/**
 * \file block_tensor_permute.hpp
 * \ingroup symmetry
 * \brief Defines zero-copy bosonic BlockTensor factor permutations.
 */

#pragma once

#include <uni20/symmetry/block_tensor_concepts.hpp>
#include <uni20/symmetry/block_tensor_mapped_view.hpp>

#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace uni20
{
namespace detail
{

template <class Tensor, std::size_t... Axis> consteval auto is_side_preserving_permutation() -> bool
{
  using tensor_type = std::remove_cvref_t<Tensor>;
  constexpr std::size_t order = tensor_type::order();
  constexpr std::size_t domain_size = tensor_type::domain_type::size();
  if constexpr (sizeof...(Axis) != order)
  {
    return false;
  }
  else
  {
    constexpr std::array<std::size_t, order> permutation{Axis...};
    if (!is_valid_axis_permutation(permutation)) return false;
    for (std::size_t output = 0; output < domain_size; ++output)
    {
      if (permutation[output] >= domain_size) return false;
    }
    for (std::size_t output = domain_size; output < order; ++output)
    {
      if (permutation[output] < domain_size) return false;
    }
    return true;
  }
}

template <class Tensor, std::size_t... Axis>
concept SidePreservingPermutation = BlockTensorView<Tensor> && is_side_preserving_permutation<Tensor, Axis...>();

template <auto FactorPermutation, class Tensor, std::size_t... I>
auto make_permuted_domain_impl(Tensor const& tensor, std::index_sequence<I...>)
{
  if constexpr (sizeof...(I) == 0)
  {
    static_cast<void>(tensor);
    return Domain<>{};
  }
  else
  {
    auto const spaces = std::tuple_cat(tensor.domain().spaces(), tensor.codomain().spaces());
    return domain_from_tuple(std::tuple{std::get<FactorPermutation[I]>(spaces)...});
  }
}

template <auto FactorPermutation, class Tensor> auto make_permuted_domain(Tensor const& tensor)
{
  using tensor_type = std::remove_cvref_t<Tensor>;
  return make_permuted_domain_impl<FactorPermutation>(tensor,
                                                      std::make_index_sequence<tensor_type::domain_type::size()>{});
}

template <auto FactorPermutation, class Tensor, std::size_t... I>
auto make_permuted_codomain_impl(Tensor const& tensor, std::index_sequence<I...>)
{
  using tensor_type = std::remove_cvref_t<Tensor>;
  constexpr std::size_t domain_size = tensor_type::domain_type::size();
  if constexpr (sizeof...(I) == 0)
  {
    static_cast<void>(tensor);
    return Codomain<>{};
  }
  else
  {
    auto const spaces = std::tuple_cat(tensor.domain().spaces(), tensor.codomain().spaces());
    return codomain_from_tuple(std::tuple{std::get<FactorPermutation[domain_size + I]>(spaces)...});
  }
}

template <auto FactorPermutation, class Tensor> auto make_permuted_codomain(Tensor const& tensor)
{
  using tensor_type = std::remove_cvref_t<Tensor>;
  return make_permuted_codomain_impl<FactorPermutation>(tensor,
                                                        std::make_index_sequence<tensor_type::codomain_type::size()>{});
}

template <class SourceTensor, std::size_t... Axis>
  requires SidePreservingPermutation<SourceTensor, Axis...>
struct PermutationViewTraits
{
    using source_tensor_type = std::remove_const_t<SourceTensor>;
    static constexpr std::array<std::size_t, source_tensor_type::order()> factor_permutation{Axis...};
    using domain_type = decltype(make_permuted_domain<factor_permutation>(std::declval<source_tensor_type const&>()));
    using codomain_type =
        decltype(make_permuted_codomain<factor_permutation>(std::declval<source_tensor_type const&>()));
    static constexpr auto key_permutation =
        make_filtered_axis_permutation<source_tensor_type, factor_permutation, false>();
    static constexpr auto dense_permutation =
        make_filtered_axis_permutation<source_tensor_type, factor_permutation, true>();
    using type = BlockTensorMappedView<SourceTensor, domain_type, codomain_type, key_permutation, dense_permutation>;
};

} // namespace detail

/// \brief Zero-copy bosonic permutation of factors within each morphism side.
/// \details Axis positions are flattened domain-then-codomain positions. The
///          permutation must retain domain factors in domain and codomain
///          factors in codomain; use `repartition` to bend a factor between
///          sides. Labels and space values move with their factors.
/// \tparam SourceTensor Source BlockTensor-like value, optionally const.
/// \tparam Axis Source axis at each output position.
template <class SourceTensor, std::size_t... Axis>
  requires detail::SidePreservingPermutation<SourceTensor, Axis...>
using BlockTensorPermutationView = typename detail::PermutationViewTraits<SourceTensor, Axis...>::type;

/// \brief Permute bosonic BlockTensor factors without moving numerical payload.
/// \details The compile-time axis list gives the source axis at each output
///          position. A temporary borrowed view may be transformed because the
///          result retains direct descriptors for the same numerical payload.
///          Owning rvalues are rejected because destroying the owner would
///          invalidate that payload. Replacing or structurally modifying the
///          ultimate source owner invalidates the view.
/// \tparam Axis Source axis at each output position.
/// \tparam Tensor BlockTensor-like source value.
/// \param tensor Source tensor whose lifetime must cover the returned view.
/// \return Zero-copy view with permuted boundaries, keys, and dense axes.
template <std::size_t... Axis, class Tensor>
  requires detail::SidePreservingPermutation<Tensor, Axis...>
auto permute(Tensor& tensor) -> BlockTensorPermutationView<Tensor, Axis...>
{
  using traits = detail::PermutationViewTraits<Tensor, Axis...>;
  using view_type = typename traits::type;
  return view_type(tensor, detail::make_permuted_domain<traits::factor_permutation>(tensor),
                   detail::make_permuted_codomain<traits::factor_permutation>(tensor));
}

/// \brief Permute a temporary borrowed BlockTensor view without retaining the intermediate view.
template <std::size_t... Axis, BorrowedBlockTensorView Tensor>
  requires(!std::is_lvalue_reference_v<Tensor &&> && detail::SidePreservingPermutation<Tensor, Axis...>)
auto permute(Tensor&& tensor)
{
  using traits = detail::PermutationViewTraits<Tensor, Axis...>;
  using view_type = typename traits::type;
  return view_type(tensor, detail::make_permuted_domain<traits::factor_permutation>(tensor),
                   detail::make_permuted_codomain<traits::factor_permutation>(tensor));
}

template <std::size_t... Axis, class Tensor>
  requires(!std::is_lvalue_reference_v<Tensor &&> && detail::SidePreservingPermutation<Tensor, Axis...> &&
           !BorrowedBlockTensorView<Tensor>)
auto permute(Tensor&&) = delete;

} // namespace uni20
