/**
 * \file block_tensor_repartition.hpp
 * \ingroup symmetry
 * \brief Defines zero-copy bosonic edge repartition views.
 */

#pragma once

#include <uni20/symmetry/block_tensor_concepts.hpp>
#include <uni20/symmetry/block_tensor_mapped_view.hpp>

#include <array>
#include <concepts>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace uni20
{

/// \brief Morphism side from which `repartition` bends one edge leg.
enum class MorphismSide
{
  Domain,
  Codomain
};

/// \brief Planar boundary end at which `repartition` bends one leg.
enum class BoundaryEnd
{
  Left,
  Right
};

namespace detail
{

template <MorphismSide Side, BoundaryEnd End, class Tensor> auto make_repartitioned_domain(Tensor const& tensor)
{
  using tensor_type = std::remove_cvref_t<Tensor>;
  constexpr std::size_t domain_size = tensor_type::domain_type::size();
  constexpr std::size_t codomain_size = tensor_type::codomain_type::size();
  auto const domain_spaces = tensor.domain().spaces();

  if constexpr (Side == MorphismSide::Domain)
  {
    if constexpr (End == BoundaryEnd::Left)
      return domain_from_tuple(tuple_slice<1, domain_size - 1>(domain_spaces));
    else
      return domain_from_tuple(tuple_slice<0, domain_size - 1>(domain_spaces));
  }
  else
  {
    auto const codomain_spaces = tensor.codomain().spaces();
    constexpr std::size_t moved_index = End == BoundaryEnd::Left ? 0 : codomain_size - 1;
    auto moved = dual(std::get<moved_index>(codomain_spaces));
    if constexpr (End == BoundaryEnd::Left)
      return domain_from_tuple(std::tuple_cat(std::tuple{std::move(moved)}, domain_spaces));
    else
      return domain_from_tuple(std::tuple_cat(domain_spaces, std::tuple{std::move(moved)}));
  }
}

template <MorphismSide Side, BoundaryEnd End, class Tensor> auto make_repartitioned_codomain(Tensor const& tensor)
{
  using tensor_type = std::remove_cvref_t<Tensor>;
  constexpr std::size_t domain_size = tensor_type::domain_type::size();
  constexpr std::size_t codomain_size = tensor_type::codomain_type::size();
  auto const codomain_spaces = tensor.codomain().spaces();

  if constexpr (Side == MorphismSide::Codomain)
  {
    if constexpr (End == BoundaryEnd::Left)
      return codomain_from_tuple(tuple_slice<1, codomain_size - 1>(codomain_spaces));
    else
      return codomain_from_tuple(tuple_slice<0, codomain_size - 1>(codomain_spaces));
  }
  else
  {
    auto const domain_spaces = tensor.domain().spaces();
    constexpr std::size_t moved_index = End == BoundaryEnd::Left ? 0 : domain_size - 1;
    auto moved = dual(std::get<moved_index>(domain_spaces));
    if constexpr (End == BoundaryEnd::Left)
      return codomain_from_tuple(std::tuple_cat(std::tuple{std::move(moved)}, codomain_spaces));
    else
      return codomain_from_tuple(std::tuple_cat(codomain_spaces, std::tuple{std::move(moved)}));
  }
}

template <class Tensor, MorphismSide Side>
concept RepartitionSourceHasLeg =
    BlockTensorView<Tensor> &&
    ((Side == MorphismSide::Domain && std::remove_cvref_t<Tensor>::domain_type::size() > 0) ||
     (Side == MorphismSide::Codomain && std::remove_cvref_t<Tensor>::codomain_type::size() > 0));

template <MorphismSide Side, BoundaryEnd End, class Tensor>
  requires RepartitionSourceHasLeg<Tensor, Side>
consteval auto make_repartition_factor_permutation()
{
  using tensor_type = std::remove_cvref_t<Tensor>;
  constexpr std::size_t domain_size = tensor_type::domain_type::size();
  constexpr std::size_t codomain_size = tensor_type::codomain_type::size();
  constexpr std::size_t order = tensor_type::order();
  std::array<std::size_t, order> permutation{};
  std::size_t output = 0;
  auto append = [&](std::size_t first, std::size_t last) {
    for (std::size_t axis = first; axis < last; ++axis)
      permutation[output++] = axis;
  };

  if constexpr (Side == MorphismSide::Domain && End == BoundaryEnd::Left)
  {
    append(1, domain_size);
    permutation[output++] = 0;
    append(domain_size, order);
  }
  else if constexpr (Side == MorphismSide::Domain && End == BoundaryEnd::Right)
  {
    append(0, domain_size - 1);
    append(domain_size, order);
    permutation[output++] = domain_size - 1;
  }
  else if constexpr (Side == MorphismSide::Codomain && End == BoundaryEnd::Left)
  {
    permutation[output++] = domain_size;
    append(0, domain_size);
    append(domain_size + 1, order);
  }
  else
  {
    append(0, domain_size);
    permutation[output++] = order - 1;
    append(domain_size, order - 1);
  }

  if (output != order || codomain_size + domain_size != order) throw "invalid repartition factor permutation";
  return permutation;
}

template <class SourceTensor, MorphismSide Side, BoundaryEnd End>
  requires RepartitionSourceHasLeg<SourceTensor, Side>
struct RepartitionViewTraits
{
    using source_tensor_type = std::remove_const_t<SourceTensor>;
    using domain_type = decltype(make_repartitioned_domain<Side, End>(std::declval<source_tensor_type const&>()));
    using codomain_type = decltype(make_repartitioned_codomain<Side, End>(std::declval<source_tensor_type const&>()));
    static constexpr auto factor_permutation = make_repartition_factor_permutation<Side, End, source_tensor_type>();
    static constexpr auto key_permutation =
        make_filtered_axis_permutation<source_tensor_type, factor_permutation, false>();
    static constexpr auto dense_permutation =
        make_filtered_axis_permutation<source_tensor_type, factor_permutation, true>();
    using type = BlockTensorMappedView<SourceTensor, domain_type, codomain_type, key_permutation, dense_permutation>;
};

} // namespace detail

/// \brief Zero-copy view of one bosonic edge leg bent across a morphism boundary.
/// \details The view owns transformed boundary and logical-key metadata while
///          retaining a reference to the source tensor's numerical storage.
///          Dense axes are exposed through strided mdspans in transformed order.
///          Replacing or structurally modifying the source invalidates the view;
///          writes to existing payload elements do not.
/// \tparam SourceTensor Source BlockTensor-like value, optionally const.
/// \tparam Side Morphism side from which the edge factor moves.
/// \tparam End Planar boundary end at which the factor bends.
template <class SourceTensor, MorphismSide Side, BoundaryEnd End>
  requires detail::RepartitionSourceHasLeg<SourceTensor, Side>
using BlockTensorRepartitionView = typename detail::RepartitionViewTraits<SourceTensor, Side, End>::type;

/// \brief Bend one planar edge leg across a BlockTensor morphism boundary.
/// \details The returned lvalue view owns only transformed metadata; numerical
///          payload remains in \p tensor. A temporary borrowed view may be
///          transformed because the result retains direct descriptors for the
///          same payload. Owning rvalues are rejected because destroying the
///          owner would invalidate the payload. Payload element writes remain
///          valid, but replacing or structurally modifying the ultimate source
///          owner invalidates the view.
/// \tparam Side Source morphism side.
/// \tparam End Planar edge to bend.
/// \tparam Tensor BlockTensor-like source value.
/// \param tensor Source tensor whose lifetime must cover the returned view.
/// \return Zero-copy transformed BlockTensor view.
template <MorphismSide Side, BoundaryEnd End, class Tensor>
  requires detail::RepartitionSourceHasLeg<Tensor, Side>
auto repartition(Tensor& tensor) -> BlockTensorRepartitionView<Tensor, Side, End>
{
  using view_type = BlockTensorRepartitionView<Tensor, Side, End>;
  return view_type(tensor, detail::make_repartitioned_domain<Side, End>(tensor),
                   detail::make_repartitioned_codomain<Side, End>(tensor));
}

/// \brief Repartition a temporary borrowed BlockTensor view without retaining the intermediate view.
template <MorphismSide Side, BoundaryEnd End, BorrowedBlockTensorView Tensor>
  requires(!std::is_lvalue_reference_v<Tensor &&> && detail::RepartitionSourceHasLeg<Tensor, Side>)
auto repartition(Tensor&& tensor)
{
  using view_type = BlockTensorRepartitionView<Tensor, Side, End>;
  return view_type(tensor, detail::make_repartitioned_domain<Side, End>(tensor),
                   detail::make_repartitioned_codomain<Side, End>(tensor));
}

template <MorphismSide Side, BoundaryEnd End, class Tensor>
  requires(!std::is_lvalue_reference_v<Tensor &&> && detail::RepartitionSourceHasLeg<Tensor, Side> &&
           !BorrowedBlockTensorView<Tensor>)
auto repartition(Tensor&&) = delete;

} // namespace uni20
