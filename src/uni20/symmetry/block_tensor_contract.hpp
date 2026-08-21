/**
 * \file block_tensor_contract.hpp
 * \ingroup symmetry
 * \brief Defines the first host pairwise BlockTensor contraction.
 */

#pragma once

#include <uni20/kernel/contract.hpp>
#include <uni20/symmetry/block_tensor_mapped_view.hpp>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace uni20
{
namespace detail
{

template <class LeftTensor, class RightTensor>
concept PairwiseContractionSources = (std::remove_cvref_t<LeftTensor>::codomain_type::size() > 0) &&
                                     (std::remove_cvref_t<RightTensor>::domain_type::size() > 0);

template <class Tensor>
using pairwise_const_block_t = decltype(std::declval<std::remove_reference_t<Tensor> const&>().block(
    std::declval<typename std::remove_cvref_t<Tensor>::key_type const&>()));

template <class Block>
concept HostReferenceReadableBlock =
    DefaultAccessorMdspanLike<Block> && (std::remove_cvref_t<Block>::rank() == 0 || StridedMdspanLike<Block>);

template <class Tensor>
concept HostReferenceContractionSource = HostReferenceReadableBlock<pairwise_const_block_t<Tensor>>;

template <class LeftTensor, class RightTensor>
  requires PairwiseContractionSources<LeftTensor, RightTensor>
auto make_pairwise_contraction_domain(LeftTensor const& left, RightTensor const& right)
{
  using right_type = std::remove_cvref_t<RightTensor>;
  constexpr std::size_t right_domain_size = right_type::domain_type::size();
  return domain_from_tuple(
      std::tuple_cat(left.domain().spaces(), tuple_slice<1, right_domain_size - 1>(right.domain().spaces())));
}

template <class LeftTensor, class RightTensor>
  requires PairwiseContractionSources<LeftTensor, RightTensor>
auto make_pairwise_contraction_codomain(LeftTensor const& left, RightTensor const& right)
{
  using left_type = std::remove_cvref_t<LeftTensor>;
  constexpr std::size_t left_codomain_size = left_type::codomain_type::size();
  return codomain_from_tuple(
      std::tuple_cat(tuple_slice<0, left_codomain_size - 1>(left.codomain().spaces()), right.codomain().spaces()));
}

template <class LeftTensor, class RightTensor>
  requires PairwiseContractionSources<LeftTensor, RightTensor>
struct PairwiseContractionTraits
{
    using left_type = std::remove_cvref_t<LeftTensor>;
    using right_type = std::remove_cvref_t<RightTensor>;
    static constexpr std::size_t left_codomain_size = left_type::codomain_type::size();
    using left_space_type = typename left_type::codomain_type::template space_type<left_codomain_size - 1>;
    using right_space_type = typename right_type::domain_type::template space_type<0>;
    static constexpr bool contracted_has_key = BlockTensorSpaceTraits<left_space_type>::has_block_coordinate;
    static constexpr bool contracted_has_dense_axis = BlockTensorSpaceTraits<left_space_type>::has_dense_axis;
    static constexpr std::size_t left_domain_key_count =
        BoundaryBlockShape<typename left_type::domain_type>::key_coordinate_count;
    static constexpr std::size_t left_codomain_key_count =
        BoundaryBlockShape<typename left_type::codomain_type>::key_coordinate_count;
    static constexpr std::size_t right_domain_key_count =
        BoundaryBlockShape<typename right_type::domain_type>::key_coordinate_count;
    static constexpr std::size_t right_codomain_key_count =
        BoundaryBlockShape<typename right_type::codomain_type>::key_coordinate_count;
    static constexpr std::size_t result_key_count =
        left_type::key_coordinate_count() + right_type::key_coordinate_count() - (contracted_has_key ? 2U : 0U);
    using result_key_type = BlockKey<result_key_count>;
    using domain_type =
        decltype(make_pairwise_contraction_domain(std::declval<left_type const&>(), std::declval<right_type const&>()));
    using codomain_type = decltype(make_pairwise_contraction_codomain(std::declval<left_type const&>(),
                                                                      std::declval<right_type const&>()));

    static_assert(std::same_as<left_space_type, right_space_type>,
                  "pairwise BlockTensor contraction requires identical contracted space types");
    static_assert(BoundaryBlockShape<domain_type>::key_coordinate_count +
                      BoundaryBlockShape<codomain_type>::key_coordinate_count ==
                  result_key_count);
};

template <class OutputStorage, class LeftTensor, class RightTensor>
  requires PairwiseContractionSources<LeftTensor, RightTensor>
struct PairwiseContractionOutputTraits
{
    using contraction_traits = PairwiseContractionTraits<LeftTensor, RightTensor>;
    using value_type = typename std::remove_cvref_t<LeftTensor>::value_type;
    static constexpr std::size_t key_count = contraction_traits::result_key_count;
    static constexpr std::size_t dense_order =
        BoundaryBlockShape<typename contraction_traits::domain_type>::dense_block_order +
        BoundaryBlockShape<typename contraction_traits::codomain_type>::dense_block_order;
    using storage_type = typename OutputStorage::template storage_t<value_type, key_count, dense_order>;
    using block_type = typename storage_type::mutable_block_type;
};

template <class OutputStorage, class LeftTensor, class RightTensor>
concept HostReferenceContractionOutput =
    PairwiseContractionSources<LeftTensor, RightTensor> && SparseBlockStorage<OutputStorage> &&
    BlockTensorStorageFor<OutputStorage,
                          typename PairwiseContractionOutputTraits<OutputStorage, LeftTensor, RightTensor>::value_type,
                          PairwiseContractionOutputTraits<OutputStorage, LeftTensor, RightTensor>::key_count,
                          PairwiseContractionOutputTraits<OutputStorage, LeftTensor, RightTensor>::dense_order> &&
    MutableMdspanLike<typename PairwiseContractionOutputTraits<OutputStorage, LeftTensor, RightTensor>::block_type> &&
    HostReferenceReadableBlock<
        typename PairwiseContractionOutputTraits<OutputStorage, LeftTensor, RightTensor>::block_type>;

template <class Traits, class LeftKey, class RightKey>
auto pairwise_contracted_coordinates_match(LeftKey const& left_key, RightKey const& right_key) -> bool
{
  if constexpr (Traits::contracted_has_key)
  {
    constexpr std::size_t left_axis = Traits::left_type::key_coordinate_count() - 1;
    return left_key.coordinate(left_axis) == right_key.coordinate(0);
  }
  else
  {
    static_cast<void>(left_key);
    static_cast<void>(right_key);
    return true;
  }
}

template <class Traits, class LeftKey, class RightKey>
auto make_pairwise_result_key(LeftKey const& left_key, RightKey const& right_key) -> typename Traits::result_key_type
{
  std::array<std::size_t, Traits::result_key_count> coordinates{};
  std::size_t output = 0;
  auto append = [&](auto const& key, std::size_t first, std::size_t last) {
    for (std::size_t axis = first; axis < last; ++axis)
      coordinates[output++] = key.coordinate(axis);
  };
  constexpr std::size_t contracted_count = Traits::contracted_has_key ? 1 : 0;
  append(left_key, 0, Traits::left_domain_key_count);
  append(right_key, contracted_count, Traits::right_domain_key_count);
  append(left_key, Traits::left_domain_key_count, Traits::left_type::key_coordinate_count() - contracted_count);
  append(right_key, Traits::right_domain_key_count, Traits::right_type::key_coordinate_count());
  return typename Traits::result_key_type{coordinates};
}

template <class Traits> struct PairwiseContractionWorkItem
{
    typename Traits::left_type::key_type left_key;
    typename Traits::right_type::key_type right_key;
    typename Traits::result_key_type result_key;
};

template <class Traits, class LeftTensor, class RightTensor>
auto make_pairwise_contraction_worklist(LeftTensor const& left,
                                        RightTensor const& right) -> std::vector<PairwiseContractionWorkItem<Traits>>
{
  std::vector<PairwiseContractionWorkItem<Traits>> worklist;
  for (auto const& left_key : left.stored_keys())
  {
    for (auto const& right_key : right.stored_keys())
    {
      if (pairwise_contracted_coordinates_match<Traits>(left_key, right_key))
      {
        worklist.push_back({left_key, right_key, make_pairwise_result_key<Traits>(left_key, right_key)});
      }
    }
  }
  return worklist;
}

template <class Traits> consteval auto make_pairwise_output_dense_permutation()
{
  constexpr std::size_t left_domain_count =
      BoundaryBlockShape<typename Traits::left_type::domain_type>::dense_block_order;
  constexpr std::size_t left_codomain_count =
      BoundaryBlockShape<typename Traits::left_type::codomain_type>::dense_block_order;
  constexpr std::size_t right_domain_count =
      BoundaryBlockShape<typename Traits::right_type::domain_type>::dense_block_order;
  constexpr std::size_t right_codomain_count =
      BoundaryBlockShape<typename Traits::right_type::codomain_type>::dense_block_order;
  constexpr std::size_t contracted_count = Traits::contracted_has_dense_axis ? 1 : 0;
  constexpr std::size_t left_codomain_external = left_codomain_count - contracted_count;
  constexpr std::size_t right_domain_external = right_domain_count - contracted_count;
  constexpr std::size_t result_count =
      left_domain_count + right_domain_external + left_codomain_external + right_codomain_count;

  std::array<std::size_t, result_count> permutation{};
  std::size_t output = 0;
  auto append = [&](std::size_t first, std::size_t last) {
    for (std::size_t axis = first; axis < last; ++axis)
      permutation[output++] = axis;
  };
  append(0, left_domain_count);
  append(left_domain_count + right_domain_external, left_domain_count + right_domain_external + left_codomain_external);
  append(left_domain_count, left_domain_count + right_domain_external);
  append(left_domain_count + right_domain_external + left_codomain_external, result_count);
  if (output != result_count) throw "invalid pairwise contraction output permutation";
  return permutation;
}

} // namespace detail

/// \brief Contract one adjacent BlockTensor factor pair into a sparse owning result.
/// \details This first host path contracts the rightmost codomain factor of
///          \p left with the leftmost domain factor of \p right. The explicit
///          axis positions must name those factors. Contracted spaces must be
///          exactly equal, including labels and explicit duality. Only stored
///          block pairs with the same contracted basis occurrence contribute;
///          no dense symmetry-erasing materialization is permitted.
/// \tparam LeftAxis Flattened domain-then-codomain axis of \p left.
/// \tparam RightAxis Flattened domain-then-codomain axis of \p right.
/// \tparam OutputStorage Sparse storage policy whose blocks provide immediate
///                       host default-accessor access for the owning result.
/// \tparam LeftTensor Left BlockTensor-like operand.
/// \tparam RightTensor Right BlockTensor-like operand.
/// \param left Left operand, whose contracted factor must be its rightmost codomain factor.
/// \param right Right operand, whose contracted factor must be its leftmost domain factor.
/// \return Sparse tensor over the uncontracted boundary factors.
/// \throws std::invalid_argument If symmetries or contracted space values differ.
template <std::size_t LeftAxis, std::size_t RightAxis, class OutputStorage = PackedSparseBlockStorage<>,
          class LeftTensor, class RightTensor>
  requires detail::PairwiseContractionSources<LeftTensor, RightTensor> &&
           detail::HostReferenceContractionSource<LeftTensor> && detail::HostReferenceContractionSource<RightTensor> &&
           detail::HostReferenceContractionOutput<OutputStorage, LeftTensor, RightTensor> &&
           (LeftAxis == std::remove_cvref_t<LeftTensor>::order() - 1) && (RightAxis == 0)
auto contract(LeftTensor const& left, RightTensor const& right)
{
  using traits = detail::PairwiseContractionTraits<LeftTensor, RightTensor>;
  using left_type = typename traits::left_type;
  using right_type = typename traits::right_type;
  using value_type = typename left_type::value_type;
  static_assert(std::same_as<value_type, typename right_type::value_type>,
                "the first BlockTensor contraction requires identical scalar types");
  static_assert(left_type::order() + right_type::order() >= 2);
  static_assert(left_type::order() + right_type::order() - 2 <= 4,
                "the first BlockTensor contraction supports result order at most four");

  if (left.symmetry() != right.symmetry())
  {
    throw std::invalid_argument("BlockTensor contraction requires matching symmetries");
  }
  auto const& left_space = left.codomain().template space<traits::left_codomain_size - 1>();
  auto const& right_space = right.domain().template space<0>();
  if (left_space != right_space)
  {
    throw std::invalid_argument("BlockTensor contraction requires exactly equal contracted spaces");
  }

  auto const worklist = detail::make_pairwise_contraction_worklist<traits>(left, right);
  std::vector<typename traits::result_key_type> result_keys;
  result_keys.reserve(worklist.size());
  for (auto const& item : worklist)
    result_keys.push_back(item.result_key);
  std::ranges::sort(result_keys);
  result_keys.erase(std::unique(result_keys.begin(), result_keys.end()), result_keys.end());

  auto result_domain = detail::make_pairwise_contraction_domain(left, right);
  auto result_codomain = detail::make_pairwise_contraction_codomain(left, right);
  using result_type =
      BlockTensor<value_type, typename traits::domain_type, typename traits::codomain_type, OutputStorage>;
  result_type result(left.symmetry(), std::move(result_domain), std::move(result_codomain), std::move(result_keys));
  auto const stored_result_keys = result.stored_keys();
  std::vector<bool> initialized(stored_result_keys.size(), false);
  constexpr auto output_permutation = detail::make_pairwise_output_dense_permutation<traits>();

  for (auto const& item : worklist)
  {
    auto const found = std::ranges::lower_bound(stored_result_keys, item.result_key);
    auto const ordinal = static_cast<std::size_t>(found - stored_result_keys.begin());
    auto output_block = detail::permute_block(result.block(item.result_key), output_permutation);
    constexpr auto left_identity = detail::make_identity_axis_permutation<left_type::dense_block_order()>();
    constexpr auto right_identity = detail::make_identity_axis_permutation<right_type::dense_block_order()>();
    auto const left_block = detail::permute_block(std::as_const(left).block(item.left_key), left_identity);
    auto const right_block = detail::permute_block(std::as_const(right).block(item.right_key), right_identity);
    value_type const beta = initialized[ordinal] ? value_type{1} : value_type{0};
    if constexpr (traits::contracted_has_dense_axis)
    {
      constexpr std::array<std::pair<std::size_t, std::size_t>, 1> dimensions{
          std::pair{left_type::dense_block_order() - 1, std::size_t{0}}};
      kernel::contract(value_type{1}, left_block, right_block, dimensions, beta, output_block);
    }
    else
    {
      constexpr std::array<std::pair<std::size_t, std::size_t>, 0> dimensions{};
      kernel::contract(value_type{1}, left_block, right_block, dimensions, beta, output_block);
    }
    initialized[ordinal] = true;
  }
  return result;
}

} // namespace uni20
