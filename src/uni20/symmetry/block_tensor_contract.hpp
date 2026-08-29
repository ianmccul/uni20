/**
 * \file block_tensor_contract.hpp
 * \ingroup symmetry
 * \brief Defines local pairwise BlockTensor contraction.
 */

#pragma once

#include <uni20/async/debug_scheduler.hpp>
#include <uni20/linalg/async.hpp>
#include <uni20/linalg/elementwise_functions.hpp>
#include <uni20/linalg/ops/contract.hpp>
#include <uni20/symmetry/block_tensor_concepts.hpp>
#include <uni20/symmetry/block_tensor_mapped_view.hpp>
#include <uni20/tensor/transform.hpp>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <memory>
#include <ranges>
#include <span>
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
concept PairwiseContractionSources =
    BlockTensorView<LeftTensor> && BlockTensorView<RightTensor> &&
    (std::remove_cvref_t<LeftTensor>::codomain_type::size() > 0) &&
    (std::remove_cvref_t<RightTensor>::domain_type::size() > 0) &&
    requires(block_tensor_value_t<LeftTensor> lhs, block_tensor_value_t<RightTensor> rhs) { lhs* rhs; } &&
    Scalar<std::remove_cvref_t<decltype(std::declval<block_tensor_value_t<LeftTensor>>() *
                                        std::declval<block_tensor_value_t<RightTensor>>())>>;

template <class LeftTensor, class RightTensor>
  requires PairwiseContractionSources<LeftTensor, RightTensor>
using pairwise_contraction_value_t = std::remove_cvref_t<decltype(std::declval<block_tensor_value_t<LeftTensor>>() *
                                                                  std::declval<block_tensor_value_t<RightTensor>>())>;

struct DefaultPairwiseContractionStorage
{};

template <class RequestedStorage, class DefaultStorage>
using selected_pairwise_storage_t =
    std::conditional_t<std::same_as<RequestedStorage, DefaultPairwiseContractionStorage>, DefaultStorage,
                       RequestedStorage>;

template <class Tensor> using pairwise_const_block_t = block_tensor_const_block_t<Tensor>;

template <class Tensor> using pairwise_mutable_block_t = block_tensor_mutable_block_t<Tensor>;

template <class Tensor>
concept LocalContractionSource =
    BlockTensorView<Tensor> && !AsyncBlockTensorView<Tensor> &&
    LocalBlockStorageFor<
        typename std::remove_cvref_t<Tensor>::storage_policy, typename std::remove_cvref_t<Tensor>::value_type,
        std::remove_cvref_t<Tensor>::key_coordinate_count(), std::remove_cvref_t<Tensor>::dense_block_order()>;

template <std::size_t ContractedCount = 1, class LeftTensor, class RightTensor>
  requires PairwiseContractionSources<LeftTensor, RightTensor>
auto make_pairwise_contraction_domain(LeftTensor const& left, RightTensor const& right)
{
  using right_type = std::remove_cvref_t<RightTensor>;
  constexpr std::size_t right_domain_size = right_type::domain_type::size();
  return domain_from_tuple(
      std::tuple_cat(left.domain().spaces(),
                     tuple_slice<ContractedCount, right_domain_size - ContractedCount>(right.domain().spaces())));
}

template <std::size_t ContractedCount = 1, class LeftTensor, class RightTensor>
  requires PairwiseContractionSources<LeftTensor, RightTensor>
auto make_pairwise_contraction_codomain(LeftTensor const& left, RightTensor const& right)
{
  using left_type = std::remove_cvref_t<LeftTensor>;
  constexpr std::size_t left_codomain_size = left_type::codomain_type::size();
  return codomain_from_tuple(std::tuple_cat(
      tuple_slice<0, left_codomain_size - ContractedCount>(left.codomain().spaces()), right.codomain().spaces()));
}

template <class Boundary, std::size_t First, std::size_t... Axis>
consteval auto boundary_key_coordinate_count(std::index_sequence<Axis...>) -> std::size_t
{
  return (
      std::size_t{0} + ... +
      (BlockTensorSpaceTraits<typename Boundary::template space_type<First + Axis>>::has_block_coordinate ? 1U : 0U));
}

template <class Boundary, std::size_t First, std::size_t... Axis>
consteval auto boundary_dense_axis_count(std::index_sequence<Axis...>) -> std::size_t
{
  return (std::size_t{0} + ... +
          (BlockTensorSpaceTraits<typename Boundary::template space_type<First + Axis>>::has_dense_axis ? 1U : 0U));
}

template <class LeftBoundary, class RightBoundary, std::size_t LeftFirst, std::size_t... Axis>
consteval auto pairwise_contracted_space_types_match(std::index_sequence<Axis...>) -> bool
{
  return (std::same_as<typename LeftBoundary::template space_type<LeftFirst + Axis>,
                       typename RightBoundary::template space_type<Axis>> &&
          ...);
}

template <class LeftTensor, class RightTensor, std::size_t ContractedCount = 1>
  requires PairwiseContractionSources<LeftTensor, RightTensor>
struct PairwiseContractionTraits
{
    using left_type = std::remove_cvref_t<LeftTensor>;
    using right_type = std::remove_cvref_t<RightTensor>;
    static constexpr std::size_t contracted_count = ContractedCount;
    static constexpr std::size_t left_codomain_size = left_type::codomain_type::size();
    static constexpr std::size_t right_domain_size = right_type::domain_type::size();
    static_assert(contracted_count > 0);
    static_assert(contracted_count <= left_codomain_size);
    static_assert(contracted_count <= right_domain_size);
    static_assert(
        pairwise_contracted_space_types_match<typename left_type::codomain_type, typename right_type::domain_type,
                                              left_codomain_size - contracted_count>(
            std::make_index_sequence<contracted_count>{}));
    static constexpr std::size_t contracted_key_count =
        boundary_key_coordinate_count<typename left_type::codomain_type, left_codomain_size - contracted_count>(
            std::make_index_sequence<contracted_count>{});
    static constexpr std::size_t contracted_dense_count =
        boundary_dense_axis_count<typename left_type::codomain_type, left_codomain_size - contracted_count>(
            std::make_index_sequence<contracted_count>{});
    static constexpr bool contracted_has_key = contracted_key_count != 0;
    static constexpr bool contracted_has_dense_axis = contracted_dense_count != 0;
    static constexpr std::size_t left_domain_key_count =
        BoundaryBlockShape<typename left_type::domain_type>::key_coordinate_count;
    static constexpr std::size_t left_codomain_key_count =
        BoundaryBlockShape<typename left_type::codomain_type>::key_coordinate_count;
    static constexpr std::size_t right_domain_key_count =
        BoundaryBlockShape<typename right_type::domain_type>::key_coordinate_count;
    static constexpr std::size_t right_codomain_key_count =
        BoundaryBlockShape<typename right_type::codomain_type>::key_coordinate_count;
    static constexpr std::size_t result_key_count =
        left_type::key_coordinate_count() + right_type::key_coordinate_count() - 2 * contracted_key_count;
    using result_key_type = BlockKey<result_key_count>;
    using domain_type = decltype(make_pairwise_contraction_domain<contracted_count>(std::declval<left_type const&>(),
                                                                                    std::declval<right_type const&>()));
    using codomain_type = decltype(make_pairwise_contraction_codomain<contracted_count>(
        std::declval<left_type const&>(), std::declval<right_type const&>()));

    static_assert(BoundaryBlockShape<domain_type>::key_coordinate_count +
                      BoundaryBlockShape<codomain_type>::key_coordinate_count ==
                  result_key_count);
};

template <class OutputStorage, class LeftTensor, class RightTensor, std::size_t ContractedCount = 1>
  requires PairwiseContractionSources<LeftTensor, RightTensor>
struct PairwiseContractionOutputTraits
{
    using contraction_traits = PairwiseContractionTraits<LeftTensor, RightTensor, ContractedCount>;
    using value_type = pairwise_contraction_value_t<LeftTensor, RightTensor>;
    static constexpr std::size_t key_count = contraction_traits::result_key_count;
    static constexpr std::size_t dense_order =
        BoundaryBlockShape<typename contraction_traits::domain_type>::dense_block_order +
        BoundaryBlockShape<typename contraction_traits::codomain_type>::dense_block_order;
    using storage_type = typename OutputStorage::template storage_t<value_type, key_count, dense_order>;
    using block_type = typename storage_type::mutable_block_type;
};

template <class OutputStorage, class LeftTensor, class RightTensor, std::size_t ContractedCount = 1>
concept LocalContractionOutput =
    PairwiseContractionSources<LeftTensor, RightTensor> && BlockTensorStorage<OutputStorage> &&
    LocalBlockStorageFor<
        OutputStorage,
        typename PairwiseContractionOutputTraits<OutputStorage, LeftTensor, RightTensor, ContractedCount>::value_type,
        PairwiseContractionOutputTraits<OutputStorage, LeftTensor, RightTensor, ContractedCount>::key_count,
        PairwiseContractionOutputTraits<OutputStorage, LeftTensor, RightTensor, ContractedCount>::dense_order> &&
    !AsyncLocalBlockStorageFor<
        OutputStorage,
        typename PairwiseContractionOutputTraits<OutputStorage, LeftTensor, RightTensor, ContractedCount>::value_type,
        PairwiseContractionOutputTraits<OutputStorage, LeftTensor, RightTensor, ContractedCount>::key_count,
        PairwiseContractionOutputTraits<OutputStorage, LeftTensor, RightTensor, ContractedCount>::dense_order> &&
    MutableTensorView<
        typename PairwiseContractionOutputTraits<OutputStorage, LeftTensor, RightTensor, ContractedCount>::block_type>;

template <class Traits, class LeftKey, class RightKey>
auto pairwise_contracted_coordinates_match(LeftKey const& left_key, RightKey const& right_key) -> bool
{
  for (std::size_t coordinate = 0; coordinate < Traits::contracted_key_count; ++coordinate)
  {
    auto const left_axis = Traits::left_type::key_coordinate_count() - Traits::contracted_key_count + coordinate;
    if (left_key.coordinate(left_axis) != right_key.coordinate(coordinate)) return false;
  }
  return true;
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
  append(left_key, 0, Traits::left_domain_key_count);
  append(right_key, Traits::contracted_key_count, Traits::right_domain_key_count);
  append(left_key, Traits::left_domain_key_count,
         Traits::left_type::key_coordinate_count() - Traits::contracted_key_count);
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

template <class Traits>
auto pairwise_contraction_result_keys(std::span<PairwiseContractionWorkItem<Traits> const> worklist)
    -> std::vector<typename Traits::result_key_type>
{
  std::vector<typename Traits::result_key_type> result_keys;
  result_keys.reserve(worklist.size());
  for (auto const& item : worklist)
    result_keys.push_back(item.result_key);
  std::ranges::sort(result_keys);
  result_keys.erase(std::unique(result_keys.begin(), result_keys.end()), result_keys.end());
  return result_keys;
}

template <class Result, class DomainType, class CodomainType, class Key>
auto make_pairwise_contraction_result(Symmetry symmetry, DomainType domain, CodomainType codomain,
                                      std::vector<Key> result_keys) -> Result
{
  if constexpr (CompleteBlockStorage<typename Result::storage_policy>)
  {
    return Result(symmetry, std::move(domain), std::move(codomain));
  }
  else
  {
    return Result(symmetry, std::move(domain), std::move(codomain), std::move(result_keys));
  }
}

template <class Traits> struct LocalPairwiseContractionBinding
{
    PairwiseContractionWorkItem<Traits> logical;
    std::size_t left_ordinal;
    std::size_t right_ordinal;
    std::size_t result_ordinal;
    bool initializes_result;
};

template <class Function> void execute_block_batch(SerialBlockExecution, std::size_t size, Function&& function)
{
  // Serial storage deliberately executes inline without requiring an active scheduler.
  for (std::size_t index = 0; index < size; ++index)
    function(index);
}

template <class Function> void execute_block_batch(SchedulerBatchBlockExecution, std::size_t size, Function&& function)
{
  async::execute_batch(size, std::forward<Function>(function));
}

template <class Key> auto stored_key_ordinal(std::span<Key const> keys, Key const& key) -> std::size_t
{
  auto const found = std::ranges::lower_bound(keys, key);
  if (found == keys.end() || *found != key)
  {
    throw std::logic_error("pairwise BlockTensor work item has no stored block binding");
  }
  return static_cast<std::size_t>(found - keys.begin());
}

template <class Traits, class LeftTensor, class RightTensor, class ResultTensor>
auto bind_pairwise_contraction_worklist(LeftTensor const& left, RightTensor const& right, ResultTensor const& result,
                                        std::span<PairwiseContractionWorkItem<Traits> const> worklist)
    -> std::vector<LocalPairwiseContractionBinding<Traits>>
{
  std::vector<LocalPairwiseContractionBinding<Traits>> bindings;
  bindings.reserve(worklist.size());
  std::vector<bool> result_initialized(result.stored_block_count(), false);
  for (auto const& item : worklist)
  {
    auto const result_ordinal = stored_key_ordinal(result.stored_keys(), item.result_key);
    bindings.push_back({.logical = item,
                        .left_ordinal = stored_key_ordinal(left.stored_keys(), item.left_key),
                        .right_ordinal = stored_key_ordinal(right.stored_keys(), item.right_key),
                        .result_ordinal = result_ordinal,
                        .initializes_result = !result_initialized[result_ordinal]});
    result_initialized[result_ordinal] = true;
  }
  return bindings;
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
  constexpr std::size_t left_codomain_external = left_codomain_count - Traits::contracted_dense_count;
  constexpr std::size_t right_domain_external = right_domain_count - Traits::contracted_dense_count;
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

template <class Traits> consteval auto is_async_matrix_product_geometry() -> bool
{
  if constexpr (!Traits::contracted_has_dense_axis || Traits::left_type::dense_block_order() != 2 ||
                Traits::right_type::dense_block_order() != 2 ||
                BoundaryBlockShape<typename Traits::domain_type>::dense_block_order +
                        BoundaryBlockShape<typename Traits::codomain_type>::dense_block_order !=
                    2)
  {
    return false;
  }
  else
  {
    constexpr auto permutation = make_pairwise_output_dense_permutation<Traits>();
    return permutation == make_identity_axis_permutation<2>();
  }
}

template <class Tensor>
concept AsyncMatrixContractionSource =
    AsyncBlockTensorView<Tensor> &&
    AsyncLocalBlockStorageFor<
        typename std::remove_cvref_t<Tensor>::storage_policy, typename std::remove_cvref_t<Tensor>::element_type,
        std::remove_cvref_t<Tensor>::key_coordinate_count(), std::remove_cvref_t<Tensor>::dense_block_order()>;

template <class OutputStorage, class LeftTensor, class RightTensor>
concept AsyncMatrixContractionOutput =
    PairwiseContractionSources<LeftTensor, RightTensor> && SparseBlockStorage<OutputStorage> &&
    std::same_as<block_tensor_value_t<LeftTensor>, block_tensor_value_t<RightTensor>> &&
    std::same_as<pairwise_contraction_value_t<LeftTensor, RightTensor>, block_tensor_value_t<LeftTensor>> &&
    AsyncLocalBlockStorageFor<
        OutputStorage, typename PairwiseContractionOutputTraits<OutputStorage, LeftTensor, RightTensor>::value_type,
        PairwiseContractionOutputTraits<OutputStorage, LeftTensor, RightTensor>::key_count,
        PairwiseContractionOutputTraits<OutputStorage, LeftTensor, RightTensor>::dense_order>;

template <class OutputTensor, class LeftTensor, class RightTensor, std::size_t ContractedCount = 1>
concept CompatibleFixedPairwiseContractionOutput =
    PairwiseContractionSources<LeftTensor, RightTensor> && MutableBlockTensorView<OutputTensor> &&
    std::same_as<typename std::remove_cvref_t<OutputTensor>::value_type,
                 pairwise_contraction_value_t<LeftTensor, RightTensor>> &&
    std::same_as<typename std::remove_cvref_t<OutputTensor>::key_type,
                 typename PairwiseContractionTraits<LeftTensor, RightTensor, ContractedCount>::result_key_type> &&
    std::same_as<typename std::remove_cvref_t<OutputTensor>::domain_type,
                 typename PairwiseContractionTraits<LeftTensor, RightTensor, ContractedCount>::domain_type> &&
    std::same_as<typename std::remove_cvref_t<OutputTensor>::codomain_type,
                 typename PairwiseContractionTraits<LeftTensor, RightTensor, ContractedCount>::codomain_type>;

template <class OutputTensor, class LeftTensor, class RightTensor, std::size_t ContractedCount = 1>
concept LocalFixedPairwiseContractionOutput =
    CompatibleFixedPairwiseContractionOutput<OutputTensor, LeftTensor, RightTensor, ContractedCount> &&
    LocalBlockStorageFor<typename std::remove_cvref_t<OutputTensor>::storage_policy,
                         typename std::remove_cvref_t<OutputTensor>::value_type,
                         std::remove_cvref_t<OutputTensor>::key_coordinate_count(),
                         std::remove_cvref_t<OutputTensor>::dense_block_order()> &&
    !MutableAsyncBlockTensorView<OutputTensor>;

template <class OutputTensor, class LeftTensor, class RightTensor>
concept AsyncMatrixFixedPairwiseContractionOutput =
    CompatibleFixedPairwiseContractionOutput<OutputTensor, LeftTensor, RightTensor> &&
    std::same_as<block_tensor_value_t<LeftTensor>, block_tensor_value_t<RightTensor>> &&
    std::same_as<pairwise_contraction_value_t<LeftTensor, RightTensor>, block_tensor_value_t<LeftTensor>> &&
    MutableAsyncBlockTensorView<OutputTensor>;

template <class Lhs, class Rhs> constexpr auto same_pairwise_tensor_object(Lhs const& lhs, Rhs const& rhs) -> bool
{
  if constexpr (std::same_as<std::remove_cvref_t<Lhs>, std::remove_cvref_t<Rhs>>)
  {
    return std::addressof(lhs) == std::addressof(rhs);
  }
  return false;
}

template <class Traits, class LeftTensor, class RightTensor, std::size_t... Axis>
auto pairwise_contracted_spaces_equal(LeftTensor const& left, RightTensor const& right,
                                      std::index_sequence<Axis...>) -> bool
{
  return ((left.codomain().template space<Traits::left_codomain_size - Traits::contracted_count + Axis>() ==
           right.domain().template space<Axis>()) &&
          ...);
}

template <class Traits, class LeftTensor, class RightTensor>
void validate_pairwise_contraction(LeftTensor const& left, RightTensor const& right)
{
  using left_type = typename Traits::left_type;
  using right_type = typename Traits::right_type;
  static_assert(Scalar<pairwise_contraction_value_t<left_type, right_type>>,
                "BlockTensor contraction requires a scalar multiplication result");
  static_assert(left_type::order() + right_type::order() >= 2 * Traits::contracted_count);
  static_assert(left_type::order() + right_type::order() - 2 * Traits::contracted_count <= 4,
                "the first BlockTensor contraction supports result order at most four");

  if (left.symmetry() != right.symmetry())
  {
    throw std::invalid_argument("BlockTensor contraction requires matching symmetries");
  }
  if (!pairwise_contracted_spaces_equal<Traits>(left, right, std::make_index_sequence<Traits::contracted_count>{}))
  {
    throw std::invalid_argument("BlockTensor contraction requires exactly equal contracted spaces");
  }
}

template <class Traits, class OutputTensor, class LeftTensor, class RightTensor>
void validate_fixed_pairwise_contraction_output(OutputTensor const& output, LeftTensor const& left,
                                                RightTensor const& right,
                                                std::span<typename Traits::result_key_type const> result_keys)
{
  if (same_pairwise_tensor_object(output, left) || same_pairwise_tensor_object(output, right))
  {
    throw std::invalid_argument("fixed BlockTensor contraction output must not be an input tensor");
  }
  if (output.symmetry() != left.symmetry() ||
      output.domain() != make_pairwise_contraction_domain<Traits::contracted_count>(left, right) ||
      output.codomain() != make_pairwise_contraction_codomain<Traits::contracted_count>(left, right))
  {
    throw std::invalid_argument("fixed BlockTensor contraction output has incompatible symmetry or boundaries");
  }
  if (!std::ranges::includes(output.stored_keys(), result_keys))
  {
    throw std::invalid_argument("fixed BlockTensor contraction output omits a required result block");
  }
}

template <class Traits, class OutputTensor, class LeftTensor, class RightTensor>
void execute_local_pairwise_contraction(OutputTensor& output, LeftTensor const& left, RightTensor const& right,
                                        std::vector<LocalPairwiseContractionBinding<Traits>> bindings)
{
  using left_type = typename Traits::left_type;
  using value_type = typename std::remove_cvref_t<OutputTensor>::value_type;
  std::ranges::stable_sort(bindings, {}, &LocalPairwiseContractionBinding<Traits>::result_ordinal);
  std::vector<std::size_t> group_offsets;
  group_offsets.reserve(output.stored_block_count() + 1);
  group_offsets.push_back(0);
  for (std::size_t index = 1; index < bindings.size(); ++index)
  {
    if (bindings[index - 1].result_ordinal != bindings[index].result_ordinal) group_offsets.push_back(index);
  }
  if (!bindings.empty()) group_offsets.push_back(bindings.size());
  constexpr auto output_permutation = make_pairwise_output_dense_permutation<Traits>();

  auto execute_group = [&](std::size_t group) {
    std::size_t const first = group_offsets[group];
    std::size_t const last = group_offsets[group + 1];
    auto output_block = permute_block(output.block_by_ordinal(bindings[first].result_ordinal), output_permutation);

    for (std::size_t index = first; index < last; ++index)
    {
      auto const& binding = bindings[index];
      auto const left_block = std::as_const(left).block_by_ordinal(binding.left_ordinal);
      auto const right_block = std::as_const(right).block_by_ordinal(binding.right_ordinal);
      value_type const beta = index == first ? value_type{0} : value_type{1};
      constexpr auto dimensions = [] {
        std::array<std::pair<std::size_t, std::size_t>, Traits::contracted_dense_count> result{};
        for (std::size_t axis = 0; axis < result.size(); ++axis)
          result[axis] = {left_type::dense_block_order() - Traits::contracted_dense_count + axis, axis};
        return result;
      }();
      linalg::contract(output_block, value_type{1}, left_block, right_block, dimensions, beta);
    }
  };
  execute_block_batch(typename std::remove_cvref_t<OutputTensor>::storage_policy::block_execution_policy{},
                      group_offsets.size() - 1, execute_group);
}

template <class OutputTensor, class Bindings>
void zero_unbound_pairwise_output_blocks(OutputTensor& output, Bindings const& bindings)
{
  std::vector<bool> bound(output.stored_block_count(), false);
  for (auto const& binding : bindings)
    bound[binding.result_ordinal] = true;

  std::vector<std::size_t> unbound_ordinals;
  unbound_ordinals.reserve(output.stored_block_count());
  for (std::size_t ordinal = 0; ordinal < bound.size(); ++ordinal)
  {
    if (!bound[ordinal]) unbound_ordinals.push_back(ordinal);
  }
  if (unbound_ordinals.empty()) return;

  if constexpr (!MutableAsyncBlockTensorView<OutputTensor>)
  {
    execute_block_batch(typename std::remove_cvref_t<OutputTensor>::storage_policy::block_execution_policy{},
                        unbound_ordinals.size(), [&](std::size_t index) {
                          auto block = output.block_by_ordinal(unbound_ordinals[index]);
                          using value_type = typename std::remove_cvref_t<OutputTensor>::value_type;
                          uni20::fill(block, value_type{});
                        });
  }
  else
  {
    using value_type = typename std::remove_cvref_t<OutputTensor>::value_type;
    for (auto const ordinal : unbound_ordinals)
    {
      uni20::fill(output.async_block_by_ordinal(ordinal), value_type{});
    }
  }
}

} // namespace detail

/// \brief Contract an adjacent group of BlockTensor boundary factors.
/// \details Contracts the last `ContractedCount` codomain factors of \p left
///          with the first `ContractedCount` domain factors of \p right in
///          planar order. Every paired space must be exactly equal. Sparse
///          worklist generation matches all contracted block coordinates, and
///          the dense leaf contraction lowers every contracted degeneracy axis
///          together through the ordinary tensor contraction dispatcher.
/// \tparam ContractedCount Number of adjacent factor pairs to contract.
/// \tparam OutputStorage Local storage for the owning result. A
///                       complete policy also stores legal zero result blocks.
/// \tparam LeftTensor Left BlockTensor-like operand.
/// \tparam RightTensor Right BlockTensor-like operand.
/// \param left Operand contributing the contracted codomain suffix.
/// \param right Operand contributing the contracted domain prefix.
/// \return Tensor over all uncontracted factors.
/// \throws std::invalid_argument If symmetries or any paired space values differ.
template <std::size_t ContractedCount, class OutputStorage = detail::DefaultPairwiseContractionStorage,
          class LeftTensor, class RightTensor>
  requires detail::PairwiseContractionSources<LeftTensor, RightTensor> && (ContractedCount > 0) &&
           (ContractedCount <= std::remove_cvref_t<LeftTensor>::codomain_type::size()) &&
           (ContractedCount <= std::remove_cvref_t<RightTensor>::domain_type::size()) &&
           detail::LocalContractionSource<LeftTensor> && detail::LocalContractionSource<RightTensor> &&
           detail::LocalContractionOutput<
               detail::selected_pairwise_storage_t<OutputStorage, PackedSparseBlockStorage<>>, LeftTensor, RightTensor,
               ContractedCount>
auto contract_adjacent(LeftTensor const& left, RightTensor const& right)
{
  using selected_output_storage = detail::selected_pairwise_storage_t<OutputStorage, PackedSparseBlockStorage<>>;
  using traits = detail::PairwiseContractionTraits<LeftTensor, RightTensor, ContractedCount>;
  using value_type = detail::pairwise_contraction_value_t<LeftTensor, RightTensor>;
  detail::validate_pairwise_contraction<traits>(left, right);

  auto const worklist = detail::make_pairwise_contraction_worklist<traits>(left, right);
  auto result_keys = detail::pairwise_contraction_result_keys<traits>(worklist);
  auto result_domain = detail::make_pairwise_contraction_domain<ContractedCount>(left, right);
  auto result_codomain = detail::make_pairwise_contraction_codomain<ContractedCount>(left, right);
  using result_type =
      BlockTensor<value_type, typename traits::domain_type, typename traits::codomain_type, selected_output_storage>;
  auto result = detail::make_pairwise_contraction_result<result_type>(
      left.symmetry(), std::move(result_domain), std::move(result_codomain), std::move(result_keys));
  auto bindings = detail::bind_pairwise_contraction_worklist<traits>(left, right, result, worklist);
  if constexpr (CompleteBlockStorage<selected_output_storage>)
    detail::zero_unbound_pairwise_output_blocks(result, bindings);
  detail::execute_local_pairwise_contraction<traits>(result, left, right, std::move(bindings));
  return result;
}

/// \brief Overwrite a fixed BlockTensor with an adjacent group contraction.
/// \details The output must have the exact result boundary and contain every
///          worklist-produced key. Additional stored blocks become zero.
/// \tparam ContractedCount Number of adjacent factor pairs to contract.
/// \param output Existing fixed-structure result tensor.
/// \param left Operand contributing the contracted codomain suffix.
/// \param right Operand contributing the contracted domain prefix.
/// \pre Output numerical storage does not overlap either input.
/// \throws std::invalid_argument If inputs or fixed output are incompatible.
template <std::size_t ContractedCount, class OutputTensor, class LeftTensor, class RightTensor>
  requires detail::PairwiseContractionSources<LeftTensor, RightTensor> && (ContractedCount > 0) &&
           (ContractedCount <= std::remove_cvref_t<LeftTensor>::codomain_type::size()) &&
           (ContractedCount <= std::remove_cvref_t<RightTensor>::domain_type::size()) &&
           detail::LocalContractionSource<LeftTensor> && detail::LocalContractionSource<RightTensor> &&
           detail::LocalFixedPairwiseContractionOutput<OutputTensor, LeftTensor, RightTensor, ContractedCount>
void contract_adjacent(OutputTensor& output, LeftTensor const& left, RightTensor const& right)
{
  using traits = detail::PairwiseContractionTraits<LeftTensor, RightTensor, ContractedCount>;
  detail::validate_pairwise_contraction<traits>(left, right);
  auto const worklist = detail::make_pairwise_contraction_worklist<traits>(left, right);
  auto const result_keys = detail::pairwise_contraction_result_keys<traits>(worklist);
  detail::validate_fixed_pairwise_contraction_output<traits>(output, left, right, result_keys);

  auto bindings = detail::bind_pairwise_contraction_worklist<traits>(left, right, output, worklist);
  detail::zero_unbound_pairwise_output_blocks(output, bindings);
  detail::execute_local_pairwise_contraction<traits>(output, left, right, std::move(bindings));
}

/// \brief Contract one adjacent BlockTensor factor pair into an owning result.
/// \details This first host path contracts the rightmost codomain factor of
///          \p left with the leftmost domain factor of \p right. The explicit
///          axis positions must name those factors. Contracted spaces must be
///          exactly equal, including labels and explicit duality. Only stored
///          block pairs with the same contracted basis occurrence contribute;
///          no dense symmetry-erasing materialization is permitted.
/// \tparam LeftAxis Flattened domain-then-codomain axis of \p left.
/// \tparam RightAxis Flattened domain-then-codomain axis of \p right.
/// \tparam OutputStorage Local storage policy for the owning result.
/// \tparam LeftTensor Left BlockTensor-like operand.
/// \tparam RightTensor Right BlockTensor-like operand.
/// \param left Left operand, whose contracted factor must be its rightmost codomain factor.
/// \param right Right operand, whose contracted factor must be its leftmost domain factor.
/// \return Tensor over the uncontracted boundary factors.
/// \throws std::invalid_argument If symmetries or contracted space values differ.
template <std::size_t LeftAxis, std::size_t RightAxis, class OutputStorage = detail::DefaultPairwiseContractionStorage,
          class LeftTensor, class RightTensor>
  requires detail::PairwiseContractionSources<LeftTensor, RightTensor> && detail::LocalContractionSource<LeftTensor> &&
           detail::LocalContractionSource<RightTensor> &&
           detail::LocalContractionOutput<
               detail::selected_pairwise_storage_t<OutputStorage, PackedSparseBlockStorage<>>, LeftTensor,
               RightTensor> &&
           (LeftAxis == std::remove_cvref_t<LeftTensor>::order() - 1) && (RightAxis == 0)
auto contract(LeftTensor const& left, RightTensor const& right)
{
  using selected_output_storage = detail::selected_pairwise_storage_t<OutputStorage, PackedSparseBlockStorage<>>;
  using traits = detail::PairwiseContractionTraits<LeftTensor, RightTensor>;
  using value_type = detail::pairwise_contraction_value_t<LeftTensor, RightTensor>;
  detail::validate_pairwise_contraction<traits>(left, right);

  auto const worklist = detail::make_pairwise_contraction_worklist<traits>(left, right);
  auto result_keys = detail::pairwise_contraction_result_keys<traits>(worklist);

  auto result_domain = detail::make_pairwise_contraction_domain(left, right);
  auto result_codomain = detail::make_pairwise_contraction_codomain(left, right);
  using result_type =
      BlockTensor<value_type, typename traits::domain_type, typename traits::codomain_type, selected_output_storage>;
  auto result = detail::make_pairwise_contraction_result<result_type>(
      left.symmetry(), std::move(result_domain), std::move(result_codomain), std::move(result_keys));
  auto bindings = detail::bind_pairwise_contraction_worklist<traits>(left, right, result, worklist);
  if constexpr (CompleteBlockStorage<selected_output_storage>)
    detail::zero_unbound_pairwise_output_blocks(result, bindings);
  detail::execute_local_pairwise_contraction<traits>(result, left, right, std::move(bindings));
  return result;
}

/// \brief Overwrite a fixed BlockTensor with one adjacent pairwise contraction.
/// \details The output structure is immutable: it must have the exact result
///          symmetry and boundaries and store every block produced by the
///          sparse worklist. Additional stored output blocks are set to zero.
///          Immediate storage completes before return.
/// \tparam LeftAxis Flattened domain-then-codomain axis of \p left.
/// \tparam RightAxis Flattened domain-then-codomain axis of \p right.
/// \param output Existing output whose numerical value is replaced.
/// \param left Left operand, whose contracted factor must be its rightmost codomain factor.
/// \param right Right operand, whose contracted factor must be its leftmost domain factor.
/// \pre Output numerical storage does not overlap either input.
/// \throws std::invalid_argument If input spaces are incompatible, the output
///         is an input object, or its fixed structure cannot represent the result.
template <std::size_t LeftAxis, std::size_t RightAxis, class OutputTensor, class LeftTensor, class RightTensor>
  requires detail::PairwiseContractionSources<LeftTensor, RightTensor> && detail::LocalContractionSource<LeftTensor> &&
           detail::LocalContractionSource<RightTensor> &&
           detail::LocalFixedPairwiseContractionOutput<OutputTensor, LeftTensor, RightTensor> &&
           (LeftAxis == std::remove_cvref_t<LeftTensor>::order() - 1) && (RightAxis == 0)
void contract(OutputTensor& output, LeftTensor const& left, RightTensor const& right)
{
  using traits = detail::PairwiseContractionTraits<LeftTensor, RightTensor>;
  detail::validate_pairwise_contraction<traits>(left, right);
  auto const worklist = detail::make_pairwise_contraction_worklist<traits>(left, right);
  auto const result_keys = detail::pairwise_contraction_result_keys<traits>(worklist);
  detail::validate_fixed_pairwise_contraction_output<traits>(output, left, right, result_keys);

  auto bindings = detail::bind_pairwise_contraction_worklist<traits>(left, right, output, worklist);
  detail::zero_unbound_pairwise_output_blocks(output, bindings);
  detail::execute_local_pairwise_contraction<traits>(output, left, right, std::move(bindings));
}

/// \brief Schedule independent dense-block matrix products for an adjacent contraction.
/// \details Structure and the sparse logical worklist are constructed
///          synchronously. Each result block has its own `Async<Tensor>`
///          timeline, so distinct blocks may execute concurrently while
///          contributions to one output block are causally serialized. The
///          returned BlockTensor is immediate; only its numerical blocks may
///          still be pending. This first lowering supports rank-two dense
///          blocks with the natural matrix-product output axis order.
/// \tparam LeftAxis Flattened domain-then-codomain axis of \p left.
/// \tparam RightAxis Flattened domain-then-codomain axis of \p right.
/// \tparam OutputStorage Async sparse output policy, or the storage-selected default.
/// \tparam LeftTensor Left async-block BlockTensor operand.
/// \tparam RightTensor Right async-block BlockTensor operand.
/// \param left Left operand, whose contracted factor must be its rightmost codomain factor.
/// \param right Right operand, whose contracted factor must be its leftmost domain factor.
/// \return Sparse tensor with scheduled per-block numerical values.
/// \pre An async scheduler is active on the submitting application thread.
/// \throws std::invalid_argument If symmetries or contracted space values differ.
template <std::size_t LeftAxis, std::size_t RightAxis, class OutputStorage = detail::DefaultPairwiseContractionStorage,
          class LeftTensor, class RightTensor>
  requires detail::PairwiseContractionSources<LeftTensor, RightTensor> &&
           detail::AsyncMatrixContractionSource<LeftTensor> && detail::AsyncMatrixContractionSource<RightTensor> &&
           detail::AsyncMatrixContractionOutput<
               detail::selected_pairwise_storage_t<OutputStorage,
                                                   typename std::remove_cvref_t<LeftTensor>::storage_policy>,
               LeftTensor, RightTensor> &&
           (detail::is_async_matrix_product_geometry<detail::PairwiseContractionTraits<LeftTensor, RightTensor>>()) &&
           (LeftAxis == std::remove_cvref_t<LeftTensor>::order() - 1) && (RightAxis == 0)
auto contract(LeftTensor const& left, RightTensor const& right)
{
  using selected_output_storage =
      detail::selected_pairwise_storage_t<OutputStorage, typename std::remove_cvref_t<LeftTensor>::storage_policy>;
  using traits = detail::PairwiseContractionTraits<LeftTensor, RightTensor>;
  using value_type = detail::pairwise_contraction_value_t<LeftTensor, RightTensor>;
  detail::validate_pairwise_contraction<traits>(left, right);

  auto const worklist = detail::make_pairwise_contraction_worklist<traits>(left, right);
  auto result_keys = detail::pairwise_contraction_result_keys<traits>(worklist);

  auto result_domain = detail::make_pairwise_contraction_domain(left, right);
  auto result_codomain = detail::make_pairwise_contraction_codomain(left, right);
  using result_type =
      BlockTensor<value_type, typename traits::domain_type, typename traits::codomain_type, selected_output_storage>;
  result_type result(left.symmetry(), std::move(result_domain), std::move(result_codomain), std::move(result_keys));
  auto const bindings = detail::bind_pairwise_contraction_worklist<traits>(left, right, result, worklist);

  for (auto const& binding : bindings)
  {
    auto& output_block = result.async_block_by_ordinal(binding.result_ordinal);
    auto const& left_block = left.async_block_by_ordinal(binding.left_ordinal);
    auto const& right_block = right.async_block_by_ordinal(binding.right_ordinal);
    if (binding.initializes_result)
    {
      linalg::assign_product(selected_output_storage::backend_selector(), output_block, left_block, right_block);
    }
    else
    {
      linalg::add_product(selected_output_storage::backend_selector(), output_block, left_block, right_block);
    }
  }
  return result;
}

/// \brief Schedule an adjacent pairwise contraction into fixed async BlockTensor storage.
/// \details Validation completes synchronously before output epochs are
///          modified. Output-only blocks are scheduled to become zero;
///          contributing blocks are initialized by their first product and
///          subsequent products accumulate in that block's epoch order.
/// \pre An async scheduler is active on the submitting application thread.
/// \pre Output numerical storage does not overlap either input.
/// \throws std::invalid_argument If input spaces are incompatible, the output
///         is an input object, or its fixed structure cannot represent the result.
template <std::size_t LeftAxis, std::size_t RightAxis, class OutputTensor, class LeftTensor, class RightTensor>
  requires detail::PairwiseContractionSources<LeftTensor, RightTensor> &&
           detail::AsyncMatrixContractionSource<LeftTensor> && detail::AsyncMatrixContractionSource<RightTensor> &&
           detail::AsyncMatrixFixedPairwiseContractionOutput<OutputTensor, LeftTensor, RightTensor> &&
           (detail::is_async_matrix_product_geometry<detail::PairwiseContractionTraits<LeftTensor, RightTensor>>()) &&
           (LeftAxis == std::remove_cvref_t<LeftTensor>::order() - 1) && (RightAxis == 0)
void contract(OutputTensor& output, LeftTensor const& left, RightTensor const& right)
{
  using traits = detail::PairwiseContractionTraits<LeftTensor, RightTensor>;
  detail::validate_pairwise_contraction<traits>(left, right);
  auto const worklist = detail::make_pairwise_contraction_worklist<traits>(left, right);
  auto const result_keys = detail::pairwise_contraction_result_keys<traits>(worklist);
  detail::validate_fixed_pairwise_contraction_output<traits>(output, left, right, result_keys);

  auto const bindings = detail::bind_pairwise_contraction_worklist<traits>(left, right, output, worklist);
  detail::zero_unbound_pairwise_output_blocks(output, bindings);
  using storage_policy = typename std::remove_cvref_t<OutputTensor>::storage_policy;
  for (auto const& binding : bindings)
  {
    auto& output_block = output.async_block_by_ordinal(binding.result_ordinal);
    auto const& left_block = left.async_block_by_ordinal(binding.left_ordinal);
    auto const& right_block = right.async_block_by_ordinal(binding.right_ordinal);
    if (binding.initializes_result)
    {
      linalg::assign_product(storage_policy::backend_selector(), output_block, left_block, right_block);
    }
    else
    {
      linalg::add_product(storage_policy::backend_selector(), output_block, left_block, right_block);
    }
  }
}

} // namespace uni20
