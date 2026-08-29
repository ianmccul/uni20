/**
 * \file block_tensor_linear.hpp
 * \ingroup symmetry
 * \brief Defines structure-preserving linear operations on BlockTensor values.
 */

#pragma once

#include <uni20/async/debug_scheduler.hpp>
#include <uni20/config.hpp>
#include <uni20/core/math.hpp>
#include <uni20/core/scalar_concepts.hpp>
#include <uni20/linalg/async/transform.hpp>
#include <uni20/linalg/elementwise_functions.hpp>
#include <uni20/symmetry/block_tensor.hpp>
#include <uni20/symmetry/block_tensor_concepts.hpp>
#include <uni20/tensor/copy_into.hpp>
#include <uni20/tensor/reductions.hpp>
#include <uni20/tensor/transform.hpp>

#if UNI20_BACKEND_CUDA
#include <uni20/linalg/backends/cuda/partitioned_buffer_linear.hpp>
#endif

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace uni20
{
namespace detail
{

template <class Lhs, class Rhs>
concept CompatibleBlockTensorValues =
    BlockTensorView<Lhs> && BlockTensorView<Rhs> &&
    std::same_as<typename block_tensor_type_t<Lhs>::value_type, typename block_tensor_type_t<Rhs>::value_type> &&
    std::same_as<typename block_tensor_type_t<Lhs>::key_type, typename block_tensor_type_t<Rhs>::key_type> &&
    std::same_as<typename block_tensor_type_t<Lhs>::domain_type, typename block_tensor_type_t<Rhs>::domain_type> &&
    std::same_as<typename block_tensor_type_t<Lhs>::codomain_type, typename block_tensor_type_t<Rhs>::codomain_type>;

template <class Output, class... Inputs>
concept CompatibleLinearOutputRepresentation =
    BlockTensorView<Output> && (BlockTensorView<Inputs> && ...) &&
    (!DiagonalBlockTensorView<Output> || (DiagonalBlockTensorView<Inputs> && ...));

template <class OutputStorage, class FirstInput, class... OtherInputs>
concept CompatibleLinearOutputStorage =
    BlockTensorStorage<OutputStorage> && BlockTensorView<FirstInput> && (BlockTensorView<OtherInputs> && ...) &&
    (!DiagonalBlockStorage<OutputStorage> ||
     (DiagonalBlockTensorView<FirstInput> && (DiagonalBlockTensorView<OtherInputs> && ...))) &&
    ((ImmediateBlockTensorView<FirstInput> && (ImmediateBlockTensorView<OtherInputs> && ...) &&
      ImmediateLocalBlockStorageFor<OutputStorage, block_tensor_value_t<FirstInput>,
                                    block_tensor_type_t<FirstInput>::key_coordinate_count(),
                                    block_tensor_type_t<FirstInput>::dense_block_order()>) ||
     (AsyncBlockTensorView<FirstInput> && (AsyncBlockTensorView<OtherInputs> && ...) &&
      AsyncLocalBlockStorageFor<OutputStorage, block_tensor_value_t<FirstInput>,
                                block_tensor_type_t<FirstInput>::key_coordinate_count(),
                                block_tensor_type_t<FirstInput>::dense_block_order()>));

template <class Lhs, class Rhs> void require_compatible_block_tensor_values(Lhs const& lhs, Rhs const& rhs)
{
  if (lhs.symmetry() != rhs.symmetry())
  {
    throw std::invalid_argument("BlockTensor linear operation requires matching symmetries");
  }
  if (lhs.domain() != rhs.domain() || lhs.codomain() != rhs.codomain())
  {
    throw std::invalid_argument("BlockTensor linear operation requires exactly equal boundaries");
  }
}

template <class Superset, class Subset> void require_stored_key_superset(Superset const& superset, Subset const& subset)
{
  if (!std::ranges::includes(superset.stored_keys(), subset.stored_keys()))
  {
    throw std::invalid_argument("fixed BlockTensor output does not store every required input block");
  }
}

template <BlockTensorView Tensor>
auto find_stored_key_ordinal(Tensor const& tensor, block_tensor_key_t<Tensor> const& key) -> std::optional<std::size_t>
{
  auto const found = std::ranges::lower_bound(tensor.stored_keys(), key);
  if (found == tensor.stored_keys().end() || *found != key) return std::nullopt;
  return static_cast<std::size_t>(found - tensor.stored_keys().begin());
}

template <class Lhs, class Rhs>
auto stored_key_union(Lhs const& lhs, Rhs const& rhs) -> std::vector<typename block_tensor_type_t<Lhs>::key_type>
{
  using key_type = typename block_tensor_type_t<Lhs>::key_type;
  std::vector<key_type> result;
  result.reserve(lhs.stored_block_count() + rhs.stored_block_count());
  std::ranges::set_union(lhs.stored_keys(), rhs.stored_keys(), std::back_inserter(result));
  return result;
}

struct BlockIntersectionBinding
{
    std::size_t lhs_ordinal;
    std::size_t rhs_ordinal;
};

template <class Lhs, class Rhs>
auto stored_key_intersection_bindings(Lhs const& lhs, Rhs const& rhs) -> std::vector<BlockIntersectionBinding>
{
  std::vector<BlockIntersectionBinding> result;
  result.reserve(std::min(lhs.stored_block_count(), rhs.stored_block_count()));
  auto const lhs_keys = lhs.stored_keys();
  auto const rhs_keys = rhs.stored_keys();
  std::size_t lhs_ordinal = 0;
  std::size_t rhs_ordinal = 0;
  while (lhs_ordinal < lhs_keys.size() && rhs_ordinal < rhs_keys.size())
  {
    if (lhs_keys[lhs_ordinal] < rhs_keys[rhs_ordinal])
    {
      ++lhs_ordinal;
    }
    else if (rhs_keys[rhs_ordinal] < lhs_keys[lhs_ordinal])
    {
      ++rhs_ordinal;
    }
    else
    {
      result.push_back({lhs_ordinal++, rhs_ordinal++});
    }
  }
  return result;
}

template <class Function> void execute_linear_block_batch(SerialBlockExecution, std::size_t size, Function&& function)
{
  for (std::size_t index = 0; index < size; ++index)
    function(index);
}

template <class Function>
void execute_linear_block_batch(SchedulerBatchBlockExecution, std::size_t size, Function&& function)
{
  async::execute_batch(size, std::forward<Function>(function));
}

template <class Tensor>
using block_execution_policy_t = typename block_tensor_type_t<Tensor>::storage_policy::block_execution_policy;

template <class Lhs, class Rhs>
using reduction_execution_policy_t =
    std::conditional_t<std::same_as<block_execution_policy_t<Lhs>, SchedulerBatchBlockExecution> ||
                           std::same_as<block_execution_policy_t<Rhs>, SchedulerBatchBlockExecution>,
                       SchedulerBatchBlockExecution, SerialBlockExecution>;

template <class Lhs, class Rhs> constexpr auto same_block_tensor_object(Lhs const& lhs, Rhs const& rhs) noexcept -> bool
{
  if constexpr (std::same_as<block_tensor_type_t<Lhs>, block_tensor_type_t<Rhs>>)
  {
    return std::addressof(lhs) == std::addressof(rhs);
  }
  else
  {
    return false;
  }
}

#if UNI20_BACKEND_CUDA

template <class Tensor>
concept CudaPartitionedBlockTensor = BlockTensorView<Tensor> && requires(Tensor& tensor, Tensor const& const_tensor) {
  { tensor.storage().buffer() } -> std::same_as<cuda::PartitionedCudaBuffer<block_tensor_value_t<Tensor>>&>;
  { const_tensor.storage().buffer() } -> std::same_as<cuda::PartitionedCudaBuffer<block_tensor_value_t<Tensor>> const&>;
  { const_tensor.storage().offsets() } -> std::same_as<std::span<std::size_t const>>;
};

template <CudaPartitionedBlockTensor Lhs, CudaPartitionedBlockTensor Rhs>
[[nodiscard]] bool same_packed_layout(Lhs const& lhs, Rhs const& rhs)
{
  return std::ranges::equal(lhs.stored_keys(), rhs.stored_keys()) &&
         std::ranges::equal(lhs.storage().offsets(), rhs.storage().offsets()) &&
         lhs.storage().buffer().size() == rhs.storage().buffer().size();
}

#endif

template <class Block> void set_zero_block(Block&& block)
{
  using value_type = tensor_element_t<Block>;
  auto span = mdspec_of(block);
  if constexpr (MutableDiagonalMdspecLike<decltype(span)>)
    uni20::fill(block.backend_selector(), diagonal_components(span), value_type{});
  else
    uni20::fill(std::forward<Block>(block), value_type{});
}

template <class Block, class Scalar> void scale_block(Block&& block, Scalar const& factor)
{
  auto span = mdspec_of(block);
  if constexpr (MutableDiagonalMdspecLike<decltype(span)>)
    uni20::transform_inplace(block.backend_selector(), diagonal_components(span), linalg::scale{factor});
  else
    uni20::transform_inplace(std::forward<Block>(block), linalg::scale{factor});
}

template <class OutputBlock, class InputBlock, class Scalar>
void assign_scale_block(OutputBlock&& output, Scalar const& factor, InputBlock&& input)
{
  auto output_span = mdspec_of(output);
  auto input_span = mdspec_of(input);
  if constexpr (MutableDiagonalMdspecLike<decltype(output_span)> && DiagonalMdspecLike<decltype(input_span)>)
    uni20::assign_transform(output.backend_selector(), diagonal_components(output_span), linalg::scale{factor},
                            diagonal_components(input_span));
  else
    uni20::assign_transform(std::forward<OutputBlock>(output), linalg::scale{factor}, std::forward<InputBlock>(input));
}

template <class OutputBlock, class LhsBlock, class RhsBlock>
void add_block(OutputBlock&& output, LhsBlock&& lhs, RhsBlock&& rhs)
{
  auto output_span = mdspec_of(output);
  auto lhs_span = mdspec_of(lhs);
  auto rhs_span = mdspec_of(rhs);
  if constexpr (MutableDiagonalMdspecLike<decltype(output_span)> && DiagonalMdspecLike<decltype(lhs_span)> &&
                DiagonalMdspecLike<decltype(rhs_span)>)
    uni20::assign_transform(output.backend_selector(), diagonal_components(output_span), linalg::add{},
                            diagonal_components(lhs_span), diagonal_components(rhs_span));
  else
    uni20::assign_transform(std::forward<OutputBlock>(output), linalg::add{}, std::forward<LhsBlock>(lhs),
                            std::forward<RhsBlock>(rhs));
}

template <class OutputBlock, class InputBlock> void add_inplace_block(OutputBlock&& output, InputBlock&& input)
{
  auto output_span = mdspec_of(output);
  auto input_span = mdspec_of(input);
  if constexpr (MutableDiagonalMdspecLike<decltype(output_span)> && DiagonalMdspecLike<decltype(input_span)>)
    uni20::transform_inplace(output.backend_selector(), diagonal_components(output_span), linalg::add{},
                             diagonal_components(input_span));
  else
    uni20::transform_inplace(std::forward<OutputBlock>(output), linalg::add{}, std::forward<InputBlock>(input));
}

template <class OutputBlock, class Scalar, class InputBlock>
void axpy_block(OutputBlock&& output, Scalar const& factor, InputBlock&& input)
{
  auto output_span = mdspec_of(output);
  auto input_span = mdspec_of(input);
  if constexpr (MutableDiagonalMdspecLike<decltype(output_span)> && DiagonalMdspecLike<decltype(input_span)>)
    uni20::transform_inplace(output.backend_selector(), diagonal_components(output_span), linalg::add_scaled{factor},
                             diagonal_components(input_span));
  else
    uni20::transform_inplace(std::forward<OutputBlock>(output), linalg::add_scaled{factor},
                             std::forward<InputBlock>(input));
}

template <class RequestedStorage, class Tensor>
using selected_linear_storage_t =
    std::conditional_t<std::is_void_v<RequestedStorage>, typename block_tensor_type_t<Tensor>::storage_policy,
                       RequestedStorage>;

} // namespace detail

/// \brief Set every stored numerical block to zero without changing structure.
/// \details Legal but unstored blocks remain implicit zero. Ordinary block
///          views dispatch through their storage-domain backend; async block
///          storage schedules one update on each block timeline.
template <MutableBlockTensorView Tensor> void set_zero(Tensor& tensor)
{
#if UNI20_BACKEND_CUDA
  if constexpr (detail::CudaPartitionedBlockTensor<Tensor> &&
                linalg::cuda_partitioned_fill_scalar<block_tensor_value_t<Tensor>>)
  {
    linalg::cuda_partitioned_set_zero(tensor.storage().buffer());
    return;
  }
#endif
  if constexpr (!MutableAsyncBlockTensorView<Tensor>)
  {
    detail::execute_linear_block_batch(detail::block_execution_policy_t<Tensor>{}, tensor.stored_block_count(),
                                       [&](std::size_t ordinal) {
                                         auto block = tensor.block_by_ordinal(ordinal);
                                         detail::set_zero_block(block);
                                       });
  }
  else
  {
    for (std::size_t ordinal = 0; ordinal < tensor.stored_block_count(); ++ordinal)
    {
      auto& block = tensor.async_block_by_ordinal(ordinal);
      using value_type = block_tensor_value_t<Tensor>;
      uni20::fill(block, value_type{});
    }
  }
}

/// \brief Multiply every stored numerical block by a scalar in place.
/// \details The stored key pattern is unchanged, including when the factor is zero.
template <MutableBlockTensorView Tensor, class Scalar> void scale(Tensor& tensor, Scalar factor)
{
#if UNI20_BACKEND_CUDA
  if constexpr (detail::CudaPartitionedBlockTensor<Tensor> &&
                linalg::cuda_partitioned_scale_scalars<block_tensor_value_t<Tensor>, Scalar>)
  {
    linalg::cuda_partitioned_scale(tensor.storage().buffer(), factor);
    return;
  }
#endif
  if constexpr (!MutableAsyncBlockTensorView<Tensor>)
  {
    detail::execute_linear_block_batch(detail::block_execution_policy_t<Tensor>{}, tensor.stored_block_count(),
                                       [&](std::size_t ordinal) {
                                         auto block = tensor.block_by_ordinal(ordinal);
                                         detail::scale_block(block, factor);
                                       });
  }
  else
  {
    for (std::size_t ordinal = 0; ordinal < tensor.stored_block_count(); ++ordinal)
    {
      uni20::transform_inplace(tensor.async_block_by_ordinal(ordinal), linalg::scale{factor});
    }
  }
}

/// \brief Overwrite a fixed BlockTensor output with a scaled input value.
/// \details The output may store additional blocks; they are set to zero. It
///          must already store every input key because BlockTensor structure is
///          immutable after construction. A generalized-diagonal output requires
///          a generalized-diagonal input.
/// \pre Distinct output and input values do not have overlapping numerical storage.
/// \throws std::invalid_argument If the symmetry or boundary values differ, or
///         the output omits an input key.
template <MutableBlockTensorView Output, class Scalar, BlockTensorView Input>
  requires detail::CompatibleBlockTensorValues<Output, Input> &&
           detail::CompatibleLinearOutputRepresentation<Output, Input> &&
           ((!MutableAsyncBlockTensorView<Output> && !AsyncBlockTensorView<Input>) ||
            (MutableAsyncBlockTensorView<Output> && AsyncBlockTensorView<Input>))
void assign_scale(Output& output, Scalar factor, Input const& input)
{
  detail::require_compatible_block_tensor_values(output, input);
  detail::require_stored_key_superset(output, input);
  if (detail::same_block_tensor_object(output, input))
  {
    scale(output, std::move(factor));
    return;
  }

#if UNI20_BACKEND_CUDA
  if constexpr (detail::CudaPartitionedBlockTensor<Output> && detail::CudaPartitionedBlockTensor<Input> &&
                linalg::cuda_partitioned_scale_scalars<block_tensor_value_t<Output>, Scalar>)
  {
    if (detail::same_packed_layout(output, input))
    {
      linalg::cuda_partitioned_assign_scale(output.storage().buffer(), factor, input.storage().buffer());
      return;
    }
  }
#endif

  if constexpr (!MutableAsyncBlockTensorView<Output>)
  {
    detail::execute_linear_block_batch(detail::block_execution_policy_t<Output>{}, output.stored_block_count(),
                                       [&](std::size_t ordinal) {
                                         auto output_block = output.block_by_ordinal(ordinal);
                                         auto const& key = output.stored_keys()[ordinal];
                                         if (auto input_ordinal = detail::find_stored_key_ordinal(input, key))
                                         {
                                           auto input_block = input.block_by_ordinal(*input_ordinal);
                                           detail::assign_scale_block(output_block, factor, input_block);
                                         }
                                         else
                                           detail::set_zero_block(output_block);
                                       });
  }
  else
  {
    using value_type = block_tensor_value_t<Output>;
    for (std::size_t ordinal = 0; ordinal < output.stored_block_count(); ++ordinal)
    {
      auto& output_block = output.async_block_by_ordinal(ordinal);
      auto const& key = output.stored_keys()[ordinal];
      if (auto const input_ordinal = std::ranges::lower_bound(input.stored_keys(), key);
          input_ordinal != input.stored_keys().end() && *input_ordinal == key)
      {
        auto const index = static_cast<std::size_t>(input_ordinal - input.stored_keys().begin());
        uni20::assign_transform(output_block, linalg::scale{factor}, input.async_block_by_ordinal(index));
      }
      else
      {
        uni20::fill(output_block, value_type{});
      }
    }
  }
}

/// \brief Copy a BlockTensor value into an existing fixed output structure.
/// \details A generalized-diagonal output requires a generalized-diagonal input.
/// \pre Distinct output and input values do not have overlapping numerical storage.
/// \throws std::invalid_argument If the symmetry or boundary values differ, or
///         the output omits an input key.
template <MutableBlockTensorView Output, BlockTensorView Input>
  requires detail::CompatibleBlockTensorValues<Output, Input> &&
           detail::CompatibleLinearOutputRepresentation<Output, Input> &&
           ((!MutableAsyncBlockTensorView<Output> && !AsyncBlockTensorView<Input>) ||
            (MutableAsyncBlockTensorView<Output> && AsyncBlockTensorView<Input>))
void copy(Output& output, Input const& input)
{
  if (detail::same_block_tensor_object(output, input)) return;
  using value_type = block_tensor_value_t<Output>;
  assign_scale(output, value_type{1}, input);
}

/// \brief Overwrite a fixed BlockTensor output with the sum of two values.
/// \details Missing input blocks are exact zero. The output must already store
///          the union of both input key sets; additional output blocks become zero.
///          A generalized-diagonal output requires generalized-diagonal inputs.
/// \pre Distinct output and input values do not have overlapping numerical storage.
/// \throws std::invalid_argument If symmetry or boundary values differ, or the
///         output omits an input key.
template <MutableBlockTensorView Output, BlockTensorView Lhs, BlockTensorView Rhs>
  requires detail::CompatibleBlockTensorValues<Output, Lhs> && detail::CompatibleBlockTensorValues<Output, Rhs> &&
           detail::CompatibleLinearOutputRepresentation<Output, Lhs, Rhs> &&
           ((!MutableAsyncBlockTensorView<Output> && !AsyncBlockTensorView<Lhs> && !AsyncBlockTensorView<Rhs>) ||
            (MutableAsyncBlockTensorView<Output> && AsyncBlockTensorView<Lhs> && AsyncBlockTensorView<Rhs>))
void add(Output& output, Lhs const& lhs, Rhs const& rhs)
{
  detail::require_compatible_block_tensor_values(output, lhs);
  detail::require_compatible_block_tensor_values(output, rhs);
  detail::require_stored_key_superset(output, lhs);
  detail::require_stored_key_superset(output, rhs);

  if (detail::same_block_tensor_object(output, lhs))
  {
    if (detail::same_block_tensor_object(output, rhs))
    {
      using value_type = block_tensor_value_t<Output>;
      scale(output, value_type{2});
    }
    else
    {
      add_inplace(output, rhs);
    }
    return;
  }
  if (detail::same_block_tensor_object(output, rhs))
  {
    add_inplace(output, lhs);
    return;
  }

  if constexpr (!MutableAsyncBlockTensorView<Output>)
  {
    detail::execute_linear_block_batch(detail::block_execution_policy_t<Output>{}, output.stored_block_count(),
                                       [&](std::size_t ordinal) {
                                         auto output_block = output.block_by_ordinal(ordinal);
                                         auto const& key = output.stored_keys()[ordinal];
                                         auto lhs_ordinal = detail::find_stored_key_ordinal(lhs, key);
                                         auto rhs_ordinal = detail::find_stored_key_ordinal(rhs, key);
                                         if (lhs_ordinal && rhs_ordinal)
                                         {
                                           auto lhs_block = lhs.block_by_ordinal(*lhs_ordinal);
                                           auto rhs_block = rhs.block_by_ordinal(*rhs_ordinal);
                                           detail::add_block(output_block, lhs_block, rhs_block);
                                         }
                                         else if (lhs_ordinal)
                                         {
                                           auto lhs_block = lhs.block_by_ordinal(*lhs_ordinal);
                                           uni20::copy(output_block, lhs_block);
                                         }
                                         else if (rhs_ordinal)
                                         {
                                           auto rhs_block = rhs.block_by_ordinal(*rhs_ordinal);
                                           uni20::copy(output_block, rhs_block);
                                         }
                                         else
                                           detail::set_zero_block(output_block);
                                       });
  }
  else
  {
    using value_type = block_tensor_value_t<Output>;
    for (std::size_t ordinal = 0; ordinal < output.stored_block_count(); ++ordinal)
    {
      auto& output_block = output.async_block_by_ordinal(ordinal);
      auto const& key = output.stored_keys()[ordinal];
      auto const lhs_found = std::ranges::lower_bound(lhs.stored_keys(), key);
      auto const rhs_found = std::ranges::lower_bound(rhs.stored_keys(), key);
      bool const has_lhs = lhs_found != lhs.stored_keys().end() && *lhs_found == key;
      bool const has_rhs = rhs_found != rhs.stored_keys().end() && *rhs_found == key;
      if (has_lhs && has_rhs)
      {
        auto const lhs_ordinal = static_cast<std::size_t>(lhs_found - lhs.stored_keys().begin());
        auto const rhs_ordinal = static_cast<std::size_t>(rhs_found - rhs.stored_keys().begin());
        uni20::assign_transform(output_block, linalg::add{}, lhs.async_block_by_ordinal(lhs_ordinal),
                                rhs.async_block_by_ordinal(rhs_ordinal));
      }
      else if (has_lhs)
      {
        auto const lhs_ordinal = static_cast<std::size_t>(lhs_found - lhs.stored_keys().begin());
        uni20::assign_transform(output_block, linalg::scale{value_type{1}}, lhs.async_block_by_ordinal(lhs_ordinal));
      }
      else if (has_rhs)
      {
        auto const rhs_ordinal = static_cast<std::size_t>(rhs_found - rhs.stored_keys().begin());
        uni20::assign_transform(output_block, linalg::scale{value_type{1}}, rhs.async_block_by_ordinal(rhs_ordinal));
      }
      else
      {
        uni20::fill(output_block, value_type{});
      }
    }
  }
}

/// \brief Add one BlockTensor value to an existing output in place.
/// \details Input blocks missing from the output cannot be inserted and are
///          rejected before any numerical block is modified. A generalized-diagonal
///          output requires a generalized-diagonal input.
/// \pre Distinct output and input values do not have overlapping numerical storage.
/// \throws std::invalid_argument If the symmetry or boundary values differ, or
///         the output omits an input key.
template <MutableBlockTensorView Output, BlockTensorView Input>
  requires detail::CompatibleBlockTensorValues<Output, Input> &&
           detail::CompatibleLinearOutputRepresentation<Output, Input> &&
           ((!MutableAsyncBlockTensorView<Output> && !AsyncBlockTensorView<Input>) ||
            (MutableAsyncBlockTensorView<Output> && AsyncBlockTensorView<Input>))
void add_inplace(Output& output, Input const& input)
{
  detail::require_compatible_block_tensor_values(output, input);
  detail::require_stored_key_superset(output, input);
  if (detail::same_block_tensor_object(output, input))
  {
    using value_type = block_tensor_value_t<Output>;
    scale(output, value_type{2});
    return;
  }
#if UNI20_BACKEND_CUDA
  if constexpr (detail::CudaPartitionedBlockTensor<Output> && detail::CudaPartitionedBlockTensor<Input> &&
                linalg::cuda_partitioned_scale_scalars<block_tensor_value_t<Output>, block_tensor_value_t<Output>>)
  {
    if (detail::same_packed_layout(output, input))
    {
      using value_type = block_tensor_value_t<Output>;
      linalg::cuda_partitioned_axpy(output.storage().buffer(), value_type{1}, input.storage().buffer());
      return;
    }
  }
#endif

  if constexpr (!MutableAsyncBlockTensorView<Output>)
  {
    detail::execute_linear_block_batch(detail::block_execution_policy_t<Output>{}, input.stored_block_count(),
                                       [&](std::size_t input_ordinal) {
                                         auto const& key = input.stored_keys()[input_ordinal];
                                         auto const output_ordinal = *detail::find_stored_key_ordinal(output, key);
                                         auto output_block = output.block_by_ordinal(output_ordinal);
                                         auto input_block = input.block_by_ordinal(input_ordinal);
                                         detail::add_inplace_block(output_block, input_block);
                                       });
  }
  else
  {
    for (std::size_t input_ordinal = 0; input_ordinal < input.stored_block_count(); ++input_ordinal)
    {
      auto const& key = input.stored_keys()[input_ordinal];
      auto const output_found = std::ranges::lower_bound(output.stored_keys(), key);
      auto const output_ordinal = static_cast<std::size_t>(output_found - output.stored_keys().begin());
      uni20::transform_inplace(output.async_block_by_ordinal(output_ordinal), linalg::add{},
                               input.async_block_by_ordinal(input_ordinal));
    }
  }
}

/// \brief Compute `output += factor * input` without changing either key pattern.
/// \details A generalized-diagonal output requires a generalized-diagonal input.
/// \pre Distinct output and input values do not have overlapping numerical storage.
/// \throws std::invalid_argument If the symmetry or boundary values differ, or
///         the output omits an input key.
template <MutableBlockTensorView Output, class Scalar, BlockTensorView Input>
  requires detail::CompatibleBlockTensorValues<Output, Input> &&
           detail::CompatibleLinearOutputRepresentation<Output, Input> &&
           ((!MutableAsyncBlockTensorView<Output> && !AsyncBlockTensorView<Input>) ||
            (MutableAsyncBlockTensorView<Output> && AsyncBlockTensorView<Input>))
void axpy(Output& output, Scalar factor, Input const& input)
{
  detail::require_compatible_block_tensor_values(output, input);
  detail::require_stored_key_superset(output, input);
  if (detail::same_block_tensor_object(output, input))
  {
    using value_type = block_tensor_value_t<Output>;
    scale(output, value_type{1} + factor);
    return;
  }
#if UNI20_BACKEND_CUDA
  if constexpr (detail::CudaPartitionedBlockTensor<Output> && detail::CudaPartitionedBlockTensor<Input> &&
                linalg::cuda_partitioned_scale_scalars<block_tensor_value_t<Output>, Scalar>)
  {
    if (detail::same_packed_layout(output, input))
    {
      linalg::cuda_partitioned_axpy(output.storage().buffer(), factor, input.storage().buffer());
      return;
    }
  }
#endif

  if constexpr (!MutableAsyncBlockTensorView<Output>)
  {
    detail::execute_linear_block_batch(detail::block_execution_policy_t<Output>{}, input.stored_block_count(),
                                       [&](std::size_t input_ordinal) {
                                         auto const& key = input.stored_keys()[input_ordinal];
                                         auto const output_ordinal = *detail::find_stored_key_ordinal(output, key);
                                         auto output_block = output.block_by_ordinal(output_ordinal);
                                         auto input_block = input.block_by_ordinal(input_ordinal);
                                         detail::axpy_block(output_block, factor, input_block);
                                       });
  }
  else
  {
    for (std::size_t input_ordinal = 0; input_ordinal < input.stored_block_count(); ++input_ordinal)
    {
      auto const& key = input.stored_keys()[input_ordinal];
      auto const output_found = std::ranges::lower_bound(output.stored_keys(), key);
      auto const output_ordinal = static_cast<std::size_t>(output_found - output.stored_keys().begin());
      uni20::transform_inplace(output.async_block_by_ordinal(output_ordinal), linalg::add_scaled{factor},
                               input.async_block_by_ordinal(input_ordinal));
    }
  }
}

/// \brief Return the sum of two BlockTensor values using the union of stored keys.
/// \details A selected generalized-diagonal output policy requires both inputs
///          to have generalized-diagonal block representations. The selected
///          storage must use the same immediate or async execution mode as both inputs.
/// \throws std::invalid_argument If the symmetry or boundary values differ.
/// \tparam OutputStorage Sparse output policy, or `void` to preserve the left policy.
template <class OutputStorage = void, BlockTensorView Lhs, BlockTensorView Rhs>
  requires detail::CompatibleBlockTensorValues<Lhs, Rhs> &&
           SparseBlockStorage<detail::selected_linear_storage_t<OutputStorage, Lhs>> &&
           detail::CompatibleLinearOutputStorage<detail::selected_linear_storage_t<OutputStorage, Lhs>, Lhs, Rhs>
[[nodiscard]] auto add(Lhs const& lhs, Rhs const& rhs)
{
  detail::require_compatible_block_tensor_values(lhs, rhs);
  using storage_type = detail::selected_linear_storage_t<OutputStorage, Lhs>;
  using lhs_type = block_tensor_type_t<Lhs>;
  using result_type = BlockTensor<typename lhs_type::value_type, typename lhs_type::domain_type,
                                  typename lhs_type::codomain_type, storage_type>;
  auto result = result_type(lhs.symmetry(), lhs.domain(), lhs.codomain(), detail::stored_key_union(lhs, rhs));
  add(result, lhs, rhs);
  return result;
}

/// \brief Return the conjugate-linear BlockTensor inner product as a host scalar.
/// \details Only the intersection of stored key sets contributes. Immediate
///          parallel policies compute block partials concurrently and combine
///          them in canonical key order. Async block inputs are awaited by this
///          explicitly blocking operation.
/// \throws std::invalid_argument If the symmetry or boundary values differ.
template <BlockTensorView Lhs, BlockTensorView Rhs>
  requires detail::CompatibleBlockTensorValues<Lhs, Rhs> && RealOrComplex<block_tensor_value_t<Lhs>> &&
               ((!AsyncBlockTensorView<Lhs> && !AsyncBlockTensorView<Rhs>) ||
                (AsyncBlockTensorView<Lhs> && AsyncBlockTensorView<Rhs>))
[[nodiscard]] auto inner_product_host(Lhs const& lhs, Rhs const& rhs) -> block_tensor_value_t<Lhs>
{
  detail::require_compatible_block_tensor_values(lhs, rhs);
  using value_type = block_tensor_value_t<Lhs>;
#if UNI20_BACKEND_CUBLAS
  if constexpr (detail::CudaPartitionedBlockTensor<Lhs> && detail::CudaPartitionedBlockTensor<Rhs> &&
                cublas::CublasLevelOneScalar<value_type>)
  {
    if (detail::same_packed_layout(lhs, rhs))
      return linalg::cuda_partitioned_inner_product_host(lhs.storage().buffer(), rhs.storage().buffer());
  }
#endif
  auto const bindings = detail::stored_key_intersection_bindings(lhs, rhs);
  std::vector<value_type> partials(bindings.size());
  if constexpr (!AsyncBlockTensorView<Lhs>)
  {
    detail::execute_linear_block_batch(detail::reduction_execution_policy_t<Lhs, Rhs>{}, bindings.size(),
                                       [&](std::size_t index) {
                                         auto const& binding = bindings[index];
                                         auto lhs_block = lhs.block_by_ordinal(binding.lhs_ordinal);
                                         auto rhs_block = rhs.block_by_ordinal(binding.rhs_ordinal);
                                         partials[index] = uni20::inner_product_host(lhs_block, rhs_block);
                                       });
  }
  else
  {
    for (std::size_t index = 0; index < bindings.size(); ++index)
    {
      auto const& binding = bindings[index];
      auto lhs_read = lhs.async_block_by_ordinal(binding.lhs_ordinal).read();
      auto rhs_read = rhs.async_block_by_ordinal(binding.rhs_ordinal).read();
      partials[index] = uni20::inner_product_host(lhs_read.get_wait(), rhs_read.get_wait());
    }
  }

  value_type result{};
  value_type compensation{};
  for (auto const& partial : partials)
  {
    if (!uni20::isfinite(result) || !uni20::isfinite(partial))
    {
      result += partial;
      compensation = value_type{};
      continue;
    }
    auto const corrected = partial - compensation;
    auto const updated = result + corrected;
    compensation = (updated - result) - corrected;
    result = updated;
  }
  return result;
}

/// \brief Return the BlockTensor inner product in a host rank-zero Tensor.
/// \throws std::invalid_argument If the symmetry or boundary values differ.
template <BlockTensorView Lhs, BlockTensorView Rhs>
  requires detail::CompatibleBlockTensorValues<Lhs, Rhs> && RealOrComplex<block_tensor_value_t<Lhs>> &&
           ((!AsyncBlockTensorView<Lhs> && !AsyncBlockTensorView<Rhs>) ||
            (AsyncBlockTensorView<Lhs> && AsyncBlockTensorView<Rhs>))
[[nodiscard]] auto inner_product(Lhs const& lhs, Rhs const& rhs)
{
  using value_type = block_tensor_value_t<Lhs>;
  ScalarTensor<value_type> result;
  result[] = inner_product_host(lhs, rhs);
  return result;
}

/// \brief Return the Euclidean BlockTensor norm as a host scalar.
/// \details Legal but unstored blocks contribute zero. Per-block norms are
///          combined in canonical key order.
template <BlockTensorView Tensor>
  requires RealOrComplex<block_tensor_value_t<Tensor>>
[[nodiscard]] auto norm_host(Tensor const& tensor) -> make_real_t<block_tensor_value_t<Tensor>>
{
#if UNI20_BACKEND_CUBLAS
  if constexpr (detail::CudaPartitionedBlockTensor<Tensor> &&
                cublas::CublasLevelOneScalar<block_tensor_value_t<Tensor>>)
    return linalg::cuda_partitioned_norm_host(tensor.storage().buffer());
#endif
  using real_type = make_real_t<block_tensor_value_t<Tensor>>;
  std::vector<real_type> partials(tensor.stored_block_count());
  if constexpr (!AsyncBlockTensorView<Tensor>)
  {
    detail::execute_linear_block_batch(detail::block_execution_policy_t<Tensor>{}, tensor.stored_block_count(),
                                       [&](std::size_t ordinal) {
                                         auto block = tensor.block_by_ordinal(ordinal);
                                         partials[ordinal] = uni20::norm_host(block);
                                       });
  }
  else
  {
    for (std::size_t ordinal = 0; ordinal < tensor.stored_block_count(); ++ordinal)
    {
      auto read = tensor.async_block_by_ordinal(ordinal).read();
      partials[ordinal] = uni20::norm_host(read.get_wait());
    }
  }

  real_type scale{};
  real_type scaled_sum{};
  for (auto const partial : partials)
  {
    if (partial == real_type{}) continue;
    if (!uni20::isfinite(partial)) return partial;
    if (scale < partial)
    {
      auto const ratio = scale / partial;
      scaled_sum = real_type{1} + scaled_sum * ratio * ratio;
      scale = partial;
    }
    else
    {
      auto const ratio = partial / scale;
      scaled_sum += ratio * ratio;
    }
  }
  if (scale == real_type{}) return real_type{};
  using std::sqrt;
  return scale * sqrt(scaled_sum);
}

/// \brief Return the Euclidean BlockTensor norm in a host rank-zero Tensor.
template <BlockTensorView Tensor>
  requires RealOrComplex<block_tensor_value_t<Tensor>>
[[nodiscard]] auto norm(Tensor const& tensor)
{
  using real_type = make_real_t<block_tensor_value_t<Tensor>>;
  ScalarTensor<real_type> result;
  result[] = norm_host(tensor);
  return result;
}

} // namespace uni20
