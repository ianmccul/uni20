#pragma once

/**
 * \file partitioned_buffer_linear.hpp
 * \ingroup linalg
 * \brief Allocation-wide CUDA linear operations for partitioned buffers.
 */

#include <uni20/backend/cuda/buffer.hpp>
#include <uni20/config.hpp>
#include <uni20/linalg/backends/cuda/elementwise_arithmetic.hpp>
#include <uni20/linalg/backends/cuda/elementwise_copy.hpp>
#include <uni20/linalg/backends/cuda/elementwise_plan.hpp>
#include <uni20/linalg/backends/cuda/elementwise_scale.hpp>

#if UNI20_BACKEND_CUBLAS
#include <uni20/backend/cublas/execution.hpp>
#include <uni20/backend/cublas/level1.hpp>
#endif

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace uni20::linalg
{
namespace partitioned_buffer_detail
{

template <std::size_t OperandCount, std::size_t MaximumRank>
[[nodiscard]] auto make_contiguous_plan(std::size_t size)
    -> detail::cuda_reference::LoweredStridedElementwisePlan<OperandCount, MaximumRank>
{
  using namespace detail::cuda_reference;
  LoweredStridedElementwisePlan<OperandCount, MaximumRank> result;
  if (size <= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
  {
    result.index_kind = ElementwiseIndexKind::index_32;
    result.plan_32.element_count = static_cast<std::uint32_t>(size);
    result.plan_32.compact_rank = size == 0 ? 0 : 1;
    if (size != 0)
    {
      result.plan_32.extents[0] = static_cast<std::uint32_t>(size);
      result.plan_32.strides[0].fill(1);
    }
  }
  else
  {
    result.index_kind = ElementwiseIndexKind::index_64;
    result.plan_64.element_count = static_cast<std::uint64_t>(size);
    result.plan_64.compact_rank = size == 0 ? 0 : 1;
    if (size != 0)
    {
      result.plan_64.extents[0] = static_cast<std::uint64_t>(size);
      result.plan_64.strides[0].fill(1);
    }
  }
  return result;
}

template <class... Accesses> void publish(cuda::Stream const& stream, Accesses&... accesses)
{
  cuda::AccessCompletion completion(stream);
  (completion.release(accesses), ...);
}

} // namespace partitioned_buffer_detail

/// \brief Whether allocation-wide CUDA fill operations support `Scalar`.
template <class Scalar>
inline constexpr bool cuda_partitioned_fill_scalar = detail::cuda_reference::supports_elementwise_arithmetic<Scalar>;

/// \brief Whether allocation-wide CUDA scale operations support this scalar pair.
template <class Scalar, class Factor>
inline constexpr bool cuda_partitioned_scale_scalars =
    detail::cuda_reference::supports_elementwise_scale<Scalar, std::remove_cvref_t<Factor>>;

/// \brief Set a complete partitioned CUDA allocation to numerical zero.
template <class Scalar>
  requires cuda_partitioned_fill_scalar<Scalar>
void cuda_partitioned_set_zero(cuda::PartitionedCudaBuffer<Scalar>& output)
{
  if (output.size() == 0) return;
  using namespace detail::cuda_reference;
  auto stream = output.resources().streams().acquire();
  auto access = output.write_synchronized_with(stream);
  auto const plan =
      partitioned_buffer_detail::make_contiguous_plan<1, elementwise_arithmetic_maximum_rank>(output.size());
  cuda::ScopedDevice guard(output.device().ordinal());
  plan.visit([&](auto const& concrete) {
    enqueue_elementwise_fill(access.data(), Scalar{}, concrete, stream.native_handle(), stream.device());
  });
  partitioned_buffer_detail::publish(stream, access);
}

/// \brief Scale a complete partitioned CUDA allocation in place.
template <class Scalar, class Factor>
  requires cuda_partitioned_scale_scalars<Scalar, Factor>
void cuda_partitioned_scale(cuda::PartitionedCudaBuffer<Scalar>& output, Factor factor)
{
  if (output.size() == 0) return;
  using namespace detail::cuda_reference;
  auto stream = output.resources().streams().acquire();
  auto access = output.write_synchronized_with(stream);
  auto const plan =
      partitioned_buffer_detail::make_contiguous_plan<1, elementwise_arithmetic_maximum_rank>(output.size());
  cuda::ScopedDevice guard(output.device().ordinal());
  plan.visit([&](auto const& concrete) {
    enqueue_elementwise_inplace_scale(access.data(), factor, concrete, stream.native_handle(), stream.device());
  });
  partitioned_buffer_detail::publish(stream, access);
}

/// \brief Overwrite one partitioned CUDA allocation with a scaled peer allocation.
template <class Scalar, class Factor>
  requires cuda_partitioned_scale_scalars<Scalar, Factor>
void cuda_partitioned_assign_scale(cuda::PartitionedCudaBuffer<Scalar>& output, Factor factor,
                                   cuda::PartitionedCudaBuffer<Scalar> const& input)
{
  CHECK_EQUAL(output.size(), input.size());
  if (output.size() == 0) return;
  CHECK_EQUAL(output.device().ordinal(), input.device().ordinal());
  using namespace detail::cuda_reference;
  auto stream = output.resources().streams().acquire();
  auto output_access = output.write_synchronized_with(stream);
  auto input_access = input.read_synchronized_with(stream);
  auto const plan = partitioned_buffer_detail::make_contiguous_plan<2, elementwise_scale_maximum_rank>(output.size());
  cuda::ScopedDevice guard(output.device().ordinal());
  plan.visit([&](auto const& concrete) {
    enqueue_elementwise_scale(output_access.data(), input_access.data(), factor, concrete, stream.native_handle(),
                              stream.device());
  });
  partitioned_buffer_detail::publish(stream, output_access, input_access);
}

/// \brief Add a scaled partitioned CUDA allocation to another in place.
template <class Scalar, class Factor>
  requires cuda_partitioned_scale_scalars<Scalar, Factor>
void cuda_partitioned_axpy(cuda::PartitionedCudaBuffer<Scalar>& output, Factor factor,
                           cuda::PartitionedCudaBuffer<Scalar> const& input)
{
  CHECK_EQUAL(output.size(), input.size());
  if (output.size() == 0) return;
  CHECK_EQUAL(output.device().ordinal(), input.device().ordinal());
  using namespace detail::cuda_reference;
  auto stream = output.resources().streams().acquire();
  auto output_access = output.write_synchronized_with(stream);
  auto input_access = input.read_synchronized_with(stream);
  auto const plan =
      partitioned_buffer_detail::make_contiguous_plan<2, elementwise_arithmetic_maximum_rank>(output.size());
  cuda::ScopedDevice guard(output.device().ordinal());
  plan.visit([&](auto const& concrete) {
    enqueue_elementwise_add_scaled(output_access.data(), input_access.data(), factor, concrete, stream.native_handle(),
                                   stream.device());
  });
  partitioned_buffer_detail::publish(stream, output_access, input_access);
}

#if UNI20_BACKEND_CUBLAS

/// \brief Return the conjugate-linear inner product of two complete partitioned CUDA allocations.
template <uni20::cublas::CublasLevelOneScalar Scalar>
[[nodiscard]] auto cuda_partitioned_inner_product_host(cuda::PartitionedCudaBuffer<Scalar> const& lhs,
                                                       cuda::PartitionedCudaBuffer<Scalar> const& rhs) -> Scalar
{
  CHECK_EQUAL(lhs.size(), rhs.size());
  CHECK(lhs.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max()), lhs.size());
  if (lhs.size() == 0) return Scalar{};
  CHECK_EQUAL(lhs.device().ordinal(), rhs.device().ordinal());
  auto execution = uni20::cublas::execution_pool(lhs.resources()).acquire();
  auto lhs_access = lhs.read_synchronized_with(execution.stream());
  auto rhs_access = rhs.read_synchronized_with(execution.stream());
  auto result = uni20::cublas::dotc(execution, static_cast<int>(lhs.size()), lhs_access.data(), rhs_access.data());
  lhs_access.release_after_synchronization();
  rhs_access.release_after_synchronization();
  return result;
}

/// \brief Return the Euclidean norm of one complete partitioned CUDA allocation.
template <uni20::cublas::CublasLevelOneScalar Scalar>
[[nodiscard]] auto cuda_partitioned_norm_host(cuda::PartitionedCudaBuffer<Scalar> const& input) -> make_real_t<Scalar>
{
  CHECK(input.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max()), input.size());
  if (input.size() == 0) return make_real_t<Scalar>{};
  auto execution = uni20::cublas::execution_pool(input.resources()).acquire();
  auto access = input.read_synchronized_with(execution.stream());
  auto result = uni20::cublas::nrm2(execution, static_cast<int>(input.size()), access.data());
  access.release_after_synchronization();
  return result;
}

#endif

} // namespace uni20::linalg
