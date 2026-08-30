#pragma once

#include <uni20/backend/cuda/cuda_error.hpp>
#include <uni20/core/compiler_attributes.hpp>

#include <cuda_runtime.h>

#include <cstddef>
#include <string_view>
#include <type_traits>
#include <utility>

namespace uni20::linalg::detail::cuda_reference
{

template <class Accessor> struct ElementwiseOperand
{
    using accessor_type = Accessor;
    using data_handle_type = typename accessor_type::data_handle_type;
    using offset_type = typename accessor_type::offset_type;

    data_handle_type handle;
    [[no_unique_address]] accessor_type accessor{};

    template <class Offset> [[nodiscard]] __device__ constexpr decltype(auto) access(Offset offset) const
    {
      return accessor.access(handle, static_cast<offset_type>(offset));
    }

    template <class Offset> [[nodiscard]] __device__ constexpr auto load(Offset offset) const
    {
      auto&& reference = this->access(offset);
      using reference_type = std::remove_cvref_t<decltype(reference)>;
      if constexpr (requires { typename reference_type::execution_type; })
        return static_cast<typename reference_type::execution_type>(reference);
      else
        return static_cast<reference_type>(reference);
    }
};

template <class Handle, class Accessor>
ElementwiseOperand(Handle, Accessor) -> ElementwiseOperand<std::remove_cvref_t<Accessor>>;

namespace elementwise_kernel_detail
{

template <bool ReadsOutput, class Operation, class OutputOperand, class Offsets, class... InputOperands,
          std::size_t... Input>
__device__ void apply(Operation const& operation, OutputOperand const& output, Offsets const& offsets,
                      std::index_sequence<Input...>, InputOperands const&... inputs)
{
  auto&& output_value = output.access(offsets[0]);
  if constexpr (ReadsOutput)
    output_value = operation(output.load(offsets[0]), inputs.load(offsets[Input + 1])...);
  else
    output_value = operation(inputs.load(offsets[Input + 1])...);
}

template <bool ReadsOutput, class Plan, class Operation, class OutputOperand, class... InputOperands>
__global__ void elementwise_kernel(Plan plan, Operation operation, OutputOperand output, InputOperands... inputs)
{
  static_assert(Plan::operand_count == sizeof...(InputOperands) + 1);
  using logical_index_type = typename Plan::logical_index_type;
  auto index = static_cast<logical_index_type>(blockIdx.x * blockDim.x + threadIdx.x);
  auto const grid_stride = static_cast<logical_index_type>(blockDim.x * gridDim.x);
  while (index < plan.element_count)
  {
    auto const offsets = plan.offsets(index);
    apply<ReadsOutput>(operation, output, offsets, std::index_sequence_for<InputOperands...>{}, inputs...);
    index += grid_stride;
  }
}

} // namespace elementwise_kernel_detail

template <bool ReadsOutput, class Plan, class Operation, class OutputOperand, class... InputOperands>
void launch_elementwise_kernel(Plan const& plan, Operation operation, OutputOperand output, cudaStream_t stream,
                               int device, std::string_view operation_name, InputOperands... inputs)
{
  if (plan.element_count == 0) return;

  constexpr unsigned int threads = 256;
  auto required_blocks = (plan.element_count + threads - 1) / threads;
  constexpr std::size_t maximum_blocks = 65535;
  auto const blocks = static_cast<unsigned int>(required_blocks < maximum_blocks ? required_blocks : maximum_blocks);
  elementwise_kernel_detail::elementwise_kernel<ReadsOutput>
      <<<blocks, threads, 0, stream>>>(plan, std::move(operation), std::move(output), std::move(inputs)...);
  uni20::cuda::check(cudaGetLastError(), operation_name, device);
}

} // namespace uni20::linalg::detail::cuda_reference
