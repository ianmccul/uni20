#include "vector_algebra_kernels.hpp"

#include <cstddef>

namespace uni20::tensorcontraction::detail
{
namespace
{

constexpr int threads_per_block = 256;

auto block_count(std::size_t size) -> int
{
  return static_cast<int>((size + static_cast<std::size_t>(threads_per_block) - 1) /
                          static_cast<std::size_t>(threads_per_block));
}

__global__ void slab_scale_kernel(double* values, std::size_t size, double alpha)
{
  auto const index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < size)
  {
    values[index] *= alpha;
  }
}

__global__ void slab_axpy_kernel(double const* x, double* y, std::size_t size, double alpha)
{
  auto const index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < size)
  {
    y[index] += alpha * x[index];
  }
}

} // namespace

void launch_slab_scale_kernel(double* values, std::size_t size, double alpha, cudaStream_t stream)
{
  if (size == 0)
  {
    return;
  }
  slab_scale_kernel<<<block_count(size), threads_per_block, 0, stream>>>(values, size, alpha);
}

void launch_slab_axpy_kernel(double const* x, double* y, std::size_t size, double alpha, cudaStream_t stream)
{
  if (size == 0)
  {
    return;
  }
  slab_axpy_kernel<<<block_count(size), threads_per_block, 0, stream>>>(x, y, size, alpha);
}

} // namespace uni20::tensorcontraction::detail
