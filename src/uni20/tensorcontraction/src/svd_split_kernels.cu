#include "svd_split_kernels.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace uni20::tensorcontraction::detail
{

namespace
{

constexpr int threads_per_block = 256;

__global__ void split_left_kernel(double const* u, double const* singular_values, double const* vt, double* left,
                                  std::uint64_t left_bond_dim, std::uint64_t left_physical_dim, std::uint64_t kept_rank,
                                  std::uint64_t rows, std::uint64_t minmn, bool transposed, bool absorb_left)
{
  auto const total = left_bond_dim * left_physical_dim * kept_rank;
  for (std::uint64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total; index += gridDim.x * blockDim.x)
  {
    auto const bond = index % kept_rank;
    auto const left_bond = (index / kept_rank) % left_bond_dim;
    auto const left_phys = index / (kept_rank * left_bond_dim);
    auto const row = left_bond * left_physical_dim + left_phys;
    double value = transposed ? vt[row * minmn + bond] : u[bond * rows + row];
    if (absorb_left)
    {
      value *= singular_values[bond];
    }
    left[index] = value;
  }
}

__global__ void split_right_kernel(double const* u, double const* singular_values, double const* vt, double* right,
                                   std::uint64_t right_physical_dim, std::uint64_t right_bond_dim,
                                   std::uint64_t kept_rank, std::uint64_t cols, std::uint64_t minmn, bool transposed,
                                   bool absorb_left)
{
  auto const total = right_physical_dim * kept_rank * right_bond_dim;
  for (std::uint64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total; index += gridDim.x * blockDim.x)
  {
    auto const right_bond = index % right_bond_dim;
    auto const bond = (index / right_bond_dim) % kept_rank;
    auto const right_phys = index / (right_bond_dim * kept_rank);
    auto const col = right_phys * right_bond_dim + right_bond;
    double value = transposed ? u[bond * cols + col] : vt[col * minmn + bond];
    if (!absorb_left)
    {
      value *= singular_values[bond];
    }
    right[index] = value;
  }
}

auto block_count(std::size_t items) -> int
{
  auto const blocks =
      (items + static_cast<std::size_t>(threads_per_block - 1)) / static_cast<std::size_t>(threads_per_block);
  return static_cast<int>(std::min<std::size_t>(blocks, 65535));
}

} // namespace

void launch_svd_split_kernels(double const* u, double const* singular_values, double const* vt, double* left,
                              double* right, std::size_t left_bond_dim, std::size_t left_physical_dim,
                              std::size_t right_physical_dim, std::size_t right_bond_dim, std::size_t kept_rank,
                              std::size_t minmn, bool transposed, bool absorb_left, cudaStream_t stream)
{
  if (kept_rank == 0)
  {
    return;
  }

  auto const rows = left_bond_dim * left_physical_dim;
  auto const cols = right_physical_dim * right_bond_dim;
  auto const left_items = rows * kept_rank;
  auto const right_items = right_physical_dim * kept_rank * right_bond_dim;

  split_left_kernel<<<block_count(left_items), threads_per_block, 0, stream>>>(
      u, singular_values, vt, left, left_bond_dim, left_physical_dim, kept_rank, rows, minmn, transposed, absorb_left);
  split_right_kernel<<<block_count(right_items), threads_per_block, 0, stream>>>(
      u, singular_values, vt, right, right_physical_dim, right_bond_dim, kept_rank, cols, minmn, transposed,
      absorb_left);
}

} // namespace uni20::tensorcontraction::detail
