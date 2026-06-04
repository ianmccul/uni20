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

__global__ void pack_svd_input_block_kernel(double const* source_block, double* destination,
                                            std::uint64_t left_bond_dim, std::uint64_t right_bond_dim,
                                            std::uint64_t left_physical_dim, std::uint64_t right_physical_dim,
                                            std::uint64_t left_physical, std::uint64_t right_physical, bool transposed)
{
  auto const total = left_bond_dim * right_bond_dim;
  auto const rows = left_bond_dim * left_physical_dim;
  auto const cols = right_physical_dim * right_bond_dim;
  for (std::uint64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total; index += gridDim.x * blockDim.x)
  {
    auto const right_bond = index % right_bond_dim;
    auto const left_bond = index / right_bond_dim;
    auto const row = left_bond * left_physical_dim + left_physical;
    auto const col = right_physical * right_bond_dim + right_bond;
    auto const value = source_block[left_bond * right_bond_dim + right_bond];
    destination[transposed ? row * cols + col : col * rows + row] = value;
  }
}

__global__ void pack_svd_input_subblock_kernel(double const* source_block, double* destination,
                                               std::uint64_t source_rows, std::uint64_t source_cols,
                                               std::uint64_t destination_rows, std::uint64_t destination_cols,
                                               std::uint64_t row_offset, std::uint64_t col_offset, bool transposed)
{
  auto const total = source_rows * source_cols;
  for (std::uint64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total; index += gridDim.x * blockDim.x)
  {
    auto const source_col = index % source_cols;
    auto const source_row = index / source_cols;
    auto const destination_row = row_offset + source_row;
    auto const destination_col = col_offset + source_col;
    auto const value = source_block[source_row * source_cols + source_col];
    destination[transposed ? destination_row * destination_cols + destination_col
                           : destination_col * destination_rows + destination_row] = value;
  }
}

__global__ void scatter_svd_left_subblock_kernel(double const* u, double const* singular_values, double const* vt,
                                                 double* destination, std::uint64_t destination_rows,
                                                 std::uint64_t sector_rows, std::uint64_t minmn,
                                                 std::uint64_t row_offset, std::uint64_t kept_rank, bool transposed,
                                                 bool absorb_left)
{
  auto const total = destination_rows * kept_rank;
  for (std::uint64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total; index += gridDim.x * blockDim.x)
  {
    auto const bond = index % kept_rank;
    auto const destination_row = index / kept_rank;
    auto const sector_row = row_offset + destination_row;
    double value = transposed ? vt[sector_row * minmn + bond] : u[bond * sector_rows + sector_row];
    if (absorb_left)
    {
      value *= singular_values[bond];
    }
    destination[destination_row * kept_rank + bond] = value;
  }
}

__global__ void scatter_svd_right_subblock_kernel(double const* u, double const* singular_values, double const* vt,
                                                  double* destination, std::uint64_t destination_cols,
                                                  std::uint64_t sector_cols, std::uint64_t minmn,
                                                  std::uint64_t col_offset, std::uint64_t kept_rank, bool transposed,
                                                  bool absorb_left)
{
  auto const total = kept_rank * destination_cols;
  for (std::uint64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total; index += gridDim.x * blockDim.x)
  {
    auto const destination_col = index % destination_cols;
    auto const bond = index / destination_cols;
    auto const sector_col = col_offset + destination_col;
    double value = transposed ? u[bond * sector_cols + sector_col] : vt[sector_col * minmn + bond];
    if (!absorb_left)
    {
      value *= singular_values[bond];
    }
    destination[bond * destination_cols + destination_col] = value;
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

void launch_pack_svd_input_block_kernel(double const* source_block, double* destination, std::size_t left_bond_dim,
                                        std::size_t right_bond_dim, std::size_t left_physical_dim,
                                        std::size_t right_physical_dim, std::size_t left_physical,
                                        std::size_t right_physical, bool transposed, cudaStream_t stream)
{
  auto const items = left_bond_dim * right_bond_dim;
  pack_svd_input_block_kernel<<<block_count(items), threads_per_block, 0, stream>>>(
      source_block, destination, left_bond_dim, right_bond_dim, left_physical_dim, right_physical_dim, left_physical,
      right_physical, transposed);
}

void launch_pack_svd_input_subblock_kernel(double const* source_block, double* destination, std::size_t source_rows,
                                           std::size_t source_cols, std::size_t destination_rows,
                                           std::size_t destination_cols, std::size_t row_offset, std::size_t col_offset,
                                           bool transposed, cudaStream_t stream)
{
  auto const items = source_rows * source_cols;
  if (items == 0)
  {
    return;
  }
  pack_svd_input_subblock_kernel<<<block_count(items), threads_per_block, 0, stream>>>(
      source_block, destination, source_rows, source_cols, destination_rows, destination_cols, row_offset, col_offset,
      transposed);
}

void launch_scatter_svd_left_subblock_kernel(double const* u, double const* singular_values, double const* vt,
                                             double* destination, std::size_t destination_rows, std::size_t sector_rows,
                                             std::size_t minmn, std::size_t row_offset, std::size_t kept_rank,
                                             bool transposed, bool absorb_left, cudaStream_t stream)
{
  auto const items = destination_rows * kept_rank;
  if (items == 0)
  {
    return;
  }
  scatter_svd_left_subblock_kernel<<<block_count(items), threads_per_block, 0, stream>>>(
      u, singular_values, vt, destination, destination_rows, sector_rows, minmn, row_offset, kept_rank, transposed,
      absorb_left);
}

void launch_scatter_svd_right_subblock_kernel(double const* u, double const* singular_values, double const* vt,
                                              double* destination, std::size_t destination_cols,
                                              std::size_t sector_cols, std::size_t minmn, std::size_t col_offset,
                                              std::size_t kept_rank, bool transposed, bool absorb_left,
                                              cudaStream_t stream)
{
  auto const items = destination_cols * kept_rank;
  if (items == 0)
  {
    return;
  }
  scatter_svd_right_subblock_kernel<<<block_count(items), threads_per_block, 0, stream>>>(
      u, singular_values, vt, destination, destination_cols, sector_cols, minmn, col_offset, kept_rank, transposed,
      absorb_left);
}

} // namespace uni20::tensorcontraction::detail
