#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace uni20::tensorcontraction::detail
{

void launch_svd_split_kernels(double const* u, double const* singular_values, double const* vt, double* left,
                              double* right, std::size_t left_bond_dim, std::size_t left_physical_dim,
                              std::size_t right_physical_dim, std::size_t right_bond_dim, std::size_t kept_rank,
                              std::size_t minmn, bool transposed, bool absorb_left, cudaStream_t stream);

void launch_pack_svd_input_block_kernel(double const* source_block, double* destination, std::size_t left_bond_dim,
                                        std::size_t right_bond_dim, std::size_t left_physical_dim,
                                        std::size_t right_physical_dim, std::size_t left_physical,
                                        std::size_t right_physical, bool transposed, cudaStream_t stream);

void launch_pack_svd_input_subblock_kernel(double const* source_block, double* destination, std::size_t source_rows,
                                           std::size_t source_cols, std::size_t destination_rows,
                                           std::size_t destination_cols, std::size_t row_offset, std::size_t col_offset,
                                           bool transposed, cudaStream_t stream);

void launch_scatter_svd_left_subblock_kernel(double const* u, double const* singular_values, double const* vt,
                                             double* destination, std::size_t destination_rows, std::size_t sector_rows,
                                             std::size_t minmn, std::size_t row_offset, std::size_t kept_rank,
                                             bool transposed, bool absorb_left, cudaStream_t stream);

void launch_scatter_svd_right_subblock_kernel(double const* u, double const* singular_values, double const* vt,
                                              double* destination, std::size_t destination_cols,
                                              std::size_t sector_cols, std::size_t minmn, std::size_t col_offset,
                                              std::size_t kept_rank, bool transposed, bool absorb_left,
                                              cudaStream_t stream);

} // namespace uni20::tensorcontraction::detail
