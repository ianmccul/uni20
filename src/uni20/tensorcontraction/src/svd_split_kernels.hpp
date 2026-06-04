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

} // namespace uni20::tensorcontraction::detail
