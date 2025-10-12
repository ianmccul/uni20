#pragma once

#include "common/mdspan.hpp"
#include "core/scalar_concepts.hpp"
#include "mdspan/strides.hpp"

#include "cpu/contract.hpp" // always available fallback

#if UNI20_BACKEND_BLAS
#include "blas/contract.hpp"
#endif

#if UNI20_BACKEND_MKL
#include "mkl/contract.hpp"
#endif

#if UNI20_BACKEND_CUDA
#include "cuda/contract.hpp"
#endif

namespace uni20::kernel
{

template <typename T, StridedMdspan AType, StridedMdspan BType, std::size_t N, typename U, MutableStridedMdspan CType,
          typename TagType>
  requires(AType::rank() + BType::rank() == CType::rank() + 2 * N)
void contract(T const& alpha, AType A, BType B, std::array<std::pair<std::size_t, std::size_t>, N> const& contractDims,
              U const& beta, CType C, TagType)
{
  auto [Mgroup, Ngroup, Kgroup] = extract_strides(A, B, contractDims, C);
  contract_strided(Mgroup, Ngroup, Kgroup, alpha, A.data_handle(), B.data_handle(), beta, C.data_handle(), TagType{});
}

template <typename T, StridedMdspan AType, StridedMdspan BType, typename U, MutableStridedMdspan CType,
          typename TagType, std::size_t N>
void contract(T const& alpha, AType A, BType B,
              const std::pair<std::size_t, std::size_t> (&dims)[N], // array reference, size known at compile time
              U const& beta, CType C, TagType t)
{
  contract(alpha, A, B, std::to_array(dims), beta, C, t);
}

} // namespace uni20::kernel
