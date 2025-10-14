#pragma once

#include "backend/blas/backend_blas.hpp"
#include "blas.hpp"
#include "core/types.hpp"

namespace uni20::kernel
{

template <BlasScalar T, std::size_t MR, std::size_t NR, std::size_t KR>
/// \brief Delegate tensor contraction to the CPU fallback when using a BLAS backend.
/// \tparam T BLAS-compatible scalar type stored in the tensors.
/// \tparam MR Number of fused M dimensions.
/// \tparam NR Number of fused N dimensions.
/// \tparam KR Number of fused K dimensions.
/// \param Mgrp Metadata describing extents and strides for the fused M dimensions.
/// \param Ngrp Metadata describing extents and strides for the fused N dimensions.
/// \param Kgrp Metadata describing extents and strides for the fused K dimensions.
/// \param alpha Scaling factor applied to the contraction output.
/// \param A Pointer to the base of the left-hand operand.
/// \param B Pointer to the base of the right-hand operand.
/// \param beta Scaling factor applied to the pre-existing contents of the destination tensor.
/// \param C Pointer to the base of the destination tensor.
/// \param tag Backend selector tag.
/// \ingroup kernel_blas
void contract_strided(static_vector<extent_strides<2>, MR> const& Mgrp,
                      static_vector<extent_strides<2>, NR> const& Ngrp,
                      static_vector<extent_strides<2>, KR> const& Kgrp, T alpha, T const* A, T const* B, T beta, T* C,
                      blas_tag tag)
{
  static_cast<void>(tag);
  contract_strided(Mgrp, Ngrp, Kgrp, alpha, A, B, beta, C, cpu_tag{});
}

} // namespace uni20::kernel
