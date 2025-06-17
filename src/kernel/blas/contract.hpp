#pragma once

#include "backend/blas/backend_blas.hpp"
#include "blas.hpp"
#include "core/types.hpp"

namespace uni20::kernel
{

template <BlasScalar T, std::size_t MR, std::size_t NR, std::size_t KR>
void contract_strided(static_vector<extent_strides<2>, MR> const& Mgrp,
                      static_vector<extent_strides<2>, NR> const& Ngrp,
                      static_vector<extent_strides<2>, KR> const& Kgrp, T alpha, T const* A, T const* B, T beta, T* C,
                      blas_tag)
{
  contract_strided(Mgrp, Ngrp, Kgrp, alpha, A, B, beta, C, cpu_tag{});
}

} // namespace uni20::kernel
