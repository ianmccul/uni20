#pragma once

#include "common/mdspan.hpp"
#include "cpu.hpp"
#include "mdspan/strides.hpp"

namespace uni20::kernel
{

template <typename T, std::size_t N>
std::pair<bool, bool> transpose_strided(T const* A,T const* B,
  std::vector<size_t> oldExtentA, std::vector<size_t> oldExtentB,
  std::vector<size_t> oldStrideA, std::vector<size_t> oldStrideB,
  std::array<std::pair<std::size_t, std::size_t>, N> const& contractDims,std::vector<T>& outputA,
  std::vector<T>& outputB,cpu_tag)//, T const* In, std::span<std::ptrdiff_t> InStrides, T* Out,                            //std::span<std::ptrdiff_t> OutStrides)
{
  for(size_t i=0;i<outputA.size();i++)  outputA[i] = A[i];
  for(size_t i=0;i<outputB.size();i++)  outputB[i] = B[i];
  return std::pair{false, false};
  
}
namespace cpu
{

/// \brief  Generic M×N×K contraction engine with full K-dim recursion.
///
/// Each group (M, N, K) has extents and two stride arrays:
///   • Mgrp: strides for A and C
///   • Ngrp: strides for B and C
///   • Kgrp: strides for A and B

template <typename T, std::size_t MR, std::size_t NR, std::size_t KR> class GemmLoop {
  public:
    /// \brief  Build the loop engine.
    GemmLoop(static_vector<extent_strides<2>, MR> const& Mgrp, static_vector<extent_strides<2>, NR> const& Ngrp,
             static_vector<extent_strides<2>, KR> const& Kgrp, T alpha, T beta) noexcept
        : Mgrp_(Mgrp), Ngrp_(Ngrp), Kgrp_(Kgrp), alpha_(alpha), beta_(beta)
    {}

    /// \brief  Perform C = β·C + α·(A ⋅ B) over all fused M, N, K dims.
    void run(T const* A0, T const* B0, T* C0) noexcept { this->loopM(0, A0, B0, C0); }

  private:
    static_vector<extent_strides<2>, MR> const Mgrp_;
    static_vector<extent_strides<2>, NR> const Ngrp_;
    static_vector<extent_strides<2>, KR> const Kgrp_;
    T const alpha_, beta_;

    //--- M-dimension recursion (A↔C) ---//
    void loopM(std::size_t dim, T const* a_ptr, T const* b_ptr, T* c_ptr) noexcept
    {
      if (dim == Mgrp_.size())
      {
        loopN(0, a_ptr, b_ptr, c_ptr);
        return;
      }
      auto extent = Mgrp_[dim].extent;
      auto sA = Mgrp_[dim].strides[0];
      auto sC = Mgrp_[dim].strides[1];
      for (decltype(extent) i = 0; i < extent; ++i)
      {
        loopM(dim + 1, a_ptr, b_ptr, c_ptr);
        a_ptr += sA;
        c_ptr += sC;
      }
    }

    //--- N-dimension recursion (B↔C) ---//
    void loopN(std::size_t dim, T const* a_ptr, T const* b_ptr, T* c_ptr) noexcept
    {
      if (dim == Ngrp_.size())
      {
        // At each M×N “cell” we do the dot-product over all K dims:
        T acc{};
        dotK(0, a_ptr, b_ptr, acc);
        *c_ptr = (beta_ * *c_ptr) + (alpha_ * acc);
        return;
      }
      auto extent = Ngrp_[dim].extent;
      auto sB = Ngrp_[dim].strides[0];
      auto sC = Ngrp_[dim].strides[1];
      for (decltype(extent) j = 0; j < extent; ++j)
      {
        loopN(dim + 1, a_ptr, b_ptr, c_ptr);
        b_ptr += sB;
        c_ptr += sC;
      }
    }

    //--- K-dimension recursion (A↔B) accumulation ---//
    void dotK(std::size_t dim, T const* a_ptr, T const* b_ptr, T& acc) noexcept
    {
      if (dim == Kgrp_.size())
      {
        // we’ve arrived at a single element to multiply:
        acc += *a_ptr * *b_ptr;
        return;
      }
      auto extent = Kgrp_[dim].extent;
      auto sA = Kgrp_[dim].strides[0];
      auto sB = Kgrp_[dim].strides[1];
      for (decltype(extent) k = 0; k < extent; ++k)
      {
        dotK(dim + 1, a_ptr, b_ptr, acc);
        a_ptr += sA;
        b_ptr += sB;
      }
    }
};

} // namespace cpu

template <typename T, std::size_t MR, std::size_t NR, std::size_t KR>
void contract_strided(static_vector<extent_strides<2>, MR> const& Mgrp,
                      static_vector<extent_strides<2>, NR> const& Ngrp,
                      static_vector<extent_strides<2>, KR> const& Kgrp, T alpha, T const* A, T const* B, T beta, T* C,
                      cpu_tag)
{
  cpu::GemmLoop Loop(Mgrp, Ngrp, Kgrp, alpha, beta);
  Loop.run(A, B, C);
}

} // namespace uni20::kernel
