#pragma once

#include "common/mdspan.hpp"
#include "cpu.hpp"
#include "mdspan/strides.hpp"

namespace uni20::kernel
{

namespace cpu
{

/// \brief Generic M×N×K contraction engine with full K-dimension recursion.
/// \details Each dimension group stores extents and operand strides used to perform
///          fused tensor contractions without materialising temporaries.
/// \tparam T Scalar type stored in the tensors.
/// \tparam MR Number of fused M dimensions.
/// \tparam NR Number of fused N dimensions.
/// \tparam KR Number of fused K dimensions.
/// \ingroup kernel_cpu
template <typename T, std::size_t MR, std::size_t NR, std::size_t KR> class GemmLoop {
  public:
    /// \brief Build the loop engine for a fused contraction.
    /// \param Mgrp Metadata describing extents and strides for the fused M dimensions.
    /// \param Ngrp Metadata describing extents and strides for the fused N dimensions.
    /// \param Kgrp Metadata describing extents and strides for the fused K dimensions.
    /// \param alpha Scaling factor applied to the contraction output.
    /// \param beta Scaling factor applied to the pre-existing contents of the destination tensor.
    /// \ingroup kernel_cpu
    GemmLoop(static_vector<extent_strides<2>, MR> const& Mgrp, static_vector<extent_strides<2>, NR> const& Ngrp,
             static_vector<extent_strides<2>, KR> const& Kgrp, T alpha, T beta) noexcept
        : Mgrp_(Mgrp), Ngrp_(Ngrp), Kgrp_(Kgrp), alpha_(alpha), beta_(beta)
    {}

    /// \brief Perform C = β·C + α·(A ⋅ B) over all fused M, N, and K dimensions.
    /// \param A0 Pointer to the base of the left-hand operand.
    /// \param B0 Pointer to the base of the right-hand operand.
    /// \param C0 Pointer to the base of the destination tensor.
    /// \ingroup kernel_cpu
    void run(T const* A0, T const* B0, T* C0) noexcept { this->loopM(0, A0, B0, C0); }

  private:
    static_vector<extent_strides<2>, MR> const Mgrp_;
    static_vector<extent_strides<2>, NR> const Ngrp_;
    static_vector<extent_strides<2>, KR> const Kgrp_;
    T const alpha_, beta_;

    /// \brief Recursively advances through the fused M dimensions.
    /// \param dim Index of the current M dimension.
    /// \param a_ptr Pointer to the active location within the left-hand operand.
    /// \param b_ptr Pointer to the active location within the right-hand operand.
    /// \param c_ptr Pointer to the active location within the destination tensor.
    /// \ingroup internal
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

    /// \brief Recursively advances through the fused N dimensions.
    /// \param dim Index of the current N dimension.
    /// \param a_ptr Pointer to the active location within the left-hand operand.
    /// \param b_ptr Pointer to the active location within the right-hand operand.
    /// \param c_ptr Pointer to the active location within the destination tensor.
    /// \ingroup internal
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

    /// \brief Recursively accumulates dot products across the fused K dimensions.
    /// \param dim Index of the current K dimension.
    /// \param a_ptr Pointer to the active location within the left-hand operand.
    /// \param b_ptr Pointer to the active location within the right-hand operand.
    /// \param acc Running contraction accumulator.
    /// \ingroup internal
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
/// \brief Execute the CPU tensor contraction using precomputed stride groupings.
/// \tparam T Scalar type stored in the tensors.
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
/// \ingroup kernel_cpu
void contract_strided(static_vector<extent_strides<2>, MR> const& Mgrp,
                      static_vector<extent_strides<2>, NR> const& Ngrp,
                      static_vector<extent_strides<2>, KR> const& Kgrp, T alpha, T const* A, T const* B, T beta, T* C,
                      cpu_tag tag)
{
  static_cast<void>(tag);
  cpu::GemmLoop Loop(Mgrp, Ngrp, Kgrp, alpha, beta);
  Loop.run(A, B, C);
}

} // namespace uni20::kernel
