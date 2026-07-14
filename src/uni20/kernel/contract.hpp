#pragma once

#include <uni20/common/mdspan.hpp>
#include <uni20/core/scalar_concepts.hpp>
#include <uni20/mdspan/strides.hpp>

/**
 * \defgroup kernel Tensor kernels
 * \brief Low-level tensor kernels operating on resolved views.
 */

#include <uni20/kernel/cpu/contract.hpp>

namespace uni20::kernel
{

template <typename T, StridedMdspan AType, StridedMdspan BType, std::size_t N, typename U, MutableStridedMdspan CType>
  requires(AType::rank() + BType::rank() == CType::rank() + 2 * N) && DefaultAccessorMdspan<AType> &&
          DefaultAccessorMdspan<BType> && DefaultAccessorMdspan<CType>
/// \brief Execute a tensor contraction with the always-available CPU reference kernel.
/// \tparam T Scalar used for scaling the contraction inputs and output.
/// \tparam AType Strided mdspan describing the left-hand tensor operand.
/// \tparam BType Strided mdspan describing the right-hand tensor operand.
/// \tparam N Number of contracted index pairs.
/// \tparam U Scalar type used to scale the destination tensor.
/// \tparam CType Mutable strided mdspan describing the output tensor.
/// \param alpha Scaling factor for the contraction result.
/// \param A Left-hand tensor operand.
/// \param B Right-hand tensor operand.
/// \param contractDims Pairing of contracted dimensions between \p A and \p B.
/// \param beta Scaling factor applied to the pre-existing contents of \p C.
/// \param C Destination tensor.
/// \ingroup kernel_ops
void contract(T const& alpha, AType A, BType B, std::array<std::pair<std::size_t, std::size_t>, N> const& contractDims,
              U const& beta, CType C)
{
  auto [Mgroup, Ngroup, Kgroup] = extract_strides(A, B, contractDims, C);
  contract_strided(Mgroup, Ngroup, Kgroup, alpha, A.data_handle(), B.data_handle(), beta, C.data_handle());
}

template <typename T, StridedMdspan AType, StridedMdspan BType, typename U, MutableStridedMdspan CType, std::size_t N>
  requires DefaultAccessorMdspan<AType> && DefaultAccessorMdspan<BType> && DefaultAccessorMdspan<CType>
/// \brief Forward compile-time dimension pairs to the CPU reference contraction.
/// \tparam T Scalar used for scaling the contraction inputs and output.
/// \tparam AType Strided mdspan describing the left-hand tensor operand.
/// \tparam BType Strided mdspan describing the right-hand tensor operand.
/// \tparam U Scalar type used to scale the destination tensor.
/// \tparam CType Mutable strided mdspan describing the output tensor.
/// \tparam N Number of contracted index pairs.
/// \param alpha Scaling factor for the contraction result.
/// \param A Left-hand tensor operand.
/// \param B Right-hand tensor operand.
/// \param dims Compile-time array reference listing contracted dimension pairs.
/// \param beta Scaling factor applied to the pre-existing contents of \p C.
/// \param C Destination tensor.
/// \ingroup kernel_ops
void contract(T const& alpha, AType A, BType B,
              std::pair<std::size_t, std::size_t> const (&dims)[N], // array reference, size known at compile time
              U const& beta, CType C)
{
  contract(alpha, A, B, std::to_array(dims), beta, C);
}

} // namespace uni20::kernel
