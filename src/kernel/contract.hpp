#pragma once

#include "common/mdspan.hpp"
#include "core/scalar_concepts.hpp"
#include "mdspan/strides.hpp"

/**
 * \defgroup kernel Tensor kernel dispatch
 * \ingroup kernel_ops
 * \brief Front-end dispatchers that route tensor contractions to backend kernels.
 */

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
    /// \brief Dispatch a tensor contraction to the backend associated with \p TagType.
    /// \tparam T Scalar used for scaling the contraction inputs and output.
    /// \tparam AType Strided mdspan describing the left-hand tensor operand.
    /// \tparam BType Strided mdspan describing the right-hand tensor operand.
    /// \tparam N Number of contracted index pairs.
    /// \tparam U Scalar type used to scale the destination tensor.
    /// \tparam CType Mutable strided mdspan describing the output tensor.
    /// \tparam TagType Backend selection tag.
    /// \param alpha Scaling factor for the contraction result.
    /// \param A Left-hand tensor operand.
    /// \param B Right-hand tensor operand.
    /// \param contractDims Pairing of contracted dimensions between \p A and \p B.
    /// \param beta Scaling factor applied to the pre-existing contents of \p C.
    /// \param C Destination tensor.
    /// \param tag Backend selector instance.
    /// \ingroup kernel_ops
    void contract(T const& alpha, AType A, BType B,
                  std::array<std::pair<std::size_t, std::size_t>, N> const& contractDims, U const& beta, CType C,
                  TagType tag)
{
  auto [Mgroup, Ngroup, Kgroup] = extract_strides(A, B, contractDims, C);
  contract_strided(Mgroup, Ngroup, Kgroup, alpha, A.data_handle(), B.data_handle(), beta, C.data_handle(), tag);
}

template <typename T, StridedMdspan AType, StridedMdspan BType, typename U, MutableStridedMdspan CType,
          typename TagType, std::size_t N>
/// \brief Overload forwarding compile-time dimension pairs to the runtime dispatcher.
/// \tparam T Scalar used for scaling the contraction inputs and output.
/// \tparam AType Strided mdspan describing the left-hand tensor operand.
/// \tparam BType Strided mdspan describing the right-hand tensor operand.
/// \tparam U Scalar type used to scale the destination tensor.
/// \tparam CType Mutable strided mdspan describing the output tensor.
/// \tparam TagType Backend selection tag.
/// \tparam N Number of contracted index pairs.
/// \param alpha Scaling factor for the contraction result.
/// \param A Left-hand tensor operand.
/// \param B Right-hand tensor operand.
/// \param dims Compile-time array reference listing contracted dimension pairs.
/// \param beta Scaling factor applied to the pre-existing contents of \p C.
/// \param C Destination tensor.
/// \param tag Backend selector instance.
/// \ingroup kernel_ops
void contract(T const& alpha, AType A, BType B,
              const std::pair<std::size_t, std::size_t> (&dims)[N], // array reference, size known at compile time
              U const& beta, CType C, TagType tag)
{
  contract(alpha, A, B, std::to_array(dims), beta, C, tag);
}

} // namespace uni20::kernel
