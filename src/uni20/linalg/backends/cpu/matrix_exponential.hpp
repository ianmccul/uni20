/**
 * \file matrix_exponential.hpp
 * \ingroup linalg
 * \brief CPU dense matrix exponential interface and dispatch adapter.
 */

#pragma once

#include <uni20/config.hpp>

#if UNI20_HAS_FLOAT128 && UNI20_FLOAT128_PROVIDER_MPLAPACK
#include <mplapack_config.h>
#endif

#include <uni20/common/trace.hpp>
#include <uni20/core/scalar_concepts.hpp>
#include <uni20/linalg/backends/cpu/dense_matrix.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/mdspan/concepts.hpp>

#include <complex>
#include <concepts>
#include <type_traits>

namespace uni20::linalg::backends::cpu
{

/// \brief Compute the matrix exponential using the adaptive scaling-and-squaring algorithm.
/// \details Follows the Pad\'e-based scaling and squaring strategy of Higham (2005) and
///          Al-Mohy & Higham (2011). The routine automatically selects between Pad\'e
///          degrees {3, 5, 7, 9, 13} based on matrix norms.
/// \tparam Scalar Dense matrix element type; may be real or complex in single, double, or extended precision.
/// \param matrix Dense matrix whose exponential will be evaluated.
/// \param t Scalar multiplier applied to \p matrix before exponentiation.
/// \return The matrix exponential of \f$\exp(t \cdot \text{matrix})\f$.
template <uni20::RealOrComplex Scalar>
DenseMatrix<Scalar> matrix_exponential(DenseMatrix<Scalar> const& matrix, uni20::make_real_t<Scalar> t);

/// \brief Compute a complex-coefficient matrix exponential, promoting real matrices to complex output.
/// \details This overload evaluates \f$\exp(t A)\f$ for complex \p t. If \p matrix is real, the result
///          is promoted to a complex dense matrix. If \p matrix is already complex, the result remains complex.
/// \tparam Scalar Real or complex matrix element type.
/// \param matrix Dense matrix whose exponential will be evaluated.
/// \param t Complex multiplier applied to \p matrix before exponentiation.
/// \return The complex matrix exponential of \f$\exp(t \cdot \text{matrix})\f$.
template <uni20::RealOrComplex Scalar>
DenseMatrix<uni20::complex<uni20::make_real_t<Scalar>>>
matrix_exponential(DenseMatrix<Scalar> const& matrix, uni20::complex<uni20::make_real_t<Scalar>> t);

} // namespace uni20::linalg::backends::cpu

namespace uni20::linalg
{

/// \brief Report compile-time eligibility for the CPU matrix exponential.
template <uni20::MutableRankedMdspanLike<2> OutputMdspan, uni20::RankedMdspanLike<2> InputMdspan, class TimeScalar>
consteval auto kernel_accepts_types(CpuReferenceBackend const&, matrix_exponential_op const&, OutputMdspan&,
                                    InputMdspan&, TimeScalar const&)
{
  using input_scalar = std::remove_cv_t<typename InputMdspan::element_type>;
  using input_matrix = backends::cpu::DenseMatrix<input_scalar>;

  if constexpr (requires(input_matrix const& input, TimeScalar time) {
                  backends::cpu::matrix_exponential(input, time);
                })
  {
    using result_matrix =
        decltype(backends::cpu::matrix_exponential(std::declval<input_matrix const&>(), std::declval<TimeScalar>()));
    if constexpr (requires(result_matrix& result, OutputMdspan& output, InputMdspan& input,
                           typename OutputMdspan::index_type output_index,
                           typename InputMdspan::index_type input_index) {
                    output.operator[](output_index, output_index) = result[std::size_t{}, std::size_t{}];
                    static_cast<input_scalar>(input.operator[](input_index, input_index));
                  })
    {
      return kernel_types_yes;
    }
    else
    {
      return kernel_types_no;
    }
  }
  else
  {
    return kernel_types_no;
  }
}

/// \brief Compute a matrix exponential through the existing CPU implementation.
template <class OutputMdspan, class InputMdspan, class TimeScalar>
KernelAttempt try_kernel(CpuReferenceBackend, matrix_exponential_op const&, OutputMdspan&& output, InputMdspan&& input,
                         TimeScalar time)
{
  using input_type = std::remove_cvref_t<InputMdspan>;
  using input_scalar = std::remove_cv_t<typename input_type::element_type>;
  using input_index_type = typename input_type::index_type;
  using output_index_type = typename std::remove_cvref_t<OutputMdspan>::index_type;

  CHECK_EQUAL(input.extent(0), input.extent(1));
  CHECK_EQUAL(output.extent(0), input.extent(0));
  CHECK_EQUAL(output.extent(1), input.extent(1));

  std::size_t const rows = static_cast<std::size_t>(input.extent(0));
  std::size_t const cols = static_cast<std::size_t>(input.extent(1));
  backends::cpu::DenseMatrix<input_scalar> materialized(rows, cols);
  for (input_index_type row = 0; row < input.extent(0); ++row)
  {
    for (input_index_type col = 0; col < input.extent(1); ++col)
    {
      materialized[static_cast<std::size_t>(row), static_cast<std::size_t>(col)] =
          static_cast<input_scalar>(input[row, col]);
    }
  }

  auto result = backends::cpu::matrix_exponential(materialized, time);
  for (output_index_type row = 0; row < output.extent(0); ++row)
  {
    for (output_index_type col = 0; col < output.extent(1); ++col)
    {
      output[row, col] = result[static_cast<std::size_t>(row), static_cast<std::size_t>(col)];
    }
  }
  return KernelAttempt::success;
}

} // namespace uni20::linalg
