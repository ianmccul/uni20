#pragma once

/**
 * \file nonsymmetric_eigen.hpp
 * \ingroup linalg
 * \brief LAPACK backend for dense nonsymmetric eigensystems.
 */

#include "common.hpp"

#include <uni20/backend/blas/blas_int.hpp>
#include <uni20/backend/lapack/lapack.hpp>
#include <uni20/common/trace.hpp>
#include <uni20/core/scalar_concepts.hpp>
#include <uni20/linalg/blas/mdspan_matrix.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/tensor/access.hpp>
#include <uni20/tensor/concepts.hpp>

#include <algorithm>
#include <concepts>
#include <span>
#include <type_traits>
#include <vector>

namespace uni20::linalg
{
namespace lapack_detail
{

/// \brief Report compile-time eligibility for resolved LAPACK nonsymmetric eigenanalysis.
template <uni20::MutableRankedStridedMdspanLike<2> MatrixMdspan, class EigenScalar,
          uni20::MutableRankedMdspanLike<2> RightEigenvectorMdspan>
consteval auto nonsymmetric_eigen_acceptance()
{
  using matrix_scalar = std::remove_cv_t<typename MatrixMdspan::element_type>;
  using real_type = uni20::make_real_t<matrix_scalar>;
  using expected_eigen_scalar = uni20::complex<real_type>;
  using vector_scalar = std::remove_cv_t<typename RightEigenvectorMdspan::element_type>;

  if constexpr (uni20::LapackScalar<matrix_scalar> && std::same_as<EigenScalar, expected_eigen_scalar> &&
                std::same_as<vector_scalar, expected_eigen_scalar> && uni20::DefaultAccessorMdspanLike<MatrixMdspan>)
  {
    return kernel_types_maybe;
  }
  else
  {
    return kernel_types_no;
  }
}

/// \brief Compute eigenvalues and optional right eigenvectors through LAPACK `geev`.
template <uni20::MutableRankedStridedMdspanLike<2> MatrixMdspan, class EigenScalar,
          uni20::MutableRankedMdspanLike<2> RightEigenvectorMdspan>
KernelAttempt try_nonsymmetric_eigen(nonsymmetric_eigen_op const& op, MatrixMdspan& matrix_work,
                                     std::span<EigenScalar> eigenvalues, RightEigenvectorMdspan& right_eigenvectors)
{
  using matrix_type = std::remove_cvref_t<MatrixMdspan>;
  using matrix_scalar = std::remove_cv_t<typename matrix_type::element_type>;
  using real_type = uni20::make_real_t<matrix_scalar>;
  using complex_type = uni20::complex<real_type>;
  using index_type = typename std::remove_cvref_t<RightEigenvectorMdspan>::index_type;

  CHECK_EQUAL(matrix_work.extent(0), matrix_work.extent(1));
  std::size_t const n = static_cast<std::size_t>(matrix_work.extent(0));
  CHECK_EQUAL(eigenvalues.size(), n);
  CHECK(!op.compute_right_vectors || (static_cast<std::size_t>(right_eigenvectors.extent(0)) == n &&
                                      static_cast<std::size_t>(right_eigenvectors.extent(1)) == n));

  blas_int const order = uni20::blas::try_blas_int(n);
  if (!uni20::blas::is_valid_blas_int(order))
  {
    return KernelAttempt::unsupported_shape;
  }

  auto matrix = uni20::linalg::blas::try_lapack_writable_matrix(matrix_work);
  if (!matrix)
  {
    return KernelAttempt::unsupported_layout;
  }
  CHECK_EQUAL(matrix->rows, order);
  CHECK_EQUAL(matrix->cols, order);
  if (order == 0)
  {
    return KernelAttempt::success;
  }

  char const jobvr = op.compute_right_vectors ? 'V' : 'N';
  blas_int const ldvr = op.compute_right_vectors ? order : 1;
  std::size_t vector_storage_size = 1;
  if (op.compute_right_vectors)
  {
    auto const square_size = lapack_detail::try_size_product(n, n);
    if (!square_size)
    {
      return KernelAttempt::unsupported_shape;
    }
    vector_storage_size = *square_size;
  }

  if constexpr (uni20::LapackReal<matrix_scalar>)
  {
    std::vector<matrix_scalar> wr(n, matrix_scalar{});
    std::vector<matrix_scalar> wi(n, matrix_scalar{});
    std::vector<matrix_scalar> vl(1, matrix_scalar{});
    std::vector<matrix_scalar> vr(vector_storage_size, matrix_scalar{});

    matrix_scalar work_query{};
    uni20::lapack::geev('N', jobvr, order, matrix->data, matrix->leading_dimension, wr.data(), wi.data(), vl.data(), 1,
                        vr.data(), ldvr, &work_query, -1);
    blas_int const lwork = lapack_detail::workspace_size(work_query);
    std::vector<matrix_scalar> work(static_cast<std::size_t>(lwork), matrix_scalar{});
    uni20::lapack::geev('N', jobvr, order, matrix->data, matrix->leading_dimension, wr.data(), wi.data(), vl.data(), 1,
                        vr.data(), ldvr, work.data(), lwork);

    for (std::size_t col = 0; col < n; ++col)
    {
      eigenvalues[col] = complex_type{wr[col], wi[col]};
    }

    if (!op.compute_right_vectors)
    {
      return KernelAttempt::success;
    }

    for (std::size_t col = 0; col < n; ++col)
    {
      if (wi[col] == real_type{})
      {
        for (std::size_t row = 0; row < n; ++row)
        {
          right_eigenvectors[static_cast<index_type>(row), static_cast<index_type>(col)] =
              complex_type{vr[row + col * n], real_type{}};
        }
      }
      else if (wi[col] > real_type{})
      {
        CHECK(col + 1 < n, "LAPACK geev returned an incomplete conjugate eigenvector pair");
        for (std::size_t row = 0; row < n; ++row)
        {
          complex_type const value{vr[row + col * n], vr[row + (col + 1) * n]};
          right_eigenvectors[static_cast<index_type>(row), static_cast<index_type>(col)] = value;
          right_eigenvectors[static_cast<index_type>(row), static_cast<index_type>(col + 1)] = uni20::conj(value);
        }
        ++col;
      }
      else
      {
        PANIC("LAPACK geev returned an unpaired negative-imaginary eigenvalue", col, wi[col]);
      }
    }
  }
  else
  {
    auto const twice_order = lapack_detail::try_size_product(2, n);
    if (!twice_order)
    {
      return KernelAttempt::unsupported_shape;
    }
    std::vector<complex_type> values(n, complex_type{});
    std::vector<complex_type> vl(1, complex_type{});
    std::vector<complex_type> vr(vector_storage_size, complex_type{});
    std::vector<real_type> rwork(std::max<std::size_t>(1, *twice_order), real_type{});

    complex_type work_query{};
    uni20::lapack::geev('N', jobvr, order, matrix->data, matrix->leading_dimension, values.data(), vl.data(), 1,
                        vr.data(), ldvr, &work_query, -1, rwork.data());
    blas_int const lwork = lapack_detail::workspace_size(work_query);
    std::vector<complex_type> work(static_cast<std::size_t>(lwork), complex_type{});
    uni20::lapack::geev('N', jobvr, order, matrix->data, matrix->leading_dimension, values.data(), vl.data(), 1,
                        vr.data(), ldvr, work.data(), lwork, rwork.data());

    std::ranges::copy(values, eigenvalues.begin());
    if (op.compute_right_vectors)
    {
      for (std::size_t col = 0; col < n; ++col)
      {
        for (std::size_t row = 0; row < n; ++row)
        {
          right_eigenvectors[static_cast<index_type>(row), static_cast<index_type>(col)] = vr[row + col * n];
        }
      }
    }
  }

  return KernelAttempt::success;
}
} // namespace lapack_detail

/// \brief Report eligibility for host-accessible mdspec nonsymmetric eigenanalysis.
template <uni20::MutableRankedStridedMdspecLike<2> MatrixMdspan, class EigenScalar,
          uni20::MutableRankedMdspecLike<2> RightEigenvectorMdspan>
  requires uni20::HostWritableMdspec<MatrixMdspan> && uni20::HostWritableMdspec<RightEigenvectorMdspan>
consteval auto kernel_accepts_types(LapackBackend const&, nonsymmetric_eigen_op const&, MatrixMdspan&,
                                    std::span<EigenScalar>&, RightEigenvectorMdspan&)
{
  using matrix_span = uni20::host_write_mdspan_t<MatrixMdspan>;
  using vector_span = uni20::host_write_mdspan_t<RightEigenvectorMdspan>;
  constexpr auto acceptance = lapack_detail::nonsymmetric_eigen_acceptance<matrix_span, EigenScalar, vector_span>();
  if constexpr (acceptance == KernelTypeAcceptance::no)
    return kernel_types_no;
  else
    return kernel_types_maybe;
}

/// \brief Resolve host access and run LAPACK nonsymmetric eigenanalysis.
template <uni20::MutableRankedStridedMdspecLike<2> MatrixMdspan, class EigenScalar,
          uni20::MutableRankedMdspecLike<2> RightEigenvectorMdspan>
  requires uni20::HostWritableMdspec<MatrixMdspan> && uni20::HostWritableMdspec<RightEigenvectorMdspan>
KernelAttempt try_kernel(LapackBackend, nonsymmetric_eigen_op const& operation, MatrixMdspan& matrix_work,
                         std::span<EigenScalar> eigenvalues, RightEigenvectorMdspan& right_eigenvectors)
{
  return lapack_detail::with_host_write_mdspans(
      [&](auto& matrix_span, auto& vector_span) {
        return lapack_detail::try_nonsymmetric_eigen(operation, matrix_span, eigenvalues, vector_span);
      },
      matrix_work, right_eigenvectors);
}

} // namespace uni20::linalg
