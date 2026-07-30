#pragma once

/**
 * \file tridiagonal_eigen.hpp
 * \ingroup linalg
 * \brief LAPACK backend for real symmetric tridiagonal eigensystems.
 */

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

/// \brief Report compile-time eligibility for resolved LAPACK tridiagonal eigensystem dispatch.
template <uni20::LapackReal Scalar, uni20::MutableRankedStridedMdspanLike<2> EigenvectorMdspan>
consteval auto tridiagonal_eigen_acceptance()
{
  using vector_scalar = std::remove_cv_t<typename EigenvectorMdspan::element_type>;
  if constexpr (std::same_as<vector_scalar, Scalar> && uni20::DefaultAccessorMdspanLike<EigenvectorMdspan>)
  {
    return kernel_types_maybe;
  }
  else
  {
    return kernel_types_no;
  }
}

/// \brief Compute a real symmetric tridiagonal eigensystem through `sterf` or `steqr`.
template <class Scalar, uni20::MutableRankedStridedMdspanLike<2> EigenvectorMdspan>
KernelAttempt try_tridiagonal_eigen(symmetric_tridiagonal_eigen_op const& op, std::span<Scalar> diagonal,
                                    std::span<Scalar> subdiagonal, EigenvectorMdspan& eigenvectors)
{
  std::size_t const n = diagonal.size();
  CHECK(subdiagonal.size() + 1 == n || (n == 0 && subdiagonal.empty()));
  CHECK(!op.compute_vectors || (static_cast<std::size_t>(eigenvectors.extent(0)) == n &&
                                static_cast<std::size_t>(eigenvectors.extent(1)) == n));

  blas_int const order = uni20::blas::try_blas_int(n);
  if (!uni20::blas::is_valid_blas_int(order))
  {
    return KernelAttempt::unsupported_shape;
  }
  if (order == 0)
  {
    return KernelAttempt::success;
  }

  Scalar dummy{};
  Scalar* e = subdiagonal.empty() ? &dummy : subdiagonal.data();
  if (!op.compute_vectors)
  {
    uni20::lapack::sterf(order, diagonal.data(), e);
    return KernelAttempt::success;
  }

  auto matrix = uni20::linalg::blas::try_lapack_writable_matrix(eigenvectors);
  if (!matrix)
  {
    return KernelAttempt::unsupported_layout;
  }

  CHECK_EQUAL(matrix->rows, order);
  CHECK_EQUAL(matrix->cols, order);
  auto const twice_order = lapack_detail::try_size_product(2, n);
  if (!twice_order)
  {
    return KernelAttempt::unsupported_shape;
  }
  std::size_t const work_size = std::max<std::size_t>(1, *twice_order - 2);
  std::vector<Scalar> work(work_size, Scalar{});
  uni20::lapack::steqr('I', order, diagonal.data(), e, matrix->data, matrix->leading_dimension, work.data());
  return KernelAttempt::success;
}
} // namespace lapack_detail

/// \brief Report eligibility for a host-accessible device-mdspan tridiagonal eigensystem.
template <uni20::LapackReal Scalar, uni20::MutableRankedStridedDeviceMdspanLike<2> EigenvectorMdspan>
  requires uni20::detail::HostWritableDeviceMdspan<EigenvectorMdspan>
consteval auto kernel_accepts_types(LapackBackend const&, symmetric_tridiagonal_eigen_op const&, std::span<Scalar>&,
                                    std::span<Scalar>&, EigenvectorMdspan&)
{
  using vector_span = uni20::detail::host_write_mdspan_t<EigenvectorMdspan>;
  constexpr auto acceptance = lapack_detail::tridiagonal_eigen_acceptance<Scalar, vector_span>();
  if constexpr (acceptance == KernelTypeAcceptance::no)
    return kernel_types_no;
  else
    return kernel_types_maybe;
}

/// \brief Resolve host access and run LAPACK tridiagonal eigenanalysis.
template <uni20::LapackReal Scalar, uni20::MutableRankedStridedDeviceMdspanLike<2> EigenvectorMdspan>
  requires uni20::detail::HostWritableDeviceMdspan<EigenvectorMdspan>
KernelAttempt try_kernel(LapackBackend, symmetric_tridiagonal_eigen_op const& operation, std::span<Scalar> diagonal,
                         std::span<Scalar> subdiagonal, EigenvectorMdspan& eigenvectors)
{
  auto access = acquire_host_write_access(eigenvectors);
  auto span = access.mdspan();
  return lapack_detail::try_tridiagonal_eigen(operation, diagonal, subdiagonal, span);
}

} // namespace uni20::linalg
