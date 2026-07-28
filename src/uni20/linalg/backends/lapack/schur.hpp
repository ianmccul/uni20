#pragma once

/**
 * \file schur.hpp
 * \ingroup linalg
 * \brief LAPACK backend for dense Schur decomposition and block reordering.
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

/// \brief Report compile-time eligibility for LAPACK Schur decomposition.
template <uni20::MutableRankedStridedMdspanLike<2> MatrixMdspan, class EigenScalar,
          uni20::MutableRankedStridedMdspanLike<2> SchurVectorMdspan>
consteval auto kernel_accepts_types(LapackBackend const&, schur_op const&, MatrixMdspan&, std::span<EigenScalar>&,
                                    SchurVectorMdspan&)
{
  using matrix_scalar = std::remove_cv_t<typename MatrixMdspan::element_type>;
  using vector_scalar = std::remove_cv_t<typename SchurVectorMdspan::element_type>;
  using expected_eigen_scalar = uni20::complex<uni20::make_real_t<matrix_scalar>>;

  if constexpr (uni20::LapackScalar<matrix_scalar> && std::same_as<vector_scalar, matrix_scalar> &&
                std::same_as<EigenScalar, expected_eigen_scalar> && uni20::DefaultAccessorMdspanLike<MatrixMdspan> &&
                uni20::DefaultAccessorMdspanLike<SchurVectorMdspan>)
  {
    return kernel_types_maybe;
  }
  else
  {
    return kernel_types_no;
  }
}

/// \brief Compute a dense Schur form through LAPACK `gees`.
template <uni20::MutableRankedStridedMdspanLike<2> MatrixMdspan, class EigenScalar,
          uni20::MutableRankedStridedMdspanLike<2> SchurVectorMdspan>
KernelAttempt try_kernel(LapackBackend, schur_op const& op, MatrixMdspan&& matrix_work,
                         std::span<EigenScalar> eigenvalues, SchurVectorMdspan&& schur_vectors)
{
  using matrix_type = std::remove_cvref_t<MatrixMdspan>;
  using scalar_type = std::remove_cv_t<typename matrix_type::element_type>;
  using real_type = uni20::make_real_t<scalar_type>;
  using complex_type = uni20::complex<real_type>;

  CHECK_EQUAL(matrix_work.extent(0), matrix_work.extent(1));
  std::size_t const n = static_cast<std::size_t>(matrix_work.extent(0));
  CHECK_EQUAL(eigenvalues.size(), n);
  CHECK(!op.compute_vectors || (static_cast<std::size_t>(schur_vectors.extent(0)) == n &&
                                static_cast<std::size_t>(schur_vectors.extent(1)) == n));

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

  decltype(uni20::linalg::blas::try_lapack_writable_matrix(schur_vectors)) vectors;
  if (op.compute_vectors)
  {
    vectors = uni20::linalg::blas::try_lapack_writable_matrix(schur_vectors);
    if (!vectors)
    {
      return KernelAttempt::unsupported_layout;
    }
  }

  CHECK_EQUAL(matrix->rows, order);
  CHECK_EQUAL(matrix->cols, order);
  if (order == 0)
  {
    return KernelAttempt::success;
  }

  scalar_type dummy{};
  scalar_type* vector_data = op.compute_vectors ? vectors->data : &dummy;
  blas_int const ldvs = op.compute_vectors ? vectors->leading_dimension : 1;
  char const jobvs = op.compute_vectors ? 'V' : 'N';
  blas_int selected_dimension = 0;
  std::vector<blas_int> bwork(std::max<std::size_t>(1, n), 0);

  if constexpr (uni20::LapackReal<scalar_type>)
  {
    std::vector<scalar_type> wr(n, scalar_type{});
    std::vector<scalar_type> wi(n, scalar_type{});
    scalar_type work_query{};
    uni20::lapack::gees(jobvs, 'N', order, matrix->data, matrix->leading_dimension, selected_dimension, wr.data(),
                        wi.data(), vector_data, ldvs, &work_query, -1, bwork.data());
    blas_int const lwork = lapack_detail::workspace_size(work_query);
    std::vector<scalar_type> work(static_cast<std::size_t>(lwork), scalar_type{});
    uni20::lapack::gees(jobvs, 'N', order, matrix->data, matrix->leading_dimension, selected_dimension, wr.data(),
                        wi.data(), vector_data, ldvs, work.data(), lwork, bwork.data());
    for (std::size_t index = 0; index < n; ++index)
    {
      eigenvalues[index] = complex_type{wr[index], wi[index]};
    }
  }
  else
  {
    std::vector<complex_type> values(n, complex_type{});
    std::vector<real_type> rwork(std::max<std::size_t>(1, n), real_type{});
    complex_type work_query{};
    uni20::lapack::gees(jobvs, 'N', order, matrix->data, matrix->leading_dimension, selected_dimension, values.data(),
                        vector_data, ldvs, &work_query, -1, rwork.data(), bwork.data());
    blas_int const lwork = lapack_detail::workspace_size(work_query);
    std::vector<complex_type> work(static_cast<std::size_t>(lwork), complex_type{});
    uni20::lapack::gees(jobvs, 'N', order, matrix->data, matrix->leading_dimension, selected_dimension, values.data(),
                        vector_data, ldvs, work.data(), lwork, rwork.data(), bwork.data());
    std::ranges::copy(values, eigenvalues.begin());
  }

  return KernelAttempt::success;
}

/// \brief Report compile-time eligibility for real upper-Hessenberg Schur decomposition.
template <uni20::MutableRankedStridedMdspanLike<2> MatrixMdspan, class EigenScalar,
          uni20::MutableRankedStridedMdspanLike<2> SchurVectorMdspan>
consteval auto kernel_accepts_types(LapackBackend const&, hessenberg_schur_op const&, MatrixMdspan&,
                                    std::span<EigenScalar>&, SchurVectorMdspan&)
{
  using matrix_scalar = std::remove_cv_t<typename MatrixMdspan::element_type>;
  using vector_scalar = std::remove_cv_t<typename SchurVectorMdspan::element_type>;
  if constexpr (uni20::LapackReal<matrix_scalar> && std::same_as<vector_scalar, matrix_scalar> &&
                std::same_as<EigenScalar, uni20::complex<matrix_scalar>> &&
                uni20::DefaultAccessorMdspanLike<MatrixMdspan> && uni20::DefaultAccessorMdspanLike<SchurVectorMdspan>)
  {
    return kernel_types_maybe;
  }
  else
  {
    return kernel_types_no;
  }
}

/// \brief Compute a real upper-Hessenberg Schur form through LAPACK `hseqr`.
template <uni20::MutableRankedStridedMdspanLike<2> MatrixMdspan, class EigenScalar,
          uni20::MutableRankedStridedMdspanLike<2> SchurVectorMdspan>
KernelAttempt try_kernel(LapackBackend, hessenberg_schur_op const& op, MatrixMdspan&& hessenberg,
                         std::span<EigenScalar> eigenvalues, SchurVectorMdspan&& schur_vectors)
{
  using scalar_type = std::remove_cv_t<typename std::remove_cvref_t<MatrixMdspan>::element_type>;

  CHECK_EQUAL(hessenberg.extent(0), hessenberg.extent(1));
  std::size_t const n = static_cast<std::size_t>(hessenberg.extent(0));
  CHECK_EQUAL(eigenvalues.size(), n);
  CHECK(!op.compute_vectors || (static_cast<std::size_t>(schur_vectors.extent(0)) == n &&
                                static_cast<std::size_t>(schur_vectors.extent(1)) == n));

  blas_int const order = uni20::blas::try_blas_int(n);
  if (!uni20::blas::is_valid_blas_int(order))
  {
    return KernelAttempt::unsupported_shape;
  }

  auto matrix = uni20::linalg::blas::try_lapack_writable_matrix(hessenberg);
  if (!matrix)
  {
    return KernelAttempt::unsupported_layout;
  }

  decltype(uni20::linalg::blas::try_lapack_writable_matrix(schur_vectors)) vectors;
  if (op.compute_vectors)
  {
    vectors = uni20::linalg::blas::try_lapack_writable_matrix(schur_vectors);
    if (!vectors)
    {
      return KernelAttempt::unsupported_layout;
    }
  }
  if (order == 0)
  {
    return KernelAttempt::success;
  }

  scalar_type dummy{};
  scalar_type* vector_data = op.compute_vectors ? vectors->data : &dummy;
  blas_int const ldz = op.compute_vectors ? vectors->leading_dimension : 1;
  std::vector<scalar_type> wr(n, scalar_type{});
  std::vector<scalar_type> wi(n, scalar_type{});
  scalar_type work_query{};
  uni20::lapack::hseqr('S', op.compute_vectors ? 'I' : 'N', order, 1, order, matrix->data, matrix->leading_dimension,
                       wr.data(), wi.data(), vector_data, ldz, &work_query, -1);
  blas_int const lwork = lapack_detail::workspace_size(work_query);
  std::vector<scalar_type> work(static_cast<std::size_t>(lwork), scalar_type{});
  uni20::lapack::hseqr('S', op.compute_vectors ? 'I' : 'N', order, 1, order, matrix->data, matrix->leading_dimension,
                       wr.data(), wi.data(), vector_data, ldz, work.data(), lwork);
  for (std::size_t index = 0; index < n; ++index)
  {
    eigenvalues[index] = uni20::complex<scalar_type>{wr[index], wi[index]};
  }
  return KernelAttempt::success;
}

/// \brief Report compile-time eligibility for LAPACK Schur reordering.
template <uni20::MutableRankedStridedMdspanLike<2> SchurFormMdspan,
          uni20::MutableRankedStridedMdspanLike<2> SchurVectorMdspan>
consteval auto kernel_accepts_types(LapackBackend const&, schur_reorder_op const&, SchurFormMdspan&, SchurVectorMdspan&)
{
  using form_scalar = std::remove_cv_t<typename SchurFormMdspan::element_type>;
  using vector_scalar = std::remove_cv_t<typename SchurVectorMdspan::element_type>;
  if constexpr (uni20::LapackScalar<form_scalar> && std::same_as<form_scalar, vector_scalar> &&
                uni20::DefaultAccessorMdspanLike<SchurFormMdspan> &&
                uni20::DefaultAccessorMdspanLike<SchurVectorMdspan>)
  {
    return kernel_types_maybe;
  }
  else
  {
    return kernel_types_no;
  }
}

/// \brief Move one real Schur block or complex Schur entry through LAPACK `trexc`.
template <uni20::MutableRankedStridedMdspanLike<2> SchurFormMdspan,
          uni20::MutableRankedStridedMdspanLike<2> SchurVectorMdspan>
KernelAttempt try_kernel(LapackBackend, schur_reorder_op const& op, SchurFormMdspan&& schur_form,
                         SchurVectorMdspan&& schur_vectors)
{
  using form_type = std::remove_cvref_t<SchurFormMdspan>;
  using scalar_type = std::remove_cv_t<typename form_type::element_type>;

  CHECK_EQUAL(schur_form.extent(0), schur_form.extent(1));
  std::size_t const n = static_cast<std::size_t>(schur_form.extent(0));
  CHECK(op.from < n);
  CHECK(op.to < n);
  CHECK(!op.update_vectors || (static_cast<std::size_t>(schur_vectors.extent(0)) == n &&
                               static_cast<std::size_t>(schur_vectors.extent(1)) == n));

  blas_int const order = uni20::blas::try_blas_int(n);
  if (!uni20::blas::is_valid_blas_int(order))
  {
    return KernelAttempt::unsupported_shape;
  }

  auto form = uni20::linalg::blas::try_lapack_writable_matrix(schur_form);
  if (!form)
  {
    return KernelAttempt::unsupported_layout;
  }

  decltype(uni20::linalg::blas::try_lapack_writable_matrix(schur_vectors)) vectors;
  if (op.update_vectors)
  {
    vectors = uni20::linalg::blas::try_lapack_writable_matrix(schur_vectors);
    if (!vectors)
    {
      return KernelAttempt::unsupported_layout;
    }
  }

  if (op.from == op.to)
  {
    return KernelAttempt::success;
  }

  scalar_type dummy{};
  scalar_type* vector_data = op.update_vectors ? vectors->data : &dummy;
  blas_int const ldq = op.update_vectors ? vectors->leading_dimension : 1;
  char const compq = op.update_vectors ? 'V' : 'N';
  blas_int first = static_cast<blas_int>(op.from + 1);
  blas_int last = static_cast<blas_int>(op.to + 1);

  if constexpr (uni20::LapackReal<scalar_type>)
  {
    std::vector<scalar_type> work(n, scalar_type{});
    uni20::lapack::trexc(compq, order, form->data, form->leading_dimension, vector_data, ldq, first, last, work.data());
  }
  else
  {
    uni20::lapack::trexc(compq, order, form->data, form->leading_dimension, vector_data, ldq, first, last);
  }
  return KernelAttempt::success;
}

/// \brief Report eligibility for host DeviceTensorView Schur decomposition.
template <class Operation, uni20::MutableRankedDeviceTensorView<2> MatrixTensor, class EigenScalar,
          uni20::MutableRankedDeviceTensorView<2> SchurVectorTensor>
  requires(std::same_as<Operation, schur_op> || std::same_as<Operation, hessenberg_schur_op>) &&
          uni20::detail::HostWritableTensor<MatrixTensor> && uni20::detail::HostWritableTensor<SchurVectorTensor>
consteval auto kernel_accepts_types(LapackBackend const&, Operation const&, MatrixTensor&, std::span<EigenScalar>&,
                                    SchurVectorTensor&)
{
  using matrix_span = uni20::detail::host_write_tensor_mdspan_t<MatrixTensor>;
  using vector_span = uni20::detail::host_write_tensor_mdspan_t<SchurVectorTensor>;
  constexpr auto acceptance =
      detail::backend_type_acceptance<LapackBackend, Operation, matrix_span&, std::span<EigenScalar>&, vector_span&>();
  if constexpr (acceptance == KernelTypeAcceptance::no)
    return kernel_types_no;
  else
    return kernel_types_maybe;
}

/// \brief Resolve host tensor access and run a LAPACK Schur decomposition.
template <class Operation, uni20::MutableRankedDeviceTensorView<2> MatrixTensor, class EigenScalar,
          uni20::MutableRankedDeviceTensorView<2> SchurVectorTensor>
  requires(std::same_as<Operation, schur_op> || std::same_as<Operation, hessenberg_schur_op>) &&
          uni20::detail::HostWritableTensor<MatrixTensor> && uni20::detail::HostWritableTensor<SchurVectorTensor>
KernelAttempt try_kernel(LapackBackend backend, Operation const& operation, MatrixTensor& matrix_work,
                         std::span<EigenScalar> eigenvalues, SchurVectorTensor& schur_vectors)
{
  return uni20::detail::with_host_write_tensor_mdspans(
      [&](auto& matrix_span, auto& vector_span) {
        return try_kernel(backend, operation, matrix_span, eigenvalues, vector_span);
      },
      matrix_work, schur_vectors);
}

/// \brief Report eligibility for host DeviceTensorView Schur reordering.
template <uni20::MutableRankedDeviceTensorView<2> SchurFormTensor,
          uni20::MutableRankedDeviceTensorView<2> SchurVectorTensor>
  requires uni20::detail::HostWritableTensor<SchurFormTensor> && uni20::detail::HostWritableTensor<SchurVectorTensor>
consteval auto kernel_accepts_types(LapackBackend const&, schur_reorder_op const&, SchurFormTensor&, SchurVectorTensor&)
{
  using form_span = uni20::detail::host_write_tensor_mdspan_t<SchurFormTensor>;
  using vector_span = uni20::detail::host_write_tensor_mdspan_t<SchurVectorTensor>;
  constexpr auto acceptance =
      detail::backend_type_acceptance<LapackBackend, schur_reorder_op, form_span&, vector_span&>();
  if constexpr (acceptance == KernelTypeAcceptance::no)
    return kernel_types_no;
  else
    return kernel_types_maybe;
}

/// \brief Resolve host tensor access and run LAPACK Schur reordering.
template <uni20::MutableRankedDeviceTensorView<2> SchurFormTensor,
          uni20::MutableRankedDeviceTensorView<2> SchurVectorTensor>
  requires uni20::detail::HostWritableTensor<SchurFormTensor> && uni20::detail::HostWritableTensor<SchurVectorTensor>
KernelAttempt try_kernel(LapackBackend backend, schur_reorder_op const& operation, SchurFormTensor& schur_form,
                         SchurVectorTensor& schur_vectors)
{
  return uni20::detail::with_host_write_tensor_mdspans(
      [&](auto& form_span, auto& vector_span) { return try_kernel(backend, operation, form_span, vector_span); },
      schur_form, schur_vectors);
}

} // namespace uni20::linalg
