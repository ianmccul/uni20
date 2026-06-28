#pragma once

#if defined(UNI20_ENABLE_MPLAPACK)
// clang-format off
#include <mplapack_config.h>
#include <complex>
#include <mplapack_binary128.h>
// clang-format on
#endif

#include <uni20/backend/lapack/lapack.hpp>
#include <uni20/common/mdspan.hpp>
#include <uni20/config.hpp>
#include <uni20/core/scalar_concepts.hpp>
#include <uni20/core/scalar_traits.hpp>
#include <uni20/linalg/backends/cpu/dense_matrix.hpp>
#include <uni20/linalg/backends/cpu/matrix_exponential.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace uni20::krylov
{

/// \brief Prototype dense matrix type used for local Krylov subspace algebra.
///
/// \details This aliases the CPU dense linear algebra matrix while the final
///          Uni20 rank-2 tensor layout policy is still being designed.
template <typename Scalar> using Matrix = uni20::linalg::backends::cpu::DenseMatrix<Scalar>;

using uni20::linalg::backends::cpu::add;
using uni20::linalg::backends::cpu::matrix_exponential;
using uni20::linalg::backends::cpu::matrix_one_norm;
using uni20::linalg::backends::cpu::matrix_one_norm_power;
using uni20::linalg::backends::cpu::matrix_power;
using uni20::linalg::backends::cpu::scale;
using uni20::linalg::backends::cpu::solve_linear_system;
using uni20::linalg::backends::cpu::subtract;

/// \brief Prototype row-major dense matrix used by mdspan-style LAPACK wrappers.
///
/// \details This is intentionally separate from the current Krylov `Matrix`,
///          which is column-major for direct Fortran LAPACK calls. `RightMatrix`
///          gives the temporary mdspan wrapper layer a layout-right owning
///          matrix while the final Uni20 rank-2 tensor layout policy is still
///          being designed.
template <typename Scalar> class RightMatrix {
  public:
    RightMatrix() = default;

    RightMatrix(std::size_t rows, std::size_t cols) : rows_(rows), cols_(cols), data_(rows * cols) {}

    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }

    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }

    [[nodiscard]] std::size_t size() const noexcept { return data_.size(); }

    Scalar& operator()(std::size_t row, std::size_t col) { return data_[col + row * cols_]; }

    Scalar const& operator()(std::size_t row, std::size_t col) const { return data_[col + row * cols_]; }

    [[nodiscard]] Scalar* data() noexcept { return data_.data(); }

    [[nodiscard]] Scalar const* data() const noexcept { return data_.data(); }

  private:
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<Scalar> data_;
};

/// \brief Return a mutable layout-right mdspan view of a prototype right-layout matrix.
/// \tparam Scalar Element type.
/// \param matrix Matrix whose storage is exposed.
/// \return Mutable rank-2 layout-right mdspan view.
template <typename Scalar> auto right_mdspan(RightMatrix<Scalar>& matrix)
{
  using extents_type = stdex::extents<std::size_t, stdex::dynamic_extent, stdex::dynamic_extent>;
  return stdex::mdspan<Scalar, extents_type, stdex::layout_right>(matrix.data(), matrix.rows(), matrix.cols());
}

/// \brief Return an immutable layout-right mdspan view of a prototype right-layout matrix.
/// \tparam Scalar Element type.
/// \param matrix Matrix whose storage is exposed.
/// \return Immutable rank-2 layout-right mdspan view.
template <typename Scalar> auto right_mdspan(RightMatrix<Scalar> const& matrix)
{
  using extents_type = stdex::extents<std::size_t, stdex::dynamic_extent, stdex::dynamic_extent>;
  return stdex::mdspan<Scalar const, extents_type, stdex::layout_right>(matrix.data(), matrix.rows(), matrix.cols());
}

/// \brief Copy a column-major Krylov matrix into a row-major prototype matrix.
/// \tparam Scalar Element type.
/// \param matrix Source column-major matrix.
/// \return Row-major matrix with the same logical entries.
template <typename Scalar> RightMatrix<Scalar> copy_left_to_right(Matrix<Scalar> const& matrix)
{
  RightMatrix<Scalar> result(matrix.rows(), matrix.cols());
  for (std::size_t row = 0; row < matrix.rows(); ++row)
  {
    for (std::size_t col = 0; col < matrix.cols(); ++col)
    {
      result(row, col) = matrix(row, col);
    }
  }
  return result;
}

/// \brief Copy a row-major prototype matrix into a column-major Krylov matrix.
/// \tparam Scalar Element type.
/// \param matrix Source row-major matrix.
/// \return Column-major matrix with the same logical entries.
template <typename Scalar> Matrix<Scalar> copy_right_to_left(RightMatrix<Scalar> const& matrix)
{
  Matrix<Scalar> result(matrix.rows(), matrix.cols());
  for (std::size_t row = 0; row < matrix.rows(); ++row)
  {
    for (std::size_t col = 0; col < matrix.cols(); ++col)
    {
      result(row, col) = matrix(row, col);
    }
  }
  return result;
}

/// \brief Matrix transpose mode for local dense Krylov operations.
enum class MatrixTranspose
{
  None,
  Transpose,
  ConjugateTranspose
};

/// \brief Matrix side selected by LAPACK-style multiply and solve operations.
enum class MatrixSide
{
  Left,
  Right
};

/// \brief Matrix region selected by LAPACK-style copy and set operations.
enum class MatrixFill
{
  All,
  Upper,
  Lower
};

/// \brief LAPACK-compatible dense matrix norm selector.
enum class MatrixNorm
{
  MaxAbs,
  One,
  Infinity,
  Frobenius
};

namespace detail
{
extern "C"
{
  void ssterf_(blas_int const* n, float* d, float* e, blas_int* info);

  void dsterf_(blas_int const* n, double* d, double* e, blas_int* info);

  void ssteqr_(char const* compz, blas_int const* n, float* d, float* e, float* z, blas_int const* ldz, float* work,
               blas_int* info);

  void dsteqr_(char const* compz, blas_int const* n, double* d, double* e, double* z, blas_int const* ldz, double* work,
               blas_int* info);

  void sstevd_(char const* jobz, blas_int const* n, float* d, float* e, float* z, blas_int const* ldz, float* work,
               blas_int const* lwork, blas_int* iwork, blas_int const* liwork, blas_int* info);

  void dstevd_(char const* jobz, blas_int const* n, double* d, double* e, double* z, blas_int const* ldz, double* work,
               blas_int const* lwork, blas_int* iwork, blas_int const* liwork, blas_int* info);

  void sstevr_(char const* jobz, char const* range, blas_int const* n, float* d, float* e, float const* vl,
               float const* vu, blas_int const* il, blas_int const* iu, float const* abstol, blas_int* m, float* w,
               float* z, blas_int const* ldz, blas_int* isuppz, float* work, blas_int const* lwork, blas_int* iwork,
               blas_int const* liwork, blas_int* info);

  void dstevr_(char const* jobz, char const* range, blas_int const* n, double* d, double* e, double const* vl,
               double const* vu, blas_int const* il, blas_int const* iu, double const* abstol, blas_int* m, double* w,
               double* z, blas_int const* ldz, blas_int* isuppz, double* work, blas_int const* lwork, blas_int* iwork,
               blas_int const* liwork, blas_int* info);

  void sbdsqr_(char const* uplo, blas_int const* n, blas_int const* ncvt, blas_int const* nru, blas_int const* ncc,
               float* d, float* e, float* vt, blas_int const* ldvt, float* u, blas_int const* ldu, float* c,
               blas_int const* ldc, float* work, blas_int* info);

  void dbdsqr_(char const* uplo, blas_int const* n, blas_int const* ncvt, blas_int const* nru, blas_int const* ncc,
               double* d, double* e, double* vt, blas_int const* ldvt, double* u, blas_int const* ldu, double* c,
               blas_int const* ldc, double* work, blas_int* info);

  void sbdsdc_(char const* uplo, char const* compq, blas_int const* n, float* d, float* e, float* u,
               blas_int const* ldu, float* vt, blas_int const* ldvt, float* q, blas_int* iq, float* work,
               blas_int* iwork, blas_int* info);

  void dbdsdc_(char const* uplo, char const* compq, blas_int const* n, double* d, double* e, double* u,
               blas_int const* ldu, double* vt, blas_int const* ldvt, double* q, blas_int* iq, double* work,
               blas_int* iwork, blas_int* info);

  void sbdsvdx_(char const* uplo, char const* jobz, char const* range, blas_int const* n, float* d, float* e,
                float const* vl, float const* vu, blas_int const* il, blas_int const* iu, blas_int* ns, float* s,
                float* z, blas_int const* ldz, float* work, blas_int* iwork, blas_int* info);

  void dbdsvdx_(char const* uplo, char const* jobz, char const* range, blas_int const* n, double* d, double* e,
                double const* vl, double const* vu, blas_int const* il, blas_int const* iu, blas_int* ns, double* s,
                double* z, blas_int const* ldz, double* work, blas_int* iwork, blas_int* info);

  void ssyev_(char const* jobz, char const* uplo, blas_int const* n, float* a, blas_int const* lda, float* w,
              float* work, blas_int const* lwork, blas_int* info);

  void dsyev_(char const* jobz, char const* uplo, blas_int const* n, double* a, blas_int const* lda, double* w,
              double* work, blas_int const* lwork, blas_int* info);

  void ssyevd_(char const* jobz, char const* uplo, blas_int const* n, float* a, blas_int const* lda, float* w,
               float* work, blas_int const* lwork, blas_int* iwork, blas_int const* liwork, blas_int* info);

  void dsyevd_(char const* jobz, char const* uplo, blas_int const* n, double* a, blas_int const* lda, double* w,
               double* work, blas_int const* lwork, blas_int* iwork, blas_int const* liwork, blas_int* info);

  void ssyevr_(char const* jobz, char const* range, char const* uplo, blas_int const* n, float* a, blas_int const* lda,
               float const* vl, float const* vu, blas_int const* il, blas_int const* iu, float const* abstol,
               blas_int* m, float* w, float* z, blas_int const* ldz, blas_int* isuppz, float* work,
               blas_int const* lwork, blas_int* iwork, blas_int const* liwork, blas_int* info);

  void dsyevr_(char const* jobz, char const* range, char const* uplo, blas_int const* n, double* a, blas_int const* lda,
               double const* vl, double const* vu, blas_int const* il, blas_int const* iu, double const* abstol,
               blas_int* m, double* w, double* z, blas_int const* ldz, blas_int* isuppz, double* work,
               blas_int const* lwork, blas_int* iwork, blas_int const* liwork, blas_int* info);

  void ssygv_(blas_int const* itype, char const* jobz, char const* uplo, blas_int const* n, float* a,
              blas_int const* lda, float* b, blas_int const* ldb, float* w, float* work, blas_int const* lwork,
              blas_int* info);

  void dsygv_(blas_int const* itype, char const* jobz, char const* uplo, blas_int const* n, double* a,
              blas_int const* lda, double* b, blas_int const* ldb, double* w, double* work, blas_int const* lwork,
              blas_int* info);

  void ssygvd_(blas_int const* itype, char const* jobz, char const* uplo, blas_int const* n, float* a,
               blas_int const* lda, float* b, blas_int const* ldb, float* w, float* work, blas_int const* lwork,
               blas_int* iwork, blas_int const* liwork, blas_int* info);

  void dsygvd_(blas_int const* itype, char const* jobz, char const* uplo, blas_int const* n, double* a,
               blas_int const* lda, double* b, blas_int const* ldb, double* w, double* work, blas_int const* lwork,
               blas_int* iwork, blas_int const* liwork, blas_int* info);

  void ssygvx_(blas_int const* itype, char const* jobz, char const* range, char const* uplo, blas_int const* n,
               float* a, blas_int const* lda, float* b, blas_int const* ldb, float const* vl, float const* vu,
               blas_int const* il, blas_int const* iu, float const* abstol, blas_int* m, float* w, float* z,
               blas_int const* ldz, float* work, blas_int const* lwork, blas_int* iwork, blas_int* ifail,
               blas_int* info);

  void dsygvx_(blas_int const* itype, char const* jobz, char const* range, char const* uplo, blas_int const* n,
               double* a, blas_int const* lda, double* b, blas_int const* ldb, double const* vl, double const* vu,
               blas_int const* il, blas_int const* iu, double const* abstol, blas_int* m, double* w, double* z,
               blas_int const* ldz, double* work, blas_int const* lwork, blas_int* iwork, blas_int* ifail,
               blas_int* info);

  void cheev_(char const* jobz, char const* uplo, blas_int const* n, uni20::complex<float>* a, blas_int const* lda,
              float* w, uni20::complex<float>* work, blas_int const* lwork, float* rwork, blas_int* info);

  void zheev_(char const* jobz, char const* uplo, blas_int const* n, uni20::complex<double>* a, blas_int const* lda,
              double* w, uni20::complex<double>* work, blas_int const* lwork, double* rwork, blas_int* info);

  void cheevd_(char const* jobz, char const* uplo, blas_int const* n, uni20::complex<float>* a, blas_int const* lda,
               float* w, uni20::complex<float>* work, blas_int const* lwork, float* rwork, blas_int const* lrwork,
               blas_int* iwork, blas_int const* liwork, blas_int* info);

  void zheevd_(char const* jobz, char const* uplo, blas_int const* n, uni20::complex<double>* a, blas_int const* lda,
               double* w, uni20::complex<double>* work, blas_int const* lwork, double* rwork, blas_int const* lrwork,
               blas_int* iwork, blas_int const* liwork, blas_int* info);

  void cheevr_(char const* jobz, char const* range, char const* uplo, blas_int const* n, uni20::complex<float>* a,
               blas_int const* lda, float const* vl, float const* vu, blas_int const* il, blas_int const* iu,
               float const* abstol, blas_int* m, float* w, uni20::complex<float>* z, blas_int const* ldz,
               blas_int* isuppz, uni20::complex<float>* work, blas_int const* lwork, float* rwork,
               blas_int const* lrwork, blas_int* iwork, blas_int const* liwork, blas_int* info);

  void zheevr_(char const* jobz, char const* range, char const* uplo, blas_int const* n, uni20::complex<double>* a,
               blas_int const* lda, double const* vl, double const* vu, blas_int const* il, blas_int const* iu,
               double const* abstol, blas_int* m, double* w, uni20::complex<double>* z, blas_int const* ldz,
               blas_int* isuppz, uni20::complex<double>* work, blas_int const* lwork, double* rwork,
               blas_int const* lrwork, blas_int* iwork, blas_int const* liwork, blas_int* info);

  void chegv_(blas_int const* itype, char const* jobz, char const* uplo, blas_int const* n, uni20::complex<float>* a,
              blas_int const* lda, uni20::complex<float>* b, blas_int const* ldb, float* w, uni20::complex<float>* work,
              blas_int const* lwork, float* rwork, blas_int* info);

  void zhegv_(blas_int const* itype, char const* jobz, char const* uplo, blas_int const* n, uni20::complex<double>* a,
              blas_int const* lda, uni20::complex<double>* b, blas_int const* ldb, double* w,
              uni20::complex<double>* work, blas_int const* lwork, double* rwork, blas_int* info);

  void chegvd_(blas_int const* itype, char const* jobz, char const* uplo, blas_int const* n, uni20::complex<float>* a,
               blas_int const* lda, uni20::complex<float>* b, blas_int const* ldb, float* w,
               uni20::complex<float>* work, blas_int const* lwork, float* rwork, blas_int const* lrwork,
               blas_int* iwork, blas_int const* liwork, blas_int* info);

  void zhegvd_(blas_int const* itype, char const* jobz, char const* uplo, blas_int const* n, uni20::complex<double>* a,
               blas_int const* lda, uni20::complex<double>* b, blas_int const* ldb, double* w,
               uni20::complex<double>* work, blas_int const* lwork, double* rwork, blas_int const* lrwork,
               blas_int* iwork, blas_int const* liwork, blas_int* info);

  void chegvx_(blas_int const* itype, char const* jobz, char const* range, char const* uplo, blas_int const* n,
               uni20::complex<float>* a, blas_int const* lda, uni20::complex<float>* b, blas_int const* ldb,
               float const* vl, float const* vu, blas_int const* il, blas_int const* iu, float const* abstol,
               blas_int* m, float* w, uni20::complex<float>* z, blas_int const* ldz, uni20::complex<float>* work,
               blas_int const* lwork, float* rwork, blas_int* iwork, blas_int* ifail, blas_int* info);

  void zhegvx_(blas_int const* itype, char const* jobz, char const* range, char const* uplo, blas_int const* n,
               uni20::complex<double>* a, blas_int const* lda, uni20::complex<double>* b, blas_int const* ldb,
               double const* vl, double const* vu, blas_int const* il, blas_int const* iu, double const* abstol,
               blas_int* m, double* w, uni20::complex<double>* z, blas_int const* ldz, uni20::complex<double>* work,
               blas_int const* lwork, double* rwork, blas_int* iwork, blas_int* ifail, blas_int* info);

  void ssytrd_(char const* uplo, blas_int const* n, float* a, blas_int const* lda, float* d, float* e, float* tau,
               float* work, blas_int const* lwork, blas_int* info);

  void dsytrd_(char const* uplo, blas_int const* n, double* a, blas_int const* lda, double* d, double* e, double* tau,
               double* work, blas_int const* lwork, blas_int* info);

  void ssytrf_(char const* uplo, blas_int const* n, float* a, blas_int const* lda, blas_int* ipiv, float* work,
               blas_int const* lwork, blas_int* info);

  void dsytrf_(char const* uplo, blas_int const* n, double* a, blas_int const* lda, blas_int* ipiv, double* work,
               blas_int const* lwork, blas_int* info);

  void ssytrs_(char const* uplo, blas_int const* n, blas_int const* nrhs, float* a, blas_int const* lda,
               blas_int const* ipiv, float* b, blas_int const* ldb, blas_int* info);

  void dsytrs_(char const* uplo, blas_int const* n, blas_int const* nrhs, double* a, blas_int const* lda,
               blas_int const* ipiv, double* b, blas_int const* ldb, blas_int* info);

  void ssytri_(char const* uplo, blas_int const* n, float* a, blas_int const* lda, blas_int const* ipiv, float* work,
               blas_int* info);

  void dsytri_(char const* uplo, blas_int const* n, double* a, blas_int const* lda, blas_int const* ipiv, double* work,
               blas_int* info);

  void ssyrfs_(char const* uplo, blas_int const* n, blas_int const* nrhs, float* a, blas_int const* lda, float* af,
               blas_int const* ldaf, blas_int* ipiv, float* b, blas_int const* ldb, float* x, blas_int const* ldx,
               float* ferr, float* berr, float* work, blas_int* iwork, blas_int* info);

  void dsyrfs_(char const* uplo, blas_int const* n, blas_int const* nrhs, double* a, blas_int const* lda, double* af,
               blas_int const* ldaf, blas_int* ipiv, double* b, blas_int const* ldb, double* x, blas_int const* ldx,
               double* ferr, double* berr, double* work, blas_int* iwork, blas_int* info);

  void ssysvx_(char const* fact, char const* uplo, blas_int const* n, blas_int const* nrhs, float* a,
               blas_int const* lda, float* af, blas_int const* ldaf, blas_int* ipiv, float* b, blas_int const* ldb,
               float* x, blas_int const* ldx, float* rcond, float* ferr, float* berr, float* work,
               blas_int const* lwork, blas_int* iwork, blas_int* info);

  void dsysvx_(char const* fact, char const* uplo, blas_int const* n, blas_int const* nrhs, double* a,
               blas_int const* lda, double* af, blas_int const* ldaf, blas_int* ipiv, double* b, blas_int const* ldb,
               double* x, blas_int const* ldx, double* rcond, double* ferr, double* berr, double* work,
               blas_int const* lwork, blas_int* iwork, blas_int* info);

  void ssycon_(char const* uplo, blas_int const* n, float* a, blas_int const* lda, blas_int const* ipiv,
               float const* anorm, float* rcond, float* work, blas_int* iwork, blas_int* info);

  void dsycon_(char const* uplo, blas_int const* n, double* a, blas_int const* lda, blas_int const* ipiv,
               double const* anorm, double* rcond, double* work, blas_int* iwork, blas_int* info);

  void strtrs_(char const* uplo, char const* trans, char const* diag, blas_int const* n, blas_int const* nrhs, float* a,
               blas_int const* lda, float* b, blas_int const* ldb, blas_int* info);

  void dtrtrs_(char const* uplo, char const* trans, char const* diag, blas_int const* n, blas_int const* nrhs,
               double* a, blas_int const* lda, double* b, blas_int const* ldb, blas_int* info);

  void strrfs_(char const* uplo, char const* trans, char const* diag, blas_int const* n, blas_int const* nrhs, float* a,
               blas_int const* lda, float* b, blas_int const* ldb, float* x, blas_int const* ldx, float* ferr,
               float* berr, float* work, blas_int* iwork, blas_int* info);

  void dtrrfs_(char const* uplo, char const* trans, char const* diag, blas_int const* n, blas_int const* nrhs,
               double* a, blas_int const* lda, double* b, blas_int const* ldb, double* x, blas_int const* ldx,
               double* ferr, double* berr, double* work, blas_int* iwork, blas_int* info);

  void strtri_(char const* uplo, char const* diag, blas_int const* n, float* a, blas_int const* lda, blas_int* info);

  void dtrtri_(char const* uplo, char const* diag, blas_int const* n, double* a, blas_int const* lda, blas_int* info);

  void strcon_(char const* norm, char const* uplo, char const* diag, blas_int const* n, float* a, blas_int const* lda,
               float* rcond, float* work, blas_int* iwork, blas_int* info);

  void dtrcon_(char const* norm, char const* uplo, char const* diag, blas_int const* n, double* a, blas_int const* lda,
               double* rcond, double* work, blas_int* iwork, blas_int* info);

  void strsyl_(char const* trana, char const* tranb, blas_int const* isgn, blas_int const* m, blas_int const* n,
               float* a, blas_int const* lda, float* b, blas_int const* ldb, float* c, blas_int const* ldc,
               float* scale, blas_int* info);

  void dtrsyl_(char const* trana, char const* tranb, blas_int const* isgn, blas_int const* m, blas_int const* n,
               double* a, blas_int const* lda, double* b, blas_int const* ldb, double* c, blas_int const* ldc,
               double* scale, blas_int* info);

  void sgeqrf_(blas_int const* m, blas_int const* n, float* a, blas_int const* lda, float* tau, float* work,
               blas_int const* lwork, blas_int* info);

  void dgeqrf_(blas_int const* m, blas_int const* n, double* a, blas_int const* lda, double* tau, double* work,
               blas_int const* lwork, blas_int* info);

  void sgelqf_(blas_int const* m, blas_int const* n, float* a, blas_int const* lda, float* tau, float* work,
               blas_int const* lwork, blas_int* info);

  void dgelqf_(blas_int const* m, blas_int const* n, double* a, blas_int const* lda, double* tau, double* work,
               blas_int const* lwork, blas_int* info);

  void sgeqlf_(blas_int const* m, blas_int const* n, float* a, blas_int const* lda, float* tau, float* work,
               blas_int const* lwork, blas_int* info);

  void dgeqlf_(blas_int const* m, blas_int const* n, double* a, blas_int const* lda, double* tau, double* work,
               blas_int const* lwork, blas_int* info);

  void sgerqf_(blas_int const* m, blas_int const* n, float* a, blas_int const* lda, float* tau, float* work,
               blas_int const* lwork, blas_int* info);

  void dgerqf_(blas_int const* m, blas_int const* n, double* a, blas_int const* lda, double* tau, double* work,
               blas_int const* lwork, blas_int* info);

  void sgebrd_(blas_int const* m, blas_int const* n, float* a, blas_int const* lda, float* d, float* e, float* tauq,
               float* taup, float* work, blas_int const* lwork, blas_int* info);

  void dgebrd_(blas_int const* m, blas_int const* n, double* a, blas_int const* lda, double* d, double* e, double* tauq,
               double* taup, double* work, blas_int const* lwork, blas_int* info);

  void sgehrd_(blas_int const* n, blas_int const* ilo, blas_int const* ihi, float* a, blas_int const* lda, float* tau,
               float* work, blas_int const* lwork, blas_int* info);

  void dgehrd_(blas_int const* n, blas_int const* ilo, blas_int const* ihi, double* a, blas_int const* lda, double* tau,
               double* work, blas_int const* lwork, blas_int* info);

  void sgeqp3_(blas_int const* m, blas_int const* n, float* a, blas_int const* lda, blas_int* jpvt, float* tau,
               float* work, blas_int const* lwork, blas_int* info);

  void dgeqp3_(blas_int const* m, blas_int const* n, double* a, blas_int const* lda, blas_int* jpvt, double* tau,
               double* work, blas_int const* lwork, blas_int* info);

  void sorgqr_(blas_int const* m, blas_int const* n, blas_int const* k, float* a, blas_int const* lda, float* tau,
               float* work, blas_int const* lwork, blas_int* info);

  void dorgqr_(blas_int const* m, blas_int const* n, blas_int const* k, double* a, blas_int const* lda, double* tau,
               double* work, blas_int const* lwork, blas_int* info);

  void sorglq_(blas_int const* m, blas_int const* n, blas_int const* k, float* a, blas_int const* lda, float* tau,
               float* work, blas_int const* lwork, blas_int* info);

  void dorglq_(blas_int const* m, blas_int const* n, blas_int const* k, double* a, blas_int const* lda, double* tau,
               double* work, blas_int const* lwork, blas_int* info);

  void sorgql_(blas_int const* m, blas_int const* n, blas_int const* k, float* a, blas_int const* lda, float* tau,
               float* work, blas_int const* lwork, blas_int* info);

  void dorgql_(blas_int const* m, blas_int const* n, blas_int const* k, double* a, blas_int const* lda, double* tau,
               double* work, blas_int const* lwork, blas_int* info);

  void sorgrq_(blas_int const* m, blas_int const* n, blas_int const* k, float* a, blas_int const* lda, float* tau,
               float* work, blas_int const* lwork, blas_int* info);

  void dorgrq_(blas_int const* m, blas_int const* n, blas_int const* k, double* a, blas_int const* lda, double* tau,
               double* work, blas_int const* lwork, blas_int* info);

  void sorgbr_(char const* vect, blas_int const* m, blas_int const* n, blas_int const* k, float* a, blas_int const* lda,
               float* tau, float* work, blas_int const* lwork, blas_int* info);

  void dorgbr_(char const* vect, blas_int const* m, blas_int const* n, blas_int const* k, double* a,
               blas_int const* lda, double* tau, double* work, blas_int const* lwork, blas_int* info);

  void sorghr_(blas_int const* n, blas_int const* ilo, blas_int const* ihi, float* a, blas_int const* lda,
               float const* tau, float* work, blas_int const* lwork, blas_int* info);

  void dorghr_(blas_int const* n, blas_int const* ilo, blas_int const* ihi, double* a, blas_int const* lda,
               double const* tau, double* work, blas_int const* lwork, blas_int* info);

  void sorgtr_(char const* uplo, blas_int const* n, float* a, blas_int const* lda, float const* tau, float* work,
               blas_int const* lwork, blas_int* info);

  void dorgtr_(char const* uplo, blas_int const* n, double* a, blas_int const* lda, double const* tau, double* work,
               blas_int const* lwork, blas_int* info);

  void sormhr_(char const* side, char const* trans, blas_int const* m, blas_int const* n, blas_int const* ilo,
               blas_int const* ihi, float* a, blas_int const* lda, float const* tau, float* c, blas_int const* ldc,
               float* work, blas_int const* lwork, blas_int* info);

  void dormhr_(char const* side, char const* trans, blas_int const* m, blas_int const* n, blas_int const* ilo,
               blas_int const* ihi, double* a, blas_int const* lda, double const* tau, double* c, blas_int const* ldc,
               double* work, blas_int const* lwork, blas_int* info);

  void sormtr_(char const* side, char const* uplo, char const* trans, blas_int const* m, blas_int const* n, float* a,
               blas_int const* lda, float const* tau, float* c, blas_int const* ldc, float* work, blas_int const* lwork,
               blas_int* info);

  void dormtr_(char const* side, char const* uplo, char const* trans, blas_int const* m, blas_int const* n, double* a,
               blas_int const* lda, double const* tau, double* c, blas_int const* ldc, double* work,
               blas_int const* lwork, blas_int* info);

  void sormbr_(char const* vect, char const* side, char const* trans, blas_int const* m, blas_int const* n,
               blas_int const* k, float* a, blas_int const* lda, float* tau, float* c, blas_int const* ldc, float* work,
               blas_int const* lwork, blas_int* info);

  void dormbr_(char const* vect, char const* side, char const* trans, blas_int const* m, blas_int const* n,
               blas_int const* k, double* a, blas_int const* lda, double* tau, double* c, blas_int const* ldc,
               double* work, blas_int const* lwork, blas_int* info);

  void sormqr_(char const* side, char const* trans, blas_int const* m, blas_int const* n, blas_int const* k, float* a,
               blas_int const* lda, float* tau, float* c, blas_int const* ldc, float* work, blas_int const* lwork,
               blas_int* info);

  void dormqr_(char const* side, char const* trans, blas_int const* m, blas_int const* n, blas_int const* k, double* a,
               blas_int const* lda, double* tau, double* c, blas_int const* ldc, double* work, blas_int const* lwork,
               blas_int* info);

  void sormlq_(char const* side, char const* trans, blas_int const* m, blas_int const* n, blas_int const* k, float* a,
               blas_int const* lda, float* tau, float* c, blas_int const* ldc, float* work, blas_int const* lwork,
               blas_int* info);

  void dormlq_(char const* side, char const* trans, blas_int const* m, blas_int const* n, blas_int const* k, double* a,
               blas_int const* lda, double* tau, double* c, blas_int const* ldc, double* work, blas_int const* lwork,
               blas_int* info);

  void sormql_(char const* side, char const* trans, blas_int const* m, blas_int const* n, blas_int const* k, float* a,
               blas_int const* lda, float* tau, float* c, blas_int const* ldc, float* work, blas_int const* lwork,
               blas_int* info);

  void dormql_(char const* side, char const* trans, blas_int const* m, blas_int const* n, blas_int const* k, double* a,
               blas_int const* lda, double* tau, double* c, blas_int const* ldc, double* work, blas_int const* lwork,
               blas_int* info);

  void sormrq_(char const* side, char const* trans, blas_int const* m, blas_int const* n, blas_int const* k, float* a,
               blas_int const* lda, float* tau, float* c, blas_int const* ldc, float* work, blas_int const* lwork,
               blas_int* info);

  void dormrq_(char const* side, char const* trans, blas_int const* m, blas_int const* n, blas_int const* k, double* a,
               blas_int const* lda, double* tau, double* c, blas_int const* ldc, double* work, blas_int const* lwork,
               blas_int* info);

  void sgesvd_(char const* jobu, char const* jobvt, blas_int const* m, blas_int const* n, float* a, blas_int const* lda,
               float* s, float* u, blas_int const* ldu, float* vt, blas_int const* ldvt, float* work,
               blas_int const* lwork, blas_int* info);

  void dgesvd_(char const* jobu, char const* jobvt, blas_int const* m, blas_int const* n, double* a,
               blas_int const* lda, double* s, double* u, blas_int const* ldu, double* vt, blas_int const* ldvt,
               double* work, blas_int const* lwork, blas_int* info);

  void sgesdd_(char const* jobz, blas_int const* m, blas_int const* n, float* a, blas_int const* lda, float* s,
               float* u, blas_int const* ldu, float* vt, blas_int const* ldvt, float* work, blas_int const* lwork,
               blas_int* iwork, blas_int* info);

  void dgesdd_(char const* jobz, blas_int const* m, blas_int const* n, double* a, blas_int const* lda, double* s,
               double* u, blas_int const* ldu, double* vt, blas_int const* ldvt, double* work, blas_int const* lwork,
               blas_int* iwork, blas_int* info);

  void sgesvdx_(char const* jobu, char const* jobvt, char const* range, blas_int const* m, blas_int const* n, float* a,
                blas_int const* lda, float const* vl, float const* vu, blas_int const* il, blas_int const* iu,
                blas_int* ns, float* s, float* u, blas_int const* ldu, float* vt, blas_int const* ldvt, float* work,
                blas_int const* lwork, blas_int* iwork, blas_int* info);

  void dgesvdx_(char const* jobu, char const* jobvt, char const* range, blas_int const* m, blas_int const* n, double* a,
                blas_int const* lda, double const* vl, double const* vu, blas_int const* il, blas_int const* iu,
                blas_int* ns, double* s, double* u, blas_int const* ldu, double* vt, blas_int const* ldvt, double* work,
                blas_int const* lwork, blas_int* iwork, blas_int* info);

  void spotrf_(char const* uplo, blas_int const* n, float* a, blas_int const* lda, blas_int* info);

  void dpotrf_(char const* uplo, blas_int const* n, double* a, blas_int const* lda, blas_int* info);

  void spbsv_(char const* uplo, blas_int const* n, blas_int const* kd, blas_int const* nrhs, float* ab,
              blas_int const* ldab, float* b, blas_int const* ldb, blas_int* info);

  void dpbsv_(char const* uplo, blas_int const* n, blas_int const* kd, blas_int const* nrhs, double* ab,
              blas_int const* ldab, double* b, blas_int const* ldb, blas_int* info);

  void spbtrf_(char const* uplo, blas_int const* n, blas_int const* kd, float* ab, blas_int const* ldab,
               blas_int* info);

  void dpbtrf_(char const* uplo, blas_int const* n, blas_int const* kd, double* ab, blas_int const* ldab,
               blas_int* info);

  void spbtrs_(char const* uplo, blas_int const* n, blas_int const* kd, blas_int const* nrhs, float* ab,
               blas_int const* ldab, float* b, blas_int const* ldb, blas_int* info);

  void dpbtrs_(char const* uplo, blas_int const* n, blas_int const* kd, blas_int const* nrhs, double* ab,
               blas_int const* ldab, double* b, blas_int const* ldb, blas_int* info);

  void spbcon_(char const* uplo, blas_int const* n, blas_int const* kd, float* ab, blas_int const* ldab,
               float const* anorm, float* rcond, float* work, blas_int* iwork, blas_int* info);

  void dpbcon_(char const* uplo, blas_int const* n, blas_int const* kd, double* ab, blas_int const* ldab,
               double const* anorm, double* rcond, double* work, blas_int* iwork, blas_int* info);

  void spbrfs_(char const* uplo, blas_int const* n, blas_int const* kd, blas_int const* nrhs, float* ab,
               blas_int const* ldab, float* afb, blas_int const* ldafb, float* b, blas_int const* ldb, float* x,
               blas_int const* ldx, float* ferr, float* berr, float* work, blas_int* iwork, blas_int* info);

  void dpbrfs_(char const* uplo, blas_int const* n, blas_int const* kd, blas_int const* nrhs, double* ab,
               blas_int const* ldab, double* afb, blas_int const* ldafb, double* b, blas_int const* ldb, double* x,
               blas_int const* ldx, double* ferr, double* berr, double* work, blas_int* iwork, blas_int* info);

  void spbsvx_(char const* fact, char const* uplo, blas_int const* n, blas_int const* kd, blas_int const* nrhs,
               float* ab, blas_int const* ldab, float* afb, blas_int const* ldafb, char* equed, float* s, float* b,
               blas_int const* ldb, float* x, blas_int const* ldx, float* rcond, float* ferr, float* berr, float* work,
               blas_int* iwork, blas_int* info);

  void dpbsvx_(char const* fact, char const* uplo, blas_int const* n, blas_int const* kd, blas_int const* nrhs,
               double* ab, blas_int const* ldab, double* afb, blas_int const* ldafb, char* equed, double* s, double* b,
               blas_int const* ldb, double* x, blas_int const* ldx, double* rcond, double* ferr, double* berr,
               double* work, blas_int* iwork, blas_int* info);

  void sptsv_(blas_int const* n, blas_int const* nrhs, float* d, float* e, float* b, blas_int const* ldb,
              blas_int* info);

  void dptsv_(blas_int const* n, blas_int const* nrhs, double* d, double* e, double* b, blas_int const* ldb,
              blas_int* info);

  void spttrf_(blas_int const* n, float* d, float* e, blas_int* info);

  void dpttrf_(blas_int const* n, double* d, double* e, blas_int* info);

  void spttrs_(blas_int const* n, blas_int const* nrhs, float* d, float* e, float* b, blas_int const* ldb,
               blas_int* info);

  void dpttrs_(blas_int const* n, blas_int const* nrhs, double* d, double* e, double* b, blas_int const* ldb,
               blas_int* info);

  void sptcon_(blas_int const* n, float* d, float* e, float const* anorm, float* rcond, float* work, blas_int* info);

  void dptcon_(blas_int const* n, double* d, double* e, double const* anorm, double* rcond, double* work,
               blas_int* info);

  void sptrfs_(blas_int const* n, blas_int const* nrhs, float* d, float* e, float* df, float* ef, float* b,
               blas_int const* ldb, float* x, blas_int const* ldx, float* ferr, float* berr, float* work,
               blas_int* info);

  void dptrfs_(blas_int const* n, blas_int const* nrhs, double* d, double* e, double* df, double* ef, double* b,
               blas_int const* ldb, double* x, blas_int const* ldx, double* ferr, double* berr, double* work,
               blas_int* info);

  void sptsvx_(char const* fact, blas_int const* n, blas_int const* nrhs, float* d, float* e, float* df, float* ef,
               float* b, blas_int const* ldb, float* x, blas_int const* ldx, float* rcond, float* ferr, float* berr,
               float* work, blas_int* info);

  void dptsvx_(char const* fact, blas_int const* n, blas_int const* nrhs, double* d, double* e, double* df, double* ef,
               double* b, blas_int const* ldb, double* x, blas_int const* ldx, double* rcond, double* ferr,
               double* berr, double* work, blas_int* info);

  void spstrf_(char const* uplo, blas_int const* n, float* a, blas_int const* lda, blas_int* piv, blas_int* rank,
               float const* tol, float* work, blas_int* info);

  void dpstrf_(char const* uplo, blas_int const* n, double* a, blas_int const* lda, blas_int* piv, blas_int* rank,
               double const* tol, double* work, blas_int* info);

  void spotri_(char const* uplo, blas_int const* n, float* a, blas_int const* lda, blas_int* info);

  void dpotri_(char const* uplo, blas_int const* n, double* a, blas_int const* lda, blas_int* info);

  void spotrs_(char const* uplo, blas_int const* n, blas_int const* nrhs, float* a, blas_int const* lda, float* b,
               blas_int const* ldb, blas_int* info);

  void dpotrs_(char const* uplo, blas_int const* n, blas_int const* nrhs, double* a, blas_int const* lda, double* b,
               blas_int const* ldb, blas_int* info);

  void sposvx_(char const* fact, char const* uplo, blas_int const* n, blas_int const* nrhs, float* a,
               blas_int const* lda, float* af, blas_int const* ldaf, char* equed, float* s, float* b,
               blas_int const* ldb, float* x, blas_int const* ldx, float* rcond, float* ferr, float* berr, float* work,
               blas_int* iwork, blas_int* info);

  void dposvx_(char const* fact, char const* uplo, blas_int const* n, blas_int const* nrhs, double* a,
               blas_int const* lda, double* af, blas_int const* ldaf, char* equed, double* s, double* b,
               blas_int const* ldb, double* x, blas_int const* ldx, double* rcond, double* ferr, double* berr,
               double* work, blas_int* iwork, blas_int* info);

  void spoequ_(blas_int const* n, float* a, blas_int const* lda, float* s, float* scond, float* amax, blas_int* info);

  void dpoequ_(blas_int const* n, double* a, blas_int const* lda, double* s, double* scond, double* amax,
               blas_int* info);

  void sporfs_(char const* uplo, blas_int const* n, blas_int const* nrhs, float* a, blas_int const* lda, float* af,
               blas_int const* ldaf, float* b, blas_int const* ldb, float* x, blas_int const* ldx, float* ferr,
               float* berr, float* work, blas_int* iwork, blas_int* info);

  void dporfs_(char const* uplo, blas_int const* n, blas_int const* nrhs, double* a, blas_int const* lda, double* af,
               blas_int const* ldaf, double* b, blas_int const* ldb, double* x, blas_int const* ldx, double* ferr,
               double* berr, double* work, blas_int* iwork, blas_int* info);

  void spocon_(char const* uplo, blas_int const* n, float* a, blas_int const* lda, float const* anorm, float* rcond,
               float* work, blas_int* iwork, blas_int* info);

  void dpocon_(char const* uplo, blas_int const* n, double* a, blas_int const* lda, double const* anorm, double* rcond,
               double* work, blas_int* iwork, blas_int* info);

  void sgeev_(char const* jobvl, char const* jobvr, blas_int const* n, float* a, blas_int const* lda, float* wr,
              float* wi, float* vl, blas_int const* ldvl, float* vr, blas_int const* ldvr, float* work,
              blas_int const* lwork, blas_int* info);

  void dgeev_(char const* jobvl, char const* jobvr, blas_int const* n, double* a, blas_int const* lda, double* wr,
              double* wi, double* vl, blas_int const* ldvl, double* vr, blas_int const* ldvr, double* work,
              blas_int const* lwork, blas_int* info);

  void sgeevx_(char const* balanc, char const* jobvl, char const* jobvr, char const* sense, blas_int const* n, float* a,
               blas_int const* lda, float* wr, float* wi, float* vl, blas_int const* ldvl, float* vr,
               blas_int const* ldvr, blas_int* ilo, blas_int* ihi, float* scale, float* abnrm, float* rconde,
               float* rcondv, float* work, blas_int const* lwork, blas_int* iwork, blas_int* info);

  void dgeevx_(char const* balanc, char const* jobvl, char const* jobvr, char const* sense, blas_int const* n,
               double* a, blas_int const* lda, double* wr, double* wi, double* vl, blas_int const* ldvl, double* vr,
               blas_int const* ldvr, blas_int* ilo, blas_int* ihi, double* scale, double* abnrm, double* rconde,
               double* rcondv, double* work, blas_int const* lwork, blas_int* iwork, blas_int* info);

  void sgebal_(char const* job, blas_int const* n, float* a, blas_int const* lda, blas_int* ilo, blas_int* ihi,
               float* scale, blas_int* info);

  void dgebal_(char const* job, blas_int const* n, double* a, blas_int const* lda, blas_int* ilo, blas_int* ihi,
               double* scale, blas_int* info);

  void sgebak_(char const* job, char const* side, blas_int const* n, blas_int const* ilo, blas_int const* ihi,
               float* scale, blas_int const* m, float* v, blas_int const* ldv, blas_int* info);

  void dgebak_(char const* job, char const* side, blas_int const* n, blas_int const* ilo, blas_int const* ihi,
               double* scale, blas_int const* m, double* v, blas_int const* ldv, blas_int* info);

  void sggev_(char const* jobvl, char const* jobvr, blas_int const* n, float* a, blas_int const* lda, float* b,
              blas_int const* ldb, float* alphar, float* alphai, float* beta, float* vl, blas_int const* ldvl,
              float* vr, blas_int const* ldvr, float* work, blas_int const* lwork, blas_int* info);

  void dggev_(char const* jobvl, char const* jobvr, blas_int const* n, double* a, blas_int const* lda, double* b,
              blas_int const* ldb, double* alphar, double* alphai, double* beta, double* vl, blas_int const* ldvl,
              double* vr, blas_int const* ldvr, double* work, blas_int const* lwork, blas_int* info);

  void sggevx_(char const* balanc, char const* jobvl, char const* jobvr, char const* sense, blas_int const* n, float* a,
               blas_int const* lda, float* b, blas_int const* ldb, float* alphar, float* alphai, float* beta, float* vl,
               blas_int const* ldvl, float* vr, blas_int const* ldvr, blas_int* ilo, blas_int* ihi, float* lscale,
               float* rscale, float* abnrm, float* bbnrm, float* rconde, float* rcondv, float* work,
               blas_int const* lwork, blas_int* iwork, blas_int* bwork, blas_int* info);

  void dggevx_(char const* balanc, char const* jobvl, char const* jobvr, char const* sense, blas_int const* n,
               double* a, blas_int const* lda, double* b, blas_int const* ldb, double* alphar, double* alphai,
               double* beta, double* vl, blas_int const* ldvl, double* vr, blas_int const* ldvr, blas_int* ilo,
               blas_int* ihi, double* lscale, double* rscale, double* abnrm, double* bbnrm, double* rconde,
               double* rcondv, double* work, blas_int const* lwork, blas_int* iwork, blas_int* bwork, blas_int* info);

  void sggbal_(char const* job, blas_int const* n, float* a, blas_int const* lda, float* b, blas_int const* ldb,
               blas_int* ilo, blas_int* ihi, float* lscale, float* rscale, float* work, blas_int* info);

  void dggbal_(char const* job, blas_int const* n, double* a, blas_int const* lda, double* b, blas_int const* ldb,
               blas_int* ilo, blas_int* ihi, double* lscale, double* rscale, double* work, blas_int* info);

  void sggbak_(char const* job, char const* side, blas_int const* n, blas_int const* ilo, blas_int const* ihi,
               float* lscale, float* rscale, blas_int const* m, float* v, blas_int const* ldv, blas_int* info);

  void dggbak_(char const* job, char const* side, blas_int const* n, blas_int const* ilo, blas_int const* ihi,
               double* lscale, double* rscale, blas_int const* m, double* v, blas_int const* ldv, blas_int* info);

  void sgges_(char const* jobvsl, char const* jobvsr, char const* sort, void* select, blas_int const* n, float* a,
              blas_int const* lda, float* b, blas_int const* ldb, blas_int* selected_dimension, float* alphar,
              float* alphai, float* beta, float* vsl, blas_int const* ldvsl, float* vsr, blas_int const* ldvsr,
              float* work, blas_int const* lwork, blas_int* bwork, blas_int* info);

  void dgges_(char const* jobvsl, char const* jobvsr, char const* sort, void* select, blas_int const* n, double* a,
              blas_int const* lda, double* b, blas_int const* ldb, blas_int* selected_dimension, double* alphar,
              double* alphai, double* beta, double* vsl, blas_int const* ldvsl, double* vsr, blas_int const* ldvsr,
              double* work, blas_int const* lwork, blas_int* bwork, blas_int* info);

  void sgghrd_(char const* compq, char const* compz, blas_int const* n, blas_int const* ilo, blas_int const* ihi,
               float* a, blas_int const* lda, float* b, blas_int const* ldb, float* q, blas_int const* ldq, float* z,
               blas_int const* ldz, blas_int* info);

  void dgghrd_(char const* compq, char const* compz, blas_int const* n, blas_int const* ilo, blas_int const* ihi,
               double* a, blas_int const* lda, double* b, blas_int const* ldb, double* q, blas_int const* ldq,
               double* z, blas_int const* ldz, blas_int* info);

  void shgeqz_(char const* job, char const* compq, char const* compz, blas_int const* n, blas_int const* ilo,
               blas_int const* ihi, float* h, blas_int const* ldh, float* t, blas_int const* ldt, float* alphar,
               float* alphai, float* beta, float* q, blas_int const* ldq, float* z, blas_int const* ldz, float* work,
               blas_int const* lwork, blas_int* info);

  void dhgeqz_(char const* job, char const* compq, char const* compz, blas_int const* n, blas_int const* ilo,
               blas_int const* ihi, double* h, blas_int const* ldh, double* t, blas_int const* ldt, double* alphar,
               double* alphai, double* beta, double* q, blas_int const* ldq, double* z, blas_int const* ldz,
               double* work, blas_int const* lwork, blas_int* info);

  void stgexc_(blas_int const* wantq, blas_int const* wantz, blas_int const* n, float* a, blas_int const* lda, float* b,
               blas_int const* ldb, float* q, blas_int const* ldq, float* z, blas_int const* ldz, blas_int* first,
               blas_int* last, float* work, blas_int const* lwork, blas_int* info);

  void dtgexc_(blas_int const* wantq, blas_int const* wantz, blas_int const* n, double* a, blas_int const* lda,
               double* b, blas_int const* ldb, double* q, blas_int const* ldq, double* z, blas_int const* ldz,
               blas_int* first, blas_int* last, double* work, blas_int const* lwork, blas_int* info);

  void stgsen_(blas_int const* ijob, blas_int const* wantq, blas_int const* wantz, blas_int* select, blas_int const* n,
               float* a, blas_int const* lda, float* b, blas_int const* ldb, float* alphar, float* alphai, float* beta,
               float* q, blas_int const* ldq, float* z, blas_int const* ldz, blas_int* selected_dimension, float* pl,
               float* pr, float* dif, float* work, blas_int const* lwork, blas_int* iwork, blas_int const* liwork,
               blas_int* info);

  void dtgsen_(blas_int const* ijob, blas_int const* wantq, blas_int const* wantz, blas_int* select, blas_int const* n,
               double* a, blas_int const* lda, double* b, blas_int const* ldb, double* alphar, double* alphai,
               double* beta, double* q, blas_int const* ldq, double* z, blas_int const* ldz,
               blas_int* selected_dimension, double* pl, double* pr, double* dif, double* work, blas_int const* lwork,
               blas_int* iwork, blas_int const* liwork, blas_int* info);

  void stgevc_(char const* side, char const* howmny, blas_int* select, blas_int const* n, float* s, blas_int const* lds,
               float* p, blas_int const* ldp, float* vl, blas_int const* ldvl, float* vr, blas_int const* ldvr,
               blas_int const* mm, blas_int* computed_vectors, float* work, blas_int* info);

  void dtgevc_(char const* side, char const* howmny, blas_int* select, blas_int const* n, double* s,
               blas_int const* lds, double* p, blas_int const* ldp, double* vl, blas_int const* ldvl, double* vr,
               blas_int const* ldvr, blas_int const* mm, blas_int* computed_vectors, double* work, blas_int* info);

  void stgsna_(char const* job, char const* howmny, blas_int* select, blas_int const* n, float* a, blas_int const* lda,
               float* b, blas_int const* ldb, float* vl, blas_int const* ldvl, float* vr, blas_int const* ldvr,
               float* s, float* dif, blas_int const* mm, blas_int* computed_estimates, float* work,
               blas_int const* lwork, blas_int* iwork, blas_int* info);

  void dtgsna_(char const* job, char const* howmny, blas_int* select, blas_int const* n, double* a, blas_int const* lda,
               double* b, blas_int const* ldb, double* vl, blas_int const* ldvl, double* vr, blas_int const* ldvr,
               double* s, double* dif, blas_int const* mm, blas_int* computed_estimates, double* work,
               blas_int const* lwork, blas_int* iwork, blas_int* info);

  void cgeev_(char const* jobvl, char const* jobvr, blas_int const* n, uni20::complex<float>* a, blas_int const* lda,
              uni20::complex<float>* w, uni20::complex<float>* vl, blas_int const* ldvl, uni20::complex<float>* vr,
              blas_int const* ldvr, uni20::complex<float>* work, blas_int const* lwork, float* rwork, blas_int* info);

  void zgeev_(char const* jobvl, char const* jobvr, blas_int const* n, uni20::complex<double>* a, blas_int const* lda,
              uni20::complex<double>* w, uni20::complex<double>* vl, blas_int const* ldvl, uni20::complex<double>* vr,
              blas_int const* ldvr, uni20::complex<double>* work, blas_int const* lwork, double* rwork, blas_int* info);

  void cgees_(char const* jobvs, char const* sort, void* select, blas_int const* n, uni20::complex<float>* a,
              blas_int const* lda, blas_int* sdim, uni20::complex<float>* w, uni20::complex<float>* vs,
              blas_int const* ldvs, uni20::complex<float>* work, blas_int const* lwork, float* rwork, blas_int* bwork,
              blas_int* info);

  void zgees_(char const* jobvs, char const* sort, void* select, blas_int const* n, uni20::complex<double>* a,
              blas_int const* lda, blas_int* sdim, uni20::complex<double>* w, uni20::complex<double>* vs,
              blas_int const* ldvs, uni20::complex<double>* work, blas_int const* lwork, double* rwork, blas_int* bwork,
              blas_int* info);

  void sgees_(char const* jobvs, char const* sort, void* select, blas_int const* n, float* a, blas_int const* lda,
              blas_int* sdim, float* wr, float* wi, float* vs, blas_int const* ldvs, float* work, blas_int const* lwork,
              blas_int* bwork, blas_int* info);

  void dgees_(char const* jobvs, char const* sort, void* select, blas_int const* n, double* a, blas_int const* lda,
              blas_int* sdim, double* wr, double* wi, double* vs, blas_int const* ldvs, double* work,
              blas_int const* lwork, blas_int* bwork, blas_int* info);

  void shseqr_(char const* job, char const* compz, blas_int const* n, blas_int const* ilo, blas_int const* ihi,
               float* h, blas_int const* ldh, float* wr, float* wi, float* z, blas_int const* ldz, float* work,
               blas_int const* lwork, blas_int* info);

  void dhseqr_(char const* job, char const* compz, blas_int const* n, blas_int const* ilo, blas_int const* ihi,
               double* h, blas_int const* ldh, double* wr, double* wi, double* z, blas_int const* ldz, double* work,
               blas_int const* lwork, blas_int* info);

  void strexc_(char const* compq, blas_int const* n, float* t, blas_int const* ldt, float* q, blas_int const* ldq,
               blas_int* ifst, blas_int* ilst, float* work, blas_int* info);

  void dtrexc_(char const* compq, blas_int const* n, double* t, blas_int const* ldt, double* q, blas_int const* ldq,
               blas_int* ifst, blas_int* ilst, double* work, blas_int* info);

  void strsen_(char const* job, char const* compq, blas_int* select, blas_int const* n, float* t, blas_int const* ldt,
               float* q, blas_int const* ldq, float* wr, float* wi, blas_int* m, float* s, float* sep, float* work,
               blas_int const* lwork, blas_int* iwork, blas_int const* liwork, blas_int* info);

  void dtrsen_(char const* job, char const* compq, blas_int* select, blas_int const* n, double* t, blas_int const* ldt,
               double* q, blas_int const* ldq, double* wr, double* wi, blas_int* m, double* s, double* sep,
               double* work, blas_int const* lwork, blas_int* iwork, blas_int const* liwork, blas_int* info);

  void strevc_(char const* side, char const* howmny, blas_int* select, blas_int const* n, float* t, blas_int const* ldt,
               float* vl, blas_int const* ldvl, float* vr, blas_int const* ldvr, blas_int const* mm, blas_int* m,
               float* work, blas_int* info);

  void dtrevc_(char const* side, char const* howmny, blas_int* select, blas_int const* n, double* t,
               blas_int const* ldt, double* vl, blas_int const* ldvl, double* vr, blas_int const* ldvr,
               blas_int const* mm, blas_int* m, double* work, blas_int* info);

  void strsna_(char const* job, char const* howmny, blas_int* select, blas_int const* n, float* t, blas_int const* ldt,
               float* vl, blas_int const* ldvl, float* vr, blas_int const* ldvr, float* s, float* sep,
               blas_int const* mm, blas_int* m, float* work, blas_int const* ldwork, blas_int* iwork, blas_int* info);

  void dtrsna_(char const* job, char const* howmny, blas_int* select, blas_int const* n, double* t, blas_int const* ldt,
               double* vl, blas_int const* ldvl, double* vr, blas_int const* ldvr, double* s, double* sep,
               blas_int const* mm, blas_int* m, double* work, blas_int const* ldwork, blas_int* iwork, blas_int* info);

  void ctrexc_(char const* compq, blas_int const* n, uni20::complex<float>* t, blas_int const* ldt,
               uni20::complex<float>* q, blas_int const* ldq, blas_int* ifst, blas_int* ilst, blas_int* info);

  void ztrexc_(char const* compq, blas_int const* n, uni20::complex<double>* t, blas_int const* ldt,
               uni20::complex<double>* q, blas_int const* ldq, blas_int* ifst, blas_int* ilst, blas_int* info);
}

inline blas_int checked_blas_int(std::size_t value)
{
  auto const converted = static_cast<blas_int>(value);
  if (static_cast<std::size_t>(converted) != value)
  {
    throw std::overflow_error("dense Krylov dimension does not fit BLAS integer type");
  }
  return converted;
}

inline char lapack_norm(MatrixNorm norm)
{
  switch (norm)
  {
    case MatrixNorm::MaxAbs:
      return 'M';
    case MatrixNorm::One:
      return '1';
    case MatrixNorm::Infinity:
      return 'I';
    case MatrixNorm::Frobenius:
      return 'F';
  }
  throw std::invalid_argument("dense real LAPACK operation received an unknown norm selector");
}

inline blas_int gelsd_iwork_size(blas_int m, blas_int n)
{
  blas_int const minmn = std::min(m, n);
  if (minmn <= 0)
  {
    return 1;
  }

  // LAPACK's bound uses SMLSIZ from ILAENV. Using divisor 2 overestimates the
  // subdivision depth, avoiding a runtime dependency on ILAENV in this wrapper.
  std::size_t levels = 0;
  for (std::size_t size = static_cast<std::size_t>(minmn) / 2; size > 0; size /= 2)
  {
    ++levels;
  }
  std::size_t const minmn_size = static_cast<std::size_t>(minmn);
  std::size_t const required = 3 * minmn_size * levels + 11 * minmn_size;
  return checked_blas_int(std::max<std::size_t>(1, required));
}

template <typename Scalar> struct is_complex : std::false_type
{};
template <typename Real> struct is_complex<uni20::complex<Real>> : std::true_type
{};

template <typename Scalar> inline constexpr bool is_complex_v = is_complex<Scalar>::value;

#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
template <typename Scalar> inline constexpr bool is_mplapack_binary128_v = std::same_as<Scalar, mplapack_binary128_t>;
#else
template <typename Scalar> inline constexpr bool is_mplapack_binary128_v = false;
#endif

template <typename Scalar> constexpr Scalar conjugate_if_complex(Scalar const& value)
{
  if constexpr (is_complex_v<Scalar>)
  {
    return std::conj(value);
  }
  else
  {
    return value;
  }
}

template <typename Scalar> void check_same_size(std::span<Scalar const> source, std::span<Scalar> destination)
{
  if (source.size() != destination.size())
  {
    throw std::invalid_argument("dense vector sizes do not agree");
  }
}

template <typename Scalar> bool in_selected_region(std::size_t row, std::size_t col, MatrixFill fill)
{
  switch (fill)
  {
    case MatrixFill::All:
      return true;
    case MatrixFill::Upper:
      return row <= col;
    case MatrixFill::Lower:
      return row >= col;
  }
  return false;
}

template <uni20::LapackReal Scalar>
Scalar lange(char norm, blas_int m, blas_int n, Scalar* a, blas_int lda, Scalar* work)
{
  return uni20::lapack::lange(norm, m, n, a, lda, work);
}

template <uni20::LapackReal Scalar>
Scalar lansy(char norm, char uplo, blas_int n, Scalar* a, blas_int lda, Scalar* work)
{
  return uni20::lapack::lansy(norm, uplo, n, a, lda, work);
}

template <uni20::LapackReal Scalar>
Scalar lantr(char norm, char uplo, char diag, blas_int m, blas_int n, Scalar* a, blas_int lda, Scalar* work)
{
  return uni20::lapack::lantr(norm, uplo, diag, m, n, a, lda, work);
}

template <uni20::LapackReal Scalar>
Scalar langb(char norm, blas_int n, blas_int kl, blas_int ku, Scalar* ab, blas_int ldab, Scalar* work)
{
  return uni20::lapack::langb(norm, n, kl, ku, ab, ldab, work);
}

template <uni20::LapackReal Scalar>
void gbsv(blas_int n, blas_int kl, blas_int ku, blas_int nrhs, Scalar* ab, blas_int ldab, blas_int* ipiv, Scalar* b,
          blas_int ldb)
{
  uni20::lapack::gbsv(n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb);
}

template <uni20::LapackReal Scalar>
void gbtrf(blas_int m, blas_int n, blas_int kl, blas_int ku, Scalar* ab, blas_int ldab, blas_int* ipiv)
{
  uni20::lapack::gbtrf(m, n, kl, ku, ab, ldab, ipiv);
}

template <uni20::LapackReal Scalar>
void gbtrs(char trans, blas_int n, blas_int kl, blas_int ku, blas_int nrhs, Scalar* ab, blas_int ldab,
           blas_int const* ipiv, Scalar* b, blas_int ldb)
{
  uni20::lapack::gbtrs(trans, n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb);
}

template <uni20::LapackReal Scalar>
Scalar gbcon(char norm, blas_int n, blas_int kl, blas_int ku, Scalar* ab, blas_int ldab, blas_int const* ipiv,
             Scalar anorm, Scalar* work, blas_int* iwork)
{
  return uni20::lapack::gbcon(norm, n, kl, ku, ab, ldab, ipiv, anorm, work, iwork);
}

template <uni20::LapackReal Scalar>
void gbrfs(char trans, blas_int n, blas_int kl, blas_int ku, blas_int nrhs, Scalar* ab, blas_int ldab, Scalar* afb,
           blas_int ldafb, blas_int const* ipiv, Scalar* b, blas_int ldb, Scalar* x, blas_int ldx,
           Scalar* forward_error, Scalar* backward_error, Scalar* work, blas_int* iwork)
{
  uni20::lapack::gbrfs(trans, n, kl, ku, nrhs, ab, ldab, afb, ldafb, ipiv, b, ldb, x, ldx, forward_error,
                       backward_error, work, iwork);
}

template <uni20::LapackReal Scalar>
bool gbsvx(char fact, char trans, blas_int n, blas_int kl, blas_int ku, blas_int nrhs, Scalar* ab, blas_int ldab,
           Scalar* afb, blas_int ldafb, blas_int* ipiv, char& equed, Scalar* row_scale, Scalar* column_scale, Scalar* b,
           blas_int ldb, Scalar* x, blas_int ldx, Scalar& rcond, Scalar* forward_error, Scalar* backward_error,
           Scalar* work, blas_int* iwork)
{
  return uni20::lapack::gbsvx(fact, trans, n, kl, ku, nrhs, ab, ldab, afb, ldafb, ipiv, equed, row_scale, column_scale,
                              b, ldb, x, ldx, rcond, forward_error, backward_error, work, iwork);
}

template <uni20::LapackReal Scalar>
void gbequ(bool power_of_two, blas_int m, blas_int n, blas_int kl, blas_int ku, Scalar* ab, blas_int ldab,
           Scalar* row_scale, Scalar* column_scale, Scalar& row_condition, Scalar& column_condition, Scalar& max_abs)
{
  uni20::lapack::gbequ(power_of_two, m, n, kl, ku, ab, ldab, row_scale, column_scale, row_condition, column_condition,
                       max_abs);
}

template <uni20::LapackReal Scalar>
void gtsv(blas_int n, blas_int nrhs, Scalar* dl, Scalar* d, Scalar* du, Scalar* b, blas_int ldb)
{
  uni20::lapack::gtsv(n, nrhs, dl, d, du, b, ldb);
}

template <uni20::LapackReal Scalar>
void gttrf(blas_int n, Scalar* dl, Scalar* d, Scalar* du, Scalar* du2, blas_int* ipiv)
{
  uni20::lapack::gttrf(n, dl, d, du, du2, ipiv);
}

template <uni20::LapackReal Scalar>
void gttrs(char trans, blas_int n, blas_int nrhs, Scalar* dl, Scalar* d, Scalar* du, Scalar* du2, blas_int const* ipiv,
           Scalar* b, blas_int ldb)
{
  uni20::lapack::gttrs(trans, n, nrhs, dl, d, du, du2, ipiv, b, ldb);
}

template <uni20::LapackReal Scalar>
Scalar gtcon(char norm, blas_int n, Scalar* dl, Scalar* d, Scalar* du, Scalar* du2, blas_int const* ipiv,
             Scalar matrix_norm, Scalar* work, blas_int* iwork)
{
  return uni20::lapack::gtcon(norm, n, dl, d, du, du2, ipiv, matrix_norm, work, iwork);
}

template <uni20::LapackReal Scalar>
void gtrfs(char trans, blas_int n, blas_int nrhs, Scalar* dl, Scalar* d, Scalar* du, Scalar* dlf, Scalar* df,
           Scalar* duf, Scalar* du2, blas_int const* ipiv, Scalar* b, blas_int ldb, Scalar* x, blas_int ldx,
           Scalar* forward_error, Scalar* backward_error, Scalar* work, blas_int* iwork)
{
  uni20::lapack::gtrfs(trans, n, nrhs, dl, d, du, dlf, df, duf, du2, ipiv, b, ldb, x, ldx, forward_error,
                       backward_error, work, iwork);
}

template <uni20::LapackReal Scalar>
bool gtsvx(char fact, char trans, blas_int n, blas_int nrhs, Scalar* dl, Scalar* d, Scalar* du, Scalar* dlf, Scalar* df,
           Scalar* duf, Scalar* du2, blas_int* ipiv, Scalar* b, blas_int ldb, Scalar* x, blas_int ldx, Scalar& rcond,
           Scalar* forward_error, Scalar* backward_error, Scalar* work, blas_int* iwork)
{
  return uni20::lapack::gtsvx(fact, trans, n, nrhs, dl, d, du, dlf, df, duf, du2, ipiv, b, ldb, x, ldx, rcond,
                              forward_error, backward_error, work, iwork);
}

template <uni20::LapackReal Scalar>
void gesv(blas_int n, blas_int nrhs, Scalar* a, blas_int lda, blas_int* ipiv, Scalar* b, blas_int ldb)
{
  uni20::lapack::gesv(n, nrhs, a, lda, ipiv, b, ldb);
}

template <uni20::LapackReal Scalar>
bool gesvx(char fact, char trans, blas_int n, blas_int nrhs, Scalar* a, blas_int lda, Scalar* af, blas_int ldaf,
           blas_int* ipiv, char& equed, Scalar* r, Scalar* c, Scalar* b, blas_int ldb, Scalar* x, blas_int ldx,
           Scalar& rcond, Scalar* ferr, Scalar* berr, Scalar* work, blas_int* iwork)
{
  return uni20::lapack::gesvx(fact, trans, n, nrhs, a, lda, af, ldaf, ipiv, equed, r, c, b, ldb, x, ldx, rcond, ferr,
                              berr, work, iwork);
}

template <uni20::LapackReal Scalar>
void geequ(blas_int m, blas_int n, Scalar* a, blas_int lda, Scalar* row_scale, Scalar* column_scale,
           Scalar& row_condition, Scalar& column_condition, Scalar& max_abs)
{
  uni20::lapack::geequ(m, n, a, lda, row_scale, column_scale, row_condition, column_condition, max_abs);
}

template <uni20::LapackReal Scalar> void getrf(blas_int m, blas_int n, Scalar* a, blas_int lda, blas_int* ipiv)
{
  uni20::lapack::getrf(m, n, a, lda, ipiv);
}

template <uni20::LapackReal Scalar>
void getrs(char trans, blas_int n, blas_int nrhs, Scalar* a, blas_int lda, blas_int const* ipiv, Scalar* b,
           blas_int ldb)
{
  uni20::lapack::getrs(trans, n, nrhs, a, lda, ipiv, b, ldb);
}

template <uni20::LapackReal Scalar>
void gerfs(char trans, blas_int n, blas_int nrhs, Scalar* a, blas_int lda, Scalar* factors, blas_int factor_lda,
           blas_int const* ipiv, Scalar* b, blas_int ldb, Scalar* x, blas_int ldx, Scalar* forward_error,
           Scalar* backward_error, Scalar* work, blas_int* iwork)
{
  uni20::lapack::gerfs(trans, n, nrhs, a, lda, factors, factor_lda, ipiv, b, ldb, x, ldx, forward_error, backward_error,
                       work, iwork);
}

template <uni20::LapackReal Scalar>
void getri(blas_int n, Scalar* a, blas_int lda, blas_int* ipiv, Scalar* work, blas_int lwork)
{
  uni20::lapack::getri(n, a, lda, ipiv, work, lwork);
}

template <uni20::LapackReal Scalar>
Scalar gecon(char norm, blas_int n, Scalar* a, blas_int lda, Scalar anorm, Scalar* work, blas_int* iwork)
{
  return uni20::lapack::gecon(norm, n, a, lda, anorm, work, iwork);
}

template <uni20::LapackReal Scalar>
void gels(char trans, blas_int m, blas_int n, blas_int nrhs, Scalar* a, blas_int lda, Scalar* b, blas_int ldb,
          Scalar* work, blas_int lwork)
{
  uni20::lapack::gels(trans, m, n, nrhs, a, lda, b, ldb, work, lwork);
}

template <uni20::LapackReal Scalar>
void gelss(blas_int m, blas_int n, blas_int nrhs, Scalar* a, blas_int lda, Scalar* b, blas_int ldb, Scalar* s,
           Scalar rcond, blas_int& rank, Scalar* work, blas_int lwork)
{
  uni20::lapack::gelss(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, lwork);
}

template <uni20::LapackReal Scalar>
void gelsd(blas_int m, blas_int n, blas_int nrhs, Scalar* a, blas_int lda, Scalar* b, blas_int ldb, Scalar* s,
           Scalar rcond, blas_int& rank, Scalar* work, blas_int lwork, blas_int* iwork)
{
  uni20::lapack::gelsd(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, lwork, iwork);
}

template <uni20::LapackReal Scalar>
void gelsy(blas_int m, blas_int n, blas_int nrhs, Scalar* a, blas_int lda, Scalar* b, blas_int ldb, blas_int* jpvt,
           Scalar rcond, blas_int& rank, Scalar* work, blas_int lwork)
{
  uni20::lapack::gelsy(m, n, nrhs, a, lda, b, ldb, jpvt, rcond, rank, work, lwork);
}

template <uni20::LapackReal Scalar>
void geqrf(blas_int m, blas_int n, Scalar* a, blas_int lda, Scalar* tau, Scalar* work, blas_int lwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sgeqrf_(&m, &n, a, &lda, tau, work, &lwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dgeqrf_(&m, &n, a, &lda, tau, work, &lwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    mplapackint mplapack_m = static_cast<mplapackint>(m);
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_info = 0;
    Rgeqrf(mplapack_m, mplapack_n, a, mplapack_lda, tau, work, mplapack_lwork, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "geqrf is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK geqrf received an invalid argument");
  }
}

template <uni20::LapackReal Scalar>
void gelqf(blas_int m, blas_int n, Scalar* a, blas_int lda, Scalar* tau, Scalar* work, blas_int lwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sgelqf_(&m, &n, a, &lda, tau, work, &lwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dgelqf_(&m, &n, a, &lda, tau, work, &lwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    mplapackint mplapack_m = static_cast<mplapackint>(m);
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_info = 0;
    Rgelqf(mplapack_m, mplapack_n, a, mplapack_lda, tau, work, mplapack_lwork, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "gelqf is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK gelqf received an invalid argument");
  }
}

template <uni20::LapackReal Scalar>
void geqlf(blas_int m, blas_int n, Scalar* a, blas_int lda, Scalar* tau, Scalar* work, blas_int lwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sgeqlf_(&m, &n, a, &lda, tau, work, &lwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dgeqlf_(&m, &n, a, &lda, tau, work, &lwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    mplapackint mplapack_m = static_cast<mplapackint>(m);
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_info = 0;
    Rgeqlf(mplapack_m, mplapack_n, a, mplapack_lda, tau, work, mplapack_lwork, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "geqlf is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK geqlf received an invalid argument");
  }
}

template <uni20::LapackReal Scalar>
void gerqf(blas_int m, blas_int n, Scalar* a, blas_int lda, Scalar* tau, Scalar* work, blas_int lwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sgerqf_(&m, &n, a, &lda, tau, work, &lwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dgerqf_(&m, &n, a, &lda, tau, work, &lwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    mplapackint mplapack_m = static_cast<mplapackint>(m);
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_info = 0;
    Rgerqf(mplapack_m, mplapack_n, a, mplapack_lda, tau, work, mplapack_lwork, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "gerqf is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK gerqf received an invalid argument");
  }
}

template <uni20::LapackReal Scalar>
void gebrd(blas_int m, blas_int n, Scalar* a, blas_int lda, Scalar* d, Scalar* e, Scalar* tauq, Scalar* taup,
           Scalar* work, blas_int lwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sgebrd_(&m, &n, a, &lda, d, e, tauq, taup, work, &lwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dgebrd_(&m, &n, a, &lda, d, e, tauq, taup, work, &lwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    mplapackint mplapack_m = static_cast<mplapackint>(m);
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_info = 0;
    Rgebrd(mplapack_m, mplapack_n, a, mplapack_lda, d, e, tauq, taup, work, mplapack_lwork, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "gebrd is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK gebrd received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK gebrd failed to reduce a matrix to bidiagonal form");
  }
}

template <uni20::LapackReal Scalar>
void gehrd(blas_int n, blas_int first, blas_int last, Scalar* a, blas_int lda, Scalar* tau, Scalar* work,
           blas_int lwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sgehrd_(&n, &first, &last, a, &lda, tau, work, &lwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dgehrd_(&n, &first, &last, a, &lda, tau, work, &lwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_first = static_cast<mplapackint>(first);
    mplapackint mplapack_last = static_cast<mplapackint>(last);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_info = 0;
    Rgehrd(mplapack_n, mplapack_first, mplapack_last, a, mplapack_lda, tau, work, mplapack_lwork, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "gehrd is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK gehrd received an invalid argument");
  }
}

template <uni20::LapackReal Scalar>
void geqp3(blas_int m, blas_int n, Scalar* a, blas_int lda, blas_int* jpvt, Scalar* tau, Scalar* work, blas_int lwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sgeqp3_(&m, &n, a, &lda, jpvt, tau, work, &lwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dgeqp3_(&m, &n, a, &lda, jpvt, tau, work, &lwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    mplapackint mplapack_m = static_cast<mplapackint>(m);
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_info = 0;
    std::vector<mplapackint> mplapack_jpvt(static_cast<std::size_t>(std::max<blas_int>(1, n)));
    for (blas_int i = 0; i < std::max<blas_int>(1, n); ++i)
    {
      mplapack_jpvt[static_cast<std::size_t>(i)] = static_cast<mplapackint>(jpvt[i]);
    }
    Rgeqp3(mplapack_m, mplapack_n, a, mplapack_lda, mplapack_jpvt.data(), tau, work, mplapack_lwork, mplapack_info);
    for (blas_int i = 0; i < std::max<blas_int>(1, n); ++i)
    {
      jpvt[i] = static_cast<blas_int>(mplapack_jpvt[static_cast<std::size_t>(i)]);
    }
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "geqp3 is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK geqp3 received an invalid argument");
  }
}

template <uni20::LapackReal Scalar>
void sytrd(char uplo, blas_int n, Scalar* a, blas_int lda, Scalar* d, Scalar* e, Scalar* tau, Scalar* work,
           blas_int lwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    ssytrd_(&uplo, &n, a, &lda, d, e, tau, work, &lwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dsytrd_(&uplo, &n, a, &lda, d, e, tau, work, &lwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char uplo_string[2] = {uplo, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_info = 0;
    Rsytrd(uplo_string, mplapack_n, a, mplapack_lda, d, e, tau, work, mplapack_lwork, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "sytrd is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK sytrd received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK sytrd failed to reduce a symmetric matrix");
  }
}

template <uni20::LapackReal Scalar>
void orgqr(blas_int m, blas_int n, blas_int k, Scalar* a, blas_int lda, Scalar* tau, Scalar* work, blas_int lwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sorgqr_(&m, &n, &k, a, &lda, tau, work, &lwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dorgqr_(&m, &n, &k, a, &lda, tau, work, &lwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    mplapackint mplapack_m = static_cast<mplapackint>(m);
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_k = static_cast<mplapackint>(k);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_info = 0;
    Rorgqr(mplapack_m, mplapack_n, mplapack_k, a, mplapack_lda, tau, work, mplapack_lwork, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "orgqr is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK orgqr received an invalid argument");
  }
}

template <uni20::LapackReal Scalar>
void orglq(blas_int m, blas_int n, blas_int k, Scalar* a, blas_int lda, Scalar* tau, Scalar* work, blas_int lwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sorglq_(&m, &n, &k, a, &lda, tau, work, &lwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dorglq_(&m, &n, &k, a, &lda, tau, work, &lwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    mplapackint mplapack_m = static_cast<mplapackint>(m);
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_k = static_cast<mplapackint>(k);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_info = 0;
    Rorglq(mplapack_m, mplapack_n, mplapack_k, a, mplapack_lda, tau, work, mplapack_lwork, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "orglq is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK orglq received an invalid argument");
  }
}

template <uni20::LapackReal Scalar>
void orgql(blas_int m, blas_int n, blas_int k, Scalar* a, blas_int lda, Scalar* tau, Scalar* work, blas_int lwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sorgql_(&m, &n, &k, a, &lda, tau, work, &lwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dorgql_(&m, &n, &k, a, &lda, tau, work, &lwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    mplapackint mplapack_m = static_cast<mplapackint>(m);
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_k = static_cast<mplapackint>(k);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_info = 0;
    Rorgql(mplapack_m, mplapack_n, mplapack_k, a, mplapack_lda, tau, work, mplapack_lwork, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "orgql is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK orgql received an invalid argument");
  }
}

template <uni20::LapackReal Scalar>
void orgrq(blas_int m, blas_int n, blas_int k, Scalar* a, blas_int lda, Scalar* tau, Scalar* work, blas_int lwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sorgrq_(&m, &n, &k, a, &lda, tau, work, &lwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dorgrq_(&m, &n, &k, a, &lda, tau, work, &lwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    mplapackint mplapack_m = static_cast<mplapackint>(m);
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_k = static_cast<mplapackint>(k);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_info = 0;
    Rorgrq(mplapack_m, mplapack_n, mplapack_k, a, mplapack_lda, tau, work, mplapack_lwork, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "orgrq is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK orgrq received an invalid argument");
  }
}

template <uni20::LapackReal Scalar>
void orgbr(char vect, blas_int m, blas_int n, blas_int k, Scalar* a, blas_int lda, Scalar* tau, Scalar* work,
           blas_int lwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sorgbr_(&vect, &m, &n, &k, a, &lda, tau, work, &lwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dorgbr_(&vect, &m, &n, &k, a, &lda, tau, work, &lwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char vect_string[2] = {vect, '\0'};
    mplapackint mplapack_m = static_cast<mplapackint>(m);
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_k = static_cast<mplapackint>(k);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_info = 0;
    Rorgbr(vect_string, mplapack_m, mplapack_n, mplapack_k, a, mplapack_lda, tau, work, mplapack_lwork, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "orgbr is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK orgbr received an invalid argument");
  }
}

template <uni20::LapackReal Scalar>
void orghr(blas_int n, blas_int first, blas_int last, Scalar* a, blas_int lda, Scalar const* tau, Scalar* work,
           blas_int lwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sorghr_(&n, &first, &last, a, &lda, tau, work, &lwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dorghr_(&n, &first, &last, a, &lda, tau, work, &lwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_first = static_cast<mplapackint>(first);
    mplapackint mplapack_last = static_cast<mplapackint>(last);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_info = 0;
    Rorghr(mplapack_n, mplapack_first, mplapack_last, a, mplapack_lda, const_cast<Scalar*>(tau), work, mplapack_lwork,
           mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "orghr is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK orghr received an invalid argument");
  }
}

template <uni20::LapackReal Scalar>
void orgtr(char uplo, blas_int n, Scalar* a, blas_int lda, Scalar const* tau, Scalar* work, blas_int lwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sorgtr_(&uplo, &n, a, &lda, tau, work, &lwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dorgtr_(&uplo, &n, a, &lda, tau, work, &lwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char uplo_string[2] = {uplo, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_info = 0;
    Rorgtr(uplo_string, mplapack_n, a, mplapack_lda, const_cast<Scalar*>(tau), work, mplapack_lwork, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "orgtr is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK orgtr received an invalid argument");
  }
}

template <uni20::LapackReal Scalar>
void ormhr(char side, char trans, blas_int m, blas_int n, blas_int first, blas_int last, Scalar* a, blas_int lda,
           Scalar const* tau, Scalar* c, blas_int ldc, Scalar* work, blas_int lwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sormhr_(&side, &trans, &m, &n, &first, &last, a, &lda, tau, c, &ldc, work, &lwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dormhr_(&side, &trans, &m, &n, &first, &last, a, &lda, tau, c, &ldc, work, &lwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char side_string[2] = {side, '\0'};
    char trans_string[2] = {trans, '\0'};
    mplapackint mplapack_m = static_cast<mplapackint>(m);
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_first = static_cast<mplapackint>(first);
    mplapackint mplapack_last = static_cast<mplapackint>(last);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_ldc = static_cast<mplapackint>(ldc);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_info = 0;
    Rormhr(side_string, trans_string, mplapack_m, mplapack_n, mplapack_first, mplapack_last, a, mplapack_lda,
           const_cast<Scalar*>(tau), c, mplapack_ldc, work, mplapack_lwork, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "ormhr is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK ormhr received an invalid argument");
  }
}

template <uni20::LapackReal Scalar>
void ormtr(char side, char uplo, char trans, blas_int m, blas_int n, Scalar* a, blas_int lda, Scalar const* tau,
           Scalar* c, blas_int ldc, Scalar* work, blas_int lwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sormtr_(&side, &uplo, &trans, &m, &n, a, &lda, tau, c, &ldc, work, &lwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dormtr_(&side, &uplo, &trans, &m, &n, a, &lda, tau, c, &ldc, work, &lwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char side_string[2] = {side, '\0'};
    char uplo_string[2] = {uplo, '\0'};
    char trans_string[2] = {trans, '\0'};
    mplapackint mplapack_m = static_cast<mplapackint>(m);
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_ldc = static_cast<mplapackint>(ldc);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_info = 0;
    Rormtr(side_string, uplo_string, trans_string, mplapack_m, mplapack_n, a, mplapack_lda, const_cast<Scalar*>(tau), c,
           mplapack_ldc, work, mplapack_lwork, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "ormtr is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK ormtr received an invalid argument");
  }
}

template <uni20::LapackReal Scalar>
void ormbr(char vect, char side, char trans, blas_int m, blas_int n, blas_int k, Scalar* a, blas_int lda, Scalar* tau,
           Scalar* c, blas_int ldc, Scalar* work, blas_int lwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sormbr_(&vect, &side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dormbr_(&vect, &side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char vect_string[2] = {vect, '\0'};
    char side_string[2] = {side, '\0'};
    char trans_string[2] = {trans, '\0'};
    mplapackint mplapack_m = static_cast<mplapackint>(m);
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_k = static_cast<mplapackint>(k);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_ldc = static_cast<mplapackint>(ldc);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_info = 0;
    Rormbr(vect_string, side_string, trans_string, mplapack_m, mplapack_n, mplapack_k, a, mplapack_lda, tau, c,
           mplapack_ldc, work, mplapack_lwork, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "ormbr is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK ormbr received an invalid argument");
  }
}

template <uni20::LapackReal Scalar>
void ormqr(char side, char trans, blas_int m, blas_int n, blas_int k, Scalar* a, blas_int lda, Scalar* tau, Scalar* c,
           blas_int ldc, Scalar* work, blas_int lwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sormqr_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dormqr_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char side_string[2] = {side, '\0'};
    char trans_string[2] = {trans, '\0'};
    mplapackint mplapack_m = static_cast<mplapackint>(m);
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_k = static_cast<mplapackint>(k);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_ldc = static_cast<mplapackint>(ldc);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_info = 0;
    Rormqr(side_string, trans_string, mplapack_m, mplapack_n, mplapack_k, a, mplapack_lda, tau, c, mplapack_ldc, work,
           mplapack_lwork, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "ormqr is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK ormqr received an invalid argument");
  }
}

template <uni20::LapackReal Scalar>
void ormlq(char side, char trans, blas_int m, blas_int n, blas_int k, Scalar* a, blas_int lda, Scalar* tau, Scalar* c,
           blas_int ldc, Scalar* work, blas_int lwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sormlq_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dormlq_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char side_string[2] = {side, '\0'};
    char trans_string[2] = {trans, '\0'};
    mplapackint mplapack_m = static_cast<mplapackint>(m);
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_k = static_cast<mplapackint>(k);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_ldc = static_cast<mplapackint>(ldc);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_info = 0;
    Rormlq(side_string, trans_string, mplapack_m, mplapack_n, mplapack_k, a, mplapack_lda, tau, c, mplapack_ldc, work,
           mplapack_lwork, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "ormlq is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK ormlq received an invalid argument");
  }
}

template <uni20::LapackReal Scalar>
void ormql(char side, char trans, blas_int m, blas_int n, blas_int k, Scalar* a, blas_int lda, Scalar* tau, Scalar* c,
           blas_int ldc, Scalar* work, blas_int lwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sormql_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dormql_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char side_string[2] = {side, '\0'};
    char trans_string[2] = {trans, '\0'};
    mplapackint mplapack_m = static_cast<mplapackint>(m);
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_k = static_cast<mplapackint>(k);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_ldc = static_cast<mplapackint>(ldc);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_info = 0;
    Rormql(side_string, trans_string, mplapack_m, mplapack_n, mplapack_k, a, mplapack_lda, tau, c, mplapack_ldc, work,
           mplapack_lwork, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "ormql is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK ormql received an invalid argument");
  }
}

template <uni20::LapackReal Scalar>
void ormrq(char side, char trans, blas_int m, blas_int n, blas_int k, Scalar* a, blas_int lda, Scalar* tau, Scalar* c,
           blas_int ldc, Scalar* work, blas_int lwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sormrq_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dormrq_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char side_string[2] = {side, '\0'};
    char trans_string[2] = {trans, '\0'};
    mplapackint mplapack_m = static_cast<mplapackint>(m);
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_k = static_cast<mplapackint>(k);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_ldc = static_cast<mplapackint>(ldc);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_info = 0;
    Rormrq(side_string, trans_string, mplapack_m, mplapack_n, mplapack_k, a, mplapack_lda, tau, c, mplapack_ldc, work,
           mplapack_lwork, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "ormrq is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK ormrq received an invalid argument");
  }
}

template <uni20::LapackReal Scalar> void potrf(char uplo, blas_int n, Scalar* a, blas_int lda)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    spotrf_(&uplo, &n, a, &lda, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dpotrf_(&uplo, &n, a, &lda, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_info = 0;
    char uplo_string[2] = {uplo, '\0'};
    Rpotrf(uplo_string, mplapack_n, a, mplapack_lda, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "potrf is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK potrf received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK potrf found a matrix that is not positive definite");
  }
}

template <uni20::LapackReal Scalar>
void pbsv(char uplo, blas_int n, blas_int kd, blas_int nrhs, Scalar* ab, blas_int ldab, Scalar* b, blas_int ldb)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    spbsv_(&uplo, &n, &kd, &nrhs, ab, &ldab, b, &ldb, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dpbsv_(&uplo, &n, &kd, &nrhs, ab, &ldab, b, &ldb, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char uplo_string[2] = {uplo, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_kd = static_cast<mplapackint>(kd);
    mplapackint mplapack_nrhs = static_cast<mplapackint>(nrhs);
    mplapackint mplapack_ldab = static_cast<mplapackint>(ldab);
    mplapackint mplapack_ldb = static_cast<mplapackint>(ldb);
    mplapackint mplapack_info = 0;
    Rpbsv(uplo_string, mplapack_n, mplapack_kd, mplapack_nrhs, ab, mplapack_ldab, b, mplapack_ldb, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "pbsv is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK pbsv received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK pbsv found a matrix that is not positive definite");
  }
}

template <uni20::LapackReal Scalar> void pbtrf(char uplo, blas_int n, blas_int kd, Scalar* ab, blas_int ldab)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    spbtrf_(&uplo, &n, &kd, ab, &ldab, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dpbtrf_(&uplo, &n, &kd, ab, &ldab, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char uplo_string[2] = {uplo, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_kd = static_cast<mplapackint>(kd);
    mplapackint mplapack_ldab = static_cast<mplapackint>(ldab);
    mplapackint mplapack_info = 0;
    Rpbtrf(uplo_string, mplapack_n, mplapack_kd, ab, mplapack_ldab, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "pbtrf is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK pbtrf received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK pbtrf found a matrix that is not positive definite");
  }
}

template <uni20::LapackReal Scalar>
void pbtrs(char uplo, blas_int n, blas_int kd, blas_int nrhs, Scalar* ab, blas_int ldab, Scalar* b, blas_int ldb)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    spbtrs_(&uplo, &n, &kd, &nrhs, ab, &ldab, b, &ldb, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dpbtrs_(&uplo, &n, &kd, &nrhs, ab, &ldab, b, &ldb, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char uplo_string[2] = {uplo, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_kd = static_cast<mplapackint>(kd);
    mplapackint mplapack_nrhs = static_cast<mplapackint>(nrhs);
    mplapackint mplapack_ldab = static_cast<mplapackint>(ldab);
    mplapackint mplapack_ldb = static_cast<mplapackint>(ldb);
    mplapackint mplapack_info = 0;
    Rpbtrs(uplo_string, mplapack_n, mplapack_kd, mplapack_nrhs, ab, mplapack_ldab, b, mplapack_ldb, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "pbtrs is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK pbtrs received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK pbtrs failed to solve the factored system");
  }
}

template <uni20::LapackReal Scalar>
Scalar pbcon(char uplo, blas_int n, blas_int kd, Scalar* ab, blas_int ldab, Scalar matrix_norm, Scalar* work,
             blas_int* iwork)
{
  Scalar rcond{};
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    spbcon_(&uplo, &n, &kd, ab, &ldab, &matrix_norm, &rcond, work, iwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dpbcon_(&uplo, &n, &kd, ab, &ldab, &matrix_norm, &rcond, work, iwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char uplo_string[2] = {uplo, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_kd = static_cast<mplapackint>(kd);
    mplapackint mplapack_ldab = static_cast<mplapackint>(ldab);
    mplapackint mplapack_info = 0;
    std::vector<mplapackint> mplapack_iwork(static_cast<std::size_t>(std::max<blas_int>(1, n)));
    Rpbcon(uplo_string, mplapack_n, mplapack_kd, ab, mplapack_ldab, matrix_norm, rcond, work, mplapack_iwork.data(),
           mplapack_info);
    for (blas_int i = 0; i < std::max<blas_int>(1, n); ++i)
    {
      iwork[i] = static_cast<blas_int>(mplapack_iwork[static_cast<std::size_t>(i)]);
    }
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "pbcon is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK pbcon received an invalid argument");
  }
  return rcond;
}

template <uni20::LapackReal Scalar>
void pbrfs(char uplo, blas_int n, blas_int kd, blas_int nrhs, Scalar* ab, blas_int ldab, Scalar* afb, blas_int ldafb,
           Scalar* b, blas_int ldb, Scalar* x, blas_int ldx, Scalar* forward_error, Scalar* backward_error,
           Scalar* work, blas_int* iwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    spbrfs_(&uplo, &n, &kd, &nrhs, ab, &ldab, afb, &ldafb, b, &ldb, x, &ldx, forward_error, backward_error, work, iwork,
            &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dpbrfs_(&uplo, &n, &kd, &nrhs, ab, &ldab, afb, &ldafb, b, &ldb, x, &ldx, forward_error, backward_error, work, iwork,
            &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char uplo_string[2] = {uplo, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_kd = static_cast<mplapackint>(kd);
    mplapackint mplapack_nrhs = static_cast<mplapackint>(nrhs);
    mplapackint mplapack_ldab = static_cast<mplapackint>(ldab);
    mplapackint mplapack_ldafb = static_cast<mplapackint>(ldafb);
    mplapackint mplapack_ldb = static_cast<mplapackint>(ldb);
    mplapackint mplapack_ldx = static_cast<mplapackint>(ldx);
    mplapackint mplapack_info = 0;
    std::vector<mplapackint> mplapack_iwork(static_cast<std::size_t>(std::max<blas_int>(1, n)));
    Rpbrfs(uplo_string, mplapack_n, mplapack_kd, mplapack_nrhs, ab, mplapack_ldab, afb, mplapack_ldafb, b, mplapack_ldb,
           x, mplapack_ldx, forward_error, backward_error, work, mplapack_iwork.data(), mplapack_info);
    for (blas_int i = 0; i < std::max<blas_int>(1, n); ++i)
    {
      iwork[i] = static_cast<blas_int>(mplapack_iwork[static_cast<std::size_t>(i)]);
    }
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "pbrfs is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK pbrfs received an invalid argument");
  }
}

template <uni20::LapackReal Scalar>
bool pbsvx(char fact, char uplo, blas_int n, blas_int kd, blas_int nrhs, Scalar* ab, blas_int ldab, Scalar* afb,
           blas_int ldafb, char& equed, Scalar* scale, Scalar* b, blas_int ldb, Scalar* x, blas_int ldx, Scalar& rcond,
           Scalar* forward_error, Scalar* backward_error, Scalar* work, blas_int* iwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    spbsvx_(&fact, &uplo, &n, &kd, &nrhs, ab, &ldab, afb, &ldafb, &equed, scale, b, &ldb, x, &ldx, &rcond,
            forward_error, backward_error, work, iwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dpbsvx_(&fact, &uplo, &n, &kd, &nrhs, ab, &ldab, afb, &ldafb, &equed, scale, b, &ldb, x, &ldx, &rcond,
            forward_error, backward_error, work, iwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char fact_string[2] = {fact, '\0'};
    char uplo_string[2] = {uplo, '\0'};
    char equed_string[2] = {equed, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_kd = static_cast<mplapackint>(kd);
    mplapackint mplapack_nrhs = static_cast<mplapackint>(nrhs);
    mplapackint mplapack_ldab = static_cast<mplapackint>(ldab);
    mplapackint mplapack_ldafb = static_cast<mplapackint>(ldafb);
    mplapackint mplapack_ldb = static_cast<mplapackint>(ldb);
    mplapackint mplapack_ldx = static_cast<mplapackint>(ldx);
    mplapackint mplapack_info = 0;
    std::vector<mplapackint> mplapack_iwork(static_cast<std::size_t>(std::max<blas_int>(1, n)));
    Rpbsvx(fact_string, uplo_string, mplapack_n, mplapack_kd, mplapack_nrhs, ab, mplapack_ldab, afb, mplapack_ldafb,
           equed_string, scale, b, mplapack_ldb, x, mplapack_ldx, rcond, forward_error, backward_error, work,
           mplapack_iwork.data(), mplapack_info);
    equed = equed_string[0];
    for (blas_int i = 0; i < std::max<blas_int>(1, n); ++i)
    {
      iwork[i] = static_cast<blas_int>(mplapack_iwork[static_cast<std::size_t>(i)]);
    }
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "pbsvx is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK pbsvx received an invalid argument");
  }
  if (info > 0 && info <= n)
  {
    throw std::runtime_error("LAPACK pbsvx found a matrix that is not positive definite");
  }
  if (info > n + 1)
  {
    throw std::runtime_error("LAPACK pbsvx failed unexpectedly");
  }
  return info == n + 1;
}

template <uni20::LapackReal Scalar> void ptsv(blas_int n, blas_int nrhs, Scalar* d, Scalar* e, Scalar* b, blas_int ldb)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sptsv_(&n, &nrhs, d, e, b, &ldb, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dptsv_(&n, &nrhs, d, e, b, &ldb, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_nrhs = static_cast<mplapackint>(nrhs);
    mplapackint mplapack_ldb = static_cast<mplapackint>(ldb);
    mplapackint mplapack_info = 0;
    Rptsv(mplapack_n, mplapack_nrhs, d, e, b, mplapack_ldb, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "ptsv is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK ptsv received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK ptsv found a matrix that is not positive definite");
  }
}

template <uni20::LapackReal Scalar> void pttrf(blas_int n, Scalar* d, Scalar* e)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    spttrf_(&n, d, e, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dpttrf_(&n, d, e, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_info = 0;
    Rpttrf(mplapack_n, d, e, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "pttrf is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK pttrf received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK pttrf found a matrix that is not positive definite");
  }
}

template <uni20::LapackReal Scalar> void pttrs(blas_int n, blas_int nrhs, Scalar* d, Scalar* e, Scalar* b, blas_int ldb)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    spttrs_(&n, &nrhs, d, e, b, &ldb, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dpttrs_(&n, &nrhs, d, e, b, &ldb, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_nrhs = static_cast<mplapackint>(nrhs);
    mplapackint mplapack_ldb = static_cast<mplapackint>(ldb);
    mplapackint mplapack_info = 0;
    Rpttrs(mplapack_n, mplapack_nrhs, d, e, b, mplapack_ldb, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "pttrs is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK pttrs received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK pttrs failed to solve the factored system");
  }
}

template <uni20::LapackReal Scalar> Scalar ptcon(blas_int n, Scalar* d, Scalar* e, Scalar matrix_norm, Scalar* work)
{
  Scalar rcond{};
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sptcon_(&n, d, e, &matrix_norm, &rcond, work, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dptcon_(&n, d, e, &matrix_norm, &rcond, work, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_info = 0;
    Rptcon(mplapack_n, d, e, matrix_norm, rcond, work, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "ptcon is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK ptcon received an invalid argument");
  }
  return rcond;
}

template <uni20::LapackReal Scalar>
void ptrfs(blas_int n, blas_int nrhs, Scalar* d, Scalar* e, Scalar* df, Scalar* ef, Scalar* b, blas_int ldb, Scalar* x,
           blas_int ldx, Scalar* forward_error, Scalar* backward_error, Scalar* work)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sptrfs_(&n, &nrhs, d, e, df, ef, b, &ldb, x, &ldx, forward_error, backward_error, work, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dptrfs_(&n, &nrhs, d, e, df, ef, b, &ldb, x, &ldx, forward_error, backward_error, work, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_nrhs = static_cast<mplapackint>(nrhs);
    mplapackint mplapack_ldb = static_cast<mplapackint>(ldb);
    mplapackint mplapack_ldx = static_cast<mplapackint>(ldx);
    mplapackint mplapack_info = 0;
    Rptrfs(mplapack_n, mplapack_nrhs, d, e, df, ef, b, mplapack_ldb, x, mplapack_ldx, forward_error, backward_error,
           work, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "ptrfs is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK ptrfs received an invalid argument");
  }
}

template <uni20::LapackReal Scalar>
bool ptsvx(char fact, blas_int n, blas_int nrhs, Scalar* d, Scalar* e, Scalar* df, Scalar* ef, Scalar* b, blas_int ldb,
           Scalar* x, blas_int ldx, Scalar& rcond, Scalar* forward_error, Scalar* backward_error, Scalar* work)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sptsvx_(&fact, &n, &nrhs, d, e, df, ef, b, &ldb, x, &ldx, &rcond, forward_error, backward_error, work, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dptsvx_(&fact, &n, &nrhs, d, e, df, ef, b, &ldb, x, &ldx, &rcond, forward_error, backward_error, work, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char fact_string[2] = {fact, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_nrhs = static_cast<mplapackint>(nrhs);
    mplapackint mplapack_ldb = static_cast<mplapackint>(ldb);
    mplapackint mplapack_ldx = static_cast<mplapackint>(ldx);
    mplapackint mplapack_info = 0;
    Rptsvx(fact_string, mplapack_n, mplapack_nrhs, d, e, df, ef, b, mplapack_ldb, x, mplapack_ldx, rcond, forward_error,
           backward_error, work, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "ptsvx is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK ptsvx received an invalid argument");
  }
  if (info > 0 && info <= n)
  {
    throw std::runtime_error("LAPACK ptsvx found a matrix that is not positive definite");
  }
  if (info > n + 1)
  {
    throw std::runtime_error("LAPACK ptsvx failed unexpectedly");
  }
  return info == n + 1;
}

template <uni20::LapackReal Scalar>
bool pstrf(char uplo, blas_int n, Scalar* a, blas_int lda, blas_int* pivots, blas_int& rank, Scalar tolerance,
           Scalar* work)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    spstrf_(&uplo, &n, a, &lda, pivots, &rank, &tolerance, work, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dpstrf_(&uplo, &n, a, &lda, pivots, &rank, &tolerance, work, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char uplo_string[2] = {uplo, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_rank = static_cast<mplapackint>(rank);
    mplapackint mplapack_info = 0;
    std::vector<mplapackint> mplapack_pivots(static_cast<std::size_t>(std::max<blas_int>(1, n)));
    for (blas_int i = 0; i < std::max<blas_int>(1, n); ++i)
    {
      mplapack_pivots[static_cast<std::size_t>(i)] = static_cast<mplapackint>(pivots[i]);
    }
    Rpstrf(uplo_string, mplapack_n, a, mplapack_lda, mplapack_pivots.data(), mplapack_rank, tolerance, work,
           mplapack_info);
    rank = static_cast<blas_int>(mplapack_rank);
    for (blas_int i = 0; i < std::max<blas_int>(1, n); ++i)
    {
      pivots[i] = static_cast<blas_int>(mplapack_pivots[static_cast<std::size_t>(i)]);
    }
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "pstrf is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK pstrf received an invalid argument");
  }
  if (info > 1)
  {
    throw std::runtime_error("LAPACK pstrf failed unexpectedly");
  }
  return info == 1;
}

template <uni20::LapackReal Scalar> void potri(char uplo, blas_int n, Scalar* a, blas_int lda)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    spotri_(&uplo, &n, a, &lda, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dpotri_(&uplo, &n, a, &lda, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_info = 0;
    char uplo_string[2] = {uplo, '\0'};
    Rpotri(uplo_string, mplapack_n, a, mplapack_lda, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "potri is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK potri received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK potri found a singular Cholesky factor");
  }
}

template <uni20::LapackReal Scalar>
void potrs(char uplo, blas_int n, blas_int nrhs, Scalar* a, blas_int lda, Scalar* b, blas_int ldb)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    spotrs_(&uplo, &n, &nrhs, a, &lda, b, &ldb, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dpotrs_(&uplo, &n, &nrhs, a, &lda, b, &ldb, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_nrhs = static_cast<mplapackint>(nrhs);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_ldb = static_cast<mplapackint>(ldb);
    mplapackint mplapack_info = 0;
    char uplo_string[2] = {uplo, '\0'};
    Rpotrs(uplo_string, mplapack_n, mplapack_nrhs, a, mplapack_lda, b, mplapack_ldb, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "potrs is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK potrs received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK potrs failed to solve the factored system");
  }
}

template <uni20::LapackReal Scalar>
void porfs(char uplo, blas_int n, blas_int nrhs, Scalar* a, blas_int lda, Scalar* factors, blas_int factor_lda,
           Scalar* b, blas_int ldb, Scalar* x, blas_int ldx, Scalar* forward_error, Scalar* backward_error,
           Scalar* work, blas_int* iwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sporfs_(&uplo, &n, &nrhs, a, &lda, factors, &factor_lda, b, &ldb, x, &ldx, forward_error, backward_error, work,
            iwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dporfs_(&uplo, &n, &nrhs, a, &lda, factors, &factor_lda, b, &ldb, x, &ldx, forward_error, backward_error, work,
            iwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char uplo_string[2] = {uplo, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_nrhs = static_cast<mplapackint>(nrhs);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_factor_lda = static_cast<mplapackint>(factor_lda);
    mplapackint mplapack_ldb = static_cast<mplapackint>(ldb);
    mplapackint mplapack_ldx = static_cast<mplapackint>(ldx);
    mplapackint mplapack_info = 0;
    std::vector<mplapackint> mplapack_iwork(static_cast<std::size_t>(std::max<blas_int>(1, n)));
    Rporfs(uplo_string, mplapack_n, mplapack_nrhs, a, mplapack_lda, factors, mplapack_factor_lda, b, mplapack_ldb, x,
           mplapack_ldx, forward_error, backward_error, work, mplapack_iwork.data(), mplapack_info);
    for (blas_int i = 0; i < std::max<blas_int>(1, n); ++i)
    {
      iwork[i] = static_cast<blas_int>(mplapack_iwork[static_cast<std::size_t>(i)]);
    }
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "porfs is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK porfs received an invalid argument");
  }
}

template <uni20::LapackReal Scalar>
bool posvx(char fact, char uplo, blas_int n, blas_int nrhs, Scalar* a, blas_int lda, Scalar* af, blas_int ldaf,
           char& equed, Scalar* scale, Scalar* b, blas_int ldb, Scalar* x, blas_int ldx, Scalar& rcond, Scalar* ferr,
           Scalar* berr, Scalar* work, blas_int* iwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sposvx_(&fact, &uplo, &n, &nrhs, a, &lda, af, &ldaf, &equed, scale, b, &ldb, x, &ldx, &rcond, ferr, berr, work,
            iwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dposvx_(&fact, &uplo, &n, &nrhs, a, &lda, af, &ldaf, &equed, scale, b, &ldb, x, &ldx, &rcond, ferr, berr, work,
            iwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char fact_string[2] = {fact, '\0'};
    char uplo_string[2] = {uplo, '\0'};
    char equed_string[2] = {equed, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_nrhs = static_cast<mplapackint>(nrhs);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_ldaf = static_cast<mplapackint>(ldaf);
    mplapackint mplapack_ldb = static_cast<mplapackint>(ldb);
    mplapackint mplapack_ldx = static_cast<mplapackint>(ldx);
    mplapackint mplapack_info = 0;
    std::vector<mplapackint> mplapack_iwork(static_cast<std::size_t>(std::max<blas_int>(1, n)));
    Rposvx(fact_string, uplo_string, mplapack_n, mplapack_nrhs, a, mplapack_lda, af, mplapack_ldaf, equed_string, scale,
           b, mplapack_ldb, x, mplapack_ldx, rcond, ferr, berr, work, mplapack_iwork.data(), mplapack_info);
    equed = equed_string[0];
    for (blas_int i = 0; i < std::max<blas_int>(1, n); ++i)
    {
      iwork[i] = static_cast<blas_int>(mplapack_iwork[static_cast<std::size_t>(i)]);
    }
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "posvx is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK posvx received an invalid argument");
  }
  if (info > 0 && info <= n)
  {
    throw std::runtime_error("LAPACK posvx found a non-positive-definite leading minor");
  }
  if (info > n + 1)
  {
    throw std::runtime_error("LAPACK posvx failed unexpectedly");
  }
  return info == n + 1;
}

template <uni20::LapackReal Scalar>
void poequ(blas_int n, Scalar* a, blas_int lda, Scalar* scale, Scalar& scale_condition, Scalar& max_abs)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    spoequ_(&n, a, &lda, scale, &scale_condition, &max_abs, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dpoequ_(&n, a, &lda, scale, &scale_condition, &max_abs, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_info = 0;
    Rpoequ(mplapack_n, a, mplapack_lda, scale, scale_condition, max_abs, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "poequ is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK poequ received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK poequ found a non-positive diagonal entry");
  }
}

template <uni20::LapackReal Scalar>
Scalar pocon(char uplo, blas_int n, Scalar* a, blas_int lda, Scalar matrix_one_norm, Scalar* work, blas_int* iwork)
{
  blas_int info = 0;
  Scalar rcond{};
  if constexpr (std::is_same_v<Scalar, float>)
  {
    spocon_(&uplo, &n, a, &lda, &matrix_one_norm, &rcond, work, iwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dpocon_(&uplo, &n, a, &lda, &matrix_one_norm, &rcond, work, iwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char uplo_string[2] = {uplo, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_info = 0;
    std::vector<mplapackint> mplapack_iwork(static_cast<std::size_t>(std::max<blas_int>(1, n)));
    Rpocon(uplo_string, mplapack_n, a, mplapack_lda, matrix_one_norm, rcond, work, mplapack_iwork.data(),
           mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "pocon is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK pocon received an invalid argument");
  }
  return rcond;
}

template <uni20::LapackReal Scalar>
void sytrf(char uplo, blas_int n, Scalar* a, blas_int lda, blas_int* ipiv, Scalar* work, blas_int lwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    ssytrf_(&uplo, &n, a, &lda, ipiv, work, &lwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dsytrf_(&uplo, &n, a, &lda, ipiv, work, &lwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_info = 0;
    std::vector<mplapackint> mplapack_ipiv(static_cast<std::size_t>(std::max<blas_int>(1, n)));
    char uplo_string[2] = {uplo, '\0'};
    Rsytrf(uplo_string, mplapack_n, a, mplapack_lda, mplapack_ipiv.data(), work, mplapack_lwork, mplapack_info);
    for (blas_int i = 0; i < std::max<blas_int>(1, n); ++i)
    {
      ipiv[i] = static_cast<blas_int>(mplapack_ipiv[static_cast<std::size_t>(i)]);
    }
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "sytrf is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK sytrf received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK sytrf found a singular diagonal block");
  }
}

template <uni20::LapackReal Scalar>
void sytrs(char uplo, blas_int n, blas_int nrhs, Scalar* a, blas_int lda, blas_int const* ipiv, Scalar* b, blas_int ldb)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    ssytrs_(&uplo, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dsytrs_(&uplo, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_nrhs = static_cast<mplapackint>(nrhs);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_ldb = static_cast<mplapackint>(ldb);
    mplapackint mplapack_info = 0;
    std::vector<mplapackint> mplapack_ipiv(static_cast<std::size_t>(std::max<blas_int>(1, n)));
    for (blas_int i = 0; i < std::max<blas_int>(1, n); ++i)
    {
      mplapack_ipiv[static_cast<std::size_t>(i)] = static_cast<mplapackint>(ipiv[i]);
    }
    char uplo_string[2] = {uplo, '\0'};
    Rsytrs(uplo_string, mplapack_n, mplapack_nrhs, a, mplapack_lda, mplapack_ipiv.data(), b, mplapack_ldb,
           mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "sytrs is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK sytrs received an invalid argument");
  }
}

template <uni20::LapackReal Scalar>
void sytri(char uplo, blas_int n, Scalar* a, blas_int lda, blas_int const* ipiv, Scalar* work)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    ssytri_(&uplo, &n, a, &lda, ipiv, work, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dsytri_(&uplo, &n, a, &lda, ipiv, work, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char uplo_string[2] = {uplo, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_info = 0;
    std::vector<mplapackint> mplapack_ipiv(static_cast<std::size_t>(std::max<blas_int>(1, n)));
    for (blas_int i = 0; i < std::max<blas_int>(1, n); ++i)
    {
      mplapack_ipiv[static_cast<std::size_t>(i)] = static_cast<mplapackint>(ipiv[i]);
    }
    Rsytri(uplo_string, mplapack_n, a, mplapack_lda, mplapack_ipiv.data(), work, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "sytri is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK sytri received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK sytri found a singular diagonal block");
  }
}

template <uni20::LapackReal Scalar>
void syrfs(char uplo, blas_int n, blas_int nrhs, Scalar* a, blas_int lda, Scalar* factors, blas_int factor_lda,
           blas_int const* ipiv, Scalar* b, blas_int ldb, Scalar* x, blas_int ldx, Scalar* forward_error,
           Scalar* backward_error, Scalar* work, blas_int* iwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    ssyrfs_(&uplo, &n, &nrhs, a, &lda, factors, &factor_lda, const_cast<blas_int*>(ipiv), b, &ldb, x, &ldx,
            forward_error, backward_error, work, iwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dsyrfs_(&uplo, &n, &nrhs, a, &lda, factors, &factor_lda, const_cast<blas_int*>(ipiv), b, &ldb, x, &ldx,
            forward_error, backward_error, work, iwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char uplo_string[2] = {uplo, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_nrhs = static_cast<mplapackint>(nrhs);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_factor_lda = static_cast<mplapackint>(factor_lda);
    mplapackint mplapack_ldb = static_cast<mplapackint>(ldb);
    mplapackint mplapack_ldx = static_cast<mplapackint>(ldx);
    mplapackint mplapack_info = 0;
    std::vector<mplapackint> mplapack_ipiv(static_cast<std::size_t>(std::max<blas_int>(1, n)));
    std::vector<mplapackint> mplapack_iwork(static_cast<std::size_t>(std::max<blas_int>(1, n)));
    for (blas_int i = 0; i < std::max<blas_int>(1, n); ++i)
    {
      mplapack_ipiv[static_cast<std::size_t>(i)] = static_cast<mplapackint>(ipiv[i]);
    }
    Rsyrfs(uplo_string, mplapack_n, mplapack_nrhs, a, mplapack_lda, factors, mplapack_factor_lda, mplapack_ipiv.data(),
           b, mplapack_ldb, x, mplapack_ldx, forward_error, backward_error, work, mplapack_iwork.data(), mplapack_info);
    for (blas_int i = 0; i < std::max<blas_int>(1, n); ++i)
    {
      iwork[i] = static_cast<blas_int>(mplapack_iwork[static_cast<std::size_t>(i)]);
    }
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "syrfs is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK syrfs received an invalid argument");
  }
}

template <uni20::LapackReal Scalar>
bool sysvx(char fact, char uplo, blas_int n, blas_int nrhs, Scalar* a, blas_int lda, Scalar* af, blas_int ldaf,
           blas_int* ipiv, Scalar* b, blas_int ldb, Scalar* x, blas_int ldx, Scalar& rcond, Scalar* ferr, Scalar* berr,
           Scalar* work, blas_int lwork, blas_int* iwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    ssysvx_(&fact, &uplo, &n, &nrhs, a, &lda, af, &ldaf, ipiv, b, &ldb, x, &ldx, &rcond, ferr, berr, work, &lwork,
            iwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dsysvx_(&fact, &uplo, &n, &nrhs, a, &lda, af, &ldaf, ipiv, b, &ldb, x, &ldx, &rcond, ferr, berr, work, &lwork,
            iwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char fact_string[2] = {fact, '\0'};
    char uplo_string[2] = {uplo, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_nrhs = static_cast<mplapackint>(nrhs);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_ldaf = static_cast<mplapackint>(ldaf);
    mplapackint mplapack_ldb = static_cast<mplapackint>(ldb);
    mplapackint mplapack_ldx = static_cast<mplapackint>(ldx);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_info = 0;
    std::vector<mplapackint> mplapack_ipiv(static_cast<std::size_t>(std::max<blas_int>(1, n)));
    std::vector<mplapackint> mplapack_iwork(static_cast<std::size_t>(std::max<blas_int>(1, n)));
    for (blas_int i = 0; i < std::max<blas_int>(1, n); ++i)
    {
      mplapack_ipiv[static_cast<std::size_t>(i)] = static_cast<mplapackint>(ipiv[i]);
    }
    Rsysvx(fact_string, uplo_string, mplapack_n, mplapack_nrhs, a, mplapack_lda, af, mplapack_ldaf,
           mplapack_ipiv.data(), b, mplapack_ldb, x, mplapack_ldx, rcond, ferr, berr, work, mplapack_lwork,
           mplapack_iwork.data(), mplapack_info);
    for (blas_int i = 0; i < std::max<blas_int>(1, n); ++i)
    {
      ipiv[i] = static_cast<blas_int>(mplapack_ipiv[static_cast<std::size_t>(i)]);
      iwork[i] = static_cast<blas_int>(mplapack_iwork[static_cast<std::size_t>(i)]);
    }
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "sysvx is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK sysvx received an invalid argument");
  }
  if (info > 0 && info <= n)
  {
    throw std::runtime_error("LAPACK sysvx found a singular diagonal block");
  }
  if (info > n + 1)
  {
    throw std::runtime_error("LAPACK sysvx failed unexpectedly");
  }
  return info == n + 1;
}

template <uni20::LapackReal Scalar>
Scalar sycon(char uplo, blas_int n, Scalar* a, blas_int lda, blas_int const* ipiv, Scalar matrix_one_norm, Scalar* work,
             blas_int* iwork)
{
  blas_int info = 0;
  Scalar rcond{};
  if constexpr (std::is_same_v<Scalar, float>)
  {
    ssycon_(&uplo, &n, a, &lda, ipiv, &matrix_one_norm, &rcond, work, iwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dsycon_(&uplo, &n, a, &lda, ipiv, &matrix_one_norm, &rcond, work, iwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char uplo_string[2] = {uplo, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_info = 0;
    std::vector<mplapackint> mplapack_ipiv(static_cast<std::size_t>(std::max<blas_int>(1, n)));
    std::vector<mplapackint> mplapack_iwork(static_cast<std::size_t>(std::max<blas_int>(1, n)));
    for (blas_int i = 0; i < std::max<blas_int>(1, n); ++i)
    {
      mplapack_ipiv[static_cast<std::size_t>(i)] = static_cast<mplapackint>(ipiv[i]);
    }
    Rsycon(uplo_string, mplapack_n, a, mplapack_lda, mplapack_ipiv.data(), matrix_one_norm, rcond, work,
           mplapack_iwork.data(), mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "sycon is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK sycon received an invalid argument");
  }
  return rcond;
}

template <uni20::LapackReal Scalar>
void trtrs(char uplo, char trans, char diag, blas_int n, blas_int nrhs, Scalar* a, blas_int lda, Scalar* b,
           blas_int ldb)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    strtrs_(&uplo, &trans, &diag, &n, &nrhs, a, &lda, b, &ldb, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dtrtrs_(&uplo, &trans, &diag, &n, &nrhs, a, &lda, b, &ldb, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_nrhs = static_cast<mplapackint>(nrhs);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_ldb = static_cast<mplapackint>(ldb);
    mplapackint mplapack_info = 0;
    char uplo_string[2] = {uplo, '\0'};
    char trans_string[2] = {trans, '\0'};
    char diag_string[2] = {diag, '\0'};
    Rtrtrs(uplo_string, trans_string, diag_string, mplapack_n, mplapack_nrhs, a, mplapack_lda, b, mplapack_ldb,
           mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "trtrs is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK trtrs received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK trtrs found a singular triangular matrix");
  }
}

template <uni20::LapackReal Scalar>
void trrfs(char uplo, char trans, char diag, blas_int n, blas_int nrhs, Scalar* a, blas_int lda, Scalar* b,
           blas_int ldb, Scalar* x, blas_int ldx, Scalar* forward_error, Scalar* backward_error, Scalar* work,
           blas_int* iwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    strrfs_(&uplo, &trans, &diag, &n, &nrhs, a, &lda, b, &ldb, x, &ldx, forward_error, backward_error, work, iwork,
            &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dtrrfs_(&uplo, &trans, &diag, &n, &nrhs, a, &lda, b, &ldb, x, &ldx, forward_error, backward_error, work, iwork,
            &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char uplo_string[2] = {uplo, '\0'};
    char trans_string[2] = {trans, '\0'};
    char diag_string[2] = {diag, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_nrhs = static_cast<mplapackint>(nrhs);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_ldb = static_cast<mplapackint>(ldb);
    mplapackint mplapack_ldx = static_cast<mplapackint>(ldx);
    mplapackint mplapack_info = 0;
    std::vector<mplapackint> mplapack_iwork(static_cast<std::size_t>(std::max<blas_int>(1, n)));
    Rtrrfs(uplo_string, trans_string, diag_string, mplapack_n, mplapack_nrhs, a, mplapack_lda, b, mplapack_ldb, x,
           mplapack_ldx, forward_error, backward_error, work, mplapack_iwork.data(), mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "trrfs is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK trrfs received an invalid argument");
  }
}

template <uni20::LapackReal Scalar> void trtri(char uplo, char diag, blas_int n, Scalar* a, blas_int lda)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    strtri_(&uplo, &diag, &n, a, &lda, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dtrtri_(&uplo, &diag, &n, a, &lda, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_info = 0;
    char uplo_string[2] = {uplo, '\0'};
    char diag_string[2] = {diag, '\0'};
    Rtrtri(uplo_string, diag_string, mplapack_n, a, mplapack_lda, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "trtri is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK trtri received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK trtri found a singular triangular matrix");
  }
}

template <uni20::LapackReal Scalar>
Scalar trcon(char norm, char uplo, char diag, blas_int n, Scalar* a, blas_int lda, Scalar* work, blas_int* iwork)
{
  blas_int info = 0;
  Scalar rcond{};
  if constexpr (std::is_same_v<Scalar, float>)
  {
    strcon_(&norm, &uplo, &diag, &n, a, &lda, &rcond, work, iwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dtrcon_(&norm, &uplo, &diag, &n, a, &lda, &rcond, work, iwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char norm_string[2] = {norm, '\0'};
    char uplo_string[2] = {uplo, '\0'};
    char diag_string[2] = {diag, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_info = 0;
    std::vector<mplapackint> mplapack_iwork(static_cast<std::size_t>(std::max<blas_int>(1, n)));
    Rtrcon(norm_string, uplo_string, diag_string, mplapack_n, a, mplapack_lda, rcond, work, mplapack_iwork.data(),
           mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "trcon is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK trcon received an invalid argument");
  }
  return rcond;
}

template <uni20::LapackReal Scalar>
bool trsyl(char trans_a, char trans_b, blas_int sign, blas_int m, blas_int n, Scalar* a, blas_int lda, Scalar* b,
           blas_int ldb, Scalar* c, blas_int ldc, Scalar& scale)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    strsyl_(&trans_a, &trans_b, &sign, &m, &n, a, &lda, b, &ldb, c, &ldc, &scale, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dtrsyl_(&trans_a, &trans_b, &sign, &m, &n, a, &lda, b, &ldb, c, &ldc, &scale, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char trans_a_string[2] = {trans_a, '\0'};
    char trans_b_string[2] = {trans_b, '\0'};
    mplapackint mplapack_sign = static_cast<mplapackint>(sign);
    mplapackint mplapack_m = static_cast<mplapackint>(m);
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_ldb = static_cast<mplapackint>(ldb);
    mplapackint mplapack_ldc = static_cast<mplapackint>(ldc);
    mplapackint mplapack_info = 0;
    Rtrsyl(trans_a_string, trans_b_string, mplapack_sign, mplapack_m, mplapack_n, a, mplapack_lda, b, mplapack_ldb, c,
           mplapack_ldc, scale, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "trsyl is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK trsyl received an invalid argument");
  }
  if (info > 1)
  {
    throw std::runtime_error("LAPACK trsyl failed unexpectedly");
  }
  return info == 1;
}

template <uni20::LapackReal Scalar>
void gesvd(char jobu, char jobvt, blas_int m, blas_int n, Scalar* a, blas_int lda, Scalar* s, Scalar* u, blas_int ldu,
           Scalar* vt, blas_int ldvt, Scalar* work, blas_int lwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sgesvd_(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dgesvd_(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    mplapackint mplapack_m = static_cast<mplapackint>(m);
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_ldu = static_cast<mplapackint>(ldu);
    mplapackint mplapack_ldvt = static_cast<mplapackint>(ldvt);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_info = 0;
    char jobu_string[2] = {jobu, '\0'};
    char jobvt_string[2] = {jobvt, '\0'};
    Rgesvd(jobu_string, jobvt_string, mplapack_m, mplapack_n, a, mplapack_lda, s, u, mplapack_ldu, vt, mplapack_ldvt,
           work, mplapack_lwork, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "gesvd is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK gesvd received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK gesvd failed to converge");
  }
}

template <uni20::LapackReal Scalar>
void gesdd(char jobz, blas_int m, blas_int n, Scalar* a, blas_int lda, Scalar* s, Scalar* u, blas_int ldu, Scalar* vt,
           blas_int ldvt, Scalar* work, blas_int lwork, blas_int* iwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sgesdd_(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, iwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dgesdd_(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, iwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char jobz_string[2] = {jobz, '\0'};
    mplapackint mplapack_m = static_cast<mplapackint>(m);
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_ldu = static_cast<mplapackint>(ldu);
    mplapackint mplapack_ldvt = static_cast<mplapackint>(ldvt);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_info = 0;
    std::vector<mplapackint> mplapack_iwork(static_cast<std::size_t>(std::max<blas_int>(1, 8 * std::min(m, n))));
    Rgesdd(jobz_string, mplapack_m, mplapack_n, a, mplapack_lda, s, u, mplapack_ldu, vt, mplapack_ldvt, work,
           mplapack_lwork, mplapack_iwork.data(), mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "gesdd is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK gesdd received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK gesdd failed to converge");
  }
}

template <uni20::LapackReal Scalar>
void gesvdx(char jobu, char jobvt, char range, blas_int m, blas_int n, Scalar* a, blas_int lda, Scalar vl, Scalar vu,
            blas_int il, blas_int iu, blas_int& selected_count, Scalar* singular_values, Scalar* u, blas_int ldu,
            Scalar* vt, blas_int ldvt, Scalar* work, blas_int lwork, blas_int* iwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sgesvdx_(&jobu, &jobvt, &range, &m, &n, a, &lda, &vl, &vu, &il, &iu, &selected_count, singular_values, u, &ldu, vt,
             &ldvt, work, &lwork, iwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dgesvdx_(&jobu, &jobvt, &range, &m, &n, a, &lda, &vl, &vu, &il, &iu, &selected_count, singular_values, u, &ldu, vt,
             &ldvt, work, &lwork, iwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char jobu_string[2] = {jobu, '\0'};
    char jobvt_string[2] = {jobvt, '\0'};
    char range_string[2] = {range, '\0'};
    mplapackint mplapack_m = static_cast<mplapackint>(m);
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_il = static_cast<mplapackint>(il);
    mplapackint mplapack_iu = static_cast<mplapackint>(iu);
    mplapackint mplapack_selected_count = static_cast<mplapackint>(selected_count);
    mplapackint mplapack_ldu = static_cast<mplapackint>(ldu);
    mplapackint mplapack_ldvt = static_cast<mplapackint>(ldvt);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_info = 0;
    std::size_t const iwork_size =
        static_cast<std::size_t>(std::max<blas_int>(1, 12 * std::max<blas_int>(1, std::min(m, n))));
    std::vector<mplapackint> mplapack_iwork(iwork_size);
    for (std::size_t i = 0; i < iwork_size; ++i)
    {
      mplapack_iwork[i] = static_cast<mplapackint>(iwork[i]);
    }
    Rgesvdx(jobu_string, jobvt_string, range_string, mplapack_m, mplapack_n, a, mplapack_lda, vl, vu, mplapack_il,
            mplapack_iu, mplapack_selected_count, singular_values, u, mplapack_ldu, vt, mplapack_ldvt, work,
            mplapack_lwork, mplapack_iwork.data(), mplapack_info);
    selected_count = static_cast<blas_int>(mplapack_selected_count);
    for (std::size_t i = 0; i < iwork_size; ++i)
    {
      iwork[i] = static_cast<blas_int>(mplapack_iwork[i]);
    }
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "gesvdx is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK gesvdx received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK gesvdx failed to converge");
  }
}

template <uni20::LapackReal Scalar>
void syev(char jobz, char uplo, blas_int n, Scalar* a, blas_int lda, Scalar* w, Scalar* work, blas_int lwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    ssyev_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dsyev_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_info = 0;
    char jobz_string[2] = {jobz, '\0'};
    char uplo_string[2] = {uplo, '\0'};
    Rsyev(jobz_string, uplo_string, mplapack_n, a, mplapack_lda, w, work, mplapack_lwork, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "syev is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK syev received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK syev failed to converge");
  }
}

template <uni20::LapackReal Scalar>
void syevd(char jobz, char uplo, blas_int n, Scalar* a, blas_int lda, Scalar* w, Scalar* work, blas_int lwork,
           blas_int* iwork, blas_int liwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    ssyevd_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, iwork, &liwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dsyevd_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, iwork, &liwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char jobz_string[2] = {jobz, '\0'};
    char uplo_string[2] = {uplo, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_liwork = static_cast<mplapackint>(liwork);
    mplapackint mplapack_info = 0;
    std::vector<mplapackint> mplapack_iwork(static_cast<std::size_t>(std::max<blas_int>(1, liwork)));
    Rsyevd(jobz_string, uplo_string, mplapack_n, a, mplapack_lda, w, work, mplapack_lwork, mplapack_iwork.data(),
           mplapack_liwork, mplapack_info);
    for (blas_int i = 0; i < std::max<blas_int>(1, liwork); ++i)
    {
      iwork[i] = static_cast<blas_int>(mplapack_iwork[static_cast<std::size_t>(i)]);
    }
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "syevd is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK syevd received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK syevd failed to converge");
  }
}

template <uni20::LapackReal Scalar>
void syevr(char jobz, char range, char uplo, blas_int n, Scalar* a, blas_int lda, Scalar vl, Scalar vu, blas_int il,
           blas_int iu, Scalar abstol, blas_int& selected_count, Scalar* w, Scalar* z, blas_int ldz, blas_int* isuppz,
           Scalar* work, blas_int lwork, blas_int* iwork, blas_int liwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    ssyevr_(&jobz, &range, &uplo, &n, a, &lda, &vl, &vu, &il, &iu, &abstol, &selected_count, w, z, &ldz, isuppz, work,
            &lwork, iwork, &liwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dsyevr_(&jobz, &range, &uplo, &n, a, &lda, &vl, &vu, &il, &iu, &abstol, &selected_count, w, z, &ldz, isuppz, work,
            &lwork, iwork, &liwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char jobz_string[2] = {jobz, '\0'};
    char range_string[2] = {range, '\0'};
    char uplo_string[2] = {uplo, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_il = static_cast<mplapackint>(il);
    mplapackint mplapack_iu = static_cast<mplapackint>(iu);
    mplapackint mplapack_selected_count = static_cast<mplapackint>(selected_count);
    mplapackint mplapack_ldz = static_cast<mplapackint>(ldz);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_liwork = static_cast<mplapackint>(liwork);
    mplapackint mplapack_info = 0;
    std::vector<mplapackint> mplapack_isuppz(
        static_cast<std::size_t>(std::max<blas_int>(1, 2 * std::max<blas_int>(1, n))));
    std::vector<mplapackint> mplapack_iwork(static_cast<std::size_t>(std::max<blas_int>(1, liwork)));
    Rsyevr(jobz_string, range_string, uplo_string, mplapack_n, a, mplapack_lda, vl, vu, mplapack_il, mplapack_iu,
           abstol, mplapack_selected_count, w, z, mplapack_ldz, mplapack_isuppz.data(), work, mplapack_lwork,
           mplapack_iwork.data(), mplapack_liwork, mplapack_info);
    selected_count = static_cast<blas_int>(mplapack_selected_count);
    for (blas_int i = 0; i < std::max<blas_int>(1, 2 * std::max<blas_int>(1, n)); ++i)
    {
      isuppz[i] = static_cast<blas_int>(mplapack_isuppz[static_cast<std::size_t>(i)]);
    }
    for (blas_int i = 0; i < std::max<blas_int>(1, liwork); ++i)
    {
      iwork[i] = static_cast<blas_int>(mplapack_iwork[static_cast<std::size_t>(i)]);
    }
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "syevr is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK syevr received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK syevr failed to converge");
  }
}

template <uni20::LapackReal Scalar>
void sygv(blas_int itype, char jobz, char uplo, blas_int n, Scalar* a, blas_int lda, Scalar* b, blas_int ldb, Scalar* w,
          Scalar* work, blas_int lwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    ssygv_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work, &lwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dsygv_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work, &lwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    mplapackint mplapack_itype = static_cast<mplapackint>(itype);
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_ldb = static_cast<mplapackint>(ldb);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_info = 0;
    char jobz_string[2] = {jobz, '\0'};
    char uplo_string[2] = {uplo, '\0'};
    Rsygv(mplapack_itype, jobz_string, uplo_string, mplapack_n, a, mplapack_lda, b, mplapack_ldb, w, work,
          mplapack_lwork, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "sygv is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK sygv received an invalid argument");
  }
  if (info > n)
  {
    throw std::runtime_error("LAPACK sygv found a metric matrix that is not positive definite");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK sygv failed to converge");
  }
}

template <uni20::LapackReal Scalar>
void sygvd(blas_int itype, char jobz, char uplo, blas_int n, Scalar* a, blas_int lda, Scalar* b, blas_int ldb,
           Scalar* w, Scalar* work, blas_int lwork, blas_int* iwork, blas_int liwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    ssygvd_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work, &lwork, iwork, &liwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dsygvd_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work, &lwork, iwork, &liwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    mplapackint mplapack_itype = static_cast<mplapackint>(itype);
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_ldb = static_cast<mplapackint>(ldb);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_liwork = static_cast<mplapackint>(liwork);
    mplapackint mplapack_info = 0;
    char jobz_string[2] = {jobz, '\0'};
    char uplo_string[2] = {uplo, '\0'};
    std::vector<mplapackint> mplapack_iwork(static_cast<std::size_t>(std::max<blas_int>(1, liwork)));
    Rsygvd(mplapack_itype, jobz_string, uplo_string, mplapack_n, a, mplapack_lda, b, mplapack_ldb, w, work,
           mplapack_lwork, mplapack_iwork.data(), mplapack_liwork, mplapack_info);
    for (blas_int i = 0; i < std::max<blas_int>(1, liwork); ++i)
    {
      iwork[i] = static_cast<blas_int>(mplapack_iwork[static_cast<std::size_t>(i)]);
    }
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "sygvd is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK sygvd received an invalid argument");
  }
  if (info > n)
  {
    throw std::runtime_error("LAPACK sygvd found a metric matrix that is not positive definite");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK sygvd failed to converge");
  }
}

template <uni20::LapackReal Scalar>
void sygvx(blas_int itype, char jobz, char range, char uplo, blas_int n, Scalar* a, blas_int lda, Scalar* b,
           blas_int ldb, Scalar vl, Scalar vu, blas_int il, blas_int iu, Scalar abstol, blas_int& selected_count,
           Scalar* w, Scalar* z, blas_int ldz, Scalar* work, blas_int lwork, blas_int* iwork, blas_int* ifail)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    ssygvx_(&itype, &jobz, &range, &uplo, &n, a, &lda, b, &ldb, &vl, &vu, &il, &iu, &abstol, &selected_count, w, z,
            &ldz, work, &lwork, iwork, ifail, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dsygvx_(&itype, &jobz, &range, &uplo, &n, a, &lda, b, &ldb, &vl, &vu, &il, &iu, &abstol, &selected_count, w, z,
            &ldz, work, &lwork, iwork, ifail, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char jobz_string[2] = {jobz, '\0'};
    char range_string[2] = {range, '\0'};
    char uplo_string[2] = {uplo, '\0'};
    mplapackint mplapack_itype = static_cast<mplapackint>(itype);
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_ldb = static_cast<mplapackint>(ldb);
    mplapackint mplapack_il = static_cast<mplapackint>(il);
    mplapackint mplapack_iu = static_cast<mplapackint>(iu);
    mplapackint mplapack_selected_count = static_cast<mplapackint>(selected_count);
    mplapackint mplapack_ldz = static_cast<mplapackint>(ldz);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_info = 0;
    std::vector<mplapackint> mplapack_iwork(
        static_cast<std::size_t>(std::max<blas_int>(1, 5 * std::max<blas_int>(1, n))));
    std::vector<mplapackint> mplapack_ifail(static_cast<std::size_t>(std::max<blas_int>(1, n)));
    Rsygvx(mplapack_itype, jobz_string, range_string, uplo_string, mplapack_n, a, mplapack_lda, b, mplapack_ldb, vl, vu,
           mplapack_il, mplapack_iu, abstol, mplapack_selected_count, w, z, mplapack_ldz, work, mplapack_lwork,
           mplapack_iwork.data(), mplapack_ifail.data(), mplapack_info);
    selected_count = static_cast<blas_int>(mplapack_selected_count);
    for (blas_int i = 0; i < std::max<blas_int>(1, 5 * std::max<blas_int>(1, n)); ++i)
    {
      iwork[i] = static_cast<blas_int>(mplapack_iwork[static_cast<std::size_t>(i)]);
    }
    for (blas_int i = 0; i < std::max<blas_int>(1, n); ++i)
    {
      ifail[i] = static_cast<blas_int>(mplapack_ifail[static_cast<std::size_t>(i)]);
    }
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "sygvx is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK sygvx received an invalid argument");
  }
  if (info > n)
  {
    throw std::runtime_error("LAPACK sygvx found a metric matrix that is not positive definite");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK sygvx failed to converge");
  }
}

template <uni20::LapackComplexReal Real>
void heev(char jobz, char uplo, blas_int n, uni20::complex<Real>* a, blas_int lda, Real* w, uni20::complex<Real>* work,
          blas_int lwork, Real* rwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Real, float>)
  {
    cheev_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, rwork, &info);
  }
  else if constexpr (std::is_same_v<Real, double>)
  {
    zheev_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, rwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Real>)
  {
    char jobz_string[2] = {jobz, '\0'};
    char uplo_string[2] = {uplo, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_info = 0;
    Cheev(jobz_string, uplo_string, mplapack_n, a, mplapack_lda, w, work, mplapack_lwork, rwork, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Real, float> || std::is_same_v<Real, double> || is_mplapack_binary128_v<Real>,
                  "heev is available only for complex float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK heev received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK heev failed to converge");
  }
}

template <uni20::LapackComplexReal Real>
void heevd(char jobz, char uplo, blas_int n, uni20::complex<Real>* a, blas_int lda, Real* w, uni20::complex<Real>* work,
           blas_int lwork, Real* rwork, blas_int lrwork, blas_int* iwork, blas_int liwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Real, float>)
  {
    cheevd_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, rwork, &lrwork, iwork, &liwork, &info);
  }
  else if constexpr (std::is_same_v<Real, double>)
  {
    zheevd_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, rwork, &lrwork, iwork, &liwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Real>)
  {
    char jobz_string[2] = {jobz, '\0'};
    char uplo_string[2] = {uplo, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_lrwork = static_cast<mplapackint>(lrwork);
    mplapackint mplapack_liwork = static_cast<mplapackint>(liwork);
    mplapackint mplapack_info = 0;
    std::vector<mplapackint> mplapack_iwork(static_cast<std::size_t>(std::max<blas_int>(1, liwork)));
    Cheevd(jobz_string, uplo_string, mplapack_n, a, mplapack_lda, w, work, mplapack_lwork, rwork, mplapack_lrwork,
           mplapack_iwork.data(), mplapack_liwork, mplapack_info);
    for (blas_int i = 0; i < std::max<blas_int>(1, liwork); ++i)
    {
      iwork[i] = static_cast<blas_int>(mplapack_iwork[static_cast<std::size_t>(i)]);
    }
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Real, float> || std::is_same_v<Real, double> || is_mplapack_binary128_v<Real>,
                  "heevd is available only for complex float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK heevd received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK heevd failed to converge");
  }
}

template <uni20::LapackComplexReal Real>
void heevr(char jobz, char range, char uplo, blas_int n, uni20::complex<Real>* a, blas_int lda, Real vl, Real vu,
           blas_int il, blas_int iu, Real abstol, blas_int& selected_count, Real* w, uni20::complex<Real>* z,
           blas_int ldz, blas_int* isuppz, uni20::complex<Real>* work, blas_int lwork, Real* rwork, blas_int lrwork,
           blas_int* iwork, blas_int liwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Real, float>)
  {
    cheevr_(&jobz, &range, &uplo, &n, a, &lda, &vl, &vu, &il, &iu, &abstol, &selected_count, w, z, &ldz, isuppz, work,
            &lwork, rwork, &lrwork, iwork, &liwork, &info);
  }
  else if constexpr (std::is_same_v<Real, double>)
  {
    zheevr_(&jobz, &range, &uplo, &n, a, &lda, &vl, &vu, &il, &iu, &abstol, &selected_count, w, z, &ldz, isuppz, work,
            &lwork, rwork, &lrwork, iwork, &liwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Real>)
  {
    char jobz_string[2] = {jobz, '\0'};
    char range_string[2] = {range, '\0'};
    char uplo_string[2] = {uplo, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_il = static_cast<mplapackint>(il);
    mplapackint mplapack_iu = static_cast<mplapackint>(iu);
    mplapackint mplapack_selected_count = static_cast<mplapackint>(selected_count);
    mplapackint mplapack_ldz = static_cast<mplapackint>(ldz);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_lrwork = static_cast<mplapackint>(lrwork);
    mplapackint mplapack_liwork = static_cast<mplapackint>(liwork);
    mplapackint mplapack_info = 0;
    std::vector<mplapackint> mplapack_isuppz(
        static_cast<std::size_t>(std::max<blas_int>(1, 2 * std::max<blas_int>(1, n))));
    std::vector<mplapackint> mplapack_iwork(static_cast<std::size_t>(std::max<blas_int>(1, liwork)));
    Cheevr(jobz_string, range_string, uplo_string, mplapack_n, a, mplapack_lda, vl, vu, mplapack_il, mplapack_iu,
           abstol, mplapack_selected_count, w, z, mplapack_ldz, mplapack_isuppz.data(), work, mplapack_lwork, rwork,
           mplapack_lrwork, mplapack_iwork.data(), mplapack_liwork, mplapack_info);
    selected_count = static_cast<blas_int>(mplapack_selected_count);
    for (blas_int i = 0; i < std::max<blas_int>(1, 2 * std::max<blas_int>(1, n)); ++i)
    {
      isuppz[i] = static_cast<blas_int>(mplapack_isuppz[static_cast<std::size_t>(i)]);
    }
    for (blas_int i = 0; i < std::max<blas_int>(1, liwork); ++i)
    {
      iwork[i] = static_cast<blas_int>(mplapack_iwork[static_cast<std::size_t>(i)]);
    }
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Real, float> || std::is_same_v<Real, double> || is_mplapack_binary128_v<Real>,
                  "heevr is available only for complex float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK heevr received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK heevr failed to converge");
  }
}

template <uni20::LapackComplexReal Real>
void hegv(blas_int itype, char jobz, char uplo, blas_int n, uni20::complex<Real>* a, blas_int lda,
          uni20::complex<Real>* b, blas_int ldb, Real* w, uni20::complex<Real>* work, blas_int lwork, Real* rwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Real, float>)
  {
    chegv_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work, &lwork, rwork, &info);
  }
  else if constexpr (std::is_same_v<Real, double>)
  {
    zhegv_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work, &lwork, rwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Real>)
  {
    char jobz_string[2] = {jobz, '\0'};
    char uplo_string[2] = {uplo, '\0'};
    mplapackint mplapack_itype = static_cast<mplapackint>(itype);
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_ldb = static_cast<mplapackint>(ldb);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_info = 0;
    Chegv(mplapack_itype, jobz_string, uplo_string, mplapack_n, a, mplapack_lda, b, mplapack_ldb, w, work,
          mplapack_lwork, rwork, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Real, float> || std::is_same_v<Real, double> || is_mplapack_binary128_v<Real>,
                  "hegv is available only for complex float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK hegv received an invalid argument");
  }
  if (info > n)
  {
    throw std::runtime_error("LAPACK hegv found a metric matrix that is not positive definite");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK hegv failed to converge");
  }
}

template <uni20::LapackComplexReal Real>
void hegvd(blas_int itype, char jobz, char uplo, blas_int n, uni20::complex<Real>* a, blas_int lda,
           uni20::complex<Real>* b, blas_int ldb, Real* w, uni20::complex<Real>* work, blas_int lwork, Real* rwork,
           blas_int lrwork, blas_int* iwork, blas_int liwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Real, float>)
  {
    chegvd_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work, &lwork, rwork, &lrwork, iwork, &liwork, &info);
  }
  else if constexpr (std::is_same_v<Real, double>)
  {
    zhegvd_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work, &lwork, rwork, &lrwork, iwork, &liwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Real>)
  {
    char jobz_string[2] = {jobz, '\0'};
    char uplo_string[2] = {uplo, '\0'};
    mplapackint mplapack_itype = static_cast<mplapackint>(itype);
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_ldb = static_cast<mplapackint>(ldb);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_lrwork = static_cast<mplapackint>(lrwork);
    mplapackint mplapack_liwork = static_cast<mplapackint>(liwork);
    mplapackint mplapack_info = 0;
    std::vector<mplapackint> mplapack_iwork(static_cast<std::size_t>(std::max<blas_int>(1, liwork)));
    Chegvd(mplapack_itype, jobz_string, uplo_string, mplapack_n, a, mplapack_lda, b, mplapack_ldb, w, work,
           mplapack_lwork, rwork, mplapack_lrwork, mplapack_iwork.data(), mplapack_liwork, mplapack_info);
    for (blas_int i = 0; i < std::max<blas_int>(1, liwork); ++i)
    {
      iwork[i] = static_cast<blas_int>(mplapack_iwork[static_cast<std::size_t>(i)]);
    }
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Real, float> || std::is_same_v<Real, double> || is_mplapack_binary128_v<Real>,
                  "hegvd is available only for complex float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK hegvd received an invalid argument");
  }
  if (info > n)
  {
    throw std::runtime_error("LAPACK hegvd found a metric matrix that is not positive definite");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK hegvd failed to converge");
  }
}

template <uni20::LapackComplexReal Real>
void hegvx(blas_int itype, char jobz, char range, char uplo, blas_int n, uni20::complex<Real>* a, blas_int lda,
           uni20::complex<Real>* b, blas_int ldb, Real vl, Real vu, blas_int il, blas_int iu, Real abstol,
           blas_int& selected_count, Real* w, uni20::complex<Real>* z, blas_int ldz, uni20::complex<Real>* work,
           blas_int lwork, Real* rwork, blas_int* iwork, blas_int* ifail)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Real, float>)
  {
    chegvx_(&itype, &jobz, &range, &uplo, &n, a, &lda, b, &ldb, &vl, &vu, &il, &iu, &abstol, &selected_count, w, z,
            &ldz, work, &lwork, rwork, iwork, ifail, &info);
  }
  else if constexpr (std::is_same_v<Real, double>)
  {
    zhegvx_(&itype, &jobz, &range, &uplo, &n, a, &lda, b, &ldb, &vl, &vu, &il, &iu, &abstol, &selected_count, w, z,
            &ldz, work, &lwork, rwork, iwork, ifail, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Real>)
  {
    char jobz_string[2] = {jobz, '\0'};
    char range_string[2] = {range, '\0'};
    char uplo_string[2] = {uplo, '\0'};
    mplapackint mplapack_itype = static_cast<mplapackint>(itype);
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_ldb = static_cast<mplapackint>(ldb);
    mplapackint mplapack_il = static_cast<mplapackint>(il);
    mplapackint mplapack_iu = static_cast<mplapackint>(iu);
    mplapackint mplapack_selected_count = static_cast<mplapackint>(selected_count);
    mplapackint mplapack_ldz = static_cast<mplapackint>(ldz);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_info = 0;
    std::vector<mplapackint> mplapack_iwork(
        static_cast<std::size_t>(std::max<blas_int>(1, 5 * std::max<blas_int>(1, n))));
    std::vector<mplapackint> mplapack_ifail(static_cast<std::size_t>(std::max<blas_int>(1, n)));
    Chegvx(mplapack_itype, jobz_string, range_string, uplo_string, mplapack_n, a, mplapack_lda, b, mplapack_ldb, vl, vu,
           mplapack_il, mplapack_iu, abstol, mplapack_selected_count, w, z, mplapack_ldz, work, mplapack_lwork, rwork,
           mplapack_iwork.data(), mplapack_ifail.data(), mplapack_info);
    selected_count = static_cast<blas_int>(mplapack_selected_count);
    for (blas_int i = 0; i < std::max<blas_int>(1, 5 * std::max<blas_int>(1, n)); ++i)
    {
      iwork[i] = static_cast<blas_int>(mplapack_iwork[static_cast<std::size_t>(i)]);
    }
    for (blas_int i = 0; i < std::max<blas_int>(1, n); ++i)
    {
      ifail[i] = static_cast<blas_int>(mplapack_ifail[static_cast<std::size_t>(i)]);
    }
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Real, float> || std::is_same_v<Real, double> || is_mplapack_binary128_v<Real>,
                  "hegvx is available only for complex float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK hegvx received an invalid argument");
  }
  if (info > n)
  {
    throw std::runtime_error("LAPACK hegvx found a metric matrix that is not positive definite");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK hegvx failed to converge");
  }
}

template <uni20::LapackReal Scalar>
void bdsqr(char uplo, blas_int n, blas_int ncvt, blas_int nru, blas_int ncc, Scalar* d, Scalar* e, Scalar* vt,
           blas_int ldvt, Scalar* u, blas_int ldu, Scalar* c, blas_int ldc, Scalar* work)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sbdsqr_(&uplo, &n, &ncvt, &nru, &ncc, d, e, vt, &ldvt, u, &ldu, c, &ldc, work, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dbdsqr_(&uplo, &n, &ncvt, &nru, &ncc, d, e, vt, &ldvt, u, &ldu, c, &ldc, work, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char uplo_string[2] = {uplo, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_ncvt = static_cast<mplapackint>(ncvt);
    mplapackint mplapack_nru = static_cast<mplapackint>(nru);
    mplapackint mplapack_ncc = static_cast<mplapackint>(ncc);
    mplapackint mplapack_ldvt = static_cast<mplapackint>(ldvt);
    mplapackint mplapack_ldu = static_cast<mplapackint>(ldu);
    mplapackint mplapack_ldc = static_cast<mplapackint>(ldc);
    mplapackint mplapack_info = 0;
    Rbdsqr(uplo_string, mplapack_n, mplapack_ncvt, mplapack_nru, mplapack_ncc, d, e, vt, mplapack_ldvt, u, mplapack_ldu,
           c, mplapack_ldc, work, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "bdsqr is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK bdsqr received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK bdsqr failed to converge");
  }
}

template <uni20::LapackReal Scalar>
void bdsdc(char uplo, char compq, blas_int n, Scalar* d, Scalar* e, Scalar* u, blas_int ldu, Scalar* vt, blas_int ldvt,
           Scalar* q, blas_int* iq, Scalar* work, blas_int* iwork)
{
  if (compq != 'N' && compq != 'I')
  {
    throw std::invalid_argument("LAPACK bdsdc wrapper supports only singular values or explicit singular vectors");
  }

  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sbdsdc_(&uplo, &compq, &n, d, e, u, &ldu, vt, &ldvt, q, iq, work, iwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dbdsdc_(&uplo, &compq, &n, d, e, u, &ldu, vt, &ldvt, q, iq, work, iwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char uplo_string[2] = {uplo, '\0'};
    char compq_string[2] = {compq, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_ldu = static_cast<mplapackint>(ldu);
    mplapackint mplapack_ldvt = static_cast<mplapackint>(ldvt);
    mplapackint mplapack_info = 0;
    std::size_t const iwork_size = static_cast<std::size_t>(std::max<blas_int>(1, 8 * std::max<blas_int>(1, n)));
    std::vector<mplapackint> mplapack_iq(1);
    std::vector<mplapackint> mplapack_iwork(iwork_size);
    mplapack_iq[0] = static_cast<mplapackint>(iq[0]);
    for (std::size_t i = 0; i < iwork_size; ++i)
    {
      mplapack_iwork[i] = static_cast<mplapackint>(iwork[i]);
    }
    Rbdsdc(uplo_string, compq_string, mplapack_n, d, e, u, mplapack_ldu, vt, mplapack_ldvt, q, mplapack_iq.data(), work,
           mplapack_iwork.data(), mplapack_info);
    iq[0] = static_cast<blas_int>(mplapack_iq[0]);
    for (std::size_t i = 0; i < iwork_size; ++i)
    {
      iwork[i] = static_cast<blas_int>(mplapack_iwork[i]);
    }
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "bdsdc is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK bdsdc received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK bdsdc failed to converge");
  }
}

template <uni20::LapackReal Scalar>
void bdsvdx(char uplo, char jobz, char range, blas_int n, Scalar* d, Scalar* e, Scalar vl, Scalar vu, blas_int il,
            blas_int iu, blas_int& selected_count, Scalar* singular_values, Scalar* z, blas_int ldz, Scalar* work,
            blas_int* iwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sbdsvdx_(&uplo, &jobz, &range, &n, d, e, &vl, &vu, &il, &iu, &selected_count, singular_values, z, &ldz, work, iwork,
             &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dbdsvdx_(&uplo, &jobz, &range, &n, d, e, &vl, &vu, &il, &iu, &selected_count, singular_values, z, &ldz, work, iwork,
             &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char uplo_string[2] = {uplo, '\0'};
    char jobz_string[2] = {jobz, '\0'};
    char range_string[2] = {range, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_il = static_cast<mplapackint>(il);
    mplapackint mplapack_iu = static_cast<mplapackint>(iu);
    mplapackint mplapack_selected_count = static_cast<mplapackint>(selected_count);
    mplapackint mplapack_ldz = static_cast<mplapackint>(ldz);
    mplapackint mplapack_info = 0;
    std::size_t const iwork_size = static_cast<std::size_t>(std::max<blas_int>(1, 12 * std::max<blas_int>(1, n)));
    std::vector<mplapackint> mplapack_iwork(iwork_size);
    for (std::size_t i = 0; i < iwork_size; ++i)
    {
      mplapack_iwork[i] = static_cast<mplapackint>(iwork[i]);
    }
    Rbdsvdx(uplo_string, jobz_string, range_string, mplapack_n, d, e, vl, vu, mplapack_il, mplapack_iu,
            mplapack_selected_count, singular_values, z, mplapack_ldz, work, mplapack_iwork.data(), mplapack_info);
    selected_count = static_cast<blas_int>(mplapack_selected_count);
    for (std::size_t i = 0; i < iwork_size; ++i)
    {
      iwork[i] = static_cast<blas_int>(mplapack_iwork[i]);
    }
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "bdsvdx is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK bdsvdx received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK bdsvdx failed to converge");
  }
}

template <uni20::LapackReal Scalar> void sterf(blas_int n, Scalar* d, Scalar* e)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    ssterf_(&n, d, e, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dsterf_(&n, d, e, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_info = 0;
    Rsterf(mplapack_n, d, e, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "sterf is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK sterf received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK sterf failed to converge");
  }
}

template <typename Scalar>
void stevd(char jobz, blas_int n, Scalar* d, Scalar* e, Scalar* z, blas_int ldz, Scalar* work, blas_int lwork,
           blas_int* iwork, blas_int liwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sstevd_(&jobz, &n, d, e, z, &ldz, work, &lwork, iwork, &liwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dstevd_(&jobz, &n, d, e, z, &ldz, work, &lwork, iwork, &liwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char jobz_string[2] = {jobz, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_ldz = static_cast<mplapackint>(ldz);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_liwork = static_cast<mplapackint>(liwork);
    mplapackint mplapack_info = 0;
    std::size_t const iwork_size = static_cast<std::size_t>(std::max<blas_int>(1, liwork == -1 ? blas_int{1} : liwork));
    std::vector<mplapackint> mplapack_iwork(iwork_size);
    for (std::size_t i = 0; i < iwork_size; ++i)
    {
      mplapack_iwork[i] = static_cast<mplapackint>(iwork[i]);
    }
    Rstevd(jobz_string, mplapack_n, d, e, z, mplapack_ldz, work, mplapack_lwork, mplapack_iwork.data(), mplapack_liwork,
           mplapack_info);
    for (std::size_t i = 0; i < iwork_size; ++i)
    {
      iwork[i] = static_cast<blas_int>(mplapack_iwork[i]);
    }
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "stevd is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK stevd received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK stevd failed to converge");
  }
}

template <typename Scalar>
void stevr(char jobz, char range, blas_int n, Scalar* d, Scalar* e, Scalar vl, Scalar vu, blas_int il, blas_int iu,
           Scalar abstol, blas_int& selected_count, Scalar* w, Scalar* z, blas_int ldz, blas_int* isuppz, Scalar* work,
           blas_int lwork, blas_int* iwork, blas_int liwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sstevr_(&jobz, &range, &n, d, e, &vl, &vu, &il, &iu, &abstol, &selected_count, w, z, &ldz, isuppz, work, &lwork,
            iwork, &liwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dstevr_(&jobz, &range, &n, d, e, &vl, &vu, &il, &iu, &abstol, &selected_count, w, z, &ldz, isuppz, work, &lwork,
            iwork, &liwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char jobz_string[2] = {jobz, '\0'};
    char range_string[2] = {range, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_il = static_cast<mplapackint>(il);
    mplapackint mplapack_iu = static_cast<mplapackint>(iu);
    mplapackint mplapack_selected_count = static_cast<mplapackint>(selected_count);
    mplapackint mplapack_ldz = static_cast<mplapackint>(ldz);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_liwork = static_cast<mplapackint>(liwork);
    mplapackint mplapack_info = 0;
    std::size_t const support_size = static_cast<std::size_t>(std::max<blas_int>(1, 2 * std::max<blas_int>(1, n)));
    std::size_t const iwork_size = static_cast<std::size_t>(std::max<blas_int>(1, liwork == -1 ? blas_int{1} : liwork));
    std::vector<mplapackint> mplapack_isuppz(support_size);
    std::vector<mplapackint> mplapack_iwork(iwork_size);
    for (std::size_t i = 0; i < support_size; ++i)
    {
      mplapack_isuppz[i] = static_cast<mplapackint>(isuppz[i]);
    }
    for (std::size_t i = 0; i < iwork_size; ++i)
    {
      mplapack_iwork[i] = static_cast<mplapackint>(iwork[i]);
    }
    Rstevr(jobz_string, range_string, mplapack_n, d, e, vl, vu, mplapack_il, mplapack_iu, abstol,
           mplapack_selected_count, w, z, mplapack_ldz, mplapack_isuppz.data(), work, mplapack_lwork,
           mplapack_iwork.data(), mplapack_liwork, mplapack_info);
    selected_count = static_cast<blas_int>(mplapack_selected_count);
    for (std::size_t i = 0; i < support_size; ++i)
    {
      isuppz[i] = static_cast<blas_int>(mplapack_isuppz[i]);
    }
    for (std::size_t i = 0; i < iwork_size; ++i)
    {
      iwork[i] = static_cast<blas_int>(mplapack_iwork[i]);
    }
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "stevr is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK stevr received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK stevr failed to converge");
  }
}

template <typename Scalar>
void steqr(char compz, blas_int n, Scalar* d, Scalar* e, Scalar* z, blas_int ldz, Scalar* work)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    ssteqr_(&compz, &n, d, e, z, &ldz, work, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dsteqr_(&compz, &n, d, e, z, &ldz, work, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_ldz = static_cast<mplapackint>(ldz);
    mplapackint mplapack_info = 0;
    char compz_string[2] = {compz, '\0'};
    Rsteqr(compz_string, mplapack_n, d, e, z, mplapack_ldz, work, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "steqr is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK steqr received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK steqr failed to converge");
  }
}

template <uni20::LapackReal Scalar>
void geev(char jobvl, char jobvr, blas_int n, Scalar* a, blas_int lda, Scalar* wr, Scalar* wi, Scalar* vl,
          blas_int ldvl, Scalar* vr, blas_int ldvr, Scalar* work, blas_int lwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sgeev_(&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, &lwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dgeev_(&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, &lwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_ldvl = static_cast<mplapackint>(ldvl);
    mplapackint mplapack_ldvr = static_cast<mplapackint>(ldvr);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_info = 0;
    char jobvl_string[2] = {jobvl, '\0'};
    char jobvr_string[2] = {jobvr, '\0'};
    Rgeev(jobvl_string, jobvr_string, mplapack_n, a, mplapack_lda, wr, wi, vl, mplapack_ldvl, vr, mplapack_ldvr, work,
          mplapack_lwork, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "real geev is available only for float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK geev received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK geev failed to converge");
  }
}

template <uni20::LapackReal Scalar>
void geevx(char balanc, char jobvl, char jobvr, char sense, blas_int n, Scalar* a, blas_int lda, Scalar* wr, Scalar* wi,
           Scalar* vl, blas_int ldvl, Scalar* vr, blas_int ldvr, blas_int& ilo, blas_int& ihi, Scalar* scale,
           Scalar& abnrm, Scalar* rconde, Scalar* rcondv, Scalar* work, blas_int lwork, blas_int* iwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sgeevx_(&balanc, &jobvl, &jobvr, &sense, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, &ilo, &ihi, scale, &abnrm,
            rconde, rcondv, work, &lwork, iwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dgeevx_(&balanc, &jobvl, &jobvr, &sense, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, &ilo, &ihi, scale, &abnrm,
            rconde, rcondv, work, &lwork, iwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char balanc_string[2] = {balanc, '\0'};
    char jobvl_string[2] = {jobvl, '\0'};
    char jobvr_string[2] = {jobvr, '\0'};
    char sense_string[2] = {sense, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_ldvl = static_cast<mplapackint>(ldvl);
    mplapackint mplapack_ldvr = static_cast<mplapackint>(ldvr);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_ilo = static_cast<mplapackint>(ilo);
    mplapackint mplapack_ihi = static_cast<mplapackint>(ihi);
    mplapackint mplapack_info = 0;
    std::vector<mplapackint> mplapack_iwork(static_cast<std::size_t>(std::max<blas_int>(1, 2 * n - 2)));
    for (blas_int i = 0; i < std::max<blas_int>(1, 2 * n - 2); ++i)
    {
      mplapack_iwork[static_cast<std::size_t>(i)] = static_cast<mplapackint>(iwork[i]);
    }
    Rgeevx(balanc_string, jobvl_string, jobvr_string, sense_string, mplapack_n, a, mplapack_lda, wr, wi, vl,
           mplapack_ldvl, vr, mplapack_ldvr, mplapack_ilo, mplapack_ihi, scale, abnrm, rconde, rcondv, work,
           mplapack_lwork, mplapack_iwork.data(), mplapack_info);
    ilo = static_cast<blas_int>(mplapack_ilo);
    ihi = static_cast<blas_int>(mplapack_ihi);
    for (blas_int i = 0; i < std::max<blas_int>(1, 2 * n - 2); ++i)
    {
      iwork[i] = static_cast<blas_int>(mplapack_iwork[static_cast<std::size_t>(i)]);
    }
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "real geevx is available only for float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK geevx received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK geevx failed to converge");
  }
}

template <uni20::LapackReal Scalar>
void gebal(char job, blas_int n, Scalar* a, blas_int lda, blas_int& first, blas_int& last, Scalar* scale)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sgebal_(&job, &n, a, &lda, &first, &last, scale, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dgebal_(&job, &n, a, &lda, &first, &last, scale, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char job_string[2] = {job, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_first = static_cast<mplapackint>(first);
    mplapackint mplapack_last = static_cast<mplapackint>(last);
    mplapackint mplapack_info = 0;
    Rgebal(job_string, mplapack_n, a, mplapack_lda, mplapack_first, mplapack_last, scale, mplapack_info);
    first = static_cast<blas_int>(mplapack_first);
    last = static_cast<blas_int>(mplapack_last);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "real gebal is available only for float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK gebal received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK gebal failed unexpectedly");
  }
}

template <uni20::LapackReal Scalar>
void gebak(char job, char side, blas_int n, blas_int first, blas_int last, Scalar* scale, blas_int vector_count,
           Scalar* vectors, blas_int leading_dimension)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sgebak_(&job, &side, &n, &first, &last, scale, &vector_count, vectors, &leading_dimension, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dgebak_(&job, &side, &n, &first, &last, scale, &vector_count, vectors, &leading_dimension, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char job_string[2] = {job, '\0'};
    char side_string[2] = {side, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_first = static_cast<mplapackint>(first);
    mplapackint mplapack_last = static_cast<mplapackint>(last);
    mplapackint mplapack_vector_count = static_cast<mplapackint>(vector_count);
    mplapackint mplapack_leading_dimension = static_cast<mplapackint>(leading_dimension);
    mplapackint mplapack_info = 0;
    Rgebak(job_string, side_string, mplapack_n, mplapack_first, mplapack_last, scale, mplapack_vector_count, vectors,
           mplapack_leading_dimension, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "real gebak is available only for float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK gebak received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK gebak failed unexpectedly");
  }
}

template <uni20::LapackReal Scalar>
void ggev(char jobvl, char jobvr, blas_int n, Scalar* a, blas_int lda, Scalar* b, blas_int ldb, Scalar* alphar,
          Scalar* alphai, Scalar* beta, Scalar* vl, blas_int ldvl, Scalar* vr, blas_int ldvr, Scalar* work,
          blas_int lwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sggev_(&jobvl, &jobvr, &n, a, &lda, b, &ldb, alphar, alphai, beta, vl, &ldvl, vr, &ldvr, work, &lwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dggev_(&jobvl, &jobvr, &n, a, &lda, b, &ldb, alphar, alphai, beta, vl, &ldvl, vr, &ldvr, work, &lwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_ldb = static_cast<mplapackint>(ldb);
    mplapackint mplapack_ldvl = static_cast<mplapackint>(ldvl);
    mplapackint mplapack_ldvr = static_cast<mplapackint>(ldvr);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_info = 0;
    char jobvl_string[2] = {jobvl, '\0'};
    char jobvr_string[2] = {jobvr, '\0'};
    Rggev(jobvl_string, jobvr_string, mplapack_n, a, mplapack_lda, b, mplapack_ldb, alphar, alphai, beta, vl,
          mplapack_ldvl, vr, mplapack_ldvr, work, mplapack_lwork, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "real ggev is available only for float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK ggev received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK ggev failed to converge");
  }
}

template <uni20::LapackReal Scalar>
void ggevx(char balanc, char jobvl, char jobvr, char sense, blas_int n, Scalar* a, blas_int lda, Scalar* b,
           blas_int ldb, Scalar* alphar, Scalar* alphai, Scalar* beta, Scalar* vl, blas_int ldvl, Scalar* vr,
           blas_int ldvr, blas_int& ilo, blas_int& ihi, Scalar* lscale, Scalar* rscale, Scalar& abnrm, Scalar& bbnrm,
           Scalar* rconde, Scalar* rcondv, Scalar* work, blas_int lwork, blas_int* iwork, blas_int* bwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sggevx_(&balanc, &jobvl, &jobvr, &sense, &n, a, &lda, b, &ldb, alphar, alphai, beta, vl, &ldvl, vr, &ldvr, &ilo,
            &ihi, lscale, rscale, &abnrm, &bbnrm, rconde, rcondv, work, &lwork, iwork, bwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dggevx_(&balanc, &jobvl, &jobvr, &sense, &n, a, &lda, b, &ldb, alphar, alphai, beta, vl, &ldvl, vr, &ldvr, &ilo,
            &ihi, lscale, rscale, &abnrm, &bbnrm, rconde, rcondv, work, &lwork, iwork, bwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char balanc_string[2] = {balanc, '\0'};
    char jobvl_string[2] = {jobvl, '\0'};
    char jobvr_string[2] = {jobvr, '\0'};
    char sense_string[2] = {sense, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_ldb = static_cast<mplapackint>(ldb);
    mplapackint mplapack_ldvl = static_cast<mplapackint>(ldvl);
    mplapackint mplapack_ldvr = static_cast<mplapackint>(ldvr);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_ilo = static_cast<mplapackint>(ilo);
    mplapackint mplapack_ihi = static_cast<mplapackint>(ihi);
    mplapackint mplapack_info = 0;
    std::size_t const iwork_size = static_cast<std::size_t>(std::max<blas_int>(1, n + 6));
    std::size_t const bwork_size = static_cast<std::size_t>(std::max<blas_int>(1, n));
    std::vector<mplapackint> mplapack_iwork(iwork_size);
    auto mplapack_bwork = std::make_unique<bool[]>(bwork_size);
    for (std::size_t i = 0; i < iwork_size; ++i)
    {
      mplapack_iwork[i] = static_cast<mplapackint>(iwork[i]);
    }
    for (std::size_t i = 0; i < bwork_size; ++i)
    {
      mplapack_bwork[i] = bwork[i] != 0;
    }
    Rggevx(balanc_string, jobvl_string, jobvr_string, sense_string, mplapack_n, a, mplapack_lda, b, mplapack_ldb,
           alphar, alphai, beta, vl, mplapack_ldvl, vr, mplapack_ldvr, mplapack_ilo, mplapack_ihi, lscale, rscale,
           abnrm, bbnrm, rconde, rcondv, work, mplapack_lwork, mplapack_iwork.data(), mplapack_bwork.get(),
           mplapack_info);
    ilo = static_cast<blas_int>(mplapack_ilo);
    ihi = static_cast<blas_int>(mplapack_ihi);
    for (std::size_t i = 0; i < iwork_size; ++i)
    {
      iwork[i] = static_cast<blas_int>(mplapack_iwork[i]);
    }
    for (std::size_t i = 0; i < bwork_size; ++i)
    {
      bwork[i] = mplapack_bwork[i] ? 1 : 0;
    }
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "real ggevx is available only for float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK ggevx received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK ggevx failed to converge");
  }
}

template <uni20::LapackReal Scalar>
void ggbal(char job, blas_int n, Scalar* a, blas_int lda, Scalar* b, blas_int ldb, blas_int& first, blas_int& last,
           Scalar* lscale, Scalar* rscale, Scalar* work)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sggbal_(&job, &n, a, &lda, b, &ldb, &first, &last, lscale, rscale, work, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dggbal_(&job, &n, a, &lda, b, &ldb, &first, &last, lscale, rscale, work, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char job_string[2] = {job, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_ldb = static_cast<mplapackint>(ldb);
    mplapackint mplapack_first = static_cast<mplapackint>(first);
    mplapackint mplapack_last = static_cast<mplapackint>(last);
    mplapackint mplapack_info = 0;
    Rggbal(job_string, mplapack_n, a, mplapack_lda, b, mplapack_ldb, mplapack_first, mplapack_last, lscale, rscale,
           work, mplapack_info);
    first = static_cast<blas_int>(mplapack_first);
    last = static_cast<blas_int>(mplapack_last);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "real ggbal is available only for float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK ggbal received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK ggbal failed unexpectedly");
  }
}

template <uni20::LapackReal Scalar>
void ggbak(char job, char side, blas_int n, blas_int first, blas_int last, Scalar* lscale, Scalar* rscale,
           blas_int vector_count, Scalar* vectors, blas_int leading_dimension)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sggbak_(&job, &side, &n, &first, &last, lscale, rscale, &vector_count, vectors, &leading_dimension, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dggbak_(&job, &side, &n, &first, &last, lscale, rscale, &vector_count, vectors, &leading_dimension, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char job_string[2] = {job, '\0'};
    char side_string[2] = {side, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_first = static_cast<mplapackint>(first);
    mplapackint mplapack_last = static_cast<mplapackint>(last);
    mplapackint mplapack_vector_count = static_cast<mplapackint>(vector_count);
    mplapackint mplapack_leading_dimension = static_cast<mplapackint>(leading_dimension);
    mplapackint mplapack_info = 0;
    Rggbak(job_string, side_string, mplapack_n, mplapack_first, mplapack_last, lscale, rscale, mplapack_vector_count,
           vectors, mplapack_leading_dimension, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "real ggbak is available only for float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK ggbak received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK ggbak failed unexpectedly");
  }
}

template <uni20::LapackReal Scalar>
void gges(char jobvsl, char jobvsr, char sort, blas_int n, Scalar* a, blas_int lda, Scalar* b, blas_int ldb,
          blas_int& selected_dimension, Scalar* alphar, Scalar* alphai, Scalar* beta, Scalar* vsl, blas_int ldvsl,
          Scalar* vsr, blas_int ldvsr, Scalar* work, blas_int lwork, blas_int* bwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sgges_(&jobvsl, &jobvsr, &sort, nullptr, &n, a, &lda, b, &ldb, &selected_dimension, alphar, alphai, beta, vsl,
           &ldvsl, vsr, &ldvsr, work, &lwork, bwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dgges_(&jobvsl, &jobvsr, &sort, nullptr, &n, a, &lda, b, &ldb, &selected_dimension, alphar, alphai, beta, vsl,
           &ldvsl, vsr, &ldvsr, work, &lwork, bwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char jobvsl_string[2] = {jobvsl, '\0'};
    char jobvsr_string[2] = {jobvsr, '\0'};
    char sort_string[2] = {sort, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_ldb = static_cast<mplapackint>(ldb);
    mplapackint mplapack_selected_dimension = static_cast<mplapackint>(selected_dimension);
    mplapackint mplapack_ldvsl = static_cast<mplapackint>(ldvsl);
    mplapackint mplapack_ldvsr = static_cast<mplapackint>(ldvsr);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_info = 0;
    std::size_t const bwork_size = static_cast<std::size_t>(std::max<blas_int>(1, n));
    auto mplapack_bwork = std::make_unique<bool[]>(bwork_size);
    Rgges(jobvsl_string, jobvsr_string, sort_string, nullptr, mplapack_n, a, mplapack_lda, b, mplapack_ldb,
          mplapack_selected_dimension, alphar, alphai, beta, vsl, mplapack_ldvsl, vsr, mplapack_ldvsr, work,
          mplapack_lwork, mplapack_bwork.get(), mplapack_info);
    selected_dimension = static_cast<blas_int>(mplapack_selected_dimension);
    for (std::size_t i = 0; i < bwork_size; ++i)
    {
      bwork[i] = mplapack_bwork[i] ? 1 : 0;
    }
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "real gges is available only for float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK gges received an invalid argument");
  }
  if (info > 0 && info <= n)
  {
    throw std::runtime_error("LAPACK gges failed to compute a generalized Schur form");
  }
  if (info > n)
  {
    throw std::runtime_error("LAPACK gges failed after generalized Schur convergence");
  }
}

template <uni20::LapackReal Scalar>
void gghrd(char compq, char compz, blas_int n, blas_int first, blas_int last, Scalar* a, blas_int lda, Scalar* b,
           blas_int ldb, Scalar* q, blas_int ldq, Scalar* z, blas_int ldz)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sgghrd_(&compq, &compz, &n, &first, &last, a, &lda, b, &ldb, q, &ldq, z, &ldz, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dgghrd_(&compq, &compz, &n, &first, &last, a, &lda, b, &ldb, q, &ldq, z, &ldz, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char compq_string[2] = {compq, '\0'};
    char compz_string[2] = {compz, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_first = static_cast<mplapackint>(first);
    mplapackint mplapack_last = static_cast<mplapackint>(last);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_ldb = static_cast<mplapackint>(ldb);
    mplapackint mplapack_ldq = static_cast<mplapackint>(ldq);
    mplapackint mplapack_ldz = static_cast<mplapackint>(ldz);
    mplapackint mplapack_info = 0;
    Rgghrd(compq_string, compz_string, mplapack_n, mplapack_first, mplapack_last, a, mplapack_lda, b, mplapack_ldb, q,
           mplapack_ldq, z, mplapack_ldz, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "real gghrd is available only for float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK gghrd received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK gghrd failed unexpectedly");
  }
}

template <uni20::LapackReal Scalar>
void hgeqz(char job, char compq, char compz, blas_int n, blas_int first, blas_int last, Scalar* h, blas_int ldh,
           Scalar* t, blas_int ldt, Scalar* alphar, Scalar* alphai, Scalar* beta, Scalar* q, blas_int ldq, Scalar* z,
           blas_int ldz, Scalar* work, blas_int lwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    shgeqz_(&job, &compq, &compz, &n, &first, &last, h, &ldh, t, &ldt, alphar, alphai, beta, q, &ldq, z, &ldz, work,
            &lwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dhgeqz_(&job, &compq, &compz, &n, &first, &last, h, &ldh, t, &ldt, alphar, alphai, beta, q, &ldq, z, &ldz, work,
            &lwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char job_string[2] = {job, '\0'};
    char compq_string[2] = {compq, '\0'};
    char compz_string[2] = {compz, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_first = static_cast<mplapackint>(first);
    mplapackint mplapack_last = static_cast<mplapackint>(last);
    mplapackint mplapack_ldh = static_cast<mplapackint>(ldh);
    mplapackint mplapack_ldt = static_cast<mplapackint>(ldt);
    mplapackint mplapack_ldq = static_cast<mplapackint>(ldq);
    mplapackint mplapack_ldz = static_cast<mplapackint>(ldz);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_info = 0;
    Rhgeqz(job_string, compq_string, compz_string, mplapack_n, mplapack_first, mplapack_last, h, mplapack_ldh, t,
           mplapack_ldt, alphar, alphai, beta, q, mplapack_ldq, z, mplapack_ldz, work, mplapack_lwork, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "real hgeqz is available only for float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK hgeqz received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK hgeqz failed to compute a generalized Schur form");
  }
}

template <uni20::LapackReal Scalar>
void tgexc(bool wantq, bool wantz, blas_int n, Scalar* a, blas_int lda, Scalar* b, blas_int ldb, Scalar* q,
           blas_int ldq, Scalar* z, blas_int ldz, blas_int& first, blas_int& last, Scalar* work, blas_int lwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    blas_int const fortran_wantq = wantq ? 1 : 0;
    blas_int const fortran_wantz = wantz ? 1 : 0;
    stgexc_(&fortran_wantq, &fortran_wantz, &n, a, &lda, b, &ldb, q, &ldq, z, &ldz, &first, &last, work, &lwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    blas_int const fortran_wantq = wantq ? 1 : 0;
    blas_int const fortran_wantz = wantz ? 1 : 0;
    dtgexc_(&fortran_wantq, &fortran_wantz, &n, a, &lda, b, &ldb, q, &ldq, z, &ldz, &first, &last, work, &lwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_ldb = static_cast<mplapackint>(ldb);
    mplapackint mplapack_ldq = static_cast<mplapackint>(ldq);
    mplapackint mplapack_ldz = static_cast<mplapackint>(ldz);
    mplapackint mplapack_first = static_cast<mplapackint>(first);
    mplapackint mplapack_last = static_cast<mplapackint>(last);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_info = 0;
    Rtgexc(wantq, wantz, mplapack_n, a, mplapack_lda, b, mplapack_ldb, q, mplapack_ldq, z, mplapack_ldz, mplapack_first,
           mplapack_last, work, mplapack_lwork, mplapack_info);
    first = static_cast<blas_int>(mplapack_first);
    last = static_cast<blas_int>(mplapack_last);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "real tgexc is available only for float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK tgexc received an invalid argument");
  }
  if (info == 1)
  {
    throw std::runtime_error("LAPACK tgexc could not swap adjacent generalized Schur blocks");
  }
  if (info > 1)
  {
    throw std::runtime_error("LAPACK tgexc failed unexpectedly");
  }
}

template <uni20::LapackReal Scalar>
void tgsen(blas_int ijob, bool wantq, bool wantz, blas_int* select, blas_int n, Scalar* a, blas_int lda, Scalar* b,
           blas_int ldb, Scalar* alphar, Scalar* alphai, Scalar* beta, Scalar* q, blas_int ldq, Scalar* z, blas_int ldz,
           blas_int& selected_dimension, Scalar& pl, Scalar& pr, Scalar* dif, Scalar* work, blas_int lwork,
           blas_int* iwork, blas_int liwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    blas_int const fortran_wantq = wantq ? 1 : 0;
    blas_int const fortran_wantz = wantz ? 1 : 0;
    stgsen_(&ijob, &fortran_wantq, &fortran_wantz, select, &n, a, &lda, b, &ldb, alphar, alphai, beta, q, &ldq, z, &ldz,
            &selected_dimension, &pl, &pr, dif, work, &lwork, iwork, &liwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    blas_int const fortran_wantq = wantq ? 1 : 0;
    blas_int const fortran_wantz = wantz ? 1 : 0;
    dtgsen_(&ijob, &fortran_wantq, &fortran_wantz, select, &n, a, &lda, b, &ldb, alphar, alphai, beta, q, &ldq, z, &ldz,
            &selected_dimension, &pl, &pr, dif, work, &lwork, iwork, &liwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    mplapackint mplapack_ijob = static_cast<mplapackint>(ijob);
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_ldb = static_cast<mplapackint>(ldb);
    mplapackint mplapack_ldq = static_cast<mplapackint>(ldq);
    mplapackint mplapack_ldz = static_cast<mplapackint>(ldz);
    mplapackint mplapack_m = static_cast<mplapackint>(selected_dimension);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_liwork = static_cast<mplapackint>(liwork);
    mplapackint mplapack_info = 0;
    std::size_t const select_size = static_cast<std::size_t>(std::max<blas_int>(1, n));
    std::size_t const iwork_size = static_cast<std::size_t>(std::max<blas_int>(1, liwork == -1 ? blas_int{1} : liwork));
    auto mplapack_select = std::make_unique<bool[]>(select_size);
    std::vector<mplapackint> mplapack_iwork(iwork_size);
    for (std::size_t i = 0; i < select_size; ++i)
    {
      mplapack_select[i] = select[i] != 0;
    }
    for (std::size_t i = 0; i < iwork_size; ++i)
    {
      mplapack_iwork[i] = static_cast<mplapackint>(iwork[i]);
    }
    Rtgsen(mplapack_ijob, wantq, wantz, mplapack_select.get(), mplapack_n, a, mplapack_lda, b, mplapack_ldb, alphar,
           alphai, beta, q, mplapack_ldq, z, mplapack_ldz, mplapack_m, pl, pr, dif, work, mplapack_lwork,
           mplapack_iwork.data(), mplapack_liwork, mplapack_info);
    selected_dimension = static_cast<blas_int>(mplapack_m);
    for (std::size_t i = 0; i < select_size; ++i)
    {
      select[i] = mplapack_select[i] ? 1 : 0;
    }
    for (std::size_t i = 0; i < iwork_size; ++i)
    {
      iwork[i] = static_cast<blas_int>(mplapack_iwork[i]);
    }
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "real tgsen is available only for float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK tgsen received an invalid argument");
  }
  if (info == 1)
  {
    throw std::runtime_error("LAPACK tgsen could not reorder the selected generalized Schur blocks");
  }
  if (info > 1)
  {
    throw std::runtime_error("LAPACK tgsen failed unexpectedly");
  }
}

template <uni20::LapackReal Scalar>
void tgevc(char side, char howmny, blas_int* select, blas_int n, Scalar* s, blas_int lds, Scalar* p, blas_int ldp,
           Scalar* vl, blas_int ldvl, Scalar* vr, blas_int ldvr, blas_int mm, blas_int& computed_vectors, Scalar* work)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    stgevc_(&side, &howmny, select, &n, s, &lds, p, &ldp, vl, &ldvl, vr, &ldvr, &mm, &computed_vectors, work, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dtgevc_(&side, &howmny, select, &n, s, &lds, p, &ldp, vl, &ldvl, vr, &ldvr, &mm, &computed_vectors, work, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char side_string[2] = {side, '\0'};
    char howmny_string[2] = {howmny, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lds = static_cast<mplapackint>(lds);
    mplapackint mplapack_ldp = static_cast<mplapackint>(ldp);
    mplapackint mplapack_ldvl = static_cast<mplapackint>(ldvl);
    mplapackint mplapack_ldvr = static_cast<mplapackint>(ldvr);
    mplapackint mplapack_mm = static_cast<mplapackint>(mm);
    mplapackint mplapack_m = static_cast<mplapackint>(computed_vectors);
    mplapackint mplapack_info = 0;
    std::size_t const select_size = static_cast<std::size_t>(std::max<blas_int>(1, n));
    auto mplapack_select = std::make_unique<bool[]>(select_size);
    for (std::size_t i = 0; i < select_size; ++i)
    {
      mplapack_select[i] = select[i] != 0;
    }
    Rtgevc(side_string, howmny_string, mplapack_select.get(), mplapack_n, s, mplapack_lds, p, mplapack_ldp, vl,
           mplapack_ldvl, vr, mplapack_ldvr, mplapack_mm, mplapack_m, work, mplapack_info);
    computed_vectors = static_cast<blas_int>(mplapack_m);
    for (std::size_t i = 0; i < select_size; ++i)
    {
      select[i] = mplapack_select[i] ? 1 : 0;
    }
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "real tgevc is available only for float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK tgevc received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK tgevc failed to compute generalized Schur eigenvectors");
  }
}

template <uni20::LapackReal Scalar>
void tgsna(char job, char howmny, blas_int* select, blas_int n, Scalar* a, blas_int lda, Scalar* b, blas_int ldb,
           Scalar* vl, blas_int ldvl, Scalar* vr, blas_int ldvr, Scalar* reciprocal_eigenvalue_condition_numbers,
           Scalar* reciprocal_eigenvector_condition_numbers, blas_int mm, blas_int& computed_estimates, Scalar* work,
           blas_int lwork, blas_int* iwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    stgsna_(&job, &howmny, select, &n, a, &lda, b, &ldb, vl, &ldvl, vr, &ldvr, reciprocal_eigenvalue_condition_numbers,
            reciprocal_eigenvector_condition_numbers, &mm, &computed_estimates, work, &lwork, iwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dtgsna_(&job, &howmny, select, &n, a, &lda, b, &ldb, vl, &ldvl, vr, &ldvr, reciprocal_eigenvalue_condition_numbers,
            reciprocal_eigenvector_condition_numbers, &mm, &computed_estimates, work, &lwork, iwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char job_string[2] = {job, '\0'};
    char howmny_string[2] = {howmny, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_ldb = static_cast<mplapackint>(ldb);
    mplapackint mplapack_ldvl = static_cast<mplapackint>(ldvl);
    mplapackint mplapack_ldvr = static_cast<mplapackint>(ldvr);
    mplapackint mplapack_mm = static_cast<mplapackint>(mm);
    mplapackint mplapack_m = static_cast<mplapackint>(computed_estimates);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_info = 0;
    std::size_t const select_size = static_cast<std::size_t>(std::max<blas_int>(1, n));
    std::size_t const iwork_size = static_cast<std::size_t>(std::max<blas_int>(1, n + 6));
    auto mplapack_select = std::make_unique<bool[]>(select_size);
    std::vector<mplapackint> mplapack_iwork(iwork_size);
    for (std::size_t i = 0; i < select_size; ++i)
    {
      mplapack_select[i] = select[i] != 0;
    }
    for (std::size_t i = 0; i < iwork_size; ++i)
    {
      mplapack_iwork[i] = static_cast<mplapackint>(iwork[i]);
    }
    Rtgsna(job_string, howmny_string, mplapack_select.get(), mplapack_n, a, mplapack_lda, b, mplapack_ldb, vl,
           mplapack_ldvl, vr, mplapack_ldvr, reciprocal_eigenvalue_condition_numbers,
           reciprocal_eigenvector_condition_numbers, mplapack_mm, mplapack_m, work, mplapack_lwork,
           mplapack_iwork.data(), mplapack_info);
    computed_estimates = static_cast<blas_int>(mplapack_m);
    for (std::size_t i = 0; i < select_size; ++i)
    {
      select[i] = mplapack_select[i] ? 1 : 0;
    }
    for (std::size_t i = 0; i < iwork_size; ++i)
    {
      iwork[i] = static_cast<blas_int>(mplapack_iwork[i]);
    }
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "real tgsna is available only for float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK tgsna received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK tgsna failed unexpectedly");
  }
}

template <uni20::LapackReal Real>
void geev(char jobvl, char jobvr, blas_int n, uni20::complex<Real>* a, blas_int lda, uni20::complex<Real>* w,
          uni20::complex<Real>* vl, blas_int ldvl, uni20::complex<Real>* vr, blas_int ldvr, uni20::complex<Real>* work,
          blas_int lwork, Real* rwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Real, float>)
  {
    cgeev_(&jobvl, &jobvr, &n, a, &lda, w, vl, &ldvl, vr, &ldvr, work, &lwork, rwork, &info);
  }
  else if constexpr (std::is_same_v<Real, double>)
  {
    zgeev_(&jobvl, &jobvr, &n, a, &lda, w, vl, &ldvl, vr, &ldvr, work, &lwork, rwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Real>)
  {
    char jobvl_string[2] = {jobvl, '\0'};
    char jobvr_string[2] = {jobvr, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_ldvl = static_cast<mplapackint>(ldvl);
    mplapackint mplapack_ldvr = static_cast<mplapackint>(ldvr);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_info = 0;
    Cgeev(jobvl_string, jobvr_string, mplapack_n, a, mplapack_lda, w, vl, mplapack_ldvl, vr, mplapack_ldvr, work,
          mplapack_lwork, rwork, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Real, float> || std::is_same_v<Real, double> || is_mplapack_binary128_v<Real>,
                  "complex geev is available only for complex float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK complex geev received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK complex geev failed to converge");
  }
}

template <uni20::LapackReal Scalar>
void gees(char jobvs, char sort, blas_int n, Scalar* a, blas_int lda, blas_int& selected_dimension, Scalar* wr,
          Scalar* wi, Scalar* vs, blas_int ldvs, Scalar* work, blas_int lwork, blas_int* bwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    sgees_(&jobvs, &sort, nullptr, &n, a, &lda, &selected_dimension, wr, wi, vs, &ldvs, work, &lwork, bwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dgees_(&jobvs, &sort, nullptr, &n, a, &lda, &selected_dimension, wr, wi, vs, &ldvs, work, &lwork, bwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_ldvs = static_cast<mplapackint>(ldvs);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_selected_dimension = static_cast<mplapackint>(selected_dimension);
    mplapackint mplapack_info = 0;
    auto mplapack_bwork = std::make_unique<bool[]>(static_cast<std::size_t>(std::max<blas_int>(1, n)));
    char jobvs_string[2] = {jobvs, '\0'};
    char sort_string[2] = {sort, '\0'};
    Rgees(jobvs_string, sort_string, nullptr, mplapack_n, a, mplapack_lda, mplapack_selected_dimension, wr, wi, vs,
          mplapack_ldvs, work, mplapack_lwork, mplapack_bwork.get(), mplapack_info);
    selected_dimension = static_cast<blas_int>(mplapack_selected_dimension);
    info = static_cast<blas_int>(mplapack_info);
    if (bwork != nullptr)
    {
      for (blas_int i = 0; i < std::max<blas_int>(1, n); ++i)
      {
        bwork[i] = mplapack_bwork[static_cast<std::size_t>(i)] ? 1 : 0;
      }
    }
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "gees is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK gees received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK gees failed to compute a Schur form");
  }
}

template <uni20::LapackReal Scalar>
void hseqr(char job, char compz, blas_int n, blas_int first, blas_int last, Scalar* h, blas_int ldh, Scalar* wr,
           Scalar* wi, Scalar* z, blas_int ldz, Scalar* work, blas_int lwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    shseqr_(&job, &compz, &n, &first, &last, h, &ldh, wr, wi, z, &ldz, work, &lwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dhseqr_(&job, &compz, &n, &first, &last, h, &ldh, wr, wi, z, &ldz, work, &lwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char job_string[2] = {job, '\0'};
    char compz_string[2] = {compz, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_first = static_cast<mplapackint>(first);
    mplapackint mplapack_last = static_cast<mplapackint>(last);
    mplapackint mplapack_ldh = static_cast<mplapackint>(ldh);
    mplapackint mplapack_ldz = static_cast<mplapackint>(ldz);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_info = 0;
    Rhseqr(job_string, compz_string, mplapack_n, mplapack_first, mplapack_last, h, mplapack_ldh, wr, wi, z,
           mplapack_ldz, work, mplapack_lwork, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "real hseqr is available only for float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK hseqr received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK hseqr failed to compute a Schur form");
  }
}

template <uni20::LapackReal Real>
void gees(char jobvs, char sort, blas_int n, uni20::complex<Real>* a, blas_int lda, blas_int& selected_dimension,
          uni20::complex<Real>* w, uni20::complex<Real>* vs, blas_int ldvs, uni20::complex<Real>* work, blas_int lwork,
          Real* rwork, blas_int* bwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Real, float>)
  {
    cgees_(&jobvs, &sort, nullptr, &n, a, &lda, &selected_dimension, w, vs, &ldvs, work, &lwork, rwork, bwork, &info);
  }
  else if constexpr (std::is_same_v<Real, double>)
  {
    zgees_(&jobvs, &sort, nullptr, &n, a, &lda, &selected_dimension, w, vs, &ldvs, work, &lwork, rwork, bwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Real>)
  {
    char jobvs_string[2] = {jobvs, '\0'};
    char sort_string[2] = {sort, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_lda = static_cast<mplapackint>(lda);
    mplapackint mplapack_ldvs = static_cast<mplapackint>(ldvs);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_selected_dimension = static_cast<mplapackint>(selected_dimension);
    mplapackint mplapack_info = 0;
    auto mplapack_bwork = std::make_unique<bool[]>(static_cast<std::size_t>(std::max<blas_int>(1, n)));
    Cgees(jobvs_string, sort_string, nullptr, mplapack_n, a, mplapack_lda, mplapack_selected_dimension, w, vs,
          mplapack_ldvs, work, mplapack_lwork, rwork, mplapack_bwork.get(), mplapack_info);
    selected_dimension = static_cast<blas_int>(mplapack_selected_dimension);
    info = static_cast<blas_int>(mplapack_info);
    if (bwork != nullptr)
    {
      for (blas_int i = 0; i < std::max<blas_int>(1, n); ++i)
      {
        bwork[i] = mplapack_bwork[static_cast<std::size_t>(i)] ? 1 : 0;
      }
    }
  }
#endif
  else
  {
    static_assert(std::is_same_v<Real, float> || std::is_same_v<Real, double> || is_mplapack_binary128_v<Real>,
                  "complex gees is available only for complex float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK complex gees received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK complex gees failed to compute a Schur form");
  }
}

template <uni20::LapackReal Scalar>
void trexc(char compq, blas_int n, Scalar* t, blas_int ldt, Scalar* q, blas_int ldq, blas_int& first, blas_int& last,
           Scalar* work)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    strexc_(&compq, &n, t, &ldt, q, &ldq, &first, &last, work, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dtrexc_(&compq, &n, t, &ldt, q, &ldq, &first, &last, work, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_ldt = static_cast<mplapackint>(ldt);
    mplapackint mplapack_ldq = static_cast<mplapackint>(ldq);
    mplapackint mplapack_first = static_cast<mplapackint>(first);
    mplapackint mplapack_last = static_cast<mplapackint>(last);
    mplapackint mplapack_info = 0;
    char compq_string[2] = {compq, '\0'};
    Rtrexc(compq_string, mplapack_n, t, mplapack_ldt, q, mplapack_ldq, mplapack_first, mplapack_last, work,
           mplapack_info);
    first = static_cast<blas_int>(mplapack_first);
    last = static_cast<blas_int>(mplapack_last);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "trexc is available only for real float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK trexc received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK trexc failed to reorder adjacent Schur blocks");
  }
}

template <uni20::LapackReal Scalar>
void trsen(char job, char compq, blas_int* select, blas_int n, Scalar* t, blas_int ldt, Scalar* q, blas_int ldq,
           Scalar* wr, Scalar* wi, blas_int& selected_dimension, Scalar& reciprocal_eigenvalue_cluster_condition,
           Scalar& reciprocal_invariant_subspace_condition, Scalar* work, blas_int lwork, blas_int* iwork,
           blas_int liwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    strsen_(&job, &compq, select, &n, t, &ldt, q, &ldq, wr, wi, &selected_dimension,
            &reciprocal_eigenvalue_cluster_condition, &reciprocal_invariant_subspace_condition, work, &lwork, iwork,
            &liwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dtrsen_(&job, &compq, select, &n, t, &ldt, q, &ldq, wr, wi, &selected_dimension,
            &reciprocal_eigenvalue_cluster_condition, &reciprocal_invariant_subspace_condition, work, &lwork, iwork,
            &liwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char job_string[2] = {job, '\0'};
    char compq_string[2] = {compq, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_ldt = static_cast<mplapackint>(ldt);
    mplapackint mplapack_ldq = static_cast<mplapackint>(ldq);
    mplapackint mplapack_m = static_cast<mplapackint>(selected_dimension);
    mplapackint mplapack_lwork = static_cast<mplapackint>(lwork);
    mplapackint mplapack_liwork = static_cast<mplapackint>(liwork);
    mplapackint mplapack_info = 0;
    std::size_t const select_size = static_cast<std::size_t>(std::max<blas_int>(1, n));
    std::size_t const iwork_size = static_cast<std::size_t>(std::max<blas_int>(1, liwork == -1 ? blas_int{1} : liwork));
    auto mplapack_select = std::make_unique<bool[]>(select_size);
    std::vector<mplapackint> mplapack_iwork(iwork_size);
    for (std::size_t i = 0; i < select_size; ++i)
    {
      mplapack_select[i] = select[i] != 0;
    }
    for (std::size_t i = 0; i < iwork_size; ++i)
    {
      mplapack_iwork[i] = static_cast<mplapackint>(iwork[i]);
    }
    Rtrsen(job_string, compq_string, mplapack_select.get(), mplapack_n, t, mplapack_ldt, q, mplapack_ldq, wr, wi,
           mplapack_m, reciprocal_eigenvalue_cluster_condition, reciprocal_invariant_subspace_condition, work,
           mplapack_lwork, mplapack_iwork.data(), mplapack_liwork, mplapack_info);
    selected_dimension = static_cast<blas_int>(mplapack_m);
    for (std::size_t i = 0; i < select_size; ++i)
    {
      select[i] = mplapack_select[i] ? 1 : 0;
    }
    for (std::size_t i = 0; i < iwork_size; ++i)
    {
      iwork[i] = static_cast<blas_int>(mplapack_iwork[i]);
    }
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "real trsen is available only for float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK trsen received an invalid argument");
  }
  if (info == 1)
  {
    throw std::runtime_error("LAPACK trsen could not reorder the selected Schur blocks");
  }
  if (info > 1)
  {
    throw std::runtime_error("LAPACK trsen failed unexpectedly");
  }
}

template <uni20::LapackReal Scalar>
void trevc(char side, char howmny, blas_int* select, blas_int n, Scalar* t, blas_int ldt, Scalar* vl, blas_int ldvl,
           Scalar* vr, blas_int ldvr, blas_int mm, blas_int& computed_vectors, Scalar* work)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    strevc_(&side, &howmny, select, &n, t, &ldt, vl, &ldvl, vr, &ldvr, &mm, &computed_vectors, work, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dtrevc_(&side, &howmny, select, &n, t, &ldt, vl, &ldvl, vr, &ldvr, &mm, &computed_vectors, work, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char side_string[2] = {side, '\0'};
    char howmny_string[2] = {howmny, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_ldt = static_cast<mplapackint>(ldt);
    mplapackint mplapack_ldvl = static_cast<mplapackint>(ldvl);
    mplapackint mplapack_ldvr = static_cast<mplapackint>(ldvr);
    mplapackint mplapack_mm = static_cast<mplapackint>(mm);
    mplapackint mplapack_m = static_cast<mplapackint>(computed_vectors);
    mplapackint mplapack_info = 0;
    std::size_t const select_size = static_cast<std::size_t>(std::max<blas_int>(1, n));
    auto mplapack_select = std::make_unique<bool[]>(select_size);
    for (std::size_t i = 0; i < select_size; ++i)
    {
      mplapack_select[i] = select[i] != 0;
    }
    Rtrevc(side_string, howmny_string, mplapack_select.get(), mplapack_n, t, mplapack_ldt, vl, mplapack_ldvl, vr,
           mplapack_ldvr, mplapack_mm, mplapack_m, work, mplapack_info);
    computed_vectors = static_cast<blas_int>(mplapack_m);
    for (std::size_t i = 0; i < select_size; ++i)
    {
      select[i] = mplapack_select[i] ? 1 : 0;
    }
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "real trevc is available only for float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK trevc received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK trevc failed to compute all requested Schur eigenvectors");
  }
}

template <uni20::LapackReal Scalar>
void trsna(char job, char howmny, blas_int* select, blas_int n, Scalar* t, blas_int ldt, Scalar* vl, blas_int ldvl,
           Scalar* vr, blas_int ldvr, Scalar* reciprocal_eigenvalue_condition_numbers,
           Scalar* reciprocal_eigenvector_condition_numbers, blas_int mm, blas_int& computed_estimates, Scalar* work,
           blas_int ldwork, blas_int* iwork)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Scalar, float>)
  {
    strsna_(&job, &howmny, select, &n, t, &ldt, vl, &ldvl, vr, &ldvr, reciprocal_eigenvalue_condition_numbers,
            reciprocal_eigenvector_condition_numbers, &mm, &computed_estimates, work, &ldwork, iwork, &info);
  }
  else if constexpr (std::is_same_v<Scalar, double>)
  {
    dtrsna_(&job, &howmny, select, &n, t, &ldt, vl, &ldvl, vr, &ldvr, reciprocal_eigenvalue_condition_numbers,
            reciprocal_eigenvector_condition_numbers, &mm, &computed_estimates, work, &ldwork, iwork, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Scalar>)
  {
    char job_string[2] = {job, '\0'};
    char howmny_string[2] = {howmny, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_ldt = static_cast<mplapackint>(ldt);
    mplapackint mplapack_ldvl = static_cast<mplapackint>(ldvl);
    mplapackint mplapack_ldvr = static_cast<mplapackint>(ldvr);
    mplapackint mplapack_mm = static_cast<mplapackint>(mm);
    mplapackint mplapack_m = static_cast<mplapackint>(computed_estimates);
    mplapackint mplapack_ldwork = static_cast<mplapackint>(ldwork);
    mplapackint mplapack_info = 0;
    std::size_t const select_size = static_cast<std::size_t>(std::max<blas_int>(1, n));
    std::size_t const iwork_size = static_cast<std::size_t>(std::max<blas_int>(1, 2 * std::max<blas_int>(1, n) - 2));
    auto mplapack_select = std::make_unique<bool[]>(select_size);
    std::vector<mplapackint> mplapack_iwork(iwork_size);
    for (std::size_t i = 0; i < select_size; ++i)
    {
      mplapack_select[i] = select[i] != 0;
    }
    for (std::size_t i = 0; i < iwork_size; ++i)
    {
      mplapack_iwork[i] = static_cast<mplapackint>(iwork[i]);
    }
    Rtrsna(job_string, howmny_string, mplapack_select.get(), mplapack_n, t, mplapack_ldt, vl, mplapack_ldvl, vr,
           mplapack_ldvr, reciprocal_eigenvalue_condition_numbers, reciprocal_eigenvector_condition_numbers,
           mplapack_mm, mplapack_m, work, mplapack_ldwork, mplapack_iwork.data(), mplapack_info);
    computed_estimates = static_cast<blas_int>(mplapack_m);
    for (std::size_t i = 0; i < select_size; ++i)
    {
      select[i] = mplapack_select[i] ? 1 : 0;
    }
    for (std::size_t i = 0; i < iwork_size; ++i)
    {
      iwork[i] = static_cast<blas_int>(mplapack_iwork[i]);
    }
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> || is_mplapack_binary128_v<Scalar>,
                  "real trsna is available only for float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK trsna received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK trsna failed unexpectedly");
  }
}

template <uni20::LapackReal Real>
void trexc(char compq, blas_int n, uni20::complex<Real>* t, blas_int ldt, uni20::complex<Real>* q, blas_int ldq,
           blas_int& first, blas_int& last)
{
  blas_int info = 0;
  if constexpr (std::is_same_v<Real, float>)
  {
    ctrexc_(&compq, &n, t, &ldt, q, &ldq, &first, &last, &info);
  }
  else if constexpr (std::is_same_v<Real, double>)
  {
    ztrexc_(&compq, &n, t, &ldt, q, &ldq, &first, &last, &info);
  }
#if defined(UNI20_ENABLE_MPLAPACK) && defined(MPLAPACK_BINARY128_MODE) &&                                              \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  else if constexpr (is_mplapack_binary128_v<Real>)
  {
    char compq_string[2] = {compq, '\0'};
    mplapackint mplapack_n = static_cast<mplapackint>(n);
    mplapackint mplapack_ldt = static_cast<mplapackint>(ldt);
    mplapackint mplapack_ldq = static_cast<mplapackint>(ldq);
    mplapackint mplapack_first = static_cast<mplapackint>(first);
    mplapackint mplapack_last = static_cast<mplapackint>(last);
    mplapackint mplapack_info = 0;
    Ctrexc(compq_string, mplapack_n, t, mplapack_ldt, q, mplapack_ldq, mplapack_first, mplapack_last, mplapack_info);
    info = static_cast<blas_int>(mplapack_info);
  }
#endif
  else
  {
    static_assert(std::is_same_v<Real, float> || std::is_same_v<Real, double> || is_mplapack_binary128_v<Real>,
                  "complex trexc is available only for complex float, double, and enabled MPLAPACK binary128");
  }

  if (info < 0)
  {
    throw std::invalid_argument("LAPACK complex trexc received an invalid argument");
  }
  if (info > 0)
  {
    throw std::runtime_error("LAPACK complex trexc failed to reorder adjacent Schur blocks");
  }
}
} // namespace detail

/// \brief Copy a local dense vector.
/// \tparam Scalar Element type.
/// \param source Source vector.
/// \param destination Destination vector overwritten with \p source.
template <typename Scalar> void copy(std::span<Scalar const> source, std::span<Scalar> destination)
{
  detail::check_same_size(source, destination);
  std::ranges::copy(source, destination.begin());
}

/// \brief Scale a local dense vector in place.
/// \tparam Scalar Element type.
/// \param alpha Scale factor.
/// \param vector Vector to scale.
template <typename Scalar> void scal(Scalar const& alpha, std::span<Scalar> vector)
{
  for (Scalar& value : vector)
  {
    value *= alpha;
  }
}

/// \brief Add a scaled local dense vector to another vector.
/// \tparam Scalar Element type.
/// \param alpha Scale factor.
/// \param x Source vector.
/// \param y Destination vector updated as `y += alpha * x`.
template <typename Scalar> void axpy(Scalar const& alpha, std::span<Scalar const> x, std::span<Scalar> y)
{
  detail::check_same_size(x, y);
  for (std::size_t i = 0; i < x.size(); ++i)
  {
    y[i] += alpha * x[i];
  }
}

/// \brief Compute an unconjugated dot product.
/// \tparam Scalar Element type.
/// \param x Left vector.
/// \param y Right vector.
/// \return Sum of `x[i] * y[i]`.
template <typename Scalar> Scalar dotu(std::span<Scalar const> x, std::span<Scalar const> y)
{
  if (x.size() != y.size())
  {
    throw std::invalid_argument("dense vector sizes do not agree");
  }

  Scalar result{};
  for (std::size_t i = 0; i < x.size(); ++i)
  {
    result += x[i] * y[i];
  }
  return result;
}

/// \brief Compute a BLAS-style real dot product.
/// \tparam Scalar Real scalar type.
/// \param x Left vector.
/// \param y Right vector.
/// \return Sum of `x[i] * y[i]`.
template <uni20::Real Scalar> Scalar dot(std::span<Scalar const> x, std::span<Scalar const> y) { return dotu(x, y); }

/// \brief Compute a conjugated dot product.
/// \tparam Scalar Element type.
/// \param x Left vector, conjugated elementwise for complex scalar types.
/// \param y Right vector.
/// \return Sum of `conj(x[i]) * y[i]`.
template <typename Scalar> Scalar dotc(std::span<Scalar const> x, std::span<Scalar const> y)
{
  if (x.size() != y.size())
  {
    throw std::invalid_argument("dense vector sizes do not agree");
  }

  Scalar result{};
  for (std::size_t i = 0; i < x.size(); ++i)
  {
    result += detail::conjugate_if_complex(x[i]) * y[i];
  }
  return result;
}

/// \brief Compute a Euclidean vector norm.
/// \tparam Scalar Element type.
/// \param vector Input vector.
/// \return Square root of the sum of squared magnitudes.
template <typename Scalar> accumulation_real_t<Scalar> nrm2(std::span<Scalar const> vector)
{
  using Real = accumulation_real_t<Scalar>;

  Real sum{};
  for (Scalar const& value : vector)
  {
    Real const magnitude = static_cast<Real>(std::abs(value));
    sum += magnitude * magnitude;
  }
  return std::sqrt(sum);
}

/// \brief Copy a selected region of a local dense matrix.
/// \tparam Scalar Element type.
/// \param source Source matrix.
/// \param destination Destination matrix with the same shape as \p source.
/// \param fill Region to copy.
template <typename Scalar> void lacpy(Matrix<Scalar> const& source, Matrix<Scalar>& destination, MatrixFill fill)
{
  if (source.rows() != destination.rows() || source.cols() != destination.cols())
  {
    throw std::invalid_argument("dense matrix sizes do not agree");
  }

  for (std::size_t col = 0; col < source.cols(); ++col)
  {
    for (std::size_t row = 0; row < source.rows(); ++row)
    {
      if (detail::in_selected_region<Scalar>(row, col, fill))
      {
        destination(row, col) = source(row, col);
      }
    }
  }
}

/// \brief Set a selected region of a local dense matrix.
/// \tparam Scalar Element type.
/// \param matrix Matrix to update.
/// \param diagonal Value written on the diagonal.
/// \param off_diagonal Value written away from the diagonal.
/// \param fill Region to set.
template <typename Scalar>
void laset(Matrix<Scalar>& matrix, Scalar const& diagonal, Scalar const& off_diagonal, MatrixFill fill)
{
  for (std::size_t col = 0; col < matrix.cols(); ++col)
  {
    for (std::size_t row = 0; row < matrix.rows(); ++row)
    {
      if (detail::in_selected_region<Scalar>(row, col, fill))
      {
        matrix(row, col) = (row == col) ? diagonal : off_diagonal;
      }
    }
  }
}

/// \brief Compute a dense matrix-vector product.
/// \tparam Scalar Element type.
/// \param alpha Scale factor for the matrix-vector product.
/// \param matrix Input matrix.
/// \param x Input vector.
/// \param beta Scale factor for the original \p y value.
/// \param y Output vector updated as `y = alpha * op(matrix) * x + beta * y`.
/// \param transpose Matrix operation applied before multiplication.
template <typename Scalar>
void gemv(Scalar const& alpha, Matrix<Scalar> const& matrix, std::span<Scalar const> x, Scalar const& beta,
          std::span<Scalar> y, MatrixTranspose transpose = MatrixTranspose::None)
{
  std::size_t const output_size = transpose == MatrixTranspose::None ? matrix.rows() : matrix.cols();
  std::size_t const input_size = transpose == MatrixTranspose::None ? matrix.cols() : matrix.rows();
  if (x.size() != input_size || y.size() != output_size)
  {
    throw std::invalid_argument("dense matrix-vector sizes do not agree");
  }

  for (Scalar& value : y)
  {
    value *= beta;
  }

  if (transpose == MatrixTranspose::None)
  {
    for (std::size_t col = 0; col < matrix.cols(); ++col)
    {
      Scalar const factor = alpha * x[col];
      for (std::size_t row = 0; row < matrix.rows(); ++row)
      {
        y[row] += factor * matrix(row, col);
      }
    }
    return;
  }

  for (std::size_t col = 0; col < matrix.cols(); ++col)
  {
    Scalar sum{};
    for (std::size_t row = 0; row < matrix.rows(); ++row)
    {
      Scalar entry = matrix(row, col);
      if (transpose == MatrixTranspose::ConjugateTranspose)
      {
        entry = detail::conjugate_if_complex(entry);
      }
      sum += entry * x[row];
    }
    y[col] += alpha * sum;
  }
}

/// \brief Apply a dense rank-one update.
/// \tparam Scalar Element type.
/// \param alpha Scale factor.
/// \param x Left vector.
/// \param y Right vector.
/// \param matrix Matrix updated as `matrix += alpha * x * y^T`.
template <typename Scalar>
void geru(Scalar const& alpha, std::span<Scalar const> x, std::span<Scalar const> y, Matrix<Scalar>& matrix)
{
  if (x.size() != matrix.rows() || y.size() != matrix.cols())
  {
    throw std::invalid_argument("dense rank-one update sizes do not agree");
  }

  for (std::size_t col = 0; col < matrix.cols(); ++col)
  {
    Scalar const factor = alpha * y[col];
    for (std::size_t row = 0; row < matrix.rows(); ++row)
    {
      matrix(row, col) += x[row] * factor;
    }
  }
}

/// \brief Apply a conjugated dense rank-one update.
/// \tparam Scalar Element type.
/// \param alpha Scale factor.
/// \param x Left vector.
/// \param y Right vector, conjugated elementwise for complex scalar types.
/// \param matrix Matrix updated as `matrix += alpha * x * conj(y)^T`.
template <typename Scalar>
void gerc(Scalar const& alpha, std::span<Scalar const> x, std::span<Scalar const> y, Matrix<Scalar>& matrix)
{
  if (x.size() != matrix.rows() || y.size() != matrix.cols())
  {
    throw std::invalid_argument("dense rank-one update sizes do not agree");
  }

  for (std::size_t col = 0; col < matrix.cols(); ++col)
  {
    Scalar const factor = alpha * detail::conjugate_if_complex(y[col]);
    for (std::size_t row = 0; row < matrix.rows(); ++row)
    {
      matrix(row, col) += x[row] * factor;
    }
  }
}

} // namespace uni20::krylov
