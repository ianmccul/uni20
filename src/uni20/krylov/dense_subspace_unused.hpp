#pragma once

#include <uni20/krylov/dense_subspace.hpp>

namespace uni20::krylov
{

/// \brief Dense LAPACK wrappers parked outside the default Krylov dense-subspace API.
/// \details This header is intentionally opt-in. It keeps prototype wrappers tested
///          while avoiding default exposure from `dense_subspace.hpp`.

namespace detail
{

/// \brief Column-major host matrix used by the direct LAPACK wrapper inventory.
/// \details These quarantined helpers pass matrix storage directly to LAPACK and
///          therefore encode the provider layout instead of relying on the
///          default layout of `uni20::DenseMatrix`.
template <typename Scalar> using ColumnMajorLapackMatrix = uni20::DenseMatrix<Scalar, uni20::ColumnMajor>;

} // namespace detail

template <uni20::LapackReal Scalar> struct RealSymmetricEigensystem
{
    std::vector<Scalar> eigenvalues;
    detail::ColumnMajorLapackMatrix<Scalar> eigenvectors;
};

/// \brief Eigenvalues and optional eigenvectors of a dense complex Hermitian matrix.
/// \tparam Real Underlying real precision.
template <uni20::LapackComplexReal Real> struct ComplexHermitianEigensystem
{
    std::vector<Real> eigenvalues;
    detail::ColumnMajorLapackMatrix<uni20::complex<Real>> eigenvectors;
};

/// \brief Singular values and optional singular vectors of a dense real matrix.
/// \tparam Scalar Real scalar type.
template <uni20::LapackReal Scalar> struct RealSingularValueDecomposition
{
    std::vector<Scalar> singular_values;
    detail::ColumnMajorLapackMatrix<Scalar> left_singular_vectors;
    detail::ColumnMajorLapackMatrix<Scalar> right_singular_vectors_transpose;
};

/// \brief LU factorization of a dense real square matrix.
/// \tparam Scalar Real scalar type.
template <uni20::LapackReal Scalar> struct RealLuFactorization
{
    detail::ColumnMajorLapackMatrix<Scalar> factors;
    std::vector<std::size_t> pivot_rows;
};

/// \brief LAPACK general band storage for a real square matrix.
/// \details `storage` uses LAPACK `gbtrf`/`gbtrs` factor-storage layout with
///          `2 * lower_bandwidth + upper_bandwidth + 1` rows and `order`
///          columns. Original matrix entries are stored at row
///          `lower_bandwidth + upper_bandwidth + i - j` in column `j`.
/// \tparam Scalar Real scalar type.
template <uni20::LapackReal Scalar> struct RealGeneralBandMatrix
{
    detail::ColumnMajorLapackMatrix<Scalar> storage;
    std::size_t order = 0;
    std::size_t lower_bandwidth = 0;
    std::size_t upper_bandwidth = 0;
};

/// \brief LU factorization of a real square general band matrix.
/// \tparam Scalar Real scalar type.
template <uni20::LapackReal Scalar> struct RealGeneralBandFactorization
{
    RealGeneralBandMatrix<Scalar> factors;
    std::vector<std::size_t> pivot_rows;
};

/// \brief LAPACK general tridiagonal storage for a real square matrix.
/// \details `diagonal` has length `n`; `lower_diagonal` and `upper_diagonal`
///          have length `n - 1`.
/// \tparam Scalar Real scalar type.
template <uni20::LapackReal Scalar> struct RealGeneralTridiagonalMatrix
{
    std::vector<Scalar> lower_diagonal;
    std::vector<Scalar> diagonal;
    std::vector<Scalar> upper_diagonal;

    [[nodiscard]] std::size_t order() const noexcept { return diagonal.size(); }
};

/// \brief LU factorization of a real square general tridiagonal matrix.
/// \details Uses LAPACK `gttrf` storage. `second_upper_diagonal` has length
///          `n - 2` when `n >= 2`; `pivot_rows` stores LAPACK row pivots as
///          zero-based row indices.
/// \tparam Scalar Real scalar type.
template <uni20::LapackReal Scalar> struct RealGeneralTridiagonalFactorization
{
    RealGeneralTridiagonalMatrix<Scalar> factors;
    std::vector<Scalar> second_upper_diagonal;
    std::vector<std::size_t> pivot_rows;
};

/// \brief Expert real general-tridiagonal linear solve result.
/// \tparam Scalar Real scalar type.
template <uni20::LapackReal Scalar> struct RealGeneralTridiagonalExpertLinearSolve
{
    detail::ColumnMajorLapackMatrix<Scalar> solution;
    RealGeneralTridiagonalFactorization<Scalar> factorization;
    Scalar reciprocal_condition = Scalar{};
    std::vector<Scalar> forward_error_bounds;
    std::vector<Scalar> backward_error_bounds;
    bool reciprocal_condition_below_machine_precision = false;
};

/// \brief Expert real general-band linear solve result.
/// \tparam Scalar Real scalar type.
template <uni20::LapackReal Scalar> struct RealGeneralBandExpertLinearSolve
{
    detail::ColumnMajorLapackMatrix<Scalar> solution;
    RealGeneralBandMatrix<Scalar> factors;
    std::vector<std::size_t> pivot_rows;
    Scalar reciprocal_condition = Scalar{};
    std::vector<Scalar> forward_error_bounds;
    std::vector<Scalar> backward_error_bounds;
    char equilibration = 'N';
    bool reciprocal_condition_below_machine_precision = false;
};

/// \brief Expert dense real linear solve result.
/// \tparam Scalar Real scalar type.
template <uni20::LapackReal Scalar> struct RealExpertLinearSolve
{
    detail::ColumnMajorLapackMatrix<Scalar> solution;
    detail::ColumnMajorLapackMatrix<Scalar> factors;
    std::vector<std::size_t> pivot_rows;
    Scalar reciprocal_condition = Scalar{};
    std::vector<Scalar> forward_error_bounds;
    std::vector<Scalar> backward_error_bounds;
    char equilibration = 'N';
    bool reciprocal_condition_below_machine_precision = false;
};

/// \brief Dense real refined linear solve result.
/// \tparam Scalar Real scalar type.
template <uni20::LapackReal Scalar> struct RealRefinedLinearSolve
{
    detail::ColumnMajorLapackMatrix<Scalar> solution;
    std::vector<Scalar> forward_error_bounds;
    std::vector<Scalar> backward_error_bounds;
};

/// \brief Dense real general matrix equilibration factors.
/// \tparam Scalar Real scalar type.
template <uni20::LapackReal Scalar> struct RealEquilibration
{
    std::vector<Scalar> row_scale;
    std::vector<Scalar> column_scale;
    Scalar row_condition = Scalar{1};
    Scalar column_condition = Scalar{1};
    Scalar max_abs = Scalar{};
};

/// \brief Expert dense real SPD linear solve result.
/// \tparam Scalar Real scalar type.
template <uni20::LapackReal Scalar> struct RealSymmetricPositiveDefiniteExpertLinearSolve
{
    detail::ColumnMajorLapackMatrix<Scalar> solution;
    detail::ColumnMajorLapackMatrix<Scalar> factors;
    Scalar reciprocal_condition = Scalar{};
    std::vector<Scalar> forward_error_bounds;
    std::vector<Scalar> backward_error_bounds;
    std::vector<Scalar> scale;
    char equilibration = 'N';
    bool reciprocal_condition_below_machine_precision = false;
};

/// \brief Dense real SPD equilibration factors.
/// \tparam Scalar Real scalar type.
template <uni20::LapackReal Scalar> struct RealSymmetricPositiveDefiniteEquilibration
{
    std::vector<Scalar> scale;
    Scalar scale_condition = Scalar{1};
    Scalar max_abs = Scalar{};
};

/// \brief Cholesky factorization of a dense real SPD matrix.
/// \tparam Scalar Real scalar type.
template <uni20::LapackReal Scalar> struct RealSymmetricPositiveDefiniteFactorization
{
    detail::ColumnMajorLapackMatrix<Scalar> factors;
    MatrixFill triangle = MatrixFill::Upper;
};

/// \brief LAPACK symmetric positive-definite band storage for a real square matrix.
/// \details `storage` has `bandwidth + 1` rows and `order` columns. For upper
///          storage, original entries are stored at row `bandwidth + i - j` in
///          column `j` for `i <= j`. For lower storage, original entries are
///          stored at row `i - j` in column `j` for `i >= j`.
/// \tparam Scalar Real scalar type.
template <uni20::LapackReal Scalar> struct RealSymmetricPositiveDefiniteBandMatrix
{
    detail::ColumnMajorLapackMatrix<Scalar> storage;
    std::size_t order = 0;
    std::size_t bandwidth = 0;
    MatrixFill triangle = MatrixFill::Upper;
};

/// \brief Cholesky factorization of a real SPD band matrix.
/// \tparam Scalar Real scalar type.
template <uni20::LapackReal Scalar> struct RealSymmetricPositiveDefiniteBandFactorization
{
    RealSymmetricPositiveDefiniteBandMatrix<Scalar> factors;
};

/// \brief Expert real SPD band linear solve result.
/// \tparam Scalar Real scalar type.
template <uni20::LapackReal Scalar> struct RealSymmetricPositiveDefiniteBandExpertLinearSolve
{
    detail::ColumnMajorLapackMatrix<Scalar> solution;
    RealSymmetricPositiveDefiniteBandFactorization<Scalar> factorization;
    Scalar reciprocal_condition = Scalar{};
    std::vector<Scalar> forward_error_bounds;
    std::vector<Scalar> backward_error_bounds;
    std::vector<Scalar> scale;
    char equilibration = 'N';
    bool reciprocal_condition_below_machine_precision = false;
};

/// \brief Real symmetric positive-definite tridiagonal matrix storage.
/// \details `diagonal` has length `n`; `offdiagonal` has length `n - 1` and
///          stores either the superdiagonal or subdiagonal, which are equal by
///          symmetry.
/// \tparam Scalar Real scalar type.
template <uni20::LapackReal Scalar> struct RealSymmetricPositiveDefiniteTridiagonalMatrix
{
    std::vector<Scalar> diagonal;
    std::vector<Scalar> offdiagonal;

    [[nodiscard]] std::size_t order() const noexcept { return diagonal.size(); }
};

/// \brief Cholesky factorization of a real SPD tridiagonal matrix.
/// \details Uses LAPACK `pttrf` storage: the diagonal is overwritten by the
///          diagonal `D` factor, and the offdiagonal stores the unit bidiagonal
///          factor multipliers.
/// \tparam Scalar Real scalar type.
template <uni20::LapackReal Scalar> struct RealSymmetricPositiveDefiniteTridiagonalFactorization
{
    RealSymmetricPositiveDefiniteTridiagonalMatrix<Scalar> factors;
};

/// \brief Expert real SPD tridiagonal linear solve result.
/// \tparam Scalar Real scalar type.
template <uni20::LapackReal Scalar> struct RealSymmetricPositiveDefiniteTridiagonalExpertLinearSolve
{
    detail::ColumnMajorLapackMatrix<Scalar> solution;
    RealSymmetricPositiveDefiniteTridiagonalFactorization<Scalar> factorization;
    Scalar reciprocal_condition = Scalar{};
    std::vector<Scalar> forward_error_bounds;
    std::vector<Scalar> backward_error_bounds;
    bool reciprocal_condition_below_machine_precision = false;
};

/// \brief Pivoted Cholesky factorization of a dense real positive semidefinite matrix.
/// \tparam Scalar Real scalar type.
template <uni20::LapackReal Scalar> struct RealPivotedCholeskyFactorization
{
    detail::ColumnMajorLapackMatrix<Scalar> factors;
    std::vector<std::size_t> pivot_order;
    std::size_t rank = 0;
    MatrixFill triangle = MatrixFill::Upper;
    bool rank_deficient = false;
};

/// \brief Bunch-Kaufman factorization of a dense real symmetric-indefinite matrix.
/// \tparam Scalar Real scalar type.
template <uni20::LapackReal Scalar> struct RealSymmetricIndefiniteFactorization
{
    detail::ColumnMajorLapackMatrix<Scalar> factors;
    std::vector<int> pivot_blocks;
    MatrixFill triangle = MatrixFill::Upper;
};

/// \brief Expert dense real symmetric-indefinite linear solve result.
/// \tparam Scalar Real scalar type.
template <uni20::LapackReal Scalar> struct RealSymmetricIndefiniteExpertLinearSolve
{
    detail::ColumnMajorLapackMatrix<Scalar> solution;
    detail::ColumnMajorLapackMatrix<Scalar> factors;
    std::vector<int> pivot_blocks;
    Scalar reciprocal_condition = Scalar{};
    std::vector<Scalar> forward_error_bounds;
    std::vector<Scalar> backward_error_bounds;
    bool reciprocal_condition_below_machine_precision = false;
};

/// \brief Rank-revealing dense real least-squares result.
/// \tparam Scalar Real scalar type.
template <uni20::LapackReal Scalar> struct RealRankRevealingLeastSquares
{
    detail::ColumnMajorLapackMatrix<Scalar> solution;
    std::size_t rank = 0;
    std::vector<std::size_t> pivot_columns;
};

/// \brief SVD-based dense real least-squares result.
/// \tparam Scalar Real scalar type.
template <uni20::LapackReal Scalar> struct RealSvdLeastSquares
{
    detail::ColumnMajorLapackMatrix<Scalar> solution;
    std::size_t rank = 0;
    std::vector<Scalar> singular_values;
};

/// \brief Dense real Sylvester equation result.
/// \tparam Scalar Real scalar type.
template <uni20::LapackReal Scalar> struct RealSylvesterSolution
{
    detail::ColumnMajorLapackMatrix<Scalar> solution;
    Scalar scale = Scalar{1};
    bool separation_perturbed = false;
};

/// \brief Reduced QR factorization of a dense real matrix.
/// \tparam Scalar Real scalar type.
template <uni20::LapackReal Scalar> struct RealQrFactorization
{
    detail::ColumnMajorLapackMatrix<Scalar> q;
    detail::ColumnMajorLapackMatrix<Scalar> r;
};

/// \brief Reduced LQ factorization of a dense real matrix.
/// \tparam Scalar Real scalar type.
template <uni20::LapackReal Scalar> struct RealLqFactorization
{
    detail::ColumnMajorLapackMatrix<Scalar> l;
    detail::ColumnMajorLapackMatrix<Scalar> q;
};

/// \brief Reduced QL factorization of a dense real matrix.
/// \tparam Scalar Real scalar type.
template <uni20::LapackReal Scalar> struct RealQlFactorization
{
    detail::ColumnMajorLapackMatrix<Scalar> q;
    detail::ColumnMajorLapackMatrix<Scalar> l;
};

/// \brief Reduced RQ factorization of a dense real matrix.
/// \tparam Scalar Real scalar type.
template <uni20::LapackReal Scalar> struct RealRqFactorization
{
    detail::ColumnMajorLapackMatrix<Scalar> r;
    detail::ColumnMajorLapackMatrix<Scalar> q;
};

/// \brief Compact Householder QR factorization of a dense real matrix.
/// \tparam Scalar Real scalar type.
template <uni20::LapackReal Scalar> struct RealCompactQrFactorization
{
    detail::ColumnMajorLapackMatrix<Scalar> reflectors;
    std::vector<Scalar> tau;
    std::size_t rank = 0;
};

/// \brief Compact Householder LQ factorization of a dense real matrix.
/// \tparam Scalar Real scalar type.
template <uni20::LapackReal Scalar> struct RealCompactLqFactorization
{
    detail::ColumnMajorLapackMatrix<Scalar> reflectors;
    std::vector<Scalar> tau;
    std::size_t rank = 0;
};

/// \brief Compact Householder QL factorization of a dense real matrix.
/// \tparam Scalar Real scalar type.
template <uni20::LapackReal Scalar> struct RealCompactQlFactorization
{
    detail::ColumnMajorLapackMatrix<Scalar> reflectors;
    std::vector<Scalar> tau;
    std::size_t rank = 0;
};

/// \brief Compact Householder RQ factorization of a dense real matrix.
/// \tparam Scalar Real scalar type.
template <uni20::LapackReal Scalar> struct RealCompactRqFactorization
{
    detail::ColumnMajorLapackMatrix<Scalar> reflectors;
    std::vector<Scalar> tau;
    std::size_t rank = 0;
};

/// \brief Compact Householder bidiagonal reduction of a dense real matrix.
/// \tparam Scalar Real scalar type.
template <uni20::LapackReal Scalar> struct RealBidiagonalReduction
{
    detail::ColumnMajorLapackMatrix<Scalar> bidiagonal;
    detail::ColumnMajorLapackMatrix<Scalar> reflectors;
    std::vector<Scalar> diagonal;
    std::vector<Scalar> offdiagonal;
    std::vector<Scalar> tauq;
    std::vector<Scalar> taup;
    bool upper = true;
};

/// \brief Singular value decomposition of a real bidiagonal matrix.
/// \tparam Scalar Real scalar type.
template <uni20::LapackReal Scalar> struct RealBidiagonalSvd
{
    std::vector<Scalar> singular_values;
    detail::ColumnMajorLapackMatrix<Scalar> u;
    detail::ColumnMajorLapackMatrix<Scalar> vt;
};

/// \brief Compact real Hessenberg reduction of a dense square matrix.
/// \tparam Scalar Real scalar type.
template <uni20::LapackReal Scalar> struct RealHessenbergReduction
{
    detail::ColumnMajorLapackMatrix<Scalar> hessenberg;
    detail::ColumnMajorLapackMatrix<Scalar> reflectors;
    std::vector<Scalar> tau;
    std::size_t first = 0;
    std::size_t last_exclusive = 0;
};

/// \brief Compact real symmetric tridiagonal reduction of a dense symmetric matrix.
/// \tparam Scalar Real scalar type.
template <uni20::LapackReal Scalar> struct RealSymmetricTridiagonalReduction
{
    detail::ColumnMajorLapackMatrix<Scalar> tridiagonal;
    detail::ColumnMajorLapackMatrix<Scalar> reflectors;
    std::vector<Scalar> diagonal;
    std::vector<Scalar> offdiagonal;
    std::vector<Scalar> tau;
    MatrixFill triangle = MatrixFill::Upper;
};

/// \brief Reduced QR factorization with column pivoting of a dense real matrix.
/// \tparam Scalar Real scalar type.
template <uni20::LapackReal Scalar> struct RealPivotedQrFactorization
{
    detail::ColumnMajorLapackMatrix<Scalar> q;
    detail::ColumnMajorLapackMatrix<Scalar> r;
    std::vector<std::size_t> pivot_columns;
};

/// \brief Eigenvalues and optional right eigenvectors of a real nonsymmetric matrix.
/// \tparam Real Underlying real precision.

template <uni20::LapackReal Real> struct RealNonsymmetricExpertEigensystem
{
    std::vector<uni20::complex<Real>> eigenvalues;
    detail::ColumnMajorLapackMatrix<uni20::complex<Real>> right_eigenvectors;
    std::vector<Real> reciprocal_eigenvalue_condition_numbers;
    std::vector<Real> reciprocal_eigenvector_condition_numbers;
    std::vector<Real> balance_scale;
    Real balanced_matrix_norm = Real{};
    std::size_t balanced_first = 0;
    std::size_t balanced_last_exclusive = 0;
};

/// \brief LAPACK balancing operations for a dense real nonsymmetric matrix.
enum class RealNonsymmetricBalanceJob
{
  None,
  Permute,
  Scale,
  Both
};

/// \brief Balanced dense real nonsymmetric matrix and LAPACK scaling metadata.
/// \tparam Real Underlying real precision.
template <uni20::LapackReal Real> struct RealNonsymmetricBalance
{
    detail::ColumnMajorLapackMatrix<Real> balanced_matrix;
    std::vector<Real> scale;
    std::size_t balanced_first = 0;
    std::size_t balanced_last_exclusive = 0;
};

/// \brief Eigenvalues and optional right eigenvectors of a real nonsymmetric matrix pencil.
/// \tparam Real Underlying real precision.
template <uni20::LapackReal Real> struct RealGeneralizedNonsymmetricEigensystem
{
    std::vector<uni20::complex<Real>> alpha;
    std::vector<Real> beta;
    std::vector<uni20::complex<Real>> eigenvalues;
    detail::ColumnMajorLapackMatrix<uni20::complex<Real>> right_eigenvectors;
};

/// \brief Expert eigensystem diagnostics for a dense real nonsymmetric matrix pencil.
/// \tparam Real Underlying real precision.
template <uni20::LapackReal Real> struct RealGeneralizedNonsymmetricExpertEigensystem
{
    std::vector<uni20::complex<Real>> alpha;
    std::vector<Real> beta;
    std::vector<uni20::complex<Real>> eigenvalues;
    detail::ColumnMajorLapackMatrix<uni20::complex<Real>> right_eigenvectors;
    std::vector<Real> reciprocal_eigenvalue_condition_numbers;
    std::vector<Real> reciprocal_eigenvector_condition_numbers;
    std::vector<Real> left_balance_scale;
    std::vector<Real> right_balance_scale;
    Real balanced_matrix_norm = Real{};
    Real balanced_metric_norm = Real{};
    std::size_t balanced_first = 0;
    std::size_t balanced_last_exclusive = 0;
};

/// \brief Balanced dense real nonsymmetric matrix pencil and LAPACK scaling metadata.
/// \tparam Real Underlying real precision.
template <uni20::LapackReal Real> struct RealGeneralizedNonsymmetricBalance
{
    detail::ColumnMajorLapackMatrix<Real> balanced_matrix;
    detail::ColumnMajorLapackMatrix<Real> balanced_metric;
    std::vector<Real> left_scale;
    std::vector<Real> right_scale;
    std::size_t balanced_first = 0;
    std::size_t balanced_last_exclusive = 0;
};

/// \brief Generalized Hessenberg reduction of a dense real matrix pencil.
/// \tparam Real Underlying real precision.
template <uni20::LapackReal Real> struct RealGeneralizedHessenbergReduction
{
    detail::ColumnMajorLapackMatrix<Real> matrix_hessenberg_form;
    detail::ColumnMajorLapackMatrix<Real> metric_triangular_form;
    detail::ColumnMajorLapackMatrix<Real> left_orthogonal_vectors;
    detail::ColumnMajorLapackMatrix<Real> right_orthogonal_vectors;
    std::size_t first = 0;
    std::size_t last_exclusive = 0;
};

/// \brief One diagonal block of a real Schur form.
/// \tparam Scalar Real scalar type.

template <uni20::LapackReal Real> struct RealGeneralizedSchurDecomposition
{
    detail::ColumnMajorLapackMatrix<Real> matrix_schur_form;
    detail::ColumnMajorLapackMatrix<Real> metric_schur_form;
    detail::ColumnMajorLapackMatrix<Real> left_schur_vectors;
    detail::ColumnMajorLapackMatrix<Real> right_schur_vectors;
    std::vector<uni20::complex<Real>> alpha;
    std::vector<Real> beta;
    std::vector<uni20::complex<Real>> eigenvalues;
    std::vector<RealSchurBlock<Real>> blocks;
    std::size_t selected_dimension = 0;
};

/// \brief Selected generalized real Schur subspace with LAPACK `tgsen` diagnostics.
/// \tparam Real Underlying real precision.
template <uni20::LapackReal Real> struct RealGeneralizedSchurSelectedSubspace
{
    RealGeneralizedSchurDecomposition<Real> decomposition;
    std::size_t selected_dimension = 0;
    Real left_projection_lower_bound = Real{};
    Real right_projection_lower_bound = Real{};
    Real upper_deflating_subspace_separation = Real{};
    Real lower_deflating_subspace_separation = Real{};
};

/// \brief Right eigenvectors of a generalized real Schur form unpacked into complex columns.
/// \tparam Real Underlying real precision.
template <uni20::LapackReal Real> struct RealGeneralizedSchurRightEigenvectors
{
    detail::ColumnMajorLapackMatrix<uni20::complex<Real>> right_eigenvectors;
    std::size_t computed_vectors = 0;
};

/// \brief Reciprocal condition estimates for eigenpairs of a generalized real Schur form.
/// \tparam Real Underlying real precision.
template <uni20::LapackReal Real> struct RealGeneralizedSchurConditionEstimates
{
    std::vector<Real> reciprocal_eigenvalue_condition_numbers;
    std::vector<Real> reciprocal_eigenvector_condition_numbers;
    std::size_t computed_estimates = 0;
};

/// \brief Eigenvalues and optional right eigenvectors of a complex nonsymmetric matrix.
/// \tparam Real Underlying real precision.

template <uni20::LapackReal Scalar> struct RealSchurSelectedSubspace
{
    RealSchurDecomposition<Scalar> decomposition;
    std::size_t selected_dimension = 0;
    Scalar reciprocal_eigenvalue_cluster_condition = Scalar{};
    Scalar reciprocal_invariant_subspace_condition = Scalar{};
};

/// \brief Right eigenvectors of a real Schur form unpacked into complex columns.
/// \tparam Scalar Real scalar type.
template <uni20::LapackReal Scalar> struct RealSchurRightEigenvectors
{
    detail::ColumnMajorLapackMatrix<uni20::complex<Scalar>> right_eigenvectors;
    std::size_t computed_vectors = 0;
};

/// \brief Reciprocal condition estimates for eigenpairs of a real Schur form.
/// \tparam Scalar Real scalar type.
template <uni20::LapackReal Scalar> struct RealSchurConditionEstimates
{
    std::vector<Scalar> reciprocal_eigenvalue_condition_numbers;
    std::vector<Scalar> reciprocal_eigenvector_condition_numbers;
    std::size_t computed_estimates = 0;
};

namespace detail
{
inline char lapack_uplo(MatrixFill fill)
{
  switch (fill)
  {
    case MatrixFill::Upper:
      return 'U';
    case MatrixFill::Lower:
      return 'L';
    case MatrixFill::All:
      throw std::invalid_argument("dense symmetric LAPACK operation requires an upper or lower triangle selector");
  }
  throw std::invalid_argument("dense symmetric LAPACK operation received an unknown triangle selector");
}

template <uni20::LapackReal Scalar>
Scalar symmetric_matrix_one_norm(detail::ColumnMajorLapackMatrix<Scalar> const& matrix, MatrixFill triangle)
{
  if (!std::cmp_equal(matrix.rows(), matrix.cols()))
  {
    throw std::invalid_argument("symmetric_matrix_one_norm requires a square matrix");
  }
  if (triangle == MatrixFill::All)
  {
    throw std::invalid_argument("symmetric_matrix_one_norm requires an upper or lower triangle selector");
  }

  Scalar norm{};
  for (uni20::index_type col = 0; col < matrix.cols(); ++col)
  {
    Scalar column_sum{};
    for (uni20::index_type row = 0; row < matrix.rows(); ++row)
    {
      if (triangle == MatrixFill::Upper)
      {
        column_sum += row <= col ? detail::adl_abs(matrix[row, col]) : detail::adl_abs(matrix[col, row]);
      }
      else
      {
        column_sum += row >= col ? detail::adl_abs(matrix[row, col]) : detail::adl_abs(matrix[col, row]);
      }
    }
    norm = std::max(norm, column_sum);
  }
  return norm;
}

template <typename Scalar>
void mirror_selected_symmetric_triangle(detail::ColumnMajorLapackMatrix<Scalar>& matrix, MatrixFill triangle)
{
  if (triangle == MatrixFill::All)
  {
    throw std::invalid_argument("symmetric triangle mirroring requires an upper or lower triangle selector");
  }
  for (uni20::index_type col = 0; col < matrix.cols(); ++col)
  {
    for (uni20::index_type row = 0; row < matrix.rows(); ++row)
    {
      if (triangle == MatrixFill::Upper && row > col)
      {
        matrix[row, col] = matrix[col, row];
      }
      if (triangle == MatrixFill::Lower && row < col)
      {
        matrix[row, col] = matrix[col, row];
      }
    }
  }
}

template <uni20::LapackReal Scalar>
std::vector<blas_int>
checked_symmetric_indefinite_pivots(RealSymmetricIndefiniteFactorization<Scalar> const& factorization)
{
  std::size_t const n = static_cast<std::size_t>(factorization.factors.rows());
  std::vector<blas_int> pivots(n);
  for (std::size_t index = 0; index < n; ++index)
  {
    int const pivot = factorization.pivot_blocks[index];
    std::size_t const pivot_magnitude =
        pivot < 0 ? static_cast<std::size_t>(-(pivot + 1)) + std::size_t{1} : static_cast<std::size_t>(pivot);
    if (pivot_magnitude == 0 || pivot_magnitude > n)
    {
      throw std::invalid_argument("symmetric-indefinite factorization has an out-of-range pivot block");
    }
    pivots[index] = static_cast<blas_int>(pivot);
  }
  return pivots;
}

inline char lapack_transpose(MatrixTranspose transpose)
{
  switch (transpose)
  {
    case MatrixTranspose::None:
      return 'N';
    case MatrixTranspose::Transpose:
      return 'T';
    case MatrixTranspose::ConjugateTranspose:
      return 'C';
  }
  throw std::invalid_argument("dense real LAPACK operation received an unknown transpose selector");
}

inline char lapack_side(MatrixSide side)
{
  switch (side)
  {
    case MatrixSide::Left:
      return 'L';
    case MatrixSide::Right:
      return 'R';
  }
  throw std::invalid_argument("dense real LAPACK operation received an unknown side selector");
}

inline char lapack_balance_job(RealNonsymmetricBalanceJob job)
{
  switch (job)
  {
    case RealNonsymmetricBalanceJob::None:
      return 'N';
    case RealNonsymmetricBalanceJob::Permute:
      return 'P';
    case RealNonsymmetricBalanceJob::Scale:
      return 'S';
    case RealNonsymmetricBalanceJob::Both:
      return 'B';
  }
  throw std::invalid_argument("dense real LAPACK operation received an unknown balance job");
}
} // namespace detail

template <uni20::LapackReal Scalar>
Scalar real_matrix_norm(detail::ColumnMajorLapackMatrix<Scalar> matrix, MatrixNorm norm)
{
  if (matrix.rows() == 0 || matrix.cols() == 0)
  {
    return Scalar{};
  }

  blas_int const rows = detail::checked_blas_int(matrix.rows());
  blas_int const cols = detail::checked_blas_int(matrix.cols());
  blas_int const lda = std::max<blas_int>(1, rows);
  std::vector<Scalar> work(static_cast<std::size_t>(std::max<blas_int>(1, rows)), Scalar{});
  return uni20::lapack::lange(detail::lapack_norm(norm), rows, cols, matrix.data(), lda, work.data());
}

/// \brief Compute a dense real symmetric matrix norm through LAPACK `lansy`.
/// \details Only the selected triangle of \p matrix is referenced.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real symmetric square matrix.
/// \param norm Matrix norm to compute.
/// \param triangle Triangle of \p matrix supplied to LAPACK.
/// \return Requested symmetric matrix norm.
template <uni20::LapackReal Scalar>
Scalar real_symmetric_matrix_norm(detail::ColumnMajorLapackMatrix<Scalar> matrix, MatrixNorm norm,
                                  MatrixFill triangle = MatrixFill::Upper)
{
  if (!std::cmp_equal(matrix.rows(), matrix.cols()))
  {
    throw std::invalid_argument("real_symmetric_matrix_norm requires a square matrix");
  }
  if (matrix.rows() == 0)
  {
    return Scalar{};
  }

  blas_int const order = detail::checked_blas_int(matrix.rows());
  blas_int const lda = std::max<blas_int>(1, order);
  char const uplo = detail::lapack_uplo(triangle);
  std::vector<Scalar> work(static_cast<std::size_t>(std::max<blas_int>(1, order)), Scalar{});
  return uni20::lapack::lansy(detail::lapack_norm(norm), uplo, order, matrix.data(), lda, work.data());
}

/// \brief Compute a dense real triangular or trapezoidal matrix norm through LAPACK `lantr`.
/// \details Only the selected triangle/trapezoid of \p matrix is referenced.
///          When \p unit_diagonal is true, the diagonal is treated as unit
///          regardless of stored values.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real rectangular matrix.
/// \param norm Matrix norm to compute.
/// \param triangle Triangle or trapezoid of \p matrix supplied to LAPACK.
/// \param unit_diagonal Whether to treat the diagonal as all ones.
/// \return Requested triangular/trapezoidal matrix norm.
template <uni20::LapackReal Scalar>
Scalar real_triangular_matrix_norm(detail::ColumnMajorLapackMatrix<Scalar> matrix, MatrixNorm norm,
                                   MatrixFill triangle = MatrixFill::Upper, bool unit_diagonal = false)
{
  if (matrix.rows() == 0 || matrix.cols() == 0)
  {
    return Scalar{};
  }

  blas_int const rows = detail::checked_blas_int(matrix.rows());
  blas_int const cols = detail::checked_blas_int(matrix.cols());
  blas_int const lda = std::max<blas_int>(1, rows);
  char const uplo = detail::lapack_uplo(triangle);
  char const diag = unit_diagonal ? 'U' : 'N';
  std::vector<Scalar> work(static_cast<std::size_t>(std::max<blas_int>(1, rows)), Scalar{});
  return uni20::lapack::lantr(detail::lapack_norm(norm), uplo, diag, rows, cols, matrix.data(), lda, work.data());
}

/// \brief Pack a dense real square matrix into LAPACK general-band factor storage.
/// \details Entries outside the requested band are dropped. The returned
///          storage has `2 * lower_bandwidth + upper_bandwidth + 1` rows so it
///          can be passed directly to `gbsv`, `gbtrf`, and `gbtrs`.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real square matrix to pack.
/// \param lower_bandwidth Number of stored subdiagonals.
/// \param upper_bandwidth Number of stored superdiagonals.
/// \return General-band matrix in LAPACK factor-storage layout.
template <uni20::LapackReal Scalar>
RealGeneralBandMatrix<Scalar> real_general_band_from_dense(detail::ColumnMajorLapackMatrix<Scalar> const& matrix,
                                                           std::size_t lower_bandwidth, std::size_t upper_bandwidth)
{
  if (!std::cmp_equal(matrix.rows(), matrix.cols()))
  {
    throw std::invalid_argument("real_general_band_from_dense requires a square matrix");
  }

  std::size_t const n = static_cast<std::size_t>(matrix.rows());
  RealGeneralBandMatrix<Scalar> result;
  result.order = n;
  result.lower_bandwidth = lower_bandwidth;
  result.upper_bandwidth = upper_bandwidth;
  result.storage = detail::ColumnMajorLapackMatrix<Scalar>(2 * lower_bandwidth + upper_bandwidth + 1, n);

  for (std::size_t col = 0; col < n; ++col)
  {
    std::size_t const first_row = col > upper_bandwidth ? col - upper_bandwidth : 0;
    std::size_t const last_row = std::min(n - 1, col + lower_bandwidth);
    for (std::size_t row = first_row; row <= last_row; ++row)
    {
      std::size_t const band_row = lower_bandwidth + upper_bandwidth + row - col;
      result.storage[band_row, col] = matrix[row, col];
    }
  }
  return result;
}

/// \brief Compute a real general-band matrix norm through LAPACK `langb`.
/// \details The band matrix must use the factor-storage layout returned by
///          `real_general_band_from_dense`. Only the original matrix band is
///          referenced; fill rows reserved for factorization are ignored.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param band_matrix Real general-band matrix in LAPACK factor-storage layout.
/// \param norm Matrix norm to compute.
/// \return Requested band-matrix norm.
template <uni20::LapackReal Scalar>
Scalar real_general_band_matrix_norm(RealGeneralBandMatrix<Scalar> band_matrix, MatrixNorm norm)
{
  if (band_matrix.order == 0)
  {
    return Scalar{};
  }
  if (!std::cmp_equal(band_matrix.storage.cols(), band_matrix.order) ||
      std::cmp_less(band_matrix.storage.rows(), 2 * band_matrix.lower_bandwidth + band_matrix.upper_bandwidth + 1))
  {
    throw std::invalid_argument("real_general_band_matrix_norm received inconsistent band storage");
  }

  blas_int const n = detail::checked_blas_int(band_matrix.order);
  blas_int const kl = detail::checked_blas_int(band_matrix.lower_bandwidth);
  blas_int const ku = detail::checked_blas_int(band_matrix.upper_bandwidth);
  blas_int const ldab = detail::checked_blas_int(band_matrix.storage.rows());
  std::vector<Scalar> work(static_cast<std::size_t>(std::max<blas_int>(1, n)), Scalar{});
  Scalar* compact_band = band_matrix.storage.data() + band_matrix.lower_bandwidth;
  return uni20::lapack::langb(detail::lapack_norm(norm), n, kl, ku, compact_band, ldab, work.data());
}

/// \brief Solve a real general-band linear system through LAPACK `gbsv`.
/// \details Solves `A * X = B` for one or more dense right-hand sides. The
///          band matrix must use LAPACK factor-storage layout.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param coefficients Real square general-band coefficient matrix.
/// \param right_hand_sides Dense right-hand-side matrix with matching row count.
/// \return Dense solution matrix.
template <uni20::LapackReal Scalar>
detail::ColumnMajorLapackMatrix<Scalar>
real_general_band_solve(RealGeneralBandMatrix<Scalar> coefficients,
                        detail::ColumnMajorLapackMatrix<Scalar> right_hand_sides)
{
  if (!std::cmp_equal(coefficients.storage.cols(), coefficients.order) ||
      std::cmp_less(coefficients.storage.rows(), 2 * coefficients.lower_bandwidth + coefficients.upper_bandwidth + 1))
  {
    throw std::invalid_argument("real_general_band_solve received inconsistent band storage");
  }
  if (!std::cmp_equal(right_hand_sides.rows(), coefficients.order))
  {
    throw std::invalid_argument("real_general_band_solve received incompatible right-hand sides");
  }
  if (coefficients.order == 0 || right_hand_sides.cols() == 0)
  {
    return right_hand_sides;
  }

  blas_int const n = detail::checked_blas_int(coefficients.order);
  blas_int const kl = detail::checked_blas_int(coefficients.lower_bandwidth);
  blas_int const ku = detail::checked_blas_int(coefficients.upper_bandwidth);
  blas_int const nrhs = detail::checked_blas_int(right_hand_sides.cols());
  blas_int const ldab = detail::checked_blas_int(coefficients.storage.rows());
  std::vector<blas_int> pivots(coefficients.order);
  uni20::lapack::gbsv(n, kl, ku, nrhs, coefficients.storage.data(), ldab, pivots.data(), right_hand_sides.data(), n);
  return right_hand_sides;
}

/// \brief Compute a real general-band LU factorization through LAPACK `gbtrf`.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real square general-band matrix in LAPACK factor-storage layout.
/// \return Banded LU factors and row-pivot metadata.
template <uni20::LapackReal Scalar>
RealGeneralBandFactorization<Scalar> real_general_band_factorization(RealGeneralBandMatrix<Scalar> matrix)
{
  if (!std::cmp_equal(matrix.storage.cols(), matrix.order) ||
      std::cmp_less(matrix.storage.rows(), 2 * matrix.lower_bandwidth + matrix.upper_bandwidth + 1))
  {
    throw std::invalid_argument("real_general_band_factorization received inconsistent band storage");
  }

  RealGeneralBandFactorization<Scalar> result;
  result.factors = std::move(matrix);
  result.pivot_rows.resize(result.factors.order);
  std::iota(result.pivot_rows.begin(), result.pivot_rows.end(), std::size_t{0});
  if (result.factors.order == 0)
  {
    return result;
  }

  blas_int const n = detail::checked_blas_int(result.factors.order);
  blas_int const kl = detail::checked_blas_int(result.factors.lower_bandwidth);
  blas_int const ku = detail::checked_blas_int(result.factors.upper_bandwidth);
  blas_int const ldab = detail::checked_blas_int(result.factors.storage.rows());
  std::vector<blas_int> pivots(result.factors.order);
  uni20::lapack::gbtrf(n, n, kl, ku, result.factors.storage.data(), ldab, pivots.data());
  for (std::size_t index = 0; index < result.factors.order; ++index)
  {
    if (pivots[index] <= 0 || static_cast<std::size_t>(pivots[index]) > result.factors.order)
    {
      throw std::runtime_error("LAPACK gbtrf returned an invalid row pivot");
    }
    result.pivot_rows[index] = static_cast<std::size_t>(pivots[index] - 1);
  }
  return result;
}

/// \brief Solve using an existing real general-band LU factorization through LAPACK `gbtrs`.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param factorization Factors returned by `real_general_band_factorization`.
/// \param right_hand_sides Dense right-hand-side matrix with matching row count.
/// \param transpose Matrix operation applied to the original coefficient matrix.
/// \return Dense solution matrix.
template <uni20::LapackReal Scalar>
detail::ColumnMajorLapackMatrix<Scalar>
real_general_band_solve(RealGeneralBandFactorization<Scalar> const& factorization,
                        detail::ColumnMajorLapackMatrix<Scalar> right_hand_sides,
                        MatrixTranspose transpose = MatrixTranspose::None)
{
  if (!std::cmp_equal(factorization.factors.storage.cols(), factorization.factors.order) ||
      std::cmp_less(factorization.factors.storage.rows(),
                    2 * factorization.factors.lower_bandwidth + factorization.factors.upper_bandwidth + 1))
  {
    throw std::invalid_argument("real_general_band_solve received inconsistent band factors");
  }
  if (factorization.pivot_rows.size() != factorization.factors.order)
  {
    throw std::invalid_argument("real_general_band_solve received inconsistent pivot metadata");
  }
  if (!std::cmp_equal(right_hand_sides.rows(), factorization.factors.order))
  {
    throw std::invalid_argument("real_general_band_solve received incompatible right-hand sides");
  }
  if (factorization.factors.order == 0 || right_hand_sides.cols() == 0)
  {
    return right_hand_sides;
  }

  detail::ColumnMajorLapackMatrix<Scalar> factors = factorization.factors.storage;
  std::vector<blas_int> pivots(factorization.factors.order);
  for (std::size_t index = 0; index < factorization.factors.order; ++index)
  {
    if (factorization.pivot_rows[index] >= factorization.factors.order)
    {
      throw std::invalid_argument("real_general_band_solve received an out-of-range row pivot");
    }
    pivots[index] = detail::checked_blas_int(factorization.pivot_rows[index] + 1);
  }

  blas_int const n = detail::checked_blas_int(factorization.factors.order);
  blas_int const kl = detail::checked_blas_int(factorization.factors.lower_bandwidth);
  blas_int const ku = detail::checked_blas_int(factorization.factors.upper_bandwidth);
  blas_int const nrhs = detail::checked_blas_int(right_hand_sides.cols());
  blas_int const ldab = detail::checked_blas_int(factors.rows());
  char const trans = detail::lapack_transpose(transpose);
  uni20::lapack::gbtrs(trans, n, kl, ku, nrhs, factors.data(), ldab, pivots.data(), right_hand_sides.data(), n);
  return right_hand_sides;
}

/// \brief Extract real general tridiagonal storage from a dense matrix.
/// \details The diagonal, first subdiagonal, and first superdiagonal are
///          copied. Entries outside the tridiagonal band are ignored.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real square tridiagonal matrix.
/// \return General tridiagonal matrix storage suitable for LAPACK `gtsv` and `gttrf`.
template <uni20::LapackReal Scalar>
RealGeneralTridiagonalMatrix<Scalar>
real_general_tridiagonal_from_dense(detail::ColumnMajorLapackMatrix<Scalar> const& matrix)
{
  if (!std::cmp_equal(matrix.rows(), matrix.cols()))
  {
    throw std::invalid_argument("real_general_tridiagonal_from_dense requires a square matrix");
  }

  RealGeneralTridiagonalMatrix<Scalar> result;
  result.diagonal.resize(matrix.rows());
  if (matrix.rows() > 0)
  {
    result.lower_diagonal.resize(matrix.rows() - 1);
    result.upper_diagonal.resize(matrix.rows() - 1);
  }
  for (uni20::index_type index = 0; index < matrix.rows(); ++index)
  {
    result.diagonal[index] = matrix[index, index];
    if (index + 1 < matrix.rows())
    {
      result.lower_diagonal[index] = matrix[index + 1, index];
      result.upper_diagonal[index] = matrix[index, index + 1];
    }
  }
  return result;
}

/// \brief Solve a real general tridiagonal linear system through LAPACK `gtsv`.
/// \details Solves `A * X = B` for one or more dense right-hand sides.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param coefficients Real square general tridiagonal coefficient matrix.
/// \param right_hand_sides Dense right-hand-side matrix with matching row count.
/// \return Dense solution matrix.
template <uni20::LapackReal Scalar>
detail::ColumnMajorLapackMatrix<Scalar>
real_general_tridiagonal_solve(RealGeneralTridiagonalMatrix<Scalar> coefficients,
                               detail::ColumnMajorLapackMatrix<Scalar> right_hand_sides)
{
  std::size_t const order = coefficients.order();
  if (coefficients.lower_diagonal.size() != (order == 0 ? 0 : order - 1) ||
      coefficients.upper_diagonal.size() != (order == 0 ? 0 : order - 1))
  {
    throw std::invalid_argument("real_general_tridiagonal_solve received inconsistent tridiagonal storage");
  }
  if (!std::cmp_equal(right_hand_sides.rows(), order))
  {
    throw std::invalid_argument("real_general_tridiagonal_solve received incompatible right-hand sides");
  }
  if (order == 0 || right_hand_sides.cols() == 0)
  {
    return right_hand_sides;
  }

  blas_int const n = detail::checked_blas_int(order);
  blas_int const nrhs = detail::checked_blas_int(right_hand_sides.cols());
  Scalar* lower_diagonal =
      coefficients.lower_diagonal.empty() ? coefficients.diagonal.data() : coefficients.lower_diagonal.data();
  Scalar* upper_diagonal =
      coefficients.upper_diagonal.empty() ? coefficients.diagonal.data() : coefficients.upper_diagonal.data();
  uni20::lapack::gtsv(n, nrhs, lower_diagonal, coefficients.diagonal.data(), upper_diagonal, right_hand_sides.data(),
                      n);
  return right_hand_sides;
}

/// \brief Compute a real general tridiagonal LU factorization through LAPACK `gttrf`.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real square general tridiagonal matrix.
/// \return Tridiagonal LU factors and row-pivot metadata.
template <uni20::LapackReal Scalar>
RealGeneralTridiagonalFactorization<Scalar>
real_general_tridiagonal_factorization(RealGeneralTridiagonalMatrix<Scalar> matrix)
{
  std::size_t const order = matrix.order();
  if (matrix.lower_diagonal.size() != (order == 0 ? 0 : order - 1) ||
      matrix.upper_diagonal.size() != (order == 0 ? 0 : order - 1))
  {
    throw std::invalid_argument("real_general_tridiagonal_factorization received inconsistent tridiagonal storage");
  }

  RealGeneralTridiagonalFactorization<Scalar> result;
  result.factors = std::move(matrix);
  result.second_upper_diagonal.resize(order > 2 ? order - 2 : 0);
  result.pivot_rows.resize(order);
  std::iota(result.pivot_rows.begin(), result.pivot_rows.end(), std::size_t{0});
  if (order == 0)
  {
    return result;
  }

  blas_int const n = detail::checked_blas_int(order);
  std::vector<blas_int> pivots(order);
  Scalar* lower_diagonal =
      result.factors.lower_diagonal.empty() ? result.factors.diagonal.data() : result.factors.lower_diagonal.data();
  Scalar* upper_diagonal =
      result.factors.upper_diagonal.empty() ? result.factors.diagonal.data() : result.factors.upper_diagonal.data();
  Scalar* second_upper_diagonal =
      result.second_upper_diagonal.empty() ? result.factors.diagonal.data() : result.second_upper_diagonal.data();
  uni20::lapack::gttrf(n, lower_diagonal, result.factors.diagonal.data(), upper_diagonal, second_upper_diagonal,
                       pivots.data());
  for (std::size_t index = 0; index < order; ++index)
  {
    if (pivots[index] <= 0 || static_cast<std::size_t>(pivots[index]) > order)
    {
      throw std::runtime_error("LAPACK gttrf returned an invalid row pivot");
    }
    result.pivot_rows[index] = static_cast<std::size_t>(pivots[index] - 1);
  }
  return result;
}

/// \brief Solve using an existing real general tridiagonal LU factorization through LAPACK `gttrs`.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param factorization Factors returned by `real_general_tridiagonal_factorization`.
/// \param right_hand_sides Dense right-hand-side matrix with matching row count.
/// \param transpose Matrix operation applied to the original coefficient matrix.
/// \return Dense solution matrix.
template <uni20::LapackReal Scalar>
detail::ColumnMajorLapackMatrix<Scalar>
real_general_tridiagonal_solve(RealGeneralTridiagonalFactorization<Scalar> const& factorization,
                               detail::ColumnMajorLapackMatrix<Scalar> right_hand_sides,
                               MatrixTranspose transpose = MatrixTranspose::None)
{
  std::size_t const order = factorization.factors.order();
  if (factorization.factors.lower_diagonal.size() != (order == 0 ? 0 : order - 1) ||
      factorization.factors.upper_diagonal.size() != (order == 0 ? 0 : order - 1) ||
      factorization.second_upper_diagonal.size() != (order > 2 ? order - 2 : 0))
  {
    throw std::invalid_argument("real_general_tridiagonal_solve received inconsistent tridiagonal factors");
  }
  if (factorization.pivot_rows.size() != order)
  {
    throw std::invalid_argument("real_general_tridiagonal_solve received inconsistent pivot metadata");
  }
  if (!std::cmp_equal(right_hand_sides.rows(), order))
  {
    throw std::invalid_argument("real_general_tridiagonal_solve received incompatible right-hand sides");
  }
  if (order == 0 || right_hand_sides.cols() == 0)
  {
    return right_hand_sides;
  }

  auto lower_diagonal_values = factorization.factors.lower_diagonal;
  auto diagonal_values = factorization.factors.diagonal;
  auto upper_diagonal_values = factorization.factors.upper_diagonal;
  auto second_upper_diagonal_values = factorization.second_upper_diagonal;
  std::vector<blas_int> pivots(order);
  for (std::size_t index = 0; index < order; ++index)
  {
    if (factorization.pivot_rows[index] >= order)
    {
      throw std::invalid_argument("real_general_tridiagonal_solve received an out-of-range row pivot");
    }
    pivots[index] = detail::checked_blas_int(factorization.pivot_rows[index] + 1);
  }

  blas_int const n = detail::checked_blas_int(order);
  blas_int const nrhs = detail::checked_blas_int(right_hand_sides.cols());
  char const trans = detail::lapack_transpose(transpose);
  Scalar* lower_diagonal = lower_diagonal_values.empty() ? diagonal_values.data() : lower_diagonal_values.data();
  Scalar* upper_diagonal = upper_diagonal_values.empty() ? diagonal_values.data() : upper_diagonal_values.data();
  Scalar* second_upper_diagonal =
      second_upper_diagonal_values.empty() ? diagonal_values.data() : second_upper_diagonal_values.data();
  uni20::lapack::gttrs(trans, n, nrhs, lower_diagonal, diagonal_values.data(), upper_diagonal, second_upper_diagonal,
                       pivots.data(), right_hand_sides.data(), n);
  return right_hand_sides;
}

/// \brief Compute the one-norm of a real general tridiagonal matrix.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real square general tridiagonal matrix.
/// \return Matrix one-norm.
template <uni20::LapackReal Scalar>
Scalar real_general_tridiagonal_one_norm(RealGeneralTridiagonalMatrix<Scalar> const& matrix)
{
  std::size_t const order = matrix.order();
  if (matrix.lower_diagonal.size() != (order == 0 ? 0 : order - 1) ||
      matrix.upper_diagonal.size() != (order == 0 ? 0 : order - 1))
  {
    throw std::invalid_argument("real_general_tridiagonal_one_norm received inconsistent tridiagonal storage");
  }

  Scalar norm{};
  for (std::size_t col = 0; col < order; ++col)
  {
    Scalar column_sum = detail::adl_abs(matrix.diagonal[col]);
    if (col > 0)
    {
      column_sum += detail::adl_abs(matrix.upper_diagonal[col - 1]);
    }
    if (col + 1 < order)
    {
      column_sum += detail::adl_abs(matrix.lower_diagonal[col]);
    }
    norm = std::max(norm, column_sum);
  }
  return norm;
}

/// \brief Estimate a real general-tridiagonal one-norm reciprocal condition number through LAPACK `gtcon`.
/// \details Uses an existing tridiagonal LU factorization. The supplied matrix
///          norm must be the one-norm of the original unfactored tridiagonal
///          matrix.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param factorization Factors returned by `real_general_tridiagonal_factorization`.
/// \param original_one_norm One-norm of the original coefficient matrix.
/// \return Estimate of `1 / (||A||_1 * ||inv(A)||_1)`.
template <uni20::LapackReal Scalar>
Scalar real_general_tridiagonal_one_norm_reciprocal_condition_number(
    RealGeneralTridiagonalFactorization<Scalar> const& factorization, Scalar original_one_norm)
{
  std::size_t const order = factorization.factors.order();
  if (factorization.factors.lower_diagonal.size() != (order == 0 ? 0 : order - 1) ||
      factorization.factors.upper_diagonal.size() != (order == 0 ? 0 : order - 1) ||
      factorization.second_upper_diagonal.size() != (order > 2 ? order - 2 : 0))
  {
    throw std::invalid_argument(
        "real_general_tridiagonal_one_norm_reciprocal_condition_number received inconsistent factors");
  }
  if (factorization.pivot_rows.size() != order)
  {
    throw std::invalid_argument(
        "real_general_tridiagonal_one_norm_reciprocal_condition_number received inconsistent pivots");
  }
  if (original_one_norm < Scalar{})
  {
    throw std::invalid_argument(
        "real_general_tridiagonal_one_norm_reciprocal_condition_number requires a nonnegative norm");
  }
  if (order == 0)
  {
    return Scalar{1};
  }

  auto lower_diagonal_values = factorization.factors.lower_diagonal;
  auto diagonal_values = factorization.factors.diagonal;
  auto upper_diagonal_values = factorization.factors.upper_diagonal;
  auto second_upper_diagonal_values = factorization.second_upper_diagonal;
  std::vector<blas_int> pivots(order);
  for (std::size_t index = 0; index < order; ++index)
  {
    if (factorization.pivot_rows[index] >= order)
    {
      throw std::invalid_argument(
          "real_general_tridiagonal_one_norm_reciprocal_condition_number received an out-of-range pivot");
    }
    pivots[index] = detail::checked_blas_int(factorization.pivot_rows[index] + 1);
  }

  blas_int const n = detail::checked_blas_int(order);
  std::vector<Scalar> work(static_cast<std::size_t>(std::max<blas_int>(1, 2 * n)), Scalar{});
  std::vector<blas_int> iwork(static_cast<std::size_t>(std::max<blas_int>(1, n)), 0);
  Scalar* lower_diagonal = lower_diagonal_values.empty() ? diagonal_values.data() : lower_diagonal_values.data();
  Scalar* upper_diagonal = upper_diagonal_values.empty() ? diagonal_values.data() : upper_diagonal_values.data();
  Scalar* second_upper_diagonal =
      second_upper_diagonal_values.empty() ? diagonal_values.data() : second_upper_diagonal_values.data();
  return uni20::lapack::gtcon('1', n, lower_diagonal, diagonal_values.data(), upper_diagonal, second_upper_diagonal,
                              pivots.data(), original_one_norm, work.data(), iwork.data());
}

/// \brief Estimate a real general-tridiagonal one-norm reciprocal condition number from a matrix.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real square general tridiagonal matrix.
/// \return Estimate of `1 / (||A||_1 * ||inv(A)||_1)`.
template <uni20::LapackReal Scalar>
Scalar real_general_tridiagonal_one_norm_reciprocal_condition_number(RealGeneralTridiagonalMatrix<Scalar> matrix)
{
  Scalar const norm = real_general_tridiagonal_one_norm(matrix);
  auto factorization = real_general_tridiagonal_factorization(std::move(matrix));
  return real_general_tridiagonal_one_norm_reciprocal_condition_number(factorization, norm);
}

/// \brief Solve and refine a real general-tridiagonal linear system through LAPACK `gtrfs`.
/// \details Computes an initial solution using `gttrs`, then refines it with
///          `gtrfs` and returns LAPACK's forward and backward error estimates.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param coefficients Real square general tridiagonal coefficient matrix.
/// \param right_hand_sides Dense right-hand-side matrix with matching row count.
/// \param transpose Matrix operation applied to the coefficient matrix.
/// \return Solution and error-bound diagnostics.
template <uni20::LapackReal Scalar>
RealRefinedLinearSolve<Scalar>
real_general_tridiagonal_refined_solve(RealGeneralTridiagonalMatrix<Scalar> coefficients,
                                       detail::ColumnMajorLapackMatrix<Scalar> right_hand_sides,
                                       MatrixTranspose transpose = MatrixTranspose::None)
{
  std::size_t const order = coefficients.order();
  if (coefficients.lower_diagonal.size() != (order == 0 ? 0 : order - 1) ||
      coefficients.upper_diagonal.size() != (order == 0 ? 0 : order - 1))
  {
    throw std::invalid_argument("real_general_tridiagonal_refined_solve received inconsistent tridiagonal storage");
  }
  if (!std::cmp_equal(right_hand_sides.rows(), order))
  {
    throw std::invalid_argument("real_general_tridiagonal_refined_solve received incompatible right-hand sides");
  }

  std::size_t const rhs_count = static_cast<std::size_t>(right_hand_sides.cols());
  RealRefinedLinearSolve<Scalar> result;
  result.solution = right_hand_sides;
  result.forward_error_bounds.resize(rhs_count);
  result.backward_error_bounds.resize(rhs_count);
  if (order == 0 || rhs_count == 0)
  {
    return result;
  }

  auto factorization = real_general_tridiagonal_factorization(coefficients);
  result.solution = real_general_tridiagonal_solve(factorization, result.solution, transpose);

  std::vector<blas_int> pivots(order);
  for (std::size_t index = 0; index < order; ++index)
  {
    if (factorization.pivot_rows[index] >= order)
    {
      throw std::invalid_argument("real_general_tridiagonal_refined_solve received an out-of-range row pivot");
    }
    pivots[index] = detail::checked_blas_int(factorization.pivot_rows[index] + 1);
  }

  blas_int const n = detail::checked_blas_int(order);
  blas_int const nrhs = detail::checked_blas_int(rhs_count);
  char const trans = detail::lapack_transpose(transpose);
  std::vector<Scalar> work(static_cast<std::size_t>(std::max<blas_int>(1, 3 * n)), Scalar{});
  std::vector<blas_int> iwork(static_cast<std::size_t>(std::max<blas_int>(1, n)), 0);
  Scalar* lower_diagonal =
      coefficients.lower_diagonal.empty() ? coefficients.diagonal.data() : coefficients.lower_diagonal.data();
  Scalar* upper_diagonal =
      coefficients.upper_diagonal.empty() ? coefficients.diagonal.data() : coefficients.upper_diagonal.data();
  Scalar* factor_lower_diagonal = factorization.factors.lower_diagonal.empty()
                                      ? factorization.factors.diagonal.data()
                                      : factorization.factors.lower_diagonal.data();
  Scalar* factor_upper_diagonal = factorization.factors.upper_diagonal.empty()
                                      ? factorization.factors.diagonal.data()
                                      : factorization.factors.upper_diagonal.data();
  Scalar* factor_second_upper_diagonal = factorization.second_upper_diagonal.empty()
                                             ? factorization.factors.diagonal.data()
                                             : factorization.second_upper_diagonal.data();
  uni20::lapack::gtrfs(trans, n, nrhs, lower_diagonal, coefficients.diagonal.data(), upper_diagonal,
                       factor_lower_diagonal, factorization.factors.diagonal.data(), factor_upper_diagonal,
                       factor_second_upper_diagonal, pivots.data(), right_hand_sides.data(), n, result.solution.data(),
                       n, result.forward_error_bounds.data(), result.backward_error_bounds.data(), work.data(),
                       iwork.data());
  return result;
}

/// \brief Solve a real general-tridiagonal linear system with LAPACK `gtsvx` diagnostics.
/// \details Solves `op(A) * X = B` and returns the solution, tridiagonal LU
///          factors, pivot metadata, reciprocal condition estimate,
///          componentwise backward errors, forward error bounds, and condition
///          diagnostics reported by LAPACK. The current wrapper requests a
///          fresh factorization (`FACT='N'`).
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param coefficients Real square general tridiagonal coefficient matrix.
/// \param right_hand_sides Dense right-hand-side matrix with matching row count.
/// \param transpose Matrix operation applied to the coefficient matrix.
/// \return Solution, factorization, and diagnostics.
template <uni20::LapackReal Scalar>
RealGeneralTridiagonalExpertLinearSolve<Scalar>
real_general_tridiagonal_expert_linear_solve(RealGeneralTridiagonalMatrix<Scalar> coefficients,
                                             detail::ColumnMajorLapackMatrix<Scalar> right_hand_sides,
                                             MatrixTranspose transpose = MatrixTranspose::None)
{
  std::size_t const order = coefficients.order();
  if (coefficients.lower_diagonal.size() != (order == 0 ? 0 : order - 1) ||
      coefficients.upper_diagonal.size() != (order == 0 ? 0 : order - 1))
  {
    throw std::invalid_argument(
        "real_general_tridiagonal_expert_linear_solve received inconsistent tridiagonal storage");
  }
  if (!std::cmp_equal(right_hand_sides.rows(), order))
  {
    throw std::invalid_argument("real_general_tridiagonal_expert_linear_solve received incompatible right-hand sides");
  }

  std::size_t const rhs_count = static_cast<std::size_t>(right_hand_sides.cols());
  RealGeneralTridiagonalExpertLinearSolve<Scalar> result;
  result.solution = detail::ColumnMajorLapackMatrix<Scalar>(right_hand_sides.rows(), rhs_count);
  result.factorization.factors.diagonal.resize(order);
  result.factorization.factors.lower_diagonal.resize(order == 0 ? 0 : order - 1);
  result.factorization.factors.upper_diagonal.resize(order == 0 ? 0 : order - 1);
  result.factorization.second_upper_diagonal.resize(order > 2 ? order - 2 : 0);
  result.factorization.pivot_rows.resize(order);
  std::iota(result.factorization.pivot_rows.begin(), result.factorization.pivot_rows.end(), std::size_t{0});
  result.forward_error_bounds.resize(rhs_count);
  result.backward_error_bounds.resize(rhs_count);
  if (order == 0 || rhs_count == 0)
  {
    return result;
  }

  std::vector<blas_int> pivots(order);
  blas_int const n = detail::checked_blas_int(order);
  blas_int const nrhs = detail::checked_blas_int(rhs_count);
  char const trans = detail::lapack_transpose(transpose);
  std::vector<Scalar> work(static_cast<std::size_t>(std::max<blas_int>(1, 3 * n)), Scalar{});
  std::vector<blas_int> iwork(static_cast<std::size_t>(std::max<blas_int>(1, n)), 0);
  Scalar* lower_diagonal =
      coefficients.lower_diagonal.empty() ? coefficients.diagonal.data() : coefficients.lower_diagonal.data();
  Scalar* upper_diagonal =
      coefficients.upper_diagonal.empty() ? coefficients.diagonal.data() : coefficients.upper_diagonal.data();
  Scalar* factor_lower_diagonal = result.factorization.factors.lower_diagonal.empty()
                                      ? result.factorization.factors.diagonal.data()
                                      : result.factorization.factors.lower_diagonal.data();
  Scalar* factor_upper_diagonal = result.factorization.factors.upper_diagonal.empty()
                                      ? result.factorization.factors.diagonal.data()
                                      : result.factorization.factors.upper_diagonal.data();
  Scalar* factor_second_upper_diagonal = result.factorization.second_upper_diagonal.empty()
                                             ? result.factorization.factors.diagonal.data()
                                             : result.factorization.second_upper_diagonal.data();
  result.reciprocal_condition_below_machine_precision = uni20::lapack::gtsvx(
      'N', trans, n, nrhs, lower_diagonal, coefficients.diagonal.data(), upper_diagonal, factor_lower_diagonal,
      result.factorization.factors.diagonal.data(), factor_upper_diagonal, factor_second_upper_diagonal, pivots.data(),
      right_hand_sides.data(), n, result.solution.data(), n, result.reciprocal_condition,
      result.forward_error_bounds.data(), result.backward_error_bounds.data(), work.data(), iwork.data());
  for (std::size_t index = 0; index < order; ++index)
  {
    if (pivots[index] <= 0 || static_cast<std::size_t>(pivots[index]) > order)
    {
      throw std::runtime_error("LAPACK gtsvx returned an invalid row pivot");
    }
    result.factorization.pivot_rows[index] = static_cast<std::size_t>(pivots[index] - 1);
  }
  return result;
}

/// \brief Estimate a real general-band one-norm reciprocal condition number through LAPACK `gbcon`.
/// \details Uses an existing banded LU factorization. The supplied matrix norm
///          must be the one-norm of the original unfactored band matrix.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param factorization Factors returned by `real_general_band_factorization`.
/// \param original_one_norm One-norm of the original coefficient matrix.
/// \return Estimate of `1 / (||A||_1 * ||inv(A)||_1)`.
template <uni20::LapackReal Scalar>
Scalar real_general_band_one_norm_reciprocal_condition_number(RealGeneralBandFactorization<Scalar> const& factorization,
                                                              Scalar original_one_norm)
{
  if (!std::cmp_equal(factorization.factors.storage.cols(), factorization.factors.order) ||
      std::cmp_less(factorization.factors.storage.rows(),
                    2 * factorization.factors.lower_bandwidth + factorization.factors.upper_bandwidth + 1))
  {
    throw std::invalid_argument("real_general_band_one_norm_reciprocal_condition_number received inconsistent factors");
  }
  if (factorization.pivot_rows.size() != factorization.factors.order)
  {
    throw std::invalid_argument("real_general_band_one_norm_reciprocal_condition_number received inconsistent pivots");
  }
  if (original_one_norm < Scalar{})
  {
    throw std::invalid_argument("real_general_band_one_norm_reciprocal_condition_number requires a nonnegative norm");
  }
  if (factorization.factors.order == 0)
  {
    return Scalar{1};
  }

  detail::ColumnMajorLapackMatrix<Scalar> factors = factorization.factors.storage;
  std::vector<blas_int> pivots(factorization.factors.order);
  for (std::size_t index = 0; index < factorization.factors.order; ++index)
  {
    if (factorization.pivot_rows[index] >= factorization.factors.order)
    {
      throw std::invalid_argument(
          "real_general_band_one_norm_reciprocal_condition_number received an out-of-range pivot");
    }
    pivots[index] = detail::checked_blas_int(factorization.pivot_rows[index] + 1);
  }

  blas_int const n = detail::checked_blas_int(factorization.factors.order);
  blas_int const kl = detail::checked_blas_int(factorization.factors.lower_bandwidth);
  blas_int const ku = detail::checked_blas_int(factorization.factors.upper_bandwidth);
  blas_int const ldab = detail::checked_blas_int(factors.rows());
  std::vector<Scalar> work(static_cast<std::size_t>(std::max<blas_int>(1, 3 * n)), Scalar{});
  std::vector<blas_int> iwork(static_cast<std::size_t>(std::max<blas_int>(1, n)), 0);
  return uni20::lapack::gbcon('1', n, kl, ku, factors.data(), ldab, pivots.data(), original_one_norm, work.data(),
                              iwork.data());
}

/// \brief Estimate a real general-band one-norm reciprocal condition number.
/// \details Computes the band one-norm, factors the matrix with LAPACK
///          `gbtrf`, and estimates the reciprocal condition number with
///          LAPACK `gbcon`.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real square general-band matrix in LAPACK factor-storage layout.
/// \return Estimate of `1 / (||A||_1 * ||inv(A)||_1)`.
template <uni20::LapackReal Scalar>
Scalar real_general_band_one_norm_reciprocal_condition_number(RealGeneralBandMatrix<Scalar> matrix)
{
  Scalar const norm = real_general_band_matrix_norm(matrix, MatrixNorm::One);
  auto factorization = real_general_band_factorization(std::move(matrix));
  return real_general_band_one_norm_reciprocal_condition_number(factorization, norm);
}

/// \brief Solve and refine a real general-band system through LAPACK `gbrfs`.
/// \details Computes a banded LU factorization, solves `op(A) * X = B`, then
///          applies iterative refinement and returns LAPACK's forward and
///          backward error estimates. The original band matrix is retained for
///          the refinement residual calculation.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param coefficients Real square general-band coefficient matrix.
/// \param right_hand_sides Dense right-hand-side matrix with matching row count.
/// \param transpose Matrix operation applied to the coefficient matrix.
/// \return Refined solution and LAPACK error estimates.
template <uni20::LapackReal Scalar>
RealRefinedLinearSolve<Scalar> real_general_band_refined_solve(RealGeneralBandMatrix<Scalar> coefficients,
                                                               detail::ColumnMajorLapackMatrix<Scalar> right_hand_sides,
                                                               MatrixTranspose transpose = MatrixTranspose::None)
{
  if (!std::cmp_equal(coefficients.storage.cols(), coefficients.order) ||
      std::cmp_less(coefficients.storage.rows(), 2 * coefficients.lower_bandwidth + coefficients.upper_bandwidth + 1))
  {
    throw std::invalid_argument("real_general_band_refined_solve received inconsistent band storage");
  }
  if (!std::cmp_equal(right_hand_sides.rows(), coefficients.order))
  {
    throw std::invalid_argument("real_general_band_refined_solve received incompatible right-hand sides");
  }

  std::size_t const rhs_count = static_cast<std::size_t>(right_hand_sides.cols());
  RealRefinedLinearSolve<Scalar> result;
  result.solution = right_hand_sides;
  result.forward_error_bounds.resize(rhs_count);
  result.backward_error_bounds.resize(rhs_count);
  if (coefficients.order == 0 || rhs_count == 0)
  {
    return result;
  }

  auto factorization = real_general_band_factorization(coefficients);
  std::vector<blas_int> pivots(factorization.factors.order);
  for (std::size_t index = 0; index < factorization.factors.order; ++index)
  {
    if (factorization.pivot_rows[index] >= factorization.factors.order)
    {
      throw std::invalid_argument("real_general_band_refined_solve received an out-of-range row pivot");
    }
    pivots[index] = detail::checked_blas_int(factorization.pivot_rows[index] + 1);
  }

  blas_int const n = detail::checked_blas_int(coefficients.order);
  blas_int const kl = detail::checked_blas_int(coefficients.lower_bandwidth);
  blas_int const ku = detail::checked_blas_int(coefficients.upper_bandwidth);
  blas_int const nrhs = detail::checked_blas_int(rhs_count);
  blas_int const ldab = detail::checked_blas_int(coefficients.storage.rows());
  blas_int const ldafb = detail::checked_blas_int(factorization.factors.storage.rows());
  char const trans = detail::lapack_transpose(transpose);

  uni20::lapack::gbtrs(trans, n, kl, ku, nrhs, factorization.factors.storage.data(), ldafb, pivots.data(),
                       result.solution.data(), n);

  std::vector<Scalar> work(static_cast<std::size_t>(std::max<blas_int>(1, 3 * n)), Scalar{});
  std::vector<blas_int> iwork(static_cast<std::size_t>(std::max<blas_int>(1, n)), 0);
  Scalar* compact_band = coefficients.storage.data() + coefficients.lower_bandwidth;
  uni20::lapack::gbrfs(trans, n, kl, ku, nrhs, compact_band, ldab, factorization.factors.storage.data(), ldafb,
                       pivots.data(), right_hand_sides.data(), n, result.solution.data(), n,
                       result.forward_error_bounds.data(), result.backward_error_bounds.data(), work.data(),
                       iwork.data());
  return result;
}

/// \brief Solve a real general-band linear system with LAPACK `gbsvx` diagnostics.
/// \details Solves `op(A) * X = B` and returns the solution, banded LU factors,
///          pivot metadata, reciprocal condition estimate, componentwise
///          backward errors, forward error bounds, and condition diagnostics
///          reported by LAPACK. The current wrapper requests a fresh
///          factorization (`FACT='N'`).
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param coefficients Real square general-band coefficient matrix.
/// \param right_hand_sides Dense right-hand-side matrix with matching row count.
/// \param transpose Matrix operation applied to the coefficient matrix.
/// \return Banded solution and LAPACK expert-driver diagnostics.
template <uni20::LapackReal Scalar>
RealGeneralBandExpertLinearSolve<Scalar>
real_general_band_expert_linear_solve(RealGeneralBandMatrix<Scalar> coefficients,
                                      detail::ColumnMajorLapackMatrix<Scalar> right_hand_sides,
                                      MatrixTranspose transpose = MatrixTranspose::None)
{
  if (!std::cmp_equal(coefficients.storage.cols(), coefficients.order) ||
      std::cmp_less(coefficients.storage.rows(), 2 * coefficients.lower_bandwidth + coefficients.upper_bandwidth + 1))
  {
    throw std::invalid_argument("real_general_band_expert_linear_solve received inconsistent band storage");
  }
  if (!std::cmp_equal(right_hand_sides.rows(), coefficients.order))
  {
    throw std::invalid_argument("real_general_band_expert_linear_solve received incompatible right-hand sides");
  }

  std::size_t const n = coefficients.order;
  std::size_t const rhs_count = static_cast<std::size_t>(right_hand_sides.cols());
  RealGeneralBandExpertLinearSolve<Scalar> result;
  result.solution = detail::ColumnMajorLapackMatrix<Scalar>(n, rhs_count);
  result.factors.order = n;
  result.factors.lower_bandwidth = coefficients.lower_bandwidth;
  result.factors.upper_bandwidth = coefficients.upper_bandwidth;
  result.factors.storage =
      detail::ColumnMajorLapackMatrix<Scalar>(coefficients.storage.rows(), coefficients.storage.cols());
  result.pivot_rows.resize(n);
  std::iota(result.pivot_rows.begin(), result.pivot_rows.end(), std::size_t{0});
  result.forward_error_bounds.resize(rhs_count);
  result.backward_error_bounds.resize(rhs_count);
  if (n == 0 || rhs_count == 0)
  {
    return result;
  }

  blas_int const order = detail::checked_blas_int(n);
  blas_int const kl = detail::checked_blas_int(coefficients.lower_bandwidth);
  blas_int const ku = detail::checked_blas_int(coefficients.upper_bandwidth);
  blas_int const nrhs = detail::checked_blas_int(rhs_count);
  blas_int const ldab = detail::checked_blas_int(coefficients.storage.rows());
  blas_int const ldafb = detail::checked_blas_int(result.factors.storage.rows());
  std::vector<blas_int> pivots(n, 0);
  std::vector<Scalar> row_scale(n, Scalar{});
  std::vector<Scalar> column_scale(n, Scalar{});
  std::vector<Scalar> work(static_cast<std::size_t>(std::max<blas_int>(1, 3 * order)), Scalar{});
  std::vector<blas_int> iwork(static_cast<std::size_t>(std::max<blas_int>(1, order)), 0);
  char const trans = detail::lapack_transpose(transpose);
  char equed = 'N';
  Scalar rcond{};
  Scalar* compact_band = coefficients.storage.data() + coefficients.lower_bandwidth;

  result.reciprocal_condition_below_machine_precision = uni20::lapack::gbsvx(
      'N', trans, order, kl, ku, nrhs, compact_band, ldab, result.factors.storage.data(), ldafb, pivots.data(), equed,
      row_scale.data(), column_scale.data(), right_hand_sides.data(), order, result.solution.data(), order, rcond,
      result.forward_error_bounds.data(), result.backward_error_bounds.data(), work.data(), iwork.data());
  result.equilibration = equed;
  result.reciprocal_condition = rcond;
  for (std::size_t index = 0; index < n; ++index)
  {
    if (pivots[index] <= 0 || static_cast<std::size_t>(pivots[index]) > n)
    {
      throw std::runtime_error("LAPACK gbsvx returned an invalid row pivot");
    }
    result.pivot_rows[index] = static_cast<std::size_t>(pivots[index] - 1);
  }
  return result;
}

/// \brief Compute real general-band row/column equilibration factors through LAPACK `gbequ`.
/// \details Returns row scales `R` and column scales `C` suitable for forming
///          `diag(R) * A * diag(C)`, along with LAPACK's scale-condition
///          diagnostics and maximum absolute matrix entry. The band matrix
///          must use the factor-storage layout returned by
///          `real_general_band_from_dense`.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param band_matrix Real square general-band matrix in LAPACK factor-storage layout.
/// \return Row and column scaling factors plus equilibration diagnostics.
template <uni20::LapackReal Scalar>
RealEquilibration<Scalar> real_general_band_equilibration(RealGeneralBandMatrix<Scalar> band_matrix)
{
  if (!std::cmp_equal(band_matrix.storage.cols(), band_matrix.order) ||
      std::cmp_less(band_matrix.storage.rows(), 2 * band_matrix.lower_bandwidth + band_matrix.upper_bandwidth + 1))
  {
    throw std::invalid_argument("real_general_band_equilibration received inconsistent band storage");
  }

  RealEquilibration<Scalar> result;
  result.row_scale.assign(band_matrix.order, Scalar{1});
  result.column_scale.assign(band_matrix.order, Scalar{1});
  if (band_matrix.order == 0)
  {
    return result;
  }

  blas_int const n = detail::checked_blas_int(band_matrix.order);
  blas_int const kl = detail::checked_blas_int(band_matrix.lower_bandwidth);
  blas_int const ku = detail::checked_blas_int(band_matrix.upper_bandwidth);
  blas_int const ldab = detail::checked_blas_int(band_matrix.storage.rows());
  Scalar* compact_band = band_matrix.storage.data() + band_matrix.lower_bandwidth;
  uni20::lapack::gbequ(false, n, n, kl, ku, compact_band, ldab, result.row_scale.data(), result.column_scale.data(),
                       result.row_condition, result.column_condition, result.max_abs);
  return result;
}

/// \brief Compute power-of-two real general-band equilibration factors through LAPACK `gbequb`.
/// \details Returns row and column scale factors constrained to radix powers
///          by LAPACK, along with scale-condition diagnostics and the maximum
///          absolute matrix entry.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param band_matrix Real square general-band matrix in LAPACK factor-storage layout.
/// \return Row and column scaling factors plus equilibration diagnostics.
template <uni20::LapackReal Scalar>
RealEquilibration<Scalar> real_general_band_power_of_two_equilibration(RealGeneralBandMatrix<Scalar> band_matrix)
{
  if (!std::cmp_equal(band_matrix.storage.cols(), band_matrix.order) ||
      std::cmp_less(band_matrix.storage.rows(), 2 * band_matrix.lower_bandwidth + band_matrix.upper_bandwidth + 1))
  {
    throw std::invalid_argument("real_general_band_power_of_two_equilibration received inconsistent band storage");
  }

  RealEquilibration<Scalar> result;
  result.row_scale.assign(band_matrix.order, Scalar{1});
  result.column_scale.assign(band_matrix.order, Scalar{1});
  if (band_matrix.order == 0)
  {
    return result;
  }

  blas_int const n = detail::checked_blas_int(band_matrix.order);
  blas_int const kl = detail::checked_blas_int(band_matrix.lower_bandwidth);
  blas_int const ku = detail::checked_blas_int(band_matrix.upper_bandwidth);
  blas_int const ldab = detail::checked_blas_int(band_matrix.storage.rows());
  Scalar* compact_band = band_matrix.storage.data() + band_matrix.lower_bandwidth;
  uni20::lapack::gbequ(true, n, n, kl, ku, compact_band, ldab, result.row_scale.data(), result.column_scale.data(),
                       result.row_condition, result.column_condition, result.max_abs);
  return result;
}

/// \brief Compute dense real row/column equilibration factors through LAPACK `geequ`.
/// \details Returns row scales `R` and column scales `C` suitable for forming
///          `diag(R) * A * diag(C)`, along with LAPACK's scale-condition
///          diagnostics and maximum absolute matrix entry.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real dense matrix.
/// \return Row and column scaling factors plus equilibration diagnostics.
template <uni20::LapackReal Scalar>
RealEquilibration<Scalar> real_equilibration(detail::ColumnMajorLapackMatrix<Scalar> matrix)
{
  RealEquilibration<Scalar> result;
  result.row_scale.assign(matrix.rows(), Scalar{1});
  result.column_scale.assign(matrix.cols(), Scalar{1});
  if (matrix.rows() == 0 || matrix.cols() == 0)
  {
    return result;
  }

  blas_int const rows = detail::checked_blas_int(matrix.rows());
  blas_int const cols = detail::checked_blas_int(matrix.cols());
  uni20::lapack::geequ(rows, cols, matrix.data(), rows, result.row_scale.data(), result.column_scale.data(),
                       result.row_condition, result.column_condition, result.max_abs);
  return result;
}

/// \brief Compute dense real SPD equilibration factors through LAPACK `poequ`.
/// \details Returns diagonal scales `S` suitable for forming
///          `diag(S) * A * diag(S)`, along with LAPACK's scale-condition
///          diagnostic and maximum absolute diagonal entry.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real symmetric positive definite square matrix.
/// \return Symmetric scaling factors plus equilibration diagnostics.
template <uni20::LapackReal Scalar>
RealSymmetricPositiveDefiniteEquilibration<Scalar>
real_symmetric_positive_definite_equilibration(detail::ColumnMajorLapackMatrix<Scalar> matrix)
{
  if (!std::cmp_equal(matrix.rows(), matrix.cols()))
  {
    throw std::invalid_argument("real_symmetric_positive_definite_equilibration requires a square matrix");
  }

  RealSymmetricPositiveDefiniteEquilibration<Scalar> result;
  result.scale.assign(matrix.rows(), Scalar{1});
  if (matrix.rows() == 0)
  {
    return result;
  }

  blas_int const order = detail::checked_blas_int(matrix.rows());
  uni20::lapack::poequ(order, matrix.data(), order, result.scale.data(), result.scale_condition, result.max_abs);
  return result;
}

/// \brief Compute a dense real SPD Cholesky factorization through LAPACK `potrf`.
/// \details Only the selected triangle of \p matrix is factorized. The
///          returned factors keep LAPACK's triangular storage.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real symmetric positive definite square matrix.
/// \param triangle Triangle of \p matrix supplied to LAPACK.
/// \return Cholesky factors and triangle metadata.
template <uni20::LapackReal Scalar>
RealSymmetricPositiveDefiniteFactorization<Scalar>
real_symmetric_positive_definite_factorization(detail::ColumnMajorLapackMatrix<Scalar> matrix,
                                               MatrixFill triangle = MatrixFill::Upper)
{
  if (!std::cmp_equal(matrix.rows(), matrix.cols()))
  {
    throw std::invalid_argument("real_symmetric_positive_definite_factorization requires a square matrix");
  }

  RealSymmetricPositiveDefiniteFactorization<Scalar> result;
  result.factors = std::move(matrix);
  result.triangle = triangle;
  if (result.factors.rows() == 0)
  {
    return result;
  }

  blas_int const order = detail::checked_blas_int(result.factors.rows());
  char const uplo = detail::lapack_uplo(triangle);
  uni20::lapack::potrf(uplo, order, result.factors.data(), order);
  return result;
}

/// \brief Pack a dense real SPD matrix into LAPACK symmetric-band storage.
/// \details Entries outside the requested triangle and bandwidth are dropped.
///          The returned storage can be passed directly to `pbsv`, `pbtrf`,
///          and `pbtrs`.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real square SPD matrix to pack.
/// \param bandwidth Number of stored offdiagonals.
/// \param triangle Triangle of \p matrix supplied to LAPACK.
/// \return SPD band matrix in LAPACK storage layout.
template <uni20::LapackReal Scalar>
RealSymmetricPositiveDefiniteBandMatrix<Scalar>
real_symmetric_positive_definite_band_from_dense(detail::ColumnMajorLapackMatrix<Scalar> const& matrix,
                                                 std::size_t bandwidth, MatrixFill triangle = MatrixFill::Upper)
{
  if (!std::cmp_equal(matrix.rows(), matrix.cols()))
  {
    throw std::invalid_argument("real_symmetric_positive_definite_band_from_dense requires a square matrix");
  }
  if (triangle != MatrixFill::Upper && triangle != MatrixFill::Lower)
  {
    throw std::invalid_argument("real_symmetric_positive_definite_band_from_dense requires upper or lower storage");
  }

  std::size_t const n = static_cast<std::size_t>(matrix.rows());
  RealSymmetricPositiveDefiniteBandMatrix<Scalar> result;
  result.order = n;
  result.bandwidth = bandwidth;
  result.triangle = triangle;
  result.storage = detail::ColumnMajorLapackMatrix<Scalar>(bandwidth + 1, n);

  for (std::size_t col = 0; col < n; ++col)
  {
    if (triangle == MatrixFill::Upper)
    {
      std::size_t const first_row = col > bandwidth ? col - bandwidth : 0;
      for (std::size_t row = first_row; row <= col; ++row)
      {
        result.storage[bandwidth + row - col, col] = matrix[row, col];
      }
    }
    else
    {
      std::size_t const last_row = std::min(n - 1, col + bandwidth);
      for (std::size_t row = col; row <= last_row; ++row)
      {
        result.storage[row - col, col] = matrix[row, col];
      }
    }
  }
  return result;
}

/// \brief Solve a real SPD band system through LAPACK `pbsv`.
/// \details Solves `A * X = B` for one or more dense right-hand sides. The
///          band matrix must use LAPACK symmetric positive-definite band
///          storage.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param coefficients Real square SPD band coefficient matrix.
/// \param right_hand_sides Dense right-hand-side matrix with matching row count.
/// \return Dense solution matrix.
template <uni20::LapackReal Scalar>
detail::ColumnMajorLapackMatrix<Scalar>
real_symmetric_positive_definite_band_solve(RealSymmetricPositiveDefiniteBandMatrix<Scalar> coefficients,
                                            detail::ColumnMajorLapackMatrix<Scalar> right_hand_sides)
{
  if (!std::cmp_equal(coefficients.storage.cols(), coefficients.order) ||
      std::cmp_less(coefficients.storage.rows(), coefficients.bandwidth + 1))
  {
    throw std::invalid_argument("real_symmetric_positive_definite_band_solve received inconsistent band storage");
  }
  if (!std::cmp_equal(right_hand_sides.rows(), coefficients.order))
  {
    throw std::invalid_argument("real_symmetric_positive_definite_band_solve received incompatible right-hand sides");
  }
  if (coefficients.order == 0 || right_hand_sides.cols() == 0)
  {
    return right_hand_sides;
  }

  blas_int const n = detail::checked_blas_int(coefficients.order);
  blas_int const kd = detail::checked_blas_int(coefficients.bandwidth);
  blas_int const nrhs = detail::checked_blas_int(right_hand_sides.cols());
  blas_int const ldab = detail::checked_blas_int(coefficients.storage.rows());
  char const uplo = detail::lapack_uplo(coefficients.triangle);
  uni20::lapack::pbsv(uplo, n, kd, nrhs, coefficients.storage.data(), ldab, right_hand_sides.data(), n);
  return right_hand_sides;
}

/// \brief Compute a real SPD band Cholesky factorization through LAPACK `pbtrf`.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real square SPD band matrix in LAPACK storage layout.
/// \return SPD band Cholesky factors and storage metadata.
template <uni20::LapackReal Scalar>
RealSymmetricPositiveDefiniteBandFactorization<Scalar>
real_symmetric_positive_definite_band_factorization(RealSymmetricPositiveDefiniteBandMatrix<Scalar> matrix)
{
  if (!std::cmp_equal(matrix.storage.cols(), matrix.order) ||
      std::cmp_less(matrix.storage.rows(), matrix.bandwidth + 1))
  {
    throw std::invalid_argument(
        "real_symmetric_positive_definite_band_factorization received inconsistent band storage");
  }
  if (matrix.triangle != MatrixFill::Upper && matrix.triangle != MatrixFill::Lower)
  {
    throw std::invalid_argument("real_symmetric_positive_definite_band_factorization requires upper or lower storage");
  }

  RealSymmetricPositiveDefiniteBandFactorization<Scalar> result;
  result.factors = std::move(matrix);
  if (result.factors.order == 0)
  {
    return result;
  }

  blas_int const n = detail::checked_blas_int(result.factors.order);
  blas_int const kd = detail::checked_blas_int(result.factors.bandwidth);
  blas_int const ldab = detail::checked_blas_int(result.factors.storage.rows());
  char const uplo = detail::lapack_uplo(result.factors.triangle);
  uni20::lapack::pbtrf(uplo, n, kd, result.factors.storage.data(), ldab);
  return result;
}

/// \brief Solve using an existing real SPD band Cholesky factorization through LAPACK `pbtrs`.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param factorization Factors returned by `real_symmetric_positive_definite_band_factorization`.
/// \param right_hand_sides Dense right-hand-side matrix with matching row count.
/// \return Dense solution matrix.
template <uni20::LapackReal Scalar>
detail::ColumnMajorLapackMatrix<Scalar>
real_symmetric_positive_definite_band_solve(RealSymmetricPositiveDefiniteBandFactorization<Scalar> const& factorization,
                                            detail::ColumnMajorLapackMatrix<Scalar> right_hand_sides)
{
  if (!std::cmp_equal(factorization.factors.storage.cols(), factorization.factors.order) ||
      std::cmp_less(factorization.factors.storage.rows(), factorization.factors.bandwidth + 1))
  {
    throw std::invalid_argument("real_symmetric_positive_definite_band_solve received inconsistent band factors");
  }
  if (!std::cmp_equal(right_hand_sides.rows(), factorization.factors.order))
  {
    throw std::invalid_argument("real_symmetric_positive_definite_band_solve received incompatible right-hand sides");
  }
  if (factorization.factors.order == 0 || right_hand_sides.cols() == 0)
  {
    return right_hand_sides;
  }

  detail::ColumnMajorLapackMatrix<Scalar> factors = factorization.factors.storage;
  blas_int const n = detail::checked_blas_int(factorization.factors.order);
  blas_int const kd = detail::checked_blas_int(factorization.factors.bandwidth);
  blas_int const nrhs = detail::checked_blas_int(right_hand_sides.cols());
  blas_int const ldab = detail::checked_blas_int(factors.rows());
  char const uplo = detail::lapack_uplo(factorization.factors.triangle);
  uni20::lapack::pbtrs(uplo, n, kd, nrhs, factors.data(), ldab, right_hand_sides.data(), n);
  return right_hand_sides;
}

/// \brief Compute the one-norm of a real SPD band matrix.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real SPD band matrix in LAPACK storage layout.
/// \return Matrix one-norm.
template <uni20::LapackReal Scalar>
Scalar real_symmetric_positive_definite_band_one_norm(RealSymmetricPositiveDefiniteBandMatrix<Scalar> const& matrix)
{
  if (!std::cmp_equal(matrix.storage.cols(), matrix.order) ||
      std::cmp_less(matrix.storage.rows(), matrix.bandwidth + 1))
  {
    throw std::invalid_argument("real_symmetric_positive_definite_band_one_norm received inconsistent band storage");
  }
  if (matrix.triangle != MatrixFill::Upper && matrix.triangle != MatrixFill::Lower)
  {
    throw std::invalid_argument("real_symmetric_positive_definite_band_one_norm requires upper or lower storage");
  }

  std::vector<Scalar> column_sums(matrix.order, Scalar{});
  for (std::size_t col = 0; col < matrix.order; ++col)
  {
    if (matrix.triangle == MatrixFill::Upper)
    {
      std::size_t const first_row = col > matrix.bandwidth ? col - matrix.bandwidth : 0;
      for (std::size_t row = first_row; row <= col; ++row)
      {
        Scalar const magnitude = detail::adl_abs(matrix.storage[matrix.bandwidth + row - col, col]);
        column_sums[col] += magnitude;
        if (row != col)
        {
          column_sums[row] += magnitude;
        }
      }
    }
    else
    {
      std::size_t const last_row = std::min(matrix.order - 1, col + matrix.bandwidth);
      for (std::size_t row = col; row <= last_row; ++row)
      {
        Scalar const magnitude = detail::adl_abs(matrix.storage[row - col, col]);
        column_sums[col] += magnitude;
        if (row != col)
        {
          column_sums[row] += magnitude;
        }
      }
    }
  }

  Scalar norm{};
  for (Scalar const value : column_sums)
  {
    norm = std::max(norm, value);
  }
  return norm;
}

/// \brief Estimate a real SPD band one-norm reciprocal condition number through LAPACK `pbcon`.
/// \details Uses an existing SPD band Cholesky factorization. The supplied
///          matrix norm must be the one-norm of the original unfactored band
///          matrix.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param factorization Factors returned by `real_symmetric_positive_definite_band_factorization`.
/// \param original_one_norm One-norm of the original coefficient matrix.
/// \return Estimate of `1 / (||A||_1 * ||inv(A)||_1)`.
template <uni20::LapackReal Scalar>
Scalar real_symmetric_positive_definite_band_one_norm_reciprocal_condition_number(
    RealSymmetricPositiveDefiniteBandFactorization<Scalar> const& factorization, Scalar original_one_norm)
{
  if (!std::cmp_equal(factorization.factors.storage.cols(), factorization.factors.order) ||
      std::cmp_less(factorization.factors.storage.rows(), factorization.factors.bandwidth + 1))
  {
    throw std::invalid_argument(
        "real_symmetric_positive_definite_band_one_norm_reciprocal_condition_number received inconsistent factors");
  }
  if (factorization.factors.triangle != MatrixFill::Upper && factorization.factors.triangle != MatrixFill::Lower)
  {
    throw std::invalid_argument(
        "real_symmetric_positive_definite_band_one_norm_reciprocal_condition_number requires upper or lower storage");
  }
  if (original_one_norm < Scalar{})
  {
    throw std::invalid_argument(
        "real_symmetric_positive_definite_band_one_norm_reciprocal_condition_number requires a nonnegative norm");
  }
  if (factorization.factors.order == 0)
  {
    return Scalar{1};
  }

  detail::ColumnMajorLapackMatrix<Scalar> factors = factorization.factors.storage;
  blas_int const n = detail::checked_blas_int(factorization.factors.order);
  blas_int const kd = detail::checked_blas_int(factorization.factors.bandwidth);
  blas_int const ldab = detail::checked_blas_int(factors.rows());
  char const uplo = detail::lapack_uplo(factorization.factors.triangle);
  std::vector<Scalar> work(static_cast<std::size_t>(std::max<blas_int>(1, 3 * n)), Scalar{});
  std::vector<blas_int> iwork(static_cast<std::size_t>(std::max<blas_int>(1, n)), 0);
  return uni20::lapack::pbcon(uplo, n, kd, factors.data(), ldab, original_one_norm, work.data(), iwork.data());
}

/// \brief Estimate a real SPD band one-norm reciprocal condition number from a matrix.
/// \details Computes the band one-norm, factors the matrix with LAPACK
///          `pbtrf`, and estimates the reciprocal condition number with
///          LAPACK `pbcon`.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real SPD band matrix in LAPACK storage layout.
/// \return Estimate of `1 / (||A||_1 * ||inv(A)||_1)`.
template <uni20::LapackReal Scalar>
Scalar real_symmetric_positive_definite_band_one_norm_reciprocal_condition_number(
    RealSymmetricPositiveDefiniteBandMatrix<Scalar> matrix)
{
  Scalar const norm = real_symmetric_positive_definite_band_one_norm(matrix);
  auto factorization = real_symmetric_positive_definite_band_factorization(std::move(matrix));
  return real_symmetric_positive_definite_band_one_norm_reciprocal_condition_number(factorization, norm);
}

/// \brief Solve and refine a real SPD band linear system through LAPACK `pbrfs`.
/// \details Computes a band Cholesky factorization, solves `A * X = B`, then
///          applies iterative refinement and returns LAPACK's forward and
///          backward error estimates.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param coefficients Real SPD band coefficient matrix.
/// \param right_hand_sides Dense right-hand-side matrix with matching row count.
/// \return Refined solution and LAPACK error estimates.
template <uni20::LapackReal Scalar>
RealRefinedLinearSolve<Scalar>
real_symmetric_positive_definite_band_refined_solve(RealSymmetricPositiveDefiniteBandMatrix<Scalar> coefficients,
                                                    detail::ColumnMajorLapackMatrix<Scalar> right_hand_sides)
{
  if (!std::cmp_equal(coefficients.storage.cols(), coefficients.order) ||
      std::cmp_less(coefficients.storage.rows(), coefficients.bandwidth + 1))
  {
    throw std::invalid_argument(
        "real_symmetric_positive_definite_band_refined_solve received inconsistent band storage");
  }
  if (!std::cmp_equal(right_hand_sides.rows(), coefficients.order))
  {
    throw std::invalid_argument(
        "real_symmetric_positive_definite_band_refined_solve received incompatible right-hand sides");
  }

  std::size_t const rhs_count = static_cast<std::size_t>(right_hand_sides.cols());
  RealRefinedLinearSolve<Scalar> result;
  result.solution = right_hand_sides;
  result.forward_error_bounds.resize(rhs_count);
  result.backward_error_bounds.resize(rhs_count);
  if (coefficients.order == 0 || rhs_count == 0)
  {
    return result;
  }

  auto factorization = real_symmetric_positive_definite_band_factorization(coefficients);
  result.solution = real_symmetric_positive_definite_band_solve(factorization, result.solution);

  blas_int const n = detail::checked_blas_int(coefficients.order);
  blas_int const kd = detail::checked_blas_int(coefficients.bandwidth);
  blas_int const nrhs = detail::checked_blas_int(rhs_count);
  blas_int const ldab = detail::checked_blas_int(coefficients.storage.rows());
  blas_int const ldafb = detail::checked_blas_int(factorization.factors.storage.rows());
  char const uplo = detail::lapack_uplo(coefficients.triangle);
  std::vector<Scalar> work(static_cast<std::size_t>(std::max<blas_int>(1, 3 * n)), Scalar{});
  std::vector<blas_int> iwork(static_cast<std::size_t>(std::max<blas_int>(1, n)), 0);

  uni20::lapack::pbrfs(uplo, n, kd, nrhs, coefficients.storage.data(), ldab, factorization.factors.storage.data(),
                       ldafb, right_hand_sides.data(), n, result.solution.data(), n, result.forward_error_bounds.data(),
                       result.backward_error_bounds.data(), work.data(), iwork.data());
  return result;
}

/// \brief Solve a real SPD band linear system with LAPACK `pbsvx` diagnostics.
/// \details Solves `A * X = B` and returns the solution, band Cholesky factors,
///          reciprocal condition estimate, componentwise backward errors,
///          forward error bounds, equilibration, and condition diagnostics
///          reported by LAPACK. The current wrapper requests a fresh
///          factorization (`FACT='N'`) and does not expose caller-supplied
///          equilibration modes.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param coefficients Real SPD band coefficient matrix.
/// \param right_hand_sides Dense right-hand-side matrix with matching row count.
/// \return Solution, factorization, and diagnostics.
template <uni20::LapackReal Scalar>
RealSymmetricPositiveDefiniteBandExpertLinearSolve<Scalar>
real_symmetric_positive_definite_band_expert_linear_solve(RealSymmetricPositiveDefiniteBandMatrix<Scalar> coefficients,
                                                          detail::ColumnMajorLapackMatrix<Scalar> right_hand_sides)
{
  if (!std::cmp_equal(coefficients.storage.cols(), coefficients.order) ||
      std::cmp_less(coefficients.storage.rows(), coefficients.bandwidth + 1))
  {
    throw std::invalid_argument(
        "real_symmetric_positive_definite_band_expert_linear_solve received inconsistent band storage");
  }
  if (!std::cmp_equal(right_hand_sides.rows(), coefficients.order))
  {
    throw std::invalid_argument(
        "real_symmetric_positive_definite_band_expert_linear_solve received incompatible right-hand sides");
  }

  std::size_t const n = coefficients.order;
  std::size_t const rhs_count = static_cast<std::size_t>(right_hand_sides.cols());
  RealSymmetricPositiveDefiniteBandExpertLinearSolve<Scalar> result;
  result.solution = detail::ColumnMajorLapackMatrix<Scalar>(n, rhs_count);
  result.factorization.factors.order = n;
  result.factorization.factors.bandwidth = coefficients.bandwidth;
  result.factorization.factors.triangle = coefficients.triangle;
  result.factorization.factors.storage =
      detail::ColumnMajorLapackMatrix<Scalar>(coefficients.storage.rows(), coefficients.storage.cols());
  result.forward_error_bounds.resize(rhs_count);
  result.backward_error_bounds.resize(rhs_count);
  result.scale.assign(n, Scalar{1});
  if (n == 0 || rhs_count == 0)
  {
    return result;
  }

  blas_int const order = detail::checked_blas_int(n);
  blas_int const kd = detail::checked_blas_int(coefficients.bandwidth);
  blas_int const nrhs = detail::checked_blas_int(rhs_count);
  blas_int const ldab = detail::checked_blas_int(coefficients.storage.rows());
  blas_int const ldafb = detail::checked_blas_int(result.factorization.factors.storage.rows());
  char const uplo = detail::lapack_uplo(coefficients.triangle);
  char equed = 'N';
  Scalar rcond{};
  std::vector<Scalar> work(static_cast<std::size_t>(std::max<blas_int>(1, 3 * order)), Scalar{});
  std::vector<blas_int> iwork(static_cast<std::size_t>(std::max<blas_int>(1, order)), 0);

  result.reciprocal_condition_below_machine_precision = uni20::lapack::pbsvx(
      'N', uplo, order, kd, nrhs, coefficients.storage.data(), ldab, result.factorization.factors.storage.data(), ldafb,
      equed, result.scale.data(), right_hand_sides.data(), order, result.solution.data(), order, rcond,
      result.forward_error_bounds.data(), result.backward_error_bounds.data(), work.data(), iwork.data());
  result.equilibration = equed;
  result.reciprocal_condition = rcond;
  return result;
}

/// \brief Extract real SPD tridiagonal storage from a dense matrix.
/// \details The diagonal and first subdiagonal are copied. Entries outside the
///          tridiagonal band are ignored.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real square SPD tridiagonal matrix.
/// \return Tridiagonal matrix storage suitable for LAPACK `ptsv` and `pttrf`.
template <uni20::LapackReal Scalar>
RealSymmetricPositiveDefiniteTridiagonalMatrix<Scalar>
real_symmetric_positive_definite_tridiagonal_from_dense(detail::ColumnMajorLapackMatrix<Scalar> const& matrix)
{
  if (!std::cmp_equal(matrix.rows(), matrix.cols()))
  {
    throw std::invalid_argument("real_symmetric_positive_definite_tridiagonal_from_dense requires a square matrix");
  }

  RealSymmetricPositiveDefiniteTridiagonalMatrix<Scalar> result;
  result.diagonal.resize(matrix.rows());
  if (matrix.rows() > 0)
  {
    result.offdiagonal.resize(matrix.rows() - 1);
  }
  for (uni20::index_type index = 0; index < matrix.rows(); ++index)
  {
    result.diagonal[index] = matrix[index, index];
    if (index + 1 < matrix.rows())
    {
      result.offdiagonal[index] = matrix[index + 1, index];
    }
  }
  return result;
}

/// \brief Solve a real SPD tridiagonal system through LAPACK `ptsv`.
/// \details Solves `A * X = B` for one or more dense right-hand sides.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param coefficients Real SPD tridiagonal coefficient matrix.
/// \param right_hand_sides Dense right-hand-side matrix with matching row count.
/// \return Dense solution matrix.
template <uni20::LapackReal Scalar>
detail::ColumnMajorLapackMatrix<Scalar>
real_symmetric_positive_definite_tridiagonal_solve(RealSymmetricPositiveDefiniteTridiagonalMatrix<Scalar> coefficients,
                                                   detail::ColumnMajorLapackMatrix<Scalar> right_hand_sides)
{
  std::size_t const order = coefficients.order();
  if (coefficients.offdiagonal.size() != (order == 0 ? 0 : order - 1))
  {
    throw std::invalid_argument(
        "real_symmetric_positive_definite_tridiagonal_solve received inconsistent tridiagonal storage");
  }
  if (!std::cmp_equal(right_hand_sides.rows(), order))
  {
    throw std::invalid_argument(
        "real_symmetric_positive_definite_tridiagonal_solve received incompatible right-hand sides");
  }
  if (order == 0 || right_hand_sides.cols() == 0)
  {
    return right_hand_sides;
  }

  blas_int const n = detail::checked_blas_int(order);
  blas_int const nrhs = detail::checked_blas_int(right_hand_sides.cols());
  Scalar* offdiagonal =
      coefficients.offdiagonal.empty() ? coefficients.diagonal.data() : coefficients.offdiagonal.data();
  uni20::lapack::ptsv(n, nrhs, coefficients.diagonal.data(), offdiagonal, right_hand_sides.data(), n);
  return right_hand_sides;
}

/// \brief Compute a real SPD tridiagonal factorization through LAPACK `pttrf`.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real SPD tridiagonal matrix.
/// \return SPD tridiagonal Cholesky factors.
template <uni20::LapackReal Scalar>
RealSymmetricPositiveDefiniteTridiagonalFactorization<Scalar>
real_symmetric_positive_definite_tridiagonal_factorization(
    RealSymmetricPositiveDefiniteTridiagonalMatrix<Scalar> matrix)
{
  std::size_t const order = matrix.order();
  if (matrix.offdiagonal.size() != (order == 0 ? 0 : order - 1))
  {
    throw std::invalid_argument(
        "real_symmetric_positive_definite_tridiagonal_factorization received inconsistent tridiagonal storage");
  }

  RealSymmetricPositiveDefiniteTridiagonalFactorization<Scalar> result;
  result.factors = std::move(matrix);
  if (order == 0)
  {
    return result;
  }

  blas_int const n = detail::checked_blas_int(order);
  Scalar* offdiagonal =
      result.factors.offdiagonal.empty() ? result.factors.diagonal.data() : result.factors.offdiagonal.data();
  uni20::lapack::pttrf(n, result.factors.diagonal.data(), offdiagonal);
  return result;
}

/// \brief Solve using an existing real SPD tridiagonal factorization through LAPACK `pttrs`.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param factorization Factors returned by `real_symmetric_positive_definite_tridiagonal_factorization`.
/// \param right_hand_sides Dense right-hand-side matrix with matching row count.
/// \return Dense solution matrix.
template <uni20::LapackReal Scalar>
detail::ColumnMajorLapackMatrix<Scalar> real_symmetric_positive_definite_tridiagonal_solve(
    RealSymmetricPositiveDefiniteTridiagonalFactorization<Scalar> const& factorization,
    detail::ColumnMajorLapackMatrix<Scalar> right_hand_sides)
{
  std::size_t const order = factorization.factors.order();
  if (factorization.factors.offdiagonal.size() != (order == 0 ? 0 : order - 1))
  {
    throw std::invalid_argument(
        "real_symmetric_positive_definite_tridiagonal_solve received inconsistent tridiagonal factors");
  }
  if (!std::cmp_equal(right_hand_sides.rows(), order))
  {
    throw std::invalid_argument(
        "real_symmetric_positive_definite_tridiagonal_solve received incompatible right-hand sides");
  }
  if (order == 0 || right_hand_sides.cols() == 0)
  {
    return right_hand_sides;
  }

  std::vector<Scalar> diagonal = factorization.factors.diagonal;
  std::vector<Scalar> offdiagonal = factorization.factors.offdiagonal;
  blas_int const n = detail::checked_blas_int(order);
  blas_int const nrhs = detail::checked_blas_int(right_hand_sides.cols());
  Scalar* offdiagonal_data = offdiagonal.empty() ? diagonal.data() : offdiagonal.data();
  uni20::lapack::pttrs(n, nrhs, diagonal.data(), offdiagonal_data, right_hand_sides.data(), n);
  return right_hand_sides;
}

/// \brief Compute the one-norm of a real SPD tridiagonal matrix.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real SPD tridiagonal matrix.
/// \return Matrix one-norm.
template <uni20::LapackReal Scalar>
Scalar real_symmetric_positive_definite_tridiagonal_one_norm(
    RealSymmetricPositiveDefiniteTridiagonalMatrix<Scalar> const& matrix)
{
  std::size_t const order = matrix.order();
  if (matrix.offdiagonal.size() != (order == 0 ? 0 : order - 1))
  {
    throw std::invalid_argument(
        "real_symmetric_positive_definite_tridiagonal_one_norm received inconsistent tridiagonal storage");
  }

  Scalar norm{};
  for (std::size_t col = 0; col < order; ++col)
  {
    Scalar column_sum = detail::adl_abs(matrix.diagonal[col]);
    if (col > 0)
    {
      column_sum += detail::adl_abs(matrix.offdiagonal[col - 1]);
    }
    if (col + 1 < order)
    {
      column_sum += detail::adl_abs(matrix.offdiagonal[col]);
    }
    norm = std::max(norm, column_sum);
  }
  return norm;
}

/// \brief Estimate a real SPD tridiagonal one-norm reciprocal condition number through LAPACK `ptcon`.
/// \details Uses an existing SPD tridiagonal factorization. The supplied matrix
///          norm must be the one-norm of the original unfactored tridiagonal
///          matrix.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param factorization Factors returned by `real_symmetric_positive_definite_tridiagonal_factorization`.
/// \param original_one_norm One-norm of the original coefficient matrix.
/// \return Estimate of `1 / (||A||_1 * ||inv(A)||_1)`.
template <uni20::LapackReal Scalar>
Scalar real_symmetric_positive_definite_tridiagonal_one_norm_reciprocal_condition_number(
    RealSymmetricPositiveDefiniteTridiagonalFactorization<Scalar> const& factorization, Scalar original_one_norm)
{
  std::size_t const order = factorization.factors.order();
  if (factorization.factors.offdiagonal.size() != (order == 0 ? 0 : order - 1))
  {
    throw std::invalid_argument(
        "real_symmetric_positive_definite_tridiagonal_one_norm_reciprocal_condition_number received inconsistent "
        "factors");
  }
  if (original_one_norm < Scalar{})
  {
    throw std::invalid_argument("real_symmetric_positive_definite_tridiagonal_one_norm_reciprocal_condition_number "
                                "requires a nonnegative norm");
  }
  if (order == 0)
  {
    return Scalar{1};
  }

  std::vector<Scalar> diagonal = factorization.factors.diagonal;
  std::vector<Scalar> offdiagonal = factorization.factors.offdiagonal;
  blas_int const n = detail::checked_blas_int(order);
  std::vector<Scalar> work(static_cast<std::size_t>(std::max<blas_int>(1, n)), Scalar{});
  Scalar* offdiagonal_data = offdiagonal.empty() ? diagonal.data() : offdiagonal.data();
  return uni20::lapack::ptcon(n, diagonal.data(), offdiagonal_data, original_one_norm, work.data());
}

/// \brief Estimate a real SPD tridiagonal one-norm reciprocal condition number from a matrix.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real SPD tridiagonal matrix.
/// \return Estimate of `1 / (||A||_1 * ||inv(A)||_1)`.
template <uni20::LapackReal Scalar>
Scalar real_symmetric_positive_definite_tridiagonal_one_norm_reciprocal_condition_number(
    RealSymmetricPositiveDefiniteTridiagonalMatrix<Scalar> matrix)
{
  Scalar const norm = real_symmetric_positive_definite_tridiagonal_one_norm(matrix);
  auto factorization = real_symmetric_positive_definite_tridiagonal_factorization(std::move(matrix));
  return real_symmetric_positive_definite_tridiagonal_one_norm_reciprocal_condition_number(factorization, norm);
}

/// \brief Solve and refine a real SPD tridiagonal linear system through LAPACK `ptrfs`.
/// \details Computes an initial solution using `pttrs`, then refines it with
///          `ptrfs` and returns LAPACK's forward and backward error estimates.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param coefficients Real SPD tridiagonal coefficient matrix.
/// \param right_hand_sides Dense right-hand-side matrix with matching row count.
/// \return Solution and error-bound diagnostics.
template <uni20::LapackReal Scalar>
RealRefinedLinearSolve<Scalar> real_symmetric_positive_definite_tridiagonal_refined_solve(
    RealSymmetricPositiveDefiniteTridiagonalMatrix<Scalar> coefficients,
    detail::ColumnMajorLapackMatrix<Scalar> right_hand_sides)
{
  std::size_t const order = coefficients.order();
  if (coefficients.offdiagonal.size() != (order == 0 ? 0 : order - 1))
  {
    throw std::invalid_argument(
        "real_symmetric_positive_definite_tridiagonal_refined_solve received inconsistent tridiagonal storage");
  }
  if (!std::cmp_equal(right_hand_sides.rows(), order))
  {
    throw std::invalid_argument(
        "real_symmetric_positive_definite_tridiagonal_refined_solve received incompatible right-hand sides");
  }

  std::size_t const rhs_count = static_cast<std::size_t>(right_hand_sides.cols());
  RealRefinedLinearSolve<Scalar> result;
  result.solution = right_hand_sides;
  result.forward_error_bounds.resize(rhs_count);
  result.backward_error_bounds.resize(rhs_count);
  if (order == 0 || rhs_count == 0)
  {
    return result;
  }

  auto factorization = real_symmetric_positive_definite_tridiagonal_factorization(coefficients);
  result.solution = real_symmetric_positive_definite_tridiagonal_solve(factorization, result.solution);

  blas_int const n = detail::checked_blas_int(order);
  blas_int const nrhs = detail::checked_blas_int(rhs_count);
  std::vector<Scalar> work(static_cast<std::size_t>(std::max<blas_int>(1, 2 * n)), Scalar{});
  Scalar* offdiagonal =
      coefficients.offdiagonal.empty() ? coefficients.diagonal.data() : coefficients.offdiagonal.data();
  Scalar* factor_offdiagonal = factorization.factors.offdiagonal.empty() ? factorization.factors.diagonal.data()
                                                                         : factorization.factors.offdiagonal.data();
  uni20::lapack::ptrfs(n, nrhs, coefficients.diagonal.data(), offdiagonal, factorization.factors.diagonal.data(),
                       factor_offdiagonal, right_hand_sides.data(), n, result.solution.data(), n,
                       result.forward_error_bounds.data(), result.backward_error_bounds.data(), work.data());
  return result;
}

/// \brief Solve a real SPD tridiagonal linear system with LAPACK `ptsvx` diagnostics.
/// \details Solves `A * X = B` and returns the solution, SPD tridiagonal
///          factors, reciprocal condition estimate, componentwise backward
///          errors, forward error bounds, and condition diagnostics reported by
///          LAPACK. The current wrapper requests a fresh factorization
///          (`FACT='N'`).
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param coefficients Real SPD tridiagonal coefficient matrix.
/// \param right_hand_sides Dense right-hand-side matrix with matching row count.
/// \return Solution, factorization, and diagnostics.
template <uni20::LapackReal Scalar>
RealSymmetricPositiveDefiniteTridiagonalExpertLinearSolve<Scalar>
real_symmetric_positive_definite_tridiagonal_expert_linear_solve(
    RealSymmetricPositiveDefiniteTridiagonalMatrix<Scalar> coefficients,
    detail::ColumnMajorLapackMatrix<Scalar> right_hand_sides)
{
  std::size_t const order = coefficients.order();
  if (coefficients.offdiagonal.size() != (order == 0 ? 0 : order - 1))
  {
    throw std::invalid_argument(
        "real_symmetric_positive_definite_tridiagonal_expert_linear_solve received inconsistent tridiagonal storage");
  }
  if (!std::cmp_equal(right_hand_sides.rows(), order))
  {
    throw std::invalid_argument(
        "real_symmetric_positive_definite_tridiagonal_expert_linear_solve received incompatible right-hand sides");
  }

  std::size_t const rhs_count = static_cast<std::size_t>(right_hand_sides.cols());
  RealSymmetricPositiveDefiniteTridiagonalExpertLinearSolve<Scalar> result;
  result.solution = detail::ColumnMajorLapackMatrix<Scalar>(right_hand_sides.rows(), rhs_count);
  result.factorization.factors.diagonal.resize(order);
  result.factorization.factors.offdiagonal.resize(order == 0 ? 0 : order - 1);
  result.forward_error_bounds.resize(rhs_count);
  result.backward_error_bounds.resize(rhs_count);
  if (order == 0 || rhs_count == 0)
  {
    return result;
  }

  blas_int const n = detail::checked_blas_int(order);
  blas_int const nrhs = detail::checked_blas_int(rhs_count);
  std::vector<Scalar> work(static_cast<std::size_t>(std::max<blas_int>(1, 2 * n)), Scalar{});
  Scalar* offdiagonal =
      coefficients.offdiagonal.empty() ? coefficients.diagonal.data() : coefficients.offdiagonal.data();
  Scalar* factor_offdiagonal = result.factorization.factors.offdiagonal.empty()
                                   ? result.factorization.factors.diagonal.data()
                                   : result.factorization.factors.offdiagonal.data();
  result.reciprocal_condition_below_machine_precision = uni20::lapack::ptsvx(
      'N', n, nrhs, coefficients.diagonal.data(), offdiagonal, result.factorization.factors.diagonal.data(),
      factor_offdiagonal, right_hand_sides.data(), n, result.solution.data(), n, result.reciprocal_condition,
      result.forward_error_bounds.data(), result.backward_error_bounds.data(), work.data());
  return result;
}

/// \brief Compute a dense real pivoted Cholesky factorization through LAPACK `pstrf`.
/// \details Computes a complete-pivoting Cholesky factorization of a symmetric
///          positive semidefinite matrix. For upper storage, LAPACK returns
///          `P^T A P = U^T U`; for lower storage, it returns
///          `P^T A P = L L^T`. A nonnegative \p tolerance is used as the
///          stopping threshold; a negative value requests LAPACK's default
///          `n * eps * max(diag(A))` threshold.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real symmetric positive semidefinite square matrix.
/// \param tolerance Pivot stopping threshold, or negative for LAPACK's default.
/// \param triangle Triangle of \p matrix supplied to LAPACK.
/// \return Pivoted Cholesky factors, 0-based pivot order, rank, and rank-deficiency flag.
template <uni20::LapackReal Scalar>
RealPivotedCholeskyFactorization<Scalar>
real_pivoted_cholesky_factorization(detail::ColumnMajorLapackMatrix<Scalar> matrix, Scalar tolerance = Scalar{-1},
                                    MatrixFill triangle = MatrixFill::Upper)
{
  if (!std::cmp_equal(matrix.rows(), matrix.cols()))
  {
    throw std::invalid_argument("real_pivoted_cholesky_factorization requires a square matrix");
  }

  std::size_t const n = static_cast<std::size_t>(matrix.rows());
  RealPivotedCholeskyFactorization<Scalar> result;
  result.factors = std::move(matrix);
  result.pivot_order.resize(n);
  std::iota(result.pivot_order.begin(), result.pivot_order.end(), std::size_t{0});
  result.triangle = triangle;
  result.rank = n;
  if (n == 0)
  {
    return result;
  }

  blas_int const order = detail::checked_blas_int(n);
  char const uplo = detail::lapack_uplo(triangle);
  std::vector<blas_int> pivots(n, 0);
  std::vector<Scalar> work(static_cast<std::size_t>(std::max<blas_int>(1, 2 * order)), Scalar{});
  blas_int rank = 0;
  result.rank_deficient =
      uni20::lapack::pstrf(uplo, order, result.factors.data(), order, pivots.data(), rank, tolerance, work.data());
  if (rank < 0 || rank > order)
  {
    throw std::runtime_error("LAPACK pstrf returned an invalid numerical rank");
  }
  result.rank = static_cast<std::size_t>(rank);
  for (std::size_t index = 0; index < n; ++index)
  {
    if (pivots[index] <= 0 || static_cast<std::size_t>(pivots[index]) > n)
    {
      throw std::runtime_error("LAPACK pstrf returned an invalid pivot order");
    }
    result.pivot_order[index] = static_cast<std::size_t>(pivots[index] - 1);
  }
  return result;
}

/// \brief Solve a dense real linear system through LAPACK `gesv`.
/// \details Solves `A * X = B` for one or more dense right-hand sides. The
///          coefficient and right-hand-side matrices are copied by value and
///          may be overwritten by LAPACK. The returned matrix stores `X`.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param coefficients Real square coefficient matrix.
/// \param right_hand_sides Dense right-hand-side matrix with matching row count.
/// \return Dense solution matrix.
template <uni20::LapackReal Scalar>
detail::ColumnMajorLapackMatrix<Scalar>
real_dense_solve_linear_system(detail::ColumnMajorLapackMatrix<Scalar> coefficients,
                               detail::ColumnMajorLapackMatrix<Scalar> right_hand_sides)
{
  if (!std::cmp_equal(coefficients.rows(), coefficients.cols()))
  {
    throw std::invalid_argument("real_dense_solve_linear_system requires a square coefficient matrix");
  }
  if (!std::cmp_equal(coefficients.rows(), right_hand_sides.rows()))
  {
    throw std::invalid_argument("real_dense_solve_linear_system received incompatible right-hand sides");
  }

  std::size_t const n = static_cast<std::size_t>(coefficients.rows());
  if (n == 0)
  {
    return right_hand_sides;
  }

  blas_int const order = detail::checked_blas_int(n);
  blas_int const nrhs = detail::checked_blas_int(right_hand_sides.cols());
  std::vector<blas_int> pivots(n);
  uni20::lapack::gesv(order, nrhs, coefficients.data(), order, pivots.data(), right_hand_sides.data(), order);
  return right_hand_sides;
}

/// \brief Solve a dense real linear system with LAPACK `gesvx` diagnostics.
/// \details Solves `A * X = B` and returns the solution, LU factors, pivot
///          metadata, reciprocal condition estimate, componentwise backward
///          errors, forward error bounds, and equilibration/condition
///          diagnostics reported by LAPACK. The current wrapper requests a
///          fresh factorization (`FACT='N'`) and does not expose LAPACK's
///          transpose or caller-supplied equilibration modes.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param coefficients Real square coefficient matrix.
/// \param right_hand_sides Dense right-hand-side matrix with matching row count.
/// \return Dense solution and LAPACK expert-driver diagnostics.
template <uni20::LapackReal Scalar>
RealExpertLinearSolve<Scalar> real_expert_linear_solve(detail::ColumnMajorLapackMatrix<Scalar> coefficients,
                                                       detail::ColumnMajorLapackMatrix<Scalar> right_hand_sides)
{
  if (!std::cmp_equal(coefficients.rows(), coefficients.cols()))
  {
    throw std::invalid_argument("real_expert_linear_solve requires a square coefficient matrix");
  }
  if (!std::cmp_equal(coefficients.rows(), right_hand_sides.rows()))
  {
    throw std::invalid_argument("real_expert_linear_solve received incompatible right-hand sides");
  }

  std::size_t const n = static_cast<std::size_t>(coefficients.rows());
  std::size_t const rhs_count = static_cast<std::size_t>(right_hand_sides.cols());
  RealExpertLinearSolve<Scalar> result;
  result.solution = detail::ColumnMajorLapackMatrix<Scalar>(n, rhs_count);
  result.factors = detail::ColumnMajorLapackMatrix<Scalar>(n, n);
  result.pivot_rows.resize(n);
  std::iota(result.pivot_rows.begin(), result.pivot_rows.end(), std::size_t{0});
  result.forward_error_bounds.resize(rhs_count);
  result.backward_error_bounds.resize(rhs_count);
  if (n == 0 || rhs_count == 0)
  {
    return result;
  }

  blas_int const order = detail::checked_blas_int(n);
  blas_int const nrhs = detail::checked_blas_int(rhs_count);
  std::vector<blas_int> pivots(n, 0);
  std::vector<Scalar> row_scale(n, Scalar{});
  std::vector<Scalar> column_scale(n, Scalar{});
  std::vector<Scalar> work(static_cast<std::size_t>(4 * order), Scalar{});
  std::vector<blas_int> iwork(n, 0);
  char equed = 'N';
  Scalar rcond{};

  result.reciprocal_condition_below_machine_precision = uni20::lapack::gesvx(
      'N', 'N', order, nrhs, coefficients.data(), order, result.factors.data(), order, pivots.data(), equed,
      row_scale.data(), column_scale.data(), right_hand_sides.data(), order, result.solution.data(), order, rcond,
      result.forward_error_bounds.data(), result.backward_error_bounds.data(), work.data(), iwork.data());
  result.equilibration = equed;
  result.reciprocal_condition = rcond;
  for (std::size_t index = 0; index < n; ++index)
  {
    if (pivots[index] <= 0 || static_cast<std::size_t>(pivots[index]) > n)
    {
      throw std::runtime_error("LAPACK gesvx returned an invalid row pivot");
    }
    result.pivot_rows[index] = static_cast<std::size_t>(pivots[index] - 1);
  }
  return result;
}

/// \brief Compute a dense real LU factorization through LAPACK `getrf`.
/// \details Factorizes a square matrix as `P * A = L * U`. Pivot rows are
///          returned as 0-based indices.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real square matrix to factor.
/// \return LU factors and row-pivot metadata.
template <uni20::LapackReal Scalar>
RealLuFactorization<Scalar> real_lu_factorization(detail::ColumnMajorLapackMatrix<Scalar> matrix)
{
  if (!std::cmp_equal(matrix.rows(), matrix.cols()))
  {
    throw std::invalid_argument("real_lu_factorization requires a square matrix");
  }

  std::size_t const n = static_cast<std::size_t>(matrix.rows());
  RealLuFactorization<Scalar> result;
  result.factors = std::move(matrix);
  result.pivot_rows.resize(n);
  std::iota(result.pivot_rows.begin(), result.pivot_rows.end(), std::size_t{0});
  if (n == 0)
  {
    return result;
  }

  blas_int const order = detail::checked_blas_int(n);
  std::vector<blas_int> pivots(n);
  uni20::lapack::getrf(order, order, result.factors.data(), order, pivots.data());
  for (std::size_t index = 0; index < n; ++index)
  {
    if (pivots[index] <= 0 || static_cast<std::size_t>(pivots[index]) > n)
    {
      throw std::runtime_error("LAPACK getrf returned an invalid row pivot");
    }
    result.pivot_rows[index] = static_cast<std::size_t>(pivots[index] - 1);
  }
  return result;
}

/// \brief Solve using an existing dense real LU factorization through LAPACK `getrs`.
/// \details Solves `op(A) * X = B`, where `A` is represented by the supplied
///          LU factorization.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param factorization LU factors returned by `real_lu_factorization`.
/// \param right_hand_sides Dense right-hand-side matrix with matching row count.
/// \param transpose Matrix operation applied to the original coefficient matrix.
/// \return Dense solution matrix.
template <uni20::LapackReal Scalar>
detail::ColumnMajorLapackMatrix<Scalar> real_lu_solve(RealLuFactorization<Scalar> const& factorization,
                                                      detail::ColumnMajorLapackMatrix<Scalar> right_hand_sides,
                                                      MatrixTranspose transpose = MatrixTranspose::None)
{
  if (!std::cmp_equal(factorization.factors.rows(), factorization.factors.cols()))
  {
    throw std::invalid_argument("real_lu_solve requires square LU factors");
  }
  std::size_t const n = static_cast<std::size_t>(factorization.factors.rows());
  if (factorization.pivot_rows.size() != n)
  {
    throw std::invalid_argument("real_lu_solve received inconsistent pivot metadata");
  }
  if (!std::cmp_equal(right_hand_sides.rows(), n))
  {
    throw std::invalid_argument("real_lu_solve received incompatible right-hand sides");
  }
  if (n == 0 || right_hand_sides.cols() == 0)
  {
    return right_hand_sides;
  }

  detail::ColumnMajorLapackMatrix<Scalar> factors = factorization.factors;
  std::vector<blas_int> pivots(n);
  for (std::size_t index = 0; index < n; ++index)
  {
    if (factorization.pivot_rows[index] >= n)
    {
      throw std::invalid_argument("real_lu_solve received an out-of-range row pivot");
    }
    pivots[index] = detail::checked_blas_int(factorization.pivot_rows[index] + 1);
  }

  blas_int const order = detail::checked_blas_int(n);
  blas_int const nrhs = detail::checked_blas_int(right_hand_sides.cols());
  char const trans = detail::lapack_transpose(transpose);
  uni20::lapack::getrs(trans, order, nrhs, factors.data(), order, pivots.data(), right_hand_sides.data(), order);
  return right_hand_sides;
}

/// \brief Solve and refine a dense real linear system through LAPACK `gerfs`.
/// \details Computes an LU factorization, solves `op(A) * X = B`, then applies
///          iterative refinement and returns LAPACK's forward and backward
///          error estimates. The original coefficient matrix is retained for
///          the refinement residual calculation.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param coefficients Real square coefficient matrix.
/// \param right_hand_sides Dense right-hand-side matrix with matching row count.
/// \param transpose Matrix operation applied to the coefficient matrix.
/// \return Refined solution and LAPACK error estimates.
template <uni20::LapackReal Scalar>
RealRefinedLinearSolve<Scalar> real_refined_linear_solve(detail::ColumnMajorLapackMatrix<Scalar> coefficients,
                                                         detail::ColumnMajorLapackMatrix<Scalar> right_hand_sides,
                                                         MatrixTranspose transpose = MatrixTranspose::None)
{
  if (!std::cmp_equal(coefficients.rows(), coefficients.cols()))
  {
    throw std::invalid_argument("real_refined_linear_solve requires a square coefficient matrix");
  }
  if (!std::cmp_equal(coefficients.rows(), right_hand_sides.rows()))
  {
    throw std::invalid_argument("real_refined_linear_solve received incompatible right-hand sides");
  }

  std::size_t const n = static_cast<std::size_t>(coefficients.rows());
  std::size_t const rhs_count = static_cast<std::size_t>(right_hand_sides.cols());
  RealRefinedLinearSolve<Scalar> result;
  result.solution = right_hand_sides;
  result.forward_error_bounds.resize(rhs_count);
  result.backward_error_bounds.resize(rhs_count);
  if (n == 0 || rhs_count == 0)
  {
    return result;
  }

  detail::ColumnMajorLapackMatrix<Scalar> factors = coefficients;
  blas_int const order = detail::checked_blas_int(n);
  blas_int const nrhs = detail::checked_blas_int(rhs_count);
  std::vector<blas_int> pivots(n);
  uni20::lapack::getrf(order, order, factors.data(), order, pivots.data());

  char const trans = detail::lapack_transpose(transpose);
  uni20::lapack::getrs(trans, order, nrhs, factors.data(), order, pivots.data(), result.solution.data(), order);

  std::vector<Scalar> work(static_cast<std::size_t>(std::max<blas_int>(1, 3 * order)), Scalar{});
  std::vector<blas_int> iwork(static_cast<std::size_t>(std::max<blas_int>(1, order)), 0);
  uni20::lapack::gerfs(trans, order, nrhs, coefficients.data(), order, factors.data(), order, pivots.data(),
                       right_hand_sides.data(), order, result.solution.data(), order,
                       result.forward_error_bounds.data(), result.backward_error_bounds.data(), work.data(),
                       iwork.data());
  return result;
}

/// \brief Estimate the one-norm reciprocal condition number from an LU factorization.
/// \details Uses LAPACK `gecon` on an existing LU factorization. The supplied
///          matrix norm must be the one-norm of the original unfactored matrix.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param factorization LU factors returned by `real_lu_factorization`.
/// \param original_one_norm One-norm of the original coefficient matrix.
/// \return Estimate of `1 / (||A||_1 * ||inv(A)||_1)`.
template <uni20::LapackReal Scalar>
Scalar real_lu_one_norm_reciprocal_condition_number(RealLuFactorization<Scalar> const& factorization,
                                                    Scalar original_one_norm)
{
  if (!std::cmp_equal(factorization.factors.rows(), factorization.factors.cols()))
  {
    throw std::invalid_argument("real_lu_one_norm_reciprocal_condition_number requires square LU factors");
  }
  if (original_one_norm < Scalar{})
  {
    throw std::invalid_argument("real_lu_one_norm_reciprocal_condition_number requires a nonnegative matrix norm");
  }

  std::size_t const n = static_cast<std::size_t>(factorization.factors.rows());
  if (n == 0)
  {
    return Scalar{1};
  }

  detail::ColumnMajorLapackMatrix<Scalar> factors = factorization.factors;
  blas_int const order = detail::checked_blas_int(n);
  std::vector<Scalar> work(static_cast<std::size_t>(4 * order), Scalar{});
  std::vector<blas_int> iwork(n);
  return uni20::lapack::gecon('1', order, factors.data(), order, original_one_norm, work.data(), iwork.data());
}

/// \brief Estimate the one-norm reciprocal condition number of a dense real matrix.
/// \details Computes the one-norm, factors the matrix with LAPACK `getrf`, and
///          estimates the reciprocal condition number with LAPACK `gecon`.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real square matrix.
/// \return Estimate of `1 / (||A||_1 * ||inv(A)||_1)`.
template <uni20::LapackReal Scalar>
Scalar real_one_norm_reciprocal_condition_number(detail::ColumnMajorLapackMatrix<Scalar> matrix)
{
  Scalar const norm = uni20::linalg::matrix_norm_host(matrix, uni20::linalg::MatrixNorm::One);
  auto factorization = real_lu_factorization(std::move(matrix));
  return real_lu_one_norm_reciprocal_condition_number(factorization, norm);
}

/// \brief Solve a dense real triangular system through LAPACK `trtrs`.
/// \details Solves `op(A) * X = B` for one or more dense right-hand sides,
///          where `A` is triangular and `op(A)` is selected by \p transpose.
///          The coefficient and right-hand-side matrices are copied by value
///          and may be overwritten by LAPACK. The returned matrix stores `X`.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param coefficients Real square triangular coefficient matrix.
/// \param right_hand_sides Dense right-hand-side matrix with matching row count.
/// \param triangle Triangle of \p coefficients supplied to LAPACK.
/// \param transpose Matrix operation applied to \p coefficients before solving.
/// \param unit_diagonal Whether the triangular diagonal is implicitly unit.
/// \return Dense solution matrix.
template <uni20::LapackReal Scalar>
detail::ColumnMajorLapackMatrix<Scalar>
real_triangular_solve(detail::ColumnMajorLapackMatrix<Scalar> coefficients,
                      detail::ColumnMajorLapackMatrix<Scalar> right_hand_sides, MatrixFill triangle = MatrixFill::Upper,
                      MatrixTranspose transpose = MatrixTranspose::None, bool unit_diagonal = false)
{
  if (!std::cmp_equal(coefficients.rows(), coefficients.cols()))
  {
    throw std::invalid_argument("real_triangular_solve requires a square coefficient matrix");
  }
  if (!std::cmp_equal(coefficients.rows(), right_hand_sides.rows()))
  {
    throw std::invalid_argument("real_triangular_solve received incompatible right-hand sides");
  }

  std::size_t const n = static_cast<std::size_t>(coefficients.rows());
  if (n == 0 || right_hand_sides.cols() == 0)
  {
    return right_hand_sides;
  }

  blas_int const order = detail::checked_blas_int(n);
  blas_int const nrhs = detail::checked_blas_int(right_hand_sides.cols());
  char const uplo = detail::lapack_uplo(triangle);
  char const trans = detail::lapack_transpose(transpose);
  char const diag = unit_diagonal ? 'U' : 'N';
  uni20::lapack::trtrs(uplo, trans, diag, order, nrhs, coefficients.data(), order, right_hand_sides.data(), order);
  return right_hand_sides;
}

/// \brief Solve and refine a dense real triangular system through LAPACK `trrfs`.
/// \details Solves `op(A) * X = B` with `trtrs`, then asks LAPACK `trrfs` for
///          forward and backward error estimates. The coefficient matrix is
///          triangular and is not factorized.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param coefficients Real square triangular coefficient matrix.
/// \param right_hand_sides Dense right-hand-side matrix with matching row count.
/// \param triangle Triangle of \p coefficients supplied to LAPACK.
/// \param transpose Matrix operation applied to \p coefficients before solving.
/// \param unit_diagonal Whether the triangular diagonal is implicitly unit.
/// \return Refined solution and LAPACK error estimates.
template <uni20::LapackReal Scalar>
RealRefinedLinearSolve<Scalar> real_triangular_refined_solve(detail::ColumnMajorLapackMatrix<Scalar> coefficients,
                                                             detail::ColumnMajorLapackMatrix<Scalar> right_hand_sides,
                                                             MatrixFill triangle = MatrixFill::Upper,
                                                             MatrixTranspose transpose = MatrixTranspose::None,
                                                             bool unit_diagonal = false)
{
  if (!std::cmp_equal(coefficients.rows(), coefficients.cols()))
  {
    throw std::invalid_argument("real_triangular_refined_solve requires a square coefficient matrix");
  }
  if (!std::cmp_equal(coefficients.rows(), right_hand_sides.rows()))
  {
    throw std::invalid_argument("real_triangular_refined_solve received incompatible right-hand sides");
  }

  std::size_t const n = static_cast<std::size_t>(coefficients.rows());
  std::size_t const rhs_count = static_cast<std::size_t>(right_hand_sides.cols());
  RealRefinedLinearSolve<Scalar> result;
  result.solution = right_hand_sides;
  result.forward_error_bounds.resize(rhs_count);
  result.backward_error_bounds.resize(rhs_count);
  if (n == 0 || rhs_count == 0)
  {
    return result;
  }

  blas_int const order = detail::checked_blas_int(n);
  blas_int const nrhs = detail::checked_blas_int(rhs_count);
  char const uplo = detail::lapack_uplo(triangle);
  char const trans = detail::lapack_transpose(transpose);
  char const diag = unit_diagonal ? 'U' : 'N';
  uni20::lapack::trtrs(uplo, trans, diag, order, nrhs, coefficients.data(), order, result.solution.data(), order);

  std::vector<Scalar> work(static_cast<std::size_t>(std::max<blas_int>(1, 3 * order)), Scalar{});
  std::vector<blas_int> iwork(static_cast<std::size_t>(std::max<blas_int>(1, order)), 0);
  uni20::lapack::trrfs(uplo, trans, diag, order, nrhs, coefficients.data(), order, right_hand_sides.data(), order,
                       result.solution.data(), order, result.forward_error_bounds.data(),
                       result.backward_error_bounds.data(), work.data(), iwork.data());
  return result;
}

/// \brief Invert a dense real triangular matrix through LAPACK `trtri`.
/// \details The input matrix is copied by value and overwritten by LAPACK.
///          Entries outside the selected triangle are zeroed before returning
///          so the result is an ordinary dense triangular inverse.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real square triangular matrix to invert.
/// \param triangle Triangle of \p matrix supplied to LAPACK.
/// \param unit_diagonal Whether the triangular diagonal is implicitly unit.
/// \return Dense triangular inverse matrix.
template <uni20::LapackReal Scalar>
detail::ColumnMajorLapackMatrix<Scalar> real_triangular_inverse(detail::ColumnMajorLapackMatrix<Scalar> matrix,
                                                                MatrixFill triangle = MatrixFill::Upper,
                                                                bool unit_diagonal = false)
{
  if (!std::cmp_equal(matrix.rows(), matrix.cols()))
  {
    throw std::invalid_argument("real_triangular_inverse requires a square matrix");
  }

  std::size_t const n = static_cast<std::size_t>(matrix.rows());
  if (n == 0)
  {
    return matrix;
  }

  blas_int const order = detail::checked_blas_int(n);
  char const uplo = detail::lapack_uplo(triangle);
  char const diag = unit_diagonal ? 'U' : 'N';
  uni20::lapack::trtri(uplo, diag, order, matrix.data(), order);

  for (std::size_t row = 0; row < n; ++row)
  {
    for (std::size_t col = 0; col < n; ++col)
    {
      bool const selected = triangle == MatrixFill::Upper ? row <= col : row >= col;
      if (!selected)
      {
        matrix[row, col] = Scalar{};
      }
    }
    if (unit_diagonal)
    {
      matrix[row, row] = Scalar{1};
    }
  }
  return matrix;
}

/// \brief Estimate a dense real triangular one-norm reciprocal condition number through LAPACK `trcon`.
/// \details The input matrix is copied by value because LAPACK accepts a mutable
///          pointer. Only the selected triangle is read. The estimate is for
///          `1 / (||A||_1 * ||inv(A)||_1)`.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real square triangular matrix.
/// \param triangle Triangle of \p matrix supplied to LAPACK.
/// \param unit_diagonal Whether the triangular diagonal is implicitly unit.
/// \return Estimated one-norm reciprocal condition number.
template <uni20::LapackReal Scalar>
Scalar real_triangular_one_norm_reciprocal_condition_number(detail::ColumnMajorLapackMatrix<Scalar> matrix,
                                                            MatrixFill triangle = MatrixFill::Upper,
                                                            bool unit_diagonal = false)
{
  if (!std::cmp_equal(matrix.rows(), matrix.cols()))
  {
    throw std::invalid_argument("real_triangular_one_norm_reciprocal_condition_number requires a square matrix");
  }

  std::size_t const n = static_cast<std::size_t>(matrix.rows());
  if (n == 0)
  {
    return Scalar{1};
  }

  blas_int const order = detail::checked_blas_int(n);
  char const uplo = detail::lapack_uplo(triangle);
  char const diag = unit_diagonal ? 'U' : 'N';
  std::vector<Scalar> work(static_cast<std::size_t>(std::max<blas_int>(1, 3 * order)), Scalar{});
  std::vector<blas_int> iwork(static_cast<std::size_t>(std::max<blas_int>(1, order)), 0);
  return uni20::lapack::trcon('1', uplo, diag, order, matrix.data(), order, work.data(), iwork.data());
}

/// \brief Solve a dense real Sylvester equation through LAPACK `trsyl`.
///
/// \details Solves
///          `op(A) * X + sign * X * op(B) = scale * C`.
///          LAPACK overwrites `C` with `X`; this wrapper returns the solution,
///          the scale factor, and whether LAPACK perturbed nearly common
///          eigenvalues while solving.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param left Real square matrix `A`.
/// \param right Real square matrix `B`.
/// \param rhs Right-hand side matrix `C` with dimensions `A.rows()` by `B.rows()`.
/// \param sign Equation sign, either `+1` or `-1`.
/// \param transpose_left Operation applied to `A`.
/// \param transpose_right Operation applied to `B`.
/// \return Dense Sylvester solution, scale factor, and perturbation diagnostic.
template <uni20::LapackReal Scalar>
RealSylvesterSolution<Scalar> real_sylvester_solve(detail::ColumnMajorLapackMatrix<Scalar> left,
                                                   detail::ColumnMajorLapackMatrix<Scalar> right,
                                                   detail::ColumnMajorLapackMatrix<Scalar> rhs, int sign = 1,
                                                   MatrixTranspose transpose_left = MatrixTranspose::None,
                                                   MatrixTranspose transpose_right = MatrixTranspose::None)
{
  if (!std::cmp_equal(left.rows(), left.cols()))
  {
    throw std::invalid_argument("real_sylvester_solve requires a square left matrix");
  }
  if (!std::cmp_equal(right.rows(), right.cols()))
  {
    throw std::invalid_argument("real_sylvester_solve requires a square right matrix");
  }
  if (!std::cmp_equal(rhs.rows(), left.rows()) || !std::cmp_equal(rhs.cols(), right.rows()))
  {
    throw std::invalid_argument("real_sylvester_solve received incompatible right-hand side dimensions");
  }
  if (sign != 1 && sign != -1)
  {
    throw std::invalid_argument("real_sylvester_solve sign must be +1 or -1");
  }

  RealSylvesterSolution<Scalar> result;
  result.solution = std::move(rhs);
  if (left.rows() == 0 || right.rows() == 0)
  {
    return result;
  }

  blas_int const m = detail::checked_blas_int(left.rows());
  blas_int const n = detail::checked_blas_int(right.rows());
  char const trans_a = detail::lapack_transpose(transpose_left);
  char const trans_b = detail::lapack_transpose(transpose_right);
  Scalar scale = Scalar{1};
  result.separation_perturbed = uni20::lapack::trsyl(trans_a, trans_b, static_cast<blas_int>(sign), m, n, left.data(),
                                                     m, right.data(), n, result.solution.data(), m, scale);
  result.scale = scale;
  return result;
}

/// \brief Invert a dense real square matrix through LAPACK `getrf` and `getri`.
/// \details The input matrix is copied by value and overwritten by LAPACK.
///          The returned matrix stores the inverse.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real square matrix to invert.
/// \return Dense inverse matrix.
template <uni20::LapackReal Scalar>
detail::ColumnMajorLapackMatrix<Scalar> real_dense_inverse(detail::ColumnMajorLapackMatrix<Scalar> matrix)
{
  if (!std::cmp_equal(matrix.rows(), matrix.cols()))
  {
    throw std::invalid_argument("real_dense_inverse requires a square matrix");
  }

  std::size_t const n = static_cast<std::size_t>(matrix.rows());
  if (n == 0)
  {
    return matrix;
  }

  blas_int const order = detail::checked_blas_int(n);
  std::vector<blas_int> pivots(n);
  uni20::lapack::getrf(order, order, matrix.data(), order, pivots.data());

  Scalar work_query{};
  blas_int const query_lwork = -1;
  uni20::lapack::getri(order, matrix.data(), order, pivots.data(), &work_query, query_lwork);

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
  std::vector<Scalar> work(static_cast<std::size_t>(lwork), Scalar{});
  uni20::lapack::getri(order, matrix.data(), order, pivots.data(), work.data(), lwork);

  return matrix;
}

/// \brief Solve a dense real least-squares or minimum-norm problem through LAPACK `gels`.
/// \details Solves `A * X = B` in the least-squares sense for overdetermined
///          systems and the minimum-norm sense for underdetermined systems.
///          The returned matrix has `A.cols()` rows and `B.cols()` columns.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param coefficients Real rectangular coefficient matrix `A`.
/// \param right_hand_sides Dense right-hand-side matrix `B` with matching row count.
/// \return Dense solution matrix `X`.
template <uni20::LapackReal Scalar>
detail::ColumnMajorLapackMatrix<Scalar> real_least_squares(detail::ColumnMajorLapackMatrix<Scalar> coefficients,
                                                           detail::ColumnMajorLapackMatrix<Scalar> right_hand_sides)
{
  if (!std::cmp_equal(coefficients.rows(), right_hand_sides.rows()))
  {
    throw std::invalid_argument("real_least_squares received incompatible right-hand sides");
  }

  std::size_t const rows = static_cast<std::size_t>(coefficients.rows());
  std::size_t const cols = static_cast<std::size_t>(coefficients.cols());
  std::size_t const rhs_count = static_cast<std::size_t>(right_hand_sides.cols());
  detail::ColumnMajorLapackMatrix<Scalar> solution(cols, rhs_count);
  if (cols == 0 || rhs_count == 0)
  {
    return solution;
  }
  if (rows == 0)
  {
    std::fill_n(solution.data(), solution.size(), Scalar{});
    return solution;
  }

  std::size_t const workspace_rows = std::max(rows, cols);
  detail::ColumnMajorLapackMatrix<Scalar> rhs_workspace(workspace_rows, rhs_count);
  for (std::size_t row = 0; row < rows; ++row)
  {
    for (std::size_t col = 0; col < rhs_count; ++col)
    {
      rhs_workspace[row, col] = right_hand_sides[row, col];
    }
  }

  blas_int const m = detail::checked_blas_int(rows);
  blas_int const n = detail::checked_blas_int(cols);
  blas_int const nrhs = detail::checked_blas_int(rhs_count);
  blas_int const lda = std::max<blas_int>(1, m);
  blas_int const ldb = detail::checked_blas_int(workspace_rows);
  char const trans = 'N';

  Scalar work_query{};
  blas_int const query_lwork = -1;
  uni20::lapack::gels(trans, m, n, nrhs, coefficients.data(), lda, rhs_workspace.data(), ldb, &work_query, query_lwork);

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
  std::vector<Scalar> work(static_cast<std::size_t>(lwork), Scalar{});
  uni20::lapack::gels(trans, m, n, nrhs, coefficients.data(), lda, rhs_workspace.data(), ldb, work.data(), lwork);

  for (std::size_t row = 0; row < cols; ++row)
  {
    for (std::size_t col = 0; col < rhs_count; ++col)
    {
      solution[row, col] = rhs_workspace[row, col];
    }
  }
  return solution;
}

/// \brief Solve a dense real SVD-based least-squares problem through LAPACK `gelss`.
/// \details Solves `A * X = B` using a singular value decomposition. The
///          returned singular values are sorted in descending order, and the
///          numerical rank is determined by \p rcond.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param coefficients Real rectangular coefficient matrix `A`.
/// \param right_hand_sides Dense right-hand-side matrix `B` with matching row count.
/// \param rcond Relative singular-value threshold used by LAPACK to determine rank.
/// \return Dense solution, numerical rank, and singular values of `A`.
template <uni20::LapackReal Scalar>
RealSvdLeastSquares<Scalar> real_svd_least_squares(detail::ColumnMajorLapackMatrix<Scalar> coefficients,
                                                   detail::ColumnMajorLapackMatrix<Scalar> right_hand_sides,
                                                   Scalar rcond = Scalar{100} *
                                                                  uni20::numeric_limits<Scalar>::epsilon())
{
  if (!std::cmp_equal(coefficients.rows(), right_hand_sides.rows()))
  {
    throw std::invalid_argument("real_svd_least_squares received incompatible right-hand sides");
  }

  std::size_t const rows = static_cast<std::size_t>(coefficients.rows());
  std::size_t const cols = static_cast<std::size_t>(coefficients.cols());
  std::size_t const rhs_count = static_cast<std::size_t>(right_hand_sides.cols());
  RealSvdLeastSquares<Scalar> result;
  result.solution = detail::ColumnMajorLapackMatrix<Scalar>(cols, rhs_count);
  result.singular_values.resize(std::min(rows, cols));
  if (cols == 0 || rhs_count == 0)
  {
    return result;
  }
  if (rows == 0)
  {
    std::fill_n(result.solution.data(), result.solution.size(), Scalar{});
    return result;
  }

  std::size_t const workspace_rows = std::max(rows, cols);
  detail::ColumnMajorLapackMatrix<Scalar> rhs_workspace(workspace_rows, rhs_count);
  for (std::size_t row = 0; row < rows; ++row)
  {
    for (std::size_t col = 0; col < rhs_count; ++col)
    {
      rhs_workspace[row, col] = right_hand_sides[row, col];
    }
  }

  blas_int const m = detail::checked_blas_int(rows);
  blas_int const n = detail::checked_blas_int(cols);
  blas_int const nrhs = detail::checked_blas_int(rhs_count);
  blas_int const lda = std::max<blas_int>(1, m);
  blas_int const ldb = detail::checked_blas_int(workspace_rows);
  blas_int rank = 0;

  Scalar work_query{};
  blas_int const query_lwork = -1;
  uni20::lapack::gelss(m, n, nrhs, coefficients.data(), lda, rhs_workspace.data(), ldb, result.singular_values.data(),
                       rcond, rank, &work_query, query_lwork);

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
  std::vector<Scalar> work(static_cast<std::size_t>(lwork), Scalar{});
  uni20::lapack::gelss(m, n, nrhs, coefficients.data(), lda, rhs_workspace.data(), ldb, result.singular_values.data(),
                       rcond, rank, work.data(), lwork);

  if (rank < 0)
  {
    throw std::runtime_error("LAPACK gelss returned a negative rank");
  }
  result.rank = static_cast<std::size_t>(rank);
  for (std::size_t row = 0; row < cols; ++row)
  {
    for (std::size_t col = 0; col < rhs_count; ++col)
    {
      result.solution[row, col] = rhs_workspace[row, col];
    }
  }
  return result;
}

/// \brief Solve a dense real divide-and-conquer SVD least-squares problem through LAPACK `gelsd`.
/// \details Solves `A * X = B` using a divide-and-conquer singular value
///          decomposition. The returned singular values are sorted in
///          descending order, and the numerical rank is determined by
///          \p rcond.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param coefficients Real rectangular coefficient matrix `A`.
/// \param right_hand_sides Dense right-hand-side matrix `B` with matching row count.
/// \param rcond Relative singular-value threshold used by LAPACK to determine rank.
/// \return Dense solution, numerical rank, and singular values of `A`.
template <uni20::LapackReal Scalar>
RealSvdLeastSquares<Scalar>
real_divide_and_conquer_svd_least_squares(detail::ColumnMajorLapackMatrix<Scalar> coefficients,
                                          detail::ColumnMajorLapackMatrix<Scalar> right_hand_sides,
                                          Scalar rcond = Scalar{100} * uni20::numeric_limits<Scalar>::epsilon())
{
  if (!std::cmp_equal(coefficients.rows(), right_hand_sides.rows()))
  {
    throw std::invalid_argument("real_divide_and_conquer_svd_least_squares received incompatible right-hand sides");
  }

  std::size_t const rows = static_cast<std::size_t>(coefficients.rows());
  std::size_t const cols = static_cast<std::size_t>(coefficients.cols());
  std::size_t const rhs_count = static_cast<std::size_t>(right_hand_sides.cols());
  RealSvdLeastSquares<Scalar> result;
  result.solution = detail::ColumnMajorLapackMatrix<Scalar>(cols, rhs_count);
  result.singular_values.resize(std::min(rows, cols));
  if (cols == 0 || rhs_count == 0)
  {
    return result;
  }
  if (rows == 0)
  {
    std::fill_n(result.solution.data(), result.solution.size(), Scalar{});
    return result;
  }

  std::size_t const workspace_rows = std::max(rows, cols);
  detail::ColumnMajorLapackMatrix<Scalar> rhs_workspace(workspace_rows, rhs_count);
  for (std::size_t row = 0; row < rows; ++row)
  {
    for (std::size_t col = 0; col < rhs_count; ++col)
    {
      rhs_workspace[row, col] = right_hand_sides[row, col];
    }
  }

  blas_int const m = detail::checked_blas_int(rows);
  blas_int const n = detail::checked_blas_int(cols);
  blas_int const nrhs = detail::checked_blas_int(rhs_count);
  blas_int const lda = std::max<blas_int>(1, m);
  blas_int const ldb = detail::checked_blas_int(workspace_rows);
  blas_int rank = 0;
  blas_int const iwork_size = detail::gelsd_iwork_size(m, n);
  std::vector<blas_int> iwork(static_cast<std::size_t>(iwork_size), 0);

  Scalar work_query{};
  blas_int const query_lwork = -1;
  uni20::lapack::gelsd(m, n, nrhs, coefficients.data(), lda, rhs_workspace.data(), ldb, result.singular_values.data(),
                       rcond, rank, &work_query, query_lwork, iwork.data());

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
  std::vector<Scalar> work(static_cast<std::size_t>(lwork), Scalar{});
  rank = 0;
  uni20::lapack::gelsd(m, n, nrhs, coefficients.data(), lda, rhs_workspace.data(), ldb, result.singular_values.data(),
                       rcond, rank, work.data(), lwork, iwork.data());

  if (rank < 0)
  {
    throw std::runtime_error("LAPACK gelsd returned a negative rank");
  }
  result.rank = static_cast<std::size_t>(rank);
  for (std::size_t row = 0; row < cols; ++row)
  {
    for (std::size_t col = 0; col < rhs_count; ++col)
    {
      result.solution[row, col] = rhs_workspace[row, col];
    }
  }
  return result;
}

/// \brief Solve a dense real rank-revealing least-squares problem through LAPACK `gelsy`.
/// \details Solves `A * X = B` using complete orthogonal factorization with
///          column pivoting. The returned pivot order is 0-based: entry `j`
///          gives the original column placed at pivoted position `j`.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param coefficients Real rectangular coefficient matrix `A`.
/// \param right_hand_sides Dense right-hand-side matrix `B` with matching row count.
/// \param rcond Relative threshold used by LAPACK to determine the effective rank.
/// \return Dense solution, numerical rank, and column pivot order.
template <uni20::LapackReal Scalar>
RealRankRevealingLeastSquares<Scalar>
real_rank_revealing_least_squares(detail::ColumnMajorLapackMatrix<Scalar> coefficients,
                                  detail::ColumnMajorLapackMatrix<Scalar> right_hand_sides,
                                  Scalar rcond = Scalar{100} * uni20::numeric_limits<Scalar>::epsilon())
{
  if (!std::cmp_equal(coefficients.rows(), right_hand_sides.rows()))
  {
    throw std::invalid_argument("real_rank_revealing_least_squares received incompatible right-hand sides");
  }

  std::size_t const rows = static_cast<std::size_t>(coefficients.rows());
  std::size_t const cols = static_cast<std::size_t>(coefficients.cols());
  std::size_t const rhs_count = static_cast<std::size_t>(right_hand_sides.cols());
  RealRankRevealingLeastSquares<Scalar> result;
  result.solution = detail::ColumnMajorLapackMatrix<Scalar>(cols, rhs_count);
  result.pivot_columns.resize(cols);
  std::iota(result.pivot_columns.begin(), result.pivot_columns.end(), std::size_t{0});
  if (cols == 0 || rhs_count == 0)
  {
    return result;
  }
  if (rows == 0)
  {
    std::fill_n(result.solution.data(), result.solution.size(), Scalar{});
    return result;
  }

  std::size_t const workspace_rows = std::max(rows, cols);
  detail::ColumnMajorLapackMatrix<Scalar> rhs_workspace(workspace_rows, rhs_count);
  for (std::size_t row = 0; row < rows; ++row)
  {
    for (std::size_t col = 0; col < rhs_count; ++col)
    {
      rhs_workspace[row, col] = right_hand_sides[row, col];
    }
  }

  blas_int const m = detail::checked_blas_int(rows);
  blas_int const n = detail::checked_blas_int(cols);
  blas_int const nrhs = detail::checked_blas_int(rhs_count);
  blas_int const lda = std::max<blas_int>(1, m);
  blas_int const ldb = detail::checked_blas_int(workspace_rows);
  std::vector<blas_int> jpvt(cols, 0);
  blas_int rank = 0;

  Scalar work_query{};
  blas_int const query_lwork = -1;
  uni20::lapack::gelsy(m, n, nrhs, coefficients.data(), lda, rhs_workspace.data(), ldb, jpvt.data(), rcond, rank,
                       &work_query, query_lwork);

  std::fill(jpvt.begin(), jpvt.end(), 0);
  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
  std::vector<Scalar> work(static_cast<std::size_t>(lwork), Scalar{});
  uni20::lapack::gelsy(m, n, nrhs, coefficients.data(), lda, rhs_workspace.data(), ldb, jpvt.data(), rcond, rank,
                       work.data(), lwork);

  if (rank < 0)
  {
    throw std::runtime_error("LAPACK gelsy returned a negative rank");
  }
  result.rank = static_cast<std::size_t>(rank);
  for (std::size_t col = 0; col < cols; ++col)
  {
    if (jpvt[col] <= 0 || static_cast<std::size_t>(jpvt[col]) > cols)
    {
      throw std::runtime_error("LAPACK gelsy returned an invalid column pivot");
    }
    result.pivot_columns[col] = static_cast<std::size_t>(jpvt[col] - 1);
  }

  for (std::size_t row = 0; row < cols; ++row)
  {
    for (std::size_t col = 0; col < rhs_count; ++col)
    {
      result.solution[row, col] = rhs_workspace[row, col];
    }
  }
  return result;
}

/// \brief Solve a dense real symmetric positive definite system through LAPACK `potrf` and `potrs`.
/// \details Solves `A * X = B` for one or more dense right-hand sides. The
///          coefficient and right-hand-side matrices are copied by value and
///          may be overwritten by LAPACK. The returned matrix stores `X`.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param coefficients Real symmetric positive definite square matrix.
/// \param right_hand_sides Dense right-hand-side matrix with matching row count.
/// \param triangle Triangle of \p coefficients supplied to LAPACK.
/// \return Dense solution matrix.
template <uni20::LapackReal Scalar>
detail::ColumnMajorLapackMatrix<Scalar>
real_symmetric_positive_definite_solve(detail::ColumnMajorLapackMatrix<Scalar> coefficients,
                                       detail::ColumnMajorLapackMatrix<Scalar> right_hand_sides,
                                       MatrixFill triangle = MatrixFill::Upper)
{
  if (!std::cmp_equal(coefficients.rows(), coefficients.cols()))
  {
    throw std::invalid_argument("real_symmetric_positive_definite_solve requires a square coefficient matrix");
  }
  if (!std::cmp_equal(coefficients.rows(), right_hand_sides.rows()))
  {
    throw std::invalid_argument("real_symmetric_positive_definite_solve received incompatible right-hand sides");
  }

  std::size_t const n = static_cast<std::size_t>(coefficients.rows());
  if (n == 0 || right_hand_sides.cols() == 0)
  {
    return right_hand_sides;
  }

  blas_int const order = detail::checked_blas_int(n);
  blas_int const nrhs = detail::checked_blas_int(right_hand_sides.cols());
  char const uplo = detail::lapack_uplo(triangle);
  uni20::lapack::potrf(uplo, order, coefficients.data(), order);
  uni20::lapack::potrs(uplo, order, nrhs, coefficients.data(), order, right_hand_sides.data(), order);
  return right_hand_sides;
}

/// \brief Solve a dense real SPD system from an existing Cholesky factorization.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param factorization Cholesky factors returned by `real_symmetric_positive_definite_factorization`.
/// \param right_hand_sides Dense right-hand-side matrix with matching row count.
/// \return Dense solution matrix.
template <uni20::LapackReal Scalar>
detail::ColumnMajorLapackMatrix<Scalar>
real_symmetric_positive_definite_solve(RealSymmetricPositiveDefiniteFactorization<Scalar> const& factorization,
                                       detail::ColumnMajorLapackMatrix<Scalar> right_hand_sides)
{
  if (!std::cmp_equal(factorization.factors.rows(), factorization.factors.cols()))
  {
    throw std::invalid_argument("real_symmetric_positive_definite_solve requires square Cholesky factors");
  }
  std::size_t const n = static_cast<std::size_t>(factorization.factors.rows());
  if (!std::cmp_equal(right_hand_sides.rows(), n))
  {
    throw std::invalid_argument("real_symmetric_positive_definite_solve received incompatible right-hand sides");
  }
  if (n == 0 || right_hand_sides.cols() == 0)
  {
    return right_hand_sides;
  }

  detail::ColumnMajorLapackMatrix<Scalar> factors = factorization.factors;
  blas_int const order = detail::checked_blas_int(n);
  blas_int const nrhs = detail::checked_blas_int(right_hand_sides.cols());
  char const uplo = detail::lapack_uplo(factorization.triangle);
  uni20::lapack::potrs(uplo, order, nrhs, factors.data(), order, right_hand_sides.data(), order);
  return right_hand_sides;
}

/// \brief Solve and refine a dense real SPD system through LAPACK `porfs`.
/// \details Computes a Cholesky factorization, solves `A * X = B`, then applies
///          iterative refinement and returns LAPACK's forward and backward
///          error estimates. Only the selected triangle of \p coefficients is
///          read by LAPACK.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param coefficients Real symmetric positive definite square matrix.
/// \param right_hand_sides Dense right-hand-side matrix with matching row count.
/// \param triangle Triangle of \p coefficients supplied to LAPACK.
/// \return Refined solution and LAPACK error estimates.
template <uni20::LapackReal Scalar>
RealRefinedLinearSolve<Scalar>
real_symmetric_positive_definite_refined_solve(detail::ColumnMajorLapackMatrix<Scalar> coefficients,
                                               detail::ColumnMajorLapackMatrix<Scalar> right_hand_sides,
                                               MatrixFill triangle = MatrixFill::Upper)
{
  if (!std::cmp_equal(coefficients.rows(), coefficients.cols()))
  {
    throw std::invalid_argument("real_symmetric_positive_definite_refined_solve requires a square coefficient matrix");
  }
  if (!std::cmp_equal(coefficients.rows(), right_hand_sides.rows()))
  {
    throw std::invalid_argument(
        "real_symmetric_positive_definite_refined_solve received incompatible right-hand sides");
  }

  std::size_t const n = static_cast<std::size_t>(coefficients.rows());
  std::size_t const rhs_count = static_cast<std::size_t>(right_hand_sides.cols());
  RealRefinedLinearSolve<Scalar> result;
  result.solution = right_hand_sides;
  result.forward_error_bounds.resize(rhs_count);
  result.backward_error_bounds.resize(rhs_count);
  if (n == 0 || rhs_count == 0)
  {
    return result;
  }

  detail::ColumnMajorLapackMatrix<Scalar> factors = coefficients;
  blas_int const order = detail::checked_blas_int(n);
  blas_int const nrhs = detail::checked_blas_int(rhs_count);
  char const uplo = detail::lapack_uplo(triangle);
  uni20::lapack::potrf(uplo, order, factors.data(), order);
  uni20::lapack::potrs(uplo, order, nrhs, factors.data(), order, result.solution.data(), order);

  std::vector<Scalar> work(static_cast<std::size_t>(std::max<blas_int>(1, 3 * order)), Scalar{});
  std::vector<blas_int> iwork(static_cast<std::size_t>(std::max<blas_int>(1, order)), 0);
  uni20::lapack::porfs(uplo, order, nrhs, coefficients.data(), order, factors.data(), order, right_hand_sides.data(),
                       order, result.solution.data(), order, result.forward_error_bounds.data(),
                       result.backward_error_bounds.data(), work.data(), iwork.data());
  return result;
}

/// \brief Solve a dense real SPD linear system with LAPACK `posvx` diagnostics.
/// \details Solves `A * X = B` and returns the solution, Cholesky factors,
///          reciprocal condition estimate, componentwise backward errors,
///          forward error bounds, and equilibration/condition diagnostics
///          reported by LAPACK. The current wrapper requests a fresh
///          factorization (`FACT='N'`) and does not expose caller-supplied
///          equilibration modes.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param coefficients Real symmetric positive definite square matrix.
/// \param right_hand_sides Dense right-hand-side matrix with matching row count.
/// \param triangle Triangle of \p coefficients supplied to LAPACK.
/// \return Dense solution and LAPACK expert-driver diagnostics.
template <uni20::LapackReal Scalar>
RealSymmetricPositiveDefiniteExpertLinearSolve<Scalar>
real_symmetric_positive_definite_expert_solve(detail::ColumnMajorLapackMatrix<Scalar> coefficients,
                                              detail::ColumnMajorLapackMatrix<Scalar> right_hand_sides,
                                              MatrixFill triangle = MatrixFill::Upper)
{
  if (!std::cmp_equal(coefficients.rows(), coefficients.cols()))
  {
    throw std::invalid_argument("real_symmetric_positive_definite_expert_solve requires a square coefficient matrix");
  }
  if (!std::cmp_equal(coefficients.rows(), right_hand_sides.rows()))
  {
    throw std::invalid_argument("real_symmetric_positive_definite_expert_solve received incompatible right-hand sides");
  }

  std::size_t const n = static_cast<std::size_t>(coefficients.rows());
  std::size_t const rhs_count = static_cast<std::size_t>(right_hand_sides.cols());
  RealSymmetricPositiveDefiniteExpertLinearSolve<Scalar> result;
  result.solution = detail::ColumnMajorLapackMatrix<Scalar>(n, rhs_count);
  result.factors = detail::ColumnMajorLapackMatrix<Scalar>(n, n);
  result.forward_error_bounds.resize(rhs_count);
  result.backward_error_bounds.resize(rhs_count);
  result.scale.assign(n, Scalar{1});
  if (n == 0 || rhs_count == 0)
  {
    return result;
  }

  blas_int const order = detail::checked_blas_int(n);
  blas_int const nrhs = detail::checked_blas_int(rhs_count);
  char const uplo = detail::lapack_uplo(triangle);
  char equed = 'N';
  Scalar rcond{};
  std::vector<Scalar> work(static_cast<std::size_t>(std::max<blas_int>(1, 3 * order)), Scalar{});
  std::vector<blas_int> iwork(static_cast<std::size_t>(std::max<blas_int>(1, order)), 0);

  result.reciprocal_condition_below_machine_precision = uni20::lapack::posvx(
      'N', uplo, order, nrhs, coefficients.data(), order, result.factors.data(), order, equed, result.scale.data(),
      right_hand_sides.data(), order, result.solution.data(), order, rcond, result.forward_error_bounds.data(),
      result.backward_error_bounds.data(), work.data(), iwork.data());
  result.equilibration = equed;
  result.reciprocal_condition = rcond;
  return result;
}

/// \brief Estimate a dense real SPD one-norm reciprocal condition number through LAPACK `pocon`.
/// \details Computes a Cholesky factorization with `potrf`, then estimates
///          `1 / (||A||_1 * ||inv(A)||_1)` from the factorization. Only the
///          selected triangle is read, but the one-norm is computed as if the
///          full symmetric matrix were materialized.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real symmetric positive definite square matrix.
/// \param triangle Triangle of \p matrix supplied to LAPACK.
/// \return Estimated one-norm reciprocal condition number.
template <uni20::LapackReal Scalar>
Scalar
real_symmetric_positive_definite_one_norm_reciprocal_condition_number(detail::ColumnMajorLapackMatrix<Scalar> matrix,
                                                                      MatrixFill triangle = MatrixFill::Upper)
{
  if (!std::cmp_equal(matrix.rows(), matrix.cols()))
  {
    throw std::invalid_argument(
        "real_symmetric_positive_definite_one_norm_reciprocal_condition_number requires a square matrix");
  }

  std::size_t const n = static_cast<std::size_t>(matrix.rows());
  if (n == 0)
  {
    return Scalar{1};
  }

  Scalar const original_one_norm = detail::symmetric_matrix_one_norm(matrix, triangle);
  blas_int const order = detail::checked_blas_int(n);
  char const uplo = detail::lapack_uplo(triangle);
  uni20::lapack::potrf(uplo, order, matrix.data(), order);
  std::vector<Scalar> work(static_cast<std::size_t>(std::max<blas_int>(1, 3 * order)), Scalar{});
  std::vector<blas_int> iwork(static_cast<std::size_t>(std::max<blas_int>(1, order)), 0);
  return uni20::lapack::pocon(uplo, order, matrix.data(), order, original_one_norm, work.data(), iwork.data());
}

/// \brief Estimate an SPD one-norm reciprocal condition number from an existing Cholesky factorization.
/// \details The supplied matrix norm must be the one-norm of the original
///          unfactored symmetric positive definite matrix.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param factorization Cholesky factors returned by `real_symmetric_positive_definite_factorization`.
/// \param original_one_norm One-norm of the original coefficient matrix.
/// \return Estimate of `1 / (||A||_1 * ||inv(A)||_1)`.
template <uni20::LapackReal Scalar>
Scalar real_symmetric_positive_definite_one_norm_reciprocal_condition_number(
    RealSymmetricPositiveDefiniteFactorization<Scalar> const& factorization, Scalar original_one_norm)
{
  if (!std::cmp_equal(factorization.factors.rows(), factorization.factors.cols()))
  {
    throw std::invalid_argument(
        "real_symmetric_positive_definite_one_norm_reciprocal_condition_number requires square Cholesky factors");
  }
  if (original_one_norm < Scalar{})
  {
    throw std::invalid_argument(
        "real_symmetric_positive_definite_one_norm_reciprocal_condition_number requires a nonnegative matrix norm");
  }

  std::size_t const n = static_cast<std::size_t>(factorization.factors.rows());
  if (n == 0)
  {
    return Scalar{1};
  }

  detail::ColumnMajorLapackMatrix<Scalar> factors = factorization.factors;
  blas_int const order = detail::checked_blas_int(n);
  char const uplo = detail::lapack_uplo(factorization.triangle);
  std::vector<Scalar> work(static_cast<std::size_t>(std::max<blas_int>(1, 3 * order)), Scalar{});
  std::vector<blas_int> iwork(static_cast<std::size_t>(std::max<blas_int>(1, order)), 0);
  return uni20::lapack::pocon(uplo, order, factors.data(), order, original_one_norm, work.data(), iwork.data());
}

/// \brief Invert a dense real SPD matrix through LAPACK `potrf` and `potri`.
/// \details Computes a Cholesky factorization, inverts it with `potri`, and
///          mirrors the selected triangle so the returned matrix is a full
///          dense symmetric inverse.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real symmetric positive definite square matrix.
/// \param triangle Triangle of \p matrix supplied to LAPACK.
/// \return Dense symmetric inverse matrix.
template <uni20::LapackReal Scalar>
detail::ColumnMajorLapackMatrix<Scalar>
real_symmetric_positive_definite_inverse(detail::ColumnMajorLapackMatrix<Scalar> matrix,
                                         MatrixFill triangle = MatrixFill::Upper)
{
  if (!std::cmp_equal(matrix.rows(), matrix.cols()))
  {
    throw std::invalid_argument("real_symmetric_positive_definite_inverse requires a square matrix");
  }

  std::size_t const n = static_cast<std::size_t>(matrix.rows());
  if (n == 0)
  {
    return matrix;
  }

  blas_int const order = detail::checked_blas_int(n);
  char const uplo = detail::lapack_uplo(triangle);
  uni20::lapack::potrf(uplo, order, matrix.data(), order);
  uni20::lapack::potri(uplo, order, matrix.data(), order);
  detail::mirror_selected_symmetric_triangle(matrix, triangle);
  return matrix;
}

/// \brief Compute a dense real symmetric-indefinite Bunch-Kaufman factorization through LAPACK `sytrf`.
/// \details Only the selected triangle of \p matrix is factorized. The signed
///          pivot block metadata is kept in LAPACK's `ipiv` encoding so it can
///          be reused by `sytrs`, `sytri`, `syrfs`, and `sycon` wrappers.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real symmetric square matrix.
/// \param triangle Triangle of \p matrix supplied to LAPACK.
/// \return Bunch-Kaufman factors, signed pivot blocks, and triangle metadata.
template <uni20::LapackReal Scalar>
RealSymmetricIndefiniteFactorization<Scalar>
real_symmetric_indefinite_factorization(detail::ColumnMajorLapackMatrix<Scalar> matrix,
                                        MatrixFill triangle = MatrixFill::Upper)
{
  if (!std::cmp_equal(matrix.rows(), matrix.cols()))
  {
    throw std::invalid_argument("real_symmetric_indefinite_factorization requires a square matrix");
  }

  std::size_t const n = static_cast<std::size_t>(matrix.rows());
  RealSymmetricIndefiniteFactorization<Scalar> result;
  result.factors = std::move(matrix);
  result.pivot_blocks.resize(n);
  result.triangle = triangle;
  if (n == 0)
  {
    return result;
  }

  blas_int const order = detail::checked_blas_int(n);
  char const uplo = detail::lapack_uplo(triangle);
  std::vector<blas_int> pivots(n);

  Scalar work_query{};
  blas_int const query_lwork = -1;
  uni20::lapack::sytrf(uplo, order, result.factors.data(), order, pivots.data(), &work_query, query_lwork);

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
  std::vector<Scalar> work(static_cast<std::size_t>(lwork), Scalar{});
  uni20::lapack::sytrf(uplo, order, result.factors.data(), order, pivots.data(), work.data(), lwork);

  for (std::size_t index = 0; index < n; ++index)
  {
    result.pivot_blocks[index] = static_cast<int>(pivots[index]);
  }
  return result;
}

/// \brief Solve a dense real symmetric indefinite system through LAPACK `sytrf` and `sytrs`.
/// \details Solves `A * X = B` for one or more dense right-hand sides using
///          a Bunch-Kaufman diagonal-pivoting factorization. The coefficient
///          and right-hand-side matrices are copied by value and may be
///          overwritten by LAPACK. The returned matrix stores `X`.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param coefficients Real symmetric square coefficient matrix.
/// \param right_hand_sides Dense right-hand-side matrix with matching row count.
/// \param triangle Triangle of \p coefficients supplied to LAPACK.
/// \return Dense solution matrix.
template <uni20::LapackReal Scalar>
detail::ColumnMajorLapackMatrix<Scalar>
real_symmetric_indefinite_solve(detail::ColumnMajorLapackMatrix<Scalar> coefficients,
                                detail::ColumnMajorLapackMatrix<Scalar> right_hand_sides,
                                MatrixFill triangle = MatrixFill::Upper)
{
  if (!std::cmp_equal(coefficients.rows(), coefficients.cols()))
  {
    throw std::invalid_argument("real_symmetric_indefinite_solve requires a square coefficient matrix");
  }
  if (!std::cmp_equal(coefficients.rows(), right_hand_sides.rows()))
  {
    throw std::invalid_argument("real_symmetric_indefinite_solve received incompatible right-hand sides");
  }

  std::size_t const n = static_cast<std::size_t>(coefficients.rows());
  if (n == 0 || right_hand_sides.cols() == 0)
  {
    return right_hand_sides;
  }

  blas_int const order = detail::checked_blas_int(n);
  blas_int const nrhs = detail::checked_blas_int(right_hand_sides.cols());
  char const uplo = detail::lapack_uplo(triangle);
  std::vector<blas_int> pivots(n);

  Scalar work_query{};
  blas_int const query_lwork = -1;
  uni20::lapack::sytrf(uplo, order, coefficients.data(), order, pivots.data(), &work_query, query_lwork);

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
  std::vector<Scalar> work(static_cast<std::size_t>(lwork), Scalar{});
  uni20::lapack::sytrf(uplo, order, coefficients.data(), order, pivots.data(), work.data(), lwork);
  uni20::lapack::sytrs(uplo, order, nrhs, coefficients.data(), order, pivots.data(), right_hand_sides.data(), order);
  return right_hand_sides;
}

/// \brief Solve a dense real symmetric-indefinite system from an existing Bunch-Kaufman factorization.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param factorization Factors returned by `real_symmetric_indefinite_factorization`.
/// \param right_hand_sides Dense right-hand-side matrix with matching row count.
/// \return Dense solution matrix.
template <uni20::LapackReal Scalar>
detail::ColumnMajorLapackMatrix<Scalar>
real_symmetric_indefinite_solve(RealSymmetricIndefiniteFactorization<Scalar> const& factorization,
                                detail::ColumnMajorLapackMatrix<Scalar> right_hand_sides)
{
  if (!std::cmp_equal(factorization.factors.rows(), factorization.factors.cols()))
  {
    throw std::invalid_argument("real_symmetric_indefinite_solve requires square Bunch-Kaufman factors");
  }
  std::size_t const n = static_cast<std::size_t>(factorization.factors.rows());
  if (factorization.pivot_blocks.size() != n)
  {
    throw std::invalid_argument("real_symmetric_indefinite_solve received inconsistent pivot metadata");
  }
  if (!std::cmp_equal(right_hand_sides.rows(), n))
  {
    throw std::invalid_argument("real_symmetric_indefinite_solve received incompatible right-hand sides");
  }
  if (n == 0 || right_hand_sides.cols() == 0)
  {
    return right_hand_sides;
  }

  detail::ColumnMajorLapackMatrix<Scalar> factors = factorization.factors;
  std::vector<blas_int> pivots = detail::checked_symmetric_indefinite_pivots(factorization);
  blas_int const order = detail::checked_blas_int(n);
  blas_int const nrhs = detail::checked_blas_int(right_hand_sides.cols());
  char const uplo = detail::lapack_uplo(factorization.triangle);
  uni20::lapack::sytrs(uplo, order, nrhs, factors.data(), order, pivots.data(), right_hand_sides.data(), order);
  return right_hand_sides;
}

/// \brief Solve and refine a dense real symmetric-indefinite system through LAPACK `syrfs`.
/// \details Computes a Bunch-Kaufman factorization, solves `A * X = B`, then
///          applies iterative refinement and returns LAPACK's forward and
///          backward error estimates. Only the selected triangle of
///          \p coefficients is read by LAPACK.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param coefficients Real symmetric square coefficient matrix.
/// \param right_hand_sides Dense right-hand-side matrix with matching row count.
/// \param triangle Triangle of \p coefficients supplied to LAPACK.
/// \return Refined solution and LAPACK error estimates.
template <uni20::LapackReal Scalar>
RealRefinedLinearSolve<Scalar>
real_symmetric_indefinite_refined_solve(detail::ColumnMajorLapackMatrix<Scalar> coefficients,
                                        detail::ColumnMajorLapackMatrix<Scalar> right_hand_sides,
                                        MatrixFill triangle = MatrixFill::Upper)
{
  if (!std::cmp_equal(coefficients.rows(), coefficients.cols()))
  {
    throw std::invalid_argument("real_symmetric_indefinite_refined_solve requires a square coefficient matrix");
  }
  if (!std::cmp_equal(coefficients.rows(), right_hand_sides.rows()))
  {
    throw std::invalid_argument("real_symmetric_indefinite_refined_solve received incompatible right-hand sides");
  }

  std::size_t const n = static_cast<std::size_t>(coefficients.rows());
  std::size_t const rhs_count = static_cast<std::size_t>(right_hand_sides.cols());
  RealRefinedLinearSolve<Scalar> result;
  result.solution = right_hand_sides;
  result.forward_error_bounds.resize(rhs_count);
  result.backward_error_bounds.resize(rhs_count);
  if (n == 0 || rhs_count == 0)
  {
    return result;
  }

  detail::ColumnMajorLapackMatrix<Scalar> factors = coefficients;
  blas_int const order = detail::checked_blas_int(n);
  blas_int const nrhs = detail::checked_blas_int(rhs_count);
  char const uplo = detail::lapack_uplo(triangle);
  std::vector<blas_int> pivots(n);

  Scalar work_query{};
  blas_int const query_lwork = -1;
  uni20::lapack::sytrf(uplo, order, factors.data(), order, pivots.data(), &work_query, query_lwork);

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
  std::vector<Scalar> factor_work(static_cast<std::size_t>(lwork), Scalar{});
  uni20::lapack::sytrf(uplo, order, factors.data(), order, pivots.data(), factor_work.data(), lwork);
  uni20::lapack::sytrs(uplo, order, nrhs, factors.data(), order, pivots.data(), result.solution.data(), order);

  std::vector<Scalar> refine_work(static_cast<std::size_t>(std::max<blas_int>(1, 3 * order)), Scalar{});
  std::vector<blas_int> iwork(static_cast<std::size_t>(std::max<blas_int>(1, order)), 0);
  uni20::lapack::syrfs(uplo, order, nrhs, coefficients.data(), order, factors.data(), order, pivots.data(),
                       right_hand_sides.data(), order, result.solution.data(), order,
                       result.forward_error_bounds.data(), result.backward_error_bounds.data(), refine_work.data(),
                       iwork.data());
  return result;
}

/// \brief Invert a dense real symmetric-indefinite matrix through LAPACK `sytrf` and `sytri`.
/// \details Computes a Bunch-Kaufman factorization, inverts it with `sytri`,
///          and mirrors the selected triangle so the returned matrix is a full
///          dense symmetric inverse.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real symmetric square matrix.
/// \param triangle Triangle of \p matrix supplied to LAPACK.
/// \return Dense symmetric inverse matrix.
template <uni20::LapackReal Scalar>
detail::ColumnMajorLapackMatrix<Scalar>
real_symmetric_indefinite_inverse(detail::ColumnMajorLapackMatrix<Scalar> matrix,
                                  MatrixFill triangle = MatrixFill::Upper)
{
  if (!std::cmp_equal(matrix.rows(), matrix.cols()))
  {
    throw std::invalid_argument("real_symmetric_indefinite_inverse requires a square matrix");
  }

  std::size_t const n = static_cast<std::size_t>(matrix.rows());
  if (n == 0)
  {
    return matrix;
  }

  blas_int const order = detail::checked_blas_int(n);
  char const uplo = detail::lapack_uplo(triangle);
  std::vector<blas_int> pivots(n);

  Scalar work_query{};
  blas_int const query_lwork = -1;
  uni20::lapack::sytrf(uplo, order, matrix.data(), order, pivots.data(), &work_query, query_lwork);

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
  std::vector<Scalar> factor_work(static_cast<std::size_t>(lwork), Scalar{});
  uni20::lapack::sytrf(uplo, order, matrix.data(), order, pivots.data(), factor_work.data(), lwork);

  std::vector<Scalar> inverse_work(static_cast<std::size_t>(std::max<blas_int>(1, order)), Scalar{});
  uni20::lapack::sytri(uplo, order, matrix.data(), order, pivots.data(), inverse_work.data());
  detail::mirror_selected_symmetric_triangle(matrix, triangle);
  return matrix;
}

/// \brief Solve a dense real symmetric-indefinite system with LAPACK `sysvx` diagnostics.
/// \details Solves `A * X = B` using a Bunch-Kaufman factorization and returns
///          the solution, factors, signed pivot-block metadata, reciprocal
///          condition estimate, componentwise backward errors, and forward
///          error bounds reported by LAPACK.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param coefficients Real symmetric square coefficient matrix.
/// \param right_hand_sides Dense right-hand-side matrix with matching row count.
/// \param triangle Triangle of \p coefficients supplied to LAPACK.
/// \return Dense solution and LAPACK expert-driver diagnostics.
template <uni20::LapackReal Scalar>
RealSymmetricIndefiniteExpertLinearSolve<Scalar>
real_symmetric_indefinite_expert_solve(detail::ColumnMajorLapackMatrix<Scalar> coefficients,
                                       detail::ColumnMajorLapackMatrix<Scalar> right_hand_sides,
                                       MatrixFill triangle = MatrixFill::Upper)
{
  if (!std::cmp_equal(coefficients.rows(), coefficients.cols()))
  {
    throw std::invalid_argument("real_symmetric_indefinite_expert_solve requires a square coefficient matrix");
  }
  if (!std::cmp_equal(coefficients.rows(), right_hand_sides.rows()))
  {
    throw std::invalid_argument("real_symmetric_indefinite_expert_solve received incompatible right-hand sides");
  }

  std::size_t const n = static_cast<std::size_t>(coefficients.rows());
  std::size_t const rhs_count = static_cast<std::size_t>(right_hand_sides.cols());
  RealSymmetricIndefiniteExpertLinearSolve<Scalar> result;
  result.solution = detail::ColumnMajorLapackMatrix<Scalar>(n, rhs_count);
  result.factors = detail::ColumnMajorLapackMatrix<Scalar>(n, n);
  result.pivot_blocks.resize(n);
  result.forward_error_bounds.resize(rhs_count);
  result.backward_error_bounds.resize(rhs_count);
  if (n == 0 || rhs_count == 0)
  {
    return result;
  }

  blas_int const order = detail::checked_blas_int(n);
  blas_int const nrhs = detail::checked_blas_int(rhs_count);
  char const uplo = detail::lapack_uplo(triangle);
  std::vector<blas_int> pivots(n, 0);
  std::vector<blas_int> iwork(static_cast<std::size_t>(std::max<blas_int>(1, order)), 0);
  Scalar rcond{};

  Scalar work_query{};
  blas_int const query_lwork = -1;
  uni20::lapack::sysvx('N', uplo, order, nrhs, coefficients.data(), order, result.factors.data(), order, pivots.data(),
                       right_hand_sides.data(), order, result.solution.data(), order, rcond,
                       result.forward_error_bounds.data(), result.backward_error_bounds.data(), &work_query,
                       query_lwork, iwork.data());

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
  std::vector<Scalar> work(static_cast<std::size_t>(lwork), Scalar{});
  result.reciprocal_condition_below_machine_precision = uni20::lapack::sysvx(
      'N', uplo, order, nrhs, coefficients.data(), order, result.factors.data(), order, pivots.data(),
      right_hand_sides.data(), order, result.solution.data(), order, rcond, result.forward_error_bounds.data(),
      result.backward_error_bounds.data(), work.data(), lwork, iwork.data());
  result.reciprocal_condition = rcond;

  for (std::size_t index = 0; index < n; ++index)
  {
    result.pivot_blocks[index] = static_cast<int>(pivots[index]);
  }
  return result;
}

/// \brief Estimate a dense real symmetric-indefinite one-norm reciprocal condition number through LAPACK `sycon`.
/// \details Computes a Bunch-Kaufman factorization with `sytrf`, then estimates
///          `1 / (||A||_1 * ||inv(A)||_1)` from the factorization. Only the
///          selected triangle is read, but the one-norm is computed as if the
///          full symmetric matrix were materialized.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real symmetric indefinite square matrix.
/// \param triangle Triangle of \p matrix supplied to LAPACK.
/// \return Estimated one-norm reciprocal condition number.
template <uni20::LapackReal Scalar>
Scalar real_symmetric_indefinite_one_norm_reciprocal_condition_number(detail::ColumnMajorLapackMatrix<Scalar> matrix,
                                                                      MatrixFill triangle = MatrixFill::Upper)
{
  if (!std::cmp_equal(matrix.rows(), matrix.cols()))
  {
    throw std::invalid_argument(
        "real_symmetric_indefinite_one_norm_reciprocal_condition_number requires a square matrix");
  }

  std::size_t const n = static_cast<std::size_t>(matrix.rows());
  if (n == 0)
  {
    return Scalar{1};
  }

  Scalar const original_one_norm = detail::symmetric_matrix_one_norm(matrix, triangle);
  blas_int const order = detail::checked_blas_int(n);
  char const uplo = detail::lapack_uplo(triangle);
  std::vector<blas_int> pivots(n);

  Scalar work_query{};
  blas_int const query_lwork = -1;
  uni20::lapack::sytrf(uplo, order, matrix.data(), order, pivots.data(), &work_query, query_lwork);

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
  std::vector<Scalar> factor_work(static_cast<std::size_t>(lwork), Scalar{});
  uni20::lapack::sytrf(uplo, order, matrix.data(), order, pivots.data(), factor_work.data(), lwork);

  std::vector<Scalar> condition_work(static_cast<std::size_t>(std::max<blas_int>(1, 2 * order)), Scalar{});
  std::vector<blas_int> iwork(static_cast<std::size_t>(std::max<blas_int>(1, order)), 0);
  return uni20::lapack::sycon(uplo, order, matrix.data(), order, pivots.data(), original_one_norm,
                              condition_work.data(), iwork.data());
}

/// \brief Estimate a symmetric-indefinite one-norm reciprocal condition number from an existing factorization.
/// \details The supplied matrix norm must be the one-norm of the original
///          unfactored symmetric matrix.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param factorization Bunch-Kaufman factors returned by `real_symmetric_indefinite_factorization`.
/// \param original_one_norm One-norm of the original coefficient matrix.
/// \return Estimate of `1 / (||A||_1 * ||inv(A)||_1)`.
template <uni20::LapackReal Scalar>
Scalar real_symmetric_indefinite_one_norm_reciprocal_condition_number(
    RealSymmetricIndefiniteFactorization<Scalar> const& factorization, Scalar original_one_norm)
{
  if (!std::cmp_equal(factorization.factors.rows(), factorization.factors.cols()))
  {
    throw std::invalid_argument(
        "real_symmetric_indefinite_one_norm_reciprocal_condition_number requires square Bunch-Kaufman factors");
  }
  if (original_one_norm < Scalar{})
  {
    throw std::invalid_argument(
        "real_symmetric_indefinite_one_norm_reciprocal_condition_number requires a nonnegative matrix norm");
  }

  std::size_t const n = static_cast<std::size_t>(factorization.factors.rows());
  if (factorization.pivot_blocks.size() != n)
  {
    throw std::invalid_argument(
        "real_symmetric_indefinite_one_norm_reciprocal_condition_number received inconsistent pivot metadata");
  }
  if (n == 0)
  {
    return Scalar{1};
  }

  detail::ColumnMajorLapackMatrix<Scalar> factors = factorization.factors;
  std::vector<blas_int> pivots = detail::checked_symmetric_indefinite_pivots(factorization);
  blas_int const order = detail::checked_blas_int(n);
  char const uplo = detail::lapack_uplo(factorization.triangle);
  std::vector<Scalar> work(static_cast<std::size_t>(std::max<blas_int>(1, 2 * order)), Scalar{});
  std::vector<blas_int> iwork(static_cast<std::size_t>(std::max<blas_int>(1, order)), 0);
  return uni20::lapack::sycon(uplo, order, factors.data(), order, pivots.data(), original_one_norm, work.data(),
                              iwork.data());
}

/// \brief Solve a dense real symmetric eigensystem through LAPACK `syev`.
///
/// \details Eigenvalues are returned in ascending order. If `compute_vectors`
///          is true, `eigenvectors` is an `n`-by-`n` column-major matrix whose
///          columns are the normalized eigenvectors.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real symmetric square matrix. Only the selected triangle is read.
/// \param compute_vectors Whether to compute eigenvectors.
/// \param triangle Triangle of \p matrix supplied to LAPACK.
/// \return Eigenvalues and, optionally, eigenvectors.
template <uni20::LapackReal Scalar>
RealSymmetricEigensystem<Scalar> real_symmetric_eigensystem(detail::ColumnMajorLapackMatrix<Scalar> matrix,
                                                            bool compute_vectors,
                                                            MatrixFill triangle = MatrixFill::Upper)
{
  if (!std::cmp_equal(matrix.rows(), matrix.cols()))
  {
    throw std::invalid_argument("real_symmetric_eigensystem requires a square matrix");
  }

  std::size_t const n = static_cast<std::size_t>(matrix.rows());
  RealSymmetricEigensystem<Scalar> result;
  result.eigenvalues.resize(n);
  if (n == 0)
  {
    result.eigenvectors = detail::ColumnMajorLapackMatrix<Scalar>(0, 0);
    return result;
  }

  blas_int const order = detail::checked_blas_int(n);
  char const jobz = compute_vectors ? 'V' : 'N';
  char const uplo = detail::lapack_uplo(triangle);
  Scalar work_query{};
  blas_int const query_lwork = -1;
  uni20::lapack::syev(jobz, uplo, order, matrix.data(), order, result.eigenvalues.data(), &work_query, query_lwork);

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
  std::vector<Scalar> work(static_cast<std::size_t>(lwork), Scalar{});
  uni20::lapack::syev(jobz, uplo, order, matrix.data(), order, result.eigenvalues.data(), work.data(), lwork);

  result.eigenvectors = compute_vectors ? std::move(matrix) : detail::ColumnMajorLapackMatrix<Scalar>(0, 0);
  return result;
}

/// \brief Solve a dense real symmetric eigensystem through LAPACK `syevd`.
///
/// \details Uses the divide-and-conquer symmetric eigensystem driver.
///          Eigenvalues are returned in ascending order. If `compute_vectors`
///          is true, `eigenvectors` is an `n`-by-`n` column-major matrix whose
///          columns are the normalized eigenvectors.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real symmetric square matrix. Only the selected triangle is read.
/// \param compute_vectors Whether to compute eigenvectors.
/// \param triangle Triangle of \p matrix supplied to LAPACK.
/// \return Eigenvalues and, optionally, eigenvectors.
template <uni20::LapackReal Scalar>
RealSymmetricEigensystem<Scalar>
real_symmetric_eigensystem_divide_and_conquer(detail::ColumnMajorLapackMatrix<Scalar> matrix, bool compute_vectors,
                                              MatrixFill triangle = MatrixFill::Upper)
{
  if (!std::cmp_equal(matrix.rows(), matrix.cols()))
  {
    throw std::invalid_argument("real_symmetric_eigensystem_divide_and_conquer requires a square matrix");
  }

  std::size_t const n = static_cast<std::size_t>(matrix.rows());
  RealSymmetricEigensystem<Scalar> result;
  result.eigenvalues.resize(n);
  if (n == 0)
  {
    result.eigenvectors = detail::ColumnMajorLapackMatrix<Scalar>(0, 0);
    return result;
  }

  blas_int const order = detail::checked_blas_int(n);
  char const jobz = compute_vectors ? 'V' : 'N';
  char const uplo = detail::lapack_uplo(triangle);
  Scalar work_query{};
  blas_int iwork_query = 0;
  blas_int const query_lwork = -1;
  uni20::lapack::syevd(jobz, uplo, order, matrix.data(), order, result.eigenvalues.data(), &work_query, query_lwork,
                       &iwork_query, query_lwork);

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
  blas_int const liwork = std::max<blas_int>(1, iwork_query);
  std::vector<Scalar> work(static_cast<std::size_t>(lwork), Scalar{});
  std::vector<blas_int> iwork(static_cast<std::size_t>(liwork), 0);
  uni20::lapack::syevd(jobz, uplo, order, matrix.data(), order, result.eigenvalues.data(), work.data(), lwork,
                       iwork.data(), liwork);

  result.eigenvectors = compute_vectors ? std::move(matrix) : detail::ColumnMajorLapackMatrix<Scalar>(0, 0);
  return result;
}

/// \brief Solve a selected dense real symmetric eigensystem through LAPACK `syevr`.
///
/// \details Selects eigenpairs by 0-based inclusive index in ascending
///          eigenvalue order. If `compute_vectors` is true, `eigenvectors` is
///          an `n`-by-`m` column-major matrix whose columns are the selected
///          normalized eigenvectors.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real symmetric square matrix. Only the selected triangle is read.
/// \param first_index First 0-based eigenvalue index to compute.
/// \param last_index Last 0-based eigenvalue index to compute.
/// \param compute_vectors Whether to compute selected eigenvectors.
/// \param triangle Triangle of \p matrix supplied to LAPACK.
/// \return Selected eigenvalues and, optionally, selected eigenvectors.
template <uni20::LapackReal Scalar>
RealSymmetricEigensystem<Scalar> real_symmetric_eigensystem_index_range(detail::ColumnMajorLapackMatrix<Scalar> matrix,
                                                                        std::size_t first_index, std::size_t last_index,
                                                                        bool compute_vectors,
                                                                        MatrixFill triangle = MatrixFill::Upper)
{
  if (!std::cmp_equal(matrix.rows(), matrix.cols()))
  {
    throw std::invalid_argument("real_symmetric_eigensystem_index_range requires a square matrix");
  }

  std::size_t const n = static_cast<std::size_t>(matrix.rows());
  if (n == 0 || first_index > last_index || last_index >= n)
  {
    throw std::invalid_argument("real_symmetric_eigensystem_index_range received an invalid index range");
  }

  std::size_t const selected = last_index - first_index + 1;
  RealSymmetricEigensystem<Scalar> result;
  result.eigenvalues.resize(selected);

  blas_int const order = detail::checked_blas_int(n);
  blas_int const first = detail::checked_blas_int(first_index + 1);
  blas_int const last = detail::checked_blas_int(last_index + 1);
  char const jobz = compute_vectors ? 'V' : 'N';
  char const range = 'I';
  char const uplo = detail::lapack_uplo(triangle);
  detail::ColumnMajorLapackMatrix<Scalar> eigenvectors(compute_vectors ? n : 1, compute_vectors ? selected : 1);
  blas_int const ldz = compute_vectors ? order : 1;
  std::vector<blas_int> support(static_cast<std::size_t>(std::max<blas_int>(1, 2 * order)), 0);
  Scalar work_query{};
  blas_int iwork_query = 0;
  blas_int selected_count = 0;
  blas_int const query_lwork = -1;
  uni20::lapack::syevr(jobz, range, uplo, order, matrix.data(), order, Scalar{}, Scalar{}, first, last, Scalar{},
                       selected_count, result.eigenvalues.data(), eigenvectors.data(), ldz, support.data(), &work_query,
                       query_lwork, &iwork_query, query_lwork);

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
  blas_int const liwork = std::max<blas_int>(1, iwork_query);
  std::vector<Scalar> work(static_cast<std::size_t>(lwork), Scalar{});
  std::vector<blas_int> iwork(static_cast<std::size_t>(liwork), 0);
  selected_count = 0;
  uni20::lapack::syevr(jobz, range, uplo, order, matrix.data(), order, Scalar{}, Scalar{}, first, last, Scalar{},
                       selected_count, result.eigenvalues.data(), eigenvectors.data(), ldz, support.data(), work.data(),
                       lwork, iwork.data(), liwork);

  if (selected_count < 0 || static_cast<std::size_t>(selected_count) != selected)
  {
    throw std::runtime_error("LAPACK syevr returned an unexpected number of eigenvalues");
  }

  result.eigenvectors = compute_vectors ? std::move(eigenvectors) : detail::ColumnMajorLapackMatrix<Scalar>(0, 0);
  return result;
}

/// \brief Solve a dense type-1 generalized real symmetric eigensystem through LAPACK `sygv`.
/// \details Solves `A * x = lambda * B * x`, where `A` is real symmetric and
///          `B` is real symmetric positive definite. Eigenvalues are returned
///          in ascending order. If `compute_vectors` is true, `eigenvectors`
///          is an `n`-by-`n` column-major matrix whose columns are normalized
///          in the `B` metric.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real symmetric square matrix `A`. Only the selected triangle is read.
/// \param metric Real symmetric positive definite square matrix `B`.
/// \param compute_vectors Whether to compute generalized eigenvectors.
/// \param triangle Triangle of \p matrix and \p metric supplied to LAPACK.
/// \return Eigenvalues and, optionally, generalized eigenvectors.
template <uni20::LapackReal Scalar>
RealSymmetricEigensystem<Scalar> real_generalized_symmetric_eigensystem(detail::ColumnMajorLapackMatrix<Scalar> matrix,
                                                                        detail::ColumnMajorLapackMatrix<Scalar> metric,
                                                                        bool compute_vectors,
                                                                        MatrixFill triangle = MatrixFill::Upper)
{
  if (!std::cmp_equal(matrix.rows(), matrix.cols()))
  {
    throw std::invalid_argument("real_generalized_symmetric_eigensystem requires a square matrix");
  }
  if (!std::cmp_equal(metric.rows(), metric.cols()))
  {
    throw std::invalid_argument("real_generalized_symmetric_eigensystem requires a square metric matrix");
  }
  if (!std::cmp_equal(matrix.rows(), metric.rows()))
  {
    throw std::invalid_argument("real_generalized_symmetric_eigensystem received incompatible matrix sizes");
  }

  std::size_t const n = static_cast<std::size_t>(matrix.rows());
  RealSymmetricEigensystem<Scalar> result;
  result.eigenvalues.resize(n);
  if (n == 0)
  {
    result.eigenvectors = detail::ColumnMajorLapackMatrix<Scalar>(0, 0);
    return result;
  }

  blas_int const order = detail::checked_blas_int(n);
  char const jobz = compute_vectors ? 'V' : 'N';
  char const uplo = detail::lapack_uplo(triangle);
  Scalar work_query{};
  blas_int const query_lwork = -1;
  uni20::lapack::sygv(1, jobz, uplo, order, matrix.data(), order, metric.data(), order, result.eigenvalues.data(),
                      &work_query, query_lwork);

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
  std::vector<Scalar> work(static_cast<std::size_t>(lwork), Scalar{});
  uni20::lapack::sygv(1, jobz, uplo, order, matrix.data(), order, metric.data(), order, result.eigenvalues.data(),
                      work.data(), lwork);

  result.eigenvectors = compute_vectors ? std::move(matrix) : detail::ColumnMajorLapackMatrix<Scalar>(0, 0);
  return result;
}

/// \brief Solve a dense type-1 generalized real symmetric eigensystem through LAPACK `sygvd`.
/// \details Uses the divide-and-conquer generalized symmetric eigensystem
///          driver. Solves `A * x = lambda * B * x`, where `A` is real
///          symmetric and `B` is real symmetric positive definite. Eigenvalues
///          are returned in ascending order. If `compute_vectors` is true,
///          `eigenvectors` is an `n`-by-`n` column-major matrix whose columns
///          are normalized in the `B` metric.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real symmetric square matrix `A`. Only the selected triangle is read.
/// \param metric Real symmetric positive definite square matrix `B`.
/// \param compute_vectors Whether to compute generalized eigenvectors.
/// \param triangle Triangle of \p matrix and \p metric supplied to LAPACK.
/// \return Eigenvalues and, optionally, generalized eigenvectors.
template <uni20::LapackReal Scalar>
RealSymmetricEigensystem<Scalar>
real_generalized_symmetric_eigensystem_divide_and_conquer(detail::ColumnMajorLapackMatrix<Scalar> matrix,
                                                          detail::ColumnMajorLapackMatrix<Scalar> metric,
                                                          bool compute_vectors, MatrixFill triangle = MatrixFill::Upper)
{
  if (!std::cmp_equal(matrix.rows(), matrix.cols()))
  {
    throw std::invalid_argument("real_generalized_symmetric_eigensystem_divide_and_conquer requires a square matrix");
  }
  if (!std::cmp_equal(metric.rows(), metric.cols()))
  {
    throw std::invalid_argument(
        "real_generalized_symmetric_eigensystem_divide_and_conquer requires a square metric matrix");
  }
  if (!std::cmp_equal(matrix.rows(), metric.rows()))
  {
    throw std::invalid_argument(
        "real_generalized_symmetric_eigensystem_divide_and_conquer received incompatible matrix sizes");
  }

  std::size_t const n = static_cast<std::size_t>(matrix.rows());
  RealSymmetricEigensystem<Scalar> result;
  result.eigenvalues.resize(n);
  if (n == 0)
  {
    result.eigenvectors = detail::ColumnMajorLapackMatrix<Scalar>(0, 0);
    return result;
  }

  blas_int const order = detail::checked_blas_int(n);
  char const jobz = compute_vectors ? 'V' : 'N';
  char const uplo = detail::lapack_uplo(triangle);
  Scalar work_query{};
  blas_int iwork_query = 0;
  blas_int const query_lwork = -1;
  uni20::lapack::sygvd(1, jobz, uplo, order, matrix.data(), order, metric.data(), order, result.eigenvalues.data(),
                       &work_query, query_lwork, &iwork_query, query_lwork);

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
  blas_int const liwork = std::max<blas_int>(1, iwork_query);
  std::vector<Scalar> work(static_cast<std::size_t>(lwork), Scalar{});
  std::vector<blas_int> iwork(static_cast<std::size_t>(liwork), 0);
  uni20::lapack::sygvd(1, jobz, uplo, order, matrix.data(), order, metric.data(), order, result.eigenvalues.data(),
                       work.data(), lwork, iwork.data(), liwork);

  result.eigenvectors = compute_vectors ? std::move(matrix) : detail::ColumnMajorLapackMatrix<Scalar>(0, 0);
  return result;
}

/// \brief Solve a selected dense type-1 generalized real symmetric eigensystem through LAPACK `sygvx`.
/// \details Selects eigenpairs by 0-based inclusive index in ascending
///          eigenvalue order for `A * x = lambda * B * x`, where `A` is real
///          symmetric and `B` is real symmetric positive definite. If
///          `compute_vectors` is true, `eigenvectors` is an `n`-by-`m`
///          column-major matrix whose columns are normalized in the `B`
///          metric.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real symmetric square matrix `A`. Only the selected triangle is read.
/// \param metric Real symmetric positive definite square matrix `B`.
/// \param first_index First 0-based eigenvalue index to compute.
/// \param last_index Last 0-based eigenvalue index to compute.
/// \param compute_vectors Whether to compute selected generalized eigenvectors.
/// \param triangle Triangle of \p matrix and \p metric supplied to LAPACK.
/// \return Selected eigenvalues and, optionally, selected generalized eigenvectors.
template <uni20::LapackReal Scalar>
RealSymmetricEigensystem<Scalar> real_generalized_symmetric_eigensystem_index_range(
    detail::ColumnMajorLapackMatrix<Scalar> matrix, detail::ColumnMajorLapackMatrix<Scalar> metric,
    std::size_t first_index, std::size_t last_index, bool compute_vectors, MatrixFill triangle = MatrixFill::Upper)
{
  if (!std::cmp_equal(matrix.rows(), matrix.cols()))
  {
    throw std::invalid_argument("real_generalized_symmetric_eigensystem_index_range requires a square matrix");
  }
  if (!std::cmp_equal(metric.rows(), metric.cols()))
  {
    throw std::invalid_argument("real_generalized_symmetric_eigensystem_index_range requires a square metric matrix");
  }
  if (!std::cmp_equal(matrix.rows(), metric.rows()))
  {
    throw std::invalid_argument(
        "real_generalized_symmetric_eigensystem_index_range received incompatible matrix sizes");
  }

  std::size_t const n = static_cast<std::size_t>(matrix.rows());
  if (n == 0 || first_index > last_index || last_index >= n)
  {
    throw std::invalid_argument("real_generalized_symmetric_eigensystem_index_range received an invalid index range");
  }

  std::size_t const selected = last_index - first_index + 1;
  RealSymmetricEigensystem<Scalar> result;
  result.eigenvalues.resize(n);

  blas_int const order = detail::checked_blas_int(n);
  blas_int const first = detail::checked_blas_int(first_index + 1);
  blas_int const last = detail::checked_blas_int(last_index + 1);
  char const jobz = compute_vectors ? 'V' : 'N';
  char const range = 'I';
  char const uplo = detail::lapack_uplo(triangle);
  detail::ColumnMajorLapackMatrix<Scalar> eigenvectors(compute_vectors ? n : 1, compute_vectors ? selected : 1);
  blas_int const ldz = compute_vectors ? order : 1;
  std::vector<blas_int> iwork(static_cast<std::size_t>(std::max<blas_int>(1, 5 * order)), 0);
  std::vector<blas_int> ifail(static_cast<std::size_t>(std::max<blas_int>(1, order)), 0);
  Scalar work_query{};
  blas_int selected_count = 0;
  blas_int const query_lwork = -1;
  uni20::lapack::sygvx(1, jobz, range, uplo, order, matrix.data(), order, metric.data(), order, Scalar{}, Scalar{},
                       first, last, Scalar{}, selected_count, result.eigenvalues.data(), eigenvectors.data(), ldz,
                       &work_query, query_lwork, iwork.data(), ifail.data());

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
  std::vector<Scalar> work(static_cast<std::size_t>(lwork), Scalar{});
  selected_count = 0;
  uni20::lapack::sygvx(1, jobz, range, uplo, order, matrix.data(), order, metric.data(), order, Scalar{}, Scalar{},
                       first, last, Scalar{}, selected_count, result.eigenvalues.data(), eigenvectors.data(), ldz,
                       work.data(), lwork, iwork.data(), ifail.data());

  if (selected_count < 0 || static_cast<std::size_t>(selected_count) != selected)
  {
    throw std::runtime_error("LAPACK sygvx returned an unexpected number of eigenvalues");
  }

  result.eigenvalues.resize(selected);
  result.eigenvectors = compute_vectors ? std::move(eigenvectors) : detail::ColumnMajorLapackMatrix<Scalar>(0, 0);
  return result;
}

/// \brief Solve a dense complex Hermitian eigensystem through LAPACK `heev`.
/// \details Eigenvalues are returned in ascending order. If `compute_vectors`
///          is true, `eigenvectors` is an `n`-by-`n` column-major matrix whose
///          columns are normalized eigenvectors.
/// \tparam Real Underlying real precision.
/// \param matrix Complex Hermitian square matrix. Only the selected triangle is read.
/// \param compute_vectors Whether to compute eigenvectors.
/// \param triangle Triangle of \p matrix supplied to LAPACK.
/// \return Eigenvalues and, optionally, eigenvectors.
template <uni20::LapackComplexReal Real>
ComplexHermitianEigensystem<Real>
complex_hermitian_eigensystem(detail::ColumnMajorLapackMatrix<uni20::complex<Real>> matrix, bool compute_vectors,
                              MatrixFill triangle = MatrixFill::Upper)
{
  if (!std::cmp_equal(matrix.rows(), matrix.cols()))
  {
    throw std::invalid_argument("complex_hermitian_eigensystem requires a square matrix");
  }

  using Complex = uni20::complex<Real>;
  std::size_t const n = static_cast<std::size_t>(matrix.rows());
  ComplexHermitianEigensystem<Real> result;
  result.eigenvalues.resize(n);
  if (n == 0)
  {
    result.eigenvectors = detail::ColumnMajorLapackMatrix<Complex>(0, 0);
    return result;
  }

  blas_int const order = detail::checked_blas_int(n);
  char const jobz = compute_vectors ? 'V' : 'N';
  char const uplo = detail::lapack_uplo(triangle);
  std::vector<Real> rwork(static_cast<std::size_t>(std::max<blas_int>(1, 3 * order - 2)), Real{});
  Complex work_query{};
  blas_int const query_lwork = -1;
  uni20::lapack::heev(jobz, uplo, order, matrix.data(), order, result.eigenvalues.data(), &work_query, query_lwork,
                      rwork.data());

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(std::real(work_query)));
  std::vector<Complex> work(static_cast<std::size_t>(lwork), Complex{});
  uni20::lapack::heev(jobz, uplo, order, matrix.data(), order, result.eigenvalues.data(), work.data(), lwork,
                      rwork.data());

  result.eigenvectors = compute_vectors ? std::move(matrix) : detail::ColumnMajorLapackMatrix<Complex>(0, 0);
  return result;
}

/// \brief Solve a dense complex Hermitian eigensystem through LAPACK `heevd`.
/// \details Uses the divide-and-conquer Hermitian eigensystem driver.
///          Eigenvalues are returned in ascending order.
/// \tparam Real Underlying real precision.
/// \param matrix Complex Hermitian square matrix. Only the selected triangle is read.
/// \param compute_vectors Whether to compute eigenvectors.
/// \param triangle Triangle of \p matrix supplied to LAPACK.
/// \return Eigenvalues and, optionally, eigenvectors.
template <uni20::LapackComplexReal Real>
ComplexHermitianEigensystem<Real>
complex_hermitian_eigensystem_divide_and_conquer(detail::ColumnMajorLapackMatrix<uni20::complex<Real>> matrix,
                                                 bool compute_vectors, MatrixFill triangle = MatrixFill::Upper)
{
  if (!std::cmp_equal(matrix.rows(), matrix.cols()))
  {
    throw std::invalid_argument("complex_hermitian_eigensystem_divide_and_conquer requires a square matrix");
  }

  using Complex = uni20::complex<Real>;
  std::size_t const n = static_cast<std::size_t>(matrix.rows());
  ComplexHermitianEigensystem<Real> result;
  result.eigenvalues.resize(n);
  if (n == 0)
  {
    result.eigenvectors = detail::ColumnMajorLapackMatrix<Complex>(0, 0);
    return result;
  }

  blas_int const order = detail::checked_blas_int(n);
  char const jobz = compute_vectors ? 'V' : 'N';
  char const uplo = detail::lapack_uplo(triangle);
  Complex work_query{};
  Real rwork_query{};
  blas_int iwork_query = 0;
  blas_int const query_lwork = -1;
  uni20::lapack::heevd(jobz, uplo, order, matrix.data(), order, result.eigenvalues.data(), &work_query, query_lwork,
                       &rwork_query, query_lwork, &iwork_query, query_lwork);

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(std::real(work_query)));
  blas_int const lrwork = std::max<blas_int>(1, static_cast<blas_int>(rwork_query));
  blas_int const liwork = std::max<blas_int>(1, iwork_query);
  std::vector<Complex> work(static_cast<std::size_t>(lwork), Complex{});
  std::vector<Real> rwork(static_cast<std::size_t>(lrwork), Real{});
  std::vector<blas_int> iwork(static_cast<std::size_t>(liwork), 0);
  uni20::lapack::heevd(jobz, uplo, order, matrix.data(), order, result.eigenvalues.data(), work.data(), lwork,
                       rwork.data(), lrwork, iwork.data(), liwork);

  result.eigenvectors = compute_vectors ? std::move(matrix) : detail::ColumnMajorLapackMatrix<Complex>(0, 0);
  return result;
}

/// \brief Solve a selected dense complex Hermitian eigensystem through LAPACK `heevr`.
/// \details Selects eigenpairs by 0-based inclusive index in ascending
///          eigenvalue order.
/// \tparam Real Underlying real precision.
/// \param matrix Complex Hermitian square matrix. Only the selected triangle is read.
/// \param first_index First 0-based eigenvalue index to compute.
/// \param last_index Last 0-based eigenvalue index to compute.
/// \param compute_vectors Whether to compute selected eigenvectors.
/// \param triangle Triangle of \p matrix supplied to LAPACK.
/// \return Selected eigenvalues and, optionally, selected eigenvectors.
template <uni20::LapackComplexReal Real>
ComplexHermitianEigensystem<Real>
complex_hermitian_eigensystem_index_range(detail::ColumnMajorLapackMatrix<uni20::complex<Real>> matrix,
                                          std::size_t first_index, std::size_t last_index, bool compute_vectors,
                                          MatrixFill triangle = MatrixFill::Upper)
{
  if (!std::cmp_equal(matrix.rows(), matrix.cols()))
  {
    throw std::invalid_argument("complex_hermitian_eigensystem_index_range requires a square matrix");
  }

  using Complex = uni20::complex<Real>;
  std::size_t const n = static_cast<std::size_t>(matrix.rows());
  if (n == 0 || first_index > last_index || last_index >= n)
  {
    throw std::invalid_argument("complex_hermitian_eigensystem_index_range received an invalid index range");
  }

  std::size_t const selected = last_index - first_index + 1;
  ComplexHermitianEigensystem<Real> result;
  result.eigenvalues.resize(selected);

  blas_int const order = detail::checked_blas_int(n);
  blas_int const first = detail::checked_blas_int(first_index + 1);
  blas_int const last = detail::checked_blas_int(last_index + 1);
  char const jobz = compute_vectors ? 'V' : 'N';
  char const range = 'I';
  char const uplo = detail::lapack_uplo(triangle);
  detail::ColumnMajorLapackMatrix<Complex> eigenvectors(compute_vectors ? n : 1, compute_vectors ? selected : 1);
  blas_int const ldz = compute_vectors ? order : 1;
  std::vector<blas_int> support(static_cast<std::size_t>(std::max<blas_int>(1, 2 * order)), 0);
  Complex work_query{};
  Real rwork_query{};
  blas_int iwork_query = 0;
  blas_int selected_count = 0;
  blas_int const query_lwork = -1;
  uni20::lapack::heevr(jobz, range, uplo, order, matrix.data(), order, Real{}, Real{}, first, last, Real{},
                       selected_count, result.eigenvalues.data(), eigenvectors.data(), ldz, support.data(), &work_query,
                       query_lwork, &rwork_query, query_lwork, &iwork_query, query_lwork);

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(std::real(work_query)));
  blas_int const lrwork = std::max<blas_int>(1, static_cast<blas_int>(rwork_query));
  blas_int const liwork = std::max<blas_int>(1, iwork_query);
  std::vector<Complex> work(static_cast<std::size_t>(lwork), Complex{});
  std::vector<Real> rwork(static_cast<std::size_t>(lrwork), Real{});
  std::vector<blas_int> iwork(static_cast<std::size_t>(liwork), 0);
  selected_count = 0;
  uni20::lapack::heevr(jobz, range, uplo, order, matrix.data(), order, Real{}, Real{}, first, last, Real{},
                       selected_count, result.eigenvalues.data(), eigenvectors.data(), ldz, support.data(), work.data(),
                       lwork, rwork.data(), lrwork, iwork.data(), liwork);

  if (selected_count < 0 || static_cast<std::size_t>(selected_count) != selected)
  {
    throw std::runtime_error("LAPACK heevr returned an unexpected number of eigenvalues");
  }

  result.eigenvectors = compute_vectors ? std::move(eigenvectors) : detail::ColumnMajorLapackMatrix<Complex>(0, 0);
  return result;
}

/// \brief Solve a dense type-1 generalized complex Hermitian eigensystem through LAPACK `hegv`.
/// \details Solves `A * x = lambda * B * x`, where `A` is complex Hermitian
///          and `B` is complex Hermitian positive definite. Eigenvalues are
///          returned in ascending order.
/// \tparam Real Underlying real precision.
/// \param matrix Complex Hermitian square matrix `A`. Only the selected triangle is read.
/// \param metric Complex Hermitian positive definite square matrix `B`.
/// \param compute_vectors Whether to compute generalized eigenvectors.
/// \param triangle Triangle of \p matrix and \p metric supplied to LAPACK.
/// \return Eigenvalues and, optionally, generalized eigenvectors.
template <uni20::LapackComplexReal Real>
ComplexHermitianEigensystem<Real>
complex_generalized_hermitian_eigensystem(detail::ColumnMajorLapackMatrix<uni20::complex<Real>> matrix,
                                          detail::ColumnMajorLapackMatrix<uni20::complex<Real>> metric,
                                          bool compute_vectors, MatrixFill triangle = MatrixFill::Upper)
{
  if (!std::cmp_equal(matrix.rows(), matrix.cols()))
  {
    throw std::invalid_argument("complex_generalized_hermitian_eigensystem requires a square matrix");
  }
  if (!std::cmp_equal(metric.rows(), metric.cols()))
  {
    throw std::invalid_argument("complex_generalized_hermitian_eigensystem requires a square metric matrix");
  }
  if (!std::cmp_equal(matrix.rows(), metric.rows()))
  {
    throw std::invalid_argument("complex_generalized_hermitian_eigensystem received incompatible matrix sizes");
  }

  using Complex = uni20::complex<Real>;
  std::size_t const n = static_cast<std::size_t>(matrix.rows());
  ComplexHermitianEigensystem<Real> result;
  result.eigenvalues.resize(n);
  if (n == 0)
  {
    result.eigenvectors = detail::ColumnMajorLapackMatrix<Complex>(0, 0);
    return result;
  }

  blas_int const order = detail::checked_blas_int(n);
  char const jobz = compute_vectors ? 'V' : 'N';
  char const uplo = detail::lapack_uplo(triangle);
  std::vector<Real> rwork(static_cast<std::size_t>(std::max<blas_int>(1, 3 * order - 2)), Real{});
  Complex work_query{};
  blas_int const query_lwork = -1;
  uni20::lapack::hegv(1, jobz, uplo, order, matrix.data(), order, metric.data(), order, result.eigenvalues.data(),
                      &work_query, query_lwork, rwork.data());

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(std::real(work_query)));
  std::vector<Complex> work(static_cast<std::size_t>(lwork), Complex{});
  uni20::lapack::hegv(1, jobz, uplo, order, matrix.data(), order, metric.data(), order, result.eigenvalues.data(),
                      work.data(), lwork, rwork.data());

  result.eigenvectors = compute_vectors ? std::move(matrix) : detail::ColumnMajorLapackMatrix<Complex>(0, 0);
  return result;
}

/// \brief Solve a dense type-1 generalized complex Hermitian eigensystem through LAPACK `hegvd`.
/// \details Uses the divide-and-conquer generalized Hermitian eigensystem
///          driver for `A * x = lambda * B * x`.
/// \tparam Real Underlying real precision.
/// \param matrix Complex Hermitian square matrix `A`. Only the selected triangle is read.
/// \param metric Complex Hermitian positive definite square matrix `B`.
/// \param compute_vectors Whether to compute generalized eigenvectors.
/// \param triangle Triangle of \p matrix and \p metric supplied to LAPACK.
/// \return Eigenvalues and, optionally, generalized eigenvectors.
template <uni20::LapackComplexReal Real>
ComplexHermitianEigensystem<Real> complex_generalized_hermitian_eigensystem_divide_and_conquer(
    detail::ColumnMajorLapackMatrix<uni20::complex<Real>> matrix,
    detail::ColumnMajorLapackMatrix<uni20::complex<Real>> metric, bool compute_vectors,
    MatrixFill triangle = MatrixFill::Upper)
{
  if (!std::cmp_equal(matrix.rows(), matrix.cols()))
  {
    throw std::invalid_argument(
        "complex_generalized_hermitian_eigensystem_divide_and_conquer requires a square matrix");
  }
  if (!std::cmp_equal(metric.rows(), metric.cols()))
  {
    throw std::invalid_argument(
        "complex_generalized_hermitian_eigensystem_divide_and_conquer requires a square metric matrix");
  }
  if (!std::cmp_equal(matrix.rows(), metric.rows()))
  {
    throw std::invalid_argument(
        "complex_generalized_hermitian_eigensystem_divide_and_conquer received incompatible matrix sizes");
  }

  using Complex = uni20::complex<Real>;
  std::size_t const n = static_cast<std::size_t>(matrix.rows());
  ComplexHermitianEigensystem<Real> result;
  result.eigenvalues.resize(n);
  if (n == 0)
  {
    result.eigenvectors = detail::ColumnMajorLapackMatrix<Complex>(0, 0);
    return result;
  }

  blas_int const order = detail::checked_blas_int(n);
  char const jobz = compute_vectors ? 'V' : 'N';
  char const uplo = detail::lapack_uplo(triangle);
  Complex work_query{};
  Real rwork_query{};
  blas_int iwork_query = 0;
  blas_int const query_lwork = -1;
  uni20::lapack::hegvd(1, jobz, uplo, order, matrix.data(), order, metric.data(), order, result.eigenvalues.data(),
                       &work_query, query_lwork, &rwork_query, query_lwork, &iwork_query, query_lwork);

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(std::real(work_query)));
  blas_int const lrwork = std::max<blas_int>(1, static_cast<blas_int>(rwork_query));
  blas_int const liwork = std::max<blas_int>(1, iwork_query);
  std::vector<Complex> work(static_cast<std::size_t>(lwork), Complex{});
  std::vector<Real> rwork(static_cast<std::size_t>(lrwork), Real{});
  std::vector<blas_int> iwork(static_cast<std::size_t>(liwork), 0);
  uni20::lapack::hegvd(1, jobz, uplo, order, matrix.data(), order, metric.data(), order, result.eigenvalues.data(),
                       work.data(), lwork, rwork.data(), lrwork, iwork.data(), liwork);

  result.eigenvectors = compute_vectors ? std::move(matrix) : detail::ColumnMajorLapackMatrix<Complex>(0, 0);
  return result;
}

/// \brief Solve a selected dense type-1 generalized complex Hermitian eigensystem through LAPACK `hegvx`.
/// \details Selects eigenpairs by 0-based inclusive index in ascending
///          eigenvalue order for `A * x = lambda * B * x`.
/// \tparam Real Underlying real precision.
/// \param matrix Complex Hermitian square matrix `A`. Only the selected triangle is read.
/// \param metric Complex Hermitian positive definite square matrix `B`.
/// \param first_index First 0-based eigenvalue index to compute.
/// \param last_index Last 0-based eigenvalue index to compute.
/// \param compute_vectors Whether to compute selected generalized eigenvectors.
/// \param triangle Triangle of \p matrix and \p metric supplied to LAPACK.
/// \return Selected eigenvalues and, optionally, selected generalized eigenvectors.
template <uni20::LapackComplexReal Real>
ComplexHermitianEigensystem<Real>
complex_generalized_hermitian_eigensystem_index_range(detail::ColumnMajorLapackMatrix<uni20::complex<Real>> matrix,
                                                      detail::ColumnMajorLapackMatrix<uni20::complex<Real>> metric,
                                                      std::size_t first_index, std::size_t last_index,
                                                      bool compute_vectors, MatrixFill triangle = MatrixFill::Upper)
{
  if (!std::cmp_equal(matrix.rows(), matrix.cols()))
  {
    throw std::invalid_argument("complex_generalized_hermitian_eigensystem_index_range requires a square matrix");
  }
  if (!std::cmp_equal(metric.rows(), metric.cols()))
  {
    throw std::invalid_argument(
        "complex_generalized_hermitian_eigensystem_index_range requires a square metric matrix");
  }
  if (!std::cmp_equal(matrix.rows(), metric.rows()))
  {
    throw std::invalid_argument(
        "complex_generalized_hermitian_eigensystem_index_range received incompatible matrix sizes");
  }

  using Complex = uni20::complex<Real>;
  std::size_t const n = static_cast<std::size_t>(matrix.rows());
  if (n == 0 || first_index > last_index || last_index >= n)
  {
    throw std::invalid_argument(
        "complex_generalized_hermitian_eigensystem_index_range received an invalid index range");
  }

  std::size_t const selected = last_index - first_index + 1;
  ComplexHermitianEigensystem<Real> result;
  result.eigenvalues.resize(n);

  blas_int const order = detail::checked_blas_int(n);
  blas_int const first = detail::checked_blas_int(first_index + 1);
  blas_int const last = detail::checked_blas_int(last_index + 1);
  char const jobz = compute_vectors ? 'V' : 'N';
  char const range = 'I';
  char const uplo = detail::lapack_uplo(triangle);
  detail::ColumnMajorLapackMatrix<Complex> eigenvectors(compute_vectors ? n : 1, compute_vectors ? selected : 1);
  blas_int const ldz = compute_vectors ? order : 1;
  std::vector<Real> rwork(static_cast<std::size_t>(std::max<blas_int>(1, 7 * order)), Real{});
  std::vector<blas_int> iwork(static_cast<std::size_t>(std::max<blas_int>(1, 5 * order)), 0);
  std::vector<blas_int> ifail(static_cast<std::size_t>(std::max<blas_int>(1, order)), 0);
  Complex work_query{};
  blas_int selected_count = 0;
  blas_int const query_lwork = -1;
  uni20::lapack::hegvx(1, jobz, range, uplo, order, matrix.data(), order, metric.data(), order, Real{}, Real{}, first,
                       last, Real{}, selected_count, result.eigenvalues.data(), eigenvectors.data(), ldz, &work_query,
                       query_lwork, rwork.data(), iwork.data(), ifail.data());

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(std::real(work_query)));
  std::vector<Complex> work(static_cast<std::size_t>(lwork), Complex{});
  selected_count = 0;
  uni20::lapack::hegvx(1, jobz, range, uplo, order, matrix.data(), order, metric.data(), order, Real{}, Real{}, first,
                       last, Real{}, selected_count, result.eigenvalues.data(), eigenvectors.data(), ldz, work.data(),
                       lwork, rwork.data(), iwork.data(), ifail.data());

  if (selected_count < 0 || static_cast<std::size_t>(selected_count) != selected)
  {
    throw std::runtime_error("LAPACK hegvx returned an unexpected number of eigenvalues");
  }

  result.eigenvalues.resize(selected);
  result.eigenvectors = compute_vectors ? std::move(eigenvectors) : detail::ColumnMajorLapackMatrix<Complex>(0, 0);
  return result;
}

/// \brief Compute a compact dense real QR factorization through LAPACK `geqrf`.
/// \details Stores Householder reflectors and scalar factors in LAPACK's
///          compact QR representation. Use `apply_real_qr_factor` to apply the
///          encoded orthogonal factor without explicitly materializing `Q`.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real matrix to factor.
/// \return Compact QR reflector data and scalar factors.
template <uni20::LapackReal Scalar>
RealCompactQrFactorization<Scalar> real_compact_qr_factorization(detail::ColumnMajorLapackMatrix<Scalar> matrix)
{
  std::size_t const rows = static_cast<std::size_t>(matrix.rows());
  std::size_t const cols = static_cast<std::size_t>(matrix.cols());
  std::size_t const rank = std::min(rows, cols);

  RealCompactQrFactorization<Scalar> result;
  result.reflectors = std::move(matrix);
  result.tau.resize(rank);
  result.rank = rank;
  if (rank == 0)
  {
    return result;
  }

  blas_int const m = detail::checked_blas_int(rows);
  blas_int const n = detail::checked_blas_int(cols);
  blas_int const lda = std::max<blas_int>(1, m);

  Scalar geqrf_work_query{};
  blas_int const query_lwork = -1;
  uni20::lapack::geqrf(m, n, result.reflectors.data(), lda, result.tau.data(), &geqrf_work_query, query_lwork);

  blas_int const geqrf_lwork = std::max<blas_int>(1, static_cast<blas_int>(geqrf_work_query));
  std::vector<Scalar> geqrf_work(static_cast<std::size_t>(geqrf_lwork), Scalar{});
  uni20::lapack::geqrf(m, n, result.reflectors.data(), lda, result.tau.data(), geqrf_work.data(), geqrf_lwork);

  return result;
}

/// \brief Apply the orthogonal factor from a compact real QR factorization.
/// \details Applies `Q`, or `Q^T`, from a Householder QR factorization to a
///          dense target matrix using LAPACK `ormqr`.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param factorization Compact QR factorization returned by
///        `real_compact_qr_factorization`.
/// \param target Matrix overwritten by `op(Q) * target` or `target * op(Q)`.
/// \param side Whether `op(Q)` multiplies \p target from the left or right.
/// \param transpose Matrix operation applied to `Q`.
/// \return Matrix after applying the encoded orthogonal factor.
template <uni20::LapackReal Scalar>
detail::ColumnMajorLapackMatrix<Scalar> apply_real_qr_factor(RealCompactQrFactorization<Scalar> const& factorization,
                                                             detail::ColumnMajorLapackMatrix<Scalar> target,
                                                             MatrixSide side = MatrixSide::Left,
                                                             MatrixTranspose transpose = MatrixTranspose::None)
{
  std::size_t const reflector_rows = static_cast<std::size_t>(factorization.reflectors.rows());
  std::size_t const reflector_cols = static_cast<std::size_t>(factorization.reflectors.cols());
  if (factorization.rank > std::min(reflector_rows, reflector_cols))
  {
    throw std::invalid_argument("apply_real_qr_factor received inconsistent QR reflector dimensions");
  }
  if (factorization.tau.size() < factorization.rank)
  {
    throw std::invalid_argument("apply_real_qr_factor received too few QR scalar factors");
  }

  if (side == MatrixSide::Left && !std::cmp_equal(target.rows(), reflector_rows))
  {
    throw std::invalid_argument("apply_real_qr_factor left application requires matching target row count");
  }
  if (side == MatrixSide::Right && !std::cmp_equal(target.cols(), reflector_rows))
  {
    throw std::invalid_argument("apply_real_qr_factor right application requires matching target column count");
  }
  if (target.rows() == 0 || target.cols() == 0 || factorization.rank == 0)
  {
    return target;
  }

  detail::ColumnMajorLapackMatrix<Scalar> reflectors = factorization.reflectors;
  std::vector<Scalar> tau = factorization.tau;
  blas_int const m = detail::checked_blas_int(target.rows());
  blas_int const n = detail::checked_blas_int(target.cols());
  blas_int const k = detail::checked_blas_int(factorization.rank);
  blas_int const lda = std::max<blas_int>(1, detail::checked_blas_int(reflector_rows));
  blas_int const ldc = std::max<blas_int>(1, m);
  char const side_char = detail::lapack_side(side);
  char const trans = detail::lapack_transpose(transpose);

  Scalar work_query{};
  blas_int const query_lwork = -1;
  uni20::lapack::ormqr(side_char, trans, m, n, k, reflectors.data(), lda, tau.data(), target.data(), ldc, &work_query,
                       query_lwork);

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
  std::vector<Scalar> work(static_cast<std::size_t>(lwork), Scalar{});
  uni20::lapack::ormqr(side_char, trans, m, n, k, reflectors.data(), lda, tau.data(), target.data(), ldc, work.data(),
                       lwork);

  return target;
}

/// \brief Compute a reduced dense real QR factorization through LAPACK `geqrf` and `orgqr`.
/// \details Returns `A = Q * R`, where `Q` has orthonormal columns and
///          `min(rows, cols)` columns. `R` has `min(rows, cols)` rows.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real matrix to factor.
/// \return Reduced QR factors.
template <uni20::LapackReal Scalar>
RealQrFactorization<Scalar> real_qr_factorization(detail::ColumnMajorLapackMatrix<Scalar> matrix)
{
  auto compact = real_compact_qr_factorization(std::move(matrix));
  std::size_t const rows = static_cast<std::size_t>(compact.reflectors.rows());
  std::size_t const cols = static_cast<std::size_t>(compact.reflectors.cols());
  std::size_t const rank = compact.rank;

  RealQrFactorization<Scalar> result;
  result.q = detail::ColumnMajorLapackMatrix<Scalar>(rows, rank);
  result.r = detail::ColumnMajorLapackMatrix<Scalar>(rank, cols);
  if (rank == 0)
  {
    return result;
  }
  laset(result.r, Scalar{}, Scalar{}, MatrixFill::All);

  blas_int const m = detail::checked_blas_int(rows);
  blas_int const k = detail::checked_blas_int(rank);
  blas_int const lda = std::max<blas_int>(1, m);

  for (std::size_t row = 0; row < rank; ++row)
  {
    for (std::size_t col = row; col < cols; ++col)
    {
      result.r[row, col] = compact.reflectors[row, col];
    }
  }

  for (std::size_t col = 0; col < rank; ++col)
  {
    for (std::size_t row = 0; row < rows; ++row)
    {
      result.q[row, col] = compact.reflectors[row, col];
    }
  }

  Scalar orgqr_work_query{};
  blas_int const query_lwork = -1;
  uni20::lapack::orgqr(m, k, k, result.q.data(), lda, compact.tau.data(), &orgqr_work_query, query_lwork);

  blas_int const orgqr_lwork = std::max<blas_int>(1, static_cast<blas_int>(orgqr_work_query));
  std::vector<Scalar> orgqr_work(static_cast<std::size_t>(orgqr_lwork), Scalar{});
  uni20::lapack::orgqr(m, k, k, result.q.data(), lda, compact.tau.data(), orgqr_work.data(), orgqr_lwork);

  return result;
}

/// \brief Compute a compact dense real LQ factorization through LAPACK `gelqf`.
/// \details Stores Householder reflectors and scalar factors in LAPACK's
///          compact LQ representation. Use `apply_real_lq_factor` to apply the
///          encoded orthogonal factor without explicitly materializing `Q`.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real matrix to factor.
/// \return Compact LQ reflector data and scalar factors.
template <uni20::LapackReal Scalar>
RealCompactLqFactorization<Scalar> real_compact_lq_factorization(detail::ColumnMajorLapackMatrix<Scalar> matrix)
{
  std::size_t const rows = static_cast<std::size_t>(matrix.rows());
  std::size_t const cols = static_cast<std::size_t>(matrix.cols());
  std::size_t const rank = std::min(rows, cols);

  RealCompactLqFactorization<Scalar> result;
  result.reflectors = std::move(matrix);
  result.tau.resize(rank);
  result.rank = rank;
  if (rank == 0)
  {
    return result;
  }

  blas_int const m = detail::checked_blas_int(rows);
  blas_int const n = detail::checked_blas_int(cols);
  blas_int const lda = std::max<blas_int>(1, m);

  Scalar gelqf_work_query{};
  blas_int const query_lwork = -1;
  uni20::lapack::gelqf(m, n, result.reflectors.data(), lda, result.tau.data(), &gelqf_work_query, query_lwork);

  blas_int const gelqf_lwork = std::max<blas_int>(1, static_cast<blas_int>(gelqf_work_query));
  std::vector<Scalar> gelqf_work(static_cast<std::size_t>(gelqf_lwork), Scalar{});
  uni20::lapack::gelqf(m, n, result.reflectors.data(), lda, result.tau.data(), gelqf_work.data(), gelqf_lwork);

  return result;
}

/// \brief Apply the orthogonal factor from a compact real LQ factorization.
/// \details Applies `Q`, or `Q^T`, from a Householder LQ factorization to a
///          dense target matrix using LAPACK `ormlq`.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param factorization Compact LQ factorization returned by
///        `real_compact_lq_factorization`.
/// \param target Matrix overwritten by `op(Q) * target` or `target * op(Q)`.
/// \param side Whether `op(Q)` multiplies \p target from the left or right.
/// \param transpose Matrix operation applied to `Q`.
/// \return Matrix after applying the encoded orthogonal factor.
template <uni20::LapackReal Scalar>
detail::ColumnMajorLapackMatrix<Scalar> apply_real_lq_factor(RealCompactLqFactorization<Scalar> const& factorization,
                                                             detail::ColumnMajorLapackMatrix<Scalar> target,
                                                             MatrixSide side = MatrixSide::Right,
                                                             MatrixTranspose transpose = MatrixTranspose::None)
{
  std::size_t const reflector_rows = static_cast<std::size_t>(factorization.reflectors.rows());
  std::size_t const reflector_cols = static_cast<std::size_t>(factorization.reflectors.cols());
  if (factorization.rank > std::min(reflector_rows, reflector_cols))
  {
    throw std::invalid_argument("apply_real_lq_factor received inconsistent LQ reflector dimensions");
  }
  if (factorization.tau.size() < factorization.rank)
  {
    throw std::invalid_argument("apply_real_lq_factor received too few LQ scalar factors");
  }

  if (side == MatrixSide::Left && !std::cmp_equal(target.rows(), reflector_cols))
  {
    throw std::invalid_argument("apply_real_lq_factor left application requires matching target row count");
  }
  if (side == MatrixSide::Right && !std::cmp_equal(target.cols(), reflector_cols))
  {
    throw std::invalid_argument("apply_real_lq_factor right application requires matching target column count");
  }
  if (target.rows() == 0 || target.cols() == 0 || factorization.rank == 0)
  {
    return target;
  }

  detail::ColumnMajorLapackMatrix<Scalar> reflectors = factorization.reflectors;
  std::vector<Scalar> tau = factorization.tau;
  blas_int const m = detail::checked_blas_int(target.rows());
  blas_int const n = detail::checked_blas_int(target.cols());
  blas_int const k = detail::checked_blas_int(factorization.rank);
  blas_int const lda = std::max<blas_int>(1, detail::checked_blas_int(reflector_rows));
  blas_int const ldc = std::max<blas_int>(1, m);
  char const side_char = detail::lapack_side(side);
  char const trans = detail::lapack_transpose(transpose);

  Scalar work_query{};
  blas_int const query_lwork = -1;
  uni20::lapack::ormlq(side_char, trans, m, n, k, reflectors.data(), lda, tau.data(), target.data(), ldc, &work_query,
                       query_lwork);

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
  std::vector<Scalar> work(static_cast<std::size_t>(lwork), Scalar{});
  uni20::lapack::ormlq(side_char, trans, m, n, k, reflectors.data(), lda, tau.data(), target.data(), ldc, work.data(),
                       lwork);

  return target;
}

/// \brief Compute a reduced dense real LQ factorization through LAPACK `gelqf` and `orglq`.
/// \details Returns `A = L * Q`, where `Q` has orthonormal rows and
///          `min(rows, cols)` rows. `L` has `min(rows, cols)` columns.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real matrix to factor.
/// \return Reduced LQ factors.
template <uni20::LapackReal Scalar>
RealLqFactorization<Scalar> real_lq_factorization(detail::ColumnMajorLapackMatrix<Scalar> matrix)
{
  auto compact = real_compact_lq_factorization(std::move(matrix));
  std::size_t const rows = static_cast<std::size_t>(compact.reflectors.rows());
  std::size_t const cols = static_cast<std::size_t>(compact.reflectors.cols());
  std::size_t const rank = compact.rank;

  RealLqFactorization<Scalar> result;
  result.l = detail::ColumnMajorLapackMatrix<Scalar>(rows, rank);
  result.q = detail::ColumnMajorLapackMatrix<Scalar>(rank, cols);
  if (rank == 0)
  {
    return result;
  }
  laset(result.l, Scalar{}, Scalar{}, MatrixFill::All);

  blas_int const n = detail::checked_blas_int(cols);
  blas_int const k = detail::checked_blas_int(rank);
  blas_int const lda = std::max<blas_int>(1, k);

  for (std::size_t row = 0; row < rows; ++row)
  {
    for (std::size_t col = 0; col < rank && col <= row; ++col)
    {
      result.l[row, col] = compact.reflectors[row, col];
    }
  }

  for (std::size_t row = 0; row < rank; ++row)
  {
    for (std::size_t col = 0; col < cols; ++col)
    {
      result.q[row, col] = compact.reflectors[row, col];
    }
  }

  Scalar orglq_work_query{};
  blas_int const query_lwork = -1;
  uni20::lapack::orglq(k, n, k, result.q.data(), lda, compact.tau.data(), &orglq_work_query, query_lwork);

  blas_int const orglq_lwork = std::max<blas_int>(1, static_cast<blas_int>(orglq_work_query));
  std::vector<Scalar> orglq_work(static_cast<std::size_t>(orglq_lwork), Scalar{});
  uni20::lapack::orglq(k, n, k, result.q.data(), lda, compact.tau.data(), orglq_work.data(), orglq_lwork);

  return result;
}

/// \brief Compute a compact dense real QL factorization through LAPACK `geqlf`.
/// \details Stores Householder reflectors and scalar factors in LAPACK's
///          compact QL representation. Use `apply_real_ql_factor` to apply the
///          encoded orthogonal factor without explicitly materializing `Q`.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real matrix to factor.
/// \return Compact QL reflector data and scalar factors.
template <uni20::LapackReal Scalar>
RealCompactQlFactorization<Scalar> real_compact_ql_factorization(detail::ColumnMajorLapackMatrix<Scalar> matrix)
{
  std::size_t const rows = static_cast<std::size_t>(matrix.rows());
  std::size_t const cols = static_cast<std::size_t>(matrix.cols());
  std::size_t const rank = std::min(rows, cols);

  RealCompactQlFactorization<Scalar> result;
  result.reflectors = std::move(matrix);
  result.tau.resize(rank);
  result.rank = rank;
  if (rank == 0)
  {
    return result;
  }

  blas_int const m = detail::checked_blas_int(rows);
  blas_int const n = detail::checked_blas_int(cols);
  blas_int const lda = std::max<blas_int>(1, m);

  Scalar geqlf_work_query{};
  blas_int const query_lwork = -1;
  uni20::lapack::geqlf(m, n, result.reflectors.data(), lda, result.tau.data(), &geqlf_work_query, query_lwork);

  blas_int const geqlf_lwork = std::max<blas_int>(1, static_cast<blas_int>(geqlf_work_query));
  std::vector<Scalar> geqlf_work(static_cast<std::size_t>(geqlf_lwork), Scalar{});
  uni20::lapack::geqlf(m, n, result.reflectors.data(), lda, result.tau.data(), geqlf_work.data(), geqlf_lwork);

  return result;
}

/// \brief Apply the orthogonal factor from a compact real QL factorization.
/// \details Applies `Q`, or `Q^T`, from a Householder QL factorization to a
///          dense target matrix using LAPACK `ormql`.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param factorization Compact QL factorization returned by
///        `real_compact_ql_factorization`.
/// \param target Matrix overwritten by `op(Q) * target` or `target * op(Q)`.
/// \param side Whether `op(Q)` multiplies \p target from the left or right.
/// \param transpose Matrix operation applied to `Q`.
/// \return Matrix after applying the encoded orthogonal factor.
template <uni20::LapackReal Scalar>
detail::ColumnMajorLapackMatrix<Scalar> apply_real_ql_factor(RealCompactQlFactorization<Scalar> const& factorization,
                                                             detail::ColumnMajorLapackMatrix<Scalar> target,
                                                             MatrixSide side = MatrixSide::Left,
                                                             MatrixTranspose transpose = MatrixTranspose::None)
{
  std::size_t const reflector_rows = static_cast<std::size_t>(factorization.reflectors.rows());
  std::size_t const reflector_cols = static_cast<std::size_t>(factorization.reflectors.cols());
  if (factorization.rank > std::min(reflector_rows, reflector_cols))
  {
    throw std::invalid_argument("apply_real_ql_factor received inconsistent QL reflector dimensions");
  }
  if (factorization.tau.size() < factorization.rank)
  {
    throw std::invalid_argument("apply_real_ql_factor received too few QL scalar factors");
  }

  if (side == MatrixSide::Left && !std::cmp_equal(target.rows(), reflector_rows))
  {
    throw std::invalid_argument("apply_real_ql_factor left application requires matching target row count");
  }
  if (side == MatrixSide::Right && !std::cmp_equal(target.cols(), reflector_rows))
  {
    throw std::invalid_argument("apply_real_ql_factor right application requires matching target column count");
  }
  if (target.rows() == 0 || target.cols() == 0 || factorization.rank == 0)
  {
    return target;
  }

  detail::ColumnMajorLapackMatrix<Scalar> reflectors = factorization.reflectors;
  std::vector<Scalar> tau = factorization.tau;
  blas_int const m = detail::checked_blas_int(target.rows());
  blas_int const n = detail::checked_blas_int(target.cols());
  blas_int const k = detail::checked_blas_int(factorization.rank);
  blas_int const lda = std::max<blas_int>(1, detail::checked_blas_int(reflector_rows));
  blas_int const ldc = std::max<blas_int>(1, m);
  char const side_char = detail::lapack_side(side);
  char const trans = detail::lapack_transpose(transpose);

  Scalar work_query{};
  blas_int const query_lwork = -1;
  uni20::lapack::ormql(side_char, trans, m, n, k, reflectors.data(), lda, tau.data(), target.data(), ldc, &work_query,
                       query_lwork);

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
  std::vector<Scalar> work(static_cast<std::size_t>(lwork), Scalar{});
  uni20::lapack::ormql(side_char, trans, m, n, k, reflectors.data(), lda, tau.data(), target.data(), ldc, work.data(),
                       lwork);

  return target;
}

/// \brief Compute a reduced dense real QL factorization through LAPACK `geqlf` and `orgql`.
/// \details Returns `A = Q * L`, where `Q` has orthonormal columns and
///          `min(rows, cols)` columns. `L` has `min(rows, cols)` rows.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real matrix to factor.
/// \return Reduced QL factors.
template <uni20::LapackReal Scalar>
RealQlFactorization<Scalar> real_ql_factorization(detail::ColumnMajorLapackMatrix<Scalar> matrix)
{
  auto compact = real_compact_ql_factorization(std::move(matrix));
  std::size_t const rows = static_cast<std::size_t>(compact.reflectors.rows());
  std::size_t const cols = static_cast<std::size_t>(compact.reflectors.cols());
  std::size_t const rank = compact.rank;

  RealQlFactorization<Scalar> result;
  result.q = detail::ColumnMajorLapackMatrix<Scalar>(rows, rank);
  result.l = detail::ColumnMajorLapackMatrix<Scalar>(rank, cols);
  if (rank == 0)
  {
    return result;
  }
  laset(result.l, Scalar{}, Scalar{}, MatrixFill::All);

  blas_int const m = detail::checked_blas_int(rows);
  blas_int const k = detail::checked_blas_int(rank);
  blas_int const lda = std::max<blas_int>(1, m);

  std::size_t const first_l_row = rows - rank;
  std::size_t const diagonal_offset = cols - rank;
  for (std::size_t row = 0; row < rank; ++row)
  {
    std::size_t const max_col = diagonal_offset + row;
    for (std::size_t col = 0; col < cols && col <= max_col; ++col)
    {
      result.l[row, col] = compact.reflectors[first_l_row + row, col];
    }
  }

  std::size_t const first_q_col = cols - rank;
  for (std::size_t col = 0; col < rank; ++col)
  {
    for (std::size_t row = 0; row < rows; ++row)
    {
      result.q[row, col] = compact.reflectors[row, first_q_col + col];
    }
  }

  Scalar orgql_work_query{};
  blas_int const query_lwork = -1;
  uni20::lapack::orgql(m, k, k, result.q.data(), lda, compact.tau.data(), &orgql_work_query, query_lwork);

  blas_int const orgql_lwork = std::max<blas_int>(1, static_cast<blas_int>(orgql_work_query));
  std::vector<Scalar> orgql_work(static_cast<std::size_t>(orgql_lwork), Scalar{});
  uni20::lapack::orgql(m, k, k, result.q.data(), lda, compact.tau.data(), orgql_work.data(), orgql_lwork);

  return result;
}

/// \brief Compute a compact dense real RQ factorization through LAPACK `gerqf`.
/// \details Stores Householder reflectors and scalar factors in LAPACK's
///          compact RQ representation. Use `apply_real_rq_factor` to apply the
///          encoded orthogonal factor without explicitly materializing `Q`.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real matrix to factor.
/// \return Compact RQ reflector data and scalar factors.
template <uni20::LapackReal Scalar>
RealCompactRqFactorization<Scalar> real_compact_rq_factorization(detail::ColumnMajorLapackMatrix<Scalar> matrix)
{
  std::size_t const rows = static_cast<std::size_t>(matrix.rows());
  std::size_t const cols = static_cast<std::size_t>(matrix.cols());
  std::size_t const rank = std::min(rows, cols);

  RealCompactRqFactorization<Scalar> result;
  result.reflectors = std::move(matrix);
  result.tau.resize(rank);
  result.rank = rank;
  if (rank == 0)
  {
    return result;
  }

  blas_int const m = detail::checked_blas_int(rows);
  blas_int const n = detail::checked_blas_int(cols);
  blas_int const lda = std::max<blas_int>(1, m);

  Scalar gerqf_work_query{};
  blas_int const query_lwork = -1;
  uni20::lapack::gerqf(m, n, result.reflectors.data(), lda, result.tau.data(), &gerqf_work_query, query_lwork);

  blas_int const gerqf_lwork = std::max<blas_int>(1, static_cast<blas_int>(gerqf_work_query));
  std::vector<Scalar> gerqf_work(static_cast<std::size_t>(gerqf_lwork), Scalar{});
  uni20::lapack::gerqf(m, n, result.reflectors.data(), lda, result.tau.data(), gerqf_work.data(), gerqf_lwork);

  return result;
}

/// \brief Apply the orthogonal factor from a compact real RQ factorization.
/// \details Applies `Q`, or `Q^T`, from a Householder RQ factorization to a
///          dense target matrix using LAPACK `ormrq`.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param factorization Compact RQ factorization returned by
///        `real_compact_rq_factorization`.
/// \param target Matrix overwritten by `op(Q) * target` or `target * op(Q)`.
/// \param side Whether `op(Q)` multiplies \p target from the left or right.
/// \param transpose Matrix operation applied to `Q`.
/// \return Matrix after applying the encoded orthogonal factor.
template <uni20::LapackReal Scalar>
detail::ColumnMajorLapackMatrix<Scalar> apply_real_rq_factor(RealCompactRqFactorization<Scalar> const& factorization,
                                                             detail::ColumnMajorLapackMatrix<Scalar> target,
                                                             MatrixSide side = MatrixSide::Right,
                                                             MatrixTranspose transpose = MatrixTranspose::None)
{
  std::size_t const reflector_rows = static_cast<std::size_t>(factorization.reflectors.rows());
  std::size_t const reflector_cols = static_cast<std::size_t>(factorization.reflectors.cols());
  if (factorization.rank > std::min(reflector_rows, reflector_cols))
  {
    throw std::invalid_argument("apply_real_rq_factor received inconsistent RQ reflector dimensions");
  }
  if (factorization.tau.size() < factorization.rank)
  {
    throw std::invalid_argument("apply_real_rq_factor received too few RQ scalar factors");
  }

  if (side == MatrixSide::Left && !std::cmp_equal(target.rows(), reflector_cols))
  {
    throw std::invalid_argument("apply_real_rq_factor left application requires matching target row count");
  }
  if (side == MatrixSide::Right && !std::cmp_equal(target.cols(), reflector_cols))
  {
    throw std::invalid_argument("apply_real_rq_factor right application requires matching target column count");
  }
  if (target.rows() == 0 || target.cols() == 0 || factorization.rank == 0)
  {
    return target;
  }

  detail::ColumnMajorLapackMatrix<Scalar> reflectors = factorization.reflectors;
  std::vector<Scalar> tau = factorization.tau;
  blas_int const m = detail::checked_blas_int(target.rows());
  blas_int const n = detail::checked_blas_int(target.cols());
  blas_int const k = detail::checked_blas_int(factorization.rank);
  blas_int const lda = std::max<blas_int>(1, detail::checked_blas_int(reflector_rows));
  blas_int const ldc = std::max<blas_int>(1, m);
  char const side_char = detail::lapack_side(side);
  char const trans = detail::lapack_transpose(transpose);

  Scalar work_query{};
  blas_int const query_lwork = -1;
  uni20::lapack::ormrq(side_char, trans, m, n, k, reflectors.data(), lda, tau.data(), target.data(), ldc, &work_query,
                       query_lwork);

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
  std::vector<Scalar> work(static_cast<std::size_t>(lwork), Scalar{});
  uni20::lapack::ormrq(side_char, trans, m, n, k, reflectors.data(), lda, tau.data(), target.data(), ldc, work.data(),
                       lwork);

  return target;
}

/// \brief Compute a reduced dense real RQ factorization through LAPACK `gerqf` and `orgrq`.
/// \details Returns `A = R * Q`, where `Q` has orthonormal rows and
///          `min(rows, cols)` rows. `R` has `min(rows, cols)` columns.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real matrix to factor.
/// \return Reduced RQ factors.
template <uni20::LapackReal Scalar>
RealRqFactorization<Scalar> real_rq_factorization(detail::ColumnMajorLapackMatrix<Scalar> matrix)
{
  auto compact = real_compact_rq_factorization(std::move(matrix));
  std::size_t const rows = static_cast<std::size_t>(compact.reflectors.rows());
  std::size_t const cols = static_cast<std::size_t>(compact.reflectors.cols());
  std::size_t const rank = compact.rank;

  RealRqFactorization<Scalar> result;
  result.r = detail::ColumnMajorLapackMatrix<Scalar>(rows, rank);
  result.q = detail::ColumnMajorLapackMatrix<Scalar>(rank, cols);
  if (rank == 0)
  {
    return result;
  }
  laset(result.r, Scalar{}, Scalar{}, MatrixFill::All);

  blas_int const n = detail::checked_blas_int(cols);
  blas_int const k = detail::checked_blas_int(rank);
  blas_int const lda = std::max<blas_int>(1, k);

  std::size_t const first_r_col = cols - rank;
  std::size_t const diagonal_offset = rows - rank;
  for (std::size_t row = 0; row < rows; ++row)
  {
    for (std::size_t col = 0; col < rank; ++col)
    {
      if (row <= diagonal_offset + col)
      {
        result.r[row, col] = compact.reflectors[row, first_r_col + col];
      }
    }
  }

  std::size_t const first_q_row = rows - rank;
  for (std::size_t row = 0; row < rank; ++row)
  {
    for (std::size_t col = 0; col < cols; ++col)
    {
      result.q[row, col] = compact.reflectors[first_q_row + row, col];
    }
  }

  Scalar orgrq_work_query{};
  blas_int const query_lwork = -1;
  uni20::lapack::orgrq(k, n, k, result.q.data(), lda, compact.tau.data(), &orgrq_work_query, query_lwork);

  blas_int const orgrq_lwork = std::max<blas_int>(1, static_cast<blas_int>(orgrq_work_query));
  std::vector<Scalar> orgrq_work(static_cast<std::size_t>(orgrq_lwork), Scalar{});
  uni20::lapack::orgrq(k, n, k, result.q.data(), lda, compact.tau.data(), orgrq_work.data(), orgrq_lwork);

  return result;
}

/// \brief Reduce a dense real matrix to bidiagonal form through LAPACK `gebrd`.
/// \details Returns the clean bidiagonal matrix plus LAPACK's compact
///          Householder storage. For an `m`-by-`n` input, the decomposition is
///          `A = Q * B * P^T`, where `B` is upper bidiagonal if `m >= n` and
///          lower bidiagonal otherwise.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real matrix to reduce.
/// \return Bidiagonal form plus compact Householder reflectors.
template <uni20::LapackReal Scalar>
RealBidiagonalReduction<Scalar> real_bidiagonal_reduction(detail::ColumnMajorLapackMatrix<Scalar> matrix)
{
  std::size_t const rows = static_cast<std::size_t>(matrix.rows());
  std::size_t const cols = static_cast<std::size_t>(matrix.cols());
  std::size_t const rank = std::min(rows, cols);

  RealBidiagonalReduction<Scalar> result;
  result.bidiagonal = detail::ColumnMajorLapackMatrix<Scalar>(rows, cols);
  result.reflectors = std::move(matrix);
  result.diagonal.resize(rank);
  result.offdiagonal.resize(rank > 0 ? rank - 1 : 0);
  result.tauq.resize(rank);
  result.taup.resize(rank);
  result.upper = rows >= cols;
  if (rank == 0)
  {
    return result;
  }
  laset(result.bidiagonal, Scalar{}, Scalar{}, MatrixFill::All);

  blas_int const m = detail::checked_blas_int(rows);
  blas_int const n = detail::checked_blas_int(cols);
  blas_int const lda = std::max<blas_int>(1, m);

  Scalar work_query{};
  blas_int const query_lwork = -1;
  uni20::lapack::gebrd(m, n, result.reflectors.data(), lda, result.diagonal.data(), result.offdiagonal.data(),
                       result.tauq.data(), result.taup.data(), &work_query, query_lwork);

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
  std::vector<Scalar> work(static_cast<std::size_t>(lwork), Scalar{});
  uni20::lapack::gebrd(m, n, result.reflectors.data(), lda, result.diagonal.data(), result.offdiagonal.data(),
                       result.tauq.data(), result.taup.data(), work.data(), lwork);

  for (std::size_t index = 0; index < rank; ++index)
  {
    result.bidiagonal[index, index] = result.diagonal[index];
  }
  for (std::size_t index = 0; index + 1 < rank; ++index)
  {
    if (result.upper)
    {
      result.bidiagonal[index, index + 1] = result.offdiagonal[index];
    }
    else
    {
      result.bidiagonal[index + 1, index] = result.offdiagonal[index];
    }
  }

  return result;
}

/// \brief Compute singular values of a real bidiagonal matrix through LAPACK `bdsqr`.
/// \details The singular values are returned in decreasing order. The
///          offdiagonal has length `diagonal.size() - 1` unless the diagonal
///          is empty.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param diagonal Main diagonal of the bidiagonal matrix.
/// \param offdiagonal Superdiagonal entries if \p upper is true, otherwise
///        subdiagonal entries.
/// \param upper Whether the bidiagonal matrix is upper or lower bidiagonal.
/// \return Singular values in decreasing order.
template <uni20::LapackReal Scalar>
std::vector<Scalar> real_bidiagonal_singular_values(std::vector<Scalar> diagonal, std::vector<Scalar> offdiagonal,
                                                    bool upper = true)
{
  std::size_t const order = diagonal.size();
  if (offdiagonal.size() != (order > 0 ? order - 1 : 0))
  {
    throw std::invalid_argument("real_bidiagonal_singular_values received inconsistent bidiagonal dimensions");
  }
  if (order == 0)
  {
    return diagonal;
  }

  blas_int const n = detail::checked_blas_int(order);
  blas_int const zero = 0;
  blas_int const dummy_ld = 1;
  Scalar dummy{};
  Scalar* e = offdiagonal.empty() ? &dummy : offdiagonal.data();
  std::vector<Scalar> work(static_cast<std::size_t>(std::max<blas_int>(1, 4 * n)), Scalar{});
  uni20::lapack::bdsqr(upper ? 'U' : 'L', n, zero, zero, zero, diagonal.data(), e, &dummy, dummy_ld, &dummy, dummy_ld,
                       &dummy, dummy_ld, work.data());
  return diagonal;
}

/// \brief Compute singular values from a compact real bidiagonal reduction.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param reduction Compact bidiagonal reduction returned by
///        `real_bidiagonal_reduction`.
/// \return Singular values in decreasing order.
template <uni20::LapackReal Scalar>
std::vector<Scalar> real_bidiagonal_singular_values(RealBidiagonalReduction<Scalar> const& reduction)
{
  return real_bidiagonal_singular_values(reduction.diagonal, reduction.offdiagonal, reduction.upper);
}

/// \brief Compute the SVD of a real bidiagonal matrix through LAPACK `bdsdc`.
///
/// \details The singular values are returned in decreasing order. If
///          `compute_vectors` is true, `u` and `vt` are `n`-by-`n`
///          column-major matrices satisfying `B = u * diag(s) * vt`, where
///          `B` is the input upper or lower bidiagonal matrix.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param diagonal Main diagonal of the bidiagonal matrix.
/// \param offdiagonal Superdiagonal entries if \p upper is true, otherwise
///        subdiagonal entries.
/// \param compute_vectors Whether to compute explicit singular vectors.
/// \param upper Whether the bidiagonal matrix is upper or lower bidiagonal.
/// \return Singular values and, optionally, explicit singular vectors.
template <uni20::LapackReal Scalar>
RealBidiagonalSvd<Scalar> real_bidiagonal_svd(std::vector<Scalar> diagonal, std::vector<Scalar> offdiagonal,
                                              bool compute_vectors, bool upper = true)
{
  std::size_t const order = diagonal.size();
  if (offdiagonal.size() != (order > 0 ? order - 1 : 0))
  {
    throw std::invalid_argument("real_bidiagonal_svd received inconsistent bidiagonal dimensions");
  }

  RealBidiagonalSvd<Scalar> result;
  result.singular_values = std::move(diagonal);
  if (order == 0)
  {
    result.u = detail::ColumnMajorLapackMatrix<Scalar>(0, 0);
    result.vt = detail::ColumnMajorLapackMatrix<Scalar>(0, 0);
    return result;
  }

  blas_int const n = detail::checked_blas_int(order);
  char const compq = compute_vectors ? 'I' : 'N';
  detail::ColumnMajorLapackMatrix<Scalar> u(compute_vectors ? order : 1, compute_vectors ? order : 1);
  detail::ColumnMajorLapackMatrix<Scalar> vt(compute_vectors ? order : 1, compute_vectors ? order : 1);
  blas_int const leading_dimension = compute_vectors ? n : 1;
  Scalar dummy_subdiagonal{};
  Scalar* e = offdiagonal.empty() ? &dummy_subdiagonal : offdiagonal.data();
  std::vector<Scalar> q(1, Scalar{});
  std::vector<blas_int> iq(1, 0);
  std::size_t const work_size = compute_vectors ? 3 * order * order + 4 * order : 4 * order;
  std::vector<Scalar> work(std::max<std::size_t>(1, work_size), Scalar{});
  std::vector<blas_int> iwork(std::max<std::size_t>(1, 8 * order), 0);
  uni20::lapack::bdsdc(upper ? 'U' : 'L', compq, n, result.singular_values.data(), e, u.data(), leading_dimension,
                       vt.data(), leading_dimension, q.data(), iq.data(), work.data(), iwork.data());

  result.u = compute_vectors ? std::move(u) : detail::ColumnMajorLapackMatrix<Scalar>(0, 0);
  result.vt = compute_vectors ? std::move(vt) : detail::ColumnMajorLapackMatrix<Scalar>(0, 0);
  return result;
}

/// \brief Compute the SVD from a compact real bidiagonal reduction.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param reduction Compact bidiagonal reduction returned by
///        `real_bidiagonal_reduction`.
/// \param compute_vectors Whether to compute explicit singular vectors.
/// \return Bidiagonal singular values and, optionally, explicit singular vectors.
template <uni20::LapackReal Scalar>
RealBidiagonalSvd<Scalar> real_bidiagonal_svd(RealBidiagonalReduction<Scalar> const& reduction, bool compute_vectors)
{
  return real_bidiagonal_svd(reduction.diagonal, reduction.offdiagonal, compute_vectors, reduction.upper);
}

/// \brief Compute a selected-index SVD of a real bidiagonal matrix through LAPACK `bdsvdx`.
///
/// \details Selects singular triplets by 0-based inclusive index in decreasing
///          singular-value order. If `compute_vectors` is true, `u` is
///          `n`-by-`m` and `vt` is `m`-by-`n`, where `m` is the number of
///          selected singular values. The selected contribution is
///          `u * diag(s) * vt`.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param diagonal Main diagonal of the bidiagonal matrix.
/// \param offdiagonal Superdiagonal entries if \p upper is true, otherwise
///        subdiagonal entries.
/// \param first_index First 0-based decreasing singular-value index to compute.
/// \param last_index Last 0-based decreasing singular-value index to compute.
/// \param compute_vectors Whether to compute explicit singular vectors.
/// \param upper Whether the bidiagonal matrix is upper or lower bidiagonal.
/// \return Selected singular values and, optionally, explicit singular vectors.
template <uni20::LapackReal Scalar>
RealBidiagonalSvd<Scalar> real_bidiagonal_svd_index_range(std::vector<Scalar> diagonal, std::vector<Scalar> offdiagonal,
                                                          std::size_t first_index, std::size_t last_index,
                                                          bool compute_vectors, bool upper = true)
{
  std::size_t const order = diagonal.size();
  if (offdiagonal.size() != (order > 0 ? order - 1 : 0))
  {
    throw std::invalid_argument("real_bidiagonal_svd_index_range received inconsistent bidiagonal dimensions");
  }
  if (order == 0 || first_index > last_index || last_index >= order)
  {
    throw std::invalid_argument("real_bidiagonal_svd_index_range received an invalid index range");
  }

  std::size_t const selected = last_index - first_index + 1;
  RealBidiagonalSvd<Scalar> result;

  blas_int const n = detail::checked_blas_int(order);
  blas_int const first = detail::checked_blas_int(first_index + 1);
  blas_int const last = detail::checked_blas_int(last_index + 1);
  char const jobz = compute_vectors ? 'V' : 'N';
  char const range = 'I';
  Scalar dummy_subdiagonal{};
  Scalar* e = offdiagonal.empty() ? &dummy_subdiagonal : offdiagonal.data();
  // bdsvdx uses S(1:N) and up to N vector columns internally before discarding
  // unselected values.
  std::vector<Scalar> singular_values_workspace(order, Scalar{});
  detail::ColumnMajorLapackMatrix<Scalar> z(compute_vectors ? 2 * order : 1, compute_vectors ? order : 1);
  blas_int const ldz = compute_vectors ? detail::checked_blas_int(2 * order) : 1;
  std::vector<Scalar> work(std::max<std::size_t>(1, 14 * order), Scalar{});
  std::vector<blas_int> iwork(std::max<std::size_t>(1, 12 * order), 0);
  blas_int selected_count = 0;
  uni20::lapack::bdsvdx(upper ? 'U' : 'L', jobz, range, n, diagonal.data(), e, Scalar{}, Scalar{}, first, last,
                        selected_count, singular_values_workspace.data(), z.data(), ldz, work.data(), iwork.data());

  if (selected_count < 0 || static_cast<std::size_t>(selected_count) != selected)
  {
    throw std::runtime_error("LAPACK bdsvdx returned an unexpected number of singular values");
  }

  result.singular_values.assign(singular_values_workspace.begin(), singular_values_workspace.begin() + selected_count);

  if (!compute_vectors)
  {
    result.u = detail::ColumnMajorLapackMatrix<Scalar>(0, 0);
    result.vt = detail::ColumnMajorLapackMatrix<Scalar>(0, 0);
    return result;
  }

  result.u = detail::ColumnMajorLapackMatrix<Scalar>(order, selected);
  result.vt = detail::ColumnMajorLapackMatrix<Scalar>(selected, order);
  for (std::size_t col = 0; col < selected; ++col)
  {
    for (std::size_t row = 0; row < order; ++row)
    {
      result.u[row, col] = z[row, col];
      result.vt[col, row] = z[order + row, col];
    }
  }
  return result;
}

/// \brief Compute a selected-index SVD from a compact real bidiagonal reduction.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param reduction Compact bidiagonal reduction returned by
///        `real_bidiagonal_reduction`.
/// \param first_index First 0-based decreasing singular-value index to compute.
/// \param last_index Last 0-based decreasing singular-value index to compute.
/// \param compute_vectors Whether to compute explicit singular vectors.
/// \return Selected singular values and, optionally, explicit singular vectors.
template <uni20::LapackReal Scalar>
RealBidiagonalSvd<Scalar> real_bidiagonal_svd_index_range(RealBidiagonalReduction<Scalar> const& reduction,
                                                          std::size_t first_index, std::size_t last_index,
                                                          bool compute_vectors)
{
  return real_bidiagonal_svd_index_range(reduction.diagonal, reduction.offdiagonal, first_index, last_index,
                                         compute_vectors, reduction.upper);
}

/// \brief Materialize the left orthogonal factor `Q` from a real bidiagonal reduction.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param reduction Compact bidiagonal reduction returned by
///        `real_bidiagonal_reduction`.
/// \return Full `Q` factor satisfying `A = Q * B * P^T`.
template <uni20::LapackReal Scalar>
detail::ColumnMajorLapackMatrix<Scalar>
real_bidiagonal_left_orthogonal_factor(RealBidiagonalReduction<Scalar> const& reduction)
{
  std::size_t const rows = static_cast<std::size_t>(reduction.reflectors.rows());
  std::size_t const cols = static_cast<std::size_t>(reduction.reflectors.cols());
  detail::ColumnMajorLapackMatrix<Scalar> factor(rows, rows);
  laset(factor, Scalar{1}, Scalar{}, MatrixFill::All);
  if (rows == 0 || cols == 0)
  {
    return factor;
  }
  if (reduction.tauq.size() < std::min(rows, cols))
  {
    throw std::invalid_argument("real_bidiagonal_left_orthogonal_factor received too few Q scalar factors");
  }

  for (std::size_t col = 0; col < std::min(cols, rows); ++col)
  {
    for (std::size_t row = 0; row < rows; ++row)
    {
      factor[row, col] = reduction.reflectors[row, col];
    }
  }

  blas_int const m = detail::checked_blas_int(rows);
  blas_int const n = detail::checked_blas_int(rows);
  blas_int const k = detail::checked_blas_int(cols);
  blas_int const lda = std::max<blas_int>(1, m);

  Scalar work_query{};
  blas_int const query_lwork = -1;
  uni20::lapack::orgbr('Q', m, n, k, factor.data(), lda, const_cast<Scalar*>(reduction.tauq.data()), &work_query,
                       query_lwork);

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
  std::vector<Scalar> work(static_cast<std::size_t>(lwork), Scalar{});
  uni20::lapack::orgbr('Q', m, n, k, factor.data(), lda, const_cast<Scalar*>(reduction.tauq.data()), work.data(),
                       lwork);

  return factor;
}

/// \brief Materialize the right transposed factor `P^T` from a real bidiagonal reduction.
/// \details This follows LAPACK `orgbr('P')`, which generates `P^T`, not `P`.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param reduction Compact bidiagonal reduction returned by
///        `real_bidiagonal_reduction`.
/// \return Full `P^T` factor satisfying `A = Q * B * P^T`.
template <uni20::LapackReal Scalar>
detail::ColumnMajorLapackMatrix<Scalar>
real_bidiagonal_right_orthogonal_factor_transpose(RealBidiagonalReduction<Scalar> const& reduction)
{
  std::size_t const rows = static_cast<std::size_t>(reduction.reflectors.rows());
  std::size_t const cols = static_cast<std::size_t>(reduction.reflectors.cols());
  detail::ColumnMajorLapackMatrix<Scalar> factor(cols, cols);
  laset(factor, Scalar{1}, Scalar{}, MatrixFill::All);
  if (rows == 0 || cols == 0)
  {
    return factor;
  }
  if (reduction.taup.size() < std::min(rows, cols))
  {
    throw std::invalid_argument("real_bidiagonal_right_orthogonal_factor_transpose received too few P scalar factors");
  }

  for (std::size_t col = 0; col < cols; ++col)
  {
    for (std::size_t row = 0; row < std::min(rows, cols); ++row)
    {
      factor[row, col] = reduction.reflectors[row, col];
    }
  }

  blas_int const m = detail::checked_blas_int(cols);
  blas_int const n = detail::checked_blas_int(cols);
  blas_int const k = detail::checked_blas_int(rows);
  blas_int const lda = std::max<blas_int>(1, m);

  Scalar work_query{};
  blas_int const query_lwork = -1;
  uni20::lapack::orgbr('P', m, n, k, factor.data(), lda, const_cast<Scalar*>(reduction.taup.data()), &work_query,
                       query_lwork);

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
  std::vector<Scalar> work(static_cast<std::size_t>(lwork), Scalar{});
  uni20::lapack::orgbr('P', m, n, k, factor.data(), lda, const_cast<Scalar*>(reduction.taup.data()), work.data(),
                       lwork);

  return factor;
}

/// \brief Apply the left orthogonal factor `Q` from a real bidiagonal reduction.
/// \details Applies `Q` or `Q^T` using LAPACK `ormbr`.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param reduction Compact bidiagonal reduction returned by
///        `real_bidiagonal_reduction`.
/// \param target Matrix overwritten by `op(Q) * target` or `target * op(Q)`.
/// \param side Whether `op(Q)` multiplies \p target from the left or right.
/// \param transpose Matrix operation applied to `Q`.
/// \return Matrix after applying the encoded orthogonal factor.
template <uni20::LapackReal Scalar>
detail::ColumnMajorLapackMatrix<Scalar>
apply_real_bidiagonal_left_factor(RealBidiagonalReduction<Scalar> const& reduction,
                                  detail::ColumnMajorLapackMatrix<Scalar> target, MatrixSide side = MatrixSide::Left,
                                  MatrixTranspose transpose = MatrixTranspose::None)
{
  std::size_t const rows = static_cast<std::size_t>(reduction.reflectors.rows());
  std::size_t const cols = static_cast<std::size_t>(reduction.reflectors.cols());
  std::size_t const rank = std::min(rows, cols);
  if (reduction.tauq.size() < rank)
  {
    throw std::invalid_argument("apply_real_bidiagonal_left_factor received too few Q scalar factors");
  }
  if (side == MatrixSide::Left && !std::cmp_equal(target.rows(), rows))
  {
    throw std::invalid_argument(
        "apply_real_bidiagonal_left_factor left application requires matching target row count");
  }
  if (side == MatrixSide::Right && !std::cmp_equal(target.cols(), rows))
  {
    throw std::invalid_argument(
        "apply_real_bidiagonal_left_factor right application requires matching target column count");
  }
  if (target.rows() == 0 || target.cols() == 0 || cols == 0)
  {
    return target;
  }

  detail::ColumnMajorLapackMatrix<Scalar> reflectors = reduction.reflectors;
  std::vector<Scalar> tau = reduction.tauq;
  blas_int const m = detail::checked_blas_int(target.rows());
  blas_int const n = detail::checked_blas_int(target.cols());
  blas_int const k = detail::checked_blas_int(cols);
  blas_int const lda = std::max<blas_int>(1, detail::checked_blas_int(rows));
  blas_int const ldc = std::max<blas_int>(1, m);
  char const side_char = detail::lapack_side(side);
  char const trans = detail::lapack_transpose(transpose);

  Scalar work_query{};
  blas_int const query_lwork = -1;
  uni20::lapack::ormbr('Q', side_char, trans, m, n, k, reflectors.data(), lda, tau.data(), target.data(), ldc,
                       &work_query, query_lwork);

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
  std::vector<Scalar> work(static_cast<std::size_t>(lwork), Scalar{});
  uni20::lapack::ormbr('Q', side_char, trans, m, n, k, reflectors.data(), lda, tau.data(), target.data(), ldc,
                       work.data(), lwork);

  return target;
}

/// \brief Apply the right orthogonal factor `P` from a real bidiagonal reduction.
/// \details Applies `P` or `P^T` using LAPACK `ormbr`.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param reduction Compact bidiagonal reduction returned by
///        `real_bidiagonal_reduction`.
/// \param target Matrix overwritten by `op(P) * target` or `target * op(P)`.
/// \param side Whether `op(P)` multiplies \p target from the left or right.
/// \param transpose Matrix operation applied to `P`.
/// \return Matrix after applying the encoded orthogonal factor.
template <uni20::LapackReal Scalar>
detail::ColumnMajorLapackMatrix<Scalar>
apply_real_bidiagonal_right_factor(RealBidiagonalReduction<Scalar> const& reduction,
                                   detail::ColumnMajorLapackMatrix<Scalar> target, MatrixSide side = MatrixSide::Right,
                                   MatrixTranspose transpose = MatrixTranspose::None)
{
  std::size_t const rows = static_cast<std::size_t>(reduction.reflectors.rows());
  std::size_t const cols = static_cast<std::size_t>(reduction.reflectors.cols());
  std::size_t const rank = std::min(rows, cols);
  if (reduction.taup.size() < rank)
  {
    throw std::invalid_argument("apply_real_bidiagonal_right_factor received too few P scalar factors");
  }
  if (side == MatrixSide::Left && !std::cmp_equal(target.rows(), cols))
  {
    throw std::invalid_argument(
        "apply_real_bidiagonal_right_factor left application requires matching target row count");
  }
  if (side == MatrixSide::Right && !std::cmp_equal(target.cols(), cols))
  {
    throw std::invalid_argument(
        "apply_real_bidiagonal_right_factor right application requires matching target column count");
  }
  if (target.rows() == 0 || target.cols() == 0 || rows == 0)
  {
    return target;
  }

  detail::ColumnMajorLapackMatrix<Scalar> reflectors = reduction.reflectors;
  std::vector<Scalar> tau = reduction.taup;
  blas_int const m = detail::checked_blas_int(target.rows());
  blas_int const n = detail::checked_blas_int(target.cols());
  blas_int const k = detail::checked_blas_int(rows);
  blas_int const lda = std::max<blas_int>(1, detail::checked_blas_int(rows));
  blas_int const ldc = std::max<blas_int>(1, m);
  char const side_char = detail::lapack_side(side);
  char const trans = detail::lapack_transpose(transpose);

  Scalar work_query{};
  blas_int const query_lwork = -1;
  uni20::lapack::ormbr('P', side_char, trans, m, n, k, reflectors.data(), lda, tau.data(), target.data(), ldc,
                       &work_query, query_lwork);

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
  std::vector<Scalar> work(static_cast<std::size_t>(lwork), Scalar{});
  uni20::lapack::ormbr('P', side_char, trans, m, n, k, reflectors.data(), lda, tau.data(), target.data(), ldc,
                       work.data(), lwork);

  return target;
}

/// \brief Reduce a dense real square matrix to upper Hessenberg form through LAPACK `gehrd`.
/// \details Returns both a clean upper Hessenberg matrix and the raw compact
///          Householder storage needed by `real_hessenberg_orthogonal_factor`.
///          The reduction satisfies `A = Q * H * Q^T` when `Q` is generated
///          from the returned compact data.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real square matrix to reduce.
/// \return Upper Hessenberg form plus compact Householder reflectors.
template <uni20::LapackReal Scalar>
RealHessenbergReduction<Scalar> real_hessenberg_reduction(detail::ColumnMajorLapackMatrix<Scalar> matrix)
{
  if (!std::cmp_equal(matrix.rows(), matrix.cols()))
  {
    throw std::invalid_argument("real_hessenberg_reduction requires a square matrix");
  }

  std::size_t const n = static_cast<std::size_t>(matrix.rows());
  RealHessenbergReduction<Scalar> result;
  result.first = 0;
  result.last_exclusive = n;
  result.tau.resize(n > 0 ? n - 1 : 0);
  if (n == 0)
  {
    result.hessenberg = detail::ColumnMajorLapackMatrix<Scalar>(0, 0);
    result.reflectors = detail::ColumnMajorLapackMatrix<Scalar>(0, 0);
    return result;
  }

  if (n > 1)
  {
    blas_int const order = detail::checked_blas_int(n);
    blas_int const first = 1;
    blas_int const last = order;

    Scalar work_query{};
    blas_int const query_lwork = -1;
    uni20::lapack::gehrd(order, first, last, matrix.data(), order, result.tau.data(), &work_query, query_lwork);

    blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
    std::vector<Scalar> work(static_cast<std::size_t>(lwork), Scalar{});
    uni20::lapack::gehrd(order, first, last, matrix.data(), order, result.tau.data(), work.data(), lwork);
  }

  result.reflectors = matrix;
  result.hessenberg = std::move(matrix);
  for (std::size_t col = 0; col < n; ++col)
  {
    for (std::size_t row = col + 2; row < n; ++row)
    {
      result.hessenberg[row, col] = Scalar{};
    }
  }
  return result;
}

/// \brief Generate the orthogonal factor from a compact real Hessenberg reduction.
/// \details The returned matrix is the `Q` in `A = Q * H * Q^T` for the result
///          returned by `real_hessenberg_reduction`.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param reduction Compact Hessenberg reduction returned by `real_hessenberg_reduction`.
/// \return Dense orthogonal matrix represented by the compact Householder data.
template <uni20::LapackReal Scalar>
detail::ColumnMajorLapackMatrix<Scalar>
real_hessenberg_orthogonal_factor(RealHessenbergReduction<Scalar> const& reduction)
{
  if (!std::cmp_equal(reduction.reflectors.rows(), reduction.reflectors.cols()))
  {
    throw std::invalid_argument("real_hessenberg_orthogonal_factor requires square compact reflector storage");
  }

  std::size_t const n = static_cast<std::size_t>(reduction.reflectors.rows());
  if (reduction.tau.size() != (n > 0 ? n - 1 : 0))
  {
    throw std::invalid_argument("real_hessenberg_orthogonal_factor received inconsistent tau data");
  }
  if (reduction.first > reduction.last_exclusive || reduction.last_exclusive > n)
  {
    throw std::invalid_argument("real_hessenberg_orthogonal_factor received invalid active interval");
  }

  detail::ColumnMajorLapackMatrix<Scalar> q(n, n);
  for (std::size_t diagonal = 0; diagonal < n; ++diagonal)
  {
    q[diagonal, diagonal] = Scalar{1};
  }
  if (n <= 1)
  {
    return q;
  }

  q = reduction.reflectors;
  blas_int const order = detail::checked_blas_int(n);
  blas_int const first = detail::checked_blas_int(reduction.first + 1);
  blas_int const last = detail::checked_blas_int(reduction.last_exclusive);

  Scalar work_query{};
  blas_int const query_lwork = -1;
  uni20::lapack::orghr(order, first, last, q.data(), order, reduction.tau.data(), &work_query, query_lwork);

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
  std::vector<Scalar> work(static_cast<std::size_t>(lwork), Scalar{});
  uni20::lapack::orghr(order, first, last, q.data(), order, reduction.tau.data(), work.data(), lwork);
  return q;
}

/// \brief Apply the orthogonal factor from a compact real Hessenberg reduction.
/// \details Applies `Q`, or `Q^T`, from the Householder data returned by
///          `real_hessenberg_reduction` to a dense target matrix using LAPACK
///          `ormhr`.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param reduction Compact Hessenberg reduction returned by `real_hessenberg_reduction`.
/// \param target Matrix overwritten by `op(Q) * target` or `target * op(Q)`.
/// \param side Whether `op(Q)` multiplies \p target from the left or right.
/// \param transpose Matrix operation applied to `Q`.
/// \return Matrix after applying the encoded orthogonal factor.
template <uni20::LapackReal Scalar>
detail::ColumnMajorLapackMatrix<Scalar> apply_real_hessenberg_orthogonal_factor(
    RealHessenbergReduction<Scalar> const& reduction, detail::ColumnMajorLapackMatrix<Scalar> target,
    MatrixSide side = MatrixSide::Left, MatrixTranspose transpose = MatrixTranspose::None)
{
  if (!std::cmp_equal(reduction.reflectors.rows(), reduction.reflectors.cols()))
  {
    throw std::invalid_argument("apply_real_hessenberg_orthogonal_factor requires square compact reflector storage");
  }

  std::size_t const order_size = static_cast<std::size_t>(reduction.reflectors.rows());
  if (reduction.tau.size() != (order_size > 0 ? order_size - 1 : 0))
  {
    throw std::invalid_argument("apply_real_hessenberg_orthogonal_factor received inconsistent tau data");
  }
  if (reduction.first > reduction.last_exclusive || reduction.last_exclusive > order_size)
  {
    throw std::invalid_argument("apply_real_hessenberg_orthogonal_factor received invalid active interval");
  }
  if (side == MatrixSide::Left && !std::cmp_equal(target.rows(), order_size))
  {
    throw std::invalid_argument("apply_real_hessenberg_orthogonal_factor left application requires matching rows");
  }
  if (side == MatrixSide::Right && !std::cmp_equal(target.cols(), order_size))
  {
    throw std::invalid_argument("apply_real_hessenberg_orthogonal_factor right application requires matching columns");
  }
  if (target.rows() == 0 || target.cols() == 0 || order_size <= 1)
  {
    return target;
  }

  detail::ColumnMajorLapackMatrix<Scalar> reflectors = reduction.reflectors;
  blas_int const m = detail::checked_blas_int(target.rows());
  blas_int const n = detail::checked_blas_int(target.cols());
  blas_int const order = detail::checked_blas_int(order_size);
  blas_int const first = detail::checked_blas_int(reduction.first + 1);
  blas_int const last = detail::checked_blas_int(reduction.last_exclusive);
  blas_int const lda = std::max<blas_int>(1, order);
  blas_int const ldc = std::max<blas_int>(1, m);
  char const side_char = detail::lapack_side(side);
  char const trans = detail::lapack_transpose(transpose);

  Scalar work_query{};
  blas_int const query_lwork = -1;
  uni20::lapack::ormhr(side_char, trans, m, n, first, last, reflectors.data(), lda, reduction.tau.data(), target.data(),
                       ldc, &work_query, query_lwork);

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
  std::vector<Scalar> work(static_cast<std::size_t>(lwork), Scalar{});
  uni20::lapack::ormhr(side_char, trans, m, n, first, last, reflectors.data(), lda, reduction.tau.data(), target.data(),
                       ldc, work.data(), lwork);
  return target;
}

/// \brief Reduce a dense real symmetric matrix to tridiagonal form through LAPACK `sytrd`.
/// \details Uses the selected triangle of \p matrix and returns both a clean
///          symmetric tridiagonal matrix and the compact Householder storage
///          needed by `real_symmetric_tridiagonal_orthogonal_factor` and
///          `apply_real_symmetric_tridiagonal_orthogonal_factor`. The reduction
///          satisfies `A = Q * T * Q^T` when `Q` is generated from the returned
///          compact data.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real symmetric matrix to reduce.
/// \param triangle Which triangle of \p matrix contains the symmetric input.
/// \return Tridiagonal form plus compact Householder reflectors.
template <uni20::LapackReal Scalar>
RealSymmetricTridiagonalReduction<Scalar>
real_symmetric_tridiagonal_reduction(detail::ColumnMajorLapackMatrix<Scalar> matrix,
                                     MatrixFill triangle = MatrixFill::Upper)
{
  if (!std::cmp_equal(matrix.rows(), matrix.cols()))
  {
    throw std::invalid_argument("real_symmetric_tridiagonal_reduction requires a square matrix");
  }

  char const uplo = detail::lapack_uplo(triangle);
  std::size_t const n = static_cast<std::size_t>(matrix.rows());
  RealSymmetricTridiagonalReduction<Scalar> result;
  result.triangle = triangle;
  result.diagonal.resize(n);
  result.offdiagonal.resize(n > 0 ? n - 1 : 0);
  result.tau.resize(n > 0 ? n - 1 : 0);
  if (n == 0)
  {
    result.tridiagonal = detail::ColumnMajorLapackMatrix<Scalar>(0, 0);
    result.reflectors = detail::ColumnMajorLapackMatrix<Scalar>(0, 0);
    return result;
  }

  blas_int const order = detail::checked_blas_int(n);
  Scalar work_query{};
  blas_int const query_lwork = -1;
  uni20::lapack::sytrd(uplo, order, matrix.data(), order, result.diagonal.data(), result.offdiagonal.data(),
                       result.tau.data(), &work_query, query_lwork);

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
  std::vector<Scalar> work(static_cast<std::size_t>(lwork), Scalar{});
  uni20::lapack::sytrd(uplo, order, matrix.data(), order, result.diagonal.data(), result.offdiagonal.data(),
                       result.tau.data(), work.data(), lwork);

  result.reflectors = matrix;
  result.tridiagonal = detail::ColumnMajorLapackMatrix<Scalar>(n, n);
  laset(result.tridiagonal, Scalar{}, Scalar{}, MatrixFill::All);
  for (std::size_t index = 0; index < n; ++index)
  {
    result.tridiagonal[index, index] = result.diagonal[index];
  }
  for (std::size_t index = 0; index + 1 < n; ++index)
  {
    result.tridiagonal[index, index + 1] = result.offdiagonal[index];
    result.tridiagonal[index + 1, index] = result.offdiagonal[index];
  }
  return result;
}

/// \brief Generate the orthogonal factor from a compact real symmetric tridiagonal reduction.
/// \details The returned matrix is the `Q` in `A = Q * T * Q^T` for the result
///          returned by `real_symmetric_tridiagonal_reduction`.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param reduction Compact tridiagonal reduction returned by
///        `real_symmetric_tridiagonal_reduction`.
/// \return Dense orthogonal matrix represented by the compact Householder data.
template <uni20::LapackReal Scalar>
detail::ColumnMajorLapackMatrix<Scalar>
real_symmetric_tridiagonal_orthogonal_factor(RealSymmetricTridiagonalReduction<Scalar> const& reduction)
{
  if (!std::cmp_equal(reduction.reflectors.rows(), reduction.reflectors.cols()))
  {
    throw std::invalid_argument(
        "real_symmetric_tridiagonal_orthogonal_factor requires square compact reflector storage");
  }

  std::size_t const n = static_cast<std::size_t>(reduction.reflectors.rows());
  if (reduction.diagonal.size() != n || reduction.offdiagonal.size() != (n > 0 ? n - 1 : 0) ||
      reduction.tau.size() != (n > 0 ? n - 1 : 0))
  {
    throw std::invalid_argument("real_symmetric_tridiagonal_orthogonal_factor received inconsistent reduction data");
  }

  detail::ColumnMajorLapackMatrix<Scalar> q(n, n);
  for (std::size_t diagonal = 0; diagonal < n; ++diagonal)
  {
    q[diagonal, diagonal] = Scalar{1};
  }
  if (n <= 1)
  {
    return q;
  }

  q = reduction.reflectors;
  blas_int const order = detail::checked_blas_int(n);
  char const uplo = detail::lapack_uplo(reduction.triangle);

  Scalar work_query{};
  blas_int const query_lwork = -1;
  uni20::lapack::orgtr(uplo, order, q.data(), order, reduction.tau.data(), &work_query, query_lwork);

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
  std::vector<Scalar> work(static_cast<std::size_t>(lwork), Scalar{});
  uni20::lapack::orgtr(uplo, order, q.data(), order, reduction.tau.data(), work.data(), lwork);
  return q;
}

/// \brief Apply the orthogonal factor from a compact real symmetric tridiagonal reduction.
/// \details Applies `Q`, or `Q^T`, from the Householder data returned by
///          `real_symmetric_tridiagonal_reduction` to a dense target matrix
///          using LAPACK `ormtr`.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param reduction Compact tridiagonal reduction returned by
///        `real_symmetric_tridiagonal_reduction`.
/// \param target Matrix overwritten by `op(Q) * target` or `target * op(Q)`.
/// \param side Whether `op(Q)` multiplies \p target from the left or right.
/// \param transpose Matrix operation applied to `Q`.
/// \return Matrix after applying the encoded orthogonal factor.
template <uni20::LapackReal Scalar>
detail::ColumnMajorLapackMatrix<Scalar> apply_real_symmetric_tridiagonal_orthogonal_factor(
    RealSymmetricTridiagonalReduction<Scalar> const& reduction, detail::ColumnMajorLapackMatrix<Scalar> target,
    MatrixSide side = MatrixSide::Left, MatrixTranspose transpose = MatrixTranspose::None)
{
  if (!std::cmp_equal(reduction.reflectors.rows(), reduction.reflectors.cols()))
  {
    throw std::invalid_argument(
        "apply_real_symmetric_tridiagonal_orthogonal_factor requires square compact reflector storage");
  }

  std::size_t const order_size = static_cast<std::size_t>(reduction.reflectors.rows());
  if (reduction.diagonal.size() != order_size ||
      reduction.offdiagonal.size() != (order_size > 0 ? order_size - 1 : 0) ||
      reduction.tau.size() != (order_size > 0 ? order_size - 1 : 0))
  {
    throw std::invalid_argument(
        "apply_real_symmetric_tridiagonal_orthogonal_factor received inconsistent reduction data");
  }
  if (side == MatrixSide::Left && !std::cmp_equal(target.rows(), order_size))
  {
    throw std::invalid_argument(
        "apply_real_symmetric_tridiagonal_orthogonal_factor left application requires matching rows");
  }
  if (side == MatrixSide::Right && !std::cmp_equal(target.cols(), order_size))
  {
    throw std::invalid_argument(
        "apply_real_symmetric_tridiagonal_orthogonal_factor right application requires matching columns");
  }
  if (target.rows() == 0 || target.cols() == 0 || order_size <= 1)
  {
    return target;
  }

  detail::ColumnMajorLapackMatrix<Scalar> reflectors = reduction.reflectors;
  blas_int const m = detail::checked_blas_int(target.rows());
  blas_int const n = detail::checked_blas_int(target.cols());
  blas_int const order = detail::checked_blas_int(order_size);
  blas_int const lda = std::max<blas_int>(1, order);
  blas_int const ldc = std::max<blas_int>(1, m);
  char const side_char = detail::lapack_side(side);
  char const uplo = detail::lapack_uplo(reduction.triangle);
  char const trans = detail::lapack_transpose(transpose);

  Scalar work_query{};
  blas_int const query_lwork = -1;
  uni20::lapack::ormtr(side_char, uplo, trans, m, n, reflectors.data(), lda, reduction.tau.data(), target.data(), ldc,
                       &work_query, query_lwork);

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
  std::vector<Scalar> work(static_cast<std::size_t>(lwork), Scalar{});
  uni20::lapack::ormtr(side_char, uplo, trans, m, n, reflectors.data(), lda, reduction.tau.data(), target.data(), ldc,
                       work.data(), lwork);
  return target;
}

/// \brief Compute a reduced dense real pivoted QR factorization through LAPACK `geqp3`.
/// \details Returns `A * P = Q * R`, where `Q` has orthonormal columns and
///          `min(rows, cols)` columns. `pivot_columns[j]` is the original
///          0-based column index stored in pivoted column `j` of `A * P`.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real matrix to factor.
/// \return Reduced pivoted QR factors and column pivot order.
template <uni20::LapackReal Scalar>
RealPivotedQrFactorization<Scalar> real_pivoted_qr_factorization(detail::ColumnMajorLapackMatrix<Scalar> matrix)
{
  std::size_t const rows = static_cast<std::size_t>(matrix.rows());
  std::size_t const cols = static_cast<std::size_t>(matrix.cols());
  std::size_t const rank = std::min(rows, cols);

  RealPivotedQrFactorization<Scalar> result;
  result.q = detail::ColumnMajorLapackMatrix<Scalar>(rows, rank);
  result.r = detail::ColumnMajorLapackMatrix<Scalar>(rank, cols);
  result.pivot_columns.resize(cols);
  std::iota(result.pivot_columns.begin(), result.pivot_columns.end(), std::size_t{0});
  if (rank == 0)
  {
    return result;
  }
  laset(result.r, Scalar{}, Scalar{}, MatrixFill::All);

  blas_int const m = detail::checked_blas_int(rows);
  blas_int const n = detail::checked_blas_int(cols);
  blas_int const k = detail::checked_blas_int(rank);
  blas_int const lda = std::max<blas_int>(1, m);
  std::vector<Scalar> tau(rank, Scalar{});
  std::vector<blas_int> jpvt(cols, 0);

  Scalar geqp3_work_query{};
  blas_int const query_lwork = -1;
  uni20::lapack::geqp3(m, n, matrix.data(), lda, jpvt.data(), tau.data(), &geqp3_work_query, query_lwork);

  std::fill(jpvt.begin(), jpvt.end(), 0);
  blas_int const geqp3_lwork = std::max<blas_int>(1, static_cast<blas_int>(geqp3_work_query));
  std::vector<Scalar> geqp3_work(static_cast<std::size_t>(geqp3_lwork), Scalar{});
  uni20::lapack::geqp3(m, n, matrix.data(), lda, jpvt.data(), tau.data(), geqp3_work.data(), geqp3_lwork);

  for (std::size_t col = 0; col < cols; ++col)
  {
    if (jpvt[col] <= 0 || static_cast<std::size_t>(jpvt[col]) > cols)
    {
      throw std::runtime_error("LAPACK geqp3 returned an invalid column pivot");
    }
    result.pivot_columns[col] = static_cast<std::size_t>(jpvt[col] - 1);
  }

  for (std::size_t row = 0; row < rank; ++row)
  {
    for (std::size_t col = row; col < cols; ++col)
    {
      result.r[row, col] = matrix[row, col];
    }
  }

  for (std::size_t col = 0; col < rank; ++col)
  {
    for (std::size_t row = 0; row < rows; ++row)
    {
      result.q[row, col] = matrix[row, col];
    }
  }

  Scalar orgqr_work_query{};
  uni20::lapack::orgqr(m, k, k, result.q.data(), lda, tau.data(), &orgqr_work_query, query_lwork);

  blas_int const orgqr_lwork = std::max<blas_int>(1, static_cast<blas_int>(orgqr_work_query));
  std::vector<Scalar> orgqr_work(static_cast<std::size_t>(orgqr_lwork), Scalar{});
  uni20::lapack::orgqr(m, k, k, result.q.data(), lda, tau.data(), orgqr_work.data(), orgqr_lwork);

  return result;
}

/// \brief Compute a dense real singular value decomposition through LAPACK `gesvd`.
/// \details Singular values are returned in descending order. If
///          `compute_vectors` is true, this computes the full decomposition
///          `A = U * Sigma * VT`, with `U` stored in `left_singular_vectors`
///          and `VT` stored in `right_singular_vectors_transpose`.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real matrix to decompose.
/// \param compute_vectors Whether to compute full left and right singular vectors.
/// \return Singular values and, optionally, full singular-vector matrices.
template <uni20::LapackReal Scalar>
RealSingularValueDecomposition<Scalar> real_singular_value_decomposition(detail::ColumnMajorLapackMatrix<Scalar> matrix,
                                                                         bool compute_vectors)
{
  std::size_t const rows = static_cast<std::size_t>(matrix.rows());
  std::size_t const cols = static_cast<std::size_t>(matrix.cols());
  std::size_t const singular_value_count = std::min(rows, cols);

  RealSingularValueDecomposition<Scalar> result;
  result.singular_values.resize(singular_value_count);
  if (singular_value_count == 0)
  {
    result.left_singular_vectors = compute_vectors ? detail::ColumnMajorLapackMatrix<Scalar>(rows, rows)
                                                   : detail::ColumnMajorLapackMatrix<Scalar>(0, 0);
    result.right_singular_vectors_transpose = compute_vectors ? detail::ColumnMajorLapackMatrix<Scalar>(cols, cols)
                                                              : detail::ColumnMajorLapackMatrix<Scalar>(0, 0);
    return result;
  }

  blas_int const m = detail::checked_blas_int(rows);
  blas_int const n = detail::checked_blas_int(cols);
  blas_int const lda = std::max<blas_int>(1, m);
  char const jobu = compute_vectors ? 'A' : 'N';
  char const jobvt = compute_vectors ? 'A' : 'N';
  detail::ColumnMajorLapackMatrix<Scalar> left(compute_vectors ? rows : 1, compute_vectors ? rows : 1);
  detail::ColumnMajorLapackMatrix<Scalar> right_transpose(compute_vectors ? cols : 1, compute_vectors ? cols : 1);
  blas_int const ldu = compute_vectors ? std::max<blas_int>(1, m) : 1;
  blas_int const ldvt = compute_vectors ? std::max<blas_int>(1, n) : 1;

  Scalar work_query{};
  blas_int const query_lwork = -1;
  uni20::lapack::gesvd(jobu, jobvt, m, n, matrix.data(), lda, result.singular_values.data(), left.data(), ldu,
                       right_transpose.data(), ldvt, &work_query, query_lwork);

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
  std::vector<Scalar> work(static_cast<std::size_t>(lwork), Scalar{});
  uni20::lapack::gesvd(jobu, jobvt, m, n, matrix.data(), lda, result.singular_values.data(), left.data(), ldu,
                       right_transpose.data(), ldvt, work.data(), lwork);

  result.left_singular_vectors = compute_vectors ? std::move(left) : detail::ColumnMajorLapackMatrix<Scalar>(0, 0);
  result.right_singular_vectors_transpose =
      compute_vectors ? std::move(right_transpose) : detail::ColumnMajorLapackMatrix<Scalar>(0, 0);
  return result;
}

/// \brief Compute a dense real singular value decomposition through LAPACK `gesdd`.
/// \details Uses the divide-and-conquer SVD driver. Singular values are
///          returned in descending order. If `compute_vectors` is true, this
///          computes the full decomposition `A = U * Sigma * VT`, with `U`
///          stored in `left_singular_vectors` and `VT` stored in
///          `right_singular_vectors_transpose`.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real matrix to decompose.
/// \param compute_vectors Whether to compute full left and right singular vectors.
/// \return Singular values and, optionally, full singular-vector matrices.
template <uni20::LapackReal Scalar>
RealSingularValueDecomposition<Scalar>
real_singular_value_decomposition_divide_and_conquer(detail::ColumnMajorLapackMatrix<Scalar> matrix,
                                                     bool compute_vectors)
{
  std::size_t const rows = static_cast<std::size_t>(matrix.rows());
  std::size_t const cols = static_cast<std::size_t>(matrix.cols());
  std::size_t const singular_value_count = std::min(rows, cols);

  RealSingularValueDecomposition<Scalar> result;
  result.singular_values.resize(singular_value_count);
  if (singular_value_count == 0)
  {
    result.left_singular_vectors = compute_vectors ? detail::ColumnMajorLapackMatrix<Scalar>(rows, rows)
                                                   : detail::ColumnMajorLapackMatrix<Scalar>(0, 0);
    result.right_singular_vectors_transpose = compute_vectors ? detail::ColumnMajorLapackMatrix<Scalar>(cols, cols)
                                                              : detail::ColumnMajorLapackMatrix<Scalar>(0, 0);
    return result;
  }

  blas_int const m = detail::checked_blas_int(rows);
  blas_int const n = detail::checked_blas_int(cols);
  blas_int const lda = std::max<blas_int>(1, m);
  char const jobz = compute_vectors ? 'A' : 'N';
  detail::ColumnMajorLapackMatrix<Scalar> left(compute_vectors ? rows : 1, compute_vectors ? rows : 1);
  detail::ColumnMajorLapackMatrix<Scalar> right_transpose(compute_vectors ? cols : 1, compute_vectors ? cols : 1);
  blas_int const ldu = compute_vectors ? std::max<blas_int>(1, m) : 1;
  blas_int const ldvt = compute_vectors ? std::max<blas_int>(1, n) : 1;
  std::vector<blas_int> iwork(
      static_cast<std::size_t>(std::max<blas_int>(1, 8 * detail::checked_blas_int(singular_value_count))));

  Scalar work_query{};
  blas_int const query_lwork = -1;
  uni20::lapack::gesdd(jobz, m, n, matrix.data(), lda, result.singular_values.data(), left.data(), ldu,
                       right_transpose.data(), ldvt, &work_query, query_lwork, iwork.data());

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
  std::vector<Scalar> work(static_cast<std::size_t>(lwork), Scalar{});
  uni20::lapack::gesdd(jobz, m, n, matrix.data(), lda, result.singular_values.data(), left.data(), ldu,
                       right_transpose.data(), ldvt, work.data(), lwork, iwork.data());

  result.left_singular_vectors = compute_vectors ? std::move(left) : detail::ColumnMajorLapackMatrix<Scalar>(0, 0);
  result.right_singular_vectors_transpose =
      compute_vectors ? std::move(right_transpose) : detail::ColumnMajorLapackMatrix<Scalar>(0, 0);
  return result;
}

/// \brief Compute a selected-index dense real SVD through LAPACK `gesvdx`.
/// \details Selects singular triplets by 0-based inclusive index in decreasing
///          singular-value order. If `compute_vectors` is true, this computes
///          the selected contribution `A_selected = U * Sigma * VT`, with `U`
///          stored in `left_singular_vectors` and `VT` stored in
///          `right_singular_vectors_transpose`.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real matrix to decompose.
/// \param first_index First 0-based decreasing singular-value index to compute.
/// \param last_index Last 0-based decreasing singular-value index to compute.
/// \param compute_vectors Whether to compute selected left and right singular vectors.
/// \return Selected singular values and, optionally, selected singular-vector matrices.
template <uni20::LapackReal Scalar>
RealSingularValueDecomposition<Scalar>
real_singular_value_decomposition_index_range(detail::ColumnMajorLapackMatrix<Scalar> matrix, std::size_t first_index,
                                              std::size_t last_index, bool compute_vectors)
{
  std::size_t const rows = static_cast<std::size_t>(matrix.rows());
  std::size_t const cols = static_cast<std::size_t>(matrix.cols());
  std::size_t const singular_value_count = std::min(rows, cols);
  if (singular_value_count == 0 || first_index > last_index || last_index >= singular_value_count)
  {
    throw std::invalid_argument("real_singular_value_decomposition_index_range received an invalid index range");
  }

  std::size_t const selected = last_index - first_index + 1;
  RealSingularValueDecomposition<Scalar> result;

  blas_int const m = detail::checked_blas_int(rows);
  blas_int const n = detail::checked_blas_int(cols);
  blas_int const lda = std::max<blas_int>(1, m);
  blas_int const first = detail::checked_blas_int(first_index + 1);
  blas_int const last = detail::checked_blas_int(last_index + 1);
  char const jobu = compute_vectors ? 'V' : 'N';
  char const jobvt = compute_vectors ? 'V' : 'N';
  char const range = 'I';
  std::vector<Scalar> singular_values_workspace(singular_value_count, Scalar{});
  detail::ColumnMajorLapackMatrix<Scalar> left(compute_vectors ? rows : 1, compute_vectors ? selected : 1);
  detail::ColumnMajorLapackMatrix<Scalar> right_transpose(compute_vectors ? selected : 1, compute_vectors ? cols : 1);
  blas_int const ldu = compute_vectors ? std::max<blas_int>(1, m) : 1;
  blas_int const ldvt = compute_vectors ? detail::checked_blas_int(selected) : 1;
  std::vector<blas_int> iwork(
      static_cast<std::size_t>(std::max<blas_int>(1, 12 * detail::checked_blas_int(singular_value_count))));
  blas_int selected_count = 0;

  Scalar work_query{};
  blas_int const query_lwork = -1;
  uni20::lapack::gesvdx(jobu, jobvt, range, m, n, matrix.data(), lda, Scalar{}, Scalar{}, first, last, selected_count,
                        singular_values_workspace.data(), left.data(), ldu, right_transpose.data(), ldvt, &work_query,
                        query_lwork, iwork.data());

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
  std::vector<Scalar> work(static_cast<std::size_t>(lwork), Scalar{});
  uni20::lapack::gesvdx(jobu, jobvt, range, m, n, matrix.data(), lda, Scalar{}, Scalar{}, first, last, selected_count,
                        singular_values_workspace.data(), left.data(), ldu, right_transpose.data(), ldvt, work.data(),
                        lwork, iwork.data());

  if (selected_count < 0 || static_cast<std::size_t>(selected_count) != selected)
  {
    throw std::runtime_error("LAPACK gesvdx returned an unexpected number of singular values");
  }

  result.singular_values.assign(singular_values_workspace.begin(), singular_values_workspace.begin() + selected_count);
  result.left_singular_vectors = compute_vectors ? std::move(left) : detail::ColumnMajorLapackMatrix<Scalar>(0, 0);
  result.right_singular_vectors_transpose =
      compute_vectors ? std::move(right_transpose) : detail::ColumnMajorLapackMatrix<Scalar>(0, 0);
  return result;
}

/// \brief Solve a real symmetric tridiagonal eigenvalue problem through LAPACK `sterf`.
///
/// \details `diagonal` has length `n`; `subdiagonal` has length `n - 1`.
///          Eigenvalues are returned in ascending order.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param diagonal Main diagonal of the tridiagonal matrix.
/// \param subdiagonal Subdiagonal/superdiagonal entries.
/// \return Eigenvalues in ascending order.

template <uni20::LapackReal Scalar>
TridiagonalEigensystem<Scalar> symmetric_tridiagonal_eigensystem_divide_and_conquer(std::vector<Scalar> diagonal,
                                                                                    std::vector<Scalar> subdiagonal,
                                                                                    bool compute_vectors)
{
  std::size_t const n = diagonal.size();
  if (subdiagonal.size() + 1 != n && !(n == 0 && subdiagonal.empty()))
  {
    throw std::invalid_argument(
        "symmetric_tridiagonal_eigensystem_divide_and_conquer received inconsistent diagonal sizes");
  }

  TridiagonalEigensystem<Scalar> result;
  result.eigenvalues = std::move(diagonal);
  if (n == 0)
  {
    result.eigenvectors = detail::ColumnMajorLapackMatrix<Scalar>(0, 0);
    return result;
  }

  blas_int const order = detail::checked_blas_int(n);
  char const jobz = compute_vectors ? 'V' : 'N';
  detail::ColumnMajorLapackMatrix<Scalar> z(compute_vectors ? n : 1, compute_vectors ? n : 1);
  blas_int const ldz = compute_vectors ? order : 1;
  Scalar dummy_subdiagonal{};
  Scalar* e = subdiagonal.empty() ? &dummy_subdiagonal : subdiagonal.data();

  Scalar work_query{};
  blas_int iwork_query = 0;
  blas_int const query_lwork = -1;
  uni20::lapack::stevd(jobz, order, result.eigenvalues.data(), e, z.data(), ldz, &work_query, query_lwork, &iwork_query,
                       query_lwork);

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
  blas_int const liwork = std::max<blas_int>(1, iwork_query);
  std::vector<Scalar> work(static_cast<std::size_t>(lwork), Scalar{});
  std::vector<blas_int> iwork(static_cast<std::size_t>(liwork), 0);
  uni20::lapack::stevd(jobz, order, result.eigenvalues.data(), e, z.data(), ldz, work.data(), lwork, iwork.data(),
                       liwork);

  result.eigenvectors = compute_vectors ? std::move(z) : detail::ColumnMajorLapackMatrix<Scalar>(0, 0);
  return result;
}

/// \brief Solve a selected real symmetric tridiagonal eigensystem through LAPACK `stevr`.
///
/// \details Selects eigenpairs by 0-based inclusive index in ascending
///          eigenvalue order. `diagonal` has length `n`; `subdiagonal` has
///          length `n - 1`. If `compute_vectors` is true, `eigenvectors` is an
///          `n`-by-`m` column-major matrix whose columns are the selected
///          normalized eigenvectors.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param diagonal Main diagonal of the tridiagonal matrix.
/// \param subdiagonal Subdiagonal/superdiagonal entries.
/// \param first_index First 0-based eigenvalue index to compute.
/// \param last_index Last 0-based eigenvalue index to compute.
/// \param compute_vectors Whether to compute selected eigenvectors.
/// \return Selected eigenvalues and, optionally, selected eigenvectors.
template <uni20::LapackReal Scalar>
TridiagonalEigensystem<Scalar>
symmetric_tridiagonal_eigensystem_index_range(std::vector<Scalar> diagonal, std::vector<Scalar> subdiagonal,
                                              std::size_t first_index, std::size_t last_index, bool compute_vectors)
{
  std::size_t const n = diagonal.size();
  if (subdiagonal.size() + 1 != n)
  {
    throw std::invalid_argument("symmetric_tridiagonal_eigensystem_index_range received inconsistent diagonal sizes");
  }
  if (n == 0 || first_index > last_index || last_index >= n)
  {
    throw std::invalid_argument("symmetric_tridiagonal_eigensystem_index_range received an invalid index range");
  }

  std::size_t const selected = last_index - first_index + 1;
  TridiagonalEigensystem<Scalar> result;
  result.eigenvalues.resize(selected);

  blas_int const order = detail::checked_blas_int(n);
  blas_int const first = detail::checked_blas_int(first_index + 1);
  blas_int const last = detail::checked_blas_int(last_index + 1);
  char const jobz = compute_vectors ? 'V' : 'N';
  char const range = 'I';
  detail::ColumnMajorLapackMatrix<Scalar> eigenvectors(compute_vectors ? n : 1, compute_vectors ? selected : 1);
  blas_int const ldz = compute_vectors ? order : 1;
  std::vector<blas_int> support(static_cast<std::size_t>(std::max<blas_int>(1, 2 * order)), 0);
  Scalar dummy_subdiagonal{};
  Scalar* e = subdiagonal.empty() ? &dummy_subdiagonal : subdiagonal.data();
  Scalar work_query{};
  blas_int iwork_query = 0;
  blas_int selected_count = 0;
  blas_int const query_lwork = -1;
  uni20::lapack::stevr(jobz, range, order, diagonal.data(), e, Scalar{}, Scalar{}, first, last, Scalar{},
                       selected_count, result.eigenvalues.data(), eigenvectors.data(), ldz, support.data(), &work_query,
                       query_lwork, &iwork_query, query_lwork);

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
  blas_int const liwork = std::max<blas_int>(1, iwork_query);
  std::vector<Scalar> work(static_cast<std::size_t>(lwork), Scalar{});
  std::vector<blas_int> iwork(static_cast<std::size_t>(liwork), 0);
  selected_count = 0;
  uni20::lapack::stevr(jobz, range, order, diagonal.data(), e, Scalar{}, Scalar{}, first, last, Scalar{},
                       selected_count, result.eigenvalues.data(), eigenvectors.data(), ldz, support.data(), work.data(),
                       lwork, iwork.data(), liwork);

  if (selected_count < 0 || static_cast<std::size_t>(selected_count) != selected)
  {
    throw std::runtime_error("LAPACK stevr returned an unexpected number of eigenvalues");
  }

  result.eigenvectors = compute_vectors ? std::move(eigenvectors) : detail::ColumnMajorLapackMatrix<Scalar>(0, 0);
  return result;
}

/// \brief Compute the real Schur decomposition of a layout-right matrix through LAPACK `gees`.
///
/// \details The public wrapper accepts the prototype row-major `RightMatrix`
///          used by the mdspan-facing LAPACK layer. Internally it copies to
///          a column-major DenseMatrix before calling Fortran
///          LAPACK, so the returned Schur vectors correspond to the logical
///          input matrix rather than its transpose.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.

template <uni20::LapackReal Scalar>
RealSchurSelectedSubspace<Scalar> real_schur_selected_subspace(RealSchurDecomposition<Scalar> decomposition,
                                                               std::vector<std::size_t> const& selected_blocks)
{
  if (!std::cmp_equal(decomposition.schur_form.rows(), decomposition.schur_form.cols()))
  {
    throw std::invalid_argument("real_schur_selected_subspace requires a square Schur form");
  }

  std::size_t const n = static_cast<std::size_t>(decomposition.schur_form.rows());
  RealSchurSelectedSubspace<Scalar> result;
  if (n == 0 || selected_blocks.empty())
  {
    result.decomposition = std::move(decomposition);
    return result;
  }
  if (decomposition.eigenvalues.size() != n)
  {
    throw std::invalid_argument("real_schur_selected_subspace received inconsistent Schur eigenvalue data");
  }
  bool const update_vectors = decomposition.schur_vectors.rows() != 0 || decomposition.schur_vectors.cols() != 0;
  if (update_vectors && (!std::cmp_equal(decomposition.schur_vectors.rows(), n) ||
                         !std::cmp_equal(decomposition.schur_vectors.cols(), n)))
  {
    throw std::invalid_argument("real_schur_selected_subspace received inconsistent Schur vectors");
  }
  if (decomposition.blocks.empty())
  {
    decomposition.blocks = detail::real_schur_blocks(decomposition.eigenvalues);
  }

  std::vector<bool> seen_blocks(decomposition.blocks.size(), false);
  std::vector<blas_int> select(n, 0);
  std::size_t expected_selected_dimension = 0;
  for (std::size_t const block_index : selected_blocks)
  {
    if (block_index >= decomposition.blocks.size())
    {
      throw std::invalid_argument("real_schur_selected_subspace received an out-of-range selected block index");
    }
    if (seen_blocks[block_index])
    {
      throw std::invalid_argument("real_schur_selected_subspace received a duplicate selected block index");
    }
    seen_blocks[block_index] = true;
    auto const& block = decomposition.blocks[block_index];
    if (block.begin + block.size > n)
    {
      throw std::invalid_argument("real_schur_selected_subspace received inconsistent Schur block metadata");
    }
    for (std::size_t offset = 0; offset < block.size; ++offset)
    {
      select[block.begin + offset] = 1;
    }
    expected_selected_dimension += block.size;
  }

  detail::ColumnMajorLapackMatrix<Scalar> schur_form = std::move(decomposition.schur_form);
  detail::ColumnMajorLapackMatrix<Scalar> schur_vectors =
      update_vectors ? std::move(decomposition.schur_vectors) : detail::ColumnMajorLapackMatrix<Scalar>(1, 1);
  blas_int const order = detail::checked_blas_int(n);
  blas_int const ldq = update_vectors ? order : 1;
  char const job = 'B';
  char const compq = update_vectors ? 'V' : 'N';
  std::vector<Scalar> wr(n, Scalar{});
  std::vector<Scalar> wi(n, Scalar{});
  blas_int selected_dimension = 0;
  Scalar reciprocal_eigenvalue_cluster_condition{};
  Scalar reciprocal_invariant_subspace_condition{};

  Scalar work_query = Scalar{};
  blas_int iwork_query = 0;
  blas_int const query_workspace = -1;
  uni20::lapack::trsen(job, compq, select.data(), order, schur_form.data(), order, schur_vectors.data(), ldq, wr.data(),
                       wi.data(), selected_dimension, reciprocal_eigenvalue_cluster_condition,
                       reciprocal_invariant_subspace_condition, &work_query, query_workspace, &iwork_query,
                       query_workspace);

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
  blas_int const liwork = std::max<blas_int>(1, iwork_query);
  std::vector<Scalar> work(static_cast<std::size_t>(lwork), Scalar{});
  std::vector<blas_int> iwork(static_cast<std::size_t>(liwork), 0);
  uni20::lapack::trsen(job, compq, select.data(), order, schur_form.data(), order, schur_vectors.data(), ldq, wr.data(),
                       wi.data(), selected_dimension, reciprocal_eigenvalue_cluster_condition,
                       reciprocal_invariant_subspace_condition, work.data(), lwork, iwork.data(), liwork);

  if (selected_dimension < 0 || static_cast<std::size_t>(selected_dimension) != expected_selected_dimension)
  {
    throw std::runtime_error("LAPACK trsen returned an unexpected selected subspace dimension");
  }

  decomposition.schur_form = std::move(schur_form);
  decomposition.schur_vectors =
      update_vectors ? std::move(schur_vectors) : detail::ColumnMajorLapackMatrix<Scalar>(0, 0);
  decomposition.eigenvalues.clear();
  decomposition.eigenvalues.reserve(n);
  for (std::size_t index = 0; index < n; ++index)
  {
    decomposition.eigenvalues.push_back(uni20::complex<Scalar>{wr[index], wi[index]});
  }
  decomposition.blocks = detail::real_schur_blocks(decomposition.eigenvalues);

  result.decomposition = std::move(decomposition);
  result.selected_dimension = static_cast<std::size_t>(selected_dimension);
  result.reciprocal_eigenvalue_cluster_condition = reciprocal_eigenvalue_cluster_condition;
  result.reciprocal_invariant_subspace_condition = reciprocal_invariant_subspace_condition;
  return result;
}

/// \brief Compute right eigenvectors of a real Schur form through LAPACK `trevc`.
/// \details The returned vectors are eigenvectors of `decomposition.schur_form`.
///          They are not multiplied by the Schur-vector matrix, so callers that
///          need eigenvectors of the original matrix must backtransform them
///          separately.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param decomposition Real Schur decomposition whose Schur form is used.
/// \return Complex right eigenvectors of the Schur form and the number computed.
template <uni20::LapackReal Scalar>
RealSchurRightEigenvectors<Scalar> real_schur_right_eigenvectors(RealSchurDecomposition<Scalar> decomposition)
{
  if (!std::cmp_equal(decomposition.schur_form.rows(), decomposition.schur_form.cols()))
  {
    throw std::invalid_argument("real_schur_right_eigenvectors requires a square Schur form");
  }

  std::size_t const n = static_cast<std::size_t>(decomposition.schur_form.rows());
  RealSchurRightEigenvectors<Scalar> result;
  result.right_eigenvectors = detail::ColumnMajorLapackMatrix<uni20::complex<Scalar>>(n, n);
  if (n == 0)
  {
    return result;
  }
  if (decomposition.eigenvalues.size() != n)
  {
    throw std::invalid_argument("real_schur_right_eigenvectors received inconsistent Schur eigenvalue data");
  }

  detail::ColumnMajorLapackMatrix<Scalar> schur_form = std::move(decomposition.schur_form);
  detail::ColumnMajorLapackMatrix<Scalar> left_vectors(1, 1);
  detail::ColumnMajorLapackMatrix<Scalar> right_vectors(n, n);
  std::vector<blas_int> select(n, 1);
  std::vector<Scalar> work(3 * n, Scalar{});
  blas_int const order = detail::checked_blas_int(n);
  blas_int computed_vectors = 0;
  char const side = 'R';
  char const howmny = 'A';
  uni20::lapack::trevc(side, howmny, select.data(), order, schur_form.data(), order, left_vectors.data(), 1,
                       right_vectors.data(), order, order, computed_vectors, work.data());

  if (computed_vectors < 0 || static_cast<std::size_t>(computed_vectors) != n)
  {
    throw std::runtime_error("LAPACK trevc returned an unexpected number of eigenvectors");
  }

  for (std::size_t col = 0; col < n;)
  {
    auto const eigenvalue = decomposition.eigenvalues[col];
    if (eigenvalue.imag() == Scalar{})
    {
      for (std::size_t row = 0; row < n; ++row)
      {
        result.right_eigenvectors[row, col] = uni20::complex<Scalar>{right_vectors[row, col], Scalar{}};
      }
      ++col;
      continue;
    }

    if (col + 1 >= n)
    {
      throw std::runtime_error("LAPACK trevc returned an incomplete complex eigenvector pair");
    }
    auto const next_eigenvalue = decomposition.eigenvalues[col + 1];
    Scalar const scale = std::max(Scalar{1}, std::max(detail::adl_abs(eigenvalue), detail::adl_abs(next_eigenvalue)));
    if (detail::adl_abs(next_eigenvalue - std::conj(eigenvalue)) >
        Scalar{100} * uni20::numeric_limits<Scalar>::epsilon() * scale)
    {
      throw std::runtime_error("LAPACK trevc returned a non-adjacent complex eigenvector pair");
    }

    for (std::size_t row = 0; row < n; ++row)
    {
      uni20::complex<Scalar> const positive_imaginary_vector{right_vectors[row, col], right_vectors[row, col + 1]};
      if (eigenvalue.imag() > Scalar{})
      {
        result.right_eigenvectors[row, col] = positive_imaginary_vector;
        result.right_eigenvectors[row, col + 1] = std::conj(positive_imaginary_vector);
      }
      else
      {
        result.right_eigenvectors[row, col] = std::conj(positive_imaginary_vector);
        result.right_eigenvectors[row, col + 1] = positive_imaginary_vector;
      }
    }
    col += 2;
  }

  result.computed_vectors = static_cast<std::size_t>(computed_vectors);
  return result;
}

/// \brief Estimate Schur-form eigenvalue and eigenvector conditioning through LAPACK `trsna`.
/// \details The returned arrays are aligned with `decomposition.eigenvalues`.
///          This helper asks LAPACK for all eigenpairs and computes the
///          required real-packed left and right Schur eigenvectors internally.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param decomposition Real Schur decomposition whose Schur form is used.
/// \return Reciprocal eigenvalue and right-eigenvector condition estimates.
template <uni20::LapackReal Scalar>
RealSchurConditionEstimates<Scalar> real_schur_condition_estimates(RealSchurDecomposition<Scalar> decomposition)
{
  if (!std::cmp_equal(decomposition.schur_form.rows(), decomposition.schur_form.cols()))
  {
    throw std::invalid_argument("real_schur_condition_estimates requires a square Schur form");
  }

  std::size_t const n = static_cast<std::size_t>(decomposition.schur_form.rows());
  RealSchurConditionEstimates<Scalar> result;
  result.reciprocal_eigenvalue_condition_numbers.resize(n);
  result.reciprocal_eigenvector_condition_numbers.resize(n);
  if (n == 0)
  {
    return result;
  }
  if (decomposition.eigenvalues.size() != n)
  {
    throw std::invalid_argument("real_schur_condition_estimates received inconsistent Schur eigenvalue data");
  }

  detail::ColumnMajorLapackMatrix<Scalar> schur_form = std::move(decomposition.schur_form);
  detail::ColumnMajorLapackMatrix<Scalar> left_vectors(n, n);
  detail::ColumnMajorLapackMatrix<Scalar> right_vectors(n, n);
  std::vector<blas_int> select(n, 1);
  std::vector<Scalar> trevc_work(3 * n, Scalar{});
  blas_int const order = detail::checked_blas_int(n);
  blas_int computed_vectors = 0;
  uni20::lapack::trevc('B', 'A', select.data(), order, schur_form.data(), order, left_vectors.data(), order,
                       right_vectors.data(), order, order, computed_vectors, trevc_work.data());

  if (computed_vectors < 0 || static_cast<std::size_t>(computed_vectors) != n)
  {
    throw std::runtime_error("LAPACK trevc returned an unexpected number of Schur eigenvectors");
  }

  blas_int computed_estimates = 0;
  blas_int const ldwork = order;
  std::vector<Scalar> work(n * (n + 6), Scalar{});
  std::vector<blas_int> iwork(static_cast<std::size_t>(std::max<blas_int>(1, 2 * order - 2)), 0);
  uni20::lapack::trsna('B', 'A', select.data(), order, schur_form.data(), order, left_vectors.data(), order,
                       right_vectors.data(), order, result.reciprocal_eigenvalue_condition_numbers.data(),
                       result.reciprocal_eigenvector_condition_numbers.data(), order, computed_estimates, work.data(),
                       ldwork, iwork.data());

  if (computed_estimates < 0 || static_cast<std::size_t>(computed_estimates) != n)
  {
    throw std::runtime_error("LAPACK trsna returned an unexpected number of condition estimates");
  }

  result.computed_estimates = static_cast<std::size_t>(computed_estimates);
  return result;
}

/// \brief Reorder a layout-right complex Schur decomposition by moving selected entries to the front.
///
/// \details Complex Schur form contains only scalar 1x1 diagonal entries.
///          `leading_order` names entries in the input decomposition's current
///          order. Entries not named are left after the requested prefix in
///          their original relative order.
/// \tparam Real `float` or `double`.
/// \param decomposition Complex Schur decomposition to reorder.
/// \param leading_order Input entry indices to move to the leading positions.
/// \return Reordered complex Schur decomposition.

template <uni20::LapackReal Real>
RealNonsymmetricBalance<Real>
real_nonsymmetric_balance(detail::ColumnMajorLapackMatrix<Real> matrix,
                          RealNonsymmetricBalanceJob job = RealNonsymmetricBalanceJob::Both)
{
  if (!std::cmp_equal(matrix.rows(), matrix.cols()))
  {
    throw std::invalid_argument("real_nonsymmetric_balance requires a square matrix");
  }

  std::size_t const n = static_cast<std::size_t>(matrix.rows());
  RealNonsymmetricBalance<Real> result;
  result.balanced_matrix = std::move(matrix);
  result.scale.resize(n);
  result.balanced_last_exclusive = n;
  if (n == 0)
  {
    return result;
  }

  blas_int const order = detail::checked_blas_int(n);
  blas_int first = 0;
  blas_int last = 0;
  uni20::lapack::gebal(detail::lapack_balance_job(job), order, result.balanced_matrix.data(), order, first, last,
                       result.scale.data());

  if (first <= 0 || last < first || last > order)
  {
    throw std::runtime_error("LAPACK gebal returned an invalid balanced block interval");
  }
  result.balanced_first = static_cast<std::size_t>(first - 1);
  result.balanced_last_exclusive = static_cast<std::size_t>(last);
  return result;
}

/// \brief Backtransform real right eigenvectors after `real_nonsymmetric_balance`.
/// \details The input vectors use LAPACK's real storage convention, so complex
///          conjugate pairs are still represented by adjacent real columns.
///          Use this before unpacking those columns into complex vectors.
/// \tparam Real Real scalar type satisfying `uni20::LapackReal`.
/// \param vectors Real right eigenvector columns of the balanced matrix.
/// \param balance Balancing metadata returned by `real_nonsymmetric_balance`.
/// \param job The same balancing operation used for the matrix.
/// \return Real right eigenvector columns backtransformed to the original matrix.
template <uni20::LapackReal Real>
detail::ColumnMajorLapackMatrix<Real>
real_nonsymmetric_balance_backtransform_right_vectors(detail::ColumnMajorLapackMatrix<Real> vectors,
                                                      RealNonsymmetricBalance<Real> const& balance,
                                                      RealNonsymmetricBalanceJob job = RealNonsymmetricBalanceJob::Both)
{
  std::size_t const n = balance.scale.size();
  if (!std::cmp_equal(vectors.rows(), n))
  {
    throw std::invalid_argument("real_nonsymmetric_balance_backtransform_right_vectors received incompatible vectors");
  }
  if (balance.balanced_first > balance.balanced_last_exclusive || balance.balanced_last_exclusive > n)
  {
    throw std::invalid_argument("real_nonsymmetric_balance_backtransform_right_vectors received invalid balance data");
  }
  if (n == 0 || vectors.cols() == 0)
  {
    return vectors;
  }

  blas_int const order = detail::checked_blas_int(n);
  blas_int const first = detail::checked_blas_int(balance.balanced_first + 1);
  blas_int const last = detail::checked_blas_int(balance.balanced_last_exclusive);
  blas_int const vector_count = detail::checked_blas_int(vectors.cols());
  std::vector<Real> scale = balance.scale;
  uni20::lapack::gebak(detail::lapack_balance_job(job), 'R', order, first, last, scale.data(), vector_count,
                       vectors.data(), order);
  return vectors;
}

/// \brief Balance a dense real nonsymmetric matrix pencil through LAPACK `ggbal`.
/// \details The input matrices are copied by value and overwritten with the
///          balanced pencil. The returned balanced interval is zero-based and
///          half-open; pass the full result to
///          `real_generalized_nonsymmetric_balance_backtransform_right_vectors`
///          to backtransform real right eigenvectors of the balanced pencil.
/// \tparam Real Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real square matrix `A` in column-major local storage.
/// \param metric Real square matrix `B` in column-major local storage.
/// \param job Balancing operation to apply.
/// \return Balanced pencil, LAPACK left/right scale data, and balanced active interval.
template <uni20::LapackReal Real>
RealGeneralizedNonsymmetricBalance<Real>
real_generalized_nonsymmetric_balance(detail::ColumnMajorLapackMatrix<Real> matrix,
                                      detail::ColumnMajorLapackMatrix<Real> metric,
                                      RealNonsymmetricBalanceJob job = RealNonsymmetricBalanceJob::Both)
{
  if (!std::cmp_equal(matrix.rows(), matrix.cols()))
  {
    throw std::invalid_argument("real_generalized_nonsymmetric_balance requires a square matrix");
  }
  if (!std::cmp_equal(metric.rows(), metric.cols()))
  {
    throw std::invalid_argument("real_generalized_nonsymmetric_balance requires a square metric matrix");
  }
  if (!std::cmp_equal(matrix.rows(), metric.rows()))
  {
    throw std::invalid_argument("real_generalized_nonsymmetric_balance received incompatible matrix sizes");
  }

  std::size_t const n = static_cast<std::size_t>(matrix.rows());
  RealGeneralizedNonsymmetricBalance<Real> result;
  result.balanced_matrix = std::move(matrix);
  result.balanced_metric = std::move(metric);
  result.left_scale.resize(n);
  result.right_scale.resize(n);
  result.balanced_last_exclusive = n;
  if (n == 0)
  {
    return result;
  }

  blas_int const order = detail::checked_blas_int(n);
  blas_int first = 0;
  blas_int last = 0;
  std::vector<Real> work(6 * n, Real{});
  uni20::lapack::ggbal(detail::lapack_balance_job(job), order, result.balanced_matrix.data(), order,
                       result.balanced_metric.data(), order, first, last, result.left_scale.data(),
                       result.right_scale.data(), work.data());

  if (first <= 0 || last < first || last > order)
  {
    throw std::runtime_error("LAPACK ggbal returned an invalid balanced block interval");
  }
  result.balanced_first = static_cast<std::size_t>(first - 1);
  result.balanced_last_exclusive = static_cast<std::size_t>(last);
  return result;
}

/// \brief Backtransform real right eigenvectors after generalized pencil balancing.
/// \details The input vectors use LAPACK's real storage convention, so complex
///          conjugate pairs are still represented by adjacent real columns.
///          Use this before unpacking those columns into complex vectors.
/// \tparam Real Real scalar type satisfying `uni20::LapackReal`.
/// \param vectors Real right eigenvector columns of the balanced pencil.
/// \param balance Balancing metadata returned by `real_generalized_nonsymmetric_balance`.
/// \param job The same balancing operation used for the pencil.
/// \return Real right eigenvector columns backtransformed to the original pencil.
template <uni20::LapackReal Real>
detail::ColumnMajorLapackMatrix<Real> real_generalized_nonsymmetric_balance_backtransform_right_vectors(
    detail::ColumnMajorLapackMatrix<Real> vectors, RealGeneralizedNonsymmetricBalance<Real> const& balance,
    RealNonsymmetricBalanceJob job = RealNonsymmetricBalanceJob::Both)
{
  std::size_t const n = balance.left_scale.size();
  if (balance.right_scale.size() != n)
  {
    throw std::invalid_argument(
        "real_generalized_nonsymmetric_balance_backtransform_right_vectors received inconsistent balance data");
  }
  if (!std::cmp_equal(vectors.rows(), n))
  {
    throw std::invalid_argument(
        "real_generalized_nonsymmetric_balance_backtransform_right_vectors received incompatible vectors");
  }
  if (balance.balanced_first > balance.balanced_last_exclusive || balance.balanced_last_exclusive > n)
  {
    throw std::invalid_argument(
        "real_generalized_nonsymmetric_balance_backtransform_right_vectors received invalid balance data");
  }
  if (n == 0 || vectors.cols() == 0)
  {
    return vectors;
  }

  blas_int const order = detail::checked_blas_int(n);
  blas_int const first = detail::checked_blas_int(balance.balanced_first + 1);
  blas_int const last = detail::checked_blas_int(balance.balanced_last_exclusive);
  blas_int const vector_count = detail::checked_blas_int(vectors.cols());
  std::vector<Real> left_scale = balance.left_scale;
  std::vector<Real> right_scale = balance.right_scale;
  uni20::lapack::ggbak(detail::lapack_balance_job(job), 'R', order, first, last, left_scale.data(), right_scale.data(),
                       vector_count, vectors.data(), order);
  return vectors;
}

/// \brief Reduce a real matrix pencil to generalized upper Hessenberg form through LAPACK `gghrd`.
/// \details LAPACK `gghrd` assumes the second matrix is already upper
///          triangular. This helper preserves that boundary: callers that have
///          a general second matrix should triangularize it explicitly before
///          using this lower-level QZ building block. The reduction satisfies
///          `A = Q * H * Z^T` and `B = Q * T * Z^T` when vectors are requested.
/// \tparam Real Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real square matrix `A` in column-major local storage.
/// \param upper_triangular_metric Real upper-triangular square matrix `B`.
/// \param compute_vectors Whether to compute the left and right orthogonal transformations.
/// \return Generalized upper Hessenberg forms and optional orthogonal factors.
template <uni20::LapackReal Real>
RealGeneralizedHessenbergReduction<Real>
real_generalized_hessenberg_reduction(detail::ColumnMajorLapackMatrix<Real> matrix,
                                      detail::ColumnMajorLapackMatrix<Real> upper_triangular_metric,
                                      bool compute_vectors)
{
  if (!std::cmp_equal(matrix.rows(), matrix.cols()))
  {
    throw std::invalid_argument("real_generalized_hessenberg_reduction requires a square matrix");
  }
  if (!std::cmp_equal(upper_triangular_metric.rows(), upper_triangular_metric.cols()))
  {
    throw std::invalid_argument("real_generalized_hessenberg_reduction requires a square metric matrix");
  }
  if (!std::cmp_equal(matrix.rows(), upper_triangular_metric.rows()))
  {
    throw std::invalid_argument("real_generalized_hessenberg_reduction received incompatible matrix sizes");
  }

  std::size_t const n = static_cast<std::size_t>(matrix.rows());
  for (std::size_t col = 0; col < n; ++col)
  {
    for (std::size_t row = col + 1; row < n; ++row)
    {
      if (upper_triangular_metric[row, col] != Real{})
      {
        throw std::invalid_argument("real_generalized_hessenberg_reduction requires an upper-triangular metric");
      }
    }
  }

  RealGeneralizedHessenbergReduction<Real> result;
  result.first = 0;
  result.last_exclusive = n;
  if (n == 0)
  {
    result.matrix_hessenberg_form = detail::ColumnMajorLapackMatrix<Real>(0, 0);
    result.metric_triangular_form = detail::ColumnMajorLapackMatrix<Real>(0, 0);
    result.left_orthogonal_vectors = detail::ColumnMajorLapackMatrix<Real>(0, 0);
    result.right_orthogonal_vectors = detail::ColumnMajorLapackMatrix<Real>(0, 0);
    return result;
  }

  detail::ColumnMajorLapackMatrix<Real> left_vectors(compute_vectors ? n : 1, compute_vectors ? n : 1);
  detail::ColumnMajorLapackMatrix<Real> right_vectors(compute_vectors ? n : 1, compute_vectors ? n : 1);
  blas_int const order = detail::checked_blas_int(n);
  blas_int const first = 1;
  blas_int const last = order;
  blas_int const ldq = compute_vectors ? order : 1;
  blas_int const ldz = compute_vectors ? order : 1;
  char const compq = compute_vectors ? 'I' : 'N';
  char const compz = compute_vectors ? 'I' : 'N';

  uni20::lapack::gghrd(compq, compz, order, first, last, matrix.data(), order, upper_triangular_metric.data(), order,
                       left_vectors.data(), ldq, right_vectors.data(), ldz);

  result.matrix_hessenberg_form = std::move(matrix);
  result.metric_triangular_form = std::move(upper_triangular_metric);
  result.left_orthogonal_vectors =
      compute_vectors ? std::move(left_vectors) : detail::ColumnMajorLapackMatrix<Real>(0, 0);
  result.right_orthogonal_vectors =
      compute_vectors ? std::move(right_vectors) : detail::ColumnMajorLapackMatrix<Real>(0, 0);
  return result;
}

/// \brief Compute generalized real Schur form from a generalized Hessenberg pencil through LAPACK `hgeqz`.
/// \details LAPACK `hgeqz` assumes the first matrix is upper Hessenberg and
///          the second matrix is upper triangular. This helper preserves that
///          boundary for projected QZ pipelines that have already called
///          `gghrd`. When vectors are requested, the returned forms satisfy
///          `H = Q * S * Z^T` and `T = Q * P * Z^T`.
/// \tparam Real Real scalar type satisfying `uni20::LapackReal`.
/// \param hessenberg Real upper-Hessenberg square matrix `H`.
/// \param upper_triangular_metric Real upper-triangular square matrix `T`.
/// \param compute_vectors Whether to compute left and right generalized Schur vectors.
/// \return Generalized Schur forms, optional Schur vectors, and projective eigenvalue data.
template <uni20::LapackReal Real>
RealGeneralizedSchurDecomposition<Real>
real_generalized_hessenberg_schur(detail::ColumnMajorLapackMatrix<Real> hessenberg,
                                  detail::ColumnMajorLapackMatrix<Real> upper_triangular_metric, bool compute_vectors)
{
  if (!std::cmp_equal(hessenberg.rows(), hessenberg.cols()))
  {
    throw std::invalid_argument("real_generalized_hessenberg_schur requires a square Hessenberg matrix");
  }
  if (!std::cmp_equal(upper_triangular_metric.rows(), upper_triangular_metric.cols()))
  {
    throw std::invalid_argument("real_generalized_hessenberg_schur requires a square metric matrix");
  }
  if (!std::cmp_equal(hessenberg.rows(), upper_triangular_metric.rows()))
  {
    throw std::invalid_argument("real_generalized_hessenberg_schur received incompatible matrix sizes");
  }

  std::size_t const n = static_cast<std::size_t>(hessenberg.rows());
  for (std::size_t col = 0; col < n; ++col)
  {
    for (std::size_t row = col + 2; row < n; ++row)
    {
      if (hessenberg[row, col] != Real{})
      {
        throw std::invalid_argument("real_generalized_hessenberg_schur requires an upper-Hessenberg matrix");
      }
    }
    for (std::size_t row = col + 1; row < n; ++row)
    {
      if (upper_triangular_metric[row, col] != Real{})
      {
        throw std::invalid_argument("real_generalized_hessenberg_schur requires an upper-triangular metric");
      }
    }
  }

  RealGeneralizedSchurDecomposition<Real> result;
  result.alpha.resize(n);
  result.beta.resize(n);
  result.eigenvalues.resize(n);
  if (n == 0)
  {
    result.matrix_schur_form = detail::ColumnMajorLapackMatrix<Real>(0, 0);
    result.metric_schur_form = detail::ColumnMajorLapackMatrix<Real>(0, 0);
    result.left_schur_vectors = detail::ColumnMajorLapackMatrix<Real>(0, 0);
    result.right_schur_vectors = detail::ColumnMajorLapackMatrix<Real>(0, 0);
    return result;
  }

  blas_int const order = detail::checked_blas_int(n);
  std::vector<Real> alphar(n, Real{});
  std::vector<Real> alphai(n, Real{});
  std::vector<Real> beta(n, Real{});
  detail::ColumnMajorLapackMatrix<Real> left_vectors(compute_vectors ? n : 1, compute_vectors ? n : 1);
  detail::ColumnMajorLapackMatrix<Real> right_vectors(compute_vectors ? n : 1, compute_vectors ? n : 1);
  blas_int const ldq = compute_vectors ? order : 1;
  blas_int const ldz = compute_vectors ? order : 1;
  blas_int const first = 1;
  blas_int const last = order;
  char const job = 'S';
  char const compq = compute_vectors ? 'I' : 'N';
  char const compz = compute_vectors ? 'I' : 'N';

  Real work_query{};
  blas_int const query_lwork = -1;
  uni20::lapack::hgeqz(job, compq, compz, order, first, last, hessenberg.data(), order, upper_triangular_metric.data(),
                       order, alphar.data(), alphai.data(), beta.data(), left_vectors.data(), ldq, right_vectors.data(),
                       ldz, &work_query, query_lwork);

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
  std::vector<Real> work(static_cast<std::size_t>(lwork), Real{});
  uni20::lapack::hgeqz(job, compq, compz, order, first, last, hessenberg.data(), order, upper_triangular_metric.data(),
                       order, alphar.data(), alphai.data(), beta.data(), left_vectors.data(), ldq, right_vectors.data(),
                       ldz, work.data(), lwork);

  result.matrix_schur_form = std::move(hessenberg);
  result.metric_schur_form = std::move(upper_triangular_metric);
  result.left_schur_vectors = compute_vectors ? std::move(left_vectors) : detail::ColumnMajorLapackMatrix<Real>(0, 0);
  result.right_schur_vectors = compute_vectors ? std::move(right_vectors) : detail::ColumnMajorLapackMatrix<Real>(0, 0);
  for (std::size_t i = 0; i < n; ++i)
  {
    result.alpha[i] = uni20::complex<Real>{alphar[i], alphai[i]};
    result.beta[i] = beta[i];
    result.eigenvalues[i] = detail::generalized_eigenvalue(alphar[i], alphai[i], beta[i]);
  }
  result.blocks = detail::real_schur_blocks(result.eigenvalues);
  return result;
}

/// \brief Compute the generalized real Schur decomposition of a matrix pencil through LAPACK `gges`.
/// \details Computes unsorted QZ form for `A - lambda B`. LAPACK overwrites
///          `matrix` with the quasi-triangular generalized Schur form `S` and
///          `metric` with the upper-triangular form `T`. Generalized
///          eigenvalues are preserved as projective pairs `alpha / beta`, with
///          finite ratios also stored in `eigenvalues`.
/// \tparam Real Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real square matrix `A` in column-major local storage.
/// \param metric Real square matrix `B` in column-major local storage.
/// \param compute_vectors Whether to compute left and right generalized Schur vectors.
/// \return Generalized Schur forms, optional Schur vectors, and projective eigenvalue data.
template <uni20::LapackReal Real>
RealGeneralizedSchurDecomposition<Real> real_generalized_schur(detail::ColumnMajorLapackMatrix<Real> matrix,
                                                               detail::ColumnMajorLapackMatrix<Real> metric,
                                                               bool compute_vectors)
{
  if (!std::cmp_equal(matrix.rows(), matrix.cols()))
  {
    throw std::invalid_argument("real_generalized_schur requires a square matrix");
  }
  if (!std::cmp_equal(metric.rows(), metric.cols()))
  {
    throw std::invalid_argument("real_generalized_schur requires a square metric matrix");
  }
  if (!std::cmp_equal(matrix.rows(), metric.rows()))
  {
    throw std::invalid_argument("real_generalized_schur received incompatible matrix sizes");
  }

  std::size_t const n = static_cast<std::size_t>(matrix.rows());
  RealGeneralizedSchurDecomposition<Real> result;
  result.alpha.resize(n);
  result.beta.resize(n);
  result.eigenvalues.resize(n);
  if (n == 0)
  {
    result.matrix_schur_form = detail::ColumnMajorLapackMatrix<Real>(0, 0);
    result.metric_schur_form = detail::ColumnMajorLapackMatrix<Real>(0, 0);
    result.left_schur_vectors = detail::ColumnMajorLapackMatrix<Real>(0, 0);
    result.right_schur_vectors = detail::ColumnMajorLapackMatrix<Real>(0, 0);
    return result;
  }

  blas_int const order = detail::checked_blas_int(n);
  std::vector<Real> alphar(n, Real{});
  std::vector<Real> alphai(n, Real{});
  std::vector<Real> beta(n, Real{});
  detail::ColumnMajorLapackMatrix<Real> left_vectors(compute_vectors ? n : 1, compute_vectors ? n : 1);
  detail::ColumnMajorLapackMatrix<Real> right_vectors(compute_vectors ? n : 1, compute_vectors ? n : 1);
  blas_int const ldvsl = compute_vectors ? order : 1;
  blas_int const ldvsr = compute_vectors ? order : 1;
  char const jobvsl = compute_vectors ? 'V' : 'N';
  char const jobvsr = compute_vectors ? 'V' : 'N';
  char const sort = 'N';
  blas_int selected_dimension = 0;
  std::vector<blas_int> bwork(std::max<std::size_t>(1, n), 0);

  Real work_query = Real{};
  blas_int const query_lwork = -1;
  uni20::lapack::gges(jobvsl, jobvsr, sort, order, matrix.data(), order, metric.data(), order, selected_dimension,
                      alphar.data(), alphai.data(), beta.data(), left_vectors.data(), ldvsl, right_vectors.data(),
                      ldvsr, &work_query, query_lwork, bwork.data());

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
  std::vector<Real> work(static_cast<std::size_t>(lwork), Real{});
  uni20::lapack::gges(jobvsl, jobvsr, sort, order, matrix.data(), order, metric.data(), order, selected_dimension,
                      alphar.data(), alphai.data(), beta.data(), left_vectors.data(), ldvsl, right_vectors.data(),
                      ldvsr, work.data(), lwork, bwork.data());

  result.matrix_schur_form = std::move(matrix);
  result.metric_schur_form = std::move(metric);
  result.left_schur_vectors = compute_vectors ? std::move(left_vectors) : detail::ColumnMajorLapackMatrix<Real>(0, 0);
  result.right_schur_vectors = compute_vectors ? std::move(right_vectors) : detail::ColumnMajorLapackMatrix<Real>(0, 0);
  result.selected_dimension = static_cast<std::size_t>(selected_dimension);
  for (std::size_t i = 0; i < n; ++i)
  {
    result.alpha[i] = uni20::complex<Real>{alphar[i], alphai[i]};
    result.beta[i] = beta[i];
    result.eigenvalues[i] = detail::generalized_eigenvalue(alphar[i], alphai[i], beta[i]);
  }
  result.blocks = detail::real_schur_blocks(result.eigenvalues);

  return result;
}

/// \brief Reorder a generalized real Schur decomposition by moving selected blocks to the front.
/// \details Uses LAPACK `tgexc` to reorder the QZ forms of `A - lambda B`.
///          Real-arithmetic 2x2 complex-conjugate blocks are moved as
///          indivisible units, and the projective `alpha`/`beta` data is kept
///          aligned with the new block order.
/// \tparam Real Real scalar type satisfying `uni20::LapackReal`.
/// \param decomposition Generalized real Schur decomposition to reorder.
/// \param leading_block_order Input block indices to move to the leading positions.
/// \return Reordered generalized real Schur decomposition.
template <uni20::LapackReal Real>
RealGeneralizedSchurDecomposition<Real>
reorder_real_generalized_schur(RealGeneralizedSchurDecomposition<Real> decomposition,
                               std::vector<std::size_t> const& leading_block_order)
{
  if (!std::cmp_equal(decomposition.matrix_schur_form.rows(), decomposition.matrix_schur_form.cols()))
  {
    throw std::invalid_argument("reorder_real_generalized_schur requires a square matrix Schur form");
  }
  if (!std::cmp_equal(decomposition.metric_schur_form.rows(), decomposition.metric_schur_form.cols()))
  {
    throw std::invalid_argument("reorder_real_generalized_schur requires a square metric Schur form");
  }

  std::size_t const n = static_cast<std::size_t>(decomposition.matrix_schur_form.rows());
  if (n == 0 || leading_block_order.empty())
  {
    return decomposition;
  }
  if (!std::cmp_equal(decomposition.metric_schur_form.rows(), n) ||
      !std::cmp_equal(decomposition.matrix_schur_form.cols(), n) ||
      !std::cmp_equal(decomposition.metric_schur_form.cols(), n))
  {
    throw std::invalid_argument("reorder_real_generalized_schur received inconsistent Schur form sizes");
  }
  if (decomposition.alpha.size() != n || decomposition.beta.size() != n || decomposition.eigenvalues.size() != n)
  {
    throw std::invalid_argument("reorder_real_generalized_schur received inconsistent eigenvalue data");
  }
  bool const has_left_vectors =
      decomposition.left_schur_vectors.rows() != 0 || decomposition.left_schur_vectors.cols() != 0;
  bool const has_right_vectors =
      decomposition.right_schur_vectors.rows() != 0 || decomposition.right_schur_vectors.cols() != 0;
  if (has_left_vectors != has_right_vectors)
  {
    throw std::invalid_argument("reorder_real_generalized_schur requires both Schur vector sets or neither");
  }
  bool const update_vectors = has_left_vectors;
  if (update_vectors && (!std::cmp_equal(decomposition.left_schur_vectors.rows(), n) ||
                         !std::cmp_equal(decomposition.left_schur_vectors.cols(), n) ||
                         !std::cmp_equal(decomposition.right_schur_vectors.rows(), n) ||
                         !std::cmp_equal(decomposition.right_schur_vectors.cols(), n)))
  {
    throw std::invalid_argument("reorder_real_generalized_schur received inconsistent Schur vectors");
  }

  std::vector<std::size_t> target_order =
      detail::complete_leading_block_order(decomposition.blocks.size(), leading_block_order);
  std::vector<std::size_t> current_order(decomposition.blocks.size());
  std::iota(current_order.begin(), current_order.end(), std::size_t{0});
  std::vector<RealSchurBlock<Real>> current_blocks = decomposition.blocks;

  blas_int const order = detail::checked_blas_int(n);
  blas_int const ldq = update_vectors ? order : 1;
  blas_int const ldz = update_vectors ? order : 1;
  detail::ColumnMajorLapackMatrix<Real> left_vectors =
      update_vectors ? std::move(decomposition.left_schur_vectors) : detail::ColumnMajorLapackMatrix<Real>(1, 1);
  detail::ColumnMajorLapackMatrix<Real> right_vectors =
      update_vectors ? std::move(decomposition.right_schur_vectors) : detail::ColumnMajorLapackMatrix<Real>(1, 1);
  blas_int const lwork = std::max<blas_int>(1, 4 * order + 16);
  std::vector<Real> work(static_cast<std::size_t>(lwork), Real{});

  for (std::size_t target_position = 0; target_position < leading_block_order.size(); ++target_position)
  {
    std::size_t const desired_original_index = target_order[target_position];
    auto const current_it = std::ranges::find(current_order, desired_original_index);
    if (current_it == current_order.end())
    {
      throw std::logic_error("generalized real Schur block reorder lost a block index");
    }
    std::size_t const current_position = static_cast<std::size_t>(current_it - current_order.begin());
    if (current_position == target_position)
    {
      continue;
    }

    blas_int first = detail::checked_blas_int(current_blocks[current_position].begin + 1);
    blas_int last = detail::checked_blas_int(current_blocks[target_position].begin + 1);
    uni20::lapack::tgexc(update_vectors, update_vectors, order, decomposition.matrix_schur_form.data(), order,
                         decomposition.metric_schur_form.data(), order, left_vectors.data(), ldq, right_vectors.data(),
                         ldz, first, last, work.data(), lwork);

    std::size_t const moved_index = current_order[current_position];
    current_order.erase(current_order.begin() + static_cast<std::ptrdiff_t>(current_position));
    current_order.insert(current_order.begin() + static_cast<std::ptrdiff_t>(target_position), moved_index);
    current_blocks = detail::ordered_schur_blocks(decomposition.blocks, current_order);
  }

  decomposition.left_schur_vectors =
      update_vectors ? std::move(left_vectors) : detail::ColumnMajorLapackMatrix<Real>(0, 0);
  decomposition.right_schur_vectors =
      update_vectors ? std::move(right_vectors) : detail::ColumnMajorLapackMatrix<Real>(0, 0);
  decomposition.alpha = detail::ordered_schur_block_values(decomposition.alpha, decomposition.blocks, current_order);
  decomposition.beta = detail::ordered_schur_block_values(decomposition.beta, decomposition.blocks, current_order);
  decomposition.blocks = detail::ordered_schur_blocks(decomposition.blocks, current_order);
  decomposition.eigenvalues = detail::eigenvalues_from_schur_blocks(decomposition.blocks);
  return decomposition;
}

/// \brief Move selected generalized real Schur blocks to the leading deflating subspace and estimate conditioning.
/// \details Uses LAPACK `tgsen` with `IJOB=4`, returning lower bounds for the
///          left/right projection quantities (`PL`, `PR`) and F-norm
///          estimates of `Difu` and `Difl`. Selection is by real Schur block,
///          so real-arithmetic 2x2 complex-conjugate blocks cannot be split.
/// \tparam Real Real scalar type satisfying `uni20::LapackReal`.
/// \param decomposition Generalized real Schur decomposition to reorder and diagnose.
/// \param selected_blocks Block indices to move into the leading deflating subspace.
/// \return Reordered generalized Schur decomposition and selected-subspace diagnostics.
template <uni20::LapackReal Real>
RealGeneralizedSchurSelectedSubspace<Real>
real_generalized_schur_selected_subspace(RealGeneralizedSchurDecomposition<Real> decomposition,
                                         std::vector<std::size_t> const& selected_blocks)
{
  if (!std::cmp_equal(decomposition.matrix_schur_form.rows(), decomposition.matrix_schur_form.cols()))
  {
    throw std::invalid_argument("real_generalized_schur_selected_subspace requires a square matrix Schur form");
  }
  if (!std::cmp_equal(decomposition.metric_schur_form.rows(), decomposition.metric_schur_form.cols()))
  {
    throw std::invalid_argument("real_generalized_schur_selected_subspace requires a square metric Schur form");
  }

  std::size_t const n = static_cast<std::size_t>(decomposition.matrix_schur_form.rows());
  RealGeneralizedSchurSelectedSubspace<Real> result;
  if (n == 0 || selected_blocks.empty())
  {
    result.decomposition = std::move(decomposition);
    return result;
  }
  if (!std::cmp_equal(decomposition.metric_schur_form.rows(), n) ||
      !std::cmp_equal(decomposition.matrix_schur_form.cols(), n) ||
      !std::cmp_equal(decomposition.metric_schur_form.cols(), n))
  {
    throw std::invalid_argument("real_generalized_schur_selected_subspace received inconsistent Schur form sizes");
  }
  if (decomposition.alpha.size() != n || decomposition.beta.size() != n || decomposition.eigenvalues.size() != n)
  {
    throw std::invalid_argument("real_generalized_schur_selected_subspace received inconsistent eigenvalue data");
  }
  bool const has_left_vectors =
      decomposition.left_schur_vectors.rows() != 0 || decomposition.left_schur_vectors.cols() != 0;
  bool const has_right_vectors =
      decomposition.right_schur_vectors.rows() != 0 || decomposition.right_schur_vectors.cols() != 0;
  if (has_left_vectors != has_right_vectors)
  {
    throw std::invalid_argument("real_generalized_schur_selected_subspace requires both Schur vector sets or neither");
  }
  bool const update_vectors = has_left_vectors;
  if (update_vectors && (!std::cmp_equal(decomposition.left_schur_vectors.rows(), n) ||
                         !std::cmp_equal(decomposition.left_schur_vectors.cols(), n) ||
                         !std::cmp_equal(decomposition.right_schur_vectors.rows(), n) ||
                         !std::cmp_equal(decomposition.right_schur_vectors.cols(), n)))
  {
    throw std::invalid_argument("real_generalized_schur_selected_subspace received inconsistent Schur vectors");
  }
  if (decomposition.blocks.empty())
  {
    decomposition.blocks = detail::real_schur_blocks(decomposition.eigenvalues);
  }

  std::vector<bool> seen_blocks(decomposition.blocks.size(), false);
  std::vector<blas_int> select(n, 0);
  std::size_t expected_selected_dimension = 0;
  for (std::size_t const block_index : selected_blocks)
  {
    if (block_index >= decomposition.blocks.size())
    {
      throw std::invalid_argument("real_generalized_schur_selected_subspace received an out-of-range selected block");
    }
    if (seen_blocks[block_index])
    {
      throw std::invalid_argument("real_generalized_schur_selected_subspace received a duplicate selected block");
    }
    seen_blocks[block_index] = true;
    auto const& block = decomposition.blocks[block_index];
    if (block.begin + block.size > n)
    {
      throw std::invalid_argument("real_generalized_schur_selected_subspace received inconsistent block metadata");
    }
    for (std::size_t offset = 0; offset < block.size; ++offset)
    {
      select[block.begin + offset] = 1;
    }
    expected_selected_dimension += block.size;
  }

  detail::ColumnMajorLapackMatrix<Real> matrix_schur_form = std::move(decomposition.matrix_schur_form);
  detail::ColumnMajorLapackMatrix<Real> metric_schur_form = std::move(decomposition.metric_schur_form);
  detail::ColumnMajorLapackMatrix<Real> left_vectors =
      update_vectors ? std::move(decomposition.left_schur_vectors) : detail::ColumnMajorLapackMatrix<Real>(1, 1);
  detail::ColumnMajorLapackMatrix<Real> right_vectors =
      update_vectors ? std::move(decomposition.right_schur_vectors) : detail::ColumnMajorLapackMatrix<Real>(1, 1);
  std::vector<Real> alphar(n, Real{});
  std::vector<Real> alphai(n, Real{});
  std::vector<Real> beta(n, Real{});
  std::array<Real, 2> dif{Real{}, Real{}};
  blas_int const order = detail::checked_blas_int(n);
  blas_int const ldq = update_vectors ? order : 1;
  blas_int const ldz = update_vectors ? order : 1;
  blas_int const ijob = 4;
  blas_int selected_dimension = 0;
  Real pl{};
  Real pr{};

  Real work_query = Real{};
  blas_int iwork_query = 0;
  blas_int const query_workspace = -1;
  uni20::lapack::tgsen(ijob, update_vectors, update_vectors, select.data(), order, matrix_schur_form.data(), order,
                       metric_schur_form.data(), order, alphar.data(), alphai.data(), beta.data(), left_vectors.data(),
                       ldq, right_vectors.data(), ldz, selected_dimension, pl, pr, dif.data(), &work_query,
                       query_workspace, &iwork_query, query_workspace);

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
  blas_int const liwork = std::max<blas_int>(1, iwork_query);
  std::vector<Real> work(static_cast<std::size_t>(lwork), Real{});
  std::vector<blas_int> iwork(static_cast<std::size_t>(liwork), 0);
  uni20::lapack::tgsen(ijob, update_vectors, update_vectors, select.data(), order, matrix_schur_form.data(), order,
                       metric_schur_form.data(), order, alphar.data(), alphai.data(), beta.data(), left_vectors.data(),
                       ldq, right_vectors.data(), ldz, selected_dimension, pl, pr, dif.data(), work.data(), lwork,
                       iwork.data(), liwork);

  if (selected_dimension < 0 || static_cast<std::size_t>(selected_dimension) != expected_selected_dimension)
  {
    throw std::runtime_error("LAPACK tgsen returned an unexpected selected subspace dimension");
  }

  decomposition.matrix_schur_form = std::move(matrix_schur_form);
  decomposition.metric_schur_form = std::move(metric_schur_form);
  decomposition.left_schur_vectors =
      update_vectors ? std::move(left_vectors) : detail::ColumnMajorLapackMatrix<Real>(0, 0);
  decomposition.right_schur_vectors =
      update_vectors ? std::move(right_vectors) : detail::ColumnMajorLapackMatrix<Real>(0, 0);
  for (std::size_t i = 0; i < n; ++i)
  {
    decomposition.alpha[i] = uni20::complex<Real>{alphar[i], alphai[i]};
    decomposition.beta[i] = beta[i];
    decomposition.eigenvalues[i] = detail::generalized_eigenvalue(alphar[i], alphai[i], beta[i]);
  }
  decomposition.blocks = detail::real_schur_blocks(decomposition.eigenvalues);

  result.decomposition = std::move(decomposition);
  result.selected_dimension = static_cast<std::size_t>(selected_dimension);
  result.left_projection_lower_bound = pl;
  result.right_projection_lower_bound = pr;
  result.upper_deflating_subspace_separation = dif[0];
  result.lower_deflating_subspace_separation = dif[1];
  return result;
}

/// \brief Compute right eigenvectors of a generalized real Schur form through LAPACK `tgevc`.
/// \details The returned vectors solve `S v = lambda T v` for the generalized
///          Schur forms `S` and `T`. They are not multiplied by the right Schur
///          vectors, so callers that need eigenvectors of the original pencil
///          must backtransform them separately.
/// \tparam Real Real scalar type satisfying `uni20::LapackReal`.
/// \param decomposition Generalized real Schur decomposition whose forms are used.
/// \return Complex right eigenvectors of the generalized Schur form and the number computed.
template <uni20::LapackReal Real>
RealGeneralizedSchurRightEigenvectors<Real>
real_generalized_schur_right_eigenvectors(RealGeneralizedSchurDecomposition<Real> decomposition)
{
  if (!std::cmp_equal(decomposition.matrix_schur_form.rows(), decomposition.matrix_schur_form.cols()))
  {
    throw std::invalid_argument("real_generalized_schur_right_eigenvectors requires a square matrix Schur form");
  }
  if (!std::cmp_equal(decomposition.metric_schur_form.rows(), decomposition.metric_schur_form.cols()))
  {
    throw std::invalid_argument("real_generalized_schur_right_eigenvectors requires a square metric Schur form");
  }

  std::size_t const n = static_cast<std::size_t>(decomposition.matrix_schur_form.rows());
  RealGeneralizedSchurRightEigenvectors<Real> result;
  result.right_eigenvectors = detail::ColumnMajorLapackMatrix<uni20::complex<Real>>(n, n);
  if (n == 0)
  {
    return result;
  }
  if (!std::cmp_equal(decomposition.metric_schur_form.rows(), n) ||
      !std::cmp_equal(decomposition.matrix_schur_form.cols(), n) ||
      !std::cmp_equal(decomposition.metric_schur_form.cols(), n))
  {
    throw std::invalid_argument("real_generalized_schur_right_eigenvectors received inconsistent Schur form sizes");
  }
  if (decomposition.eigenvalues.size() != n)
  {
    throw std::invalid_argument("real_generalized_schur_right_eigenvectors received inconsistent eigenvalue data");
  }

  detail::ColumnMajorLapackMatrix<Real> matrix_schur_form = std::move(decomposition.matrix_schur_form);
  detail::ColumnMajorLapackMatrix<Real> metric_schur_form = std::move(decomposition.metric_schur_form);
  detail::ColumnMajorLapackMatrix<Real> left_vectors(1, 1);
  detail::ColumnMajorLapackMatrix<Real> right_vectors(n, n);
  std::vector<blas_int> select(n, 1);
  std::vector<Real> work(6 * n, Real{});
  blas_int const order = detail::checked_blas_int(n);
  blas_int computed_vectors = 0;
  char const side = 'R';
  char const howmny = 'A';
  uni20::lapack::tgevc(side, howmny, select.data(), order, matrix_schur_form.data(), order, metric_schur_form.data(),
                       order, left_vectors.data(), 1, right_vectors.data(), order, order, computed_vectors,
                       work.data());

  if (computed_vectors < 0 || static_cast<std::size_t>(computed_vectors) != n)
  {
    throw std::runtime_error("LAPACK tgevc returned an unexpected number of generalized Schur eigenvectors");
  }

  for (std::size_t col = 0; col < n;)
  {
    auto const eigenvalue = decomposition.eigenvalues[col];
    if (eigenvalue.imag() == Real{})
    {
      for (std::size_t row = 0; row < n; ++row)
      {
        result.right_eigenvectors[row, col] = uni20::complex<Real>{right_vectors[row, col], Real{}};
      }
      ++col;
      continue;
    }

    if (col + 1 >= n)
    {
      throw std::runtime_error("LAPACK tgevc returned an incomplete complex eigenvector pair");
    }
    auto const next_eigenvalue = decomposition.eigenvalues[col + 1];
    Real const scale = std::max(Real{1}, std::max(detail::adl_abs(eigenvalue), detail::adl_abs(next_eigenvalue)));
    if (detail::adl_abs(next_eigenvalue - std::conj(eigenvalue)) >
        Real{100} * uni20::numeric_limits<Real>::epsilon() * scale)
    {
      throw std::runtime_error("LAPACK tgevc returned a non-adjacent complex eigenvector pair");
    }

    for (std::size_t row = 0; row < n; ++row)
    {
      uni20::complex<Real> const positive_imaginary_vector{right_vectors[row, col], right_vectors[row, col + 1]};
      if (eigenvalue.imag() > Real{})
      {
        result.right_eigenvectors[row, col] = positive_imaginary_vector;
        result.right_eigenvectors[row, col + 1] = std::conj(positive_imaginary_vector);
      }
      else
      {
        result.right_eigenvectors[row, col] = std::conj(positive_imaginary_vector);
        result.right_eigenvectors[row, col + 1] = positive_imaginary_vector;
      }
    }
    col += 2;
  }

  result.computed_vectors = static_cast<std::size_t>(computed_vectors);
  return result;
}

/// \brief Estimate generalized Schur eigenvalue and eigenvector conditioning through LAPACK `tgsna`.
/// \details Computes left and right generalized Schur eigenvectors internally
///          with `tgevc`, then estimates reciprocal condition numbers for all
///          eigenpairs of the generalized Schur pencil.
/// \tparam Real Real scalar type satisfying `uni20::LapackReal`.
/// \param decomposition Generalized real Schur decomposition whose forms are used.
/// \return Reciprocal generalized eigenvalue and eigenvector condition estimates.
template <uni20::LapackReal Real>
RealGeneralizedSchurConditionEstimates<Real>
real_generalized_schur_condition_estimates(RealGeneralizedSchurDecomposition<Real> decomposition)
{
  if (!std::cmp_equal(decomposition.matrix_schur_form.rows(), decomposition.matrix_schur_form.cols()))
  {
    throw std::invalid_argument("real_generalized_schur_condition_estimates requires a square matrix Schur form");
  }
  if (!std::cmp_equal(decomposition.metric_schur_form.rows(), decomposition.metric_schur_form.cols()))
  {
    throw std::invalid_argument("real_generalized_schur_condition_estimates requires a square metric Schur form");
  }

  std::size_t const n = static_cast<std::size_t>(decomposition.matrix_schur_form.rows());
  RealGeneralizedSchurConditionEstimates<Real> result;
  result.reciprocal_eigenvalue_condition_numbers.resize(n);
  result.reciprocal_eigenvector_condition_numbers.resize(n);
  if (n == 0)
  {
    return result;
  }
  if (!std::cmp_equal(decomposition.metric_schur_form.rows(), n) ||
      !std::cmp_equal(decomposition.matrix_schur_form.cols(), n) ||
      !std::cmp_equal(decomposition.metric_schur_form.cols(), n))
  {
    throw std::invalid_argument("real_generalized_schur_condition_estimates received inconsistent Schur form sizes");
  }
  if (decomposition.eigenvalues.size() != n)
  {
    throw std::invalid_argument("real_generalized_schur_condition_estimates received inconsistent eigenvalue data");
  }

  detail::ColumnMajorLapackMatrix<Real> matrix_schur_form = std::move(decomposition.matrix_schur_form);
  detail::ColumnMajorLapackMatrix<Real> metric_schur_form = std::move(decomposition.metric_schur_form);
  detail::ColumnMajorLapackMatrix<Real> left_vectors(n, n);
  detail::ColumnMajorLapackMatrix<Real> right_vectors(n, n);
  std::vector<blas_int> select(n, 1);
  std::vector<Real> tgevc_work(6 * n, Real{});
  blas_int const order = detail::checked_blas_int(n);
  blas_int computed_vectors = 0;
  uni20::lapack::tgevc('B', 'A', select.data(), order, matrix_schur_form.data(), order, metric_schur_form.data(), order,
                       left_vectors.data(), order, right_vectors.data(), order, order, computed_vectors,
                       tgevc_work.data());

  if (computed_vectors < 0 || static_cast<std::size_t>(computed_vectors) != n)
  {
    throw std::runtime_error("LAPACK tgevc returned an unexpected number of generalized Schur eigenvectors");
  }

  blas_int computed_estimates = 0;
  blas_int const lwork = std::max<blas_int>(1, 2 * order * (order + 2) + 16);
  std::vector<Real> work(static_cast<std::size_t>(lwork), Real{});
  std::vector<blas_int> iwork(static_cast<std::size_t>(std::max<blas_int>(1, order + 6)), 0);
  uni20::lapack::tgsna('B', 'A', select.data(), order, matrix_schur_form.data(), order, metric_schur_form.data(), order,
                       left_vectors.data(), order, right_vectors.data(), order,
                       result.reciprocal_eigenvalue_condition_numbers.data(),
                       result.reciprocal_eigenvector_condition_numbers.data(), order, computed_estimates, work.data(),
                       lwork, iwork.data());

  if (computed_estimates < 0 || static_cast<std::size_t>(computed_estimates) != n)
  {
    throw std::runtime_error("LAPACK tgsna returned an unexpected number of condition estimates");
  }

  result.computed_estimates = static_cast<std::size_t>(computed_estimates);
  return result;
}

/// \brief Solve a dense real nonsymmetric eigensystem through LAPACK `geev`.
///
/// \details The input matrix is copied because LAPACK overwrites it. Eigenvalues
///          are returned in LAPACK order. If right eigenvectors are requested,
///          real eigenvectors are returned as complex vectors with zero
///          imaginary part, and complex conjugate pairs are unpacked from
///          LAPACK's adjacent real-vector representation.
/// \tparam Real Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real square matrix in column-major local storage.
/// \param compute_right_vectors Whether to compute right eigenvectors.
/// \return Complex eigenvalues and, optionally, right eigenvectors.

template <uni20::LapackReal Real>
RealNonsymmetricExpertEigensystem<Real>
real_nonsymmetric_expert_eigensystem(detail::ColumnMajorLapackMatrix<Real> matrix, bool compute_right_vectors)
{
  if (!std::cmp_equal(matrix.rows(), matrix.cols()))
  {
    throw std::invalid_argument("real_nonsymmetric_expert_eigensystem requires a square matrix");
  }

  std::size_t const n = static_cast<std::size_t>(matrix.rows());
  RealNonsymmetricExpertEigensystem<Real> result;
  result.eigenvalues.resize(n);
  result.right_eigenvectors = detail::ColumnMajorLapackMatrix<uni20::complex<Real>>(compute_right_vectors ? n : 0,
                                                                                    compute_right_vectors ? n : 0);
  result.reciprocal_eigenvalue_condition_numbers.resize(n);
  result.reciprocal_eigenvector_condition_numbers.resize(n);
  result.balance_scale.resize(n);
  if (n == 0)
  {
    return result;
  }

  blas_int const order = detail::checked_blas_int(n);
  std::vector<Real> wr(n, Real{});
  std::vector<Real> wi(n, Real{});
  detail::ColumnMajorLapackMatrix<Real> vl(n, n);
  detail::ColumnMajorLapackMatrix<Real> vr(n, n);
  blas_int const ldvl = order;
  blas_int const ldvr = order;
  char const balanc = 'B';
  char const jobvl = 'V';
  char const jobvr = 'V';
  char const sense = 'B';
  blas_int ilo = 0;
  blas_int ihi = 0;
  std::vector<blas_int> iwork(static_cast<std::size_t>(std::max<blas_int>(1, 2 * order - 2)), 0);

  Real work_query = Real{};
  blas_int const query_lwork = -1;
  uni20::lapack::geevx(balanc, jobvl, jobvr, sense, order, matrix.data(), order, wr.data(), wi.data(), vl.data(), ldvl,
                       vr.data(), ldvr, ilo, ihi, result.balance_scale.data(), result.balanced_matrix_norm,
                       result.reciprocal_eigenvalue_condition_numbers.data(),
                       result.reciprocal_eigenvector_condition_numbers.data(), &work_query, query_lwork, iwork.data());

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
  std::vector<Real> work(static_cast<std::size_t>(lwork), Real{});
  uni20::lapack::geevx(balanc, jobvl, jobvr, sense, order, matrix.data(), order, wr.data(), wi.data(), vl.data(), ldvl,
                       vr.data(), ldvr, ilo, ihi, result.balance_scale.data(), result.balanced_matrix_norm,
                       result.reciprocal_eigenvalue_condition_numbers.data(),
                       result.reciprocal_eigenvector_condition_numbers.data(), work.data(), lwork, iwork.data());

  if (ilo <= 0 || ihi < ilo || ihi > order)
  {
    throw std::runtime_error("LAPACK geevx returned an invalid balanced block interval");
  }
  result.balanced_first = static_cast<std::size_t>(ilo - 1);
  result.balanced_last_exclusive = static_cast<std::size_t>(ihi);

  for (std::size_t i = 0; i < n; ++i)
  {
    result.eigenvalues[i] = uni20::complex<Real>{wr[i], wi[i]};
  }

  if (!compute_right_vectors)
  {
    return result;
  }

  for (std::size_t col = 0; col < n; ++col)
  {
    if (wi[col] == Real{})
    {
      for (std::size_t row = 0; row < n; ++row)
      {
        result.right_eigenvectors[row, col] = uni20::complex<Real>{vr[row, col], Real{}};
      }
    }
    else if (wi[col] > Real{})
    {
      if (col + 1 >= n)
      {
        throw std::runtime_error("LAPACK geevx returned an incomplete complex conjugate eigenvector pair");
      }
      for (std::size_t row = 0; row < n; ++row)
      {
        uni20::complex<Real> const vector_value{vr[row, col], vr[row, col + 1]};
        result.right_eigenvectors[row, col] = vector_value;
        result.right_eigenvectors[row, col + 1] = std::conj(vector_value);
      }
      ++col;
    }
  }

  return result;
}

/// \brief Solve a dense real generalized nonsymmetric eigensystem through LAPACK `ggev`.
///
/// \details Solves the matrix pencil `A * x = lambda * B * x`. LAPACK reports
///          generalized eigenvalues as projective pairs `alpha / beta`; this
///          wrapper preserves both `alpha` and `beta` and also stores the
///          ratio in `eigenvalues`. If right eigenvectors are requested, real
///          eigenvectors are returned as complex vectors with zero imaginary
///          part, and complex conjugate pairs are unpacked from LAPACK's
///          adjacent real-vector representation.
/// \tparam Real Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real square matrix `A` in column-major local storage.
/// \param metric Real square matrix `B` in column-major local storage.
/// \param compute_right_vectors Whether to compute right eigenvectors.
/// \return Projective eigenvalue data and, optionally, right eigenvectors.
template <uni20::LapackReal Real>
RealGeneralizedNonsymmetricEigensystem<Real>
real_generalized_nonsymmetric_eigensystem(detail::ColumnMajorLapackMatrix<Real> matrix,
                                          detail::ColumnMajorLapackMatrix<Real> metric, bool compute_right_vectors)
{
  if (!std::cmp_equal(matrix.rows(), matrix.cols()))
  {
    throw std::invalid_argument("real_generalized_nonsymmetric_eigensystem requires a square matrix");
  }
  if (!std::cmp_equal(metric.rows(), metric.cols()))
  {
    throw std::invalid_argument("real_generalized_nonsymmetric_eigensystem requires a square metric matrix");
  }
  if (!std::cmp_equal(matrix.rows(), metric.rows()))
  {
    throw std::invalid_argument("real_generalized_nonsymmetric_eigensystem received incompatible matrix sizes");
  }

  std::size_t const n = static_cast<std::size_t>(matrix.rows());
  RealGeneralizedNonsymmetricEigensystem<Real> result;
  result.alpha.resize(n);
  result.beta.resize(n);
  result.eigenvalues.resize(n);
  result.right_eigenvectors = detail::ColumnMajorLapackMatrix<uni20::complex<Real>>(compute_right_vectors ? n : 0,
                                                                                    compute_right_vectors ? n : 0);
  if (n == 0)
  {
    return result;
  }

  blas_int const order = detail::checked_blas_int(n);
  std::vector<Real> alphar(n, Real{});
  std::vector<Real> alphai(n, Real{});
  std::vector<Real> beta(n, Real{});
  std::vector<Real> vl(1, Real{});
  detail::ColumnMajorLapackMatrix<Real> vr(compute_right_vectors ? n : 1, compute_right_vectors ? n : 1);
  blas_int const ldvl = 1;
  blas_int const ldvr = compute_right_vectors ? order : 1;
  char const jobvl = 'N';
  char const jobvr = compute_right_vectors ? 'V' : 'N';

  Real work_query = Real{};
  blas_int const query_lwork = -1;
  uni20::lapack::ggev(jobvl, jobvr, order, matrix.data(), order, metric.data(), order, alphar.data(), alphai.data(),
                      beta.data(), vl.data(), ldvl, vr.data(), ldvr, &work_query, query_lwork);

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
  std::vector<Real> work(static_cast<std::size_t>(lwork), Real{});
  uni20::lapack::ggev(jobvl, jobvr, order, matrix.data(), order, metric.data(), order, alphar.data(), alphai.data(),
                      beta.data(), vl.data(), ldvl, vr.data(), ldvr, work.data(), lwork);

  for (std::size_t i = 0; i < n; ++i)
  {
    result.alpha[i] = uni20::complex<Real>{alphar[i], alphai[i]};
    result.beta[i] = beta[i];
    result.eigenvalues[i] = detail::generalized_eigenvalue(alphar[i], alphai[i], beta[i]);
  }

  if (!compute_right_vectors)
  {
    return result;
  }

  for (std::size_t col = 0; col < n; ++col)
  {
    if (alphai[col] == Real{})
    {
      for (std::size_t row = 0; row < n; ++row)
      {
        result.right_eigenvectors[row, col] = uni20::complex<Real>{vr[row, col], Real{}};
      }
    }
    else if (alphai[col] > Real{})
    {
      if (col + 1 >= n)
      {
        throw std::runtime_error("LAPACK ggev returned an incomplete complex conjugate eigenvector pair");
      }
      for (std::size_t row = 0; row < n; ++row)
      {
        uni20::complex<Real> const vector_value{vr[row, col], vr[row, col + 1]};
        result.right_eigenvectors[row, col] = vector_value;
        result.right_eigenvectors[row, col + 1] = std::conj(vector_value);
      }
      ++col;
    }
  }

  return result;
}

/// \brief Solve a dense real generalized nonsymmetric eigensystem through LAPACK `ggevx`.
///
/// \details Solves the matrix pencil `A * x = lambda * B * x` with balancing
///          (`BALANC='B'`) and reciprocal condition estimates for both
///          eigenvalues and right eigenvectors (`SENSE='B'`). It computes left
///          and right eigenvectors internally because LAPACK needs them for
///          these estimates. The wrapper preserves LAPACK's projective
///          `alpha / beta` eigenvalue representation and stores finite ratios
///          in `eigenvalues`.
/// \tparam Real Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real square matrix `A` in column-major local storage.
/// \param metric Real square matrix `B` in column-major local storage.
/// \param compute_right_vectors Whether to return right eigenvectors.
/// \return Projective eigenvalue data, optional right eigenvectors, balancing
///         data, and reciprocal condition estimates.
template <uni20::LapackReal Real>
RealGeneralizedNonsymmetricExpertEigensystem<Real>
real_generalized_nonsymmetric_expert_eigensystem(detail::ColumnMajorLapackMatrix<Real> matrix,
                                                 detail::ColumnMajorLapackMatrix<Real> metric,
                                                 bool compute_right_vectors)
{
  if (!std::cmp_equal(matrix.rows(), matrix.cols()))
  {
    throw std::invalid_argument("real_generalized_nonsymmetric_expert_eigensystem requires a square matrix");
  }
  if (!std::cmp_equal(metric.rows(), metric.cols()))
  {
    throw std::invalid_argument("real_generalized_nonsymmetric_expert_eigensystem requires a square metric matrix");
  }
  if (!std::cmp_equal(matrix.rows(), metric.rows()))
  {
    throw std::invalid_argument("real_generalized_nonsymmetric_expert_eigensystem received incompatible matrix sizes");
  }

  std::size_t const n = static_cast<std::size_t>(matrix.rows());
  RealGeneralizedNonsymmetricExpertEigensystem<Real> result;
  result.alpha.resize(n);
  result.beta.resize(n);
  result.eigenvalues.resize(n);
  result.right_eigenvectors = detail::ColumnMajorLapackMatrix<uni20::complex<Real>>(compute_right_vectors ? n : 0,
                                                                                    compute_right_vectors ? n : 0);
  result.reciprocal_eigenvalue_condition_numbers.resize(n);
  result.reciprocal_eigenvector_condition_numbers.resize(n);
  result.left_balance_scale.resize(n);
  result.right_balance_scale.resize(n);
  if (n == 0)
  {
    return result;
  }

  blas_int const order = detail::checked_blas_int(n);
  std::vector<Real> alphar(n, Real{});
  std::vector<Real> alphai(n, Real{});
  std::vector<Real> beta(n, Real{});
  detail::ColumnMajorLapackMatrix<Real> vl(n, n);
  detail::ColumnMajorLapackMatrix<Real> vr(n, n);
  blas_int const ldvl = order;
  blas_int const ldvr = order;
  char const balanc = 'B';
  char const jobvl = 'V';
  char const jobvr = 'V';
  char const sense = 'B';
  blas_int ilo = 0;
  blas_int ihi = 0;
  std::vector<blas_int> iwork(static_cast<std::size_t>(std::max<blas_int>(1, order + 6)), 0);
  std::vector<blas_int> bwork(static_cast<std::size_t>(std::max<blas_int>(1, order)), 0);

  Real work_query = Real{};
  blas_int const query_lwork = -1;
  uni20::lapack::ggevx(balanc, jobvl, jobvr, sense, order, matrix.data(), order, metric.data(), order, alphar.data(),
                       alphai.data(), beta.data(), vl.data(), ldvl, vr.data(), ldvr, ilo, ihi,
                       result.left_balance_scale.data(), result.right_balance_scale.data(), result.balanced_matrix_norm,
                       result.balanced_metric_norm, result.reciprocal_eigenvalue_condition_numbers.data(),
                       result.reciprocal_eigenvector_condition_numbers.data(), &work_query, query_lwork, iwork.data(),
                       bwork.data());

  blas_int const lwork = std::max<blas_int>(1, static_cast<blas_int>(work_query));
  std::vector<Real> work(static_cast<std::size_t>(lwork), Real{});
  uni20::lapack::ggevx(balanc, jobvl, jobvr, sense, order, matrix.data(), order, metric.data(), order, alphar.data(),
                       alphai.data(), beta.data(), vl.data(), ldvl, vr.data(), ldvr, ilo, ihi,
                       result.left_balance_scale.data(), result.right_balance_scale.data(), result.balanced_matrix_norm,
                       result.balanced_metric_norm, result.reciprocal_eigenvalue_condition_numbers.data(),
                       result.reciprocal_eigenvector_condition_numbers.data(), work.data(), lwork, iwork.data(),
                       bwork.data());

  if (ilo <= 0 || ihi < ilo || ihi > order)
  {
    throw std::runtime_error("LAPACK ggevx returned an invalid balanced block interval");
  }
  result.balanced_first = static_cast<std::size_t>(ilo - 1);
  result.balanced_last_exclusive = static_cast<std::size_t>(ihi);

  for (std::size_t i = 0; i < n; ++i)
  {
    result.alpha[i] = uni20::complex<Real>{alphar[i], alphai[i]};
    result.beta[i] = beta[i];
    result.eigenvalues[i] = detail::generalized_eigenvalue(alphar[i], alphai[i], beta[i]);
  }

  if (!compute_right_vectors)
  {
    return result;
  }

  for (std::size_t col = 0; col < n; ++col)
  {
    if (alphai[col] == Real{})
    {
      for (std::size_t row = 0; row < n; ++row)
      {
        result.right_eigenvectors[row, col] = uni20::complex<Real>{vr[row, col], Real{}};
      }
    }
    else if (alphai[col] > Real{})
    {
      if (col + 1 >= n)
      {
        throw std::runtime_error("LAPACK ggevx returned an incomplete complex conjugate eigenvector pair");
      }
      for (std::size_t row = 0; row < n; ++row)
      {
        uni20::complex<Real> const vector_value{vr[row, col], vr[row, col + 1]};
        result.right_eigenvectors[row, col] = vector_value;
        result.right_eigenvectors[row, col + 1] = std::conj(vector_value);
      }
      ++col;
    }
  }

  return result;
}

} // namespace uni20::krylov
