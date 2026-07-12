#pragma once

#include <uni20/backend/blas/blas_int.hpp>
#include <uni20/backend/lapack/lapack.hpp>
#include <uni20/common/mdspan.hpp>
#include <uni20/config.hpp>
#include <uni20/core/scalar_concepts.hpp>
#include <uni20/core/scalar_traits.hpp>
#include <uni20/krylov/detail_math.hpp>
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

    Scalar& operator[](std::size_t row, std::size_t col) { return data_[col + row * cols_]; }

    Scalar const& operator[](std::size_t row, std::size_t col) const { return data_[col + row * cols_]; }

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
      result[row, col] = matrix[row, col];
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
      result[row, col] = matrix[row, col];
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

using uni20::blas::checked_blas_int;

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

template <typename Scalar> constexpr Scalar conjugate_if_complex(Scalar const& value)
{
  if constexpr (uni20::is_complex_v<Scalar>)
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
} // namespace detail

/// \brief Copy a local dense vector.
/// \tparam Scalar Element type.
/// \param destination Destination vector overwritten with \p source.
/// \param source Source vector.
template <typename Scalar> void copy(std::span<Scalar> destination, std::span<Scalar const> source)
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
/// \param y Destination vector updated as `y += alpha * x`.
/// \param alpha Scale factor.
/// \param x Source vector.
template <typename Scalar> void axpy(std::span<Scalar> y, Scalar const& alpha, std::span<Scalar const> x)
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
    Real const magnitude = static_cast<Real>(detail::adl_abs(value));
    sum += magnitude * magnitude;
  }
  return detail::adl_sqrt(sum);
}

/// \brief Copy a selected region of a local dense matrix.
/// \tparam Scalar Element type.
/// \param destination Destination matrix with the same shape as \p source.
/// \param source Source matrix.
/// \param fill Region to copy.
template <typename Scalar> void lacpy(Matrix<Scalar>& destination, Matrix<Scalar> const& source, MatrixFill fill)
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
        destination[row, col] = source[row, col];
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
        matrix[row, col] = (row == col) ? diagonal : off_diagonal;
      }
    }
  }
}

/// \brief Compute a dense matrix-vector product.
/// \tparam Scalar Element type.
/// \param y Output vector updated as `y = alpha * op(matrix) * x + beta * y`.
/// \param alpha Scale factor for the matrix-vector product.
/// \param matrix Input matrix.
/// \param x Input vector.
/// \param beta Scale factor for the original \p y value.
/// \param transpose Matrix operation applied before multiplication.
template <typename Scalar>
void gemv(std::span<Scalar> y, Scalar const& alpha, Matrix<Scalar> const& matrix, std::span<Scalar const> x,
          Scalar const& beta, MatrixTranspose transpose = MatrixTranspose::None)
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
        y[row] += factor * matrix[row, col];
      }
    }
    return;
  }

  for (std::size_t col = 0; col < matrix.cols(); ++col)
  {
    Scalar sum{};
    for (std::size_t row = 0; row < matrix.rows(); ++row)
    {
      Scalar entry = matrix[row, col];
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
/// \param matrix Matrix updated as `matrix += alpha * x * y^T`.
/// \param alpha Scale factor.
/// \param x Left vector.
/// \param y Right vector.
template <typename Scalar>
void geru(Matrix<Scalar>& matrix, Scalar const& alpha, std::span<Scalar const> x, std::span<Scalar const> y)
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
      matrix[row, col] += x[row] * factor;
    }
  }
}

/// \brief Apply a conjugated dense rank-one update.
/// \tparam Scalar Element type.
/// \param matrix Matrix updated as `matrix += alpha * x * conj(y)^T`.
/// \param alpha Scale factor.
/// \param x Left vector.
/// \param y Right vector, conjugated elementwise for complex scalar types.
template <typename Scalar>
void gerc(Matrix<Scalar>& matrix, Scalar const& alpha, std::span<Scalar const> x, std::span<Scalar const> y)
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
      matrix[row, col] += x[row] * factor;
    }
  }
}

} // namespace uni20::krylov
