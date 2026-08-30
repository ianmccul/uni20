#pragma once

#include <uni20/core/scalar_traits.hpp>
#include <uni20/linalg/backends/cpu/detail/compensated_sum.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace uni20::linalg::backends::cpu
{

/// \brief Minimal column-major dense matrix implementation used by the CPU dense linear algebra routines.
/// \tparam T Element type stored in the matrix.
template <typename T> class DenseMatrix {
  public:
    DenseMatrix() = default;

    DenseMatrix(std::size_t rows, std::size_t cols) : rows_(rows), cols_(cols), data_(rows * cols) {}

    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }

    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }

    [[nodiscard]] std::size_t size() const noexcept { return data_.size(); }

    T& operator[](std::size_t row, std::size_t col) { return data_[row + col * rows_]; }

    T const& operator[](std::size_t row, std::size_t col) const { return data_[row + col * rows_]; }

    [[nodiscard]] T* data() noexcept { return data_.data(); }

    [[nodiscard]] T const* data() const noexcept { return data_.data(); }

    void swap(DenseMatrix& other) noexcept
    {
      std::swap(rows_, other.rows_);
      std::swap(cols_, other.cols_);
      data_.swap(other.data_);
    }

  private:
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<T> data_;
};

/// \brief Create an identity matrix of size \p n.
/// \tparam T Element type used for the identity matrix.
/// \param n Dimension of the matrix.
/// \return An \p n-by-\p n identity matrix.
template <typename T> DenseMatrix<T> make_identity(std::size_t n)
{
  DenseMatrix<T> result(n, n);
  for (std::size_t i = 0; i < n; ++i)
  {
    for (std::size_t j = 0; j < n; ++j)
    {
      result[i, j] = (i == j) ? T{1} : T{};
    }
  }
  return result;
}

/// \brief Multiply matrix \p lhs by matrix \p rhs.
/// \tparam T Element type of the matrices.
/// \param lhs Left-hand operand.
/// \param rhs Right-hand operand.
/// \return The matrix product \p lhs * \p rhs.
template <typename T> DenseMatrix<T> multiply(DenseMatrix<T> const& lhs, DenseMatrix<T> const& rhs)
{
  if (lhs.cols() != rhs.rows())
  {
    throw std::invalid_argument("matrix dimensions do not agree for multiplication");
  }

  DenseMatrix<T> result(lhs.rows(), rhs.cols());
  std::fill_n(result.data(), result.size(), T{});
  for (std::size_t i = 0; i < lhs.rows(); ++i)
  {
    for (std::size_t k = 0; k < lhs.cols(); ++k)
    {
      T const factor = lhs[i, k];
      if (factor == T{})
      {
        continue;
      }
      for (std::size_t j = 0; j < rhs.cols(); ++j)
      {
        result[i, j] += factor * rhs[k, j];
      }
    }
  }
  return result;
}

/// \brief Add matrices \p lhs and \p rhs.
/// \tparam T Element type of the matrices.
/// \param lhs Left-hand operand.
/// \param rhs Right-hand operand.
/// \return The element-wise sum of \p lhs and \p rhs.
template <typename T> DenseMatrix<T> add(DenseMatrix<T> const& lhs, DenseMatrix<T> const& rhs)
{
  if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols())
  {
    throw std::invalid_argument("matrix dimensions do not agree for addition");
  }

  DenseMatrix<T> result(lhs.rows(), lhs.cols());
  for (std::size_t i = 0; i < lhs.size(); ++i)
  {
    result.data()[i] = lhs.data()[i] + rhs.data()[i];
  }
  return result;
}

/// \brief Subtract matrix \p rhs from \p lhs.
/// \tparam T Element type of the matrices.
/// \param lhs Left-hand operand.
/// \param rhs Right-hand operand.
/// \return The element-wise difference of \p lhs and \p rhs.
template <typename T> DenseMatrix<T> subtract(DenseMatrix<T> const& lhs, DenseMatrix<T> const& rhs)
{
  if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols())
  {
    throw std::invalid_argument("matrix dimensions do not agree for subtraction");
  }

  DenseMatrix<T> result(lhs.rows(), lhs.cols());
  for (std::size_t i = 0; i < lhs.size(); ++i)
  {
    result.data()[i] = lhs.data()[i] - rhs.data()[i];
  }
  return result;
}

/// \brief Multiply matrix \p mat by scalar \p scalar.
/// \tparam T Element type of the matrix.
/// \param mat DenseMatrix to scale.
/// \param scalar Scalar factor.
/// \return A matrix where each element is \p mat[i, j] * \p scalar.
template <typename T, typename Scalar> DenseMatrix<T> scale(DenseMatrix<T> const& mat, Scalar const& scalar)
{
  DenseMatrix<T> result(mat.rows(), mat.cols());
  for (std::size_t i = 0; i < mat.size(); ++i)
  {
    result.data()[i] = mat.data()[i] * scalar;
  }
  return result;
}

/// \brief Multiply an owned matrix by scalar \p scalar, reusing the existing storage.
/// \tparam T Element type of the matrix.
/// \tparam Scalar Scalar factor type.
/// \param mat Owned DenseMatrix to scale.
/// \param scalar Scalar factor.
/// \return The scaled matrix, moved from \p mat.
template <typename T, typename Scalar> DenseMatrix<T> scale(DenseMatrix<T>&& mat, Scalar const& scalar)
{
  for (std::size_t i = 0; i < mat.size(); ++i)
  {
    mat.data()[i] *= scalar;
  }
  return std::move(mat);
}

/// \brief Compute the 1-norm (maximum absolute column sum) of a matrix.
/// \tparam T Element type of the matrix.
/// \param mat Input matrix.
/// \return The induced matrix 1-norm of \p mat.
template <typename T> uni20::make_real_t<T> matrix_one_norm(DenseMatrix<T> const& mat)
{
  using Real = uni20::make_real_t<T>;
  using std::abs;

  Real result = Real{};
  for (std::size_t j = 0; j < mat.cols(); ++j)
  {
    detail::CompensatedRealSum<Real> column_sum;
    for (std::size_t i = 0; i < mat.rows(); ++i)
    {
      column_sum.add(abs(mat[i, j]));
    }
    result = std::max(result, column_sum.value());
  }
  return result;
}

/// \brief Raise a square matrix to an integer power.
/// \tparam T Element type of the matrix.
/// \param mat Input matrix to be exponentiated.
/// \param power Non-negative integer exponent.
/// \return DenseMatrix power \f$mat^{\text{power}}\f$.
/// \throws std::invalid_argument if \p mat is not square.
template <typename T> DenseMatrix<T> matrix_power(DenseMatrix<T> const& mat, unsigned int power)
{
  if (mat.rows() != mat.cols())
  {
    throw std::invalid_argument("matrix_power requires a square matrix");
  }

  if (power == 0U)
  {
    return make_identity<T>(mat.rows());
  }

  DenseMatrix<T> result = make_identity<T>(mat.rows());
  DenseMatrix<T> base = mat;
  unsigned int exponent = power;
  while (exponent > 0U)
  {
    if ((exponent & 1U) != 0U)
    {
      result = multiply(result, base);
    }
    exponent >>= 1U;
    if (exponent != 0U)
    {
      base = multiply(base, base);
    }
  }

  return result;
}

/// \brief Compute the matrix 1-norm of a matrix power.
/// \tparam T Element type of the matrix.
/// \param mat Input matrix.
/// \param power Non-negative integer exponent.
/// \return The 1-norm of \f$mat^{\text{power}}\f$.
template <typename T> uni20::make_real_t<T> matrix_one_norm_power(DenseMatrix<T> const& mat, unsigned int power)
{
  DenseMatrix<T> powered = matrix_power(mat, power);
  return matrix_one_norm(powered);
}

/// \brief Swap two rows in a matrix.
/// \tparam T Element type of the matrix.
/// \param mat DenseMatrix whose rows will be swapped.
/// \param lhs First row index.
/// \param rhs Second row index.
template <typename T> void swap_rows(DenseMatrix<T>& mat, std::size_t lhs, std::size_t rhs)
{
  if (lhs == rhs)
  {
    return;
  }
  for (std::size_t j = 0; j < mat.cols(); ++j)
  {
    std::swap(mat[lhs, j], mat[rhs, j]);
  }
}

/// \brief Solve the linear system A * X = B using Gaussian elimination with partial pivoting.
/// \tparam T Element type.
/// \param A Coefficient matrix copied or moved into the solver and overwritten during elimination.
/// \param B Right-hand side matrix copied or moved into the solver and overwritten during elimination.
/// \return Solution matrix X satisfying A * X = B.
/// \throws std::runtime_error if the system is singular.
template <typename T> DenseMatrix<T> solve_linear_system(DenseMatrix<T> A, DenseMatrix<T> B)
{
  if (A.rows() != A.cols() || A.rows() != B.rows())
  {
    throw std::invalid_argument("solve_linear_system requires square coefficient matrix");
  }

  std::size_t n = A.rows();
  std::size_t nrhs = B.cols();
  using Real = uni20::make_real_t<T>;
  using std::abs;

  for (std::size_t k = 0; k < n; ++k)
  {
    std::size_t pivot_row = k;
    Real pivot_value = abs(A[k, k]);
    for (std::size_t i = k + 1; i < n; ++i)
    {
      Real candidate = abs(A[i, k]);
      if (candidate > pivot_value)
      {
        pivot_value = candidate;
        pivot_row = i;
      }
    }

    if (pivot_value == Real{})
    {
      throw std::runtime_error("singular matrix in solve_linear_system");
    }

    swap_rows(A, k, pivot_row);
    swap_rows(B, k, pivot_row);

    T const pivot = A[k, k];
    for (std::size_t i = k + 1; i < n; ++i)
    {
      T const factor = A[i, k] / pivot;
      if (factor == T{})
      {
        continue;
      }
      A[i, k] = T{};
      for (std::size_t j = k + 1; j < n; ++j)
      {
        A[i, j] -= factor * A[k, j];
      }
      for (std::size_t j = 0; j < nrhs; ++j)
      {
        B[i, j] -= factor * B[k, j];
      }
    }
  }

  for (std::size_t i = n; i-- > 0;)
  {
    T const pivot = A[i, i];
    for (std::size_t j = 0; j < nrhs; ++j)
    {
      T value = B[i, j];
      for (std::size_t k = i + 1; k < n; ++k)
      {
        value -= A[i, k] * B[k, j];
      }
      B[i, j] = value / pivot;
    }
  }

  return B;
}

} // namespace uni20::linalg::backends::cpu
