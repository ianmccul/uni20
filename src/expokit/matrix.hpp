#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace EXPOKIT
{

/// \brief Minimal dense matrix implementation used by the EXPOKIT adapters.
/// \tparam T Element type stored in the matrix.
template <typename T>
class Matrix
{
public:
    Matrix() = default;

    Matrix(std::size_t rows, std::size_t cols)
        : rows_(rows)
        , cols_(cols)
        , data_(rows * cols)
    {
    }

    [[nodiscard]] std::size_t rows() const noexcept
    {
        return rows_;
    }

    [[nodiscard]] std::size_t cols() const noexcept
    {
        return cols_;
    }

    [[nodiscard]] std::size_t size() const noexcept
    {
        return data_.size();
    }

    T& operator()(std::size_t row, std::size_t col)
    {
        return data_[row * cols_ + col];
    }

    T const& operator()(std::size_t row, std::size_t col) const
    {
        return data_[row * cols_ + col];
    }

    T* data() noexcept
    {
        return data_.data();
    }

    T const* data() const noexcept
    {
        return data_.data();
    }

    void swap(Matrix& other) noexcept
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
template <typename T>
Matrix<T> make_identity(std::size_t n)
{
    Matrix<T> result(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            result(i, j) = (i == j) ? T{1} : T{};
        }
    }
    return result;
}

/// \brief Multiply matrix \p lhs by matrix \p rhs.
/// \tparam T Element type of the matrices.
/// \param lhs Left-hand operand.
/// \param rhs Right-hand operand.
/// \return The matrix product \p lhs * \p rhs.
template <typename T>
Matrix<T> multiply(Matrix<T> const& lhs, Matrix<T> const& rhs)
{
    if (lhs.cols() != rhs.rows()) {
        throw std::invalid_argument("matrix dimensions do not agree for multiplication");
    }

    Matrix<T> result(lhs.rows(), rhs.cols());
    for (std::size_t i = 0; i < lhs.rows(); ++i) {
        for (std::size_t j = 0; j < rhs.cols(); ++j) {
            T value{};
            for (std::size_t k = 0; k < lhs.cols(); ++k) {
                value += lhs(i, k) * rhs(k, j);
            }
            result(i, j) = value;
        }
    }
    return result;
}

/// \brief Add matrices \p lhs and \p rhs.
/// \tparam T Element type of the matrices.
/// \param lhs Left-hand operand.
/// \param rhs Right-hand operand.
/// \return The element-wise sum of \p lhs and \p rhs.
template <typename T>
Matrix<T> add(Matrix<T> const& lhs, Matrix<T> const& rhs)
{
    if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) {
        throw std::invalid_argument("matrix dimensions do not agree for addition");
    }

    Matrix<T> result(lhs.rows(), lhs.cols());
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        result.data()[i] = lhs.data()[i] + rhs.data()[i];
    }
    return result;
}

/// \brief Subtract matrix \p rhs from \p lhs.
/// \tparam T Element type of the matrices.
/// \param lhs Left-hand operand.
/// \param rhs Right-hand operand.
/// \return The element-wise difference of \p lhs and \p rhs.
template <typename T>
Matrix<T> subtract(Matrix<T> const& lhs, Matrix<T> const& rhs)
{
    if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) {
        throw std::invalid_argument("matrix dimensions do not agree for subtraction");
    }

    Matrix<T> result(lhs.rows(), lhs.cols());
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        result.data()[i] = lhs.data()[i] - rhs.data()[i];
    }
    return result;
}

/// \brief Multiply matrix \p mat by scalar \p scalar.
/// \tparam T Element type of the matrix.
/// \param mat Matrix to scale.
/// \param scalar Scalar factor.
/// \return A matrix where each element is \p mat(i, j) * \p scalar.
template <typename T, typename Scalar>
Matrix<T> scale(Matrix<T> const& mat, Scalar const& scalar)
{
    Matrix<T> result(mat.rows(), mat.cols());
    for (std::size_t i = 0; i < mat.size(); ++i) {
        result.data()[i] = mat.data()[i] * scalar;
    }
    return result;
}

/// \brief Compute the 1-norm (maximum absolute column sum) of a matrix.
/// \tparam T Element type of the matrix.
/// \param mat Input matrix.
/// \return The induced matrix 1-norm of \p mat.
template <typename T>
double matrix_one_norm(Matrix<T> const& mat)
{
    double result = 0.0;
    for (std::size_t j = 0; j < mat.cols(); ++j) {
        double column_sum = 0.0;
        for (std::size_t i = 0; i < mat.rows(); ++i) {
            column_sum += std::abs(mat(i, j));
        }
        result = std::max(result, column_sum);
    }
    return result;
}

/// \brief Raise a square matrix to an integer power.
/// \tparam T Element type of the matrix.
/// \param mat Input matrix to be exponentiated.
/// \param power Non-negative integer exponent.
/// \return Matrix power \f$mat^{\text{power}}\f$.
/// \throws std::invalid_argument if \p mat is not square.
template <typename T>
Matrix<T> matrix_power(Matrix<T> const& mat, unsigned int power)
{
    if (mat.rows() != mat.cols()) {
        throw std::invalid_argument("matrix_power requires a square matrix");
    }

    if (power == 0U) {
        return make_identity<T>(mat.rows());
    }

    Matrix<T> result = make_identity<T>(mat.rows());
    Matrix<T> base = mat;
    unsigned int exponent = power;
    while (exponent > 0U) {
        if ((exponent & 1U) != 0U) {
            result = multiply(result, base);
        }
        exponent >>= 1U;
        if (exponent != 0U) {
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
template <typename T>
double matrix_one_norm_power(Matrix<T> const& mat, unsigned int power)
{
    Matrix<T> powered = matrix_power(mat, power);
    return matrix_one_norm(powered);
}

/// \brief Swap two rows in a matrix.
/// \tparam T Element type of the matrix.
/// \param mat Matrix whose rows will be swapped.
/// \param lhs First row index.
/// \param rhs Second row index.
template <typename T>
void swap_rows(Matrix<T>& mat, std::size_t lhs, std::size_t rhs)
{
    if (lhs == rhs) {
        return;
    }
    for (std::size_t j = 0; j < mat.cols(); ++j) {
        std::swap(mat(lhs, j), mat(rhs, j));
    }
}

/// \brief Solve the linear system A * X = B using Gaussian elimination with partial pivoting.
/// \tparam T Element type.
/// \param A Coefficient matrix (will be copied internally).
/// \param B Right-hand side matrix (will be copied internally).
/// \return Solution matrix X satisfying A * X = B.
/// \throws std::runtime_error if the system is singular.
template <typename T>
Matrix<T> solve_linear_system(Matrix<T> A, Matrix<T> B)
{
    if (A.rows() != A.cols() || A.rows() != B.rows()) {
        throw std::invalid_argument("solve_linear_system requires square coefficient matrix");
    }

    std::size_t n = A.rows();
    std::size_t nrhs = B.cols();

    for (std::size_t k = 0; k < n; ++k) {
        std::size_t pivot_row = k;
        double pivot_value = std::abs(A(k, k));
        for (std::size_t i = k + 1; i < n; ++i) {
            double candidate = std::abs(A(i, k));
            if (candidate > pivot_value) {
                pivot_value = candidate;
                pivot_row = i;
            }
        }

        if (pivot_value == 0.0) {
            throw std::runtime_error("singular matrix in solve_linear_system");
        }

        swap_rows(A, k, pivot_row);
        swap_rows(B, k, pivot_row);

        T const pivot = A(k, k);
        for (std::size_t i = k + 1; i < n; ++i) {
            T const factor = A(i, k) / pivot;
            if (factor == T{}) {
                continue;
            }
            A(i, k) = T{};
            for (std::size_t j = k + 1; j < n; ++j) {
                A(i, j) -= factor * A(k, j);
            }
            for (std::size_t j = 0; j < nrhs; ++j) {
                B(i, j) -= factor * B(k, j);
            }
        }
    }

    Matrix<T> X = B;
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
        T const pivot = A(static_cast<std::size_t>(i), static_cast<std::size_t>(i));
        for (std::size_t j = 0; j < nrhs; ++j) {
            T value = X(static_cast<std::size_t>(i), j);
            for (std::size_t k = static_cast<std::size_t>(i) + 1; k < n; ++k) {
                value -= A(static_cast<std::size_t>(i), k) * X(k, j);
            }
            X(static_cast<std::size_t>(i), j) = value / pivot;
        }
    }

    return X;
}

} // namespace EXPOKIT

