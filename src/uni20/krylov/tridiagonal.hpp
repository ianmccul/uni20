#pragma once

#include <uni20/common/trace.hpp>
#include <uni20/core/scalar_concepts.hpp>

#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace uni20::krylov
{

namespace detail
{
inline void check_symmetric_tridiagonal_shape(std::size_t diagonal_size, std::size_t offdiagonal_size)
{
  if (diagonal_size == 0)
  {
    PRECONDITION(offdiagonal_size == 0, "invalid symmetric tridiagonal shape", diagonal_size, offdiagonal_size);
    return;
  }

  PRECONDITION(offdiagonal_size + 1 == diagonal_size, "invalid symmetric tridiagonal shape", diagonal_size,
               offdiagonal_size);
}
} // namespace detail

/// \brief Non-owning read-only view of a real symmetric tridiagonal matrix.
/// \tparam Real Real floating-point scalar type.
///
/// The matrix is represented by its diagonal `d[0..n)` and offdiagonal
/// `e[0..n-1)`, where `e[i]` is both element `(i, i + 1)` and `(i + 1, i)`.
/// This is the natural projected matrix format produced by Hermitian Lanczos,
/// including complex-Hermitian problems, whose Lanczos projection is real
/// symmetric tridiagonal.
template <uni20::Real Real> struct symmetric_tridiagonal_view
{
    std::span<Real const> const diagonal;
    std::span<Real const> const offdiagonal;

    symmetric_tridiagonal_view(std::span<Real const> diagonal_in, std::span<Real const> offdiagonal_in)
        : diagonal(diagonal_in), offdiagonal(offdiagonal_in)
    {
      detail::check_symmetric_tridiagonal_shape(diagonal.size(), offdiagonal.size());
    }

    [[nodiscard]] constexpr std::size_t size() const noexcept { return diagonal.size(); }
    [[nodiscard]] constexpr bool empty() const noexcept { return diagonal.empty(); }

    [[nodiscard]] Real operator[](std::size_t row, std::size_t column) const
    {
      PRECONDITION(row < this->size(), "symmetric tridiagonal row index out of range", row, this->size());
      PRECONDITION(column < this->size(), "symmetric tridiagonal column index out of range", column, this->size());
      if (row == column)
      {
        return diagonal[row];
      }
      if (row + 1 == column)
      {
        return offdiagonal[row];
      }
      if (column + 1 == row)
      {
        return offdiagonal[column];
      }
      return Real{};
    }
};

template <uni20::Real Real>
[[nodiscard]] symmetric_tridiagonal_view<Real> symmetric_tridiagonal(std::span<Real const> diagonal,
                                                                     std::span<Real const> offdiagonal)
{
  return symmetric_tridiagonal_view<Real>{diagonal, offdiagonal};
}

/// \brief Fixed-size owning real symmetric tridiagonal matrix.
/// \tparam Real Real floating-point scalar type.
///
/// The public spans are const handles into the owned storage: callers may mutate
/// entries through `diagonal[i]` and `offdiagonal[i]`, but cannot rebind the
/// handles or resize the matrix. Use external vectors and
/// `symmetric_tridiagonal_view` directly when resizable Lanczos state is needed.
template <uni20::Real Real> class symmetric_tridiagonal_matrix {
  private:
    std::vector<Real> diagonal_storage_;
    std::vector<Real> offdiagonal_storage_;

  public:
    std::span<Real> const diagonal;
    std::span<Real> const offdiagonal;

    explicit symmetric_tridiagonal_matrix(std::size_t size)
        : diagonal_storage_(size), offdiagonal_storage_(size == 0 ? 0 : size - 1), diagonal(diagonal_storage_),
          offdiagonal(offdiagonal_storage_)
    {}

    symmetric_tridiagonal_matrix(symmetric_tridiagonal_matrix const& other)
        : diagonal_storage_(other.diagonal_storage_), offdiagonal_storage_(other.offdiagonal_storage_),
          diagonal(diagonal_storage_), offdiagonal(offdiagonal_storage_)
    {}

    symmetric_tridiagonal_matrix(symmetric_tridiagonal_matrix&& other) noexcept
        : diagonal_storage_(std::move(other.diagonal_storage_)),
          offdiagonal_storage_(std::move(other.offdiagonal_storage_)), diagonal(diagonal_storage_),
          offdiagonal(offdiagonal_storage_)
    {}

    symmetric_tridiagonal_matrix& operator=(symmetric_tridiagonal_matrix const&) = delete;
    symmetric_tridiagonal_matrix& operator=(symmetric_tridiagonal_matrix&&) = delete;

    [[nodiscard]] std::size_t size() const noexcept { return diagonal.size(); }
    [[nodiscard]] bool empty() const noexcept { return diagonal.empty(); }

    [[nodiscard]] symmetric_tridiagonal_view<Real> view() const noexcept
    {
      return symmetric_tridiagonal_view<Real>{diagonal, offdiagonal};
    }

    [[nodiscard]] Real operator[](std::size_t row, std::size_t column) const { return this->view()[row, column]; }
};

} // namespace uni20::krylov
