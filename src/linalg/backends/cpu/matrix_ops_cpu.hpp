#pragma once

#include "core/types.hpp"
#include "tags/cpu.hpp"
#include "tensor/tensor_view.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace uni20::linalg::backends::cpu::detail
{

namespace util
{
template <typename T, typename Traits> constexpr void require_rank_two(TensorView<T, Traits> const&)
{
  static_assert(TensorView<T, Traits>::rank() == 2, "TensorView must model a rank-2 matrix");
}

template <typename T, typename Traits> void require_square(TensorView<T, Traits> view)
{
  require_rank_two(view);
  if (view.rows() != view.cols())
  {
    throw std::invalid_argument("matrix must be square");
  }
}

template <typename TLhs, typename TraitsLhs, typename TRhs, typename TraitsRhs>
void require_same_shape(TensorView<TLhs, TraitsLhs> lhs, TensorView<TRhs, TraitsRhs> rhs)
{
  require_rank_two(lhs);
  require_rank_two(rhs);
  if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols())
  {
    throw std::invalid_argument("matrix dimensions do not match");
  }
}

template <typename T, typename Traits> auto mutable_span(TensorView<T, Traits> view) { return view.mutable_mdspan(); }

template <typename T, typename Traits> auto const_span(TensorView<T, Traits> view) { return view.mdspan(); }
} // namespace util

/// \brief Copy the contents of one matrix view into another.
/// \tparam TSrc Element type stored by the source view.
/// \tparam SrcTraits Trait bundle describing the source view.
/// \tparam TDst Element type stored by the destination view.
/// \tparam DstTraits Trait bundle describing the destination view.
/// \param src Source matrix view.
/// \param dst Destination matrix view.
template <typename TSrc, typename SrcTraits, typename TDst, typename DstTraits>
void copy(TensorView<TSrc const, SrcTraits> src, TensorView<TDst, DstTraits> dst)
{
  util::require_same_shape(src, dst);

  auto src_span = util::const_span(src);
  auto dst_span = util::mutable_span(dst);

  for (index_type i = 0; i < src.rows(); ++i)
  {
    for (index_type j = 0; j < src.cols(); ++j)
    {
      dst_span[i, j] = src_span[i, j];
    }
  }
}

/// \brief Fill a matrix view with the identity matrix.
/// \tparam T Element type stored by the destination view.
/// \tparam Traits Trait bundle describing the destination view.
/// \param out Destination matrix view which will be overwritten with the identity matrix.
template <typename T, typename Traits> void fill_identity(TensorView<T, Traits> out)
{
  util::require_square(out);

  using value_type = std::remove_cv_t<typename TensorView<T, Traits>::value_type>;
  auto out_span = util::mutable_span(out);

  for (index_type i = 0; i < out.rows(); ++i)
  {
    for (index_type j = 0; j < out.cols(); ++j)
    {
      out_span[i, j] = (i == j) ? value_type{1} : value_type{};
    }
  }
}

/// \brief Multiply two matrices and store the product in a destination view.
/// \tparam TLhs Element type stored by the left operand view.
/// \tparam LhsTraits Trait bundle describing the left operand view.
/// \tparam TRhs Element type stored by the right operand view.
/// \tparam RhsTraits Trait bundle describing the right operand view.
/// \tparam TOut Element type stored by the destination view.
/// \tparam OutTraits Trait bundle describing the destination view.
/// \param lhs Left-hand operand view.
/// \param rhs Right-hand operand view.
/// \param out View receiving the multiplication result.
template <typename TLhs, typename LhsTraits, typename TRhs, typename RhsTraits, typename TOut, typename OutTraits>
void multiply(TensorView<TLhs const, LhsTraits> lhs, TensorView<TRhs const, RhsTraits> rhs,
              TensorView<TOut, OutTraits> out)
{
  util::require_rank_two(lhs);
  util::require_rank_two(rhs);
  util::require_rank_two(out);

  if (lhs.cols() != rhs.rows())
  {
    throw std::invalid_argument("matrix dimensions do not agree for multiplication");
  }

  if (out.rows() != lhs.rows() || out.cols() != rhs.cols())
  {
    throw std::invalid_argument("output matrix has incompatible dimensions for multiplication");
  }

  using value_type = std::remove_cv_t<typename TensorView<TOut, OutTraits>::value_type>;

  auto lhs_span = util::const_span(lhs);
  auto rhs_span = util::const_span(rhs);
  auto out_span = util::mutable_span(out);

  for (index_type i = 0; i < lhs.rows(); ++i)
  {
    for (index_type j = 0; j < rhs.cols(); ++j)
    {
      value_type value{};
      for (index_type k = 0; k < lhs.cols(); ++k)
      {
        value += lhs_span[i, k] * rhs_span[k, j];
      }
      out_span[i, j] = value;
    }
  }
}

/// \brief Add two matrices and store the result in a destination view.
/// \tparam TLhs Element type stored by the left operand view.
/// \tparam LhsTraits Trait bundle describing the left operand view.
/// \tparam TRhs Element type stored by the right operand view.
/// \tparam RhsTraits Trait bundle describing the right operand view.
/// \tparam TOut Element type stored by the destination view.
/// \tparam OutTraits Trait bundle describing the destination view.
/// \param lhs Left-hand operand view.
/// \param rhs Right-hand operand view.
/// \param out View receiving the addition result.
template <typename TLhs, typename LhsTraits, typename TRhs, typename RhsTraits, typename TOut, typename OutTraits>
void add(TensorView<TLhs const, LhsTraits> lhs, TensorView<TRhs const, RhsTraits> rhs, TensorView<TOut, OutTraits> out)
{
  util::require_same_shape(lhs, rhs);
  util::require_same_shape(lhs, out);

  auto lhs_span = util::const_span(lhs);
  auto rhs_span = util::const_span(rhs);
  auto out_span = util::mutable_span(out);

  for (index_type i = 0; i < lhs.rows(); ++i)
  {
    for (index_type j = 0; j < lhs.cols(); ++j)
    {
      out_span[i, j] = lhs_span[i, j] + rhs_span[i, j];
    }
  }
}

/// \brief Subtract one matrix from another and store the result.
/// \tparam TLhs Element type stored by the left operand view.
/// \tparam LhsTraits Trait bundle describing the left operand view.
/// \tparam TRhs Element type stored by the right operand view.
/// \tparam RhsTraits Trait bundle describing the right operand view.
/// \tparam TOut Element type stored by the destination view.
/// \tparam OutTraits Trait bundle describing the destination view.
/// \param lhs Left-hand operand view.
/// \param rhs Right-hand operand view.
/// \param out View receiving the subtraction result.
template <typename TLhs, typename LhsTraits, typename TRhs, typename RhsTraits, typename TOut, typename OutTraits>
void subtract(TensorView<TLhs const, LhsTraits> lhs, TensorView<TRhs const, RhsTraits> rhs,
              TensorView<TOut, OutTraits> out)
{
  util::require_same_shape(lhs, rhs);
  util::require_same_shape(lhs, out);

  auto lhs_span = util::const_span(lhs);
  auto rhs_span = util::const_span(rhs);
  auto out_span = util::mutable_span(out);

  for (index_type i = 0; i < lhs.rows(); ++i)
  {
    for (index_type j = 0; j < lhs.cols(); ++j)
    {
      out_span[i, j] = lhs_span[i, j] - rhs_span[i, j];
    }
  }
}

/// \brief Scale a matrix by a scalar factor.
/// \tparam TMat Element type stored by the input matrix view.
/// \tparam MatTraits Trait bundle describing the input matrix view.
/// \tparam Scalar Scalar factor type.
/// \tparam TOut Element type stored by the destination view.
/// \tparam OutTraits Trait bundle describing the destination view.
/// \param mat Matrix view to scale.
/// \param scalar Scalar factor applied to each element.
/// \param out View receiving the scaled matrix.
template <typename TMat, typename MatTraits, typename Scalar, typename TOut, typename OutTraits>
void scale(TensorView<TMat const, MatTraits> mat, Scalar const& scalar, TensorView<TOut, OutTraits> out)
{
  util::require_same_shape(mat, out);

  auto mat_span = util::const_span(mat);
  auto out_span = util::mutable_span(out);

  for (index_type i = 0; i < mat.rows(); ++i)
  {
    for (index_type j = 0; j < mat.cols(); ++j)
    {
      out_span[i, j] = mat_span[i, j] * scalar;
    }
  }
}

/// \brief Compute the induced 1-norm (maximum column sum) of a matrix.
/// \tparam T Element type stored by the matrix view.
/// \tparam Traits Trait bundle describing the matrix view.
/// \param mat Matrix view whose norm is computed.
/// \return The induced matrix 1-norm.
template <typename T, typename Traits> double matrix_one_norm(TensorView<T const, Traits> mat)
{
  util::require_rank_two(mat);

  auto mat_span = util::const_span(mat);
  double result = 0.0;

  for (index_type j = 0; j < mat.cols(); ++j)
  {
    double column_sum = 0.0;
    for (index_type i = 0; i < mat.rows(); ++i)
    {
      column_sum += std::abs(mat_span[i, j]);
    }
    result = std::max(result, column_sum);
  }

  return result;
}

/// \brief Swap two rows of a mutable matrix view.
/// \tparam T Element type stored by the matrix view.
/// \tparam Traits Trait bundle describing the matrix view.
/// \param mat Matrix view to modify.
/// \param lhs Index of the first row.
/// \param rhs Index of the second row.
template <typename T, typename Traits> void swap_rows(TensorView<T, Traits> mat, index_type lhs, index_type rhs)
{
  util::require_rank_two(mat);

  if (lhs == rhs)
  {
    return;
  }

  if (lhs < 0 || rhs < 0 || lhs >= mat.rows() || rhs >= mat.rows())
  {
    throw std::out_of_range("row index out of bounds in swap_rows");
  }

  auto span = util::mutable_span(mat);
  for (index_type j = 0; j < mat.cols(); ++j)
  {
    using std::swap;
    swap(span[lhs, j], span[rhs, j]);
  }
}

/// \brief Solve the linear system A * X = B using Gaussian elimination with partial pivoting.
/// \tparam TA Element type stored by the coefficient matrix view.
/// \tparam ATraits Trait bundle describing the coefficient matrix view.
/// \tparam TB Element type stored by the right-hand side view.
/// \tparam BTraits Trait bundle describing the right-hand side view.
/// \param A Coefficient matrix (modified in-place to its LU factorisation).
/// \param B Right-hand side matrix; overwritten with the solution.
template <typename TA, typename ATraits, typename TB, typename BTraits>
void solve_linear_system(TensorView<TA, ATraits> A, TensorView<TB, BTraits> B)
{
  static_assert(!std::is_const_v<TA>, "Coefficient matrix must be mutable");
  static_assert(!std::is_const_v<TB>, "Right-hand side matrix must be mutable");

  util::require_rank_two(A);
  util::require_rank_two(B);

  if (A.rows() != A.cols() || A.rows() != B.rows())
  {
    throw std::invalid_argument("solve_linear_system requires square coefficient matrix");
  }

  index_type const n = A.rows();
  index_type const nrhs = B.cols();

  auto A_span = util::mutable_span(A);
  auto B_span = util::mutable_span(B);

  using value_type = std::remove_cv_t<typename TensorView<TB, BTraits>::value_type>;

  for (index_type k = 0; k < n; ++k)
  {
    index_type pivot_row = k;
    double pivot_value = std::abs(A_span[k, k]);
    for (index_type i = k + 1; i < n; ++i)
    {
      double const candidate = std::abs(A_span[i, k]);
      if (candidate > pivot_value)
      {
        pivot_value = candidate;
        pivot_row = i;
      }
    }

    if (pivot_value == 0.0)
    {
      throw std::runtime_error("singular matrix in solve_linear_system");
    }

    if (pivot_row != k)
    {
      swap_rows(A, k, pivot_row);
      swap_rows(B, k, pivot_row);
      A_span = util::mutable_span(A);
      B_span = util::mutable_span(B);
    }

    value_type const pivot = A_span[k, k];
    for (index_type i = k + 1; i < n; ++i)
    {
      value_type const factor = A_span[i, k] / pivot;
      if (factor == value_type{})
      {
        continue;
      }
      A_span[i, k] = value_type{};
      for (index_type j = k + 1; j < n; ++j)
      {
        A_span[i, j] -= factor * A_span[k, j];
      }
      for (index_type j = 0; j < nrhs; ++j)
      {
        B_span[i, j] -= factor * B_span[k, j];
      }
    }
  }

  for (index_type i = n; i-- > 0;)
  {
    value_type const pivot = A_span[i, i];
    for (index_type j = 0; j < nrhs; ++j)
    {
      value_type value = B_span[i, j];
      for (index_type k = i + 1; k < n; ++k)
      {
        value -= A_span[i, k] * B_span[k, j];
      }
      B_span[i, j] = value / pivot;
    }
  }
}

} // namespace uni20::linalg::backends::cpu::detail

namespace uni20::linalg
{
template <typename TSrc, typename SrcTraits, typename TDst, typename DstTraits>
void copy(TensorView<TSrc const, SrcTraits> src, TensorView<TDst, DstTraits> dst, cpu_tag)
{
  backends::cpu::detail::copy(src, dst);
}

template <typename T, typename Traits> void fill_identity(TensorView<T, Traits> out, cpu_tag)
{
  backends::cpu::detail::fill_identity(out);
}

template <typename TLhs, typename LhsTraits, typename TRhs, typename RhsTraits, typename TOut, typename OutTraits>
void multiply_into(TensorView<TLhs const, LhsTraits> lhs, TensorView<TRhs const, RhsTraits> rhs,
                   TensorView<TOut, OutTraits> out, cpu_tag)
{
  backends::cpu::detail::multiply(lhs, rhs, out);
}

template <typename TLhs, typename LhsTraits, typename TRhs, typename RhsTraits, typename TOut, typename OutTraits>
void add_into(TensorView<TLhs const, LhsTraits> lhs, TensorView<TRhs const, RhsTraits> rhs,
              TensorView<TOut, OutTraits> out, cpu_tag)
{
  backends::cpu::detail::add(lhs, rhs, out);
}

template <typename TLhs, typename LhsTraits, typename TRhs, typename RhsTraits, typename TOut, typename OutTraits>
void subtract_into(TensorView<TLhs const, LhsTraits> lhs, TensorView<TRhs const, RhsTraits> rhs,
                   TensorView<TOut, OutTraits> out, cpu_tag)
{
  backends::cpu::detail::subtract(lhs, rhs, out);
}

template <typename TMat, typename MatTraits, typename Scalar, typename TOut, typename OutTraits>
void scale_into(TensorView<TMat const, MatTraits> mat, Scalar const& scalar, TensorView<TOut, OutTraits> out, cpu_tag)
{
  backends::cpu::detail::scale(mat, scalar, out);
}

template <typename T, typename Traits> double matrix_one_norm(TensorView<T const, Traits> mat, cpu_tag)
{
  return backends::cpu::detail::matrix_one_norm(mat);
}

template <typename T, typename Traits>
void swap_rows(TensorView<T, Traits> mat, index_type lhs, index_type rhs, cpu_tag)
{
  backends::cpu::detail::swap_rows(mat, lhs, rhs);
}

template <typename TA, typename ATraits, typename TB, typename BTraits>
void solve_linear_system(TensorView<TA, ATraits> A, TensorView<TB, BTraits> B, cpu_tag)
{
  backends::cpu::detail::solve_linear_system(A, B);
}
} // namespace uni20::linalg
