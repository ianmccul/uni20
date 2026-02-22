#pragma once

#include "core/types.hpp"
#include "linalg/backend_manifest.hpp"
#include "storage/vectorstorage.hpp"
#include "tensor/basic_tensor.hpp"
#include "tensor/tensor_view.hpp"

#include <stdexcept>
#include <type_traits>
#include <utility>

namespace uni20::linalg
{

namespace detail
{
template <typename View> using view_t = std::remove_cvref_t<View>;

template <typename View> using value_type_t = std::remove_cv_t<typename view_t<View>::value_type>;

template <typename View> using default_tag_t = typename view_t<View>::default_tag;

template <typename FirstView, typename... RestViews>
using select_tag_t = std::common_type_t<default_tag_t<FirstView>, default_tag_t<RestViews>...>;

template <typename FirstView, typename... RestViews>
constexpr auto select_tag(FirstView const&, RestViews const&...) -> select_tag_t<FirstView, RestViews...>
{
  using tag_type = select_tag_t<FirstView, RestViews...>;
  static_assert((std::is_convertible_v<default_tag_t<FirstView>, tag_type>)&&(
                    ... && std::is_convertible_v<default_tag_t<RestViews>, tag_type>),
                "TensorView arguments must share a compatible backend tag");
  return tag_type{};
}

template <typename View> constexpr void require_rank_two()
{
  static_assert(view_t<View>::rank() == 2, "TensorView must model a rank-2 matrix");
}

template <typename View> void require_square(View const& view)
{
  require_rank_two<View>();
  if (view.rows() != view.cols())
  {
    throw std::invalid_argument("matrix must be square");
  }
}

template <typename LhsView, typename RhsView> void require_same_shape(LhsView const& lhs, RhsView const& rhs)
{
  require_rank_two<LhsView>();
  require_rank_two<RhsView>();
  if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols())
  {
    throw std::invalid_argument("matrix dimensions do not match");
  }
}

template <typename View> auto make_extents(View view)
{
  using extents_type = stdex::dextents<index_type, 2>;
  return extents_type(view.rows(), view.cols());
}

template <typename BackendTag, typename T, typename Traits> auto clone(TensorView<T const, Traits> view, BackendTag tag)
{
  using value_type = value_type_t<decltype(view)>;
  using tensor_type = uni20::BasicTensor<value_type, stdex::dextents<index_type, 2>, VectorStorage>;
  tensor_type result(make_extents(view));
  ::uni20::linalg::copy(view, result.view(), tag);
  return result;
}
} // namespace detail

/// \brief Copy the contents of one matrix view into another using the default backend.
/// \tparam T Element type stored by the matrix views.
/// \tparam SrcTraits Trait bundle describing the source view.
/// \tparam DstTraits Trait bundle describing the destination view.
/// \param src Source matrix view.
/// \param dst Destination matrix view.
template <typename T, typename SrcTraits, typename DstTraits>
void copy(TensorView<T const, SrcTraits> src, TensorView<T, DstTraits> dst)
{
  auto const tag = detail::select_tag(src, dst);
  ::uni20::linalg::copy(src, dst, tag);
}

/// \brief Fill a matrix view with the identity matrix using the default backend.
/// \tparam T Element type stored by the matrix view.
/// \tparam Traits Trait bundle describing the matrix view.
/// \param out Destination matrix view to receive the identity.
template <typename T, typename Traits> void fill_identity(TensorView<T, Traits> out)
{
  auto const tag = detail::select_tag(out);
  ::uni20::linalg::fill_identity(out, tag);
}

/// \brief Multiply two matrices and store the result in an output view using the default backend.
/// \tparam T Element type stored by the matrix views.
/// \tparam LhsTraits Trait bundle describing the left operand view.
/// \tparam RhsTraits Trait bundle describing the right operand view.
/// \tparam OutTraits Trait bundle describing the destination view.
/// \param lhs Left-hand operand view.
/// \param rhs Right-hand operand view.
/// \param out View receiving the multiplication result.
template <typename T, typename LhsTraits, typename RhsTraits, typename OutTraits>
void multiply_into(TensorView<T const, LhsTraits> lhs, TensorView<T const, RhsTraits> rhs, TensorView<T, OutTraits> out)
{
  auto const tag = detail::select_tag(lhs, rhs, out);
  ::uni20::linalg::multiply_into(lhs, rhs, out, tag);
}

/// \brief Allocate a new tensor containing the matrix product lhs * rhs.
/// \tparam T Element type stored by the matrix views.
/// \tparam LhsTraits Trait bundle describing the left operand view.
/// \tparam RhsTraits Trait bundle describing the right operand view.
/// \tparam BackendTag Backend selection tag.
/// \return Owning tensor containing the product.
template <typename T, typename LhsTraits, typename RhsTraits, typename BackendTag>
auto multiply(TensorView<T const, LhsTraits> lhs, TensorView<T const, RhsTraits> rhs, BackendTag tag)
{
  detail::require_rank_two<decltype(lhs)>();
  detail::require_rank_two<decltype(rhs)>();
  if (lhs.cols() != rhs.rows())
  {
    throw std::invalid_argument("matrix dimensions do not agree for multiplication");
  }

  using value_type = detail::value_type_t<decltype(lhs)>;
  using tensor_type = uni20::BasicTensor<value_type, stdex::dextents<index_type, 2>, VectorStorage>;

  stdex::dextents<index_type, 2> exts(lhs.rows(), rhs.cols());
  tensor_type result(exts);
  ::uni20::linalg::multiply_into(lhs, rhs, result.view(), tag);
  return result;
}

template <typename T, typename LhsTraits, typename RhsTraits>
auto multiply(TensorView<T const, LhsTraits> lhs, TensorView<T const, RhsTraits> rhs)
{
  auto const tag = detail::select_tag(lhs, rhs);
  return multiply(lhs, rhs, tag);
}

/// \brief Add two matrices and store the result in an output view using the default backend.
/// \tparam T Element type stored by the matrix views.
/// \tparam LhsTraits Trait bundle describing the left operand view.
/// \tparam RhsTraits Trait bundle describing the right operand view.
/// \tparam OutTraits Trait bundle describing the destination view.
/// \param lhs Left-hand operand view.
/// \param rhs Right-hand operand view.
/// \param out View receiving the addition result.
template <typename T, typename LhsTraits, typename RhsTraits, typename OutTraits>
void add_into(TensorView<T const, LhsTraits> lhs, TensorView<T const, RhsTraits> rhs, TensorView<T, OutTraits> out)
{
  auto const tag = detail::select_tag(lhs, rhs, out);
  ::uni20::linalg::add_into(lhs, rhs, out, tag);
}

/// \brief Allocate a new tensor containing the element-wise sum lhs + rhs.
/// \tparam T Element type stored by the matrix views.
/// \tparam LhsTraits Trait bundle describing the left operand view.
/// \tparam RhsTraits Trait bundle describing the right operand view.
/// \tparam BackendTag Backend selection tag.
/// \return Owning tensor containing the sum.
template <typename T, typename LhsTraits, typename RhsTraits, typename BackendTag>
auto add(TensorView<T const, LhsTraits> lhs, TensorView<T const, RhsTraits> rhs, BackendTag tag)
{
  detail::require_same_shape(lhs, rhs);
  using value_type = detail::value_type_t<decltype(lhs)>;
  using tensor_type = uni20::BasicTensor<value_type, stdex::dextents<index_type, 2>, VectorStorage>;

  tensor_type result(detail::make_extents(lhs));
  ::uni20::linalg::add_into(lhs, rhs, result.view(), tag);
  return result;
}

template <typename T, typename LhsTraits, typename RhsTraits>
auto add(TensorView<T const, LhsTraits> lhs, TensorView<T const, RhsTraits> rhs)
{
  auto const tag = detail::select_tag(lhs, rhs);
  return add(lhs, rhs, tag);
}

/// \brief Subtract one matrix from another and store the result using the default backend.
/// \tparam T Element type stored by the matrix views.
/// \tparam LhsTraits Trait bundle describing the left operand view.
/// \tparam RhsTraits Trait bundle describing the right operand view.
/// \tparam OutTraits Trait bundle describing the destination view.
/// \param lhs Left-hand operand view.
/// \param rhs Right-hand operand view.
/// \param out View receiving the subtraction result.
template <typename T, typename LhsTraits, typename RhsTraits, typename OutTraits>
void subtract_into(TensorView<T const, LhsTraits> lhs, TensorView<T const, RhsTraits> rhs, TensorView<T, OutTraits> out)
{
  auto const tag = detail::select_tag(lhs, rhs, out);
  ::uni20::linalg::subtract_into(lhs, rhs, out, tag);
}

/// \brief Allocate a new tensor containing the element-wise difference lhs - rhs.
/// \tparam T Element type stored by the matrix views.
/// \tparam LhsTraits Trait bundle describing the left operand view.
/// \tparam RhsTraits Trait bundle describing the right operand view.
/// \tparam BackendTag Backend selection tag.
/// \return Owning tensor containing the difference.
template <typename T, typename LhsTraits, typename RhsTraits, typename BackendTag>
auto subtract(TensorView<T const, LhsTraits> lhs, TensorView<T const, RhsTraits> rhs, BackendTag tag)
{
  detail::require_same_shape(lhs, rhs);
  using value_type = detail::value_type_t<decltype(lhs)>;
  using tensor_type = uni20::BasicTensor<value_type, stdex::dextents<index_type, 2>, VectorStorage>;

  tensor_type result(detail::make_extents(lhs));
  ::uni20::linalg::subtract_into(lhs, rhs, result.view(), tag);
  return result;
}

template <typename T, typename LhsTraits, typename RhsTraits>
auto subtract(TensorView<T const, LhsTraits> lhs, TensorView<T const, RhsTraits> rhs)
{
  auto const tag = detail::select_tag(lhs, rhs);
  return subtract(lhs, rhs, tag);
}

/// \brief Scale a matrix by a scalar factor and store the result using the default backend.
/// \tparam T Element type stored by the matrix views.
/// \tparam Scalar Scalar factor type.
/// \tparam MatTraits Trait bundle describing the input matrix view.
/// \tparam OutTraits Trait bundle describing the destination view.
/// \param mat Matrix view to scale.
/// \param scalar Scalar factor applied to each element.
/// \param out View receiving the scaled matrix.
template <typename T, typename Scalar, typename MatTraits, typename OutTraits>
void scale_into(TensorView<T const, MatTraits> mat, Scalar const& scalar, TensorView<T, OutTraits> out)
{
  auto const tag = detail::select_tag(mat, out);
  ::uni20::linalg::scale_into(mat, scalar, out, tag);
}

/// \brief Allocate a new tensor containing scalar * mat.
/// \tparam T Element type stored by the matrix view.
/// \tparam Scalar Scalar factor type.
/// \tparam MatTraits Trait bundle describing the input matrix view.
/// \tparam BackendTag Backend selection tag.
/// \return Owning tensor containing the scaled matrix.
template <typename T, typename Scalar, typename MatTraits, typename BackendTag>
auto scale(TensorView<T const, MatTraits> mat, Scalar const& scalar, BackendTag tag)
{
  using value_type = detail::value_type_t<decltype(mat)>;
  using tensor_type = uni20::BasicTensor<value_type, stdex::dextents<index_type, 2>, VectorStorage>;

  tensor_type result(detail::make_extents(mat));
  ::uni20::linalg::scale_into(mat, scalar, result.view(), tag);
  return result;
}

template <typename T, typename Scalar, typename MatTraits>
auto scale(TensorView<T const, MatTraits> mat, Scalar const& scalar)
{
  auto const tag = detail::select_tag(mat);
  return scale(mat, scalar, tag);
}

/// \brief Compute the induced matrix 1-norm (maximum absolute column sum).
/// \tparam T Element type stored by the matrix view.
/// \tparam Traits Trait bundle describing the matrix view.
/// \param mat Matrix view whose norm is computed.
/// \return The induced 1-norm of mat.
template <typename T, typename Traits> double matrix_one_norm(TensorView<T const, Traits> mat)
{
  auto const tag = detail::select_tag(mat);
  return ::uni20::linalg::matrix_one_norm(mat, tag);
}

/// \brief Compute the matrix power mat^power and store the result in an output view.
/// \tparam T Element type stored by the matrix views.
/// \tparam MatTraits Trait bundle describing the input matrix view.
/// \tparam OutTraits Trait bundle describing the destination matrix view.
/// \tparam BackendTag Backend selection tag.
/// \param mat Matrix to exponentiate.
/// \param power Non-negative integer exponent.
/// \param out View receiving the matrix power result.
template <typename T, typename MatTraits, typename OutTraits, typename BackendTag>
void matrix_power_into(TensorView<T const, MatTraits> mat, unsigned int power, TensorView<T, OutTraits> out,
                       BackendTag tag)
{
  detail::require_square(mat);
  detail::require_square(out);
  if (mat.rows() != out.rows() || mat.cols() != out.cols())
  {
    throw std::invalid_argument("output matrix has incompatible dimensions for matrix_power");
  }

  using value_type = detail::value_type_t<decltype(mat)>;
  using tensor_type = uni20::BasicTensor<value_type, stdex::dextents<index_type, 2>, VectorStorage>;

  tensor_type result(detail::make_extents(mat));
  ::uni20::linalg::fill_identity(result.view(), tag);

  if (power == 0U)
  {
    ::uni20::linalg::copy(result.view(), out, tag);
    return;
  }

  tensor_type base = detail::clone(mat, tag);
  tensor_type scratch(detail::make_extents(mat));

  unsigned int exponent = power;
  while (exponent > 0U)
  {
    if ((exponent & 1U) != 0U)
    {
      ::uni20::linalg::multiply_into(result.view(), base.view(), scratch.view(), tag);
      ::uni20::linalg::copy(scratch.view(), result.view(), tag);
    }
    exponent >>= 1U;
    if (exponent != 0U)
    {
      ::uni20::linalg::multiply_into(base.view(), base.view(), scratch.view(), tag);
      ::uni20::linalg::copy(scratch.view(), base.view(), tag);
    }
  }

  ::uni20::linalg::copy(result.view(), out, tag);
}

template <typename T, typename MatTraits, typename OutTraits>
void matrix_power_into(TensorView<T const, MatTraits> mat, unsigned int power, TensorView<T, OutTraits> out)
{
  auto const tag = detail::select_tag(mat, out);
  matrix_power_into(mat, power, out, tag);
}

/// \brief Allocate a new tensor containing mat^power.
/// \tparam T Element type stored by the matrix view.
/// \tparam MatTraits Trait bundle describing the input matrix view.
/// \tparam BackendTag Backend selection tag.
/// \param mat Matrix to exponentiate.
/// \param power Non-negative integer exponent.
/// \return Owning tensor containing the exponentiated matrix.
template <typename T, typename MatTraits, typename BackendTag>
auto matrix_power(TensorView<T const, MatTraits> mat, unsigned int power, BackendTag tag)
{
  using value_type = detail::value_type_t<decltype(mat)>;
  using tensor_type = uni20::BasicTensor<value_type, stdex::dextents<index_type, 2>, VectorStorage>;

  tensor_type result(detail::make_extents(mat));
  matrix_power_into(mat, power, result.view(), tag);
  return result;
}

template <typename T, typename MatTraits> auto matrix_power(TensorView<T const, MatTraits> mat, unsigned int power)
{
  auto const tag = detail::select_tag(mat);
  return matrix_power(mat, power, tag);
}

/// \brief Compute the 1-norm of mat^power without exposing the intermediate matrix.
/// \tparam T Element type stored by the matrix view.
/// \tparam MatTraits Trait bundle describing the input matrix view.
/// \tparam BackendTag Backend selection tag.
/// \param mat Matrix to exponentiate.
/// \param power Non-negative integer exponent.
/// \return Induced matrix 1-norm of mat^power.
template <typename T, typename MatTraits, typename BackendTag>
double matrix_one_norm_power(TensorView<T const, MatTraits> mat, unsigned int power, BackendTag tag)
{
  auto powered = matrix_power(mat, power, tag);
  return ::uni20::linalg::matrix_one_norm(powered.view(), tag);
}

template <typename T, typename MatTraits>
double matrix_one_norm_power(TensorView<T const, MatTraits> mat, unsigned int power)
{
  auto const tag = detail::select_tag(mat);
  return matrix_one_norm_power(mat, power, tag);
}

/// \brief Swap two rows of a mutable matrix view using the default backend.
/// \tparam T Element type stored by the matrix view.
/// \tparam Traits Trait bundle describing the matrix view.
/// \param mat Matrix whose rows will be swapped.
/// \param lhs Index of the first row.
/// \param rhs Index of the second row.
template <typename T, typename Traits> void swap_rows(TensorView<T, Traits> mat, index_type lhs, index_type rhs)
{
  auto const tag = detail::select_tag(mat);
  ::uni20::linalg::swap_rows(mat, lhs, rhs, tag);
}

/// \brief Solve the linear system A * X = B and return an owning tensor with the solution.
/// \tparam T Element type stored by the coefficient matrix.
/// \tparam MatTraits Trait bundle describing the coefficient matrix view.
/// \tparam RhsTraits Trait bundle describing the right-hand side view.
/// \tparam BackendTag Backend selection tag.
/// \param A Coefficient matrix view.
/// \param B Right-hand side matrix view.
/// \return Owning tensor containing the solution X.
template <typename T, typename MatTraits, typename RhsTraits, typename BackendTag>
auto solve_linear_system(TensorView<T const, MatTraits> A, TensorView<T const, RhsTraits> B, BackendTag tag)
{
  detail::require_square(A);
  detail::require_rank_two<decltype(B)>();
  if (A.rows() != B.rows())
  {
    throw std::invalid_argument("solve_linear_system requires matching row counts");
  }

  using value_type = detail::value_type_t<decltype(B)>;
  using tensor_type = uni20::BasicTensor<value_type, stdex::dextents<index_type, 2>, VectorStorage>;

  tensor_type A_work = detail::clone(A, tag);
  tensor_type B_work = detail::clone(B, tag);

  ::uni20::linalg::solve_linear_system(A_work.view(), B_work.view(), tag);
  return B_work;
}

template <typename T, typename MatTraits, typename RhsTraits>
auto solve_linear_system(TensorView<T const, MatTraits> A, TensorView<T const, RhsTraits> B)
{
  auto const tag = detail::select_tag(A, B);
  return solve_linear_system<T, MatTraits, RhsTraits, decltype(tag)>(A, B, tag);
}

/// \brief Solve the linear system A * X = B and store the solution in an output view.
/// \tparam T Element type stored by the matrix views.
/// \tparam MatTraits Trait bundle describing the coefficient matrix view.
/// \tparam RhsTraits Trait bundle describing the right-hand side view.
/// \tparam OutTraits Trait bundle describing the destination view.
/// \tparam BackendTag Backend selection tag.
/// \param A Coefficient matrix view.
/// \param B Right-hand side matrix view.
/// \param out View receiving the solution.
template <typename T, typename MatTraits, typename RhsTraits, typename OutTraits, typename BackendTag>
void solve_linear_system_into(TensorView<T const, MatTraits> A, TensorView<T const, RhsTraits> B,
                              TensorView<T, OutTraits> out, BackendTag tag)
{
  auto solution = solve_linear_system(A, B, tag);
  ::uni20::linalg::copy(solution.view(), out, tag);
}

template <typename T, typename MatTraits, typename RhsTraits, typename OutTraits>
void solve_linear_system_into(TensorView<T const, MatTraits> A, TensorView<T const, RhsTraits> B,
                              TensorView<T, OutTraits> out)
{
  auto const tag = detail::select_tag(A, B, out);
  solve_linear_system_into(A, B, out, tag);
}

/// \brief Create an identity matrix with the specified order.
/// \tparam T Value type stored in the matrix.
/// \param order Dimension of the identity matrix.
/// \return Owning tensor initialised to the identity matrix.
template <typename T, typename BackendTag> auto make_identity(index_type order, BackendTag tag)
{
  using tensor_type = uni20::BasicTensor<T, stdex::dextents<index_type, 2>, VectorStorage>;
  stdex::dextents<index_type, 2> exts(order, order);
  tensor_type result(exts);
  ::uni20::linalg::fill_identity(result.view(), tag);
  return result;
}

template <typename T> auto make_identity(index_type order)
{
  return make_identity<T>(order, VectorStorage::default_tag{});
}

namespace ops
{
using ::uni20::linalg::add;
using ::uni20::linalg::add_into;
using ::uni20::linalg::copy;
using ::uni20::linalg::fill_identity;
using ::uni20::linalg::make_identity;
using ::uni20::linalg::matrix_one_norm;
using ::uni20::linalg::matrix_one_norm_power;
using ::uni20::linalg::matrix_power;
using ::uni20::linalg::matrix_power_into;
using ::uni20::linalg::multiply;
using ::uni20::linalg::multiply_into;
using ::uni20::linalg::scale;
using ::uni20::linalg::scale_into;
using ::uni20::linalg::solve_linear_system;
using ::uni20::linalg::solve_linear_system_into;
using ::uni20::linalg::subtract;
using ::uni20::linalg::subtract_into;
using ::uni20::linalg::swap_rows;
} // namespace ops

} // namespace uni20::linalg
