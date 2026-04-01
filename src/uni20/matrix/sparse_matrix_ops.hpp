/**
 * \file sparse_matrix_ops.hpp
 * \brief Basic algebra for `uni20::SparseMatrix<T>`.
 * \details See `docs/matrix.md` for the current scope and intended operator/MPO use.
 */

#pragma once

#include <uni20/matrix/sparse_matrix.hpp>

#include <concepts>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace uni20
{

namespace detail
{
template <typename Lhs, typename Rhs, typename CombineFn>
using sparse_add_result_t =
    std::remove_cvref_t<std::invoke_result_t<CombineFn const&, Lhs const&, Rhs const&>>;

template <typename Lhs, typename Rhs, typename MultiplyFn>
using sparse_product_t =
    std::remove_cvref_t<std::invoke_result_t<MultiplyFn const&, Lhs const&, Rhs const&>>;

template <typename Product, typename AddFn>
using sparse_accum_t =
    std::remove_cvref_t<std::invoke_result_t<AddFn const&, Product const&, Product const&>>;

template <typename T, typename Scalar, typename MultiplyFn>
using sparse_scale_result_t =
    std::remove_cvref_t<std::invoke_result_t<MultiplyFn const&, T const&, Scalar const&>>;

template <typename Lhs, typename Rhs>
void require_same_shape(SparseMatrix<Lhs> const& lhs, SparseMatrix<Rhs> const& rhs)
{
    if (lhs.shape() != rhs.shape())
    {
        throw std::invalid_argument("SparseMatrix dimensions do not match");
    }
}

template <typename Lhs, typename Rhs>
void require_multiply_shape(SparseMatrix<Lhs> const& lhs, SparseMatrix<Rhs> const& rhs)
{
    if (lhs.cols() != rhs.rows())
    {
        throw std::invalid_argument("SparseMatrix dimensions do not agree for multiplication");
    }
}
} // namespace detail

/// \brief Add two sparse matrices entry-wise using a custom overlap combiner.
/// \details Entries present in only one operand are copied through unchanged.
///          Entries present in both operands are combined with `combine_fn`.
/// \tparam Lhs Value type stored by the left operand.
/// \tparam Rhs Value type stored by the right operand.
/// \tparam CombineFn Function object used for overlapping entries.
/// \param lhs Left-hand sparse matrix.
/// \param rhs Right-hand sparse matrix.
/// \param combine_fn Function used to combine `lhs(i,j)` and `rhs(i,j)` when both exist.
/// \return Sparse matrix containing the merged entries.
template <typename Lhs, typename Rhs, typename CombineFn>
auto add(SparseMatrix<Lhs> const& lhs, SparseMatrix<Rhs> const& rhs, CombineFn combine_fn)
{
    detail::require_same_shape(lhs, rhs);

    using result_type = detail::sparse_add_result_t<Lhs, Rhs, CombineFn>;
    static_assert(std::constructible_from<result_type, Lhs const&>,
                  "SparseMatrix add result type must be constructible from the left value type");
    static_assert(std::constructible_from<result_type, Rhs const&>,
                  "SparseMatrix add result type must be constructible from the right value type");

    SparseMatrix<result_type> result(lhs.rows(), lhs.cols());

    for (std::size_t row = 0; row < lhs.rows(); ++row)
    {
        auto const lhs_row = lhs.row(row);
        auto const rhs_row = rhs.row(row);

        std::size_t lhs_index = 0;
        std::size_t rhs_index = 0;
        while (lhs_index < lhs_row.size() || rhs_index < rhs_row.size())
        {
            if (rhs_index == rhs_row.size() || (lhs_index < lhs_row.size() && lhs_row[lhs_index].column < rhs_row[rhs_index].column))
            {
                auto const& entry = lhs_row[lhs_index++];
                result.insert_or_assign(row, entry.column, result_type(entry.value));
            }
            else if (lhs_index == lhs_row.size() || rhs_row[rhs_index].column < lhs_row[lhs_index].column)
            {
                auto const& entry = rhs_row[rhs_index++];
                result.insert_or_assign(row, entry.column, result_type(entry.value));
            }
            else
            {
                auto const column = lhs_row[lhs_index].column;
                result.insert_or_assign(row, column, std::invoke(combine_fn, lhs_row[lhs_index].value, rhs_row[rhs_index].value));
                ++lhs_index;
                ++rhs_index;
            }
        }
    }

    return result;
}

/// \brief Add two sparse matrices entry-wise using `std::plus<>` on overlaps.
/// \tparam Lhs Value type stored by the left operand.
/// \tparam Rhs Value type stored by the right operand.
/// \param lhs Left-hand sparse matrix.
/// \param rhs Right-hand sparse matrix.
/// \return Sparse matrix containing the merged entries.
template <typename Lhs, typename Rhs>
auto add(SparseMatrix<Lhs> const& lhs, SparseMatrix<Rhs> const& rhs)
{
    return add(lhs, rhs, std::plus<>{});
}

/// \brief Scale a sparse matrix by a scalar using a custom element-wise action.
/// \tparam T Value type stored by the sparse matrix.
/// \tparam Scalar Scalar factor type.
/// \tparam MultiplyFn Function object used to scale one stored entry.
/// \param matrix Input sparse matrix.
/// \param scalar Scalar factor supplied to `multiply_fn`.
/// \param multiply_fn Function used to compute `multiply_fn(value, scalar)`.
/// \return Sparse matrix containing the scaled entries.
template <typename T, typename Scalar, typename MultiplyFn>
auto scale(SparseMatrix<T> const& matrix, Scalar const& scalar, MultiplyFn multiply_fn)
{
    using result_type = detail::sparse_scale_result_t<T, Scalar, MultiplyFn>;

    SparseMatrix<result_type> result(matrix.rows(), matrix.cols());
    for (std::size_t row = 0; row < matrix.rows(); ++row)
    {
        for (auto const& entry : matrix.row(row))
        {
            result.insert_or_assign(row, entry.column, std::invoke(multiply_fn, entry.value, scalar));
        }
    }
    return result;
}

/// \brief Scale a sparse matrix by a scalar using `std::multiplies<>`.
/// \tparam T Value type stored by the sparse matrix.
/// \tparam Scalar Scalar factor type.
/// \param matrix Input sparse matrix.
/// \param scalar Scalar factor.
/// \return Sparse matrix containing the scaled entries.
template <typename T, typename Scalar>
auto scale(SparseMatrix<T> const& matrix, Scalar const& scalar)
{
    return scale(matrix, scalar, std::multiplies<>{});
}

/// \brief Multiply two sparse matrices using custom product and accumulation functors.
/// \details The multiplication walks nonzero rows of `lhs` and matching nonzero rows of `rhs`.
///          Products contributing to the same output slot are combined with `add_fn`.
/// \tparam Lhs Value type stored by the left operand.
/// \tparam Rhs Value type stored by the right operand.
/// \tparam MultiplyFn Function object used for the nested product.
/// \tparam AddFn Function object used to accumulate repeated contributions.
/// \param lhs Left-hand sparse matrix.
/// \param rhs Right-hand sparse matrix.
/// \param multiply_fn Function used to compute one pairwise product.
/// \param add_fn Function used to accumulate two contributions targeting the same output entry.
/// \return Sparse matrix containing the matrix product.
template <typename Lhs, typename Rhs, typename MultiplyFn, typename AddFn>
auto multiply(SparseMatrix<Lhs> const& lhs, SparseMatrix<Rhs> const& rhs, MultiplyFn multiply_fn, AddFn add_fn)
{
    detail::require_multiply_shape(lhs, rhs);

    using product_type = detail::sparse_product_t<Lhs, Rhs, MultiplyFn>;
    using result_type = detail::sparse_accum_t<product_type, AddFn>;
    static_assert(std::constructible_from<result_type, product_type>,
                  "SparseMatrix multiply result type must be constructible from one product contribution");

    SparseMatrix<result_type> result(lhs.rows(), rhs.cols());

    for (std::size_t row = 0; row < lhs.rows(); ++row)
    {
        for (auto const& lhs_entry : lhs.row(row))
        {
            for (auto const& rhs_entry : rhs.row(lhs_entry.column))
            {
                auto product = std::invoke(multiply_fn, lhs_entry.value, rhs_entry.value);
                if (auto* existing = result.find(row, rhs_entry.column); existing != nullptr)
                {
                    *existing = std::invoke(add_fn, *existing, product);
                }
                else
                {
                    result.insert_or_assign(row, rhs_entry.column, result_type(std::move(product)));
                }
            }
        }
    }

    return result;
}

/// \brief Multiply two sparse matrices using `std::multiplies<>` and `std::plus<>`.
/// \tparam Lhs Value type stored by the left operand.
/// \tparam Rhs Value type stored by the right operand.
/// \param lhs Left-hand sparse matrix.
/// \param rhs Right-hand sparse matrix.
/// \return Sparse matrix containing the matrix product.
template <typename Lhs, typename Rhs>
auto multiply(SparseMatrix<Lhs> const& lhs, SparseMatrix<Rhs> const& rhs)
{
    return multiply(lhs, rhs, std::multiplies<>{}, std::plus<>{});
}

/// \brief Form the Kronecker product of two sparse matrices using a custom nested product.
/// \tparam Lhs Value type stored by the left operand.
/// \tparam Rhs Value type stored by the right operand.
/// \tparam MultiplyFn Function object used for the nested product.
/// \param lhs Left-hand sparse matrix.
/// \param rhs Right-hand sparse matrix.
/// \param multiply_fn Function used to compute one pairwise block product.
/// \return Sparse matrix containing the Kronecker product.
template <typename Lhs, typename Rhs, typename MultiplyFn>
auto kronecker(SparseMatrix<Lhs> const& lhs, SparseMatrix<Rhs> const& rhs, MultiplyFn multiply_fn)
{
    using result_type = detail::sparse_product_t<Lhs, Rhs, MultiplyFn>;

    SparseMatrix<result_type> result(lhs.rows() * rhs.rows(), lhs.cols() * rhs.cols());
    for (std::size_t lhs_row = 0; lhs_row < lhs.rows(); ++lhs_row)
    {
        for (auto const& lhs_entry : lhs.row(lhs_row))
        {
            for (std::size_t rhs_row = 0; rhs_row < rhs.rows(); ++rhs_row)
            {
                for (auto const& rhs_entry : rhs.row(rhs_row))
                {
                    auto const row = lhs_row * rhs.rows() + rhs_row;
                    auto const col = lhs_entry.column * rhs.cols() + rhs_entry.column;
                    result.insert_or_assign(row, col, std::invoke(multiply_fn, lhs_entry.value, rhs_entry.value));
                }
            }
        }
    }
    return result;
}

/// \brief Form the Kronecker product of two sparse matrices using `std::multiplies<>`.
/// \tparam Lhs Value type stored by the left operand.
/// \tparam Rhs Value type stored by the right operand.
/// \param lhs Left-hand sparse matrix.
/// \param rhs Right-hand sparse matrix.
/// \return Sparse matrix containing the Kronecker product.
template <typename Lhs, typename Rhs>
auto kronecker(SparseMatrix<Lhs> const& lhs, SparseMatrix<Rhs> const& rhs)
{
    return kronecker(lhs, rhs, std::multiplies<>{});
}

/// \brief Alias for `kronecker(lhs, rhs, ...)`.
/// \tparam Args Argument pack forwarded to `kronecker`.
/// \param args Arguments forwarded to `kronecker`.
/// \return Result of `kronecker(std::forward<Args>(args)...)`.
template <typename... Args>
auto kron(Args&&... args)
{
    return kronecker(std::forward<Args>(args)...);
}

} // namespace uni20
