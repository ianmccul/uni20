#pragma once

/**
 * \file operation_tags.hpp
 * \ingroup linalg
 * \brief Backend-independent tensor-view operation values used by dense linalg dispatch.
 */

#include <uni20/linalg/reduction_axes.hpp>

#include <cstddef>
#include <string_view>
#include <type_traits>

namespace uni20::linalg
{

/// \brief Accessor-respecting element copy operation tag.
struct copy_op
{
    static constexpr std::string_view name = "copy";
};

/// \brief Accessor-respecting in-place scalar conjugation operation tag.
struct conjugate_inplace_op
{
    static constexpr std::string_view name = "conjugate_inplace";
};

/// \brief Callable-carrying elementwise overwrite operation.
/// \details The function is invoked as a const object. Backends may specialize
///          on its concrete type while preserving one generic front-end API.
///          Input elements are passed to the function as values.
template <class Function> struct transform_op
{
    static constexpr std::string_view name = "transform";
    [[no_unique_address]] Function function;
};

/// \brief Deduce an overwrite operation that owns a decayed callable value.
template <class Function> transform_op(Function) -> transform_op<std::decay_t<Function>>;

/// \brief Callable-carrying elementwise update operation.
/// \details The old output value is passed by value as the callable's first
///          argument. The output remains one read/write operand rather than a
///          duplicated input.
template <class Function> struct transform_inplace_op
{
    static constexpr std::string_view name = "transform_inplace";
    [[no_unique_address]] Function function;
};

/// \brief Deduce an update operation that owns a decayed callable value.
template <class Function> transform_inplace_op(Function) -> transform_inplace_op<std::decay_t<Function>>;

/// \brief Dense matrix multiplication operation tag.
struct gemm_op
{
    static constexpr std::string_view name = "gemm";
};

/// \brief Dense matrix-vector multiplication operation tag.
struct gemv_op
{
    static constexpr std::string_view name = "gemv";
};

/// \brief Conjugate-linear-left tensor inner-product operation tag.
struct inner_product_op
{
    static constexpr std::string_view name = "inner_product";
};

/// \brief Euclidean tensor norm operation tag.
struct norm_op
{
    static constexpr std::string_view name = "norm";
};

/// \brief Sum reduction over one or more fixed-rank input axes.
/// \details The diagnostic name remains the user-facing operation name while
///          the C++ type makes its reduction role explicit.
template <std::size_t InputRank, std::size_t ReducedRank> struct sum_reduction_op
{
    static constexpr std::string_view name = "sum";
    ReductionAxes<InputRank, ReducedRank> axes;
};

/// \brief Dense matrix exponential operation tag.
struct matrix_exponential_op
{
    static constexpr std::string_view name = "matrix_exponential";
};

/// \brief Matrix region selected by structured initialization operations.
enum class MatrixRegion
{
  All,
  Upper,
  Lower
};

/// \brief Structured dense matrix initialization operation tag.
struct matrix_set_op
{
    static constexpr std::string_view name = "matrix_set";
    MatrixRegion region = MatrixRegion::All;
};

/// \brief Triangle of a symmetric or Hermitian matrix supplied to an operation.
enum class MatrixTriangle
{
  Upper,
  Lower
};

/// \brief Options for a dense symmetric or Hermitian eigensystem.
struct SelfAdjointEighOptions
{
    bool compute_vectors = true;
    MatrixTriangle triangle = MatrixTriangle::Upper;
};

/// \brief Dense symmetric or Hermitian eigensystem operation tag.
struct self_adjoint_eigh_op
{
    static constexpr std::string_view name = "self_adjoint_eigh";
    bool compute_vectors = true;
    MatrixTriangle triangle = MatrixTriangle::Upper;
};

/// \brief Extent policy for one side of an exact singular value decomposition.
enum class SvdVectorExtent
{
  Reduced,
  Full
};

/// \brief Output-shape options for an exact singular value decomposition.
struct SvdOptions
{
    SvdVectorExtent left = SvdVectorExtent::Reduced;
    SvdVectorExtent right = SvdVectorExtent::Reduced;
};

/// \brief Input allocation overwritten by a singular-vector factor.
enum class SvdOverwrite
{
  None,
  Left,
  Right
};

/// \brief Exact dense singular-values-only operation tag.
struct singular_values_op
{
    static constexpr std::string_view name = "singular_values";
};

/// \brief Exact dense left singular-vector operation tag.
struct svd_left_op
{
    static constexpr std::string_view name = "svd_left";
    SvdVectorExtent left = SvdVectorExtent::Reduced;
    bool overwrite_input = false;
};

/// \brief Exact dense right singular-vector operation tag.
struct svd_right_op
{
    static constexpr std::string_view name = "svd_right";
    SvdVectorExtent right = SvdVectorExtent::Reduced;
    bool overwrite_input = false;
};

/// \brief Exact dense singular value decomposition operation tag.
struct svd_op
{
    static constexpr std::string_view name = "svd";
    SvdVectorExtent left = SvdVectorExtent::Reduced;
    SvdVectorExtent right = SvdVectorExtent::Reduced;
    SvdOverwrite overwrite = SvdOverwrite::None;
};

/// \brief Symmetric tridiagonal eigensystem operation tag.
struct symmetric_tridiagonal_eigen_op
{
    static constexpr std::string_view name = "symmetric_tridiagonal_eigen";
    bool compute_vectors = true;
};

/// \brief Dense nonsymmetric eigensystem operation tag.
struct nonsymmetric_eigen_op
{
    static constexpr std::string_view name = "nonsymmetric_eigen";
    bool compute_right_vectors = true;
};

/// \brief Dense Schur decomposition operation tag.
struct schur_op
{
    static constexpr std::string_view name = "schur";
    bool compute_vectors = true;
};

/// \brief Schur decomposition of an already upper-Hessenberg real matrix.
struct hessenberg_schur_op
{
    static constexpr std::string_view name = "hessenberg_schur";
    bool compute_vectors = true;
};

/// \brief Move one Schur block or entry to a requested position.
struct schur_reorder_op
{
    static constexpr std::string_view name = "reorder_schur";
    std::size_t from = 0;
    std::size_t to = 0;
    bool update_vectors = true;
};

} // namespace uni20::linalg
