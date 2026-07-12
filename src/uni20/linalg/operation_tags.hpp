#pragma once

/**
 * \file operation_tags.hpp
 * \ingroup linalg
 * \brief Backend-independent operation values used by dense linalg dispatch.
 */

#include <cstddef>
#include <string_view>

namespace uni20::linalg
{

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
