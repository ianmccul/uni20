#pragma once

#include <uni20/krylov/dense_linalg.hpp>

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace uni20::krylov
{

/// \brief Eigenvalues and optional eigenvectors of a symmetric tridiagonal matrix.
/// \tparam Scalar Real scalar type.

template <typename Scalar> struct TridiagonalEigensystem
{
    std::vector<Scalar> eigenvalues;
    uni20::DenseMatrix<Scalar> eigenvectors;
};

/// \brief Eigenvalues and optional eigenvectors of a dense real symmetric matrix.

template <uni20::LapackReal Real> struct RealNonsymmetricEigensystem
{
    std::vector<uni20::complex<Real>> eigenvalues;
    uni20::DenseMatrix<uni20::complex<Real>> right_eigenvectors;
};

/// \brief Expert eigensystem diagnostics for a dense real nonsymmetric matrix.
/// \tparam Real Underlying real precision.

template <uni20::LapackReal Scalar> struct RealSchurBlock
{
    std::size_t begin = 0;
    std::size_t size = 0;
    uni20::complex<Scalar> first_eigenvalue{};
    uni20::complex<Scalar> second_eigenvalue{};
};

/// \brief Generalized real Schur decomposition of a dense real matrix pencil.

template <uni20::LapackReal Real> struct ComplexNonsymmetricEigensystem
{
    std::vector<uni20::complex<Real>> eigenvalues;
    uni20::DenseMatrix<uni20::complex<Real>> right_eigenvectors;
};

/// \brief Complex Schur decomposition stored in column-major DenseMatrix values.
/// \tparam Real Underlying real precision.
template <uni20::LapackReal Real> struct ComplexSchurDecomposition
{
    uni20::DenseMatrix<uni20::complex<Real>> schur_form;
    uni20::DenseMatrix<uni20::complex<Real>> schur_vectors;
    std::vector<uni20::complex<Real>> eigenvalues;
};

/// \brief Complex Schur decomposition stored in prototype layout-right matrices.
/// \tparam Real Underlying real precision.
template <uni20::LapackReal Real> struct RightComplexSchurDecomposition
{
    RightMatrix<uni20::complex<Real>> schur_form;
    RightMatrix<uni20::complex<Real>> schur_vectors;
    std::vector<uni20::complex<Real>> eigenvalues;
};

template <uni20::LapackReal Scalar> struct RealSchurDecomposition
{
    uni20::DenseMatrix<Scalar> schur_form;
    uni20::DenseMatrix<Scalar> schur_vectors;
    std::vector<uni20::complex<Scalar>> eigenvalues;
    std::vector<RealSchurBlock<Scalar>> blocks;
};

/// \brief Real Schur decomposition stored in prototype layout-right matrices.
/// \tparam Scalar Real scalar type.
template <uni20::LapackReal Scalar> struct RightRealSchurDecomposition
{
    RightMatrix<Scalar> schur_form;
    RightMatrix<Scalar> schur_vectors;
    std::vector<uni20::complex<Scalar>> eigenvalues;
    std::vector<RealSchurBlock<Scalar>> blocks;
};

namespace detail
{
template <uni20::LapackReal Real> uni20::complex<Real> generalized_eigenvalue(Real alphar, Real alphai, Real beta)
{
  if (beta == Real{})
  {
    Real const infinity = uni20::numeric_limits<Real>::infinity();
    Real const real_part = alphar < Real{} ? -infinity : infinity;
    Real const imaginary_part = alphai == Real{} ? Real{} : (alphai < Real{} ? -infinity : infinity);
    return uni20::complex<Real>{real_part, imaginary_part};
  }
  return uni20::complex<Real>{alphar, alphai} / beta;
}

template <uni20::LapackReal Scalar>
std::vector<RealSchurBlock<Scalar>> real_schur_blocks(std::vector<uni20::complex<Scalar>> const& eigenvalues)
{
  std::vector<RealSchurBlock<Scalar>> blocks;
  for (std::size_t index = 0; index < eigenvalues.size();)
  {
    if (eigenvalues[index].imag() == Scalar{})
    {
      blocks.push_back(RealSchurBlock<Scalar>{
          .begin = index, .size = 1, .first_eigenvalue = eigenvalues[index], .second_eigenvalue = {}});
      ++index;
      continue;
    }

    if (index + 1 >= eigenvalues.size())
    {
      throw std::runtime_error("LAPACK gees returned an incomplete complex Schur block");
    }
    if (detail::adl_abs(eigenvalues[index + 1] - std::conj(eigenvalues[index])) >
        Scalar{100} * uni20::numeric_limits<Scalar>::epsilon() *
            std::max(Scalar{1}, std::max(detail::adl_abs(eigenvalues[index]), detail::adl_abs(eigenvalues[index + 1]))))
    {
      throw std::runtime_error("LAPACK gees returned a non-adjacent complex conjugate Schur block");
    }

    blocks.push_back(RealSchurBlock<Scalar>{.begin = index,
                                            .size = 2,
                                            .first_eigenvalue = eigenvalues[index],
                                            .second_eigenvalue = eigenvalues[index + 1]});
    index += 2;
  }
  return blocks;
}

template <uni20::LapackReal Scalar>
std::vector<uni20::complex<Scalar>> eigenvalues_from_schur_blocks(std::vector<RealSchurBlock<Scalar>> const& blocks)
{
  std::vector<uni20::complex<Scalar>> eigenvalues;
  for (auto const& block : blocks)
  {
    eigenvalues.push_back(block.first_eigenvalue);
    if (block.size == 2)
    {
      eigenvalues.push_back(block.second_eigenvalue);
    }
  }
  return eigenvalues;
}

template <typename Value, uni20::LapackReal Scalar>
std::vector<Value> ordered_schur_block_values(std::vector<Value> const& values,
                                              std::vector<RealSchurBlock<Scalar>> const& original_blocks,
                                              std::vector<std::size_t> const& order)
{
  std::vector<Value> ordered;
  ordered.reserve(values.size());
  for (std::size_t const index : order)
  {
    if (index >= original_blocks.size())
    {
      throw std::invalid_argument("real Schur block value order contains an out-of-range block index");
    }
    RealSchurBlock<Scalar> const& block = original_blocks[index];
    for (std::size_t offset = 0; offset < block.size; ++offset)
    {
      ordered.push_back(values[block.begin + offset]);
    }
  }
  return ordered;
}

template <uni20::LapackReal Scalar>
std::vector<RealSchurBlock<Scalar>> ordered_schur_blocks(std::vector<RealSchurBlock<Scalar>> const& original_blocks,
                                                         std::vector<std::size_t> const& order)
{
  std::vector<RealSchurBlock<Scalar>> ordered;
  ordered.reserve(order.size());
  std::size_t begin = 0;
  for (std::size_t const index : order)
  {
    if (index >= original_blocks.size())
    {
      throw std::invalid_argument("real Schur block order contains an out-of-range block index");
    }
    auto block = original_blocks[index];
    block.begin = begin;
    begin += block.size;
    ordered.push_back(block);
  }
  return ordered;
}

inline std::vector<std::size_t> complete_leading_block_order(std::size_t block_count,
                                                             std::vector<std::size_t> const& leading_block_order)
{
  std::vector<bool> seen(block_count, false);
  std::vector<std::size_t> order;
  order.reserve(block_count);
  for (std::size_t const index : leading_block_order)
  {
    if (index >= block_count)
    {
      throw std::invalid_argument("real Schur leading block order contains an out-of-range block index");
    }
    if (seen[index])
    {
      throw std::invalid_argument("real Schur leading block order contains a duplicate block index");
    }
    seen[index] = true;
    order.push_back(index);
  }
  for (std::size_t index = 0; index < block_count; ++index)
  {
    if (!seen[index])
    {
      order.push_back(index);
    }
  }
  return order;
}

inline std::vector<std::size_t> complete_leading_index_order(std::size_t count,
                                                             std::vector<std::size_t> const& leading_order)
{
  std::vector<bool> seen(count, false);
  std::vector<std::size_t> order;
  order.reserve(count);
  for (std::size_t const index : leading_order)
  {
    if (index >= count)
    {
      throw std::invalid_argument("Schur leading order contains an out-of-range index");
    }
    if (seen[index])
    {
      throw std::invalid_argument("Schur leading order contains a duplicate index");
    }
    seen[index] = true;
    order.push_back(index);
  }
  for (std::size_t index = 0; index < count; ++index)
  {
    if (!seen[index])
    {
      order.push_back(index);
    }
  }
  return order;
}

} // namespace detail

template <uni20::LapackReal Scalar>
std::vector<Scalar> symmetric_tridiagonal_eigenvalues(std::vector<Scalar> diagonal, std::vector<Scalar> subdiagonal)
{
  std::size_t const n = diagonal.size();
  if (subdiagonal.size() + 1 != n && !(n == 0 && subdiagonal.empty()))
  {
    throw std::invalid_argument("symmetric_tridiagonal_eigenvalues received inconsistent diagonal sizes");
  }
  if (n == 0)
  {
    return diagonal;
  }

  uni20::DenseMatrix<Scalar> no_eigenvectors(0, 0);
  uni20::linalg::symmetric_tridiagonal_eigen(std::span<Scalar>(diagonal), std::span<Scalar>(subdiagonal),
                                             no_eigenvectors, false);
  return diagonal;
}

/// \brief Solve a real symmetric tridiagonal eigensystem through LAPACK `sterf` or `steqr`.
///
/// \details `diagonal` has length `n`; `subdiagonal` has length `n - 1`.
///          Eigenvalues are returned in ascending order. If `compute_vectors`
///          is false, LAPACK `sterf` is used. If `compute_vectors` is true,
///          LAPACK `steqr` is used and `eigenvectors` is an `n`-by-`n`
///          column-major matrix whose columns are the normalized eigenvectors.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param diagonal Main diagonal of the tridiagonal matrix.
/// \param subdiagonal Subdiagonal/superdiagonal entries.
/// \param compute_vectors Whether to compute eigenvectors.
/// \return Eigenvalues and, optionally, eigenvectors.
template <uni20::LapackReal Scalar>
TridiagonalEigensystem<Scalar> symmetric_tridiagonal_eigensystem(std::vector<Scalar> diagonal,
                                                                 std::vector<Scalar> subdiagonal, bool compute_vectors)
{
  std::size_t const n = diagonal.size();
  if (subdiagonal.size() + 1 != n && !(n == 0 && subdiagonal.empty()))
  {
    throw std::invalid_argument("symmetric_tridiagonal_eigensystem received inconsistent diagonal sizes");
  }

  TridiagonalEigensystem<Scalar> result;
  if (!compute_vectors)
  {
    result.eigenvalues = symmetric_tridiagonal_eigenvalues(std::move(diagonal), std::move(subdiagonal));
    result.eigenvectors = uni20::DenseMatrix<Scalar>(0, 0);
    return result;
  }

  result.eigenvalues = std::move(diagonal);
  if (n == 0)
  {
    result.eigenvectors = uni20::DenseMatrix<Scalar>(0, 0);
    return result;
  }

  uni20::DenseMatrix<Scalar> z(n, n);
  uni20::linalg::symmetric_tridiagonal_eigen(std::span<Scalar>(result.eigenvalues), std::span<Scalar>(subdiagonal), z,
                                             true);

  result.eigenvectors = std::move(z);
  return result;
}

/// \brief Solve a real symmetric tridiagonal eigensystem through LAPACK `stevd`.
///
/// \details Uses the divide-and-conquer symmetric tridiagonal eigensystem
///          driver. `diagonal` has length `n`; `subdiagonal` has length
///          `n - 1`. Eigenvalues are returned in ascending order. If
///          `compute_vectors` is true, `eigenvectors` is an `n`-by-`n`
///          column-major matrix whose columns are the normalized eigenvectors.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param diagonal Main diagonal of the tridiagonal matrix.
/// \param subdiagonal Subdiagonal/superdiagonal entries.
/// \param compute_vectors Whether to compute eigenvectors.
/// \return Eigenvalues and, optionally, eigenvectors.

template <uni20::LapackReal Scalar>
RightRealSchurDecomposition<Scalar> real_schur_layout_right(RightMatrix<Scalar> matrix, bool compute_vectors)
{
  if (matrix.rows() != matrix.cols())
  {
    throw std::invalid_argument("real_schur_layout_right requires a square matrix");
  }

  std::size_t const n = static_cast<std::size_t>(matrix.rows());
  RightRealSchurDecomposition<Scalar> result;
  result.eigenvalues.resize(n);
  if (n == 0)
  {
    result.schur_form = RightMatrix<Scalar>(0, 0);
    result.schur_vectors = RightMatrix<Scalar>(0, 0);
    return result;
  }

  uni20::DenseMatrix<Scalar> lapack_matrix = copy_right_to_left(matrix);
  uni20::DenseMatrix<Scalar> schur_vectors_left(compute_vectors ? n : 0, compute_vectors ? n : 0);
  uni20::linalg::schur(lapack_matrix, std::span<uni20::complex<Scalar>>(result.eigenvalues), schur_vectors_left,
                       compute_vectors);

  result.blocks = detail::real_schur_blocks(result.eigenvalues);
  result.schur_form = copy_left_to_right(lapack_matrix);
  result.schur_vectors = compute_vectors ? copy_left_to_right(schur_vectors_left) : RightMatrix<Scalar>(0, 0);
  return result;
}

/// \brief Compute the real Schur decomposition of a column-major DenseMatrix.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real square matrix in current Krylov local storage.
/// \param compute_vectors Whether to compute Schur vectors.
/// \return Real Schur form, optional Schur vectors, eigenvalues, and block metadata.
template <uni20::LapackReal Scalar>
RealSchurDecomposition<Scalar> real_schur(uni20::DenseMatrix<Scalar> matrix, bool compute_vectors)
{
  RealSchurDecomposition<Scalar> result;
  if (matrix.rows() != matrix.cols())
  {
    throw std::invalid_argument("real_schur requires a square matrix");
  }
  std::size_t const n = static_cast<std::size_t>(matrix.rows());

  result.schur_form = std::move(matrix);
  result.schur_vectors = uni20::DenseMatrix<Scalar>(compute_vectors ? n : 0, compute_vectors ? n : 0);
  result.eigenvalues.resize(n);
  if (n != 0)
  {
    uni20::linalg::schur(result.schur_form, std::span<uni20::complex<Scalar>>(result.eigenvalues), result.schur_vectors,
                         compute_vectors);
    result.blocks = detail::real_schur_blocks(result.eigenvalues);
  }
  return result;
}

/// \brief Compute the real Schur decomposition of an upper Hessenberg matrix through LAPACK `hseqr`.
/// \details This helper does not reduce a dense matrix to Hessenberg form. It
///          is intended for projected Arnoldi matrices that are already upper
///          Hessenberg, avoiding an unnecessary `gees` reduction step.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param hessenberg Upper Hessenberg matrix in column-major local storage.
/// \param compute_vectors Whether to compute Schur vectors of the Hessenberg matrix.
/// \return Real Schur form, optional Schur vectors, eigenvalues, and block metadata.
template <uni20::LapackReal Scalar>
RealSchurDecomposition<Scalar> real_hessenberg_schur(uni20::DenseMatrix<Scalar> hessenberg, bool compute_vectors)
{
  if (hessenberg.rows() != hessenberg.cols())
  {
    throw std::invalid_argument("real_hessenberg_schur requires a square matrix");
  }

  std::size_t const n = static_cast<std::size_t>(hessenberg.rows());
  RealSchurDecomposition<Scalar> result;
  result.eigenvalues.resize(n);
  if (n == 0)
  {
    result.schur_form = uni20::DenseMatrix<Scalar>(0, 0);
    result.schur_vectors = uni20::DenseMatrix<Scalar>(0, 0);
    return result;
  }

  uni20::DenseMatrix<Scalar> schur_vectors(compute_vectors ? n : 0, compute_vectors ? n : 0);
  uni20::linalg::hessenberg_schur(hessenberg, std::span<uni20::complex<Scalar>>(result.eigenvalues), schur_vectors,
                                  compute_vectors);
  result.blocks = detail::real_schur_blocks(result.eigenvalues);
  result.schur_form = std::move(hessenberg);
  result.schur_vectors = compute_vectors ? std::move(schur_vectors) : uni20::DenseMatrix<Scalar>(0, 0);
  return result;
}

/// \brief Compute the complex Schur decomposition of a layout-right matrix through LAPACK `gees`.
///
/// \details The public wrapper accepts the prototype row-major `RightMatrix`
///          and copies to a column-major DenseMatrix before
///          calling Fortran LAPACK. Complex Schur blocks are all scalar 1x1
///          entries, so eigenvalues are the diagonal of the Schur form.
/// \tparam Real `float` or `double`.
/// \param matrix Complex square matrix in logical layout-right storage.
/// \param compute_vectors Whether to compute Schur vectors.
/// \return Complex Schur form, optional Schur vectors, and eigenvalues.
template <uni20::LapackReal Real>
RightComplexSchurDecomposition<Real> complex_schur_layout_right(RightMatrix<uni20::complex<Real>> matrix,
                                                                bool compute_vectors)
{
  if (matrix.rows() != matrix.cols())
  {
    throw std::invalid_argument("complex_schur_layout_right requires a square matrix");
  }

  using Complex = uni20::complex<Real>;

  std::size_t const n = static_cast<std::size_t>(matrix.rows());
  RightComplexSchurDecomposition<Real> result;
  result.eigenvalues.resize(n);
  if (n == 0)
  {
    result.schur_form = RightMatrix<Complex>(0, 0);
    result.schur_vectors = RightMatrix<Complex>(0, 0);
    return result;
  }

  uni20::DenseMatrix<Complex> lapack_matrix = copy_right_to_left(matrix);
  uni20::DenseMatrix<Complex> schur_vectors_left(compute_vectors ? n : 0, compute_vectors ? n : 0);
  uni20::linalg::schur(lapack_matrix, std::span<Complex>(result.eigenvalues), schur_vectors_left, compute_vectors);

  result.schur_form = copy_left_to_right(lapack_matrix);
  result.schur_vectors = compute_vectors ? copy_left_to_right(schur_vectors_left) : RightMatrix<Complex>(0, 0);
  return result;
}

/// \brief Compute the complex Schur decomposition of a column-major DenseMatrix.
/// \tparam Real `float` or `double`.
/// \param matrix Complex square matrix in current Krylov local storage.
/// \param compute_vectors Whether to compute Schur vectors.
/// \return Complex Schur form, optional Schur vectors, and eigenvalues.
template <uni20::LapackReal Real>
ComplexSchurDecomposition<Real> complex_schur(uni20::DenseMatrix<uni20::complex<Real>> matrix, bool compute_vectors)
{
  using Complex = uni20::complex<Real>;
  ComplexSchurDecomposition<Real> result;
  if (matrix.rows() != matrix.cols())
  {
    throw std::invalid_argument("complex_schur requires a square matrix");
  }
  std::size_t const n = static_cast<std::size_t>(matrix.rows());

  result.schur_form = std::move(matrix);
  result.schur_vectors = uni20::DenseMatrix<Complex>(compute_vectors ? n : 0, compute_vectors ? n : 0);
  result.eigenvalues.resize(n);
  if (n != 0)
  {
    uni20::linalg::schur(result.schur_form, std::span<Complex>(result.eigenvalues), result.schur_vectors,
                         compute_vectors);
  }
  return result;
}

/// \brief Reorder a layout-right real Schur decomposition by moving selected blocks to the front.
///
/// \details `leading_block_order` names blocks in the input decomposition's
///          current block order. Any blocks not named are left after the
///          requested prefix in their original relative order. Complex
///          conjugate 2x2 real Schur blocks are moved as indivisible blocks.
///          The Schur form and, when present, Schur vectors are updated through
///          LAPACK `trexc`.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param decomposition Real Schur decomposition to reorder.
/// \param leading_block_order Input block indices to move to the leading positions.
/// \return Reordered real Schur decomposition.
template <uni20::LapackReal Scalar>
RightRealSchurDecomposition<Scalar> reorder_real_schur_layout_right(RightRealSchurDecomposition<Scalar> decomposition,
                                                                    std::vector<std::size_t> const& leading_block_order)
{
  if (decomposition.schur_form.rows() != decomposition.schur_form.cols())
  {
    throw std::invalid_argument("reorder_real_schur_layout_right requires a square Schur form");
  }

  auto const order = decomposition.schur_form.rows();
  std::size_t const n = static_cast<std::size_t>(order);
  if (n == 0 || leading_block_order.empty())
  {
    return decomposition;
  }
  if (decomposition.schur_form.cols() != order)
  {
    throw std::invalid_argument("reorder_real_schur_layout_right received an inconsistent Schur form");
  }
  bool const update_vectors = decomposition.schur_vectors.rows() != 0 || decomposition.schur_vectors.cols() != 0;
  if (update_vectors && (decomposition.schur_vectors.rows() != order || decomposition.schur_vectors.cols() != order))
  {
    throw std::invalid_argument("reorder_real_schur_layout_right received inconsistent Schur vectors");
  }

  std::vector<std::size_t> target_order =
      detail::complete_leading_block_order(decomposition.blocks.size(), leading_block_order);
  std::vector<std::size_t> current_order(decomposition.blocks.size());
  std::iota(current_order.begin(), current_order.end(), std::size_t{0});
  std::vector<RealSchurBlock<Scalar>> current_blocks = decomposition.blocks;

  uni20::DenseMatrix<Scalar> schur_form = copy_right_to_left(decomposition.schur_form);
  uni20::DenseMatrix<Scalar> schur_vectors =
      update_vectors ? copy_right_to_left(decomposition.schur_vectors) : uni20::DenseMatrix<Scalar>(0, 0);

  for (std::size_t target_position = 0; target_position < leading_block_order.size(); ++target_position)
  {
    std::size_t const desired_original_index = target_order[target_position];
    auto const current_it = std::ranges::find(current_order, desired_original_index);
    if (current_it == current_order.end())
    {
      throw std::logic_error("real Schur block reorder lost a block index");
    }
    std::size_t const current_position = static_cast<std::size_t>(current_it - current_order.begin());
    if (current_position == target_position)
    {
      continue;
    }

    uni20::linalg::reorder_schur(schur_form, schur_vectors, current_blocks[current_position].begin,
                                 current_blocks[target_position].begin, update_vectors);

    std::size_t const moved_index = current_order[current_position];
    current_order.erase(current_order.begin() + static_cast<std::ptrdiff_t>(current_position));
    current_order.insert(current_order.begin() + static_cast<std::ptrdiff_t>(target_position), moved_index);
    current_blocks = detail::ordered_schur_blocks(decomposition.blocks, current_order);
  }

  decomposition.schur_form = copy_left_to_right(schur_form);
  decomposition.schur_vectors = update_vectors ? copy_left_to_right(schur_vectors) : RightMatrix<Scalar>(0, 0);
  decomposition.blocks = detail::ordered_schur_blocks(decomposition.blocks, current_order);
  decomposition.eigenvalues = detail::eigenvalues_from_schur_blocks(decomposition.blocks);
  return decomposition;
}

/// \brief Reorder a column-major real Schur decomposition by moving selected blocks to the front.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param decomposition Real Schur decomposition to reorder.
/// \param leading_block_order Input block indices to move to the leading positions.
/// \return Reordered real Schur decomposition.
template <uni20::LapackReal Scalar>
RealSchurDecomposition<Scalar> reorder_real_schur(RealSchurDecomposition<Scalar> decomposition,
                                                  std::vector<std::size_t> const& leading_block_order)
{
  RightRealSchurDecomposition<Scalar> right;
  right.schur_form = copy_left_to_right(decomposition.schur_form);
  right.schur_vectors = copy_left_to_right(decomposition.schur_vectors);
  right.eigenvalues = std::move(decomposition.eigenvalues);
  right.blocks = std::move(decomposition.blocks);

  right = reorder_real_schur_layout_right(std::move(right), leading_block_order);

  RealSchurDecomposition<Scalar> result;
  result.schur_form = copy_right_to_left(right.schur_form);
  result.schur_vectors = copy_right_to_left(right.schur_vectors);
  result.eigenvalues = std::move(right.eigenvalues);
  result.blocks = std::move(right.blocks);
  return result;
}

/// \brief Move selected real Schur blocks to the leading invariant subspace and estimate conditioning.
/// \details Uses LAPACK `trsen` with `JOB='B'`, returning reciprocal condition
///          estimates for the selected eigenvalue cluster and invariant
///          subspace. Selection is by Schur block, so real-arithmetic 2x2
///          complex-conjugate blocks cannot be split accidentally.
/// \tparam Scalar Real scalar type satisfying `uni20::LapackReal`.
/// \param decomposition Real Schur decomposition to reorder and diagnose.

template <uni20::LapackReal Real>
RightComplexSchurDecomposition<Real>
reorder_complex_schur_layout_right(RightComplexSchurDecomposition<Real> decomposition,
                                   std::vector<std::size_t> const& leading_order)
{
  if (decomposition.schur_form.rows() != decomposition.schur_form.cols())
  {
    throw std::invalid_argument("reorder_complex_schur_layout_right requires a square Schur form");
  }

  using Complex = uni20::complex<Real>;

  auto const order = decomposition.schur_form.rows();
  std::size_t const n = static_cast<std::size_t>(order);
  if (n == 0 || leading_order.empty())
  {
    return decomposition;
  }
  if (decomposition.schur_form.cols() != order || decomposition.eigenvalues.size() != n)
  {
    throw std::invalid_argument("reorder_complex_schur_layout_right received inconsistent Schur data");
  }
  bool const update_vectors = decomposition.schur_vectors.rows() != 0 || decomposition.schur_vectors.cols() != 0;
  if (update_vectors && (decomposition.schur_vectors.rows() != order || decomposition.schur_vectors.cols() != order))
  {
    throw std::invalid_argument("reorder_complex_schur_layout_right received inconsistent Schur vectors");
  }

  std::vector<std::size_t> target_order = detail::complete_leading_index_order(n, leading_order);
  std::vector<std::size_t> current_order(n);
  std::iota(current_order.begin(), current_order.end(), std::size_t{0});

  uni20::DenseMatrix<Complex> schur_form = copy_right_to_left(decomposition.schur_form);
  uni20::DenseMatrix<Complex> schur_vectors =
      update_vectors ? copy_right_to_left(decomposition.schur_vectors) : uni20::DenseMatrix<Complex>(0, 0);

  for (std::size_t target_position = 0; target_position < leading_order.size(); ++target_position)
  {
    std::size_t const desired_original_index = target_order[target_position];
    auto const current_it = std::ranges::find(current_order, desired_original_index);
    if (current_it == current_order.end())
    {
      throw std::logic_error("complex Schur reorder lost an index");
    }
    std::size_t const current_position = static_cast<std::size_t>(current_it - current_order.begin());
    if (current_position == target_position)
    {
      continue;
    }

    uni20::linalg::reorder_schur(schur_form, schur_vectors, current_position, target_position, update_vectors);

    std::size_t const moved_index = current_order[current_position];
    current_order.erase(current_order.begin() + static_cast<std::ptrdiff_t>(current_position));
    current_order.insert(current_order.begin() + static_cast<std::ptrdiff_t>(target_position), moved_index);
  }

  decomposition.schur_form = copy_left_to_right(schur_form);
  decomposition.schur_vectors = update_vectors ? copy_left_to_right(schur_vectors) : RightMatrix<Complex>(0, 0);

  std::vector<Complex> reordered_eigenvalues;
  reordered_eigenvalues.reserve(n);
  for (std::size_t const index : current_order)
  {
    reordered_eigenvalues.push_back(decomposition.eigenvalues[index]);
  }
  decomposition.eigenvalues = std::move(reordered_eigenvalues);
  return decomposition;
}

/// \brief Reorder a column-major complex Schur decomposition by moving selected entries to the front.
/// \tparam Real `float` or `double`.
/// \param decomposition Complex Schur decomposition to reorder.
/// \param leading_order Input entry indices to move to the leading positions.
/// \return Reordered complex Schur decomposition.
template <uni20::LapackReal Real>
ComplexSchurDecomposition<Real> reorder_complex_schur(ComplexSchurDecomposition<Real> decomposition,
                                                      std::vector<std::size_t> const& leading_order)
{
  RightComplexSchurDecomposition<Real> right;
  right.schur_form = copy_left_to_right(decomposition.schur_form);
  right.schur_vectors = copy_left_to_right(decomposition.schur_vectors);
  right.eigenvalues = std::move(decomposition.eigenvalues);

  right = reorder_complex_schur_layout_right(std::move(right), leading_order);

  ComplexSchurDecomposition<Real> result;
  result.schur_form = copy_right_to_left(right.schur_form);
  result.schur_vectors = copy_right_to_left(right.schur_vectors);
  result.eigenvalues = std::move(right.eigenvalues);
  return result;
}

template <uni20::LapackReal Real>
RealNonsymmetricEigensystem<Real> real_nonsymmetric_eigensystem(uni20::DenseMatrix<Real> matrix,
                                                                bool compute_right_vectors)
{
  if (matrix.rows() != matrix.cols())
  {
    throw std::invalid_argument("real_nonsymmetric_eigensystem requires a square matrix");
  }

  std::size_t const n = static_cast<std::size_t>(matrix.rows());
  RealNonsymmetricEigensystem<Real> result;
  result.eigenvalues.resize(n);
  result.right_eigenvectors =
      uni20::DenseMatrix<uni20::complex<Real>>(compute_right_vectors ? n : 0, compute_right_vectors ? n : 0);
  if (n == 0)
  {
    return result;
  }

  uni20::linalg::nonsymmetric_eigen(matrix, std::span<uni20::complex<Real>>(result.eigenvalues),
                                    result.right_eigenvectors, compute_right_vectors);

  return result;
}

/// \brief Solve a dense real nonsymmetric eigensystem through LAPACK `geevx`.
///
/// \details The expert driver balances the matrix (`BALANC='B'`) and computes
///          reciprocal condition estimates for both eigenvalues and right
///          eigenvectors (`SENSE='B'`). It computes left and right eigenvectors
///          internally because LAPACK needs them for these estimates. The input
///          matrix is copied because LAPACK overwrites it. If right
///          eigenvectors are requested, real eigenvectors are returned as
///          complex vectors with zero imaginary part, and complex conjugate
///          pairs are unpacked from LAPACK's adjacent real-vector
///          representation.
/// \tparam Real Real scalar type satisfying `uni20::LapackReal`.
/// \param matrix Real square matrix in column-major local storage.
/// \param compute_right_vectors Whether to compute right eigenvectors.
/// \return Complex eigenvalues, optional right eigenvectors, balancing data,
///         and reciprocal condition estimates.

template <uni20::LapackReal Real>
ComplexNonsymmetricEigensystem<Real> complex_nonsymmetric_eigensystem(uni20::DenseMatrix<uni20::complex<Real>> matrix,
                                                                      bool compute_right_vectors)
{
  if (matrix.rows() != matrix.cols())
  {
    throw std::invalid_argument("complex_nonsymmetric_eigensystem requires a square matrix");
  }

  using Complex = uni20::complex<Real>;

  std::size_t const n = static_cast<std::size_t>(matrix.rows());
  ComplexNonsymmetricEigensystem<Real> result;
  result.eigenvalues.resize(n);
  result.right_eigenvectors = uni20::DenseMatrix<Complex>(compute_right_vectors ? n : 0, compute_right_vectors ? n : 0);
  if (n == 0)
  {
    return result;
  }

  uni20::linalg::nonsymmetric_eigen(matrix, std::span<Complex>(result.eigenvalues), result.right_eigenvectors,
                                    compute_right_vectors);
  return result;
}

} // namespace uni20::krylov
