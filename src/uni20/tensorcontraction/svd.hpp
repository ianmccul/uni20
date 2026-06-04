#pragma once

#include <uni20/config.hpp>
#include <uni20/tensorcontraction/matrix_family.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace uni20::tensorcontraction
{

class VectorAlgebraEngine;

struct SvdOptions
{
    std::size_t max_rank = std::numeric_limits<std::size_t>::max();
    double cutoff = 0.0;
};

/// \brief Singular-value spectrum and truncation metadata for an SVD-derived operation.
struct SvdSpectrum
{
    std::vector<double> singular_values;
    double discarded_weight = 0.0;
    std::size_t full_rank = 0;
};

struct SingleBlockSvd
{
    MatrixFamily u;
    std::vector<double> singular_values;
    MatrixFamily vt;
    double discarded_weight = 0.0;
    std::size_t full_rank = 0;
};

/// \brief Selects which split factor receives the singular values.
enum class SvdAbsorbSingularValues
{
  Left,
  Right,
};

/// \brief Dense two-site layout used to split a single matrix block into physical site blocks.
struct SingleBlockSvdSplitLayout
{
    std::size_t left_bond_dim = 0;
    std::size_t left_physical_dim = 0;
    std::size_t right_physical_dim = 0;
    std::size_t right_bond_dim = 0;
};

/// \brief Truncated two-site split result without host copies of full SVD factors.
struct SingleBlockSvdSplit
{
    MatrixFamily left;
    MatrixFamily right;
    SvdSpectrum spectrum;
};

namespace detail
{

struct svd_op
{};

struct reference_svd_capability
{};

struct lapack_svd_capability
{};

struct cusolver_svd_capability
{};

template <typename... Capabilities> struct backend_stack
{
    using capabilities = std::tuple<Capabilities...>;
};

#if UNI20_TENSORCONTRACTION_HAS_CUSOLVER
#if UNI20_TENSORCONTRACTION_HAS_LAPACK
using default_svd_backend = backend_stack<cusolver_svd_capability, lapack_svd_capability, reference_svd_capability>;
#else
using default_svd_backend = backend_stack<cusolver_svd_capability, reference_svd_capability>;
#endif
#else
#if UNI20_TENSORCONTRACTION_HAS_LAPACK
using default_svd_backend = backend_stack<lapack_svd_capability, reference_svd_capability>;
#else
using default_svd_backend = backend_stack<reference_svd_capability>;
#endif
#endif

struct SymmetricEigenSystem
{
    std::vector<double> eigenvalues;
    std::vector<double> eigenvectors;
};

inline std::size_t svd_dense_index(std::size_t n, std::size_t row, std::size_t col) { return row * n + col; }

inline SymmetricEigenSystem symmetric_eigensystem(std::vector<double> matrix, std::size_t n)
{
  if (n == 0)
  {
    throw std::invalid_argument("SVD eigensolver requires a non-empty matrix");
  }

  std::vector<double> eigenvectors(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i)
  {
    eigenvectors[svd_dense_index(n, i, i)] = 1.0;
  }

  auto const max_sweeps = std::max<std::size_t>(32, 32 * n * n);
  for (std::size_t sweep = 0; sweep < max_sweeps; ++sweep)
  {
    std::size_t p = 0;
    std::size_t q = 0;
    double max_offdiag = 0.0;
    for (std::size_t row = 0; row < n; ++row)
    {
      for (std::size_t col = row + 1; col < n; ++col)
      {
        double const value = std::abs(matrix[svd_dense_index(n, row, col)]);
        if (value > max_offdiag)
        {
          max_offdiag = value;
          p = row;
          q = col;
        }
      }
    }

    if (max_offdiag <= 100.0 * std::numeric_limits<double>::epsilon())
    {
      break;
    }

    double const app = matrix[svd_dense_index(n, p, p)];
    double const aqq = matrix[svd_dense_index(n, q, q)];
    double const apq = matrix[svd_dense_index(n, p, q)];
    double const tau = (aqq - app) / (2.0 * apq);
    double const sign = tau < 0.0 ? -1.0 : 1.0;
    double const t = sign / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
    double const c = 1.0 / std::sqrt(1.0 + t * t);
    double const s = t * c;

    for (std::size_t k = 0; k < n; ++k)
    {
      if (k == p || k == q)
      {
        continue;
      }
      double const akp = matrix[svd_dense_index(n, k, p)];
      double const akq = matrix[svd_dense_index(n, k, q)];
      matrix[svd_dense_index(n, k, p)] = matrix[svd_dense_index(n, p, k)] = c * akp - s * akq;
      matrix[svd_dense_index(n, k, q)] = matrix[svd_dense_index(n, q, k)] = s * akp + c * akq;
    }

    matrix[svd_dense_index(n, p, p)] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
    matrix[svd_dense_index(n, q, q)] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
    matrix[svd_dense_index(n, p, q)] = 0.0;
    matrix[svd_dense_index(n, q, p)] = 0.0;

    for (std::size_t k = 0; k < n; ++k)
    {
      double const vkp = eigenvectors[svd_dense_index(n, k, p)];
      double const vkq = eigenvectors[svd_dense_index(n, k, q)];
      eigenvectors[svd_dense_index(n, k, p)] = c * vkp - s * vkq;
      eigenvectors[svd_dense_index(n, k, q)] = s * vkp + c * vkq;
    }
  }

  SymmetricEigenSystem result;
  result.eigenvalues.resize(n);
  result.eigenvectors = std::move(eigenvectors);
  for (std::size_t i = 0; i < n; ++i)
  {
    result.eigenvalues[i] = matrix[svd_dense_index(n, i, i)];
  }
  return result;
}

inline std::vector<double> normal_equations(std::span<double const> values, std::size_t rows, std::size_t cols)
{
  std::vector<double> gram(cols * cols, 0.0);
  for (std::size_t row = 0; row < rows; ++row)
  {
    for (std::size_t left = 0; left < cols; ++left)
    {
      for (std::size_t right = 0; right < cols; ++right)
      {
        gram[svd_dense_index(cols, left, right)] += values[row * cols + left] * values[row * cols + right];
      }
    }
  }
  return gram;
}

inline void validate_single_block_svd_inputs(MatrixFamily const& matrix, SvdOptions options)
{
  if (matrix.size() != 1)
  {
    throw std::invalid_argument("single_block_svd requires exactly one MatrixFamily block");
  }
  if (options.max_rank == 0)
  {
    throw std::invalid_argument("single_block_svd requires a positive max_rank");
  }
  if (options.cutoff < 0.0 || std::isnan(options.cutoff))
  {
    throw std::invalid_argument("single_block_svd requires a finite non-negative cutoff");
  }

  auto const block = matrix.block(0);
  if (block.rows == 0 || block.cols == 0)
  {
    throw std::invalid_argument("single_block_svd requires a non-empty block");
  }
}

inline std::size_t svd_checked_product(std::size_t lhs, std::size_t rhs, char const* name)
{
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
  {
    throw std::length_error(std::string(name) + " overflows size_t");
  }
  return lhs * rhs;
}

inline void validate_single_block_svd_split_inputs(MatrixFamily const& matrix, SingleBlockSvdSplitLayout layout,
                                                   SvdOptions options)
{
  validate_single_block_svd_inputs(matrix, options);
  if (layout.left_bond_dim == 0 || layout.left_physical_dim == 0 || layout.right_physical_dim == 0 ||
      layout.right_bond_dim == 0)
  {
    throw std::invalid_argument("single_block_svd_split requires non-empty layout dimensions");
  }

  auto const rows = svd_checked_product(layout.left_bond_dim, layout.left_physical_dim, "SVD split row count");
  auto const cols = svd_checked_product(layout.right_physical_dim, layout.right_bond_dim, "SVD split column count");
  if (matrix.block(0) != MatrixFamily::Block{rows, cols})
  {
    throw std::invalid_argument("single_block_svd_split matrix shape does not match the split layout");
  }
}

inline void validate_resident_svd_split_inputs(MatrixFamily const& vector, SingleBlockSvdSplitLayout layout,
                                               SvdOptions options)
{
  if (options.max_rank == 0)
  {
    throw std::invalid_argument("single_block_svd_split_resident requires a positive max_rank");
  }
  if (options.cutoff < 0.0 || std::isnan(options.cutoff))
  {
    throw std::invalid_argument("single_block_svd_split_resident requires a finite non-negative cutoff");
  }
  if (layout.left_bond_dim == 0 || layout.left_physical_dim == 0 || layout.right_physical_dim == 0 ||
      layout.right_bond_dim == 0)
  {
    throw std::invalid_argument("single_block_svd_split_resident requires non-empty layout dimensions");
  }

  auto const block_count = svd_checked_product(layout.left_physical_dim, layout.right_physical_dim,
                                               "resident SVD split two-site block count");
  if (vector.size() != block_count)
  {
    throw std::invalid_argument("single_block_svd_split_resident vector block count does not match the split layout");
  }

  for (std::size_t block = 0; block < vector.size(); ++block)
  {
    if (vector.block(block) != MatrixFamily::Block{layout.left_bond_dim, layout.right_bond_dim})
    {
      throw std::invalid_argument("single_block_svd_split_resident vector block shape does not match the split layout");
    }
  }
}

inline std::size_t singular_rank(std::span<double const> singular_values)
{
  return static_cast<std::size_t>(
      std::count_if(singular_values.begin(), singular_values.end(), [](double s) { return s > 0.0; }));
}

inline std::size_t kept_singular_count(std::span<double const> singular_values, SvdOptions options)
{
  std::size_t kept = 0;
  for (; kept < singular_values.size() && kept < options.max_rank; ++kept)
  {
    if (singular_values[kept] <= options.cutoff)
    {
      break;
    }
  }
  if (kept == 0 && !singular_values.empty() && singular_values.front() > 0.0)
  {
    kept = 1;
  }
  return kept;
}

inline double discarded_singular_weight(std::span<double const> singular_values, std::size_t kept)
{
  double discarded_weight = 0.0;
  for (std::size_t i = kept; i < singular_values.size(); ++i)
  {
    discarded_weight += singular_values[i] * singular_values[i];
  }
  return discarded_weight;
}

} // namespace detail

inline SingleBlockSvd single_block_svd_reference(MatrixFamily const& matrix, SvdOptions options = {})
{
  detail::validate_single_block_svd_inputs(matrix, options);

  auto const block = matrix.block(0);
  auto const rows = block.rows;
  auto const cols = block.cols;

  auto eig = detail::symmetric_eigensystem(detail::normal_equations(matrix.values(0), rows, cols), cols);
  std::vector<std::size_t> order(cols);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&](std::size_t lhs, std::size_t rhs) { return eig.eigenvalues[lhs] > eig.eigenvalues[rhs]; });

  std::vector<double> all_singular_values;
  all_singular_values.reserve(std::min(rows, cols));
  for (std::size_t i = 0; i < std::min(rows, cols); ++i)
  {
    double const value = eig.eigenvalues[order[i]];
    all_singular_values.push_back(std::sqrt(std::max(0.0, value)));
  }

  auto const full_rank = detail::singular_rank(all_singular_values);
  auto const kept = detail::kept_singular_count(all_singular_values, options);
  double const discarded_weight = detail::discarded_singular_weight(all_singular_values, kept);

  std::vector<MatrixFamily::Block> u_blocks{MatrixFamily::Block{rows, kept}};
  std::vector<MatrixFamily::Block> vt_blocks{MatrixFamily::Block{kept, cols}};
  MatrixFamily u(u_blocks);
  MatrixFamily vt(vt_blocks);
  auto u_values = u.values(0);
  auto vt_values = vt.values(0);
  auto const input_values = matrix.values(0);

  std::vector<double> kept_singular_values;
  kept_singular_values.reserve(kept);
  for (std::size_t rank = 0; rank < kept; ++rank)
  {
    auto const eig_col = order[rank];
    double const singular = all_singular_values[rank];
    kept_singular_values.push_back(singular);

    for (std::size_t col = 0; col < cols; ++col)
    {
      vt_values[rank * cols + col] = eig.eigenvectors[detail::svd_dense_index(cols, col, eig_col)];
    }

    double const inv_singular = 1.0 / singular;
    for (std::size_t row = 0; row < rows; ++row)
    {
      double value = 0.0;
      for (std::size_t col = 0; col < cols; ++col)
      {
        value += input_values[row * cols + col] * vt_values[rank * cols + col];
      }
      u_values[row * kept + rank] = value * inv_singular;
    }
  }

  return SingleBlockSvd{.u = std::move(u),
                        .singular_values = std::move(kept_singular_values),
                        .vt = std::move(vt),
                        .discarded_weight = discarded_weight,
                        .full_rank = full_rank};
}

namespace detail
{

inline SingleBlockSvd dispatch(svd_op, reference_svd_capability, MatrixFamily const& matrix, SvdOptions options)
{
  return single_block_svd_reference(matrix, options);
}

#if UNI20_TENSORCONTRACTION_HAS_LAPACK
inline SingleBlockSvd dispatch(svd_op, lapack_svd_capability, MatrixFamily const& matrix, SvdOptions options);
#endif

#if UNI20_TENSORCONTRACTION_HAS_CUSOLVER
std::optional<SingleBlockSvd> single_block_svd_cusolver(MatrixFamily const& matrix, SvdOptions options);
std::optional<SingleBlockSvdSplit> single_block_svd_split_cusolver(MatrixFamily const& matrix,
                                                                   SingleBlockSvdSplitLayout layout,
                                                                   SvdAbsorbSingularValues absorb, SvdOptions options);
std::optional<SingleBlockSvdSplit> single_block_svd_split_resident_cusolver(MatrixFamily const& vector,
                                                                            SingleBlockSvdSplitLayout layout,
                                                                            SvdAbsorbSingularValues absorb,
                                                                            SvdOptions options,
                                                                            VectorAlgebraEngine& algebra);

inline SingleBlockSvd dispatch(svd_op op, cusolver_svd_capability, MatrixFamily const& matrix, SvdOptions options)
{
  if (auto svd = single_block_svd_cusolver(matrix, options); svd.has_value())
  {
    return std::move(*svd);
  }

#if UNI20_TENSORCONTRACTION_HAS_LAPACK
  return dispatch(op, lapack_svd_capability{}, matrix, options);
#else
  return dispatch(op, reference_svd_capability{}, matrix, options);
#endif
}
#endif

#if UNI20_TENSORCONTRACTION_HAS_LAPACK

extern "C" void dgesdd_(char* jobz, ::uni20::blas_int* m, ::uni20::blas_int* n, double* a, ::uni20::blas_int* lda,
                        double* s, double* u, ::uni20::blas_int* ldu, double* vt, ::uni20::blas_int* ldvt, double* work,
                        ::uni20::blas_int* lwork, ::uni20::blas_int* iwork, ::uni20::blas_int* info);

inline ::uni20::blas_int checked_lapack_int(std::size_t value, char const* name)
{
  if (value > static_cast<std::size_t>(std::numeric_limits<::uni20::blas_int>::max()))
  {
    throw std::length_error(std::string(name) + " exceeds LAPACK integer range");
  }
  return static_cast<::uni20::blas_int>(value);
}

inline SingleBlockSvd dispatch(svd_op, lapack_svd_capability, MatrixFamily const& matrix, SvdOptions options)
{
  validate_single_block_svd_inputs(matrix, options);

  auto const block = matrix.block(0);
  auto const rows = block.rows;
  auto const cols = block.cols;
  auto m = checked_lapack_int(rows, "SVD row count");
  auto n = checked_lapack_int(cols, "SVD column count");
  auto minmn = std::min(m, n);
  auto lda = std::max<::uni20::blas_int>(1, m);
  auto ldu = std::max<::uni20::blas_int>(1, m);
  auto ldvt = std::max<::uni20::blas_int>(1, minmn);

  std::vector<double> a(rows * cols);
  auto const input = matrix.values(0);
  for (std::size_t row = 0; row < rows; ++row)
  {
    for (std::size_t col = 0; col < cols; ++col)
    {
      a[col * rows + row] = input[row * cols + col];
    }
  }

  std::vector<double> singular_values(static_cast<std::size_t>(minmn));
  std::vector<double> u(static_cast<std::size_t>(ldu) * static_cast<std::size_t>(minmn));
  std::vector<double> vt(static_cast<std::size_t>(ldvt) * cols);
  std::vector<::uni20::blas_int> iwork(static_cast<std::size_t>(8 * minmn));
  char jobz = 'S';
  ::uni20::blas_int info = 0;
  ::uni20::blas_int lwork = -1;
  double workspace_query = 0.0;

  dgesdd_(&jobz, &m, &n, a.data(), &lda, singular_values.data(), u.data(), &ldu, vt.data(), &ldvt, &workspace_query,
          &lwork, iwork.data(), &info);
  if (info != 0)
  {
    throw std::runtime_error("LAPACK dgesdd workspace query failed with info=" + std::to_string(info));
  }

  lwork = std::max<::uni20::blas_int>(1, static_cast<::uni20::blas_int>(workspace_query));
  std::vector<double> work(static_cast<std::size_t>(lwork));
  dgesdd_(&jobz, &m, &n, a.data(), &lda, singular_values.data(), u.data(), &ldu, vt.data(), &ldvt, work.data(), &lwork,
          iwork.data(), &info);
  if (info < 0)
  {
    throw std::invalid_argument("LAPACK dgesdd rejected argument " + std::to_string(-info));
  }
  if (info > 0)
  {
    return dispatch(svd_op{}, reference_svd_capability{}, matrix, options);
  }

  auto const full_rank = singular_rank(singular_values);
  auto const kept = kept_singular_count(singular_values, options);
  double const discarded_weight = discarded_singular_weight(singular_values, kept);

  std::vector<MatrixFamily::Block> u_blocks{MatrixFamily::Block{rows, kept}};
  std::vector<MatrixFamily::Block> vt_blocks{MatrixFamily::Block{kept, cols}};
  MatrixFamily result_u(u_blocks);
  MatrixFamily result_vt(vt_blocks);
  auto result_u_values = result_u.values(0);
  auto result_vt_values = result_vt.values(0);

  for (std::size_t row = 0; row < rows; ++row)
  {
    for (std::size_t col = 0; col < kept; ++col)
    {
      result_u_values[row * kept + col] = u[col * rows + row];
    }
  }
  for (std::size_t row = 0; row < kept; ++row)
  {
    for (std::size_t col = 0; col < cols; ++col)
    {
      result_vt_values[row * cols + col] = vt[col * static_cast<std::size_t>(ldvt) + row];
    }
  }

  singular_values.resize(kept);
  return SingleBlockSvd{.u = std::move(result_u),
                        .singular_values = std::move(singular_values),
                        .vt = std::move(result_vt),
                        .discarded_weight = discarded_weight,
                        .full_rank = full_rank};
}
#endif

template <typename Capability, typename... Rest>
inline SingleBlockSvd dispatch_first(svd_op op, std::tuple<Capability, Rest...>, MatrixFamily const& matrix,
                                     SvdOptions options)
{
  if constexpr (requires { dispatch(op, Capability{}, matrix, options); })
  {
    return dispatch(op, Capability{}, matrix, options);
  }
  else
  {
    static_assert(sizeof...(Rest) > 0, "no SVD implementation available for backend stack");
    return dispatch_first(op, std::tuple<Rest...>{}, matrix, options);
  }
}

template <typename Backend>
inline SingleBlockSvd dispatch(svd_op op, Backend, MatrixFamily const& matrix, SvdOptions options)
{
  return dispatch_first(op, typename Backend::capabilities{}, matrix, options);
}

} // namespace detail

inline SingleBlockSvd single_block_svd(MatrixFamily const& matrix, SvdOptions options = {})
{
  return detail::dispatch(detail::svd_op{}, detail::default_svd_backend{}, matrix, options);
}

/// \brief Extract spectrum and truncation metadata from a complete host SVD.
/// \param svd Complete SVD result to summarize.
/// \return Spectrum metadata without copying host factor matrices.
inline auto svd_spectrum(SingleBlockSvd const& svd) -> SvdSpectrum
{
  return SvdSpectrum{
      .singular_values = svd.singular_values, .discarded_weight = svd.discarded_weight, .full_rank = svd.full_rank};
}

namespace detail
{

inline auto split_svd_factors_on_host(SingleBlockSvd const& svd, SingleBlockSvdSplitLayout layout,
                                      SvdAbsorbSingularValues absorb) -> SingleBlockSvdSplit
{
  auto const rank = svd.singular_values.size();
  auto const rows = svd_checked_product(layout.left_bond_dim, layout.left_physical_dim, "SVD split row count");
  auto const cols = svd_checked_product(layout.right_physical_dim, layout.right_bond_dim, "SVD split column count");
  if (svd.u.size() != 1 || svd.vt.size() != 1 || svd.u.block(0) != MatrixFamily::Block{rows, rank} ||
      svd.vt.block(0) != MatrixFamily::Block{rank, cols})
  {
    throw std::invalid_argument("single_block_svd_split received SVD factors with incompatible shapes");
  }

  std::vector<MatrixFamily::Block> left_blocks(layout.left_physical_dim,
                                               MatrixFamily::Block{layout.left_bond_dim, rank});
  std::vector<MatrixFamily::Block> right_blocks(layout.right_physical_dim,
                                                MatrixFamily::Block{rank, layout.right_bond_dim});
  MatrixFamily left(left_blocks);
  MatrixFamily right(right_blocks);

  auto const u_values = svd.u.values(0);
  auto const vt_values = svd.vt.values(0);
  for (std::size_t left_phys = 0; left_phys < layout.left_physical_dim; ++left_phys)
  {
    auto dst = left.values(left_phys);
    for (std::size_t left_bond = 0; left_bond < layout.left_bond_dim; ++left_bond)
    {
      auto const row = left_bond * layout.left_physical_dim + left_phys;
      for (std::size_t bond = 0; bond < rank; ++bond)
      {
        double value = u_values[row * rank + bond];
        if (absorb == SvdAbsorbSingularValues::Left)
        {
          value *= svd.singular_values[bond];
        }
        dst[left_bond * rank + bond] = value;
      }
    }
  }

  for (std::size_t right_phys = 0; right_phys < layout.right_physical_dim; ++right_phys)
  {
    auto dst = right.values(right_phys);
    for (std::size_t bond = 0; bond < rank; ++bond)
    {
      for (std::size_t right_bond = 0; right_bond < layout.right_bond_dim; ++right_bond)
      {
        auto const col = right_phys * layout.right_bond_dim + right_bond;
        double value = vt_values[bond * cols + col];
        if (absorb == SvdAbsorbSingularValues::Right)
        {
          value *= svd.singular_values[bond];
        }
        dst[bond * layout.right_bond_dim + right_bond] = value;
      }
    }
  }

  return SingleBlockSvdSplit{.left = std::move(left), .right = std::move(right), .spectrum = svd_spectrum(svd)};
}

} // namespace detail

/// \brief Split one dense SVD block into physical left/right factors.
/// \param matrix Single-block matrix with shape implied by `layout`.
/// \param layout Dense two-site split dimensions.
/// \param absorb Side that receives the singular values.
/// \param options SVD truncation options.
/// \return Split factors and spectrum metadata.
inline auto single_block_svd_split(MatrixFamily const& matrix, SingleBlockSvdSplitLayout layout,
                                   SvdAbsorbSingularValues absorb, SvdOptions options = {}) -> SingleBlockSvdSplit
{
  detail::validate_single_block_svd_split_inputs(matrix, layout, options);

#if UNI20_TENSORCONTRACTION_HAS_CUSOLVER
  if (auto split = detail::single_block_svd_split_cusolver(matrix, layout, absorb, options); split.has_value())
  {
    return std::move(*split);
  }
#endif

  auto svd = single_block_svd(matrix, options);
  return detail::split_svd_factors_on_host(svd, layout, absorb);
}

/// \brief Split a resident two-site block vector without first materializing it on the host.
/// \param vector Resident two-site vector blocks ordered by left/right physical index.
/// \param layout Dense two-site split dimensions.
/// \param absorb Side that receives the singular values.
/// \param options SVD truncation options.
/// \param algebra Resident vector algebra engine that owns the GPU buffers.
/// \return Split factors and spectrum metadata when a resident backend is available.
inline auto single_block_svd_split_resident(MatrixFamily const& vector, SingleBlockSvdSplitLayout layout,
                                            SvdAbsorbSingularValues absorb, SvdOptions options,
                                            VectorAlgebraEngine& algebra) -> std::optional<SingleBlockSvdSplit>
{
  detail::validate_resident_svd_split_inputs(vector, layout, options);

#if UNI20_TENSORCONTRACTION_HAS_CUSOLVER
  return detail::single_block_svd_split_resident_cusolver(vector, layout, absorb, options, algebra);
#else
  (void)vector;
  (void)layout;
  (void)absorb;
  (void)options;
  (void)algebra;
  return std::nullopt;
#endif
}

} // namespace uni20::tensorcontraction
