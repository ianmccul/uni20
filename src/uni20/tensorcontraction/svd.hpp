#pragma once

#include <uni20/tensorcontraction/matrix_family.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace uni20::tensorcontraction
{

struct SvdOptions
{
    std::size_t max_rank = std::numeric_limits<std::size_t>::max();
    double cutoff = 0.0;
};

struct SingleBlockSvd
{
    MatrixFamily u;
    std::vector<double> singular_values;
    MatrixFamily vt;
    double discarded_weight = 0.0;
    std::size_t full_rank = 0;
};

namespace detail
{

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

} // namespace detail

inline SingleBlockSvd single_block_svd(MatrixFamily const& matrix, SvdOptions options = {})
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
  auto const rows = block.rows;
  auto const cols = block.cols;
  if (rows == 0 || cols == 0)
  {
    throw std::invalid_argument("single_block_svd requires a non-empty block");
  }

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

  auto const full_rank = static_cast<std::size_t>(
      std::count_if(all_singular_values.begin(), all_singular_values.end(), [](double s) { return s > 0.0; }));

  std::size_t kept = 0;
  for (; kept < all_singular_values.size() && kept < options.max_rank; ++kept)
  {
    if (all_singular_values[kept] <= options.cutoff)
    {
      break;
    }
  }
  if (kept == 0 && !all_singular_values.empty() && all_singular_values.front() > 0.0)
  {
    kept = 1;
  }

  double discarded_weight = 0.0;
  for (std::size_t i = kept; i < all_singular_values.size(); ++i)
  {
    discarded_weight += all_singular_values[i] * all_singular_values[i];
  }

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

} // namespace uni20::tensorcontraction
