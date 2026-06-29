#pragma once

#include <uni20/core/scalar_concepts.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace uni20::krylov::test_matrices
{

enum class MatrixOracleKind
{
  ExactEigenvalues,
  ExactEigenvaluesAndCondition,
  ExactInertia,
  StressOnly
};

struct KnownMatrixMetadata
{
    std::string_view id;
    std::string_view source;
    std::string_view license;
    std::string_view reference;
    std::string_view purpose;
    MatrixOracleKind oracle;
    bool symmetric;
    bool positive_definite;
    bool sparse;
};

template <uni20::Real Scalar> struct KnownSymmetricMatrix
{
    KnownMatrixMetadata metadata;
    std::size_t dimension = 0;
    std::vector<Scalar> row_major_values;
    std::vector<Scalar> eigenvalues_ascending;
    std::optional<Scalar> two_norm_condition_number;
};

template <uni20::Real Scalar> [[nodiscard]] Scalar matrix_test_pi() { return std::acos(Scalar{-1}); }

template <uni20::Real Scalar>
[[nodiscard]] std::optional<Scalar> condition_number_from_eigenvalues(std::vector<Scalar> eigenvalues)
{
  if (eigenvalues.empty())
  {
    return std::nullopt;
  }

  Scalar min_abs = std::numeric_limits<Scalar>::infinity();
  Scalar max_abs = Scalar{};
  for (Scalar value : eigenvalues)
  {
    using std::abs;
    Scalar const abs_value = abs(value);
    if (abs_value == Scalar{})
    {
      return std::nullopt;
    }
    min_abs = std::min(min_abs, abs_value);
    max_abs = std::max(max_abs, abs_value);
  }
  return max_abs / min_abs;
}

template <uni20::Real Scalar>
[[nodiscard]] KnownSymmetricMatrix<Scalar> prescribed_diagonal_spectrum(std::vector<Scalar> eigenvalues_ascending)
{
  if (eigenvalues_ascending.empty())
  {
    throw std::invalid_argument("prescribed diagonal spectrum must be nonempty");
  }
  if (!std::ranges::is_sorted(eigenvalues_ascending))
  {
    throw std::invalid_argument("prescribed diagonal spectrum must be sorted ascending");
  }

  std::size_t const n = eigenvalues_ascending.size();
  std::vector<Scalar> matrix(n * n, Scalar{});
  for (std::size_t i = 0; i < n; ++i)
  {
    matrix[i * n + i] = eigenvalues_ascending[i];
  }
  auto condition_number = condition_number_from_eigenvalues(eigenvalues_ascending);

  return KnownSymmetricMatrix<Scalar>{
      .metadata =
          {
              .id = "prescribed_diagonal_spectrum",
              .source = "Uni20 generated test matrix",
              .license = "Uni20 project license",
              .reference = "Direct diagonal construction",
              .purpose = "Exact-spectrum baseline with controllable gaps and clusters.",
              .oracle = MatrixOracleKind::ExactEigenvaluesAndCondition,
              .symmetric = true,
              .positive_definite = eigenvalues_ascending.front() > Scalar{},
              .sparse = true,
          },
      .dimension = n,
      .row_major_values = std::move(matrix),
      .eigenvalues_ascending = std::move(eigenvalues_ascending),
      .two_norm_condition_number = condition_number,
  };
}

template <uni20::Real Scalar>
[[nodiscard]] KnownSymmetricMatrix<Scalar> symmetric_tridiagonal_toeplitz(std::size_t n, Scalar diagonal,
                                                                          Scalar offdiagonal)
{
  if (n == 0)
  {
    throw std::invalid_argument("tridiagonal Toeplitz dimension must be nonzero");
  }

  std::vector<Scalar> matrix(n * n, Scalar{});
  for (std::size_t i = 0; i < n; ++i)
  {
    matrix[i * n + i] = diagonal;
    if (i + 1 < n)
    {
      matrix[i * n + i + 1] = offdiagonal;
      matrix[(i + 1) * n + i] = offdiagonal;
    }
  }

  std::vector<Scalar> eigenvalues;
  eigenvalues.reserve(n);
  Scalar const pi = matrix_test_pi<Scalar>();
  using std::cos;
  for (std::size_t j = 1; j <= n; ++j)
  {
    Scalar const angle = static_cast<Scalar>(j) * pi / static_cast<Scalar>(n + 1);
    eigenvalues.push_back(diagonal + Scalar{2} * offdiagonal * cos(angle));
  }
  std::ranges::sort(eigenvalues);
  auto condition_number = condition_number_from_eigenvalues(eigenvalues);

  bool const positive_definite = !eigenvalues.empty() && eigenvalues.front() > Scalar{};
  return KnownSymmetricMatrix<Scalar>{
      .metadata =
          {
              .id = "symmetric_tridiagonal_toeplitz",
              .source = "Classical tridiagonal Toeplitz test matrix",
              .license = "Formula in public mathematical literature",
              .reference = "Eigenvalues a + 2 b cos(j pi/(n+1)), j=1..n",
              .purpose = "Sparse exact-spectrum symmetric baseline.",
              .oracle = MatrixOracleKind::ExactEigenvaluesAndCondition,
              .symmetric = true,
              .positive_definite = positive_definite,
              .sparse = true,
          },
      .dimension = n,
      .row_major_values = std::move(matrix),
      .eigenvalues_ascending = std::move(eigenvalues),
      .two_norm_condition_number = condition_number,
  };
}

template <uni20::Real Scalar> [[nodiscard]] KnownSymmetricMatrix<Scalar> path_laplacian(std::size_t n)
{
  if (n == 0)
  {
    throw std::invalid_argument("path Laplacian dimension must be nonzero");
  }

  std::vector<Scalar> matrix(n * n, Scalar{});
  for (std::size_t i = 0; i < n; ++i)
  {
    matrix[i * n + i] = i == 0 || i + 1 == n ? Scalar{1} : Scalar{2};
    if (i + 1 < n)
    {
      matrix[i * n + i + 1] = Scalar{-1};
      matrix[(i + 1) * n + i] = Scalar{-1};
    }
  }

  std::vector<Scalar> eigenvalues;
  eigenvalues.reserve(n);
  Scalar const pi = matrix_test_pi<Scalar>();
  using std::cos;
  for (std::size_t j = 0; j < n; ++j)
  {
    Scalar const angle = static_cast<Scalar>(j) * pi / static_cast<Scalar>(n);
    eigenvalues.push_back(Scalar{2} - Scalar{2} * cos(angle));
  }
  std::ranges::sort(eigenvalues);

  return KnownSymmetricMatrix<Scalar>{
      .metadata =
          {
              .id = "path_laplacian",
              .source = "Uni20 generated graph Laplacian",
              .license = "Formula in public mathematical literature",
              .reference = "Path graph Laplacian spectrum 2 - 2 cos(j pi/n), j=0..n-1",
              .purpose = "Sparse positive-semidefinite exact-spectrum graph test.",
              .oracle = MatrixOracleKind::ExactEigenvalues,
              .symmetric = true,
              .positive_definite = false,
              .sparse = true,
          },
      .dimension = n,
      .row_major_values = std::move(matrix),
      .eigenvalues_ascending = std::move(eigenvalues),
      .two_norm_condition_number = std::nullopt,
  };
}

template <uni20::Real Scalar>
[[nodiscard]] KnownSymmetricMatrix<Scalar> shifted_path_laplacian(std::size_t n, Scalar shift)
{
  auto matrix = path_laplacian<Scalar>(n);
  for (std::size_t i = 0; i < n; ++i)
  {
    matrix.row_major_values[i * n + i] += shift;
  }
  for (Scalar& value : matrix.eigenvalues_ascending)
  {
    value += shift;
  }
  matrix.metadata = KnownMatrixMetadata{
      .id = "shifted_path_laplacian",
      .source = "Uni20 generated graph Laplacian",
      .license = "Formula in public mathematical literature",
      .reference = "Path graph Laplacian spectrum shifted by c",
      .purpose = "Large identity-shift stress test for residual scaling and breakdown thresholds.",
      .oracle = MatrixOracleKind::ExactEigenvaluesAndCondition,
      .symmetric = true,
      .positive_definite = shift > Scalar{},
      .sparse = true,
  };
  matrix.two_norm_condition_number = condition_number_from_eigenvalues(matrix.eigenvalues_ascending);
  return matrix;
}

template <uni20::Real Scalar> [[nodiscard]] KnownSymmetricMatrix<Scalar> diagonal_clustered_extremes(std::size_t n)
{
  if (n < 12)
  {
    throw std::invalid_argument("clustered-extremes stress matrix needs dimension at least 12");
  }

  std::vector<Scalar> eigenvalues;
  eigenvalues.reserve(n);
  eigenvalues.push_back(Scalar{-1000});
  eigenvalues.push_back(Scalar{-999});
  eigenvalues.push_back(Scalar{-998});

  std::size_t const interior_count = n - 6;
  for (std::size_t i = 0; i < interior_count; ++i)
  {
    Scalar const fraction =
        interior_count == 1 ? Scalar{} : static_cast<Scalar>(i) / static_cast<Scalar>(interior_count - 1);
    eigenvalues.push_back(Scalar{1} + fraction);
  }

  eigenvalues.push_back(Scalar{998});
  eigenvalues.push_back(Scalar{999});
  eigenvalues.push_back(Scalar{1000});
  std::ranges::sort(eigenvalues);

  auto result = prescribed_diagonal_spectrum(std::move(eigenvalues));
  result.metadata = KnownMatrixMetadata{
      .id = "diagonal_clustered_extremes",
      .source = "Uni20 generated stress matrix",
      .license = "Uni20 project license",
      .reference = "Diagonal matrix with clustered extremal eigenvalues and dense interior spectrum",
      .purpose = "Lanczos ghost/duplicate-Ritz stress test for extremal convergence.",
      .oracle = MatrixOracleKind::ExactEigenvaluesAndCondition,
      .symmetric = true,
      .positive_definite = false,
      .sparse = true,
  };
  return result;
}

template <uni20::Real Scalar>
[[nodiscard]] KnownSymmetricMatrix<Scalar> symmetric_interior_gap(std::size_t n, Scalar gap)
{
  if (n < 12 || gap <= Scalar{})
  {
    throw std::invalid_argument("interior-gap stress matrix needs dimension at least 12 and positive gap");
  }

  std::vector<Scalar> eigenvalues;
  eigenvalues.reserve(n);
  std::size_t const half = (n - 2) / 2;
  for (std::size_t i = 0; i < half; ++i)
  {
    eigenvalues.push_back(-static_cast<Scalar>(half - i));
  }
  eigenvalues.push_back(-gap);
  eigenvalues.push_back(gap);
  while (eigenvalues.size() < n)
  {
    eigenvalues.push_back(static_cast<Scalar>(eigenvalues.size() - half - 1));
  }
  std::ranges::sort(eigenvalues);

  auto result = prescribed_diagonal_spectrum(std::move(eigenvalues));
  result.metadata = KnownMatrixMetadata{
      .id = "symmetric_interior_gap",
      .source = "Uni20 generated stress matrix",
      .license = "Uni20 project license",
      .reference = "Diagonal spectrum with wanted smallest-magnitude eigenvalues buried in the interior",
      .purpose = "Ordinary Ritz interior-target trap; should require shift-invert or fail cleanly.",
      .oracle = MatrixOracleKind::ExactEigenvaluesAndCondition,
      .symmetric = true,
      .positive_definite = false,
      .sparse = true,
  };
  return result;
}

template <uni20::Real Scalar>
[[nodiscard]] KnownSymmetricMatrix<Scalar> diagonal_clustered_wanted_end(std::size_t n, std::size_t cluster_size,
                                                                         Scalar gap)
{
  if (n <= cluster_size + 4 || cluster_size == 0 || gap <= Scalar{})
  {
    throw std::invalid_argument("clustered-wanted-end stress matrix has invalid dimensions or gap");
  }

  std::vector<Scalar> eigenvalues;
  eigenvalues.reserve(n);
  std::size_t const bulk_count = n - cluster_size;
  for (std::size_t i = 0; i < bulk_count; ++i)
  {
    Scalar const fraction = bulk_count == 1 ? Scalar{} : static_cast<Scalar>(i) / static_cast<Scalar>(bulk_count - 1);
    eigenvalues.push_back(Scalar{-1} + Scalar{1.5} * fraction);
  }
  for (std::size_t i = 0; i < cluster_size; ++i)
  {
    eigenvalues.push_back(Scalar{1} - static_cast<Scalar>(i) * gap);
  }
  std::ranges::sort(eigenvalues);

  auto result = prescribed_diagonal_spectrum(std::move(eigenvalues));
  result.metadata = KnownMatrixMetadata{
      .id = "diagonal_clustered_wanted_end",
      .source = "Uni20 generated stress matrix",
      .license = "Uni20 project license",
      .reference = "Diagonal spectrum with a tightly clustered wanted end",
      .purpose = "Restart torture test for small search spaces and clustered wanted Ritz values.",
      .oracle = MatrixOracleKind::ExactEigenvaluesAndCondition,
      .symmetric = true,
      .positive_definite = false,
      .sparse = true,
  };
  return result;
}

template <uni20::Real Scalar> [[nodiscard]] std::vector<Scalar> soules_matrix(std::size_t n)
{
  if (n == 0)
  {
    throw std::invalid_argument("Soules matrix dimension must be nonzero");
  }

  using std::sqrt;
  std::vector<Scalar> x(n, Scalar{1} / sqrt(static_cast<Scalar>(n)));
  std::vector<Scalar> q(n * n, Scalar{});
  for (std::size_t i = 0; i < n; ++i)
  {
    q[i * n + i] = Scalar{1};
  }

  for (std::size_t i = n - 1; i > 0; --i)
  {
    if (x[i] != Scalar{})
    {
      using std::hypot;
      Scalar const norm = hypot(x[i - 1], x[i]);
      Scalar const c = x[i - 1] / norm;
      Scalar const s = x[i] / norm;
      x[i - 1] = norm;
      x[i] = Scalar{};

      for (std::size_t row = 0; row < n; ++row)
      {
        Scalar const left = q[row * n + i - 1];
        Scalar const right = q[row * n + i];
        q[row * n + i - 1] = c * left - s * right;
        q[row * n + i] = s * left + c * right;
      }
    }
  }

  std::vector<Scalar> reversed(n * n, Scalar{});
  for (std::size_t row = 0; row < n; ++row)
  {
    for (std::size_t col = 0; col < n; ++col)
    {
      reversed[row * n + col] = q[(n - 1 - row) * n + col];
    }
  }
  return reversed;
}

template <uni20::Real Scalar>
[[nodiscard]] KnownSymmetricMatrix<Scalar> symmstoch_with_spectrum(std::vector<Scalar> spectrum_descending)
{
  if (spectrum_descending.empty())
  {
    throw std::invalid_argument("symmstoch spectrum must be nonempty");
  }
  if (!std::ranges::is_sorted(spectrum_descending, std::ranges::greater{}))
  {
    throw std::invalid_argument("symmstoch spectrum must be sorted descending");
  }
  if (spectrum_descending.front() <= Scalar{})
  {
    throw std::invalid_argument("symmstoch leading spectrum value must be positive");
  }

  std::size_t const n = spectrum_descending.size();
  Scalar nonnegativity_check = spectrum_descending[0] / static_cast<Scalar>(n);
  for (std::size_t j = 1; j < n; ++j)
  {
    std::size_t const denominator = (n - j + 1) * (n - j);
    nonnegativity_check += spectrum_descending[j] / static_cast<Scalar>(denominator);
  }
  if (nonnegativity_check < Scalar{})
  {
    throw std::invalid_argument("symmstoch spectrum does not satisfy the Soules nonnegativity condition");
  }

  std::vector<Scalar> q = soules_matrix<Scalar>(n);
  std::vector<Scalar> matrix(n * n, Scalar{});
  for (std::size_t row = 0; row < n; ++row)
  {
    for (std::size_t col = 0; col < n; ++col)
    {
      Scalar value{};
      for (std::size_t k = 0; k < n; ++k)
      {
        value += q[row * n + k] * spectrum_descending[k] * q[col * n + k];
      }
      matrix[row * n + col] = value / spectrum_descending.front();
    }
  }

  for (std::size_t row = 0; row < n; ++row)
  {
    for (std::size_t col = row + 1; col < n; ++col)
    {
      Scalar const symmetric_value = (matrix[row * n + col] + matrix[col * n + row]) / Scalar{2};
      matrix[row * n + col] = symmetric_value;
      matrix[col * n + row] = symmetric_value;
    }
  }

  std::vector<Scalar> eigenvalues;
  eigenvalues.reserve(n);
  for (Scalar value : spectrum_descending)
  {
    eigenvalues.push_back(value / spectrum_descending.front());
  }
  std::ranges::sort(eigenvalues);
  auto condition_number = condition_number_from_eigenvalues(eigenvalues);

  return KnownSymmetricMatrix<Scalar>{
      .metadata =
          {
              .id = "anymatrix_core_symmstoch",
              .source = "Anymatrix core/symmstoch and core/soules",
              .license = "BSD-2-Clause",
              .reference = "Higham and Mikaitis, Anymatrix; Elsner, Nabben and Neumann; Soules; Stuart",
              .purpose = "Dense symmetric stochastic matrix with prescribed exact spectrum.",
              .oracle = MatrixOracleKind::ExactEigenvaluesAndCondition,
              .symmetric = true,
              .positive_definite = !eigenvalues.empty() && eigenvalues.front() > Scalar{},
              .sparse = false,
          },
      .dimension = n,
      .row_major_values = std::move(matrix),
      .eigenvalues_ascending = std::move(eigenvalues),
      .two_norm_condition_number = condition_number,
  };
}

template <uni20::Real Scalar>
[[nodiscard]] KnownSymmetricMatrix<Scalar> blockhouse_coordinate_reflector(std::size_t n, std::size_t rank)
{
  if (n == 0 || rank > n)
  {
    throw std::invalid_argument("blockhouse coordinate reflector needs 0 <= rank <= dimension and nonzero dimension");
  }

  std::vector<Scalar> matrix(n * n, Scalar{});
  std::vector<Scalar> eigenvalues;
  eigenvalues.reserve(n);
  for (std::size_t i = 0; i < n; ++i)
  {
    Scalar const value = i < rank ? Scalar{-1} : Scalar{1};
    matrix[i * n + i] = value;
    eigenvalues.push_back(value);
  }
  std::ranges::sort(eigenvalues);

  return KnownSymmetricMatrix<Scalar>{
      .metadata =
          {
              .id = "anymatrix_core_blockhouse_coordinate_reflector",
              .source = "Anymatrix core/blockhouse special case",
              .license = "BSD-2-Clause",
              .reference = "Schreiber and Parlett, Block Reflectors, SIAM J. Numer. Anal. 25(1), 1988",
              .purpose = "Symmetric orthogonal/involutory matrix with exact repeated eigenvalues.",
              .oracle = MatrixOracleKind::ExactEigenvaluesAndCondition,
              .symmetric = true,
              .positive_definite = false,
              .sparse = true,
          },
      .dimension = n,
      .row_major_values = std::move(matrix),
      .eigenvalues_ascending = std::move(eigenvalues),
      .two_norm_condition_number = Scalar{1},
  };
}

} // namespace uni20::krylov::test_matrices
