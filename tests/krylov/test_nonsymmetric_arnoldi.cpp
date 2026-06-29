#include <uni20/krylov/dense_host_vector.hpp>
#include <uni20/krylov/nonsymmetric_arnoldi.hpp>

#include "krylov_test_types.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{

#ifndef UNI20_KRYLOV_MATRIX_MARKET_DIR
#define UNI20_KRYLOV_MATRIX_MARKET_DIR "."
#endif

using uni20::krylov::DenseHostVector;
using uni20::krylov::DenseHostVectorOps;
using uni20::krylov::KrylovDiagnosticsLevel;
using uni20::krylov::NonsymmetricEigenParams;
using uni20::krylov::NonsymmetricStatus;
using uni20::krylov::RitzReality;
using uni20::krylov::SpectrumPart;

template <typename Scalar> class KrylovNonsymmetricArnoldiRealTypedTest : public ::testing::Test {};
template <typename Scalar> class KrylovNonsymmetricArnoldiComplexTypedTest : public ::testing::Test {};
using RealTypes = uni20::krylov::test::KrylovRealTestTypes;
using ComplexTypes = uni20::krylov::test::KrylovComplexTestTypes;
TYPED_TEST_SUITE(KrylovNonsymmetricArnoldiRealTypedTest, RealTypes);
TYPED_TEST_SUITE(KrylovNonsymmetricArnoldiComplexTypedTest, ComplexTypes);

template <typename Scalar> double arnoldi_tolerance()
{
  if constexpr (std::is_same_v<Scalar, float>)
  {
    return 1.0e-4;
  }
  else
  {
    return 1.0e-12;
  }
}

template <typename Scalar> double abs_as_double(Scalar const& value) { return static_cast<double>(std::abs(value)); }

template <typename Complex> std::vector<Complex> transfer_like_clustered_spectrum(std::size_t dimension)
{
  using Real = typename Complex::value_type;
  if (dimension < 8)
  {
    throw std::invalid_argument("transfer-like stress matrix requires dimension at least 8");
  }

  std::vector<Complex> spectrum;
  spectrum.reserve(dimension);
  spectrum.push_back(Complex{Real{1}, Real{}});
  spectrum.push_back(std::polar(Real{0.996}, Real{0.19}));
  spectrum.push_back(std::polar(Real{0.993}, Real{-0.23}));
  spectrum.push_back(std::polar(Real{0.985}, Real{0.41}));
  spectrum.push_back(std::polar(Real{0.972}, Real{-0.37}));
  for (std::size_t i = spectrum.size(); i < dimension; ++i)
  {
    Real const radius = Real{0.90} - Real{0.004} * static_cast<Real>(i - 5);
    Real const angle = Real{0.11} * static_cast<Real>(i + 1);
    spectrum.push_back(std::polar(std::max(radius, Real{0.45}), angle));
  }
  return spectrum;
}

template <typename Complex> std::vector<Complex> transfer_like_nonnormal_matrix(std::size_t dimension)
{
  using Real = typename Complex::value_type;

  std::vector<Complex> const spectrum = transfer_like_clustered_spectrum<Complex>(dimension);
  std::vector<Complex> matrix(dimension * dimension, Complex{});
  for (std::size_t row = 0; row < dimension; ++row)
  {
    matrix[row * dimension + row] = spectrum[row];
    for (std::size_t distance = 1; distance <= 5 && distance <= row; ++distance)
    {
      std::size_t const col = row - distance;
      Real const phase = Real{0.17} * static_cast<Real>((row + 1) * (distance + 2));
      Real const scale = Real{0.018} / static_cast<Real>(distance);
      matrix[row * dimension + col] = std::polar(scale, phase);
    }
  }
  return matrix;
}

template <typename Complex> double typed_vector_norm(DenseHostVector<Complex> const& vector)
{
  double norm_squared = 0.0;
  for (auto const& value : vector.values)
  {
    norm_squared += static_cast<double>(std::norm(value));
  }
  return std::sqrt(norm_squared);
}

template <typename Complex>
double typed_relative_eigen_residual(DenseHostVectorOps<Complex>& ops, DenseHostVector<Complex> const& vector,
                                     Complex eigenvalue)
{
  auto applied = ops.allocate_like(vector);
  ops.matvec(applied, vector);
  for (std::size_t i = 0; i < applied.values.size(); ++i)
  {
    applied.values[i] -= eigenvalue * vector.values[i];
  }
  return typed_vector_norm(applied) /
         std::max(1.0, static_cast<double>(std::abs(eigenvalue)) * typed_vector_norm(vector));
}

struct SparseEntry
{
    std::size_t row = 0;
    std::size_t col = 0;
    double value = 0.0;
};

struct SparseMatrixMarketMatrix
{
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<SparseEntry> entries;
};

struct ComplexSparseEntry
{
    std::size_t row = 0;
    std::size_t col = 0;
    uni20::complex<double> value{};
};

struct ComplexSparseMatrixMarketMatrix
{
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<ComplexSparseEntry> entries;
};

class SparseHostVectorOps {
  public:
    explicit SparseHostVectorOps(SparseMatrixMarketMatrix matrix) : matrix_(std::move(matrix))
    {
      if (matrix_.rows != matrix_.cols)
      {
        throw std::invalid_argument("nonsymmetric Matrix Market test matrix must be square");
      }
    }

    [[nodiscard]] std::size_t problem_dimension() const noexcept { return matrix_.rows; }

    [[nodiscard]] std::size_t vector_dimension(DenseHostVector<double> const& x) const noexcept
    {
      return x.values.size();
    }

    [[nodiscard]] int matvec_count() const noexcept { return matvec_count_; }

    [[nodiscard]] DenseHostVector<double> allocate_like(DenseHostVector<double> const& x)
    {
      return DenseHostVector<double>{std::vector<double>(x.values.size())};
    }

    void copy(DenseHostVector<double>& dst, DenseHostVector<double> const& src)
    {
      require_same_size(dst, src);
      dst.values = src.values;
    }

    void axpy(DenseHostVector<double>& y, double alpha, DenseHostVector<double> const& x)
    {
      require_same_size(y, x);
      for (std::size_t i = 0; i < y.values.size(); ++i)
      {
        y.values[i] += alpha * x.values[i];
      }
    }

    void scal(DenseHostVector<double>& x, double alpha)
    {
      for (double& value : x.values)
      {
        value *= alpha;
      }
    }

    void set_zero(DenseHostVector<double>& x)
    {
      for (double& value : x.values)
      {
        value = 0.0;
      }
    }

    [[nodiscard]] double inner_product(DenseHostVector<double> const& x, DenseHostVector<double> const& y)
    {
      require_same_size(x, y);
      double result = 0.0;
      for (std::size_t i = 0; i < x.values.size(); ++i)
      {
        result += x.values[i] * y.values[i];
      }
      return result;
    }

    void matvec(DenseHostVector<double>& y, DenseHostVector<double> const& x)
    {
      if (x.values.size() != matrix_.cols || y.values.size() != matrix_.rows)
      {
        throw std::invalid_argument("nonsymmetric Matrix Market test vector has the wrong size");
      }
      ++matvec_count_;
      this->set_zero(y);
      for (auto const& entry : matrix_.entries)
      {
        y.values[entry.row] += entry.value * x.values[entry.col];
      }
    }

  private:
    static void require_same_size(DenseHostVector<double> const& lhs, DenseHostVector<double> const& rhs)
    {
      if (lhs.values.size() != rhs.values.size())
      {
        throw std::invalid_argument("nonsymmetric Matrix Market test vectors have different sizes");
      }
    }

    SparseMatrixMarketMatrix matrix_;
    int matvec_count_ = 0;
};

class ComplexSparseHostVectorOps {
  public:
    explicit ComplexSparseHostVectorOps(ComplexSparseMatrixMarketMatrix matrix) : matrix_(std::move(matrix))
    {
      if (matrix_.rows != matrix_.cols)
      {
        throw std::invalid_argument("complex nonsymmetric Matrix Market test matrix must be square");
      }
    }

    [[nodiscard]] std::size_t problem_dimension() const noexcept { return matrix_.rows; }

    [[nodiscard]] std::size_t vector_dimension(DenseHostVector<uni20::complex<double>> const& x) const noexcept
    {
      return x.values.size();
    }

    [[nodiscard]] int matvec_count() const noexcept { return matvec_count_; }

    [[nodiscard]] DenseHostVector<uni20::complex<double>>
    allocate_like(DenseHostVector<uni20::complex<double>> const& x)
    {
      return DenseHostVector<uni20::complex<double>>{std::vector<uni20::complex<double>>(x.values.size())};
    }

    void copy(DenseHostVector<uni20::complex<double>>& dst, DenseHostVector<uni20::complex<double>> const& src)
    {
      require_same_size(dst, src);
      dst.values = src.values;
    }

    void axpy(DenseHostVector<uni20::complex<double>>& y, uni20::complex<double> alpha,
              DenseHostVector<uni20::complex<double>> const& x)
    {
      require_same_size(y, x);
      for (std::size_t i = 0; i < y.values.size(); ++i)
      {
        y.values[i] += alpha * x.values[i];
      }
    }

    void scal(DenseHostVector<uni20::complex<double>>& x, uni20::complex<double> alpha)
    {
      for (auto& value : x.values)
      {
        value *= alpha;
      }
    }

    void set_zero(DenseHostVector<uni20::complex<double>>& x)
    {
      for (auto& value : x.values)
      {
        value = {};
      }
    }

    [[nodiscard]] uni20::complex<double> inner_product(DenseHostVector<uni20::complex<double>> const& x,
                                                       DenseHostVector<uni20::complex<double>> const& y)
    {
      require_same_size(x, y);
      uni20::complex<double> result{};
      for (std::size_t i = 0; i < x.values.size(); ++i)
      {
        result += std::conj(x.values[i]) * y.values[i];
      }
      return result;
    }

    [[nodiscard]] double norm(DenseHostVector<uni20::complex<double>> const& x)
    {
      double norm_squared = 0.0;
      for (auto const& value : x.values)
      {
        norm_squared += std::norm(value);
      }
      return norm_squared > 0.0 ? std::sqrt(norm_squared) : 0.0;
    }

    void matvec(DenseHostVector<uni20::complex<double>>& y, DenseHostVector<uni20::complex<double>> const& x)
    {
      if (x.values.size() != matrix_.cols || y.values.size() != matrix_.rows)
      {
        throw std::invalid_argument("complex nonsymmetric Matrix Market test vector has the wrong size");
      }
      ++matvec_count_;
      this->set_zero(y);
      for (auto const& entry : matrix_.entries)
      {
        y.values[entry.row] += entry.value * x.values[entry.col];
      }
    }

  private:
    static void require_same_size(DenseHostVector<uni20::complex<double>> const& lhs,
                                  DenseHostVector<uni20::complex<double>> const& rhs)
    {
      if (lhs.values.size() != rhs.values.size())
      {
        throw std::invalid_argument("complex nonsymmetric Matrix Market test vectors have different sizes");
      }
    }

    ComplexSparseMatrixMarketMatrix matrix_;
    int matvec_count_ = 0;
};

SparseMatrixMarketMatrix read_real_general_matrix_market(std::string const& path)
{
  std::ifstream input(path);
  if (!input)
  {
    throw std::invalid_argument("could not open Matrix Market fixture: " + path);
  }

  std::string header;
  std::getline(input, header);
  std::istringstream header_stream(header);
  std::string banner;
  std::string object;
  std::string format;
  std::string field;
  std::string symmetry;
  header_stream >> banner >> object >> format >> field >> symmetry;
  if (banner != "%%MatrixMarket" || object != "matrix" || format != "coordinate" || field != "real" ||
      symmetry != "general")
  {
    throw std::invalid_argument("expected a real general coordinate Matrix Market fixture");
  }

  std::string line;
  do
  {
    if (!std::getline(input, line))
    {
      throw std::invalid_argument("Matrix Market fixture is missing its size line");
    }
  }
  while (line.empty() || line[0] == '%');

  std::istringstream size_stream(line);
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::size_t stored_entries = 0;
  size_stream >> rows >> cols >> stored_entries;

  SparseMatrixMarketMatrix matrix{.rows = rows, .cols = cols, .entries = {}};
  matrix.entries.reserve(stored_entries);
  for (std::size_t entry_index = 0; entry_index < stored_entries; ++entry_index)
  {
    std::size_t row = 0;
    std::size_t col = 0;
    double value = 0.0;
    input >> row >> col >> value;
    if (row == 0 || row > rows || col == 0 || col > cols)
    {
      throw std::invalid_argument("Matrix Market fixture contains an out-of-range entry");
    }
    matrix.entries.push_back(SparseEntry{.row = row - 1, .col = col - 1, .value = value});
  }

  return matrix;
}

ComplexSparseMatrixMarketMatrix read_complex_general_matrix_market(std::string const& path)
{
  std::ifstream input(path);
  if (!input)
  {
    throw std::invalid_argument("could not open Matrix Market fixture: " + path);
  }

  std::string header;
  std::getline(input, header);
  std::istringstream header_stream(header);
  std::string banner;
  std::string object;
  std::string format;
  std::string field;
  std::string symmetry;
  header_stream >> banner >> object >> format >> field >> symmetry;
  if (banner != "%%MatrixMarket" || object != "matrix" || format != "coordinate" || field != "complex" ||
      symmetry != "general")
  {
    throw std::invalid_argument("expected a complex general coordinate Matrix Market fixture");
  }

  std::string line;
  do
  {
    if (!std::getline(input, line))
    {
      throw std::invalid_argument("complex Matrix Market fixture is missing its size line");
    }
  }
  while (line.empty() || line[0] == '%');

  std::istringstream size_stream(line);
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::size_t stored_entries = 0;
  size_stream >> rows >> cols >> stored_entries;

  ComplexSparseMatrixMarketMatrix matrix{.rows = rows, .cols = cols, .entries = {}};
  matrix.entries.reserve(stored_entries);
  for (std::size_t entry_index = 0; entry_index < stored_entries; ++entry_index)
  {
    std::size_t row = 0;
    std::size_t col = 0;
    double real = 0.0;
    double imaginary = 0.0;
    input >> row >> col >> real >> imaginary;
    if (row == 0 || row > rows || col == 0 || col > cols)
    {
      throw std::invalid_argument("complex Matrix Market fixture contains an out-of-range entry");
    }
    matrix.entries.push_back(ComplexSparseEntry{.row = row - 1, .col = col - 1, .value = {real, imaginary}});
  }

  return matrix;
}

double vector_norm(DenseHostVector<double> const& vector)
{
  double norm_squared = 0.0;
  for (double const value : vector.values)
  {
    norm_squared += value * value;
  }
  return std::sqrt(norm_squared);
}

DenseHostVector<double> projected_column(std::vector<DenseHostVector<double>> const& basis,
                                         uni20::krylov::Matrix<double> const& hessenberg, std::size_t column)
{
  DenseHostVector<double> result{std::vector<double>(basis.front().values.size(), 0.0)};
  for (std::size_t basis_index = 0; basis_index < basis.size(); ++basis_index)
  {
    double const coefficient = hessenberg[basis_index, column];
    for (std::size_t row = 0; row < result.values.size(); ++row)
    {
      result.values[row] += coefficient * basis[basis_index].values[row];
    }
  }
  return result;
}

uni20::complex<double> nearest_eigenvalue(std::vector<uni20::complex<double>> const& values,
                                          uni20::complex<double> target)
{
  if (values.empty())
  {
    throw std::invalid_argument("nearest_eigenvalue requires at least one value");
  }

  uni20::complex<double> nearest = values.front();
  double best_distance = std::abs(nearest - target);
  for (auto const& value : values)
  {
    double const distance = std::abs(value - target);
    if (distance < best_distance)
    {
      nearest = value;
      best_distance = distance;
    }
  }
  return nearest;
}

double relative_eigen_residual(DenseHostVectorOps<double>& ops, DenseHostVector<double> const& vector,
                               double eigenvalue)
{
  auto applied = ops.allocate_like(vector);
  ops.matvec(applied, vector);
  for (std::size_t i = 0; i < applied.values.size(); ++i)
  {
    applied.values[i] -= eigenvalue * vector.values[i];
  }
  return vector_norm(applied) / std::max(1.0, std::abs(eigenvalue) * vector_norm(vector));
}

double vector_norm(DenseHostVector<uni20::complex<double>> const& vector)
{
  double norm_squared = 0.0;
  for (auto const& value : vector.values)
  {
    norm_squared += std::norm(value);
  }
  return std::sqrt(norm_squared);
}

template <typename Ops>
double relative_eigen_residual(Ops& ops, DenseHostVector<uni20::complex<double>> const& vector,
                               uni20::complex<double> eigenvalue)
{
  auto applied = ops.allocate_like(vector);
  ops.matvec(applied, vector);
  for (std::size_t i = 0; i < applied.values.size(); ++i)
  {
    applied.values[i] -= eigenvalue * vector.values[i];
  }
  return vector_norm(applied) / std::max(1.0, std::abs(eigenvalue) * vector_norm(vector));
}

} // namespace

TEST(KrylovNonsymmetricArnoldi, DenseHostVectorOpsUsesConjugatedComplexInnerProduct)
{
  using Complex = uni20::complex<double>;

  DenseHostVectorOps<Complex> ops(2, {Complex{1.0, 0.0}, Complex{}, Complex{}, Complex{1.0, 0.0}});
  DenseHostVector<Complex> x{{Complex{1.0, 2.0}, Complex{-3.0, 4.0}}};
  DenseHostVector<Complex> y{{Complex{5.0, -7.0}, Complex{2.0, 11.0}}};

  Complex const expected = std::conj(x.values[0]) * y.values[0] + std::conj(x.values[1]) * y.values[1];
  Complex const actual = ops.inner_product(x, y);

  EXPECT_NEAR(actual.real(), expected.real(), 1.0e-14);
  EXPECT_NEAR(actual.imag(), expected.imag(), 1.0e-14);
}

TEST(KrylovNonsymmetricArnoldi, SolvesDenseComplexNonsymmetricProjectedProblem)
{
  using Complex = uni20::complex<double>;

  uni20::krylov::Matrix<Complex> matrix(2, 2);
  matrix[0, 0] = Complex{1.0, 2.0};
  matrix[1, 0] = Complex{};
  matrix[0, 1] = Complex{3.0, -1.0};
  matrix[1, 1] = Complex{-2.0, 0.5};

  auto eigensystem = uni20::krylov::complex_nonsymmetric_eigensystem<double>(std::move(matrix), true);

  ASSERT_EQ(eigensystem.eigenvalues.size(), 2);
  auto const first = nearest_eigenvalue(eigensystem.eigenvalues, Complex{1.0, 2.0});
  auto const second = nearest_eigenvalue(eigensystem.eigenvalues, Complex{-2.0, 0.5});
  EXPECT_NEAR(std::abs(first - Complex{1.0, 2.0}), 0.0, 1.0e-12);
  EXPECT_NEAR(std::abs(second - Complex{-2.0, 0.5}), 0.0, 1.0e-12);
  ASSERT_EQ(eigensystem.right_eigenvectors.rows(), 2);
  ASSERT_EQ(eigensystem.right_eigenvectors.cols(), 2);
}

TEST(KrylovNonsymmetricArnoldi, BuildsOrthonormalBasisAndArnoldiRelation)
{
  std::vector<double> const matrix{
      3.0, 2.0, 0.0, 0.0, //
      0.0, 2.0, 1.0, 0.0, //
      0.0, 0.0, 1.0, 4.0, //
      0.5, 0.0, 0.0, 0.25,
  };
  DenseHostVectorOps<double> ops(4, matrix);
  DenseHostVector<double> initial{{1.0, 2.0, 3.0, 5.0}};

  auto factorization = uni20::krylov::arnoldi_factorize<double>(ops, initial, 3);

  ASSERT_EQ(factorization.step_count, 3);
  ASSERT_FALSE(factorization.happy_breakdown);
  ASSERT_EQ(factorization.basis.size(), 4);
  ASSERT_EQ(factorization.hessenberg.rows(), 4);
  ASSERT_EQ(factorization.hessenberg.cols(), 3);
  EXPECT_EQ(factorization.op_count, 3);
  EXPECT_EQ(ops.matvec_count(), 3);

  for (std::size_t i = 0; i < factorization.basis.size(); ++i)
  {
    for (std::size_t j = 0; j < factorization.basis.size(); ++j)
    {
      double const expected = i == j ? 1.0 : 0.0;
      EXPECT_NEAR(ops.inner_product(factorization.basis[i], factorization.basis[j]), expected, 1.0e-12);
    }
  }

  for (std::size_t column = 0; column < static_cast<std::size_t>(factorization.step_count); ++column)
  {
    auto applied = ops.allocate_like(initial);
    ops.matvec(applied, factorization.basis[column]);
    auto projected = projected_column(factorization.basis, factorization.hessenberg, column);
    for (std::size_t row = 0; row < applied.values.size(); ++row)
    {
      applied.values[row] -= projected.values[row];
    }
    EXPECT_LT(vector_norm(applied), 1.0e-11);
  }
}

TEST(KrylovNonsymmetricArnoldi, CompressesArnoldiFactorizationThroughRealSchurRestart)
{
  std::vector<double> const matrix{
      3.0, 2.0, 0.0, 0.0, //
      0.0, 2.0, 1.0, 0.0, //
      0.0, 0.0, 1.0, 4.0, //
      0.5, 0.0, 0.0, 0.25,
  };
  DenseHostVectorOps<double> ops(4, matrix);
  DenseHostVector<double> initial{{1.0, 2.0, 3.0, 5.0}};

  auto factorization = uni20::krylov::arnoldi_factorize<double>(ops, initial, 3);
  NonsymmetricEigenParams<double> params;
  params.eigenvalue_count = 1;
  params.retained_ritz_count = 2;
  params.spectrum = SpectrumPart::LargestMagnitude;

  auto compressed = uni20::krylov::compress_real_schur_arnoldi_restart<double>(ops, factorization, params);

  ASSERT_EQ(compressed.basis.size(), compressed.schur_form.rows());
  ASSERT_EQ(compressed.schur_form.rows(), compressed.schur_form.cols());
  ASSERT_EQ(compressed.residual_coupling.size(), compressed.basis.size());
  ASSERT_EQ(compressed.eigenvalues.size(), compressed.basis.size());
  EXPECT_FALSE(compressed.happy_breakdown);
  EXPECT_GT(compressed.residual_norm, 0.0);

  for (std::size_t i = 0; i < compressed.basis.size(); ++i)
  {
    for (std::size_t j = 0; j < compressed.basis.size(); ++j)
    {
      double const expected = i == j ? 1.0 : 0.0;
      EXPECT_NEAR(ops.inner_product(compressed.basis[i], compressed.basis[j]), expected, 1.0e-11);
    }
  }

  for (std::size_t column = 0; column < compressed.basis.size(); ++column)
  {
    auto applied = ops.allocate_like(initial);
    ops.matvec(applied, compressed.basis[column]);

    auto expected = ops.allocate_like(initial);
    ops.set_zero(expected);
    for (std::size_t row = 0; row < compressed.basis.size(); ++row)
    {
      ops.axpy(expected, compressed.schur_form[row, column], compressed.basis[row]);
    }
    ops.axpy(expected, compressed.residual_coupling[column], compressed.residual);

    for (std::size_t row = 0; row < applied.values.size(); ++row)
    {
      applied.values[row] -= expected.values[row];
    }
    EXPECT_LT(vector_norm(applied), 1.0e-10);
  }
}

TEST(KrylovNonsymmetricArnoldi, ExpandsRealSchurRestartBackToArnoldiRelation)
{
  std::vector<double> const matrix{
      3.0, 2.0, 0.0, 0.0, //
      0.0, 2.0, 1.0, 0.0, //
      0.0, 0.0, 1.0, 4.0, //
      0.5, 0.0, 0.0, 0.25,
  };
  DenseHostVectorOps<double> ops(4, matrix);
  DenseHostVector<double> initial{{1.0, 2.0, 3.0, 5.0}};

  auto factorization = uni20::krylov::arnoldi_factorize<double>(ops, initial, 3);
  NonsymmetricEigenParams<double> params;
  params.eigenvalue_count = 1;
  params.retained_ritz_count = 2;
  params.spectrum = SpectrumPart::LargestMagnitude;

  auto compressed = uni20::krylov::compress_real_schur_arnoldi_restart<double>(ops, factorization, params);
  auto retained_only = uni20::krylov::expand_real_schur_arnoldi_restart<double>(ops, compressed, 2);
  auto expanded = uni20::krylov::expand_real_schur_arnoldi_restart<double>(ops, compressed, 3);

  ASSERT_EQ(retained_only.step_count, 2);
  ASSERT_FALSE(retained_only.happy_breakdown);
  ASSERT_EQ(retained_only.basis.size(), 3);
  ASSERT_EQ(retained_only.hessenberg.rows(), 3);
  ASSERT_EQ(retained_only.hessenberg.cols(), 2);
  EXPECT_EQ(retained_only.op_count, 0);

  for (std::size_t column = 0; column < static_cast<std::size_t>(retained_only.step_count); ++column)
  {
    auto applied = ops.allocate_like(initial);
    ops.matvec(applied, retained_only.basis[column]);
    auto projected = projected_column(retained_only.basis, retained_only.hessenberg, column);
    for (std::size_t row = 0; row < applied.values.size(); ++row)
    {
      applied.values[row] -= projected.values[row];
    }
    EXPECT_LT(vector_norm(applied), 1.0e-10);
  }

  ASSERT_EQ(expanded.step_count, 3);
  ASSERT_FALSE(expanded.happy_breakdown);
  ASSERT_EQ(expanded.basis.size(), 4);
  ASSERT_EQ(expanded.hessenberg.rows(), 4);
  ASSERT_EQ(expanded.hessenberg.cols(), 3);
  EXPECT_EQ(expanded.op_count, 1);

  for (std::size_t i = 0; i < expanded.basis.size(); ++i)
  {
    for (std::size_t j = 0; j < expanded.basis.size(); ++j)
    {
      double const expected = i == j ? 1.0 : 0.0;
      EXPECT_NEAR(ops.inner_product(expanded.basis[i], expanded.basis[j]), expected, 1.0e-11);
    }
  }

  for (std::size_t column = 0; column < static_cast<std::size_t>(expanded.step_count); ++column)
  {
    auto applied = ops.allocate_like(initial);
    ops.matvec(applied, expanded.basis[column]);
    auto projected = projected_column(expanded.basis, expanded.hessenberg, column);
    for (std::size_t row = 0; row < applied.values.size(); ++row)
    {
      applied.values[row] -= projected.values[row];
    }
    EXPECT_LT(vector_norm(applied), 1.0e-10);
  }
}

TEST(KrylovNonsymmetricArnoldi, ReportsHappyBreakdownOnInvariantSubspace)
{
  std::vector<double> const matrix{
      2.0,
      0.0,
      0.0,
      3.0,
  };
  DenseHostVectorOps<double> ops(2, matrix);
  DenseHostVector<double> initial{{1.0, 0.0}};

  auto factorization = uni20::krylov::arnoldi_factorize<double>(ops, initial, 2);

  EXPECT_EQ(factorization.step_count, 1);
  EXPECT_TRUE(factorization.happy_breakdown);
  EXPECT_EQ(factorization.basis.size(), 1);
  EXPECT_NEAR((factorization.hessenberg[0, 0]), 2.0, 1.0e-14);
  EXPECT_NEAR(factorization.residual_norm, 0.0, 1.0e-14);
}

TEST(KrylovNonsymmetricArnoldi, UsesSingleArnoldiReorthogonalizationPassWhenStable)
{
  std::vector<double> const matrix{
      0.0,
      0.0, //
      1.0,
      0.0,
  };
  DenseHostVectorOps<double> ops(2, matrix);
  DenseHostVector<double> initial{{1.0, 0.0}};

  auto factorization = uni20::krylov::arnoldi_factorize<double>(ops, initial, 1);

  EXPECT_FALSE(factorization.happy_breakdown);
  EXPECT_EQ(factorization.step_count, 1);
  EXPECT_EQ(ops.inner_product_count(), 1);
  EXPECT_EQ(factorization.basis.size(), 2);
}

TEST(KrylovNonsymmetricArnoldi, BreakdownThresholdScalesWithLocalRelation)
{
  double const unit_threshold = uni20::krylov::detail::arnoldi_breakdown_threshold(0.0, 1.0);
  double const large_threshold = uni20::krylov::detail::arnoldi_breakdown_threshold(0.0, 1.0e12);

  EXPECT_LT(unit_threshold, 1.0e-12);
  EXPECT_GT(large_threshold, 1.0e-6);
  EXPECT_LT(large_threshold, 1.0e-2);
}

TEST(KrylovNonsymmetricArnoldi, ExtractsRealRitzValuesFromInvariantSubspace)
{
  std::vector<double> const matrix{
      2.0, 0.0, 0.0, //
      0.0, 3.0, 0.0, //
      0.0, 0.0, 5.0,
  };
  DenseHostVectorOps<double> ops(3, matrix);
  DenseHostVector<double> initial{{1.0, 1.0, 1.0}};

  auto factorization = uni20::krylov::arnoldi_factorize<double>(ops, initial, 3);
  auto ritz = uni20::krylov::extract_arnoldi_ritz(factorization, 1.0e-12);

  ASSERT_EQ(factorization.step_count, 3);
  ASSERT_TRUE(factorization.happy_breakdown);
  ASSERT_EQ(ritz.ritz_values.size(), 3);
  ASSERT_EQ(ritz.residual_bounds.size(), 3);
  ASSERT_EQ(ritz.reality.size(), 3);
  for (double const eigenvalue : {2.0, 3.0, 5.0})
  {
    auto const nearest = nearest_eigenvalue(ritz.ritz_values, uni20::complex<double>{eigenvalue, 0.0});
    EXPECT_NEAR(nearest.real(), eigenvalue, 1.0e-12);
    EXPECT_NEAR(nearest.imag(), 0.0, 1.0e-12);
  }
  for (std::size_t i = 0; i < ritz.ritz_values.size(); ++i)
  {
    EXPECT_EQ(ritz.reality[i], RitzReality::Real);
    EXPECT_NEAR(ritz.residual_bounds[i], 0.0, 1.0e-12);
  }
}

TEST(KrylovNonsymmetricArnoldi, ExtractsComplexConjugateRitzPairFromRealOperator)
{
  std::vector<double> const matrix{
      1.0,
      2.0, //
      -2.0,
      1.0,
  };
  DenseHostVectorOps<double> ops(2, matrix);
  DenseHostVector<double> initial{{1.0, 0.0}};

  auto factorization = uni20::krylov::arnoldi_factorize<double>(ops, initial, 2);
  auto ritz = uni20::krylov::extract_arnoldi_ritz(factorization, 1.0e-12);

  ASSERT_EQ(factorization.step_count, 2);
  ASSERT_TRUE(factorization.happy_breakdown);
  ASSERT_EQ(ritz.ritz_values.size(), 2);

  auto const upper = nearest_eigenvalue(ritz.ritz_values, uni20::complex<double>{1.0, 2.0});
  auto const lower = nearest_eigenvalue(ritz.ritz_values, uni20::complex<double>{1.0, -2.0});
  EXPECT_NEAR(upper.real(), 1.0, 1.0e-12);
  EXPECT_NEAR(upper.imag(), 2.0, 1.0e-12);
  EXPECT_NEAR(lower.real(), 1.0, 1.0e-12);
  EXPECT_NEAR(lower.imag(), -2.0, 1.0e-12);

  for (std::size_t i = 0; i < ritz.ritz_values.size(); ++i)
  {
    EXPECT_EQ(ritz.reality[i], RitzReality::Complex);
    EXPECT_NEAR(ritz.residual_bounds[i], 0.0, 1.0e-12);
  }
}

TEST(KrylovNonsymmetricArnoldi, MeasuresProjectedDepartureFromNormality)
{
  uni20::krylov::Matrix<double> normal(2, 2);
  normal[0, 0] = 1.0;
  normal[1, 0] = -2.0;
  normal[0, 1] = 2.0;
  normal[1, 1] = 1.0;

  uni20::krylov::Matrix<double> nonnormal(2, 2);
  nonnormal[0, 0] = 1.0;
  nonnormal[1, 0] = 0.0;
  nonnormal[0, 1] = 10.0;
  nonnormal[1, 1] = 1.0;

  EXPECT_NEAR(uni20::krylov::projected_departure_from_normality(normal), 0.0, 1.0e-14);
  EXPECT_GT(uni20::krylov::projected_departure_from_normality(nonnormal), 1.0);
}

TEST(KrylovNonsymmetricArnoldi, SelectsNonsymmetricRitzValuesByRequestedSpectrum)
{
  std::vector<uni20::complex<double>> const values{
      {2.0, 0.0},
      {-4.0, 0.0},
      {1.0, 3.0},
      {1.0, -3.0},
  };
  NonsymmetricEigenParams<double> params;
  params.eigenvalue_count = 2;

  params.spectrum = SpectrumPart::LargestMagnitude;
  auto indices = uni20::krylov::select_nonsymmetric_ritz_indices(values, params);
  ASSERT_EQ(indices.size(), 2);
  EXPECT_EQ(indices[0], 1);
  EXPECT_EQ(indices[1], 2);

  params.spectrum = SpectrumPart::SmallestReal;
  indices = uni20::krylov::select_nonsymmetric_ritz_indices(values, params);
  ASSERT_EQ(indices.size(), 2);
  EXPECT_EQ(indices[0], 1);
  EXPECT_EQ(indices[1], 2);

  params.eigenvalue_count = 1;
  params.spectrum = SpectrumPart::LargestImaginary;
  indices = uni20::krylov::select_nonsymmetric_ritz_indices(values, params);
  ASSERT_EQ(indices.size(), 1);
  EXPECT_EQ(indices[0], 2);

  params.spectrum = SpectrumPart::LargestAlgebraic;
  EXPECT_THROW((void)uni20::krylov::select_nonsymmetric_ritz_indices(values, params), std::invalid_argument);
}

TEST(KrylovNonsymmetricArnoldi, SelectsRealSchurRestartBlocksWithoutSplittingComplexPairs)
{
  std::vector<uni20::krylov::RealSchurBlock<double>> const blocks{
      {.begin = 0, .size = 1, .first_eigenvalue = {5.0, 0.0}, .second_eigenvalue = {}},
      {.begin = 1, .size = 2, .first_eigenvalue = {1.0, 2.0}, .second_eigenvalue = {1.0, -2.0}},
      {.begin = 3, .size = 1, .first_eigenvalue = {-7.0, 0.0}, .second_eigenvalue = {}},
  };

  NonsymmetricEigenParams<double> params;
  params.eigenvalue_count = 2;
  params.retained_ritz_count = 2;
  params.spectrum = SpectrumPart::LargestMagnitude;
  auto selection = uni20::krylov::select_real_schur_restart_blocks(blocks, params);
  ASSERT_EQ(selection.block_indices.size(), 2);
  EXPECT_EQ(selection.block_indices[0], 2);
  EXPECT_EQ(selection.block_indices[1], 0);
  EXPECT_EQ(selection.scalar_count, 2);
  EXPECT_FALSE(selection.enlarged_to_preserve_block);

  params.eigenvalue_count = 1;
  params.retained_ritz_count = 1;
  params.spectrum = SpectrumPart::LargestImaginary;
  selection = uni20::krylov::select_real_schur_restart_blocks(blocks, params);
  ASSERT_EQ(selection.block_indices.size(), 1);
  EXPECT_EQ(selection.block_indices[0], 1);
  EXPECT_EQ(selection.scalar_count, 2);
  EXPECT_TRUE(selection.enlarged_to_preserve_block);

  params.eigenvalue_count = 1;
  params.retained_ritz_count = 3;
  params.spectrum = SpectrumPart::LargestReal;
  selection = uni20::krylov::select_real_schur_restart_blocks(blocks, params);
  ASSERT_EQ(selection.block_indices.size(), 2);
  EXPECT_EQ(selection.block_indices[0], 0);
  EXPECT_EQ(selection.block_indices[1], 1);
  EXPECT_EQ(selection.scalar_count, 3);
  EXPECT_FALSE(selection.enlarged_to_preserve_block);

  params.spectrum = SpectrumPart::LargestAlgebraic;
  EXPECT_THROW((void)uni20::krylov::select_real_schur_restart_blocks(blocks, params), std::invalid_argument);
}

TEST(KrylovNonsymmetricArnoldi, ReportsPolicySpecificComplexWantedStatus)
{
  std::vector<double> const matrix{
      1.0,
      2.0, //
      -2.0,
      1.0,
  };
  DenseHostVector<double> initial{{1.0, 0.0}};

  auto solve_with_policy = [&](uni20::krylov::RealNonsymmetricPolicy policy) {
    DenseHostVectorOps<double> ops(2, matrix);
    NonsymmetricEigenParams<double> params;
    params.eigenvalue_count = 1;
    params.krylov_dimension = 2;
    params.spectrum = SpectrumPart::LargestImaginary;
    params.tolerance = 1.0e-12;
    params.compute_eigenvectors = true;
    params.real_policy = policy;
    return uni20::krylov::real_nonsymmetric_arnoldi_standard(ops, initial, params);
  };

  EXPECT_EQ(solve_with_policy(uni20::krylov::RealNonsymmetricPolicy::RequireRealEigenpairs).status,
            NonsymmetricStatus::ComplexPairEncountered);
  EXPECT_EQ(solve_with_policy(uni20::krylov::RealNonsymmetricPolicy::PromoteToComplexSuggested).status,
            NonsymmetricStatus::ComplexPromotionRecommended);
  EXPECT_EQ(solve_with_policy(uni20::krylov::RealNonsymmetricPolicy::AllowRealSchurPairs).status,
            NonsymmetricStatus::RealSchurPairRequired);
}

TEST(KrylovNonsymmetricArnoldi, RealPathSuppressesEigenvectorsForMixedRealitySelection)
{
  std::vector<double> const matrix{
      5.0, 0.0,  0.0, //
      0.0, 1.0,  2.0, //
      0.0, -2.0, 1.0,
  };
  DenseHostVectorOps<double> ops(3, matrix);
  DenseHostVector<double> initial{{1.0, 1.0, 0.0}};
  NonsymmetricEigenParams<double> params;
  params.eigenvalue_count = 2;
  params.krylov_dimension = 3;
  params.spectrum = SpectrumPart::LargestMagnitude;
  params.tolerance = 1.0e-12;
  params.compute_eigenvectors = true;

  auto result = uni20::krylov::real_nonsymmetric_arnoldi_standard(ops, initial, params);

  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.reality.size(), 2);
  EXPECT_EQ(result.status, NonsymmetricStatus::ComplexPairEncountered);
  EXPECT_TRUE(std::ranges::contains(result.reality, RitzReality::Real));
  EXPECT_TRUE(std::ranges::contains(result.reality, RitzReality::Complex));
  EXPECT_TRUE(result.right_eigenvectors.empty());
}

TEST(KrylovNonsymmetricArnoldi, SolvesNonrestartedRealNonsymmetricDiagonalProblem)
{
  std::vector<double> const matrix{
      2.0, 0.0, 0.0, //
      0.0, 3.0, 0.0, //
      0.0, 0.0, 5.0,
  };
  DenseHostVectorOps<double> ops(3, matrix);
  DenseHostVector<double> initial{{1.0, 1.0, 1.0}};
  NonsymmetricEigenParams<double> params;
  params.eigenvalue_count = 2;
  params.krylov_dimension = 3;
  params.spectrum = SpectrumPart::LargestReal;
  params.tolerance = 1.0e-12;
  params.compute_eigenvectors = true;
  params.diagnostics = KrylovDiagnosticsLevel::Summary;

  auto result = uni20::krylov::real_nonsymmetric_arnoldi_standard(ops, initial, params);

  ASSERT_EQ(result.status, NonsymmetricStatus::Converged);
  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.residual_bounds.size(), 2);
  ASSERT_EQ(result.reality.size(), 2);
  ASSERT_EQ(result.right_eigenvectors.size(), 2);
  EXPECT_EQ(result.converged_count, 2);
  EXPECT_EQ(result.iteration_count, 3);
  EXPECT_EQ(result.matvec_count, 3);
  EXPECT_NEAR(result.eigenvalues[0].real(), 5.0, 1.0e-12);
  EXPECT_NEAR(result.eigenvalues[1].real(), 3.0, 1.0e-12);
  EXPECT_NEAR(result.eigenvalues[0].imag(), 0.0, 1.0e-12);
  EXPECT_NEAR(result.eigenvalues[1].imag(), 0.0, 1.0e-12);
  EXPECT_EQ(result.reality[0], RitzReality::Real);
  EXPECT_EQ(result.reality[1], RitzReality::Real);
  EXPECT_LT(relative_eigen_residual(ops, result.right_eigenvectors[0], result.eigenvalues[0].real()), 1.0e-12);
  EXPECT_LT(relative_eigen_residual(ops, result.right_eigenvectors[1], result.eigenvalues[1].real()), 1.0e-12);
  ASSERT_TRUE(result.diagnostics.has_value());
  EXPECT_NEAR(result.diagnostics->projected_departure_from_normality, 0.0, 1.0e-12);
}

TEST(KrylovNonsymmetricArnoldi, ReportsRealHappyBreakdownBeforeRequestedEigenvalueCount)
{
  std::vector<double> const matrix{
      2.0, 0.0, 0.0, //
      0.0, 3.0, 0.0, //
      0.0, 0.0, 5.0,
  };
  DenseHostVectorOps<double> ops(3, matrix);
  DenseHostVector<double> initial{{0.0, 0.0, 1.0}};
  NonsymmetricEigenParams<double> params;
  params.eigenvalue_count = 2;
  params.krylov_dimension = 3;
  params.spectrum = SpectrumPart::LargestReal;
  params.tolerance = 1.0e-12;
  params.compute_eigenvectors = false;
  params.diagnostics = KrylovDiagnosticsLevel::Summary;

  auto result = uni20::krylov::real_nonsymmetric_arnoldi_standard(ops, initial, params);

  EXPECT_EQ(result.status, NonsymmetricStatus::Breakdown);
  ASSERT_EQ(result.eigenvalues.size(), 1);
  ASSERT_EQ(result.residual_bounds.size(), 1);
  EXPECT_EQ(result.converged_count, 1);
  EXPECT_NEAR(result.eigenvalues[0].real(), 5.0, 1.0e-12);
  EXPECT_NEAR(result.eigenvalues[0].imag(), 0.0, 1.0e-12);
  ASSERT_TRUE(result.diagnostics.has_value());
  EXPECT_EQ(result.diagnostics->final_projected_dimension, 1);
}

TYPED_TEST(KrylovNonsymmetricArnoldiRealTypedTest, SolvesNonrestartedRealNonsymmetricDiagonalProblem)
{
  using Real = TypeParam;

  std::vector<Real> const matrix{
      Real{2}, Real{},  Real{}, //
      Real{},  Real{3}, Real{}, //
      Real{},  Real{},  Real{5},
  };
  DenseHostVectorOps<Real> ops(3, matrix);
  DenseHostVector<Real> initial{{Real{1}, Real{1}, Real{1}}};
  NonsymmetricEigenParams<Real> params;
  params.eigenvalue_count = 2;
  params.krylov_dimension = 3;
  params.spectrum = SpectrumPart::LargestReal;
  params.tolerance = static_cast<Real>(arnoldi_tolerance<Real>());
  params.compute_eigenvectors = true;
  params.diagnostics = KrylovDiagnosticsLevel::Summary;

  auto result = uni20::krylov::real_nonsymmetric_arnoldi_standard(ops, initial, params);

  ASSERT_EQ(result.status, NonsymmetricStatus::Converged);
  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.residual_bounds.size(), 2);
  ASSERT_EQ(result.reality.size(), 2);
  ASSERT_EQ(result.right_eigenvectors.size(), 2);
  EXPECT_EQ(result.converged_count, 2);
  EXPECT_EQ(result.iteration_count, 3);
  EXPECT_EQ(result.matvec_count, 3);
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[0].real()), 5.0, arnoldi_tolerance<Real>());
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[1].real()), 3.0, arnoldi_tolerance<Real>());
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[0].imag()), 0.0, arnoldi_tolerance<Real>());
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[1].imag()), 0.0, arnoldi_tolerance<Real>());
  EXPECT_EQ(result.reality[0], RitzReality::Real);
  EXPECT_EQ(result.reality[1], RitzReality::Real);
  ASSERT_TRUE(result.diagnostics.has_value());
  EXPECT_NEAR(static_cast<double>(result.diagnostics->projected_departure_from_normality), 0.0,
              arnoldi_tolerance<Real>());
}

TEST(KrylovNonsymmetricArnoldi, SolvesNonrestartedComplexNonsymmetricDiagonalProblem)
{
  using Complex = uni20::complex<double>;

  std::vector<Complex> const matrix{
      Complex{1.0, 1.0}, Complex{},         Complex{},          Complex{}, //
      Complex{},         Complex{3.0, 0.5}, Complex{},          Complex{}, //
      Complex{},         Complex{},         Complex{-2.0, 2.5}, Complex{}, //
      Complex{},         Complex{},         Complex{},          Complex{0.25, -0.1},
  };
  DenseHostVectorOps<Complex> ops(4, matrix);
  DenseHostVector<Complex> initial{{Complex{1.0, 0.0}, Complex{1.0, -0.5}, Complex{-0.25, 1.0}, Complex{2.0, 0.25}}};
  NonsymmetricEigenParams<double> params;
  params.eigenvalue_count = 2;
  params.krylov_dimension = 4;
  params.spectrum = SpectrumPart::LargestMagnitude;
  params.tolerance = 1.0e-12;
  params.compute_eigenvectors = true;
  params.diagnostics = KrylovDiagnosticsLevel::Summary;

  auto result = uni20::krylov::complex_nonsymmetric_arnoldi_standard<double>(ops, initial, params);

  ASSERT_EQ(result.status, NonsymmetricStatus::Converged);
  ASSERT_EQ(result.converged_count, 2);
  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.residual_bounds.size(), 2);
  ASSERT_EQ(result.right_eigenvectors.size(), 2);
  EXPECT_EQ(result.matvec_count, 4);
  ASSERT_TRUE(result.diagnostics.has_value());
  EXPECT_EQ(result.diagnostics->final_projected_dimension, 4);

  auto const dominant = nearest_eigenvalue(result.eigenvalues, Complex{-2.0, 2.5});
  auto const subdominant = nearest_eigenvalue(result.eigenvalues, Complex{3.0, 0.5});
  EXPECT_NEAR(std::abs(dominant - Complex{-2.0, 2.5}), 0.0, 1.0e-12);
  EXPECT_NEAR(std::abs(subdominant - Complex{3.0, 0.5}), 0.0, 1.0e-12);

  for (std::size_t i = 0; i < result.eigenvalues.size(); ++i)
  {
    EXPECT_LT(result.residual_bounds[i], 1.0e-12);
    EXPECT_LT(relative_eigen_residual(ops, result.right_eigenvectors[i], result.eigenvalues[i]), 1.0e-11);
  }
}

TEST(KrylovNonsymmetricArnoldi, SolvesOneDimensionalComplexFullSubspaceProblem)
{
  using Complex = uni20::complex<double>;

  DenseHostVectorOps<Complex> ops(1, {Complex{2.0, -0.5}});
  DenseHostVector<Complex> initial{{Complex{1.0, 1.0}}};

  NonsymmetricEigenParams<double> params;
  params.eigenvalue_count = 1;
  params.tolerance = 1.0e-12;
  params.compute_eigenvectors = true;
  params.diagnostics = KrylovDiagnosticsLevel::Summary;

  auto result = uni20::krylov::complex_nonsymmetric_arnoldi_standard<double>(ops, initial, params);

  ASSERT_EQ(result.status, NonsymmetricStatus::Converged);
  ASSERT_EQ(result.converged_count, 1);
  ASSERT_EQ(result.eigenvalues.size(), 1);
  ASSERT_EQ(result.residual_bounds.size(), 1);
  ASSERT_EQ(result.right_eigenvectors.size(), 1);
  EXPECT_EQ(result.iteration_count, 1);
  EXPECT_EQ(result.matvec_count, 1);
  EXPECT_EQ(ops.matvec_count(), 1);
  EXPECT_NEAR(std::abs(result.eigenvalues[0] - Complex{2.0, -0.5}), 0.0, 1.0e-12);
  EXPECT_LT(result.residual_bounds[0], 1.0e-12);
  EXPECT_LT(relative_eigen_residual(ops, result.right_eigenvectors[0], result.eigenvalues[0]), 1.0e-12);
  ASSERT_TRUE(result.diagnostics.has_value());
  EXPECT_EQ(result.diagnostics->final_projected_dimension, 1);
}

TEST(KrylovNonsymmetricArnoldi, ReportsComplexHappyBreakdownBeforeRequestedEigenvalueCount)
{
  using Complex = uni20::complex<double>;

  std::vector<Complex> const matrix{
      Complex{1.0, 1.0}, Complex{},         Complex{}, //
      Complex{},         Complex{3.0, 0.5}, Complex{}, //
      Complex{},         Complex{},         Complex{5.0, -0.25},
  };
  DenseHostVectorOps<Complex> ops(3, matrix);
  DenseHostVector<Complex> initial{{Complex{}, Complex{}, Complex{1.0, 0.0}}};
  NonsymmetricEigenParams<double> params;
  params.eigenvalue_count = 2;
  params.krylov_dimension = 3;
  params.spectrum = SpectrumPart::LargestReal;
  params.tolerance = 1.0e-12;
  params.compute_eigenvectors = false;
  params.diagnostics = KrylovDiagnosticsLevel::Summary;

  auto result = uni20::krylov::complex_nonsymmetric_arnoldi_standard<double>(ops, initial, params);

  EXPECT_EQ(result.status, NonsymmetricStatus::Breakdown);
  ASSERT_EQ(result.eigenvalues.size(), 1);
  ASSERT_EQ(result.residual_bounds.size(), 1);
  EXPECT_EQ(result.converged_count, 1);
  EXPECT_NEAR(result.eigenvalues[0].real(), 5.0, 1.0e-12);
  EXPECT_NEAR(result.eigenvalues[0].imag(), -0.25, 1.0e-12);
  ASSERT_TRUE(result.diagnostics.has_value());
  EXPECT_EQ(result.diagnostics->final_projected_dimension, 1);
}

TYPED_TEST(KrylovNonsymmetricArnoldiComplexTypedTest, SolvesNonrestartedComplexNonsymmetricDiagonalProblem)
{
  using Complex = TypeParam;
  using Real = typename Complex::value_type;

  std::vector<Complex> const matrix{
      Complex{Real{1}, Real{1}},
      Complex{},
      Complex{},
      Complex{}, //
      Complex{},
      Complex{Real{3}, Real{0.5}},
      Complex{},
      Complex{}, //
      Complex{},
      Complex{},
      Complex{Real{-2}, Real{2.5}},
      Complex{}, //
      Complex{},
      Complex{},
      Complex{},
      Complex{Real{0.25}, Real{-0.1}},
  };
  DenseHostVectorOps<Complex> ops(4, matrix);
  DenseHostVector<Complex> initial{{Complex{Real{1}, Real{}}, Complex{Real{1}, Real{-0.5}},
                                    Complex{Real{-0.25}, Real{1}}, Complex{Real{2}, Real{0.25}}}};
  NonsymmetricEigenParams<Real> params;
  params.eigenvalue_count = 2;
  params.krylov_dimension = 4;
  params.spectrum = SpectrumPart::LargestMagnitude;
  params.tolerance = static_cast<Real>(arnoldi_tolerance<Real>());
  params.compute_eigenvectors = true;
  params.diagnostics = KrylovDiagnosticsLevel::Summary;

  auto result = uni20::krylov::complex_nonsymmetric_arnoldi_standard<Real>(ops, initial, params);

  ASSERT_EQ(result.status, NonsymmetricStatus::Converged);
  ASSERT_EQ(result.converged_count, 2);
  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.residual_bounds.size(), 2);
  ASSERT_EQ(result.right_eigenvectors.size(), 2);
  EXPECT_EQ(result.matvec_count, 4);
  ASSERT_TRUE(result.diagnostics.has_value());
  EXPECT_EQ(result.diagnostics->final_projected_dimension, 4);

  auto nearest = [&](Complex target) {
    return std::abs(result.eigenvalues[0] - target) < std::abs(result.eigenvalues[1] - target) ? result.eigenvalues[0]
                                                                                               : result.eigenvalues[1];
  };
  Complex const dominant = nearest(Complex{Real{-2}, Real{2.5}});
  Complex const subdominant = nearest(Complex{Real{3}, Real{0.5}});
  EXPECT_NEAR(abs_as_double(dominant - Complex{Real{-2}, Real{2.5}}), 0.0, arnoldi_tolerance<Real>());
  EXPECT_NEAR(abs_as_double(subdominant - Complex{Real{3}, Real{0.5}}), 0.0, arnoldi_tolerance<Real>());
}

TYPED_TEST(KrylovNonsymmetricArnoldiComplexTypedTest, RestartedSolveConvergesOnComplexNonsymmetricDiagonalProblem)
{
  using Complex = TypeParam;
  using Real = typename Complex::value_type;

  std::vector<Complex> const matrix{
      Complex{Real{1}, Real{1}},
      Complex{},
      Complex{},
      Complex{}, //
      Complex{},
      Complex{Real{3}, Real{0.5}},
      Complex{},
      Complex{}, //
      Complex{},
      Complex{},
      Complex{Real{-2}, Real{2.5}},
      Complex{}, //
      Complex{},
      Complex{},
      Complex{},
      Complex{Real{0.25}, Real{-0.1}},
  };
  DenseHostVectorOps<Complex> ops(4, matrix);
  DenseHostVector<Complex> initial{{Complex{Real{1}, Real{}}, Complex{Real{1}, Real{-0.5}},
                                    Complex{Real{-0.25}, Real{1}}, Complex{Real{2}, Real{0.25}}}};
  NonsymmetricEigenParams<Real> params;
  params.eigenvalue_count = 2;
  params.retained_ritz_count = 2;
  params.krylov_dimension = 3;
  params.max_iterations = 80;
  params.spectrum = SpectrumPart::LargestMagnitude;
  params.tolerance = static_cast<Real>(arnoldi_tolerance<Real>());
  params.compute_eigenvectors = true;
  params.diagnostics = KrylovDiagnosticsLevel::Summary;

  auto result = uni20::krylov::complex_nonsymmetric_arnoldi_standard<Real>(ops, initial, params);

  ASSERT_EQ(result.status, NonsymmetricStatus::Converged);
  ASSERT_EQ(result.converged_count, 2);
  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.residual_bounds.size(), 2);
  ASSERT_EQ(result.right_eigenvectors.size(), 2);
  EXPECT_GT(result.matvec_count, params.krylov_dimension);
  ASSERT_TRUE(result.diagnostics.has_value());
  EXPECT_GT(result.diagnostics->restart_count, 0);

  NonsymmetricEigenParams<Real> no_vector_params = params;
  no_vector_params.compute_eigenvectors = false;
  DenseHostVectorOps<Complex> no_vector_ops(4, matrix);
  auto no_vector_result =
      uni20::krylov::complex_nonsymmetric_arnoldi_standard<Real>(no_vector_ops, initial, no_vector_params);
  ASSERT_EQ(no_vector_result.status, NonsymmetricStatus::Converged);
  EXPECT_TRUE(no_vector_result.right_eigenvectors.empty());
  int const reconstruction_axpy_count = ops.axpy_count() - no_vector_ops.axpy_count();
  EXPECT_GE(reconstruction_axpy_count, 0);
  EXPECT_LE(reconstruction_axpy_count, params.eigenvalue_count * params.krylov_dimension);

  auto nearest = [&](Complex target) {
    return std::abs(result.eigenvalues[0] - target) < std::abs(result.eigenvalues[1] - target) ? result.eigenvalues[0]
                                                                                               : result.eigenvalues[1];
  };
  Complex const dominant = nearest(Complex{Real{-2}, Real{2.5}});
  Complex const subdominant = nearest(Complex{Real{3}, Real{0.5}});
  EXPECT_NEAR(abs_as_double(dominant - Complex{Real{-2}, Real{2.5}}), 0.0, arnoldi_tolerance<Real>());
  EXPECT_NEAR(abs_as_double(subdominant - Complex{Real{3}, Real{0.5}}), 0.0, arnoldi_tolerance<Real>());
}

TYPED_TEST(KrylovNonsymmetricArnoldiComplexTypedTest, RestartedSolveConvergesOnTransferLikeClusteredSpectrum)
{
  using Complex = TypeParam;
  using Real = typename Complex::value_type;

  constexpr std::size_t dimension = 64;
  std::vector<Complex> const spectrum = transfer_like_clustered_spectrum<Complex>(dimension);
  std::vector<Complex> const matrix = transfer_like_nonnormal_matrix<Complex>(dimension);
  DenseHostVectorOps<Complex> ops(dimension, matrix);
  DenseHostVector<Complex> initial{std::vector<Complex>(dimension)};
  for (std::size_t i = 0; i < dimension; ++i)
  {
    Real const real = static_cast<Real>((13 * i + 5) % 29 + 1);
    Real const imaginary = static_cast<Real>((7 * i + 3) % 23 - 11) / Real{10};
    initial.values[i] = Complex{real, imaginary};
  }

  NonsymmetricEigenParams<Real> params;
  params.eigenvalue_count = 3;
  params.retained_ritz_count = 8;
  params.krylov_dimension = 32;
  params.max_iterations = 900;
  params.spectrum = SpectrumPart::LargestMagnitude;
  params.tolerance = std::is_same_v<Real, float> ? Real{2.0e-5F} : Real{1.0e-10};
  params.compute_eigenvectors = true;
  params.diagnostics = KrylovDiagnosticsLevel::Summary;

  auto result = uni20::krylov::complex_nonsymmetric_arnoldi_standard<Real>(ops, initial, params);

  double const eigen_tolerance = std::is_same_v<Real, float> ? 2.0e-4 : 1.0e-9;
  double const residual_tolerance = std::is_same_v<Real, float> ? 2.0e-4 : 1.0e-8;
  ASSERT_EQ(result.status, NonsymmetricStatus::Converged);
  ASSERT_EQ(result.converged_count, params.eigenvalue_count);
  ASSERT_EQ(result.eigenvalues.size(), static_cast<std::size_t>(params.eigenvalue_count));
  ASSERT_EQ(result.right_eigenvectors.size(), static_cast<std::size_t>(params.eigenvalue_count));
  EXPECT_GT(result.matvec_count, params.krylov_dimension);
  EXPECT_EQ(ops.matvec_count(), result.matvec_count);
  ASSERT_TRUE(result.diagnostics.has_value());
  EXPECT_GT(result.diagnostics->restart_count, 0);
  EXPECT_TRUE(std::isfinite(result.diagnostics->projected_departure_from_normality));
  EXPECT_GT(result.diagnostics->projected_departure_from_normality, 1.0e-5);

  for (std::size_t i = 0; i < static_cast<std::size_t>(params.eigenvalue_count); ++i)
  {
    auto const nearest = std::ranges::min_element(result.eigenvalues, [&](Complex lhs, Complex rhs) {
      return std::abs(lhs - spectrum[i]) < std::abs(rhs - spectrum[i]);
    });
    ASSERT_NE(nearest, result.eigenvalues.end());
    EXPECT_LT(static_cast<double>(std::abs(*nearest - spectrum[i])), eigen_tolerance);
  }

  for (std::size_t i = 0; i < result.eigenvalues.size(); ++i)
  {
    EXPECT_LT(static_cast<double>(result.residual_bounds[i]), residual_tolerance);
    EXPECT_LT(typed_relative_eigen_residual(ops, result.right_eigenvectors[i], result.eigenvalues[i]),
              Real{20} * residual_tolerance);
  }
}

TEST(KrylovNonsymmetricArnoldi, RestartedSolveConvergesOnRealNonsymmetricDiagonalProblem)
{
  std::vector<double> const matrix{
      2.0, 0.0, 0.0, 0.0, //
      0.0, 3.0, 0.0, 0.0, //
      0.0, 0.0, 5.0, 0.0, //
      0.0, 0.0, 0.0, 7.0,
  };
  DenseHostVectorOps<double> ops(4, matrix);
  DenseHostVector<double> initial{{1.0, 1.0, 1.0, 1.0}};
  NonsymmetricEigenParams<double> params;
  params.eigenvalue_count = 2;
  params.retained_ritz_count = 2;
  params.krylov_dimension = 3;
  params.max_iterations = 60;
  params.spectrum = SpectrumPart::LargestReal;
  params.tolerance = 1.0e-12;
  params.compute_eigenvectors = true;
  params.diagnostics = KrylovDiagnosticsLevel::Summary;

  auto result = uni20::krylov::real_nonsymmetric_arnoldi_restarted_standard(ops, initial, params);

  ASSERT_EQ(result.status, NonsymmetricStatus::Converged);
  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.residual_bounds.size(), 2);
  ASSERT_EQ(result.reality.size(), 2);
  ASSERT_EQ(result.right_eigenvectors.size(), 2);
  EXPECT_EQ(result.converged_count, 2);
  EXPECT_GT(result.iteration_count, params.krylov_dimension);
  EXPECT_EQ(result.matvec_count, result.iteration_count);
  EXPECT_EQ(ops.matvec_count(), result.matvec_count);
  EXPECT_NEAR(result.eigenvalues[0].real(), 7.0, 1.0e-11);
  EXPECT_NEAR(result.eigenvalues[1].real(), 5.0, 1.0e-11);
  EXPECT_NEAR(result.eigenvalues[0].imag(), 0.0, 1.0e-12);
  EXPECT_NEAR(result.eigenvalues[1].imag(), 0.0, 1.0e-12);
  NonsymmetricEigenParams<double> no_vector_params = params;
  no_vector_params.compute_eigenvectors = false;
  DenseHostVectorOps<double> no_vector_ops(4, matrix);
  auto no_vector_result =
      uni20::krylov::real_nonsymmetric_arnoldi_restarted_standard(no_vector_ops, initial, no_vector_params);
  ASSERT_EQ(no_vector_result.status, NonsymmetricStatus::Converged);
  EXPECT_TRUE(no_vector_result.right_eigenvectors.empty());
  int const reconstruction_axpy_count = ops.axpy_count() - no_vector_ops.axpy_count();
  EXPECT_GE(reconstruction_axpy_count, 0);
  EXPECT_LE(reconstruction_axpy_count, params.eigenvalue_count * params.krylov_dimension);
  EXPECT_LT(relative_eigen_residual(ops, result.right_eigenvectors[0], result.eigenvalues[0].real()), 1.0e-10);
  EXPECT_LT(relative_eigen_residual(ops, result.right_eigenvectors[1], result.eigenvalues[1].real()), 1.0e-10);
  ASSERT_TRUE(result.diagnostics.has_value());
  EXPECT_GT(result.diagnostics->restart_count, 0);
  EXPECT_EQ(result.diagnostics->op_count, result.matvec_count);
}

TYPED_TEST(KrylovNonsymmetricArnoldiRealTypedTest, RestartedSolveConvergesOnRealNonsymmetricDiagonalProblem)
{
  using Real = TypeParam;

  std::vector<Real> const matrix{
      Real{2}, Real{},  Real{},  Real{}, //
      Real{},  Real{3}, Real{},  Real{}, //
      Real{},  Real{},  Real{5}, Real{}, //
      Real{},  Real{},  Real{},  Real{7},
  };
  DenseHostVectorOps<Real> ops(4, matrix);
  DenseHostVector<Real> initial{{Real{1}, Real{1}, Real{1}, Real{1}}};
  NonsymmetricEigenParams<Real> params;
  params.eigenvalue_count = 2;
  params.retained_ritz_count = 2;
  params.krylov_dimension = 3;
  params.max_iterations = 60;
  params.spectrum = SpectrumPart::LargestReal;
  params.tolerance = static_cast<Real>(arnoldi_tolerance<Real>());
  params.compute_eigenvectors = true;
  params.diagnostics = KrylovDiagnosticsLevel::Summary;

  auto result = uni20::krylov::real_nonsymmetric_arnoldi_restarted_standard(ops, initial, params);

  ASSERT_EQ(result.status, NonsymmetricStatus::Converged);
  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.residual_bounds.size(), 2);
  ASSERT_EQ(result.reality.size(), 2);
  ASSERT_EQ(result.right_eigenvectors.size(), 2);
  EXPECT_EQ(result.converged_count, 2);
  EXPECT_GT(result.iteration_count, params.krylov_dimension);
  EXPECT_EQ(result.matvec_count, result.iteration_count);
  EXPECT_EQ(ops.matvec_count(), result.matvec_count);
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[0].real()), 7.0, arnoldi_tolerance<Real>());
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[1].real()), 5.0, arnoldi_tolerance<Real>());
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[0].imag()), 0.0, arnoldi_tolerance<Real>());
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[1].imag()), 0.0, arnoldi_tolerance<Real>());
  ASSERT_TRUE(result.diagnostics.has_value());
  EXPECT_GT(result.diagnostics->restart_count, 0);
  EXPECT_EQ(result.diagnostics->op_count, result.matvec_count);
}

TEST(KrylovNonsymmetricArnoldi, ReportsComplexWantedRitzValueFromRealOperator)
{
  std::vector<double> const matrix{
      1.0,
      2.0, //
      -2.0,
      1.0,
  };
  DenseHostVectorOps<double> ops(2, matrix);
  DenseHostVector<double> initial{{1.0, 0.0}};
  NonsymmetricEigenParams<double> params;
  params.eigenvalue_count = 1;
  params.krylov_dimension = 2;
  params.spectrum = SpectrumPart::LargestImaginary;
  params.tolerance = 1.0e-12;
  params.compute_eigenvectors = true;

  auto result = uni20::krylov::real_nonsymmetric_arnoldi_standard(ops, initial, params);

  ASSERT_EQ(result.status, NonsymmetricStatus::ComplexPairEncountered);
  ASSERT_EQ(result.eigenvalues.size(), 1);
  ASSERT_EQ(result.residual_bounds.size(), 1);
  ASSERT_EQ(result.reality.size(), 1);
  EXPECT_EQ(result.converged_count, 1);
  EXPECT_NEAR(result.eigenvalues[0].real(), 1.0, 1.0e-12);
  EXPECT_NEAR(result.eigenvalues[0].imag(), 2.0, 1.0e-12);
  EXPECT_EQ(result.reality[0], RitzReality::Complex);
  EXPECT_TRUE(result.right_eigenvectors.empty());
}

TEST(KrylovNonsymmetricArnoldi, BuildsShortFactorizationForNepLop163Fixture)
{
  auto const matrix =
      read_real_general_matrix_market(std::string(UNI20_KRYLOV_MATRIX_MARKET_DIR) + "/nep/stoch/lop163.mtx");
  ASSERT_EQ(matrix.rows, 163);
  ASSERT_EQ(matrix.cols, 163);
  ASSERT_EQ(matrix.entries.size(), 935);

  SparseHostVectorOps ops(matrix);
  DenseHostVector<double> initial{std::vector<double>(matrix.rows)};
  for (std::size_t i = 0; i < initial.values.size(); ++i)
  {
    initial.values[i] = static_cast<double>((7 * i + 3) % 17 + 1);
  }

  auto factorization = uni20::krylov::arnoldi_factorize<double>(ops, initial, 6);

  EXPECT_EQ(factorization.step_count, 6);
  EXPECT_FALSE(factorization.happy_breakdown);
  EXPECT_EQ(factorization.basis.size(), 7);
  EXPECT_EQ(factorization.op_count, 6);
  EXPECT_EQ(ops.matvec_count(), 6);
  EXPECT_GT(factorization.residual_norm, 0.0);

  for (std::size_t i = 0; i < factorization.basis.size(); ++i)
  {
    for (std::size_t j = 0; j < factorization.basis.size(); ++j)
    {
      double const expected = i == j ? 1.0 : 0.0;
      EXPECT_NEAR(ops.inner_product(factorization.basis[i], factorization.basis[j]), expected, 1.0e-11);
    }
  }

  auto ritz = uni20::krylov::extract_arnoldi_ritz(factorization);
  ASSERT_EQ(ritz.ritz_values.size(), 6);
  ASSERT_EQ(ritz.residual_bounds.size(), 6);
  ASSERT_EQ(ritz.reality.size(), 6);
  for (double const bound : ritz.residual_bounds)
  {
    EXPECT_TRUE(std::isfinite(bound));
    EXPECT_GE(bound, 0.0);
  }
}

TEST(KrylovNonsymmetricArnoldi, LoadsImportedNepFixturesWithExpectedDimensions)
{
  struct Fixture
  {
      std::string path;
      std::size_t dimension = 0;
      std::size_t entry_count = 0;
  };

  std::vector<Fixture> const fixtures{
      {.path = "nep/brussel/rdb200.mtx", .dimension = 200, .entry_count = 1120},
      {.path = "nep/dwave/dwa512.mtx", .dimension = 512, .entry_count = 2480},
      {.path = "nep/dwave/dw2048.mtx", .dimension = 2048, .entry_count = 10114},
      {.path = "nep/mhd/mhd416a.mtx", .dimension = 416, .entry_count = 8562},
      {.path = "nep/olmstead/olm100.mtx", .dimension = 100, .entry_count = 396},
      {.path = "nep/olmstead/olm500.mtx", .dimension = 500, .entry_count = 1996},
      {.path = "nep/stoch/lop163.mtx", .dimension = 163, .entry_count = 935},
      {.path = "nep/tubular/tub1000.mtx", .dimension = 1000, .entry_count = 3996},
  };

  for (auto const& fixture : fixtures)
  {
    SCOPED_TRACE(fixture.path);
    auto const matrix =
        read_real_general_matrix_market(std::string(UNI20_KRYLOV_MATRIX_MARKET_DIR) + "/" + fixture.path);
    EXPECT_EQ(matrix.rows, fixture.dimension);
    EXPECT_EQ(matrix.cols, fixture.dimension);
    EXPECT_EQ(matrix.entries.size(), fixture.entry_count);
  }
}

TEST(KrylovNonsymmetricArnoldi, LoadsGeneratedComplexMatrixMarketFixture)
{
  auto const matrix = read_complex_general_matrix_market(std::string(UNI20_KRYLOV_MATRIX_MARKET_DIR) +
                                                         "/complex_phase_triangular_4.mtx");

  ASSERT_EQ(matrix.rows, 4);
  ASSERT_EQ(matrix.cols, 4);
  ASSERT_EQ(matrix.entries.size(), 7);
  EXPECT_EQ(matrix.entries.front().row, 0);
  EXPECT_EQ(matrix.entries.front().col, 0);
  EXPECT_EQ(matrix.entries.front().value, uni20::complex<double>(1.0, 1.0));
  EXPECT_EQ(matrix.entries.back().row, 3);
  EXPECT_EQ(matrix.entries.back().col, 3);
  EXPECT_EQ(matrix.entries.back().value, uni20::complex<double>(0.25, -0.1));
}

TEST(KrylovNonsymmetricArnoldi, RunsNonrestartedSolverOnNepLop163Fixture)
{
  auto const matrix =
      read_real_general_matrix_market(std::string(UNI20_KRYLOV_MATRIX_MARKET_DIR) + "/nep/stoch/lop163.mtx");
  SparseHostVectorOps ops(matrix);
  DenseHostVector<double> initial{std::vector<double>(matrix.rows)};
  for (std::size_t i = 0; i < initial.values.size(); ++i)
  {
    initial.values[i] = static_cast<double>((11 * i + 5) % 23 + 1);
  }

  NonsymmetricEigenParams<double> params;
  params.eigenvalue_count = 3;
  params.krylov_dimension = 10;
  params.spectrum = SpectrumPart::LargestMagnitude;
  params.compute_eigenvectors = false;
  params.diagnostics = KrylovDiagnosticsLevel::Summary;

  auto result = uni20::krylov::real_nonsymmetric_arnoldi_standard(ops, initial, params);

  ASSERT_EQ(result.eigenvalues.size(), 3);
  ASSERT_EQ(result.residual_bounds.size(), 3);
  ASSERT_EQ(result.reality.size(), 3);
  EXPECT_EQ(result.iteration_count, 10);
  EXPECT_EQ(result.matvec_count, 10);
  EXPECT_EQ(ops.matvec_count(), 10);
  EXPECT_TRUE(result.right_eigenvectors.empty());
  ASSERT_TRUE(result.diagnostics.has_value());
  EXPECT_EQ(result.diagnostics->op_count, 10);
  EXPECT_EQ(result.diagnostics->final_projected_dimension, 10);
  ASSERT_EQ(result.diagnostics->final_ritz_values.size(), 10);
  ASSERT_EQ(result.diagnostics->final_ritz_bounds.size(), 10);
  EXPECT_TRUE(std::isfinite(result.diagnostics->projected_departure_from_normality));
  EXPECT_GE(result.diagnostics->projected_departure_from_normality, 0.0);
  for (double const bound : result.residual_bounds)
  {
    EXPECT_TRUE(std::isfinite(bound));
    EXPECT_GE(bound, 0.0);
  }
}

TEST(KrylovNonsymmetricArnoldi, RunsComplexSolverOnGeneratedMatrixMarketFixture)
{
  using Complex = uni20::complex<double>;

  auto const matrix = read_complex_general_matrix_market(std::string(UNI20_KRYLOV_MATRIX_MARKET_DIR) +
                                                         "/complex_phase_triangular_4.mtx");
  ComplexSparseHostVectorOps ops(matrix);
  DenseHostVector<Complex> initial{{Complex{1.0, 0.0}, Complex{1.0, -0.5}, Complex{-0.25, 1.0}, Complex{2.0, 0.25}}};

  NonsymmetricEigenParams<double> params;
  params.eigenvalue_count = 2;
  params.krylov_dimension = 4;
  params.spectrum = SpectrumPart::LargestMagnitude;
  params.tolerance = 1.0e-12;
  params.compute_eigenvectors = true;
  params.diagnostics = KrylovDiagnosticsLevel::Summary;

  auto result = uni20::krylov::complex_nonsymmetric_arnoldi_standard<double>(ops, initial, params);

  ASSERT_EQ(result.status, NonsymmetricStatus::Converged);
  ASSERT_EQ(result.converged_count, 2);
  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.residual_bounds.size(), 2);
  ASSERT_EQ(result.right_eigenvectors.size(), 2);
  EXPECT_EQ(result.iteration_count, 4);
  EXPECT_EQ(result.matvec_count, 4);
  EXPECT_EQ(ops.matvec_count(), 4);

  auto const dominant = nearest_eigenvalue(result.eigenvalues, Complex{-2.0, 2.5});
  auto const subdominant = nearest_eigenvalue(result.eigenvalues, Complex{3.0, 0.5});
  EXPECT_NEAR(std::abs(dominant - Complex{-2.0, 2.5}), 0.0, 1.0e-12);
  EXPECT_NEAR(std::abs(subdominant - Complex{3.0, 0.5}), 0.0, 1.0e-12);

  ASSERT_TRUE(result.diagnostics.has_value());
  EXPECT_EQ(result.diagnostics->op_count, 4);
  EXPECT_EQ(result.diagnostics->final_projected_dimension, 4);
  EXPECT_TRUE(std::isfinite(result.diagnostics->projected_departure_from_normality));
  for (std::size_t i = 0; i < result.eigenvalues.size(); ++i)
  {
    EXPECT_LT(result.residual_bounds[i], 1.0e-12);
    EXPECT_LT(relative_eigen_residual(ops, result.right_eigenvectors[i], result.eigenvalues[i]), 1.0e-11);
  }
}
