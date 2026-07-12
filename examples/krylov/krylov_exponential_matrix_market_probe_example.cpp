#include <uni20/common/presentation.hpp>
#include <uni20/config.hpp>
#include <uni20/core/numeric_limits.hpp>
#include <uni20/core/types.hpp>
#include <uni20/krylov/dense_host_vector.hpp>
#include <uni20/krylov/krylov_exponential.hpp>
#include <uni20/krylov/taylor_exponential.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <exception>
#include <fmt/core.h>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace
{

namespace presentation = uni20::presentation;

using uni20::krylov::DenseHostVector;
using uni20::krylov::DenseHostVectorOps;
using uni20::krylov::KrylovDiagnosticsLevel;
using uni20::krylov::TaylorExponentialParams;

enum class ScalarPrecision
{
  Float32,
  Float64
#if UNI20_HAS_FLOAT128
  ,
  Float128
#endif
};

struct Options
{
    std::string matrix_path =
        std::string(UNI20_KRYLOV_EXAMPLE_MATRIX_MARKET_DIR) + "/tdvp_lanczos/tdvp_ho_n8_hamiltonian.mtx";
    std::string vector_path = std::string(UNI20_KRYLOV_EXAMPLE_MATRIX_MARKET_DIR) + "/tdvp_lanczos/tdvp_ho_n8_v0.mtx";
    ScalarPrecision precision = ScalarPrecision::Float64;
    double tolerance = 1.0e-8;
    double time_real = 0.0;
    double time_imag = -0.1968473663975394;
    int max_dimension = 40;
    bool all_dimensions = false;
    bool check = false;
};

struct DenseRealMatrix
{
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<double> values;

    [[nodiscard]] double operator[](std::size_t row, std::size_t col) const { return values[row * cols + col]; }

    double& operator[](std::size_t row, std::size_t col) { return values[row * cols + col]; }
};

template <typename Scalar> using RealOf = uni20::make_real_t<Scalar>;

#if UNI20_HAS_FLOAT128
using ReferenceReal = uni20::float128;
#else
using ReferenceReal = double;
#endif

[[nodiscard]] bool starts_with(std::string_view value, std::string_view prefix)
{
  return value.substr(0, prefix.size()) == prefix;
}

[[nodiscard]] std::string_view option_value(std::string_view argument, std::string_view prefix)
{
  if (!starts_with(argument, prefix))
  {
    throw std::invalid_argument(fmt::format("expected option prefix '{}'", prefix));
  }
  return argument.substr(prefix.size());
}

[[nodiscard]] double parse_double(std::string_view text, std::string_view option)
{
  try
  {
    return std::stod(std::string(text));
  }
  catch (std::exception const&)
  {
    throw std::invalid_argument(fmt::format("invalid numeric value for {}: {}", option, text));
  }
}

[[nodiscard]] int parse_int(std::string_view text, std::string_view option)
{
  try
  {
    return std::stoi(std::string(text));
  }
  catch (std::exception const&)
  {
    throw std::invalid_argument(fmt::format("invalid integer value for {}: {}", option, text));
  }
}

void print_usage(char const* program)
{
  fmt::print("Usage: {} [matrix.mtx] [options]\n\n", program);
  fmt::print("Options:\n");
  fmt::print("  --matrix=PATH          real symmetric Matrix Market coordinate file\n");
  fmt::print("  --vector=PATH          real Matrix Market column-vector file\n");
  fmt::print("  --precision=float|double");
#if UNI20_HAS_FLOAT128
  fmt::print("|float128");
#endif
  fmt::print(", default double\n");
  fmt::print("  --tol=VALUE            requested relative tolerance marker, default 1e-8\n");
  fmt::print("  --time-real=VALUE      real part of the exponential coefficient, default 0\n");
  fmt::print("  --time-imag=VALUE      imaginary part of the exponential coefficient, default -0.1968473663975394\n");
  fmt::print("  --max-dim=N            largest Lanczos dimension in the sweep, default 40\n");
  fmt::print("  --all-dimensions       sweep every Krylov dimension from 2 through max-dim\n");
  fmt::print("  --check                fail unless the default diagnostic pathology is observed\n\n");
  fmt::print("Default matrix: {}/tdvp_lanczos/tdvp_ho_n8_hamiltonian.mtx\n", UNI20_KRYLOV_EXAMPLE_MATRIX_MARKET_DIR);
  fmt::print("Default vector: {}/tdvp_lanczos/tdvp_ho_n8_v0.mtx\n", UNI20_KRYLOV_EXAMPLE_MATRIX_MARKET_DIR);
}

[[nodiscard]] Options parse_options(int argc, char** argv)
{
  Options options;
  for (int i = 1; i < argc; ++i)
  {
    std::string_view const argument = argv[i];
    if (argument == "--help" || argument == "-h")
    {
      print_usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    }
    if (argument == "--check")
    {
      options.check = true;
    }
    else if (argument == "--all-dimensions")
    {
      options.all_dimensions = true;
    }
    else if (argument == "--precision=float")
    {
      options.precision = ScalarPrecision::Float32;
    }
    else if (argument == "--precision=double")
    {
      options.precision = ScalarPrecision::Float64;
    }
    else if (argument == "--precision=float128" || argument == "--precision=quad" ||
             argument == "--precision=binary128")
    {
#if UNI20_HAS_FLOAT128
      options.precision = ScalarPrecision::Float128;
#else
      throw std::invalid_argument("float128 precision requires a UNI20_HAS_FLOAT128 build");
#endif
    }
    else if (starts_with(argument, "--matrix="))
    {
      options.matrix_path = std::string(option_value(argument, "--matrix="));
    }
    else if (starts_with(argument, "--vector="))
    {
      options.vector_path = std::string(option_value(argument, "--vector="));
    }
    else if (starts_with(argument, "--tol="))
    {
      options.tolerance = parse_double(option_value(argument, "--tol="), "--tol");
    }
    else if (starts_with(argument, "--time-real="))
    {
      options.time_real = parse_double(option_value(argument, "--time-real="), "--time-real");
    }
    else if (starts_with(argument, "--time-imag="))
    {
      options.time_imag = parse_double(option_value(argument, "--time-imag="), "--time-imag");
    }
    else if (starts_with(argument, "--max-dim="))
    {
      options.max_dimension = parse_int(option_value(argument, "--max-dim="), "--max-dim");
    }
    else if (starts_with(argument, "--"))
    {
      throw std::invalid_argument(fmt::format("unknown option: {}", argument));
    }
    else
    {
      options.matrix_path = std::string(argument);
    }
  }

  if (options.tolerance <= 0.0 || !std::isfinite(options.tolerance))
  {
    throw std::invalid_argument("tolerance must be positive and finite");
  }
  if (!std::isfinite(options.time_real) || !std::isfinite(options.time_imag))
  {
    throw std::invalid_argument("time coefficient must be finite");
  }
  if (options.max_dimension <= 1)
  {
    throw std::invalid_argument("max-dim must be greater than one");
  }
  return options;
}

[[nodiscard]] DenseRealMatrix read_real_coordinate_matrix_market(std::string const& path)
{
  std::ifstream input(path);
  if (!input)
  {
    throw std::invalid_argument("could not open Matrix Market file: " + path);
  }

  std::string header;
  std::getline(input, header);
  std::string banner;
  std::string object;
  std::string format;
  std::string field;
  std::string symmetry;
  std::istringstream header_stream(header);
  header_stream >> banner >> object >> format >> field >> symmetry;
  if (banner != "%%MatrixMarket" || object != "matrix" || format != "coordinate")
  {
    throw std::invalid_argument("example supports Matrix Market coordinate matrices only");
  }
  if (field != "real" && field != "integer")
  {
    throw std::invalid_argument("example supports real or integer Matrix Market fields only");
  }
  if (symmetry != "symmetric" && symmetry != "general")
  {
    throw std::invalid_argument("example supports symmetric or explicitly symmetric general matrices only");
  }

  std::string line;
  do
  {
    if (!std::getline(input, line))
    {
      throw std::invalid_argument("Matrix Market file is missing its size line");
    }
  }
  while (line.empty() || line[0] == '%');

  std::istringstream size_stream(line);
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::size_t stored_entries = 0;
  if (!(size_stream >> rows >> cols >> stored_entries))
  {
    throw std::invalid_argument("Matrix Market file has a malformed size line");
  }
  if (rows != cols)
  {
    throw std::invalid_argument("Hermitian exponential Matrix Market probe requires a square matrix");
  }

  DenseRealMatrix matrix{.rows = rows, .cols = cols, .values = std::vector<double>(rows * cols, 0.0)};
  for (std::size_t entry_index = 0; entry_index < stored_entries; ++entry_index)
  {
    std::size_t row = 0;
    std::size_t col = 0;
    double value = 0.0;
    input >> row >> col >> value;
    if (row == 0 || row > rows || col == 0 || col > cols)
    {
      throw std::invalid_argument("Matrix Market file contains an out-of-range entry");
    }
    --row;
    --col;
    matrix[row, col] += value;
    if (symmetry == "symmetric" && row != col)
    {
      matrix[col, row] += value;
    }
  }

  double max_symmetry_error = 0.0;
  for (std::size_t row = 0; row < rows; ++row)
  {
    for (std::size_t col = row + 1; col < cols; ++col)
    {
      max_symmetry_error = std::max(max_symmetry_error, std::abs(matrix[row, col] - matrix[col, row]));
    }
  }
  if (max_symmetry_error > 1.0e-10)
  {
    throw std::invalid_argument("Hermitian exponential Matrix Market probe requires a symmetric matrix");
  }

  return matrix;
}

[[nodiscard]] std::vector<double> read_real_vector_matrix_market(std::string const& path)
{
  std::ifstream input(path);
  if (!input)
  {
    throw std::invalid_argument("could not open Matrix Market vector file: " + path);
  }

  std::string header;
  std::getline(input, header);
  std::string banner;
  std::string object;
  std::string format;
  std::string field;
  std::string symmetry;
  std::istringstream header_stream(header);
  header_stream >> banner >> object >> format >> field >> symmetry;
  if (banner != "%%MatrixMarket" || object != "matrix" || format != "coordinate")
  {
    throw std::invalid_argument("example supports Matrix Market coordinate vectors only");
  }
  if (field != "real" && field != "integer")
  {
    throw std::invalid_argument("example supports real or integer Matrix Market vectors only");
  }
  if (symmetry != "general")
  {
    throw std::invalid_argument("example supports general Matrix Market vectors only");
  }

  std::string line;
  do
  {
    if (!std::getline(input, line))
    {
      throw std::invalid_argument("Matrix Market vector file is missing its size line");
    }
  }
  while (line.empty() || line[0] == '%');

  std::istringstream size_stream(line);
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::size_t stored_entries = 0;
  size_stream >> rows >> cols >> stored_entries;
  if (cols != 1)
  {
    throw std::invalid_argument("Matrix Market vector file must have exactly one column");
  }

  std::vector<double> vector(rows, 0.0);
  for (std::size_t entry_index = 0; entry_index < stored_entries; ++entry_index)
  {
    std::size_t row = 0;
    std::size_t col = 0;
    double value = 0.0;
    input >> row >> col >> value;
    if (row == 0 || row > rows || col != 1)
    {
      throw std::invalid_argument("Matrix Market vector file contains an out-of-range entry");
    }
    vector[row - 1] += value;
  }
  return vector;
}

[[nodiscard]] double matrix_one_norm_bound(DenseRealMatrix const& matrix)
{
  double result = 0.0;
  for (std::size_t col = 0; col < matrix.cols; ++col)
  {
    double column_sum = 0.0;
    for (std::size_t row = 0; row < matrix.rows; ++row)
    {
      column_sum += std::abs(matrix[row, col]);
    }
    result = std::max(result, column_sum);
  }
  return result;
}

template <typename Scalar> [[nodiscard]] std::vector<Scalar> complex_matrix(DenseRealMatrix const& matrix)
{
  std::vector<Scalar> result(matrix.values.size());
  for (std::size_t i = 0; i < matrix.values.size(); ++i)
  {
    result[i] = Scalar{static_cast<RealOf<Scalar>>(matrix.values[i]), RealOf<Scalar>{}};
  }
  return result;
}

template <typename Scalar> [[nodiscard]] DenseHostVector<Scalar> complex_vector(std::vector<double> const& values)
{
  DenseHostVector<Scalar> vector{{}};
  vector.values.reserve(values.size());
  for (double const value : values)
  {
    vector.values.push_back(Scalar{static_cast<RealOf<Scalar>>(value), RealOf<Scalar>{}});
  }
  return vector;
}

template <typename Scalar> [[nodiscard]] RealOf<Scalar> vector_norm(DenseHostVector<Scalar> const& vector)
{
  RealOf<Scalar> sum{};
  for (auto const& value : vector.values)
  {
    RealOf<Scalar> const magnitude = static_cast<RealOf<Scalar>>(std::abs(value));
    sum += magnitude * magnitude;
  }
  return std::sqrt(sum);
}

template <typename Scalar, typename ReferenceScalar>
[[nodiscard]] RealOf<ReferenceScalar> difference_norm(DenseHostVector<Scalar> const& lhs,
                                                      DenseHostVector<ReferenceScalar> const& rhs)
{
  if (lhs.values.size() != rhs.values.size())
  {
    throw std::invalid_argument("difference_norm requires equal vector sizes");
  }
  using ReferenceReal = RealOf<ReferenceScalar>;

  ReferenceReal sum{};
  for (std::size_t i = 0; i < lhs.values.size(); ++i)
  {
    ReferenceScalar const left{static_cast<ReferenceReal>(std::real(lhs.values[i])),
                               static_cast<ReferenceReal>(std::imag(lhs.values[i]))};
    ReferenceReal const magnitude = static_cast<ReferenceReal>(std::abs(left - rhs.values[i]));
    sum += magnitude * magnitude;
  }
  return std::sqrt(sum);
}

template <typename Scalar>
[[nodiscard]] Scalar host_inner_product(DenseHostVector<Scalar> const& lhs, DenseHostVector<Scalar> const& rhs)
{
  if (lhs.values.size() != rhs.values.size())
  {
    throw std::invalid_argument("host_inner_product requires equal vector sizes");
  }
  Scalar result{};
  for (std::size_t i = 0; i < lhs.values.size(); ++i)
  {
    result += std::conj(lhs.values[i]) * rhs.values[i];
  }
  return result;
}

template <typename Scalar> void host_axpy(DenseHostVector<Scalar>& y, Scalar alpha, DenseHostVector<Scalar> const& x)
{
  if (y.values.size() != x.values.size())
  {
    throw std::invalid_argument("host_axpy requires equal vector sizes");
  }
  for (std::size_t i = 0; i < y.values.size(); ++i)
  {
    y.values[i] += alpha * x.values[i];
  }
}

template <typename Scalar> void host_scale(DenseHostVector<Scalar>& vector, Scalar alpha)
{
  for (auto& value : vector.values)
  {
    value *= alpha;
  }
}

template <typename Scalar> void host_set_zero(DenseHostVector<Scalar>& vector)
{
  for (auto& value : vector.values)
  {
    value = Scalar{};
  }
}

template <typename Real> struct ProbeReorthogonalizationStats
{
    Real max_correction = Real{};
    Real max_correction_ratio = Real{};
    int max_passes = 0;
};

template <typename Real>
void record_probe_reorthogonalization_pass(ProbeReorthogonalizationStats<Real>* stats, Real max_correction,
                                           Real reference_norm, int pass_count)
{
  if (stats == nullptr)
  {
    return;
  }

  stats->max_correction = std::max(stats->max_correction, max_correction);
  if (reference_norm > Real{})
  {
    stats->max_correction_ratio = std::max(stats->max_correction_ratio, max_correction / reference_norm);
  }
  stats->max_passes = std::max(stats->max_passes, pass_count);
}

template <typename Scalar>
[[nodiscard]] RealOf<Scalar> orthogonalize_probe_residual(std::vector<DenseHostVector<Scalar>> const& basis,
                                                          DenseHostVector<Scalar>& residual,
                                                          ProbeReorthogonalizationStats<RealOf<Scalar>>* stats)
{
  using Real = RealOf<Scalar>;
  auto norm = [&]() { return vector_norm(residual); };
  auto orthogonalize_once = [&]() {
    Real max_correction{};
    for (auto const& basis_vector : basis)
    {
      Scalar const correction = host_inner_product(basis_vector, residual);
      max_correction = std::max(max_correction, static_cast<Real>(std::abs(correction)));
      host_axpy(residual, -correction, basis_vector);
    }
    return max_correction;
  };

  Real const original_norm = norm();
  Real const first_max_correction = orthogonalize_once();
  record_probe_reorthogonalization_pass(stats, first_max_correction, original_norm, 1);
  Real residual_norm = norm();
  if (original_norm == Real{} || residual_norm > Real{0.717} * original_norm)
  {
    return residual_norm;
  }

  Real const first_refined_norm = residual_norm;
  Real const second_max_correction = orthogonalize_once();
  record_probe_reorthogonalization_pass(stats, second_max_correction, first_refined_norm, 2);
  residual_norm = norm();
  if (first_refined_norm == Real{} || residual_norm <= Real{0.717} * first_refined_norm)
  {
    host_set_zero(residual);
    return Real{};
  }
  return residual_norm;
}

template <typename Scalar>
[[nodiscard]] DenseHostVector<Scalar> dense_matvec(DenseRealMatrix const& matrix, DenseHostVector<Scalar> const& x)
{
  if (x.values.size() != matrix.cols)
  {
    throw std::invalid_argument("dense Matrix Market matvec vector has the wrong size");
  }
  DenseHostVector<Scalar> y{{std::vector<Scalar>(matrix.rows)}};
  for (std::size_t row = 0; row < matrix.rows; ++row)
  {
    Scalar value{};
    for (std::size_t col = 0; col < matrix.cols; ++col)
    {
      value += static_cast<RealOf<Scalar>>(matrix[row, col]) * x.values[col];
    }
    y.values[row] = value;
  }
  return y;
}

template <typename Scalar>
[[nodiscard]] RealOf<Scalar> basis_max_offdiag(std::vector<DenseHostVector<Scalar>> const& basis)
{
  RealOf<Scalar> result{};
  for (std::size_t row = 0; row < basis.size(); ++row)
  {
    for (std::size_t col = row + 1; col < basis.size(); ++col)
    {
      result = std::max(result, static_cast<RealOf<Scalar>>(std::abs(host_inner_product(basis[row], basis[col]))));
    }
  }
  return result;
}

enum class LanczosProbeVariant
{
  FullReorthogonalized,
  LegacyThreeTerm
};

[[nodiscard]] std::vector<int> sweep_dimensions(int max_dimension, std::size_t matrix_dimension, bool all_dimensions)
{
  int const capped = std::min(max_dimension, static_cast<int>(matrix_dimension));
  std::vector<int> result;
  if (all_dimensions)
  {
    result.reserve(static_cast<std::size_t>(std::max(0, capped - 1)));
    for (int dimension = 2; dimension <= capped; ++dimension)
    {
      result.push_back(dimension);
    }
    return result;
  }

  for (int dimension : {4, 6, 8, 9, 10, 12, 16, 24, 28, 30, 31, 32, 33, 34, 40, 48, 56, 64})
  {
    if (dimension <= capped)
    {
      result.push_back(dimension);
    }
  }
  if (result.empty() || result.back() != capped)
  {
    result.push_back(capped);
  }
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

template <typename Real> struct RunRow
{
    std::string method;
    int krylov_dimension = 0;
    int projected_dimension = 0;
    int matvecs = 0;
    Real error_scaled = Real{};
    Real tail_coefficient = Real{};
    Real endpoint_defect_scaled = Real{};
    Real defect_integral_scaled = Real{};
    Real orthogonality_offdiag = Real{};
    Real reorthogonalization_ratio = Real{};
    int reorthogonalization_passes = 0;
};

template <typename Scalar, typename ReferenceScalar>
[[nodiscard]] RunRow<RealOf<Scalar>> run_lanczos_probe(DenseRealMatrix const& matrix,
                                                       DenseHostVector<Scalar> const& initial,
                                                       DenseHostVector<ReferenceScalar> const& reference, Scalar time,
                                                       int krylov_dimension, LanczosProbeVariant variant)
{
  using Real = RealOf<Scalar>;
  using ReferenceReal = RealOf<ReferenceScalar>;

  Real const initial_norm = vector_norm(initial);
  DenseHostVector<Scalar> current = initial;
  host_scale(current, Scalar{Real{1} / initial_norm, Real{}});
  DenseHostVector<Scalar> previous{{std::vector<Scalar>(initial.values.size(), Scalar{})}};

  std::vector<DenseHostVector<Scalar>> basis;
  std::vector<Real> diagonal;
  std::vector<Real> offdiagonal;
  basis.reserve(static_cast<std::size_t>(krylov_dimension));
  diagonal.reserve(static_cast<std::size_t>(krylov_dimension));
  offdiagonal.reserve(static_cast<std::size_t>(krylov_dimension));

  Real residual_norm{};
  ProbeReorthogonalizationStats<Real> reorthogonalization_stats;
  int matvecs = 0;
  for (int step = 0; step < krylov_dimension; ++step)
  {
    basis.push_back(current);
    auto residual = dense_matvec(matrix, current);
    ++matvecs;

    if (variant == LanczosProbeVariant::FullReorthogonalized && step > 0)
    {
      host_axpy(residual, Scalar{-offdiagonal.back(), Real{}}, previous);
    }

    Real const alpha = static_cast<Real>(std::real(host_inner_product(current, residual)));
    diagonal.push_back(alpha);
    host_axpy(residual, Scalar{-alpha, Real{}}, current);
    if (variant == LanczosProbeVariant::LegacyThreeTerm && step > 0)
    {
      host_axpy(residual, Scalar{-offdiagonal.back(), Real{}}, previous);
    }

    residual_norm = variant == LanczosProbeVariant::FullReorthogonalized
                        ? orthogonalize_probe_residual(basis, residual, &reorthogonalization_stats)
                        : vector_norm(residual);
    offdiagonal.push_back(residual_norm);
    if (residual_norm == Real{})
    {
      break;
    }

    previous = current;
    current = std::move(residual);
    host_scale(current, Scalar{Real{1} / residual_norm, Real{}});
  }

  std::size_t const projected_size = diagonal.size();
  uni20::krylov::Matrix<Real> projected(projected_size, projected_size);
  for (std::size_t i = 0; i < projected_size; ++i)
  {
    projected[i, i] = diagonal[i];
    if (i + 1 < projected_size)
    {
      projected[i, i + 1] = offdiagonal[i];
      projected[i + 1, i] = offdiagonal[i];
    }
  }

  uni20::krylov::Matrix<Scalar> exponential(projected_size, projected_size);
  uni20::linalg::matrix_exponential(exponential, projected, time);
  DenseHostVector<Scalar> action{{std::vector<Scalar>(initial.values.size(), Scalar{})}};
  std::vector<Scalar> coefficients(projected_size);
  for (std::size_t row = 0; row < projected_size; ++row)
  {
    coefficients[row] = Scalar{initial_norm, Real{}} * exponential[row, 0];
    host_axpy(action, coefficients[row], basis[row]);
  }

  Real const endpoint_defect_estimate =
      coefficients.empty() ? Real{} : residual_norm * static_cast<Real>(std::abs(coefficients.back()));
  Real const tail_coefficient =
      coefficients.empty() ? Real{} : static_cast<Real>(std::abs(coefficients.back())) / initial_norm;
  Real const defect_integral =
      uni20::krylov::detail::hermitian_projected_defect_integral_estimate(projected, initial_norm, residual_norm, time);

  return RunRow<Real>{
      .method = variant == LanczosProbeVariant::FullReorthogonalized ? "full reorth" : "three-term",
      .krylov_dimension = krylov_dimension,
      .projected_dimension = static_cast<int>(projected_size),
      .matvecs = matvecs,
      .error_scaled = static_cast<Real>(difference_norm(action, reference) / static_cast<ReferenceReal>(initial_norm)),
      .tail_coefficient = tail_coefficient,
      .endpoint_defect_scaled = endpoint_defect_estimate / initial_norm,
      .defect_integral_scaled = defect_integral / initial_norm,
      .orthogonality_offdiag = basis_max_offdiag(basis),
      .reorthogonalization_ratio = reorthogonalization_stats.max_correction_ratio,
      .reorthogonalization_passes = reorthogonalization_stats.max_passes};
}

template <typename Scalar, typename ReferenceScalar>
[[nodiscard]] RunRow<RealOf<Scalar>>
run_full_reorthogonalized(DenseRealMatrix const& matrix, DenseHostVector<Scalar> const& initial,
                          DenseHostVector<ReferenceScalar> const& reference, Scalar time, int krylov_dimension)
{
  return run_lanczos_probe(matrix, initial, reference, time, krylov_dimension,
                           LanczosProbeVariant::FullReorthogonalized);
}

template <typename Scalar, typename ReferenceScalar>
[[nodiscard]] RunRow<RealOf<Scalar>>
run_legacy_three_term(DenseRealMatrix const& matrix, DenseHostVector<Scalar> const& initial,
                      DenseHostVector<ReferenceScalar> const& reference, Scalar time, int krylov_dimension)
{
  return run_lanczos_probe(matrix, initial, reference, time, krylov_dimension, LanczosProbeVariant::LegacyThreeTerm);
}

template <typename Real> [[nodiscard]] Real reference_taylor_tolerance()
{
  return Real{1000} * uni20::numeric_limits<Real>::epsilon();
}

template <typename Real>
[[nodiscard]] DenseHostVector<uni20::complex<Real>>
reference_taylor_action(DenseRealMatrix const& matrix, DenseHostVector<uni20::complex<Real>> const& initial,
                        uni20::complex<Real> time, Real operator_norm)
{
  using Scalar = uni20::complex<Real>;

  DenseHostVectorOps<Scalar> ops(matrix.rows, complex_matrix<Scalar>(matrix));
  TaylorExponentialParams<Real> params;
  params.tolerance = reference_taylor_tolerance<Real>();
  params.step_norm_limit = 0.5;
  params.max_taylor_degree = 200;
  params.max_scaling_steps = 20000;
  params.diagnostics = KrylovDiagnosticsLevel::None;

  auto result = uni20::krylov::taylor_exponential_action<Scalar>(ops, initial, time, operator_norm, params);
  if (!result.converged)
  {
    throw std::runtime_error("Taylor reference did not converge");
  }
  return std::move(result.action);
}

template <typename Real> struct DemoResult
{
    std::string matrix_path;
    std::string vector_path;
    std::size_t dimension = 0;
    Real tolerance = Real{};
    Real non_reorthogonalized_floor = Real{};
    uni20::complex<Real> time{};
    Real operator_norm_bound = Real{};
    Real initial_norm = Real{};
    Real reference_norm = Real{};
    Real reference_tolerance = Real{};
    bool all_dimensions = false;
    std::vector<RunRow<Real>> rows;
    bool old_indicator_false_pass = false;
    bool endpoint_defect_false_pass = false;
    bool defect_integral_false_pass = false;
    bool legacy_lost_orthogonality = false;
    int first_full_reorth_error_pass_m = 0;
    int first_full_reorth_tail_pass_m = 0;
    int first_full_reorth_defect_pass_m = 0;
    int first_full_reorth_integral_pass_m = 0;
};

template <typename Real> [[nodiscard]] DemoResult<Real> run_demo(Options const& options)
{
  using Scalar = uni20::complex<Real>;

  DenseRealMatrix const matrix = read_real_coordinate_matrix_market(options.matrix_path);
  std::vector<double> const initial_values = read_real_vector_matrix_market(options.vector_path);
  if (initial_values.size() != matrix.rows)
  {
    throw std::invalid_argument("Matrix Market vector dimension does not match matrix dimension");
  }

  double const operator_norm_bound = matrix_one_norm_bound(matrix);
  if (operator_norm_bound <= 0.0)
  {
    throw std::invalid_argument("Matrix Market operator norm bound is zero");
  }

  DenseHostVector<Scalar> const initial = complex_vector<Scalar>(initial_values);
  DenseHostVector<uni20::complex<ReferenceReal>> const reference_initial =
      complex_vector<uni20::complex<ReferenceReal>>(initial_values);
  Scalar const time{static_cast<Real>(options.time_real), static_cast<Real>(options.time_imag)};
  uni20::complex<ReferenceReal> const reference_time{static_cast<ReferenceReal>(options.time_real),
                                                     static_cast<ReferenceReal>(options.time_imag)};
  auto const reference = reference_taylor_action(matrix, reference_initial, reference_time,
                                                 static_cast<ReferenceReal>(operator_norm_bound));

  DemoResult<Real> result{.matrix_path = options.matrix_path,
                          .vector_path = options.vector_path,
                          .dimension = matrix.rows,
                          .tolerance = static_cast<Real>(options.tolerance),
                          .non_reorthogonalized_floor = std::sqrt(uni20::numeric_limits<Real>::epsilon()),
                          .time = time,
                          .operator_norm_bound = static_cast<Real>(operator_norm_bound),
                          .initial_norm = vector_norm(initial),
                          .reference_norm = static_cast<Real>(vector_norm(reference)),
                          .reference_tolerance = static_cast<Real>(reference_taylor_tolerance<ReferenceReal>()),
                          .all_dimensions = options.all_dimensions,
                          .rows = {}};

  for (int const dimension : sweep_dimensions(options.max_dimension, matrix.rows, options.all_dimensions))
  {
    result.rows.push_back(run_full_reorthogonalized(matrix, initial, reference, time, dimension));
    result.rows.push_back(run_legacy_three_term(matrix, initial, reference, time, dimension));
  }

  for (auto const& row : result.rows)
  {
    bool const old_indicator_passes = row.tail_coefficient <= result.tolerance;
    bool const endpoint_defect_passes = row.endpoint_defect_scaled <= result.tolerance;
    bool const defect_integral_passes = row.defect_integral_scaled <= result.tolerance;
    bool const error_passes = row.error_scaled <= result.tolerance;
    result.old_indicator_false_pass = result.old_indicator_false_pass || (old_indicator_passes && !error_passes);
    result.endpoint_defect_false_pass = result.endpoint_defect_false_pass || (endpoint_defect_passes && !error_passes);
    result.defect_integral_false_pass = result.defect_integral_false_pass || (defect_integral_passes && !error_passes);
    if (row.method == "full reorth")
    {
      if (error_passes && result.first_full_reorth_error_pass_m == 0)
      {
        result.first_full_reorth_error_pass_m = row.krylov_dimension;
      }
      if (old_indicator_passes && result.first_full_reorth_tail_pass_m == 0)
      {
        result.first_full_reorth_tail_pass_m = row.krylov_dimension;
      }
      if (endpoint_defect_passes && result.first_full_reorth_defect_pass_m == 0)
      {
        result.first_full_reorth_defect_pass_m = row.krylov_dimension;
      }
      if (defect_integral_passes && result.first_full_reorth_integral_pass_m == 0)
      {
        result.first_full_reorth_integral_pass_m = row.krylov_dimension;
      }
    }
    if (row.method == "three-term")
    {
      result.legacy_lost_orthogonality = result.legacy_lost_orthogonality || row.orthogonality_offdiag > Real{0.1};
    }
  }

  return result;
}

template <typename Real> [[nodiscard]] bool demo_check_passes(DemoResult<Real> const& result)
{
  return result.tolerance < result.non_reorthogonalized_floor && result.old_indicator_false_pass;
}

[[nodiscard]] presentation::numeric_format_options full_precision_numeric_format()
{
  presentation::numeric_format_options numeric;
  numeric.notation = presentation::real_notation::scientific;
  numeric.float32_precision = uni20::numeric_limits<float>::max_digits10;
  numeric.float64_precision = uni20::numeric_limits<double>::max_digits10;
  numeric.long_double_precision = uni20::numeric_limits<long double>::max_digits10;
#if UNI20_HAS_FLOAT128
  numeric.float128_precision = uni20::numeric_limits<uni20::float128>::max_digits10;
#endif
  return numeric;
}

template <typename Real> [[nodiscard]] std::string format_real(Real value)
{
  return presentation::format_real(value, full_precision_numeric_format());
}

template <typename Real> [[nodiscard]] std::string format_complex(uni20::complex<Real> value)
{
  return presentation::format_complex(value, full_precision_numeric_format());
}

template <typename Real> [[nodiscard]] std::string_view scalar_name()
{
  if constexpr (std::is_same_v<Real, float>)
  {
    return "complex<float>";
  }
  else if constexpr (std::is_same_v<Real, double>)
  {
    return "complex<double>";
  }
#if UNI20_HAS_FLOAT128
  else if constexpr (std::is_same_v<Real, uni20::float128>)
  {
    return "complex<float128>";
  }
#endif
  else
  {
    return "complex<unknown>";
  }
}

[[nodiscard]] std::string format_first_dimension(int dimension)
{
  return dimension == 0 ? "not observed" : fmt::format("{}", dimension);
}

template <typename Real> void print_demo(DemoResult<Real> const& result)
{
  presentation::report_builder report("Krylov exponential Matrix Market probe");
  report
      .status(demo_check_passes(result) ? presentation::semantic_glyph::success : presentation::semantic_glyph::warning,
              demo_check_passes(result) ? "Matrix Market pathology exposed" : "inspect diagnostics")
      .field("matrix", result.matrix_path)
      .field("vector", result.vector_path)
      .field("scalar", scalar_name<Real>())
      .field("dimension", fmt::format("{}", result.dimension))
      .field("requested rel tol", format_real(result.tolerance))
      .field("non-reorth floor", format_real(result.non_reorthogonalized_floor))
      .field("time", format_complex(result.time))
      .field("||A||_1 bound", format_real(result.operator_norm_bound))
      .field("||v||", format_real(result.initial_norm))
      .field("||reference||", format_real(result.reference_norm))
      .field("reference scalar", scalar_name<ReferenceReal>())
      .field("reference tol", format_real(result.reference_tolerance))
      .field("dimension sweep", result.all_dimensions ? "all" : "sampled");

  auto& summary = report.table("Diagnostic Checks");
  summary.column("condition").column("observed");
  summary.row("old tail coeff <= tol while Taylor error > tol", result.old_indicator_false_pass ? "yes" : "no");
  summary.row("defect <= tol while Taylor error > tol", result.endpoint_defect_false_pass ? "yes" : "no");
  summary.row("defect integral <= tol while Taylor error > tol", result.defect_integral_false_pass ? "yes" : "no");
  summary.row("legacy three-term basis loses orthogonality", result.legacy_lost_orthogonality ? "yes" : "no");

  auto& first_pass = report.table("First Full-Reorth Pass");
  first_pass.column("criterion").column("requested m");
  first_pass.row("Taylor error <= tol", format_first_dimension(result.first_full_reorth_error_pass_m));
  first_pass.row("tail coeff <= tol", format_first_dimension(result.first_full_reorth_tail_pass_m));
  first_pass.row("defect <= tol", format_first_dimension(result.first_full_reorth_defect_pass_m));
  first_pass.row("defect integral <= tol", format_first_dimension(result.first_full_reorth_integral_pass_m));

  auto& table = report.table("Lanczos Dimension Sweep");
  table.column("method")
      .column("requested m")
      .column("projected m")
      .column("matvecs")
      .column("err/||v||")
      .column("tail coeff")
      .column("defect/||v||")
      .column("int defect/||v||")
      .column("tail<=tol")
      .column("defect<=tol")
      .column("int<=tol")
      .column("err<=tol")
      .column("orth offdiag")
      .column("reorth ratio")
      .column("passes");

  for (auto const& row : result.rows)
  {
    table.row(row.method, fmt::format("{}", row.krylov_dimension), fmt::format("{}", row.projected_dimension),
              fmt::format("{}", row.matvecs), format_real(row.error_scaled), format_real(row.tail_coefficient),
              format_real(row.endpoint_defect_scaled), format_real(row.defect_integral_scaled),
              row.tail_coefficient <= result.tolerance ? "yes" : "no",
              row.endpoint_defect_scaled <= result.tolerance ? "yes" : "no",
              row.defect_integral_scaled <= result.tolerance ? "yes" : "no",
              row.error_scaled <= result.tolerance ? "yes" : "no", format_real(row.orthogonality_offdiag),
              format_real(row.reorthogonalization_ratio), fmt::format("{}", row.reorthogonalization_passes));
  }

  auto policy = presentation::terminal_policy(stdout);
  policy.glyphs = presentation::glyph_set::unicode;
  policy.charset = presentation::text_charset::utf8;
  policy.width = presentation::width_mode::display_cells;
  presentation::styled_text text;
  text.append(presentation::render_report(report, policy));
  fmt::print("{}", presentation::render_terminal(text, policy, stdout));
}

template <typename Real> int run(Options const& options)
{
  auto const result = run_demo<Real>(options);
  print_demo(result);
  if (options.check && !demo_check_passes(result))
  {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char** argv)
{
  try
  {
    Options const options = parse_options(argc, argv);
    switch (options.precision)
    {
      case ScalarPrecision::Float32:
        return run<float>(options);
      case ScalarPrecision::Float64:
        return run<double>(options);
#if UNI20_HAS_FLOAT128
      case ScalarPrecision::Float128:
        return run<uni20::float128>(options);
#endif
    }
  }
  catch (std::exception const& error)
  {
    fmt::print(stderr, "error: {}\n", error.what());
    return EXIT_FAILURE;
  }
  return EXIT_FAILURE;
}
