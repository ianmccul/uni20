#include <uni20/common/presentation.hpp>
#include <uni20/config.hpp>
#include <uni20/core/numeric_limits.hpp>
#include <uni20/core/types.hpp>
#include <uni20/krylov/dense_host_vector.hpp>
#include <uni20/krylov/dense_linalg.hpp>
#include <uni20/krylov/symmetric_lanczos.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <fmt/core.h>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{

namespace presentation = uni20::presentation;

#ifndef UNI20_KRYLOV_EXAMPLE_MATRIX_MARKET_DIR
#define UNI20_KRYLOV_EXAMPLE_MATRIX_MARKET_DIR "."
#endif

using uni20::krylov::DenseHostVector;
using uni20::krylov::KrylovDiagnosticsLevel;
using uni20::krylov::SpectrumPart;
using uni20::krylov::SymmetricEigenParams;
using uni20::krylov::SymmetricSpectralTransform;
using uni20::krylov::SymmetricSpectralTransformOptions;

enum class ExampleMode
{
  Regular,
  ShiftInvert
};

enum class ScalarPrecision
{
  Float32,
  Float64
#if UNI20_HAS_FLOAT128
  ,
  Float128
#endif
};

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

template <typename Scalar> class SparseHostVectorOps {
  public:
    explicit SparseHostVectorOps(SparseMatrixMarketMatrix matrix) : matrix_(std::move(matrix))
    {
      if (matrix_.rows != matrix_.cols)
      {
        throw std::invalid_argument("Krylov example requires a square matrix");
      }
    }

    [[nodiscard]] std::size_t problem_dimension() const noexcept { return matrix_.rows; }

    [[nodiscard]] std::size_t vector_dimension(DenseHostVector<Scalar> const& x) const noexcept
    {
      return x.values.size();
    }

    [[nodiscard]] int matvec_count() const noexcept { return matvec_count_; }

    [[nodiscard]] DenseHostVector<Scalar> allocate_like(DenseHostVector<Scalar> const& x)
    {
      return DenseHostVector<Scalar>{std::vector<Scalar>(x.values.size())};
    }

    void copy(DenseHostVector<Scalar>& dst, DenseHostVector<Scalar> const& src)
    {
      require_same_size(dst, src);
      dst.values = src.values;
    }

    void axpy(DenseHostVector<Scalar>& y, Scalar alpha, DenseHostVector<Scalar> const& x)
    {
      require_same_size(y, x);
      for (std::size_t i = 0; i < y.values.size(); ++i)
      {
        y.values[i] += alpha * x.values[i];
      }
    }

    void scal(DenseHostVector<Scalar>& x, Scalar alpha)
    {
      for (Scalar& value : x.values)
      {
        value *= alpha;
      }
    }

    void set_zero(DenseHostVector<Scalar>& x)
    {
      for (Scalar& value : x.values)
      {
        value = Scalar{};
      }
    }

    [[nodiscard]] Scalar inner_product(DenseHostVector<Scalar> const& x, DenseHostVector<Scalar> const& y)
    {
      require_same_size(x, y);
      Scalar result{};
      for (std::size_t i = 0; i < x.values.size(); ++i)
      {
        result += x.values[i] * y.values[i];
      }
      return result;
    }

    void matvec(DenseHostVector<Scalar>& y, DenseHostVector<Scalar> const& x)
    {
      if (x.values.size() != matrix_.cols || y.values.size() != matrix_.rows)
      {
        throw std::invalid_argument("Krylov example matvec vector has the wrong size");
      }
      ++matvec_count_;
      this->set_zero(y);
      for (auto const& entry : matrix_.entries)
      {
        y.values[entry.row] += static_cast<Scalar>(entry.value) * x.values[entry.col];
      }
    }

  private:
    static void require_same_size(DenseHostVector<Scalar> const& lhs, DenseHostVector<Scalar> const& rhs)
    {
      if (lhs.values.size() != rhs.values.size())
      {
        throw std::invalid_argument("Krylov example vectors have different sizes");
      }
    }

    SparseMatrixMarketMatrix matrix_;
    int matvec_count_ = 0;
};

template <typename Scalar> class DenseShiftInvertOps {
  public:
    DenseShiftInvertOps(SparseMatrixMarketMatrix const& matrix, Scalar sigma)
        : coefficient_(matrix.rows, matrix.cols), sigma_(sigma)
    {
      if (matrix.rows != matrix.cols)
      {
        throw std::invalid_argument("Krylov shift-invert example requires a square matrix");
      }
      for (auto const& entry : matrix.entries)
      {
        coefficient_(entry.row, entry.col) += static_cast<Scalar>(entry.value);
      }
      for (std::size_t i = 0; i < matrix.rows; ++i)
      {
        coefficient_(i, i) -= sigma_;
      }
    }

    [[nodiscard]] std::size_t problem_dimension() const noexcept { return coefficient_.rows(); }

    [[nodiscard]] std::size_t vector_dimension(DenseHostVector<Scalar> const& x) const noexcept
    {
      return x.values.size();
    }

    [[nodiscard]] int matvec_count() const noexcept { return solve_count_; }

    [[nodiscard]] DenseHostVector<Scalar> allocate_like(DenseHostVector<Scalar> const& x)
    {
      return DenseHostVector<Scalar>{std::vector<Scalar>(x.values.size())};
    }

    void copy(DenseHostVector<Scalar>& dst, DenseHostVector<Scalar> const& src)
    {
      require_same_size(dst, src);
      dst.values = src.values;
    }

    void axpy(DenseHostVector<Scalar>& y, Scalar alpha, DenseHostVector<Scalar> const& x)
    {
      require_same_size(y, x);
      for (std::size_t i = 0; i < y.values.size(); ++i)
      {
        y.values[i] += alpha * x.values[i];
      }
    }

    void scal(DenseHostVector<Scalar>& x, Scalar alpha)
    {
      for (Scalar& value : x.values)
      {
        value *= alpha;
      }
    }

    void set_zero(DenseHostVector<Scalar>& x)
    {
      for (Scalar& value : x.values)
      {
        value = Scalar{};
      }
    }

    [[nodiscard]] Scalar inner_product(DenseHostVector<Scalar> const& x, DenseHostVector<Scalar> const& y)
    {
      require_same_size(x, y);
      Scalar result{};
      for (std::size_t i = 0; i < x.values.size(); ++i)
      {
        result += x.values[i] * y.values[i];
      }
      return result;
    }

    void matvec(DenseHostVector<Scalar>& y, DenseHostVector<Scalar> const& x)
    {
      if (x.values.size() != coefficient_.cols() || y.values.size() != coefficient_.rows())
      {
        throw std::invalid_argument("Krylov shift-invert example vector has the wrong size");
      }
      ++solve_count_;
      uni20::krylov::Matrix<Scalar> rhs(x.values.size(), 1);
      for (std::size_t i = 0; i < x.values.size(); ++i)
      {
        rhs(i, 0) = x.values[i];
      }
      auto solution = uni20::krylov::solve_linear_system(coefficient_, std::move(rhs));
      for (std::size_t i = 0; i < y.values.size(); ++i)
      {
        y.values[i] = solution(i, 0);
      }
    }

  private:
    static void require_same_size(DenseHostVector<Scalar> const& lhs, DenseHostVector<Scalar> const& rhs)
    {
      if (lhs.values.size() != rhs.values.size())
      {
        throw std::invalid_argument("Krylov shift-invert example vectors have different sizes");
      }
    }

    uni20::krylov::Matrix<Scalar> coefficient_;
    Scalar sigma_ = Scalar{};
    int solve_count_ = 0;
};

struct ExampleOptions
{
    std::string matrix_path = std::string(UNI20_KRYLOV_EXAMPLE_MATRIX_MARKET_DIR) + "/path_laplacian_30.mtx";
    ScalarPrecision precision = ScalarPrecision::Float64;
    ExampleMode mode = ExampleMode::Regular;
    SpectrumPart spectrum = SpectrumPart::LargestAlgebraic;
    bool spectrum_explicit = false;
    int eigenvalue_count = 3;
    int retained_ritz_count = 0;
    int krylov_dimension = 0;
    int max_iterations = 300;
    double sigma = 0.0;
    double tolerance = 1.0e-10;
    bool eigenvectors = true;
    KrylovDiagnosticsLevel diagnostics = KrylovDiagnosticsLevel::Summary;
};

[[nodiscard]] std::string mode_name(ExampleMode mode)
{
  switch (mode)
  {
    case ExampleMode::Regular:
      return "regular";
    case ExampleMode::ShiftInvert:
      return "shift-invert";
  }
  return "unknown";
}

[[nodiscard]] std::string precision_name(ScalarPrecision precision)
{
  switch (precision)
  {
    case ScalarPrecision::Float32:
      return "float";
    case ScalarPrecision::Float64:
      return "double";
#if UNI20_HAS_FLOAT128
    case ScalarPrecision::Float128:
      return "float128";
#endif
  }
  return "unknown";
}

[[nodiscard]] ScalarPrecision parse_precision(std::string_view value)
{
  if (value == "float" || value == "float32" || value == "single")
  {
    return ScalarPrecision::Float32;
  }
  if (value == "double" || value == "float64")
  {
    return ScalarPrecision::Float64;
  }
  if (value == "float128" || value == "quad" || value == "binary128")
  {
#if UNI20_HAS_FLOAT128
    return ScalarPrecision::Float128;
#else
    throw std::invalid_argument("float128 precision requires a UNI20_HAS_FLOAT128 build");
#endif
  }
  throw std::invalid_argument("unsupported --precision value; use float, double, or float128 when available");
}

[[nodiscard]] ExampleMode parse_mode(std::string_view value)
{
  if (value == "regular")
  {
    return ExampleMode::Regular;
  }
  if (value == "shift-invert")
  {
    return ExampleMode::ShiftInvert;
  }
  throw std::invalid_argument("unsupported --mode value; use regular or shift-invert");
}

[[nodiscard]] std::string spectrum_name(SpectrumPart spectrum)
{
  switch (spectrum)
  {
    case SpectrumPart::LargestMagnitude:
      return "LM";
    case SpectrumPart::SmallestMagnitude:
      return "SM";
    case SpectrumPart::LargestAlgebraic:
      return "LA";
    case SpectrumPart::SmallestAlgebraic:
      return "SA";
    case SpectrumPart::BothEnds:
      return "BE";
    case SpectrumPart::LargestReal:
      return "LR";
    case SpectrumPart::SmallestReal:
      return "SR";
    case SpectrumPart::LargestImaginary:
      return "LI";
    case SpectrumPart::SmallestImaginary:
      return "SI";
  }
  return "unknown";
}

[[nodiscard]] SpectrumPart parse_spectrum(std::string_view value)
{
  if (value == "LM")
  {
    return SpectrumPart::LargestMagnitude;
  }
  if (value == "SM")
  {
    return SpectrumPart::SmallestMagnitude;
  }
  if (value == "LA")
  {
    return SpectrumPart::LargestAlgebraic;
  }
  if (value == "SA")
  {
    return SpectrumPart::SmallestAlgebraic;
  }
  if (value == "BE")
  {
    return SpectrumPart::BothEnds;
  }
  throw std::invalid_argument("unsupported --which value; use LM, SM, LA, SA, or BE");
}

[[nodiscard]] bool parse_bool(std::string_view value)
{
  if (value == "1" || value == "true" || value == "yes" || value == "on")
  {
    return true;
  }
  if (value == "0" || value == "false" || value == "no" || value == "off")
  {
    return false;
  }
  throw std::invalid_argument("expected boolean value");
}

[[nodiscard]] KrylovDiagnosticsLevel parse_diagnostics(std::string_view value)
{
  if (value == "none")
  {
    return KrylovDiagnosticsLevel::None;
  }
  if (value == "summary")
  {
    return KrylovDiagnosticsLevel::Summary;
  }
  if (value == "full")
  {
    return KrylovDiagnosticsLevel::Full;
  }
  throw std::invalid_argument("unsupported --diagnostics value; use none, summary, or full");
}

[[nodiscard]] std::string_view option_value(std::string_view argument, std::string_view prefix)
{
  if (!argument.starts_with(prefix))
  {
    throw std::invalid_argument("internal option parser error");
  }
  return argument.substr(prefix.size());
}

[[nodiscard]] ExampleOptions parse_options(int argc, char** argv)
{
  ExampleOptions options;
  for (int i = 1; i < argc; ++i)
  {
    std::string_view const argument(argv[i]);
    if (argument == "--help" || argument == "-h")
    {
      throw std::runtime_error("help");
    }
    if (argument.starts_with("--matrix="))
    {
      options.matrix_path = std::string(option_value(argument, "--matrix="));
    }
    else if (argument.starts_with("--precision="))
    {
      options.precision = parse_precision(option_value(argument, "--precision="));
    }
    else if (argument.starts_with("--which="))
    {
      options.spectrum = parse_spectrum(option_value(argument, "--which="));
      options.spectrum_explicit = true;
    }
    else if (argument.starts_with("--mode="))
    {
      options.mode = parse_mode(option_value(argument, "--mode="));
    }
    else if (argument.starts_with("--nev="))
    {
      options.eigenvalue_count = std::stoi(std::string(option_value(argument, "--nev=")));
    }
    else if (argument.starts_with("--nkeep="))
    {
      options.retained_ritz_count = std::stoi(std::string(option_value(argument, "--nkeep=")));
    }
    else if (argument.starts_with("--ncv="))
    {
      options.krylov_dimension = std::stoi(std::string(option_value(argument, "--ncv=")));
    }
    else if (argument.starts_with("--max-iters="))
    {
      options.max_iterations = std::stoi(std::string(option_value(argument, "--max-iters=")));
    }
    else if (argument.starts_with("--sigma="))
    {
      options.sigma = std::stod(std::string(option_value(argument, "--sigma=")));
    }
    else if (argument.starts_with("--tol="))
    {
      options.tolerance = std::stod(std::string(option_value(argument, "--tol=")));
    }
    else if (argument.starts_with("--eigenvectors="))
    {
      options.eigenvectors = parse_bool(option_value(argument, "--eigenvectors="));
    }
    else if (argument.starts_with("--diagnostics="))
    {
      options.diagnostics = parse_diagnostics(option_value(argument, "--diagnostics="));
    }
    else if (argument.starts_with("--"))
    {
      throw std::invalid_argument(fmt::format("unknown option '{}'", argument));
    }
    else
    {
      options.matrix_path = std::string(argument);
    }
  }
  if (options.mode == ExampleMode::ShiftInvert && !options.spectrum_explicit)
  {
    options.spectrum = SpectrumPart::LargestMagnitude;
  }
  return options;
}

void print_usage(char const* program)
{
  fmt::print("Usage: {} [matrix.mtx] [options]\n\n", program);
  fmt::print("Options:\n");
  fmt::print("  --matrix=PATH          Matrix Market coordinate file\n");
  fmt::print("  --precision=float|double");
#if UNI20_HAS_FLOAT128
  fmt::print("|float128");
#endif
  fmt::print(", default double\n");
  fmt::print("  --mode=regular|shift-invert, default regular\n");
  fmt::print("  --which=LM|SM|LA|SA|BE Spectrum selector, default LA; shift-invert default LM\n");
  fmt::print("  --nev=N                Number of eigenvalues, default 3\n");
  fmt::print("  --nkeep=N              Internal retained Ritz count, default nev\n");
  fmt::print("  --ncv=N                Krylov subspace size, default solver policy\n");
  fmt::print("  --sigma=VALUE          Shift for shift-invert mode, default 0\n");
  fmt::print("  --tol=VALUE            Convergence tolerance, default 1e-10\n");
  fmt::print("  --max-iters=N          Restart cycle limit, default 300\n");
  fmt::print("  --eigenvectors=BOOL    Compute vectors/residuals, default true\n");
  fmt::print("  --diagnostics=none|summary|full, default summary\n\n");
  fmt::print("Default matrix: {}/path_laplacian_30.mtx\n", UNI20_KRYLOV_EXAMPLE_MATRIX_MARKET_DIR);
}

[[nodiscard]] SparseMatrixMarketMatrix read_numeric_coordinate_matrix_market(std::string const& path)
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
  if (symmetry != "symmetric")
  {
    throw std::invalid_argument("symmetric Lanczos example requires a symmetric Matrix Market matrix");
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
  size_stream >> rows >> cols >> stored_entries;

  SparseMatrixMarketMatrix matrix{.rows = rows, .cols = cols, .entries = {}};
  matrix.entries.reserve(symmetry == "symmetric" ? 2 * stored_entries : stored_entries);
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
    matrix.entries.push_back(SparseEntry{.row = row, .col = col, .value = value});
    if (symmetry == "symmetric" && row != col)
    {
      matrix.entries.push_back(SparseEntry{.row = col, .col = row, .value = value});
    }
  }

  return matrix;
}

template <typename Scalar> [[nodiscard]] DenseHostVector<Scalar> deterministic_initial_vector(std::size_t dimension)
{
  DenseHostVector<Scalar> vector{std::vector<Scalar>(dimension)};
  for (std::size_t i = 0; i < dimension; ++i)
  {
    vector.values[i] = static_cast<Scalar>((7 * i + 3) % 17 + 1);
  }
  return vector;
}

template <typename Scalar> [[nodiscard]] Scalar vector_norm(DenseHostVector<Scalar> const& vector)
{
  Scalar norm_squared{};
  for (Scalar const value : vector.values)
  {
    norm_squared += value * value;
  }
  return std::sqrt(norm_squared);
}

template <typename Scalar>
[[nodiscard]] Scalar relative_residual_norm(SparseMatrixMarketMatrix const& matrix,
                                            DenseHostVector<Scalar> const& eigenvector, Scalar eigenvalue)
{
  SparseHostVectorOps<Scalar> ops(matrix);
  auto residual = ops.allocate_like(eigenvector);
  ops.matvec(residual, eigenvector);
  ops.axpy(residual, -eigenvalue, eigenvector);
  return vector_norm(residual) / std::max(Scalar{1}, std::abs(eigenvalue));
}

template <typename Scalar>
[[nodiscard]] Scalar
max_original_relative_residual(SparseMatrixMarketMatrix const& matrix,
                               uni20::krylov::EigenResult<Scalar, DenseHostVector<Scalar>> const& result)
{
  Scalar max_residual{};
  std::size_t const count = std::min(result.eigenvalues.size(), result.eigenvectors.size());
  for (std::size_t i = 0; i < count; ++i)
  {
    max_residual =
        std::max(max_residual, relative_residual_norm(matrix, result.eigenvectors[i], result.eigenvalues[i]));
  }
  return max_residual;
}

[[nodiscard]] terminal::TerminalStyle style(std::string_view spec) { return terminal::TerminalStyle(spec); }

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

void append_key_value(presentation::styled_text& text, presentation::output_policy const& policy, std::string_view key,
                      std::string_view value)
{
  text.append("  ")
      .append(presentation::pad_right(key, 18, policy), style("LightGray"))
      .append(" ")
      .append(value)
      .append("\n");
}

template <typename Scalar>
void append_result_table(presentation::styled_text& text, presentation::output_policy const& policy,
                         uni20::krylov::EigenResult<Scalar, DenseHostVector<Scalar>> const& result,
                         SparseMatrixMarketMatrix const& matrix)
{
  auto const numeric = full_precision_numeric_format();
  int constexpr value_width = 26;

  text.append("\nEigenpairs\n", style("Cyan;Bold"));
  text.append("  ")
      .append(presentation::pad_right("#", 4, policy), style("LightGray"))
      .append(presentation::pad_left("lambda", value_width, policy), style("LightGray"))
      .append("  ")
      .append(presentation::pad_left("ritz bound", value_width, policy), style("LightGray"))
      .append("  ")
      .append(presentation::pad_left("||Av-lv||/scale", value_width, policy), style("LightGray"))
      .append("\n");

  for (std::size_t i = 0; i < result.eigenvalues.size(); ++i)
  {
    std::string residual = "-";
    if (i < result.eigenvectors.size())
    {
      residual = presentation::format_real(
          relative_residual_norm(matrix, result.eigenvectors[i], result.eigenvalues[i]), numeric);
    }
    std::string const bound =
        i < result.residual_bounds.size() ? presentation::format_real(result.residual_bounds[i], numeric) : "-";
    text.append("  ")
        .append(presentation::pad_right(fmt::format("{}", i), 4, policy))
        .append(presentation::pad_left(presentation::format_real(result.eigenvalues[i], numeric), value_width, policy))
        .append("  ")
        .append(presentation::pad_left(bound, value_width, policy))
        .append("  ")
        .append(presentation::pad_left(residual, value_width, policy))
        .append("\n");
  }
}

template <typename T> [[nodiscard]] std::string format_index_vector(std::vector<T> const& values)
{
  std::string result = "[";
  for (std::size_t i = 0; i < values.size(); ++i)
  {
    if (i != 0)
    {
      result += ", ";
    }
    result += fmt::format("{}", values[i]);
  }
  result += "]";
  return result;
}

[[nodiscard]] std::string format_bool_vector(std::vector<bool> const& values)
{
  std::string result = "[";
  for (std::size_t i = 0; i < values.size(); ++i)
  {
    if (i != 0)
    {
      result += ", ";
    }
    result += values[i] ? "true" : "false";
  }
  result += "]";
  return result;
}

template <typename Scalar>
[[nodiscard]] std::string format_real_vector(std::vector<Scalar> const& values,
                                             presentation::numeric_format_options const& numeric,
                                             std::size_t max_count = 4)
{
  std::string result = "[";
  std::size_t const count = std::min(values.size(), max_count);
  for (std::size_t i = 0; i < count; ++i)
  {
    if (i != 0)
    {
      result += ", ";
    }
    result += presentation::format_real(values[i], numeric);
  }
  if (values.size() > count)
  {
    if (count != 0)
    {
      result += ", ";
    }
    result += fmt::format("... {} more", values.size() - count);
  }
  result += "]";
  return result;
}

template <typename Scalar>
void append_restart_trace(presentation::styled_text& text,
                          std::vector<uni20::krylov::SymmetricRestartCycleDiagnostics<Scalar>> const& cycles,
                          presentation::numeric_format_options const& numeric)
{
  if (cycles.empty())
  {
    return;
  }

  auto retained_range = std::minmax_element(cycles.begin(), cycles.end(), [](auto const& lhs, auto const& rhs) {
    return lhs.retained_count < rhs.retained_count;
  });
  auto shift_range = std::minmax_element(
      cycles.begin(), cycles.end(), [](auto const& lhs, auto const& rhs) { return lhs.shift_count < rhs.shift_count; });
  std::size_t const protected_cycles = static_cast<std::size_t>(std::count_if(
      cycles.begin(), cycles.end(), [](auto const& cycle) { return cycle.protected_zero_bound_count != 0; }));
  std::size_t const enlarged_cycles = static_cast<std::size_t>(std::count_if(
      cycles.begin(), cycles.end(), [](auto const& cycle) { return cycle.enlarged_for_partial_convergence; }));

  text.append("\nRestart Trace\n", style("Cyan;Bold"));
  text.append(fmt::format("  retained range:      {}..{}\n", retained_range.first->retained_count,
                          retained_range.second->retained_count));
  text.append(
      fmt::format("  shift range:         {}..{}\n", shift_range.first->shift_count, shift_range.second->shift_count));
  text.append(fmt::format("  zero-bound protected cycles: {} / {}\n", protected_cycles, cycles.size()));
  text.append(fmt::format("  partial-convergence enlarged cycles: {} / {}\n", enlarged_cycles, cycles.size()));

  std::vector<std::size_t> shown;
  std::size_t constexpr head_count = 3;
  std::size_t constexpr tail_count = 5;
  std::size_t const first_count = std::min(cycles.size(), head_count);
  for (std::size_t i = 0; i < first_count; ++i)
  {
    shown.push_back(i);
  }
  std::size_t const tail_begin = cycles.size() > tail_count ? cycles.size() - tail_count : first_count;
  for (std::size_t i = tail_begin; i < cycles.size(); ++i)
  {
    if (std::ranges::find(shown, i) == shown.end())
    {
      shown.push_back(i);
    }
  }

  text.append(fmt::format("  cycles shown:        {} of {}\n", shown.size(), cycles.size()));
  for (std::size_t const cycle_index : shown)
  {
    auto const& cycle = cycles[cycle_index];
    text.append(fmt::format("  cycle {:>5}: conv={} kept={} shifts={} protected={} enlarged={} residual={}\n",
                            cycle.cycle, cycle.converged_count, cycle.retained_count, cycle.shift_count,
                            cycle.protected_zero_bound_count, cycle.enlarged_for_partial_convergence ? "yes" : "no",
                            presentation::format_real(cycle.residual_norm, numeric)));
    text.append("              wanted idx=")
        .append(format_index_vector(cycle.wanted_indices))
        .append(" converged=")
        .append(format_bool_vector(cycle.wanted_converged))
        .append("\n");
    text.append("              shift idx=").append(format_index_vector(cycle.shift_indices)).append("\n");
    text.append("              shifts=").append(format_real_vector(cycle.shifts, numeric)).append("\n");
  }
}

template <typename Scalar>
void append_diagnostics(presentation::styled_text& text,
                        uni20::krylov::EigenResult<Scalar, DenseHostVector<Scalar>> const& result)
{
  if (!result.diagnostics.has_value())
  {
    return;
  }

  auto const& diagnostics = *result.diagnostics;
  auto const numeric = full_precision_numeric_format();
  text.append("\nDiagnostics\n", style("Cyan;Bold"));
  text.append(fmt::format("  projected dimension: {}\n", diagnostics.final_projected_dimension));
  text.append(fmt::format("  operator count:      {}\n", diagnostics.op_count));
  text.append(fmt::format("  restart cycles:      {}\n", diagnostics.restart_count));
  text.append("  final Lanczos residual norm: ")
      .append(presentation::format_real(diagnostics.final_residual_norm, numeric))
      .append("\n");

  if (!diagnostics.restart_cycles.empty())
  {
    auto const& last = diagnostics.restart_cycles.back();
    text.append("  last restart:        ")
        .append(fmt::format("retained={} shifts={} protected={} enlarged={}\n", last.retained_count, last.shift_count,
                            last.protected_zero_bound_count, last.enlarged_for_partial_convergence ? "yes" : "no"));
    append_restart_trace(text, diagnostics.restart_cycles, numeric);
  }
}

template <typename Scalar> int run_example(ExampleOptions const& options)
{
  auto const matrix = read_numeric_coordinate_matrix_market(options.matrix_path);
  DenseHostVector<Scalar> initial = deterministic_initial_vector<Scalar>(matrix.rows);

  SymmetricEigenParams<Scalar> params;
  params.eigenvalue_count = options.eigenvalue_count;
  params.retained_ritz_count = options.retained_ritz_count;
  params.krylov_dimension = options.krylov_dimension;
  params.max_iterations = options.max_iterations;
  params.tolerance = static_cast<Scalar>(options.tolerance);
  params.spectrum = options.spectrum;
  params.compute_eigenvectors = options.eigenvectors;
  params.diagnostics = options.diagnostics;

  auto const start = std::chrono::steady_clock::now();
  uni20::krylov::EigenResult<Scalar, DenseHostVector<Scalar>> result;
  int observed_op_count = 0;
  if (options.mode == ExampleMode::ShiftInvert)
  {
    DenseShiftInvertOps<Scalar> ops(matrix, static_cast<Scalar>(options.sigma));
    SymmetricSpectralTransformOptions<Scalar> transform_options{.transform = SymmetricSpectralTransform::ShiftInvert,
                                                                .sigma = static_cast<Scalar>(options.sigma)};
    result = uni20::krylov::symmetric_lanczos_restarted_transformed<Scalar>(ops, initial, params, transform_options);
    observed_op_count = ops.matvec_count();
  }
  else
  {
    SparseHostVectorOps<Scalar> ops(matrix);
    result = uni20::krylov::symmetric_lanczos_restarted_standard<Scalar>(ops, initial, params);
    observed_op_count = ops.matvec_count();
  }
  auto const stop = std::chrono::steady_clock::now();
  double const elapsed_ms = std::chrono::duration<double, std::milli>(stop - start).count();
  Scalar max_original_residual{};
  if (options.eigenvectors)
  {
    max_original_residual = max_original_relative_residual(matrix, result);
  }
  Scalar const tolerance = params.tolerance;
  bool const original_residual_ok =
      !options.eigenvectors ||
      max_original_residual <= std::max(tolerance, Scalar{100} * uni20::numeric_limits<Scalar>::epsilon());
  int const display_status =
      (result.status == 0 && original_residual_ok) ? 0 : (result.status == 0 ? 3 : result.status);

  auto policy = presentation::terminal_policy(stdout);
  policy.glyphs = presentation::glyph_set::unicode;
  policy.charset = presentation::text_charset::utf8;
  policy.width = presentation::width_mode::display_cells;
  auto const numeric = full_precision_numeric_format();

  presentation::styled_text report;
  report.append("Native symmetric Lanczos Matrix Market example\n", style("Yellow;Bold"));
  report.append(display_status == 0 ? presentation::semantic_glyph::success : presentation::semantic_glyph::warning,
                display_status == 0 ? style("Green;Bold") : style("Yellow;Bold"));
  if (display_status == 0)
  {
    report.append(" converged\n");
  }
  else if (result.status == 0)
  {
    report.append(" transformed solve converged, original residual check failed\n");
  }
  else
  {
    report.append(" not fully converged\n");
  }

  append_key_value(report, policy, "matrix", options.matrix_path);
  append_key_value(report, policy, "precision", precision_name(options.precision));
  append_key_value(report, policy, "dimension", fmt::format("{} x {}", matrix.rows, matrix.cols));
  append_key_value(report, policy, "entries", fmt::format("{}", matrix.entries.size()));
  append_key_value(report, policy, "mode", mode_name(options.mode));
  if (options.mode == ExampleMode::ShiftInvert)
  {
    append_key_value(report, policy, "sigma", presentation::format_real(static_cast<Scalar>(options.sigma), numeric));
  }
  append_key_value(report, policy, "which", spectrum_name(options.spectrum));
  int const effective_ncv = uni20::krylov::effective_symmetric_krylov_dimension(params, matrix.rows);
  append_key_value(report, policy, "nev / ncv", fmt::format("{} / {}", params.eigenvalue_count, effective_ncv));
  if (params.retained_ritz_count != 0)
  {
    append_key_value(report, policy, "nkeep", fmt::format("{}", params.retained_ritz_count));
  }
  append_key_value(report, policy, "tol", presentation::format_real(params.tolerance, numeric));
  append_key_value(report, policy, "status", fmt::format("{}", result.status));
  append_key_value(report, policy, "converged", fmt::format("{}", result.converged_count));
  if (options.eigenvectors)
  {
    append_key_value(report, policy, "max residual", presentation::format_real(max_original_residual, numeric));
  }
  append_key_value(report, policy, "matvecs",
                   fmt::format("{} reported, {} observed", result.matvec_count, observed_op_count));
  append_key_value(report, policy, "iterations", fmt::format("{}", result.iteration_count));
  append_key_value(report, policy, "elapsed", fmt::format("{:.3f} ms", elapsed_ms));

  append_result_table(report, policy, result, matrix);
  append_diagnostics(report, result);

  fmt::print("{}", presentation::render_terminal(report, policy, stdout));
  return display_status == 0 ? 0 : 2;
}

} // namespace

int main(int argc, char** argv)
{
  try
  {
    ExampleOptions const options = parse_options(argc, argv);
    switch (options.precision)
    {
      case ScalarPrecision::Float32:
        return run_example<float>(options);
      case ScalarPrecision::Float64:
        return run_example<double>(options);
#if UNI20_HAS_FLOAT128
      case ScalarPrecision::Float128:
        return run_example<uni20::float128>(options);
#endif
    }
    throw std::invalid_argument("unsupported scalar precision");
  }
  catch (std::runtime_error const& error)
  {
    if (std::string_view(error.what()) == "help")
    {
      print_usage(argv[0]);
      return 0;
    }
    fmt::print(stderr, "error: {}\n", error.what());
    return 1;
  }
  catch (std::exception const& error)
  {
    fmt::print(stderr, "error: {}\n", error.what());
    fmt::print(stderr, "Run with --help for usage.\n");
    return 1;
  }
}
