#include <uni20/common/presentation.hpp>
#include <uni20/config.hpp>
#include <uni20/core/numeric_limits.hpp>
#include <uni20/core/types.hpp>
#include <uni20/krylov/dense_host_vector.hpp>
#include <uni20/krylov/nonsymmetric_arnoldi.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
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
using uni20::krylov::NonsymmetricEigenParams;
using uni20::krylov::NonsymmetricEigenResult;
using uni20::krylov::NonsymmetricStatus;
using uni20::krylov::RealNonsymmetricPolicy;
using uni20::krylov::RitzReality;
using uni20::krylov::SpectrumPart;

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
        throw std::invalid_argument("nonsymmetric Krylov example requires a square matrix");
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

    [[nodiscard]] Scalar norm(DenseHostVector<Scalar> const& x)
    {
      Scalar norm_squared{};
      for (Scalar const value : x.values)
      {
        norm_squared += value * value;
      }
      return std::sqrt(norm_squared);
    }

    void matvec(DenseHostVector<Scalar>& y, DenseHostVector<Scalar> const& x)
    {
      if (x.values.size() != matrix_.cols || y.values.size() != matrix_.rows)
      {
        throw std::invalid_argument("nonsymmetric Krylov example matvec vector has the wrong size");
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
        throw std::invalid_argument("nonsymmetric Krylov example vectors have different sizes");
      }
    }

    SparseMatrixMarketMatrix matrix_;
    int matvec_count_ = 0;
};

struct ExampleOptions
{
    std::string matrix_path = std::string(UNI20_KRYLOV_EXAMPLE_MATRIX_MARKET_DIR) + "/nep/stoch/lop163.mtx";
    ScalarPrecision precision = ScalarPrecision::Float64;
    int eigenvalue_count = 3;
    int retained_ritz_count = 0;
    int krylov_dimension = 0;
    int max_iterations = 300;
    double tolerance = 1.0e-10;
    double complex_pair_tolerance = 0.0;
    SpectrumPart spectrum = SpectrumPart::LargestMagnitude;
    bool eigenvectors = false;
    RealNonsymmetricPolicy real_policy = RealNonsymmetricPolicy::RequireRealEigenpairs;
    KrylovDiagnosticsLevel diagnostics = KrylovDiagnosticsLevel::Summary;
};

[[nodiscard]] std::string_view precision_name(ScalarPrecision precision)
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

[[nodiscard]] bool parse_bool(std::string_view value)
{
  if (value == "true" || value == "1" || value == "yes")
  {
    return true;
  }
  if (value == "false" || value == "0" || value == "no")
  {
    return false;
  }
  throw std::invalid_argument(fmt::format("invalid boolean value '{}'", value));
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
  if (value == "LR")
  {
    return SpectrumPart::LargestReal;
  }
  if (value == "SR")
  {
    return SpectrumPart::SmallestReal;
  }
  if (value == "LI")
  {
    return SpectrumPart::LargestImaginary;
  }
  if (value == "SI")
  {
    return SpectrumPart::SmallestImaginary;
  }
  throw std::invalid_argument("nonsymmetric selector must be LM, SM, LR, SR, LI, or SI");
}

[[nodiscard]] std::string_view spectrum_name(SpectrumPart spectrum)
{
  switch (spectrum)
  {
    case SpectrumPart::LargestMagnitude:
      return "LM";
    case SpectrumPart::SmallestMagnitude:
      return "SM";
    case SpectrumPart::LargestReal:
      return "LR";
    case SpectrumPart::SmallestReal:
      return "SR";
    case SpectrumPart::LargestImaginary:
      return "LI";
    case SpectrumPart::SmallestImaginary:
      return "SI";
    case SpectrumPart::LargestAlgebraic:
    case SpectrumPart::SmallestAlgebraic:
    case SpectrumPart::BothEnds:
      return "invalid";
  }
  return "invalid";
}

[[nodiscard]] RealNonsymmetricPolicy parse_real_policy(std::string_view value)
{
  if (value == "require-real")
  {
    return RealNonsymmetricPolicy::RequireRealEigenpairs;
  }
  if (value == "promote-complex")
  {
    return RealNonsymmetricPolicy::PromoteToComplexSuggested;
  }
  if (value == "real-schur")
  {
    return RealNonsymmetricPolicy::AllowRealSchurPairs;
  }
  throw std::invalid_argument("real policy must be require-real, promote-complex, or real-schur");
}

[[nodiscard]] std::string_view real_policy_name(RealNonsymmetricPolicy policy)
{
  switch (policy)
  {
    case RealNonsymmetricPolicy::RequireRealEigenpairs:
      return "require-real";
    case RealNonsymmetricPolicy::PromoteToComplexSuggested:
      return "promote-complex";
    case RealNonsymmetricPolicy::AllowRealSchurPairs:
      return "real-schur";
  }
  return "unknown";
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
  throw std::invalid_argument("diagnostics must be none, summary, or full");
}

[[nodiscard]] std::string_view diagnostics_name(KrylovDiagnosticsLevel level)
{
  switch (level)
  {
    case KrylovDiagnosticsLevel::None:
      return "none";
    case KrylovDiagnosticsLevel::Summary:
      return "summary";
    case KrylovDiagnosticsLevel::Full:
      return "full";
  }
  return "unknown";
}

[[nodiscard]] std::string_view status_name(NonsymmetricStatus status)
{
  switch (status)
  {
    case NonsymmetricStatus::Converged:
      return "converged";
    case NonsymmetricStatus::NotConverged:
      return "not converged";
    case NonsymmetricStatus::ComplexPairEncountered:
      return "complex pair encountered";
    case NonsymmetricStatus::ComplexPromotionRecommended:
      return "complex promotion recommended";
    case NonsymmetricStatus::RealSchurPairRequired:
      return "real Schur pair required";
    case NonsymmetricStatus::AmbiguousReality:
      return "ambiguous reality";
    case NonsymmetricStatus::Breakdown:
      return "breakdown";
    case NonsymmetricStatus::InvalidInput:
      return "invalid input";
  }
  return "unknown";
}

[[nodiscard]] std::string_view reality_name(RitzReality reality)
{
  switch (reality)
  {
    case RitzReality::Real:
      return "real";
    case RitzReality::Ambiguous:
      return "ambiguous";
    case RitzReality::Complex:
      return "complex";
  }
  return "unknown";
}

[[nodiscard]] ExampleOptions parse_options(int argc, char** argv)
{
  ExampleOptions options;
  for (int i = 1; i < argc; ++i)
  {
    std::string_view argument(argv[i]);
    auto const option_value = [&](std::string_view prefix) -> std::string_view {
      if (!argument.starts_with(prefix))
      {
        return {};
      }
      return argument.substr(prefix.size());
    };

    if (argument == "--help" || argument == "-h")
    {
      throw std::runtime_error("help");
    }
    if (auto const value = option_value("--matrix="); !value.empty())
    {
      options.matrix_path = std::string(value);
    }
    else if (auto const value = option_value("--which="); !value.empty())
    {
      options.spectrum = parse_spectrum(value);
    }
    else if (auto const value = option_value("--precision="); !value.empty())
    {
      options.precision = parse_precision(value);
    }
    else if (auto const value = option_value("--nev="); !value.empty())
    {
      options.eigenvalue_count = std::stoi(std::string(value));
    }
    else if (auto const value = option_value("--nkeep="); !value.empty())
    {
      options.retained_ritz_count = std::stoi(std::string(value));
    }
    else if (auto const value = option_value("--ncv="); !value.empty())
    {
      options.krylov_dimension = std::stoi(std::string(value));
    }
    else if (auto const value = option_value("--max-iters="); !value.empty())
    {
      options.max_iterations = std::stoi(std::string(value));
    }
    else if (auto const value = option_value("--tol="); !value.empty())
    {
      options.tolerance = std::stod(std::string(value));
    }
    else if (auto const value = option_value("--complex-tol="); !value.empty())
    {
      options.complex_pair_tolerance = std::stod(std::string(value));
    }
    else if (auto const value = option_value("--real-policy="); !value.empty())
    {
      options.real_policy = parse_real_policy(value);
    }
    else if (auto const value = option_value("--eigenvectors="); !value.empty())
    {
      options.eigenvectors = parse_bool(value);
    }
    else if (auto const value = option_value("--diagnostics="); !value.empty())
    {
      options.diagnostics = parse_diagnostics(value);
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
  return options;
}

void print_usage(char const* program)
{
  fmt::print("Usage: {} [matrix.mtx] [options]\n\n", program);
  fmt::print("Options:\n");
  fmt::print("  --matrix=PATH          Matrix Market real general coordinate file\n");
  fmt::print("  --precision=float|double");
#if UNI20_HAS_FLOAT128
  fmt::print("|float128");
#endif
  fmt::print(", default double\n");
  fmt::print("  --which=LM|SM|LR|SR|LI|SI, default LM\n");
  fmt::print("  --nev=N                Number of Ritz values, default 3\n");
  fmt::print("  --nkeep=N              Retained Schur dimension after restart, default solver policy\n");
  fmt::print("  --ncv=N                Arnoldi subspace size, default solver policy\n");
  fmt::print("  --tol=VALUE            Convergence tolerance, default 1e-10\n");
  fmt::print("  --complex-tol=VALUE    Imaginary-part reality tolerance, default machine-derived\n");
  fmt::print("  --real-policy=require-real|promote-complex|real-schur, default require-real\n");
  fmt::print("  --eigenvectors=BOOL    Reconstruct real Ritz vectors when possible, default false\n");
  fmt::print("  --diagnostics=none|summary|full, default summary\n\n");
  fmt::print("Default matrix: {}/nep/stoch/lop163.mtx\n", UNI20_KRYLOV_EXAMPLE_MATRIX_MARKET_DIR);
}

[[nodiscard]] SparseMatrixMarketMatrix read_real_general_coordinate_matrix_market(std::string const& path)
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
  if (banner != "%%MatrixMarket" || object != "matrix" || format != "coordinate" || field != "real" ||
      symmetry != "general")
  {
    throw std::invalid_argument("example supports real general coordinate Matrix Market matrices only");
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
  matrix.entries.reserve(stored_entries);
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
    matrix.entries.push_back(SparseEntry{.row = row - 1, .col = col - 1, .value = value});
  }

  return matrix;
}

template <typename Scalar> [[nodiscard]] DenseHostVector<Scalar> deterministic_initial_vector(std::size_t dimension)
{
  DenseHostVector<Scalar> vector{std::vector<Scalar>(dimension)};
  for (std::size_t i = 0; i < dimension; ++i)
  {
    vector.values[i] = static_cast<Scalar>((11 * i + 5) % 23 + 1);
  }
  return vector;
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

template <typename Real>
[[nodiscard]] std::string format_ritz_complex(uni20::complex<Real> value,
                                              presentation::numeric_format_options const& numeric)
{
  std::string const real = presentation::format_real(value.real(), numeric);
  std::string const imag = presentation::format_real(std::abs(value.imag()), numeric);
  return fmt::format("{} {} {}i", real, value.imag() < Real{} ? "-" : "+", imag);
}

[[nodiscard]] terminal::TerminalStyle style(std::string_view spec) { return terminal::TerminalStyle(spec); }

template <typename Scalar>
[[nodiscard]] Scalar max_selected_ritz_bound(NonsymmetricEigenResult<Scalar, DenseHostVector<Scalar>> const& result)
{
  Scalar max_bound{};
  for (Scalar const bound : result.residual_bounds)
  {
    max_bound = std::max(max_bound, std::abs(bound));
  }
  return max_bound;
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
                         NonsymmetricEigenResult<Scalar, DenseHostVector<Scalar>> const& result)
{
  auto const numeric = full_precision_numeric_format();
  int constexpr value_width = 48;
  int constexpr scalar_width = 26;

  text.append("\nRitz Values\n", style("Cyan;Bold"));
  text.append("  ")
      .append(presentation::pad_right("#", 4, policy), style("LightGray"))
      .append(presentation::pad_left("theta", value_width, policy), style("LightGray"))
      .append("  ")
      .append(presentation::pad_left("bound", scalar_width, policy), style("LightGray"))
      .append("  ")
      .append(presentation::pad_left("reality", 12, policy), style("LightGray"))
      .append("\n");

  for (std::size_t i = 0; i < result.eigenvalues.size(); ++i)
  {
    std::string const bound =
        i < result.residual_bounds.size() ? presentation::format_real(result.residual_bounds[i], numeric) : "-";
    std::string_view const reality = i < result.reality.size() ? reality_name(result.reality[i]) : "unknown";
    text.append("  ")
        .append(presentation::pad_right(fmt::format("{}", i), 4, policy))
        .append(presentation::pad_left(format_ritz_complex(result.eigenvalues[i], numeric), value_width, policy))
        .append("  ")
        .append(presentation::pad_left(bound, scalar_width, policy))
        .append("  ")
        .append(presentation::pad_left(reality, 12, policy))
        .append("\n");
  }
}

template <typename Scalar>
void append_diagnostics(presentation::styled_text& text,
                        NonsymmetricEigenResult<Scalar, DenseHostVector<Scalar>> const& result)
{
  if (!result.diagnostics.has_value())
  {
    return;
  }

  auto const& diagnostics = *result.diagnostics;
  auto const numeric = full_precision_numeric_format();
  text.append("\nDiagnostics\n", style("Cyan;Bold"));
  text.append(
      fmt::format("  selected converged:                {} / {}\n", result.converged_count, result.eigenvalues.size()));
  text.append("  max selected Ritz bound:           ")
      .append(presentation::format_real(max_selected_ritz_bound(result), numeric))
      .append("\n");
  text.append(fmt::format("  projected dimension:               {}\n", diagnostics.final_projected_dimension));
  text.append(fmt::format("  operator count:                    {}\n", diagnostics.op_count));
  text.append(fmt::format("  restart count:                     {}\n", diagnostics.restart_count));
  text.append("  final Arnoldi expansion residual:  ")
      .append(presentation::format_real(diagnostics.final_residual_norm, numeric))
      .append("\n");
  text.append("  projected departure from normality: ")
      .append(presentation::format_real(diagnostics.projected_departure_from_normality, numeric))
      .append("\n");
}

template <typename Scalar> int run_example(ExampleOptions const& options)
{
  auto matrix = read_real_general_coordinate_matrix_market(options.matrix_path);
  DenseHostVector<Scalar> initial = deterministic_initial_vector<Scalar>(matrix.rows);

  NonsymmetricEigenParams<Scalar> params;
  params.eigenvalue_count = options.eigenvalue_count;
  params.retained_ritz_count = options.retained_ritz_count;
  params.krylov_dimension = options.krylov_dimension;
  params.max_iterations = options.max_iterations;
  params.tolerance = static_cast<Scalar>(options.tolerance);
  params.complex_pair_tolerance = static_cast<Scalar>(options.complex_pair_tolerance);
  params.spectrum = options.spectrum;
  params.compute_eigenvectors = options.eigenvectors;
  params.real_policy = options.real_policy;
  params.diagnostics = options.diagnostics;

  SparseHostVectorOps<Scalar> ops(matrix);
  auto const start = std::chrono::steady_clock::now();
  auto result = uni20::krylov::real_nonsymmetric_arnoldi_restarted_standard(ops, initial, params);
  auto const stop = std::chrono::steady_clock::now();
  double const elapsed_ms = std::chrono::duration<double, std::milli>(stop - start).count();

  auto policy = presentation::terminal_policy(stdout);
  policy.glyphs = presentation::glyph_set::unicode;
  policy.charset = presentation::text_charset::utf8;
  policy.width = presentation::width_mode::display_cells;
  auto const numeric = full_precision_numeric_format();

  presentation::styled_text report;
  report.append("Native restarted real nonsymmetric Arnoldi Matrix Market example\n", style("Yellow;Bold"));
  bool const converged_status = result.status == NonsymmetricStatus::Converged;
  report.append(converged_status ? presentation::semantic_glyph::success : presentation::semantic_glyph::warning,
                converged_status ? style("Green;Bold") : style("Yellow;Bold"));
  report.append(" ").append(status_name(result.status)).append("\n");

  append_key_value(report, policy, "matrix", options.matrix_path);
  append_key_value(report, policy, "precision", precision_name(options.precision));
  append_key_value(report, policy, "dimension", fmt::format("{} x {}", matrix.rows, matrix.cols));
  append_key_value(report, policy, "entries", fmt::format("{}", matrix.entries.size()));
  append_key_value(report, policy, "which", spectrum_name(options.spectrum));
  int const effective_ncv = uni20::krylov::effective_nonsymmetric_krylov_dimension(params, matrix.rows);
  int const effective_nkeep = uni20::krylov::effective_nonsymmetric_retained_ritz_count(params, effective_ncv);
  append_key_value(report, policy, "nev / nkeep / ncv",
                   fmt::format("{} / {} / {}", params.eigenvalue_count, effective_nkeep, effective_ncv));
  append_key_value(report, policy, "tol", presentation::format_real(params.tolerance, numeric));
  append_key_value(report, policy, "complex tol", presentation::format_real(params.complex_pair_tolerance, numeric));
  append_key_value(report, policy, "real policy", real_policy_name(params.real_policy));
  append_key_value(report, policy, "diagnostics", diagnostics_name(params.diagnostics));
  append_key_value(report, policy, "converged", fmt::format("{}", result.converged_count));
  append_key_value(report, policy, "matvecs",
                   fmt::format("{} reported, {} observed", result.matvec_count, ops.matvec_count()));
  append_key_value(report, policy, "iterations", fmt::format("{}", result.iteration_count));
  append_key_value(report, policy, "elapsed", fmt::format("{:.3f} ms", elapsed_ms));

  append_result_table(report, policy, result);
  append_diagnostics(report, result);

  fmt::print("{}", presentation::render_terminal(report, policy, stdout));
  return converged_status ? 0 : 2;
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
