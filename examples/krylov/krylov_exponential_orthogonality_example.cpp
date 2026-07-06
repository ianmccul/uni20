#include <uni20/common/presentation.hpp>
#include <uni20/krylov/dense_host_vector.hpp>
#include <uni20/krylov/krylov_exponential.hpp>

#include <algorithm>
#include <cmath>
#include <exception>
#include <fmt/core.h>
#include <limits>
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
using uni20::krylov::KrylovExponentialParams;

enum class ScalarPrecision
{
  Float32,
  Float64
};

struct Options
{
    ScalarPrecision precision = ScalarPrecision::Float32;
    double tolerance = 1.0e-8;
    bool check = false;
};

template <typename Scalar> using RealOf = uni20::make_real_t<Scalar>;

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

[[nodiscard]] Options parse_options(int argc, char** argv)
{
  Options options;
  for (int i = 1; i < argc; ++i)
  {
    std::string_view const argument = argv[i];
    if (argument == "--check")
    {
      options.check = true;
    }
    else if (argument == "--precision=float")
    {
      options.precision = ScalarPrecision::Float32;
    }
    else if (argument == "--precision=double")
    {
      options.precision = ScalarPrecision::Float64;
    }
    else if (starts_with(argument, "--tol="))
    {
      options.tolerance = parse_double(option_value(argument, "--tol="), "--tol");
    }
    else if (argument == "--help" || argument == "-h")
    {
      fmt::print("Usage: krylov_exponential_orthogonality_example "
                 "[--precision=float|double] [--tol=value] [--check]\n");
      std::exit(EXIT_SUCCESS);
    }
    else
    {
      throw std::invalid_argument(fmt::format("unknown argument: {}", argument));
    }
  }
  return options;
}

template <typename Real> [[nodiscard]] std::string format_real(Real value)
{
  if (std::isnan(static_cast<double>(value)))
  {
    return "nan";
  }
  if (std::isinf(static_cast<double>(value)))
  {
    return value > Real{} ? "inf" : "-inf";
  }
  return fmt::format("{:.3e}", static_cast<double>(value));
}

template <typename Scalar> [[nodiscard]] RealOf<Scalar> vector_norm(DenseHostVector<Scalar> const& vector)
{
  long double sum = 0.0L;
  for (auto const& value : vector.values)
  {
    long double const magnitude = static_cast<long double>(std::abs(value));
    sum += magnitude * magnitude;
  }
  return static_cast<RealOf<Scalar>>(std::sqrt(sum));
}

template <typename Scalar>
[[nodiscard]] RealOf<Scalar> difference_norm(DenseHostVector<Scalar> const& lhs, DenseHostVector<Scalar> const& rhs)
{
  if (lhs.values.size() != rhs.values.size())
  {
    throw std::invalid_argument("difference_norm requires equal vector sizes");
  }

  long double sum = 0.0L;
  for (std::size_t i = 0; i < lhs.values.size(); ++i)
  {
    long double const magnitude = static_cast<long double>(std::abs(lhs.values[i] - rhs.values[i]));
    sum += magnitude * magnitude;
  }
  return static_cast<RealOf<Scalar>>(std::sqrt(sum));
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
    result += lhs.values[i] * rhs.values[i];
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

template <typename Scalar>
[[nodiscard]] std::vector<Scalar> arithmetic_spectrum(std::size_t size, Scalar start, Scalar end)
{
  std::vector<Scalar> values(size);
  if (size == 1)
  {
    values.front() = start;
    return values;
  }

  for (std::size_t i = 0; i < size; ++i)
  {
    Scalar const fraction = static_cast<Scalar>(i) / static_cast<Scalar>(size - 1);
    values[i] = start + fraction * (end - start);
  }
  return values;
}

template <typename Scalar> [[nodiscard]] std::vector<Scalar> diagonal_matrix(std::vector<Scalar> const& eigenvalues)
{
  std::size_t const n = eigenvalues.size();
  std::vector<Scalar> matrix(n * n, Scalar{});
  for (std::size_t i = 0; i < n; ++i)
  {
    matrix[i * n + i] = eigenvalues[i];
  }
  return matrix;
}

template <typename Scalar> [[nodiscard]] DenseHostVector<Scalar> deterministic_initial(std::size_t size)
{
  DenseHostVector<Scalar> vector{{}};
  vector.values.reserve(size);
  for (std::size_t i = 0; i < size; ++i)
  {
    Scalar const value = std::sin(static_cast<Scalar>(0.19) * static_cast<Scalar>(i + 1)) +
                         Scalar{0.25} * std::cos(static_cast<Scalar>(0.07) * static_cast<Scalar>(i + 3));
    vector.values.push_back(value);
  }
  return vector;
}

template <typename Scalar>
[[nodiscard]] DenseHostVector<Scalar> exact_diagonal_action(std::vector<Scalar> const& eigenvalues,
                                                            DenseHostVector<Scalar> const& initial, Scalar time)
{
  DenseHostVector<Scalar> result{{}};
  result.values.resize(initial.values.size());
  for (std::size_t i = 0; i < initial.values.size(); ++i)
  {
    result.values[i] = std::exp(time * eigenvalues[i]) * initial.values[i];
  }
  return result;
}

template <typename Scalar>
[[nodiscard]] DenseHostVector<Scalar> diagonal_matvec(std::vector<Scalar> const& eigenvalues,
                                                      DenseHostVector<Scalar> const& vector)
{
  DenseHostVector<Scalar> result{{}};
  result.values.resize(vector.values.size());
  for (std::size_t i = 0; i < vector.values.size(); ++i)
  {
    result.values[i] = eigenvalues[i] * vector.values[i];
  }
  return result;
}

template <typename Scalar>
[[nodiscard]] RealOf<Scalar> basis_max_offdiag(std::vector<DenseHostVector<Scalar>> const& basis)
{
  using Real = RealOf<Scalar>;
  Real result{};
  for (std::size_t row = 0; row < basis.size(); ++row)
  {
    for (std::size_t col = row + 1; col < basis.size(); ++col)
    {
      result = std::max(result, static_cast<Real>(std::abs(host_inner_product(basis[row], basis[col]))));
    }
  }
  return result;
}

template <typename Scalar> struct RunRow
{
    std::string method;
    int krylov_dimension = 0;
    int matvecs = 0;
    RealOf<Scalar> error_scaled = RealOf<Scalar>{};
    RealOf<Scalar> estimate_scaled = RealOf<Scalar>{};
    RealOf<Scalar> orthogonality_offdiag = RealOf<Scalar>{};
    RealOf<Scalar> reorthogonalization_ratio = RealOf<Scalar>{};
    int reorthogonalization_passes = 0;
};

template <typename Scalar>
[[nodiscard]] RunRow<Scalar>
run_full_reorthogonalized(std::vector<Scalar> const& eigenvalues, DenseHostVector<Scalar> const& initial,
                          DenseHostVector<Scalar> const& exact, Scalar time, int krylov_dimension)
{
  using Real = RealOf<Scalar>;
  DenseHostVectorOps<Scalar> ops(eigenvalues.size(), diagonal_matrix(eigenvalues));

  KrylovExponentialParams<Real> params;
  params.krylov_dimension = krylov_dimension;
  params.diagnostics = KrylovDiagnosticsLevel::Summary;

  auto result = uni20::krylov::hermitian_krylov_exponential_action<Scalar>(ops, initial, time, params);
  if (!result.diagnostics.has_value())
  {
    throw std::logic_error("full reorthogonalized run did not return diagnostics");
  }

  Real const initial_norm = vector_norm(initial);
  return RunRow<Scalar>{.method = "full reorth",
                        .krylov_dimension = krylov_dimension,
                        .matvecs = result.matvec_count,
                        .error_scaled = difference_norm(result.action, exact) / initial_norm,
                        .estimate_scaled = result.error_estimate / initial_norm,
                        .orthogonality_offdiag = result.diagnostics->basis_max_offdiag,
                        .reorthogonalization_ratio = result.diagnostics->max_reorthogonalization_correction_ratio,
                        .reorthogonalization_passes = result.diagnostics->max_reorthogonalization_passes};
}

template <typename Scalar>
[[nodiscard]] RunRow<Scalar>
run_legacy_three_term(std::vector<Scalar> const& eigenvalues, DenseHostVector<Scalar> const& initial,
                      DenseHostVector<Scalar> const& exact, Scalar time, int krylov_dimension)
{
  using Real = RealOf<Scalar>;

  Real const initial_norm = vector_norm(initial);
  DenseHostVector<Scalar> current = initial;
  host_scale(current, Scalar{1} / initial_norm);
  DenseHostVector<Scalar> previous{{std::vector<Scalar>(initial.values.size(), Scalar{})}};

  std::vector<DenseHostVector<Scalar>> basis;
  std::vector<Real> diagonal;
  std::vector<Real> offdiagonal;
  basis.reserve(static_cast<std::size_t>(krylov_dimension));
  diagonal.reserve(static_cast<std::size_t>(krylov_dimension));
  offdiagonal.reserve(static_cast<std::size_t>(krylov_dimension));

  Real residual_norm{};
  int matvecs = 0;
  for (int step = 0; step < krylov_dimension; ++step)
  {
    basis.push_back(current);
    auto residual = diagonal_matvec(eigenvalues, current);
    ++matvecs;

    Real const alpha = host_inner_product(current, residual);
    diagonal.push_back(alpha);
    host_axpy(residual, -alpha, current);
    if (step > 0)
    {
      host_axpy(residual, -offdiagonal.back(), previous);
    }

    residual_norm = vector_norm(residual);
    offdiagonal.push_back(residual_norm);
    if (residual_norm == Real{})
    {
      break;
    }

    previous = current;
    current = std::move(residual);
    host_scale(current, Scalar{1} / residual_norm);
  }

  std::size_t const projected_size = diagonal.size();
  uni20::krylov::Matrix<Scalar> projected(projected_size, projected_size);
  for (std::size_t i = 0; i < projected_size; ++i)
  {
    projected[i, i] = diagonal[i];
    if (i + 1 < projected_size)
    {
      projected[i, i + 1] = offdiagonal[i];
      projected[i + 1, i] = offdiagonal[i];
    }
  }

  uni20::krylov::Matrix<Scalar> scaled(projected_size, projected_size);
  for (std::size_t row = 0; row < projected_size; ++row)
  {
    for (std::size_t col = 0; col < projected_size; ++col)
    {
      scaled[row, col] = time * projected[row, col];
    }
  }

  auto const exponential = uni20::krylov::matrix_exponential(scaled, Scalar{1});
  DenseHostVector<Scalar> action{{std::vector<Scalar>(initial.values.size(), Scalar{})}};
  std::vector<Scalar> coefficients(projected_size);
  for (std::size_t row = 0; row < projected_size; ++row)
  {
    coefficients[row] = initial_norm * exponential[row, 0];
    host_axpy(action, coefficients[row], basis[row]);
  }

  Real const endpoint_defect_estimate =
      coefficients.empty() ? Real{} : residual_norm * static_cast<Real>(std::abs(coefficients.back()));

  return RunRow<Scalar>{.method = "three-term",
                        .krylov_dimension = krylov_dimension,
                        .matvecs = matvecs,
                        .error_scaled = difference_norm(action, exact) / initial_norm,
                        .estimate_scaled = endpoint_defect_estimate / initial_norm,
                        .orthogonality_offdiag = basis_max_offdiag(basis),
                        .reorthogonalization_ratio = Real{},
                        .reorthogonalization_passes = 0};
}

template <typename Scalar> struct DemoResult
{
    Scalar tolerance = Scalar{};
    Scalar non_reorthogonalized_floor = Scalar{};
    Scalar initial_norm = Scalar{};
    Scalar reference_norm = Scalar{};
    std::vector<RunRow<Scalar>> rows;
    bool false_estimator_pass = false;
    bool legacy_lost_orthogonality = false;
    bool full_reorthogonalization_pressure = false;
};

template <typename Scalar> [[nodiscard]] DemoResult<Scalar> run_demo(Scalar tolerance)
{
  std::vector<Scalar> const eigenvalues = arithmetic_spectrum<Scalar>(96, Scalar{0}, Scalar{80});
  DenseHostVector<Scalar> const initial = deterministic_initial<Scalar>(eigenvalues.size());
  Scalar const time = Scalar{-0.25};
  DenseHostVector<Scalar> const exact = exact_diagonal_action(eigenvalues, initial, time);
  std::vector<int> const dimensions{16, 20, 24, 32, 48, 64, 80, 96};

  DemoResult<Scalar> result{.tolerance = tolerance,
                            .non_reorthogonalized_floor = std::sqrt(uni20::numeric_limits<Scalar>::epsilon()),
                            .initial_norm = vector_norm(initial),
                            .reference_norm = vector_norm(exact),
                            .rows = {}};

  for (int const dimension : dimensions)
  {
    result.rows.push_back(run_full_reorthogonalized(eigenvalues, initial, exact, time, dimension));
    result.rows.push_back(run_legacy_three_term(eigenvalues, initial, exact, time, dimension));
  }

  for (auto const& row : result.rows)
  {
    bool const estimate_passes = row.estimate_scaled <= tolerance;
    bool const error_passes = row.error_scaled <= tolerance;
    result.false_estimator_pass = result.false_estimator_pass || (estimate_passes && !error_passes);
    if (row.method == "three-term")
    {
      result.legacy_lost_orthogonality = result.legacy_lost_orthogonality || row.orthogonality_offdiag > Scalar{0.1};
    }
    else
    {
      result.full_reorthogonalization_pressure =
          result.full_reorthogonalization_pressure ||
          (row.reorthogonalization_ratio > Scalar{0.1} && row.reorthogonalization_passes >= 2);
    }
  }

  return result;
}

template <typename Scalar> [[nodiscard]] bool demo_check_passes(DemoResult<Scalar> const& result)
{
  return result.tolerance < result.non_reorthogonalized_floor && result.false_estimator_pass &&
         result.legacy_lost_orthogonality && result.full_reorthogonalization_pressure;
}

template <typename Scalar> void print_demo(DemoResult<Scalar> const& result)
{
  presentation::report_builder report("Krylov exponential orthogonality floor example");
  report
      .status(demo_check_passes(result) ? presentation::semantic_glyph::success : presentation::semantic_glyph::warning,
              demo_check_passes(result) ? "orthogonality floor exposed" : "inspect diagnostics")
      .field("scalar", std::is_same_v<Scalar, float> ? "float" : "double")
      .field("requested rel tol", format_real(result.tolerance))
      .field("non-reorth floor", format_real(result.non_reorthogonalized_floor))
      .field("tol below floor", result.tolerance < result.non_reorthogonalized_floor ? "yes" : "no")
      .field("||v||", format_real(result.initial_norm))
      .field("||reference||", format_real(result.reference_norm));

  auto& summary = report.table("Diagnostic Checks");
  summary.column("condition").column("observed");
  summary.row("estimate <= tol while true error > tol", result.false_estimator_pass ? "yes" : "no");
  summary.row("legacy three-term basis loses orthogonality", result.legacy_lost_orthogonality ? "yes" : "no");
  summary.row("full reorth path needs strong cleanup", result.full_reorthogonalization_pressure ? "yes" : "no");

  auto& table = report.table("Lanczos Dimension Sweep");
  table.column("method")
      .column("m")
      .column("err/||v||")
      .column("defect/||v||")
      .column("est<=tol")
      .column("err<=tol")
      .column("orth offdiag")
      .column("reorth ratio")
      .column("passes");

  for (auto const& row : result.rows)
  {
    table.row(row.method, fmt::format("{}", row.krylov_dimension), format_real(row.error_scaled),
              format_real(row.estimate_scaled), row.estimate_scaled <= result.tolerance ? "yes" : "no",
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

template <typename Scalar> int run(Options const& options)
{
  auto const result = run_demo<Scalar>(static_cast<Scalar>(options.tolerance));
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
    }
  }
  catch (std::exception const& error)
  {
    fmt::print(stderr, "error: {}\n", error.what());
    return EXIT_FAILURE;
  }
  return EXIT_FAILURE;
}
