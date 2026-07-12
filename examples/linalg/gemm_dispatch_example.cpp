#include <uni20/common/display.hpp>
#include <uni20/common/presentation_mdspan.hpp>
#include <uni20/core/numeric_limits.hpp>
#include <uni20/core/types.hpp>
#include <uni20/linalg/ops/gemm.hpp>
#include <uni20/tensor/tensor.hpp>

#include <array>
#include <cstdio>
#include <fmt/core.h>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>

namespace
{
enum class Precision
{
  fp32,
  fp64,
  fp128
};

struct Options
{
    Precision precision = Precision::fp64;
    bool diagnostics = true;
    bool show_help = false;
};

constexpr std::string_view acceptance_name(uni20::linalg::KernelTypeAcceptance acceptance)
{
  switch (acceptance)
  {
    case uni20::linalg::KernelTypeAcceptance::no:
      return "no";
    case uni20::linalg::KernelTypeAcceptance::maybe:
      return "maybe";
    case uni20::linalg::KernelTypeAcceptance::yes:
      return "yes";
  }
  return "unknown";
}

[[nodiscard]] Precision parse_precision(std::string_view value)
{
  if (value == "fp32") return Precision::fp32;
  if (value == "fp64") return Precision::fp64;
  if (value == "fp128") return Precision::fp128;
  throw std::invalid_argument("unsupported precision '" + std::string(value) + "'");
}

[[nodiscard]] bool parse_diagnostics(std::string_view value)
{
  if (value == "on") return true;
  if (value == "off") return false;
  throw std::invalid_argument("--diagnostics must be on or off");
}

[[nodiscard]] Options parse_options(int argc, char** argv)
{
  Options options;
  for (int index = 1; index < argc; ++index)
  {
    std::string_view const argument(argv[index]);
    if (argument == "--help" || argument == "-h")
    {
      options.show_help = true;
    }
    else if (argument.starts_with("--precision="))
    {
      options.precision = parse_precision(argument.substr(std::string_view("--precision=").size()));
    }
    else if (argument.starts_with("--diagnostics="))
    {
      options.diagnostics = parse_diagnostics(argument.substr(std::string_view("--diagnostics=").size()));
    }
    else
    {
      throw std::invalid_argument("unknown option '" + std::string(argument) + "'");
    }
  }
  return options;
}

void print_usage(char const* program)
{
#if UNI20_HAS_FLOAT128
  constexpr std::string_view precisions = "fp32|fp64|fp128";
#else
  constexpr std::string_view precisions = "fp32|fp64";
#endif
  fmt::print("Usage: {} [--precision={}] [--diagnostics=on|off]\n", program, precisions);
  fmt::print("  --precision       scalar type used by Tensor and GEMM; default fp64\n");
  fmt::print("  --diagnostics     emit the ordered runtime backend walk; default on\n");
}

template <class Matrix, class Scalar> void fill_matrix(Matrix& matrix, std::initializer_list<Scalar> values)
{
  auto const expected_size = static_cast<std::size_t>(matrix.extent(0) * matrix.extent(1));
  if (values.size() != expected_size) throw std::logic_error("matrix initializer has the wrong size");

  auto value = values.begin();
  for (uni20::index_type row = 0; row < matrix.extent(0); ++row)
  {
    for (uni20::index_type col = 0; col < matrix.extent(1); ++col)
    {
      matrix[row, col] = *value;
      ++value;
    }
  }
}

template <class Scalar> [[nodiscard]] Scalar absolute_value(Scalar value) { return value < Scalar{} ? -value : value; }

[[nodiscard]] uni20::presentation::mdspan_format_options matrix_format_options()
{
  uni20::presentation::mdspan_format_options options;
  options.numeric.float32_precision = uni20::numeric_limits<uni20::float32>::max_digits10;
  options.numeric.float64_precision = uni20::numeric_limits<uni20::float64>::max_digits10;
  options.numeric.float128_precision = 36;
  return options;
}

template <class Matrix>
void print_matrix(std::string_view name, Matrix const& matrix, uni20::presentation::output_policy const& policy,
                  uni20::presentation::mdspan_format_options const& format)
{
  fmt::print("{}\n{}\n\n", name, uni20::presentation::format_mdspan(matrix.mdspan(), policy, format));
}

template <class Scalar> int run_gemm(std::string_view precision_name, bool diagnostics)
{
  using tensor_type = uni20::Tensor<Scalar, 2>;
  using extents_type = typename tensor_type::extents_type;

  tensor_type lhs(extents_type{2, 3});
  tensor_type rhs(extents_type{3, 2});
  tensor_type output(extents_type{2, 2});

  fill_matrix(lhs,
              {Scalar{1} / Scalar{3}, Scalar{2}, -Scalar{1}, Scalar{4}, Scalar{1} / Scalar{7}, Scalar{1} / Scalar{2}});
  fill_matrix(rhs, {Scalar{3}, Scalar{1}, -Scalar{2}, Scalar{5} / Scalar{9}, Scalar{4}, -Scalar{3}});
  fill_matrix(output, {Scalar{1} / Scalar{4}, -Scalar{1} / Scalar{2}, Scalar{3} / Scalar{4}, Scalar{5} / Scalar{4}});

  Scalar const alpha = Scalar{5} / Scalar{4};
  Scalar const beta = Scalar{1} / Scalar{2};
  auto const format = matrix_format_options();
  auto const policy = uni20::presentation::terminal_policy(stdout);

  auto selector = output.backend_selector();
  auto output_span = output.mdspan();
  auto lhs_span = lhs.mdspan();
  auto rhs_span = rhs.mdspan();
  auto const acceptance = uni20::linalg::probe_dispatch_kernel(selector, uni20::linalg::gemm_op{}, output_span, alpha,
                                                               lhs_span, rhs_span, beta);

  fmt::print("Tensor GEMM dispatch\n");
  fmt::print("  operation:           C <- alpha A B + beta C\n");
  fmt::print("  precision:           {}\n", precision_name);
  fmt::print("  type acceptance:     {}\n", acceptance_name(acceptance));
  fmt::print("  runtime diagnostics: {}\n", diagnostics ? "on" : "off");
  fmt::print("  alpha:               {}\n", uni20::presentation::format_real(alpha, format.numeric));
  fmt::print("  beta:                {}\n\n", uni20::presentation::format_real(beta, format.numeric));

  print_matrix("A", lhs, policy, format);
  print_matrix("B", rhs, policy, format);
  print_matrix("C before", output, policy, format);

  // Owning tensors provide the selector; GEMM lowers their mdspans into the backend walk.
  uni20::linalg::gemm(output, alpha, lhs, rhs, beta);

  print_matrix("C after", output, policy, format);

  std::array<Scalar, 4> const expected{-Scalar{69} / Scalar{8}, Scalar{191} / Scalar{36}, Scalar{981} / Scalar{56},
                                       Scalar{485} / Scalar{126}};
  Scalar maximum_error{};
  for (uni20::index_type row = 0; row < 2; ++row)
  {
    for (uni20::index_type col = 0; col < 2; ++col)
    {
      Scalar const error = absolute_value(output[row, col] - expected[static_cast<std::size_t>(2 * row + col)]);
      if (error > maximum_error) maximum_error = error;
    }
  }

  Scalar const tolerance = Scalar{4096} * uni20::numeric_limits<Scalar>::epsilon();
  bool const correct = maximum_error <= tolerance;
  fmt::print("maximum absolute error: {}\n", uni20::presentation::format_real(maximum_error, format.numeric));
  fmt::print("validation: {}\n", correct ? "passed" : "failed");
  return correct ? 0 : 1;
}

[[nodiscard]] int run_selected_precision(Options const& options)
{
  switch (options.precision)
  {
    case Precision::fp32:
      return run_gemm<uni20::float32>("fp32", options.diagnostics);
    case Precision::fp64:
      return run_gemm<uni20::float64>("fp64", options.diagnostics);
    case Precision::fp128:
#if UNI20_HAS_FLOAT128
      return run_gemm<uni20::float128>("fp128", options.diagnostics);
#else
      throw std::invalid_argument("fp128 requires a build configured with MPLAPACK binary128 support");
#endif
  }
  throw std::logic_error("unhandled precision selection");
}
} // namespace

int main(int argc, char** argv)
{
  try
  {
    Options const options = parse_options(argc, argv);
    if (options.show_help)
    {
      print_usage(argv[0]);
      return 0;
    }

    if (!options.diagnostics) return run_selected_precision(options);

    uni20::linalg::dispatch_diagnostics::scoped_sink diagnostics(
        [](uni20::linalg::dispatch_diagnostics::event const& diagnostic) {
          uni20::display::emit(uni20::linalg::diagnostic_report(diagnostic), uni20::display::stream::out);
        });
    return run_selected_precision(options);
  }
  catch (std::exception const& error)
  {
    fmt::print(stderr, "error: {}\n", error.what());
    print_usage(argv[0]);
    return 2;
  }
}
