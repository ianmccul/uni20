/**
 * \file async_tbb_matrix_product_batch_example.cpp
 * \brief Runs a configurable batch of independent async matrix products with TBB.
 */

#include <uni20/async/async.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/async/tbb_scheduler.hpp>
#include <uni20/common/display.hpp>
#include <uni20/common/presentation.hpp>
#include <uni20/core/numeric_limits.hpp>
#include <uni20/core/scalar_precision.hpp>
#include <uni20/core/types.hpp>
#include <uni20/linalg/async.hpp>
#include <uni20/tensor/tensor.hpp>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <fmt/core.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace
{

enum class Backend
{
  Cpu,
  Automatic
};

struct Options
{
    std::size_t size = 128;
    std::size_t products = 8;
    std::size_t threads = 4;
    Backend backend = Backend::Cpu;
    uni20::ScalarPrecision precision = uni20::ScalarPrecision::fp64;
    bool show_help = false;
};

[[nodiscard]] std::size_t parse_positive_size(std::string_view value, std::string_view option)
{
  std::size_t result = 0;
  auto const [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
  if (error != std::errc{} || end != value.data() + value.size() || result == 0)
    throw std::invalid_argument(std::string(option) + " requires a positive integer");
  return result;
}

[[nodiscard]] Backend parse_backend(std::string_view value)
{
  if (value == "cpu") return Backend::Cpu;
  if (value == "auto") return Backend::Automatic;
  throw std::invalid_argument("--backend must be cpu or auto");
}

[[nodiscard]] uni20::ScalarPrecision parse_precision(std::string_view value)
{
  auto const precision = uni20::parse_scalar_precision(value);
  if (!precision) throw std::invalid_argument("unsupported precision '" + std::string(value) + "'");
  return *precision;
}

[[nodiscard]] Options parse_options(int argc, char** argv)
{
  Options options;
  for (int index = 1; index < argc; ++index)
  {
    std::string_view const argument(argv[index]);
    if (argument == "--help" || argument == "-h")
      options.show_help = true;
    else if (argument.starts_with("--size="))
      options.size = parse_positive_size(argument.substr(std::string_view("--size=").size()), "--size");
    else if (argument.starts_with("--products="))
      options.products = parse_positive_size(argument.substr(std::string_view("--products=").size()), "--products");
    else if (argument.starts_with("--threads="))
      options.threads = parse_positive_size(argument.substr(std::string_view("--threads=").size()), "--threads");
    else if (argument.starts_with("--backend="))
      options.backend = parse_backend(argument.substr(std::string_view("--backend=").size()));
    else if (argument.starts_with("--precision="))
      options.precision = parse_precision(argument.substr(std::string_view("--precision=").size()));
    else
      throw std::invalid_argument("unknown option '" + std::string(argument) + "'");
  }

  if (options.size > static_cast<std::size_t>(std::numeric_limits<uni20::index_type>::max()))
    throw std::invalid_argument("--size exceeds Uni20's index range");
  if (options.threads > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::invalid_argument("--threads exceeds TBB's concurrency range");
  return options;
}

void print_usage(char const* program)
{
  fmt::print("Usage: {} [--size=N] [--products=N] [--threads=N] [--backend=cpu|auto] "
             "[--precision={}]\n",
             program, uni20::configured_scalar_precision_choices());
  fmt::print("  --size       rows and columns in each square matrix; default 128\n");
  fmt::print("  --products   number of independent matrix products; default 8\n");
  fmt::print("  --threads    TBB arena concurrency, including the application thread; default 4\n");
  fmt::print("  --backend    cpu gives predictable task parallelism; auto enables normal BLAS fallback; default cpu\n");
  fmt::print("  --precision  matrix scalar precision; default fp64\n");
}

template <class T> uni20::async::AsyncTask publish(uni20::async::WriteBuffer<T> output, T value)
{
  co_await output = std::move(value);
  co_return;
}

template <class Scalar> [[nodiscard]] uni20::DenseMatrix<Scalar> make_lhs(std::size_t product, uni20::index_type size)
{
  uni20::DenseMatrix<Scalar> result(size, size);
  for (uni20::index_type row = 0; row < size; ++row)
  {
    for (uni20::index_type col = 0; col < size; ++col)
    {
      auto const pattern =
          static_cast<int>((7 * product + 3 * static_cast<std::size_t>(row) + 5 * static_cast<std::size_t>(col)) % 19) -
          9;
      result[row, col] = static_cast<Scalar>(pattern) / Scalar{8};
    }
  }
  return result;
}

template <class Scalar>
[[nodiscard]] uni20::DenseMatrix<Scalar> make_rhs(std::size_t product, uni20::index_type size, Scalar& diagonal,
                                                  Scalar& shift)
{
  uni20::DenseMatrix<Scalar> result(size, size);
  diagonal = Scalar{1} + static_cast<Scalar>(product % 5) / Scalar{16};
  shift = Scalar{1} / Scalar{4} + static_cast<Scalar>(product % 3) / Scalar{32};
  for (uni20::index_type index = 0; index < size; ++index)
  {
    result[index, index] += diagonal;
    result[index, (index + 1) % size] += shift;
  }
  return result;
}

template <class Scalar> [[nodiscard]] Scalar absolute_value(Scalar value) { return value < Scalar{} ? -value : value; }

template <class Scalar> int run(Options const& options)
{
  using matrix_type = uni20::DenseMatrix<Scalar>;
  using async_matrix_type = uni20::async::Async<matrix_type>;

  auto const size = static_cast<uni20::index_type>(options.size);
  uni20::async::TbbScheduler scheduler(static_cast<int>(options.threads));
  scheduler.pause();
  uni20::async::ScopedScheduler scoped_scheduler(&scheduler);

  std::vector<async_matrix_type> lhs;
  std::vector<async_matrix_type> rhs;
  std::vector<async_matrix_type> output(options.products);
  std::vector<Scalar> diagonal(options.products);
  std::vector<Scalar> shift(options.products);
  lhs.reserve(options.products);
  rhs.reserve(options.products);

  for (std::size_t product = 0; product < options.products; ++product)
  {
    lhs.emplace_back(make_lhs<Scalar>(product, size));
    rhs.emplace_back(make_rhs<Scalar>(product, size, diagonal[product], shift[product]));
  }

  // Every product waits on one shared scalar epoch, giving the batch a small fan-out dependency.
  uni20::async::Async<Scalar> alpha;
  uni20::async::schedule(publish(alpha.write(), Scalar{1}));
  for (std::size_t product = 0; product < options.products; ++product)
  {
    if (options.backend == Backend::Cpu)
      uni20::linalg::assign_product(uni20::linalg::CpuReferenceBackend{}, output[product], lhs[product], rhs[product],
                                    alpha);
    else
      uni20::linalg::assign_product(output[product], lhs[product], rhs[product], alpha);
  }

  auto const start = std::chrono::steady_clock::now();
  scheduler.run_all();
  auto const elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

  Scalar maximum_error{};
  Scalar maximum_expected{};
  for (std::size_t product = 0; product < options.products; ++product)
  {
    auto const& a = lhs[product].get_wait(scheduler);
    auto const& c = output[product].get_wait(scheduler);
    for (uni20::index_type row = 0; row < size; ++row)
    {
      for (uni20::index_type col = 0; col < size; ++col)
      {
        auto const previous_col = (col + size - 1) % size;
        Scalar const expected = diagonal[product] * a[row, col] + shift[product] * a[row, previous_col];
        maximum_error = std::max(maximum_error, absolute_value(c[row, col] - expected));
        maximum_expected = std::max(maximum_expected, absolute_value(expected));
      }
    }
  }

  Scalar const tolerance = Scalar{64} * uni20::numeric_limits<Scalar>::epsilon() * static_cast<Scalar>(options.size) *
                           std::max(Scalar{1}, maximum_expected);
  bool const correct = maximum_error <= tolerance;
  long double const operations = 2.0L * static_cast<long double>(options.products) *
                                 static_cast<long double>(options.size) * static_cast<long double>(options.size) *
                                 static_cast<long double>(options.size);
  long double const gibibytes = 3.0L * static_cast<long double>(options.products) *
                                static_cast<long double>(options.size) * static_cast<long double>(options.size) *
                                static_cast<long double>(sizeof(Scalar)) / (1024.0L * 1024.0L * 1024.0L);

  uni20::presentation::numeric_format_options numeric_format;
  numeric_format.float32_precision = uni20::numeric_limits<uni20::float32>::max_digits10;
  numeric_format.float64_precision = uni20::numeric_limits<uni20::float64>::max_digits10;

  uni20::presentation::report_builder report("Async TBB matrix-product batch");
  report
      .status(correct ? uni20::presentation::semantic_glyph::success : uni20::presentation::semantic_glyph::failure,
              correct ? "validation passed" : "validation failed")
      .field("precision", uni20::scalar_precision_name(options.precision))
      .field("backend", options.backend == Backend::Cpu ? "cpu_reference" : "automatic")
      .field("matrix shape", fmt::format("{} x {}", options.size, options.size))
      .field("products", options.products)
      .field("TBB concurrency", options.threads);

  report.table("Execution summary")
      .grid()
      .column("quantity", uni20::presentation::table_alignment::left)
      .column("value", uni20::presentation::table_alignment::decimal)
      .column("unit", uni20::presentation::table_alignment::left)
      .row("matrix storage", fmt::format("{:.3f}", static_cast<double>(gibibytes)), "GiB")
      .row("elapsed", fmt::format("{:.6f}", elapsed), "s")
      .row("nominal rate", fmt::format("{:.3f}", static_cast<double>(operations / elapsed / 1.0e9L)), "GFLOP/s")
      .row("maximum error", uni20::presentation::format_real(maximum_error, numeric_format), "")
      .row("tolerance", uni20::presentation::format_real(tolerance, numeric_format), "");

  uni20::display::emit(std::move(report), uni20::display::stream::out);
  return correct ? 0 : 1;
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

    return uni20::visit_scalar_precision(options.precision, [&]<typename Scalar>() { return run<Scalar>(options); });
  }
  catch (std::exception const& error)
  {
    uni20::display::failure("{}", error.what());
    print_usage(argv[0]);
    return 2;
  }
}
