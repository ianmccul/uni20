/**
 * \file async_coroutine_failure_example.cpp
 * \brief Demonstrates structured exception propagation and automatic async DAG diagnostics.
 */

#include <uni20/async/async.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/async/task_registry.hpp>
#include <uni20/common/diagnostic_error.hpp>
#include <uni20/common/display.hpp>
#include <uni20/common/presentation.hpp>
#include <uni20/common/trace.hpp>
#include <uni20/config.hpp>

#include <exception>
#include <filesystem>
#include <string>
#include <string_view>
#include <utility>

namespace
{
namespace async = uni20::async;
namespace presentation = uni20::presentation;

struct Options
{
    std::filesystem::path output_dir{"/tmp"};
    std::string file_prefix{"uni20-coroutine-failure"};
    bool write_graphviz{true};
    bool help{false};
};

struct ParseResult
{
    Options options{};
    std::string error{};
};

class ReciprocalInputError : public uni20::diagnostic_error {
  public:
    explicit ReciprocalInputError(double input)
        : uni20::diagnostic_error("reciprocal requires a nonzero input"), input_(input)
    {}

    [[nodiscard]] double input() const noexcept { return input_; }

  private:
    double input_;
};

presentation::report_builder diagnostic_report(ReciprocalInputError const& error)
{
  presentation::report_builder report;
  report.status(presentation::semantic_glyph::failure, "Async coroutine failed for 'reciprocal'")
      .field("input", error.input())
      .field("precondition", "input != 0")
      .field("propagation", "writer epoch -> dependent coroutine -> observing get_wait() boundary");
  return report;
}

class ErrorBoundaryGuard {
  public:
    ErrorBoundaryGuard() : previous_(trace::get_formatting_options().errors_abort())
    {
      trace::get_formatting_options().set_errors_abort(false);
    }

    ~ErrorBoundaryGuard() { trace::get_formatting_options().set_errors_abort(previous_); }

  private:
    bool previous_;
};

class CoroutineExceptionDiagnosticsGuard {
  public:
    explicit CoroutineExceptionDiagnosticsGuard(uni20::TaskRegistry::CoroutineExceptionDiagnosticsOptions options)
        : previous_(uni20::TaskRegistry::coroutine_exception_diagnostics_options())
    {
      uni20::TaskRegistry::set_coroutine_exception_diagnostics_options(options);
    }

    ~CoroutineExceptionDiagnosticsGuard()
    {
      uni20::TaskRegistry::set_coroutine_exception_diagnostics_options(previous_);
    }

  private:
    uni20::TaskRegistry::CoroutineExceptionDiagnosticsOptions previous_;
};

[[nodiscard]] ParseResult parse_options(int argc, char** argv)
{
  ParseResult result;
  for (int i = 1; i < argc; ++i)
  {
    std::string_view argument(argv[i]);
    if (argument == "--help" || argument == "-h")
    {
      result.options.help = true;
    }
    else if (argument == "--no-dot")
    {
      result.options.write_graphviz = false;
    }
    else if (argument.starts_with("--output-dir="))
    {
      auto const value = argument.substr(std::string_view("--output-dir=").size());
      if (value.empty())
      {
        result.error = "--output-dir requires a non-empty path";
        return result;
      }
      result.options.output_dir = value;
    }
    else if (argument.starts_with("--file-prefix="))
    {
      auto const value = argument.substr(std::string_view("--file-prefix=").size());
      if (value.empty())
      {
        result.error = "--file-prefix requires a non-empty value";
        return result;
      }
      result.options.file_prefix = value;
    }
    else
    {
      result.error = "unknown option: " + std::string(argument);
      return result;
    }
  }
  return result;
}

[[nodiscard]] presentation::report_builder usage_report()
{
  presentation::report_builder report("Async coroutine failure example");
  report.status(presentation::semantic_glyph::info,
                "Raise a structured error inside a coroutine and observe it at an async value boundary");
  report.table("Options")
      .column("option", presentation::table_alignment::left)
      .column("effect", presentation::table_alignment::left)
      .row("--output-dir=PATH", "Directory for the automatically named DOT snapshot")
      .row("--file-prefix=TEXT", "Prefix for the DOT filename")
      .row("--no-dot", "Emit the live registry report without writing Graphviz DOT")
      .row("--help", "Show this option summary");
  return report;
}

async::AsyncTask reciprocal(async::ReadBuffer<double> input, async::WriteBuffer<double> output)
{
  double const value = co_await input;
  input.release();
  if (value == 0.0) trace::raise(ReciprocalInputError(value));
  co_await output = 1.0 / value;
  co_return;
}

async::AsyncTask scale(async::ReadBuffer<double> input, async::WriteBuffer<double> output, double factor)
{
  double const value = co_await input;
  input.release();
  co_await output = factor * value;
  co_return;
}

int run(Options const& options)
{
  ErrorBoundaryGuard error_boundary;
  CoroutineExceptionDiagnosticsGuard diagnostics({
      .enabled = true,
      .write_graphviz = options.write_graphviz,
      .dump_options = {.output_dir = options.output_dir.string(), .file_prefix = options.file_prefix},
  });

  constexpr bool registry_available = UNI20_DEBUG_ASYNC_TASKS;
  std::string_view const automatic_dot =
      !options.write_graphviz ? "disabled" : (registry_available ? "enabled" : "unavailable (debug registry disabled)");
  presentation::report_builder setup("Async coroutine failure demonstration");
  setup.status(presentation::semantic_glyph::info, "the zero input deliberately violates reciprocal's precondition")
      .field("task registry", UNI20_DEBUG_ASYNC_TASKS ? "enabled" : "disabled")
      .field("DAG capture", UNI20_DEBUG_DAG ? "enabled" : "disabled")
      .field("automatic DOT", automatic_dot);
  if (options.write_graphviz && registry_available)
  {
    setup.field("DOT directory", options.output_dir.string()).field("DOT prefix", options.file_prefix);
  }
  uni20::display::emit(std::move(setup), uni20::display::stream::out);

  async::DebugScheduler scheduler;
  async::Async<double> input{0.0};
  async::Async<double> reciprocal_result;
  async::Async<double> scaled_result;
  input.debug_name("invalid reciprocal input");
  reciprocal_result.debug_name("reciprocal result");
  scaled_result.debug_name("downstream scaled result");

  auto producer = reciprocal(input.read(), reciprocal_result.write());
  producer.debug_name("reciprocal(input)");
  scheduler.schedule(std::move(producer));

  auto consumer = scale(reciprocal_result.read(), scaled_result.write(), 4.0);
  consumer.debug_name("4 * reciprocal(input)");
  scheduler.schedule(std::move(consumer));
  scheduler.run_all();

  try
  {
    static_cast<void>(scaled_result.get_wait(scheduler));
  }
  catch (ReciprocalInputError const& error)
  {
    uni20::display::emit(trace::format_diagnostic(error), uni20::display::stream::err);

    presentation::report_builder observed("Exception observation");
    observed
        .status(presentation::semantic_glyph::success,
                "the original structured exception crossed both async writer boundaries")
        .field("observed at", "scaled_result.get_wait(scheduler)")
        .field("concrete type preserved", "ReciprocalInputError");
    uni20::display::emit(std::move(observed), uni20::display::stream::err);
    return 0;
  }
  catch (std::exception const& error)
  {
    uni20::display::failure("Unexpected exception type reached get_wait(): {}", error.what());
    return 1;
  }

  uni20::display::failure("The deliberately invalid coroutine completed without raising an exception");
  return 1;
}

} // namespace

int main(int argc, char** argv)
{
  auto const parsed = parse_options(argc, argv);
  if (!parsed.error.empty())
  {
    uni20::display::failure("{}; run with --help for accepted options", parsed.error);
    return 2;
  }
  if (parsed.options.help)
  {
    uni20::display::emit(usage_report(), uni20::display::stream::out);
    return 0;
  }
  return run(parsed.options);
}
