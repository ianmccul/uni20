/**
 * \file async_diagnostics_guide_example.cpp
 * \brief Interactive guide to Uni20 async diagnostics, Graphviz snapshots, and presentation controls.
 */

#include <uni20/async/async.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/async/task_registry.hpp>
#include <uni20/async/task_registry_presentation.hpp>
#include <uni20/common/presentation.hpp>
#include <uni20/common/terminal.hpp>
#include <uni20/config.hpp>

#include <charconv>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fmt/format.h>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace
{
namespace async = uni20::async;
namespace presentation = uni20::presentation;

struct Options
{
    std::filesystem::path output{"async-diagnostics-guide.dot"};
    std::optional<std::size_t> width{};
    std::optional<std::size_t> stacktrace_frames{};
    std::optional<bool> internal_frames{};
    std::optional<presentation::glyph_set> glyphs{};
    std::optional<presentation::text_charset> charset{};
    std::optional<presentation::color_mode> color{};
    bool write_dot{true};
    bool help{false};
};

struct ParseResult
{
    Options options{};
    std::string error{};
};

[[nodiscard]] std::string compiler_name()
{
#if defined(__clang__)
  return fmt::format("Clang {}.{}.{}", __clang_major__, __clang_minor__, __clang_patchlevel__);
#elif defined(__GNUC__)
  return fmt::format("GCC {}.{}.{}", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#else
  return "unknown compiler";
#endif
}

[[nodiscard]] bool parse_positive_size(std::string_view text, std::size_t& value)
{
  if (text.empty()) return false;
  std::size_t parsed = 0;
  auto const [end, error] = std::from_chars(text.data(), text.data() + text.size(), parsed);
  if (error != std::errc{} || end != text.data() + text.size() || parsed == 0) return false;
  value = parsed;
  return true;
}

[[nodiscard]] std::optional<bool> parse_bool(std::string_view text)
{
  if (text == "yes" || text == "true" || text == "on" || text == "1") return true;
  if (text == "no" || text == "false" || text == "off" || text == "0") return false;
  return std::nullopt;
}

[[nodiscard]] ParseResult parse_options(int argc, char** argv)
{
  ParseResult result;
  for (int i = 1; i < argc; ++i)
  {
    std::string_view argument(argv[i]);
    auto value_after = [&](std::string_view prefix) { return argument.substr(prefix.size()); };

    if (argument == "--help" || argument == "-h")
    {
      result.options.help = true;
    }
    else if (argument == "--no-dot")
    {
      result.options.write_dot = false;
    }
    else if (argument.starts_with("--output="))
    {
      auto const value = value_after("--output=");
      if (value.empty())
      {
        result.error = "--output requires a non-empty path";
        return result;
      }
      result.options.output = value;
    }
    else if (argument.starts_with("--width="))
    {
      std::size_t width = 0;
      if (!parse_positive_size(value_after("--width="), width))
      {
        result.error = "--width requires a positive integer";
        return result;
      }
      result.options.width = width;
    }
    else if (argument.starts_with("--stacktrace-frames="))
    {
      auto const value = value_after("--stacktrace-frames=");
      if (value == "all" || value == "unlimited" || value == "max")
      {
        result.options.stacktrace_frames = std::numeric_limits<std::size_t>::max();
      }
      else
      {
        std::size_t frames = 0;
        if (value == "0")
        {
          frames = 0;
        }
        else if (!parse_positive_size(value, frames))
        {
          result.error = "--stacktrace-frames requires 0, a positive integer, or all";
          return result;
        }
        result.options.stacktrace_frames = frames;
      }
    }
    else if (argument.starts_with("--internal-frames="))
    {
      result.options.internal_frames = parse_bool(value_after("--internal-frames="));
      if (!result.options.internal_frames)
      {
        result.error = "--internal-frames requires yes or no";
        return result;
      }
    }
    else if (argument.starts_with("--glyphs="))
    {
      auto const value = value_after("--glyphs=");
      if (value == "emoji")
        result.options.glyphs = presentation::glyph_set::emoji;
      else if (value == "unicode")
        result.options.glyphs = presentation::glyph_set::unicode;
      else if (value == "ascii")
        result.options.glyphs = presentation::glyph_set::ascii;
      else
      {
        result.error = "--glyphs requires emoji, unicode, or ascii";
        return result;
      }
    }
    else if (argument.starts_with("--charset="))
    {
      auto const value = value_after("--charset=");
      if (value == "utf8" || value == "utf-8")
        result.options.charset = presentation::text_charset::utf8;
      else if (value == "escape" || value == "ascii-escape")
        result.options.charset = presentation::text_charset::ascii_escape;
      else if (value == "replace" || value == "ascii-replace")
        result.options.charset = presentation::text_charset::ascii_replace;
      else
      {
        result.error = "--charset requires utf8, escape, or replace";
        return result;
      }
    }
    else if (argument.starts_with("--color="))
    {
      auto const value = value_after("--color=");
      if (value == "auto")
        result.options.color = presentation::color_mode::automatic;
      else if (value == "always" || value == "yes")
        result.options.color = presentation::color_mode::always;
      else if (value == "never" || value == "no")
        result.options.color = presentation::color_mode::never;
      else
      {
        result.error = "--color requires auto, always, or never";
        return result;
      }
    }
    else
    {
      result.error = fmt::format("unknown option: {}", argument);
      return result;
    }
  }
  return result;
}

[[nodiscard]] presentation::output_policy output_policy(Options const& options)
{
  auto policy = presentation::terminal_policy(stdout);
  policy.wrap_width = options.width.value_or(static_cast<std::size_t>(terminal::columns()));
  if (options.glyphs) policy.glyphs = *options.glyphs;
  if (options.charset) policy.charset = *options.charset;
  if (options.color) policy.color = *options.color;
  return policy;
}

void emit(presentation::report_builder const& report, presentation::output_policy const& policy)
{
  auto const rendered = presentation::render_terminal(report, policy, stdout);
  std::fputs(rendered.c_str(), stdout);
  std::fputc('\n', stdout);
}

[[nodiscard]] presentation::report_builder usage_report()
{
  presentation::report_builder report("Async diagnostics guide options");
  report.status(presentation::semantic_glyph::info,
                "Run without options for a complete configuration and DAG walkthrough");
  report.table("Command-line controls")
      .column("option", presentation::table_alignment::left)
      .column("effect", presentation::table_alignment::left)
      .row("--output=PATH", "Write the Graphviz snapshot to PATH")
      .row("--no-dot", "Show the guide and terminal report without writing DOT")
      .row("--width=N", "Override the presentation wrapping width for this process")
      .row("--glyphs=emoji|unicode|ascii", "Override semantic status and table glyphs")
      .row("--charset=utf8|escape|replace", "Select UTF-8 preservation or strict ASCII fallback")
      .row("--color=auto|always|never", "Override terminal color selection")
      .row("--stacktrace-frames=N|all", "Limit stack frames serialized into snapshot provenance")
      .row("--internal-frames=yes|no", "Include or hide Uni20 and standard-library implementation frames")
      .row("--help", "Show the guide without constructing a demonstration DAG");
  return report;
}

[[nodiscard]] presentation::report_builder build_report(Options const& options)
{
#if defined(NDEBUG)
  constexpr std::string_view build_assertions = "disabled (NDEBUG)";
#else
  constexpr std::string_view build_assertions = "enabled";
#endif

  presentation::report_builder report("Uni20 async diagnostics");
  report.status(presentation::semantic_glyph::info, "build capability and runtime configuration");
  report.field("compiler", compiler_name())
      .field("build assertions", build_assertions)
      .field("effective output width", options.width.value_or(static_cast<std::size_t>(terminal::columns())))
      .field("configured fallback width", UNI20_FALLBACK_TERMINAL_WIDTH);

  report.table("Build capabilities")
      .column("facility", presentation::table_alignment::left)
      .column("state", presentation::table_alignment::left)
      .column("what it provides", presentation::table_alignment::left)
      .row("Task registry", UNI20_DEBUG_ASYNC_TASKS ? "enabled" : "disabled",
           "Coroutine lifecycle, epoch state, and focused terminal reports")
      .row("DAG capture", UNI20_DEBUG_DAG ? "enabled" : "disabled",
           "Async value nodes plus argument, co_await, and live epoch edges")
      .row("std::stacktrace", UNI20_HAS_STACKTRACE ? "enabled" : "disabled",
           "Creation, scheduling, transition, and await-site provenance");
  return report;
}

[[nodiscard]] presentation::report_builder activation_report()
{
  presentation::report_builder report("Enable async diagnostics");
#if UNI20_DEBUG_DAG && UNI20_HAS_STACKTRACE
  report.status(presentation::semantic_glyph::success, "full DAG and stacktrace support is active in this build");
#else
  report.status(presentation::semantic_glyph::warning,
                "reconfigure a separate debug build to enable every diagnostic facility");
#endif

  report.table("Recommended debug build")
      .column("step", presentation::table_alignment::left)
      .column("command or requirement", presentation::table_alignment::left)
      .row("configure", "cmake -S . -B ./build_codex/build_gcc14_debug_dag -DCMAKE_BUILD_TYPE=Debug "
                        "-DCMAKE_CXX_COMPILER=g++-14 -DUNI20_DEBUG_DAG=ON -DUNI20_ENABLE_STACKTRACE=ON")
      .row("build", "cmake --build ./build_codex/build_gcc14_debug_dag --target async_diagnostics_guide_example")
      .row("run", "./build_codex/build_gcc14_debug_dag/examples/async_diagnostics_guide_example")
      .row("project compiler floor", "GCC 13 or newer, or upstream Clang 19 or newer")
      .row("stacktrace toolchain",
           "CMake compile-links std::stacktrace and any required stdc++exp library. GCC 14 or newer is the "
           "recommended starting point; other compilers require a compatible standard library.");

  report.table("CMake controls")
      .column("setting", presentation::table_alignment::left)
      .column("meaning", presentation::table_alignment::left)
      .row("UNI20_DEBUG_ASYNC_TASKS=ON", "Enable task and epoch registry instrumentation")
      .row("UNI20_DEBUG_DAG=ON", "Enable dependency capture; this also enables task instrumentation")
      .row("UNI20_ENABLE_STACKTRACE=ON", "Require the detected C++23 std::stacktrace provider")
      .row("UNI20_FALLBACK_TERMINAL_WIDTH=132",
           "Set the width used only when COLUMNS and terminal-size detection are unavailable");
  return report;
}

[[nodiscard]] presentation::report_builder customization_report()
{
  presentation::report_builder report("Customize diagnostic output");
  report.table("Presentation environment")
      .column("variable", presentation::table_alignment::left)
      .column("examples and effect", presentation::table_alignment::left)
      .row("COLUMNS", "132; overrides detected wrapping width")
      .row("UNI20_COLOR", "auto, always, never")
      .row("NO_COLOR", "disable automatic color using the common terminal convention")
      .row("UNI20_GLYPHS", "emoji, unicode, ascii")
      .row("UNI20_CHARSET", "utf8, escape, replace");

  report.table("Async diagnostics environment")
      .column("variable", presentation::table_alignment::left)
      .column("effect", presentation::table_alignment::left)
      .row("UNI20_DEBUG_ASYNC_TASKS", "none, basic, or full runtime dump selection")
      .row("UNI20_DEBUG_DAG_OUTPUT_DIR", "Directory for automatic and signal-triggered DOT files")
      .row("UNI20_DEBUG_DAG_FILE_PREFIX", "Filename prefix for generated DOT snapshots")
      .row("UNI20_DEBUG_DAG_DUMP_ON_EXCEPTION", "Capture a live registry report and DOT when a coroutine fails")
      .row("UNI20_DEBUG_DAG_SIGNAL", "Install the diagnostics service for a signal such as SIGUSR1")
      .row("UNI20_DEBUG_DAG_REQUEST_FILE", "Control file consumed by the diagnostics service")
      .row("UNI20_DEBUG_DAG_POLL_MS", "Diagnostics-service polling interval")
      .row("UNI20_DEBUG_DAG_STACKTRACE_FRAMES", "Maximum serialized frames; 0 hides trace text and all removes the cap")
      .row("UNI20_DEBUG_DAG_STACKTRACE_INTERNAL_FRAMES", "Include or hide internal implementation frames");
  return report;
}

#if UNI20_DEBUG_ASYNC_TASKS
[[nodiscard]] presentation::report_builder interpretation_report(std::filesystem::path const& output)
{
  presentation::report_builder report("Read the async DAG");
  report.table("Graph elements")
      .column("element", presentation::table_alignment::left)
      .column("meaning", presentation::table_alignment::left)
      .row("data N", "An Async<T> value and its current storage/construction state")
      .row("task N", "A coroutine task and its lifecycle state")
      .row("epoch N", "One read/write ordering generation for an async value")
      .row("arg read/write", "A coarse dependency discovered from coroutine buffer arguments")
      .row("co_await read/write", "A concrete dependency observed when the coroutine awaited")
      .row("await read/write", "A live task suspension on an epoch")
      .row("red/pink", "A missing writer or inferred dependency cycle")
      .row("yellow", "Blocked work that may still become runnable");

  report.table("Inspect the artifact")
      .column("tool", presentation::table_alignment::left)
      .column("command and behavior", presentation::table_alignment::left)
      .row("xdot", fmt::format("xdot {}; hover task, epoch, and co_await elements for provenance", output.string()))
      .row("SVG", fmt::format("dot -Tsvg {} -o async-diagnostics-guide.svg", output.string()))
      .row("terminal report",
           "The presentation report summarizes the same structured snapshot without requiring Graphviz")
      .row("exception stacktrace",
           "Watchdog and raised-error diagnostics show the active C++ call stack; the DAG explains the suspended work");
  return report;
}

async::AsyncTask triple_after_read(async::ReadBuffer<int> input, async::WriteBuffer<int> output)
{
  auto const value = co_await input;
  input.release();
  co_await output = 3 * value;
  co_return;
}

async::AsyncTask write_value(async::WriteBuffer<int> output, int value)
{
  co_await output = value;
  co_return;
}

bool write_dot(std::filesystem::path const& path, std::string_view dot)
{
  if (auto const parent = path.parent_path(); !parent.empty()) std::filesystem::create_directories(parent);
  std::ofstream output(path);
  output << dot;
  return output.good();
}

void run_demonstration(Options const& options, presentation::output_policy const& policy)
{
  async::DebugScheduler scheduler;
  async::Async<int> input;
  async::Async<int> result;
  input.debug_name("input with no writer yet");
  result.debug_name("three times input");

  auto task = triple_after_read(input.read(), result.write());
  task.debug_name("result = 3 * input");
  scheduler.schedule(std::move(task));
  scheduler.run();

  auto const snapshot = uni20::TaskRegistry::snapshot();
  auto const diagnostics = uni20::TaskRegistry::diagnose_snapshot(snapshot);
  emit(uni20::task_registry_report(snapshot, diagnostics,
                                   {.title = "Demonstration snapshot",
                                    .reason = "the reader is deliberately suspended before its writer is scheduled"}),
       policy);

  if (options.write_dot)
  {
    auto const dot = uni20::TaskRegistry::graphviz_dot(snapshot, diagnostics);
    presentation::report_builder artifact("Graphviz artifact");
    if (write_dot(options.output, dot))
      artifact.status(presentation::semantic_glyph::success, "DOT snapshot written")
          .field("path", options.output.string());
    else
      artifact.status(presentation::semantic_glyph::failure, "could not write DOT snapshot")
          .field("path", options.output.string());
    emit(artifact, policy);
  }

  auto writer = write_value(input.write(), 7);
  writer.debug_name("input = 7");
  scheduler.schedule(std::move(writer));
  scheduler.run_all();

  presentation::report_builder completion("Demonstration completion");
  completion.status(presentation::semantic_glyph::success, "the late writer released the suspended task")
      .field("result", result.get_wait(scheduler))
      .field("expected", 21);
  emit(completion, policy);
}
#endif

} // namespace

int main(int argc, char** argv)
{
  auto const parsed = parse_options(argc, argv);
  auto const policy = output_policy(parsed.options);
  if (!parsed.error.empty())
  {
    presentation::report_builder error("Async diagnostics guide");
    error.status(presentation::semantic_glyph::failure, parsed.error)
        .field("hint", "run with --help to list accepted options");
    emit(error, policy);
    return 2;
  }

  auto stacktrace_options = uni20::TaskRegistry::stacktrace_options();
  if (parsed.options.stacktrace_frames) stacktrace_options.max_frames = *parsed.options.stacktrace_frames;
  if (parsed.options.internal_frames) stacktrace_options.include_internal_frames = *parsed.options.internal_frames;
  uni20::TaskRegistry::set_stacktrace_options(stacktrace_options);

  emit(build_report(parsed.options), policy);
  emit(activation_report(), policy);
  emit(customization_report(), policy);
  emit(usage_report(), policy);

  if (parsed.options.help) return 0;

#if UNI20_DEBUG_ASYNC_TASKS
  run_demonstration(parsed.options, policy);
  if (parsed.options.write_dot) emit(interpretation_report(parsed.options.output), policy);
#else
  presentation::report_builder unavailable("Demonstration snapshot");
  unavailable.status(presentation::semantic_glyph::skipped,
                     "this build has no task-registry instrumentation; use the CMake recipe above");
  emit(unavailable, policy);
#endif
  return 0;
}
