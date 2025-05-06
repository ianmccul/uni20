#pragma once

#include "config.hpp"
#include "demangle.hpp"
#include "namedenum.hpp"
#include "terminal.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <complex>
#include <cstdio>
#include <fmt/core.h>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

// Check for stacktrace support (C++23 and GCC 13+ or Clang+libc++)
#if defined(__cpp_lib_stacktrace) && (__cpp_lib_stacktrace >= 202011L)
#define TRACE_HAS_STACKTRACE 1
#include <stacktrace>
#else
#define TRACE_HAS_STACKTRACE 0
#endif

// stringize helper (two levels needed so that macro args expand first)
#define TRACE_STRINGIZE_IMPL(x) #x
#define TRACE_STRINGIZE(x) TRACE_STRINGIZE_IMPL(x)

// COMPILER_NOTE emits an information message during compilation. msg must be a string literal.
#if defined(_MSC_VER)
// On MSVC, use __pragma without needing quotes around the entire pragma.
#define COMPILER_NOTE(msg) __pragma(message(msg))
#else
// On GCC/Clang, use the standard _Pragma operator with a string.
#define COMPILER_NOTE(msg) _Pragma(TRACE_STRINGIZE(message msg))
#endif

// COMPILER_WARNING_NOTE(msg):
//   – On GCC/Clang: emits a compile‐time warning via an unnamed lambda with a deprecation attribute.
//   – On MSVC: no effect.
#if defined(__clang__) || defined(__GNUC__)
#define COMPILER_WARNING_NOTE(msg)                                                                                     \
  do                                                                                                                   \
  {                                                                                                                    \
    ([]() __attribute__((deprecated(msg))){})();                                                                       \
  }                                                                                                                    \
  while (0)
#else
#define COMPILER_WARNING_NOTE(msg)                                                                                     \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#endif

// TRACE MACROS
// These macros forward both the stringified expression list and the evaluated
// arguments, along with file and line info, to the corresponding trace functions.
// In constexpr context there doesn't seem to be any way to make these work - the only
// option is that in constexpr context they expand to nothing
#define TRACE(...)                                                                                                     \
  do                                                                                                                   \
  {                                                                                                                    \
    if consteval                                                                                                       \
    {}                                                                                                                 \
    else                                                                                                               \
    {                                                                                                                  \
      trace::TraceCall(#__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));                                    \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#define TRACE_IF(cond, ...)                                                                                            \
  do                                                                                                                   \
  {                                                                                                                    \
    if (cond)                                                                                                          \
    {                                                                                                                  \
      if consteval                                                                                                     \
      {}                                                                                                               \
      else                                                                                                             \
      {                                                                                                                \
        trace::TraceCall(#__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));                                  \
      }                                                                                                                \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#define TRACE_MODULE(m, ...)                                                                                           \
  do                                                                                                                   \
  {                                                                                                                    \
    if constexpr (ENABLE_TRACE_##m)                                                                                    \
    {                                                                                                                  \
      if consteval                                                                                                     \
      {}                                                                                                               \
      else                                                                                                             \
      {                                                                                                                \
        trace::TraceModuleCall(#m, #__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));                        \
      }                                                                                                                \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#define TRACE_MODULE_IF(m, cond, ...)                                                                                  \
  do                                                                                                                   \
  {                                                                                                                    \
    if constexpr (ENABLE_TRACE_##m)                                                                                    \
    {                                                                                                                  \
      if (cond)                                                                                                        \
      {                                                                                                                \
        if consteval                                                                                                   \
        {}                                                                                                             \
        else                                                                                                           \
        {                                                                                                              \
          trace::TraceModuleCall(#m, #__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));                      \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

// CHECK and PRECONDITION MACROS
// These macros check a condition and, if false, print diagnostic information and abort.
// They forward additional debug information similarly to the TRACE macros.

#define CHECK(cond, ...)                                                                                               \
  do                                                                                                                   \
  {                                                                                                                    \
    if (!(cond))                                                                                                       \
    {                                                                                                                  \
      trace::CheckCall(#cond, #__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));                             \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#define CHECK_EQUAL(a, b, ...)                                                                                         \
  do                                                                                                                   \
  {                                                                                                                    \
    if (!((a) == (b)))                                                                                                 \
    {                                                                                                                  \
      trace::CheckEqualCall(#a, #b, (#a "," #b __VA_OPT__("," #__VA_ARGS__)), __FILE__, __LINE__, a,                   \
                            b __VA_OPT__(, __VA_ARGS__));                                                              \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#define PRECONDITION(cond, ...)                                                                                        \
  do                                                                                                                   \
  {                                                                                                                    \
    if (!(cond))                                                                                                       \
    {                                                                                                                  \
      trace::PreconditionCall(#cond, #__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));                      \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#define PRECONDITION_EQUAL(a, b, ...)                                                                                  \
  do                                                                                                                   \
  {                                                                                                                    \
    if (!((a) == (b)))                                                                                                 \
    {                                                                                                                  \
      trace::PreconditionEqualCall(#a, #b, (#a "," #b __VA_OPT__("," #__VA_ARGS__)), __FILE__, __LINE__, a,            \
                                   b __VA_OPT__(, __VA_ARGS__));                                                       \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

// PANIC is used to unconditionally abort
#define PANIC(...) trace::PanicCall(#__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));

// ERROR MACROS
// These macros report an error, printing debug information and then either abort
// or throw an exception based on a global configuration flag.
#define ERROR(...) trace::ErrorCall(#__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));

#define ERROR_IF(cond, ...)                                                                                            \
  do                                                                                                                   \
  {                                                                                                                    \
    if (cond)                                                                                                          \
    {                                                                                                                  \
      trace::ErrorIfCall(#cond, #__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));                           \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

// ---------------------------------------------------------------------------
// DEBUG MACROS (compile to nothing if NDEBUG is defined)
#if defined(NDEBUG)
#define DEBUG_TRACE(...)                                                                                               \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_TRACE_IF(...)                                                                                            \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_TRACE_MODULE(...)                                                                                        \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_TRACE_MODULE_IF(...)                                                                                     \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_CHECK(...)                                                                                               \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_CHECK_EQUAL(...)                                                                                         \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_PRECONDITION(...)                                                                                        \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_PRECONDITION_EQUAL(...)                                                                                  \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#else

#define DEBUG_TRACE(...) trace::DebugTraceCall(#__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));

#define DEBUG_TRACE_IF(cond, ...)                                                                                      \
  do                                                                                                                   \
  {                                                                                                                    \
    if (cond)                                                                                                          \
    {                                                                                                                  \
      trace::DebugTraceCall(#__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));                               \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#define DEBUG_TRACE_MODULE(m, ...)                                                                                     \
  do                                                                                                                   \
  {                                                                                                                    \
    if constexpr (ENABLE_TRACE_##m)                                                                                    \
    {                                                                                                                  \
      trace::DebugTraceModuleCall(#m, #__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));                     \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#define DEBUG_TRACE_MODULE_IF(m, cond, ...)                                                                            \
  do                                                                                                                   \
  {                                                                                                                    \
    if constexpr (ENABLE_TRACE_##m)                                                                                    \
    {                                                                                                                  \
      if (cond)                                                                                                        \
      {                                                                                                                \
        trace::DebugTraceModuleCall(#m, #__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));                   \
      }                                                                                                                \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#define DEBUG_CHECK(cond, ...)                                                                                         \
  do                                                                                                                   \
  {                                                                                                                    \
    if (!(cond))                                                                                                       \
    {                                                                                                                  \
      trace::DebugCheckCall(#cond, #__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));                        \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#define DEBUG_CHECK_EQUAL(a, b, ...)                                                                                   \
  do                                                                                                                   \
  {                                                                                                                    \
    if (!((a) == (b)))                                                                                                 \
    {                                                                                                                  \
      trace::DebugCheckEqualCall(#a, #b, (#a "," #b __VA_OPT__("," #__VA_ARGS__)), __FILE__, __LINE__, a,              \
                                 b __VA_OPT__(, __VA_ARGS__));                                                         \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#define DEBUG_PRECONDITION(cond, ...)                                                                                  \
  do                                                                                                                   \
  {                                                                                                                    \
    if (!(cond))                                                                                                       \
    {                                                                                                                  \
      trace::DebugPreconditionCall(#cond, #__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));                 \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#define DEBUG_PRECONDITION_EQUAL(a, b, ...)                                                                            \
  do                                                                                                                   \
  {                                                                                                                    \
    if (!((a) == (b)))                                                                                                 \
    {                                                                                                                  \
      trace::DebugPreconditionEqualCall(#a, #b, (#a "," #b __VA_OPT__("," #__VA_ARGS__)), __FILE__, __LINE__, a,       \
                                        b __VA_OPT__(, __VA_ARGS__));                                                  \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#endif

// trace namespace: Global settings for our trace facility.
namespace trace
{

struct FormattingOptions;

/// \brief get the FormattingOptions object for a given module; use "" as the module name (or default parameter) for the
/// global format
inline FormattingOptions& get_formatting_options(const std::string& module = "");

/// \brief Configuration and formatting options for a trace module.
struct FormattingOptions
{
    //--- Precision settings ---------------------------------------------------

    /// Floating-point precision for formatting float32 values.
    int fp_precision_float32 = 6;

    /// Floating-point precision for formatting float64 values.
    int fp_precision_float64 = 15;

    //--- Output layout ---------------------------------------------------------

    /// Maximum width (in characters) before switching to multi-line.
    int terminal_width = terminal::columns();

    //--- Color configuration --------------------------------------------------

    /// Enables/disables/auto color for this module (from UNI20_TRACE_COLOR or module override).
    struct ColorOptionTraits
    {
        enum Enum
        {
          yes,
          no,
          autocolor
        };
        static constexpr Enum Default = autocolor;
        static constexpr const char* StaticName = "Color options (yes/no/auto)";
        static constexpr std::array<const char*, 3> Names = {"yes", "no", "auto"};
    };

    using ColorOptions = NamedEnumeration<ColorOptionTraits>;
    ColorOptions color = terminal::getenv_or_default<ColorOptions>("UNI20_TRACE_COLOR", ColorOptions());

    /// Whether to actually emit color sequences.
    bool showColor = terminal::is_a_terminal(stderr);

    //--- Global flags ----------------------------------------------------------

    /// Abort on error if true. This is static, since it only makes sense globally
    inline static bool errorsAbort = true;

    /// Prefix each trace with a timestamp (from UNI20_TRACE_TIMESTAMP or module override).
    bool timestamp = false;

    /// Prefix each trace with a thread‐ID (from UNI20_TRACE_THREAD_ID or module override).
    bool showThreadId = false;

    //--- Output sink -----------------------------------------------------------

    /// File to which this module writes (stderr by default, or UNI20_TRACEFILE).
    FILE* outputStream = stderr;

    // The output sink is a function that takes a string and returns void
    // Default version writes to stderr
    using Sink = std::function<void(std::string)>;

    /// Function that actually emits strings (defaults to fputs to stderr).
    Sink sink = [](std::string s) { std::fputs(s.c_str(), stderr); };

    //--- Style map -------------------------------------------------------------

    /// Holds per-kind styles (keys like "TRACE", "TRACE_LINE", etc).
    mutable std::map<std::string, terminal::TerminalStyle> Styles;

    //--- Constructors ---------------------------------------------------------

    /// \brief Default constructor
    ///
    /// Seeds all built-in style keys, reads:
    ///   - UNI20_TRACEFILE for global sink
    ///   - UNI20_TRACE_TIMESTAMP
    ///   - UNI20_TRACE_THREAD_ID
    ///   - UNI20_TRACE_COLOR
    FormattingOptions()
    {
      // Default style definitions
      static constexpr std::pair<std::string_view, std::string_view> kDefaults[] = {{"TRACE", "Cyan"},
                                                                                    {"DEBUG_TRACE", "Green"},
                                                                                    {"TRACE_EXPR", "Blue"},
                                                                                    {"TRACE_VALUE", ""},
                                                                                    {"TRACE_MODULE", "Cyan;Bold"},
                                                                                    {"TRACE_FILENAME", "Red"},
                                                                                    {"TRACE_LINE", "Bold"},
                                                                                    {"TRACE_STRING", "LightBlue"},
                                                                                    {"CHECK", "Red"},
                                                                                    {"DEBUG_CHECK", "Red"},
                                                                                    {"PRECONDITION", "Red"},
                                                                                    {"DEBUG_PRECONDITION", "Red"},
                                                                                    {"PANIC", "Red"},
                                                                                    {"ERROR", "Red"},
                                                                                    {"TIMESTAMP", "LightGray"},
                                                                                    {"THREAD_ID", "LightMagenta"}};

      for (auto [kind, def] : kDefaults)
      {
        std::string env = std::string("UNI20_COLOR_") + kind;
        Styles[std::string(kind)] = terminal::getenv_or_default<terminal::TerminalStyle>(env, def);
      }

      // Global sink override
      if (auto* path = std::getenv("UNI20_TRACEFILE"))
      {
        FILE* out = nullptr;

        if (std::strcmp(path, "-") == 0 || std::strcmp(path, "stdout") == 0)
        {
          out = stdout;
        }
        else if (std::strcmp(path, "stderr") == 0)
        {
          out = stderr;
        }
        else if (std::strlen(path) > 0)
        {
          bool append = false;
          if (path[0] == '+')
          {
            ++path;
            append = true;
          }
          out = std::fopen(path, append ? "a" : "w");
        }

        if (out)
        {
          this->set_output_stream(out);
        }
      }

      // Precision
      fp_precision_float32 = terminal::getenv_or_default<int>("UNI20_FP_PRECISION_FLOAT32", 6);
      fp_precision_float64 = terminal::getenv_or_default<int>("UNI20_FP_PRECISION_FLOAT64", 15);

      // Global flags
      timestamp = terminal::getenv_or_default<terminal::toggle>("UNI20_TRACE_TIMESTAMP", true);
      showThreadId = terminal::getenv_or_default<terminal::toggle>("UNI20_TRACE_THREAD_ID", true);

      // Global color control
      color = terminal::getenv_or_default<ColorOptions>("UNI20_TRACE_COLOR", color);
      updateShowColor();
    }

    /// \brief Module‐specific constructor
    ///
    /// Delegates to the default ctor then applies only module overrides:
    ///   - UNI20_TRACEFILE_<MODULE>
    ///   - UNI20_TRACE_COLOR_MODULE_<MODULE>
    ///   - UNI20_TRACE_TIMESTAMP_MODULE_<MODULE>
    ///   - UNI20_TRACE_THREAD_ID_MODULE_<MODULE>
    FormattingOptions(std::string_view module)
    {
      *this = trace::get_formatting_options(""); // inherit all global settings

      std::string mod{module};

      // Style "TRACE" is special: the inherited version comes from "TRACE_MODULE", as a default for all modules
      Styles["TRACE"] = Styles["TRACE_MODULE"];
      // Override with _MODULE_XXXX versions from the environment, if they exist
      static constexpr std::string_view kDefaults[] = {"TRACE",        "DEBUG_TRACE",    "TRACE_EXPR",
                                                       "TRACE_VALUE",  "TRACE_FILENAME", "TRACE_LINE",
                                                       "TRACE_STRING", "TIMESTAMP",      "THREAD_ID"};

      for (auto kind : kDefaults)
      {
        std::string env = fmt::format("UNI20_COLOR_{}_MODULE_{}", kind, module);
        terminal::TerminalStyle def = Styles[std::string(kind)];
        Styles[std::string(kind)] = terminal::getenv_or_default<terminal::TerminalStyle>(env, def);
      }

      // Module sink override via UNI20_TRACEFILE_<MODULE>
      if (auto* path = std::getenv(("UNI20_TRACEFILE_MODULE_" + mod).c_str()))
      {
        FILE* out = nullptr;

        if (std::strcmp(path, "-") == 0 || std::strcmp(path, "stdout") == 0)
        {
          out = stdout;
        }
        else if (std::strcmp(path, "stderr") == 0)
        {
          out = stderr;
        }
        else if (std::strlen(path) > 0)
        {
          bool append = false;
          if (path[0] == '+')
          {
            ++path;
            append = true;
          }
          out = std::fopen(path, append ? "a" : "w");
        }

        if (out)
        {
          this->set_output_stream(out);
        }
      }

      // Precision overrides
      fp_precision_float32 =
          terminal::getenv_or_default<int>("UNI20_FP_PRECISION_FLOAT32_MODULE_" + mod, fp_precision_float32);
      fp_precision_float64 =
          terminal::getenv_or_default<int>("UNI20_FP_PRECISION_FLOAT64_MODULE_" + mod, fp_precision_float64);

      // flags overrides
      timestamp = terminal::getenv_or_default<terminal::toggle>("UNI20_TRACE_TIMESTAMP_MODULE_" + mod, timestamp);
      showThreadId = terminal::getenv_or_default<terminal::toggle>("UNI20_TRACE_THREAD_ID_MODULE_" + mod, showThreadId);

      // color override
      color = terminal::getenv_or_default<ColorOptions>("UNI20_TRACE_COLOR_MODULE_" + mod, color);
      this->updateShowColor();
    }

    //--- Public Interface ------------------------------------------------------

    /// Set a custom sink function for this module.
    void set_sink(Sink s)
    {
      sink = std::move(s);
      outputStream = nullptr;
    }

    /// Change the output FILE* for this module.
    void set_output_stream(FILE* f)
    {
      outputStream = f;
      sink = [f](std::string s) { std::fputs(s.c_str(), f); };
      updateShowColor();
    }

    /// Enable or disable color output for this module.
    void set_color_output(ColorOptions c)
    {
      color = c;
      updateShowColor();
    }

    /// Query whether color should be used in this module.
    bool should_show_color() const { return showColor; }

    /// Enable or disable abort-on-error for this module.
    static void set_errors_abort(bool b) { errorsAbort = b; }

    /// Query the abort-on-error setting.
    static bool errors_abort() { return errorsAbort; }

    /// Get or compute the terminal style for a given kind in this module.
    // terminal::TerminalStyle get_module_terminal_style(const std::string& kind, const std::string& module) const
    // {
    //   std::string key = kind + "_MODULE_" + module;
    //   if (!Styles.count(key))
    //   {
    //     std::string env_mod = "UNI20_COLOR_" + kind + "_MODULE_" + module;
    //     std::string env_glob = "UNI20_COLOR_" + kind;
    //     Styles[key] = terminal::getenv_or_default<terminal::TerminalStyle>(
    //         env_mod, terminal::getenv_or_default<terminal::TerminalStyle>(env_glob, Styles[kind]));
    //   }
    //   return Styles[key];
    // }

    /// \brief Format text using the "global" style (no module) for the given kind.
    std::string format_style(const std::string& str, const std::string& kind) const
    {
      // empty module name = global/default styles
      if (Styles.find(kind) == Styles.end())
      {
        fmt::print(stderr, "UNEXPECTED: unknown format style: {}\n", kind);
      }
      return showColor ? terminal::color_text(str, Styles[kind]) : str;
    }

    /// Format text using this module's style for the given kind.
    // std::string format_module_style(const std::string& str, const std::string& kind, const std::string& module) const
    // {
    //   return showColor ? terminal::color_text(str, get_module_terminal_style(kind, module)) : str;
    // }

  private:
    /// \brief Update showColor based on the `color` setting and current outputStream.
    void updateShowColor()
    {
      using CO = ColorOptions::Enum;
      if (color == CO::yes)
        showColor = true;
      else if (color == CO::no)
        showColor = false;
      else /* auto */
        showColor = terminal::is_a_terminal(outputStream);
    }

    friend FormattingOptions& get_formatting_options(const std::string& module);
};

/// Returns the FormattingOptions for a module, or the “default” when called with no args.
/// Empty module name ⇒ use the no-arg ctor (global defaults only).
inline FormattingOptions& get_formatting_options(const std::string& module)
{
  static std::recursive_mutex mtx;
  static std::unordered_map<std::string, FormattingOptions> table;
  std::lock_guard lock(mtx);

  if (module.empty())
  {
    // Use default-constructed instance with no recursion
    auto [it, _] = table.try_emplace("", FormattingOptions());
    return it->second;
  }
  else
  {
    // Copy from global instance
    auto [it, _] = table.try_emplace(module, module);
    return it->second;
  }
}

// Concept for a type that has a fmt::formatter specialization
template <typename T, typename CharT = char>
concept HasFormatter =
    fmt::formattable<T, CharT>; // requires(const T& t) { fmt::is_formattable<T, CharT>::value == true; };

// Formatted output of containers, if they look like a range and have no existing formatter
template <typename T>
concept Container = std::ranges::forward_range<T> && (!HasFormatter<T>);

// formatValue: Converts a value to a string using fmt::format.
// The generic version works for most types.

/// \brief Format a non-container, non-floating-point type with fmt::formatter.
/// \tparam T             Any type that has an fmt::formatter and is not a floating-point or container.
/// \param value         The value to format.
/// \param opts          Formatting options (currently unused for this overload).
/// \returns             The string produced by `fmt::format("{}", value)`.
template <typename T>
std::string formatValue(const T& value, const FormattingOptions& opts)
  requires(!Container<T> && HasFormatter<T> && !std::floating_point<T>)
{
  return fmt::format("{}", value);
}

/// \brief Format a 32-bit float using user-configured precision.
/// \param value         The float to format.
/// \param opts          Controls the precision via `opts.fp_precision_float32`.
/// \returns             A string like `"3.14"` (precision configurable).
inline std::string formatValue(float value, const FormattingOptions& opts)
{
  // use the user-configurable float32 precision
  return fmt::format("{:.{}f}", value, opts.fp_precision_float32);
}

/// \brief Format a 64-bit float using user-configured precision.
/// \param value         The double to format.
/// \param opts          Controls the precision via `opts.fp_precision_float64`.
/// \returns             A string like `"2.71828"` (precision configurable).
inline std::string formatValue(double value, const FormattingOptions& opts)
{
  // use the user-configurable float64 precision
  return fmt::format("{:.{}f}", value, opts.fp_precision_float64);
}

/// \brief Format a complex<float> as "a+bi" using the float32 precision.
/// \param value the complex value
/// \param opts   formatting options (controls precision)
/// \returns a string like "1.23+4.56i"
inline std::string formatValue(const std::complex<float>& value, const FormattingOptions& opts)
{
  // {:+.{}f} prints a leading +/-, "{:.{}f}" uses dynamic precision
  return fmt::format("{:.{}f}{:+.{}f}i", value.real(), opts.fp_precision_float32, value.imag(),
                     opts.fp_precision_float32);
}

/// \brief Format a complex<double> as "a+bi" using the float64 precision.
/// \param value the complex value
/// \param opts   formatting options (controls precision)
/// \returns a string like "1.234567+8.765432i"
inline std::string formatValue(const std::complex<double>& value, const FormattingOptions& opts)
{
  return fmt::format("{:.{}f}{:+.{}f}i", value.real(), opts.fp_precision_float64, value.imag(),
                     opts.fp_precision_float64);
}

/// \brief Format each element of a container by recursively calling `formatValue`.
/// \tparam ContainerType  Any container with `std::begin`/`std::end`.
/// \param c               The container whose elements to format.
/// \param opts            Formatting options forwarded to each element call.
/// \returns               A `std::vector<std::string>` of the formatted elements.
template <Container ContainerType>
auto formatValue(const ContainerType& c, const FormattingOptions& opts)
    -> std::vector<decltype(formatValue(*std::begin(c), opts))>
{
  std::vector<decltype(formatValue(*std::begin(c), opts))> result;
  result.reserve(std::ranges::distance(c));
  for (auto const& elem : c)
  {
    result.push_back(formatValue(elem, opts));
  }
  return result;
}

/// \brief Format a C‐string (null‐terminated) as a normal string.
/// \param s    Pointer to a null‐terminated character array.
/// \param opts Formatting options (unused here).
/// \returns    The contents of the string, or "(null)" if `s==nullptr`.
inline std::string formatValue(const char* s, const FormattingOptions& /*opts*/)
{
  if (!s) return std::string("(null)");
  return fmt::format("{}", s);
}

/// \brief Format a mutable C‐string.  Delegates to the `const char*` overload.
/// \param s    Pointer to a null‐terminated character array.
/// \param opts Formatting options.
/// \returns    The contents of the string, or "(null)" if `s==nullptr`.
inline std::string formatValue(char* s, const FormattingOptions& opts)
{
  return formatValue(static_cast<const char*>(s), opts);
}

/// \brief Format any non‐character pointer by showing its pointee type and address.
/// \tparam U  The pointee type.
/// \param ptr Pointer to format.
/// \param opts Formatting options (unused here).
/// \requires `U` is not `char` or `const char`.
/// \returns A string like `"MyType* @ 0x7fffdeadbeef"`.
template <typename U>
inline std::string formatValue(U* ptr, const FormattingOptions& /*opts*/)
  requires(!std::is_same_v<U, char> && !std::is_same_v<U, const char>)
{
  return fmt::format("{}* @ {:p}", uni20::demangle::demangle(typeid(U).name()), fmt::ptr(ptr));
}

// Helper function to trim leading and trailing whitespace.
inline std::string trim(const std::string& s)
{
  // Find the first non-space character.
  size_t start = 0;
  while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start])))
    ++start;
  if (start == s.size())
  {
    return "";
  }
  // Find the last non-space character.
  size_t end = s.size() - 1;
  while (end > start && std::isspace(static_cast<unsigned char>(s[end])))
    --end;
  return s.substr(start, end - start + 1);
}

// parseNames: Splits the stringified parameter list into tokens.
// Each token is paired with a boolean flag that is true if a top-level
// string or character literal was encountered.
inline std::vector<std::pair<std::string, bool>> parseNames(std::string_view s)
{
  std::vector<std::pair<std::string, bool>> tokens;
  std::string current;
  bool tokenHasTopLevelLiteral = false;

  // Counters for grouping symbols.
  int parenCount = 0, squareCount = 0, curlyCount = 0;

  // Flags for being inside a literal.
  bool inDoubleQuote = false, inSingleQuote = false;

  for (size_t i = 0; i < s.size(); ++i)
  {
    char c = s[i];

    // Inside a double-quoted string literal.
    if (inDoubleQuote)
    {
      current.push_back(c);
      if (c == '\\')
      {
        // Copy the escaped character.
        if (i + 1 < s.size())
        {
          current.push_back(s[i + 1]);
          ++i;
        }
      }
      else if (c == '"')
      {
        inDoubleQuote = false;
      }
      continue;
    }

    // Inside a single-quoted character literal.
    if (inSingleQuote)
    {
      current.push_back(c);
      if (c == '\\')
      {
        if (i + 1 < s.size())
        {
          current.push_back(s[i + 1]);
          ++i;
        }
      }
      else if (c == '\'')
      {
        inSingleQuote = false;
      }
      continue;
    }

    // At top-level, if we see a literal-start, mark the token.
    if ((c == '"' || c == '\'') && parenCount == 0 && squareCount == 0 && curlyCount == 0)
    {
      tokenHasTopLevelLiteral = true;
      if (c == '"')
        inDoubleQuote = true;
      else
        inSingleQuote = true;
      current.push_back(c);
      continue;
    }

    // Split at commas if we are not inside any grouping.
    if (c == ',' && parenCount == 0 && squareCount == 0 && curlyCount == 0)
    {
      tokens.push_back({trim(current), tokenHasTopLevelLiteral});
      current.clear();
      tokenHasTopLevelLiteral = false;
      continue; // Skip adding the comma.
    }

    // Update grouping counters.
    if (c == '(')
      ++parenCount;
    else if (c == ')')
    {
      if (parenCount > 0) --parenCount;
    }
    else if (c == '[')
      ++squareCount;
    else if (c == ']')
    {
      if (squareCount > 0) --squareCount;
    }
    else if (c == '{')
      ++curlyCount;
    else if (c == '}')
    {
      if (curlyCount > 0) --curlyCount;
    }

    current.push_back(c);
  }

  // Push any remaining text as the final token.
  if (!current.empty()) tokens.push_back({trim(current), tokenHasTopLevelLiteral});

  return tokens;
}

// returns true if str spans more than one line
inline bool isMultiline(const std::string& str) { return std::find(str.begin(), str.end(), '\n') != str.end(); }

template <typename T> bool isMultiline(const std::vector<T>& str)
{
  for (const auto& s : str)
  {
    if (isMultiline(s)) return true;
  }
  return false;
}

// get the maximum length of a line for a multi-line string
inline int getMaxLineWidth(const std::string& str)
{
  int maxWidth = 0;
  int currentWidth = 0;
  for (char ch : str)
  {
    if (ch == '\n')
    {
      maxWidth = std::max(maxWidth, currentWidth);
      currentWidth = 0;
    }
    else
    {
      ++currentWidth;
    }
  }
  // Check the last line (in case the string doesn't end with a newline)
  maxWidth = std::max(maxWidth, currentWidth);
  return maxWidth;
}

// Indent the second and subsequent lines of a multi-line string
inline std::string indentMultiline(const std::string& str, int indentSpaces)
{
  std::string result;
  result.reserve(str.size());

  for (auto ch : str)
  {
    result.push_back(ch);
    if (ch == '\n')
    {
      result.append(std::string(indentSpaces, ' '));
    }
  }
  return result;
}

// recursively get the maximum width of a string or nested vectors of strings.
inline size_t maxWidth(const std::string& str) { return str.size(); }

template <typename T> size_t maxWidth(const std::vector<T>& elems)
{
  size_t max_width = 0;
  for (const auto& e : elems)
    max_width = std::max(max_width, maxWidth(e));
  return max_width;
}

inline std::string formatContainerToString(const std::string& elem, size_t max_width = 0)
{
  return fmt::format("{:>{}}", elem, max_width);
}

inline std::string formatContainerToString(const std::vector<std::string>& elems, size_t max_width = 0)
{
  std::string inlineStr;

  // The multiline case; put line breaks between each element
  if (isMultiline(elems))
  {
    inlineStr = "[\n";
    for (int i = 0; i < std::ssize(elems); ++i)
    {
      if (i > 0) inlineStr += ",\n  ";
      inlineStr += indentMultiline(elems[i], 2);
    }
    inlineStr += "\n]";
  }
  else
  {
    // Get the maximum width of the components
    max_width = std::max(max_width, maxWidth(elems));
    inlineStr = "[ ";
    for (int i = 0; i < std::ssize(elems); ++i)
    {
      if (i > 0) inlineStr += ", ";
      inlineStr += formatContainerToString(elems[i], max_width);
    }
    inlineStr += " ]";
  }
  return inlineStr;
}

// This probably works fine for arbitrary nesting
inline std::string formatContainerToString(const std::vector<std::vector<std::string>>& elems, size_t max_width = 0)
{
  std::string inlineStr = "[ ";
  if (isMultiline(elems))
  {
    for (int i = 0; i < std::ssize(elems); ++i)
    {
      if (i > 0) inlineStr += ",\n  ";
      inlineStr += indentMultiline(formatContainerToString(elems[i]), 2);
    }
    inlineStr += "\n]";
  }
  else
  {
    // Get the maximum width of the components
    max_width = std::max(max_width, maxWidth(elems));
    for (size_t i = 0; i < elems.size(); ++i)
    {
      if (i > 0) inlineStr += ",\n  ";
      inlineStr += formatContainerToString(elems[i], max_width);
    }
    inlineStr += " ]";
  }
  return inlineStr;
}

// For non-container (plain string) output.
inline std::string formatItemString(const std::pair<std::string, bool>& name, const std::string& value,
                                    const FormattingOptions& opts, int available_width)
{
  // If the name is a string literal, then just return the value separately
  if (name.second) return opts.format_style(value, "TRACE_STRING");

  // If we'd spill over onto another line, then insert a line break
  if (isMultiline(value)) // || (getMaxLineWidth(value) + std::ssize(name.first) + 3 > available_width))
  {
    int indent = name.first.size() + 3;
    return "\n" + opts.format_style(name.first, "TRACE_EXPR") + " = " +
           opts.format_style(indentMultiline(value, indent), "TRACE_VALUE");
  }
  return fmt::format("{} = {}", opts.format_style(name.first, "TRACE_EXPR"), opts.format_style(value, "TRACE_VALUE"));
}

// Containers (can be nested)
template <typename T>
std::string formatItemString(const std::pair<std::string, bool>& name, const std::vector<T>& values,
                             const FormattingOptions& opts, int available_width)
{
  std::string formatted = formatContainerToString(values);
  return formatItemString(name, formatted, opts, available_width);
}

// formatTrace: Recursively formats all trace items into one string.

inline std::string formatParameters(std::vector<std::pair<std::string, bool>>::const_iterator b,
                                    const FormattingOptions& opts)
{
  return std::string();
}

template <typename T, typename... Ts>
std::string formatParameters(std::vector<std::pair<std::string, bool>>::const_iterator b, const FormattingOptions& opts,
                             const T& first, const Ts&... rest)
{

  std::string result = formatItemString(*b, formatValue(first, opts), opts, 80);

  ++b;
  if constexpr (sizeof...(rest) > 0)
  {
    result += ", ";
    result += formatParameters(b, opts, rest...);
  }
  return result;
}

template <typename... Args>
std::string formatParameterList(const char* exprList, const FormattingOptions& opts, const Args&... args)
{
  auto names = parseNames(exprList);
  return formatParameters(names.begin(), opts, args...);
}

/// \brief Print formatted string via the default (empty-module) sink.
template <typename... Args> void print(fmt::format_string<Args...> fmt_str, Args&&... args)
{
  auto& opts = get_formatting_options(); // empty-module defaults
  opts.sink(fmt::format(fmt_str, std::forward<Args>(args)...));
}

//-----------------------------------------------------------------------------
// Non-module TRACE
//-----------------------------------------------------------------------------
template <typename... Args> void TraceCall(const char* exprList, const char* file, int line, const Args&... args)
{
  auto& opts = get_formatting_options();

  // format argument list
  std::string trace_str = formatParameterList(exprList, opts, args...);

  // optional timestamp
  std::string ts;
  if (opts.timestamp)
  {
    auto now = std::chrono::system_clock::now();
    auto us_since_epoch = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(us_since_epoch);
    auto micros = us_since_epoch - seconds;

    std::time_t t = seconds.count();
    std::tm tm = *std::localtime(&t);
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%F %T", &tm); // e.g. "2025-05-01 17:42:03"

    ts = fmt::format("[{}.{:06}] ", buffer, micros.count());
    ts = opts.format_style(ts, "TIMESTAMP");
  }

  // optional thread-ID
  std::string th;
  if (opts.showThreadId)
  {
    auto id = std::hash<std::thread::id>{}(std::this_thread::get_id());
    th = fmt::format("[TID {:>8x}] ", id);
    th = opts.format_style(th, "THREAD_ID");
  }

  // build preamble
  std::string pre = opts.format_style("TRACE", "TRACE") + " at " + opts.format_style(file, "TRACE_FILENAME") +
                    opts.format_style(fmt::format(":{}", line), "TRACE_LINE");

  // emit
  opts.sink(ts + th + fmt::format("{}{}{}\n", pre, trace_str.empty() ? "" : " : ", trace_str));
}

//-----------------------------------------------------------------------------
// Module-aware TRACE
//-----------------------------------------------------------------------------
template <typename... Args>
void TraceModuleCall(const char* module, const char* exprList, const char* file, int line, const Args&... args)
{
  auto& opts = get_formatting_options(module);

  std::string trace_str = formatParameterList(exprList, opts, args...);

  std::string ts;
  if (opts.timestamp)
  {
    auto now = std::chrono::system_clock::now();
    auto us_since_epoch = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(us_since_epoch);
    auto micros = us_since_epoch - seconds;

    std::time_t t = seconds.count();
    std::tm tm = *std::localtime(&t);
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%F %T", &tm); // e.g. "2025-05-01 17:42:03"

    ts = fmt::format("[{}.{:06}] ", buffer, micros.count());
    ts = opts.format_style(ts, "TIMESTAMP");
  }

  std::string th;
  if (opts.showThreadId)
  {
    auto id = std::hash<std::thread::id>{}(std::this_thread::get_id());
    th = fmt::format("[TID {:>8x}] ", id);
    th = opts.format_style(th, "THREAD_ID");
  }

  std::string pre = opts.format_style("TRACE", "TRACE") + " in module " + opts.format_style(module, "TRACE") + " at " +
                    opts.format_style(file, "TRACE_FILENAME") +
                    opts.format_style(fmt::format(":{}", line), "TRACE_LINE");

  opts.sink(ts + th + fmt::format("{}{}{}\n", pre, trace_str.empty() ? "" : " : ", trace_str));
}

//-----------------------------------------------------------------------------
// Non-module DEBUG_TRACE
//-----------------------------------------------------------------------------
template <typename... Args> void DebugTraceCall(const char* exprList, const char* file, int line, const Args&... args)
{
  auto& opts = get_formatting_options();

  std::string trace_str = formatParameterList(exprList, opts, args...);

  std::string ts;
  if (opts.timestamp)
  {
    auto now = std::chrono::system_clock::now();
    auto us_since_epoch = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(us_since_epoch);
    auto micros = us_since_epoch - seconds;

    std::time_t t = seconds.count();
    std::tm tm = *std::localtime(&t);
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%F %T", &tm); // e.g. "2025-05-01 17:42:03"

    ts = fmt::format("[{}.{:06}] ", buffer, micros.count());
    ts = opts.format_style(ts, "TIMESTAMP");
  }

  std::string th;
  if (opts.showThreadId)
  {
    auto id = std::hash<std::thread::id>{}(std::this_thread::get_id());
    th = fmt::format("[TID {:>8x}] ", id);
    th = opts.format_style(th, "THREAD_ID");
  }

  std::string pre = opts.format_style("DEBUG_TRACE", "DEBUG_TRACE") + " at " +
                    opts.format_style(file, "TRACE_FILENAME") +
                    opts.format_style(fmt::format(":{}", line), "TRACE_LINE");

  opts.sink(ts + th + fmt::format("{}{}{}\n", pre, trace_str.empty() ? "" : " : ", trace_str));
}

//-----------------------------------------------------------------------------
// Module-aware DEBUG_TRACE
//-----------------------------------------------------------------------------
template <typename... Args>
void DebugTraceModuleCall(const char* module, const char* exprList, const char* file, int line, const Args&... args)
{
  auto& opts = get_formatting_options(module);

  std::string trace_str = formatParameterList(exprList, opts, args...);

  std::string ts;
  if (opts.timestamp)
  {
    auto now = std::chrono::system_clock::now();
    auto us_since_epoch = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(us_since_epoch);
    auto micros = us_since_epoch - seconds;

    std::time_t t = seconds.count();
    std::tm tm = *std::localtime(&t);
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%F %T", &tm); // e.g. "2025-05-01 17:42:03"

    ts = fmt::format("[{}.{:06}] ", buffer, micros.count());
    ts = opts.format_style(ts, "TIMESTAMP");
  }

  std::string th;
  if (opts.showThreadId)
  {
    auto id = std::hash<std::thread::id>{}(std::this_thread::get_id());
    th = fmt::format("[TID {:>8x}] ", id);
    th = opts.format_style(th, "THREAD_ID");
  }

  std::string pre = opts.format_style("DEBUG_TRACE", "DEBUG_TRACE") + " in module " +
                    opts.format_style(module, module) + " at " + opts.format_style(file, "TRACE_FILENAME") +
                    opts.format_style(fmt::format(":{}", line), "TRACE_LINE");

  opts.sink(ts + th + fmt::format("{}{}{}\n", pre, trace_str.empty() ? "" : " : ", trace_str));
}

template <typename... Args>
void CheckCall(const char* cond, const char* exprList, const char* file, int line, const Args&... args)
{
  auto& opts = get_formatting_options();

  std::string trace_str = formatParameterList(exprList, opts, args...);

  std::string preamble =
      opts.format_style("CHECK", "CHECK") + " at " + opts.format_style(file, "TRACE_FILENAME") +
      opts.format_style(fmt::format(":{}", line), "TRACE_LINE") +
      fmt::format("\n{} is {}!", opts.format_style(cond, "TRACE_EXPR"), opts.format_style("false", "TRACE_VALUE"));

  print("{}{}{}\n", preamble, trace_str.empty() ? "" : "\n : ", trace_str);
  std::fflush(nullptr); // flush all output streams
  std::abort();
}

//------------------------------------------------------------------------------
// DEBUG_CHECK
//------------------------------------------------------------------------------
template <typename... Args>
void DebugCheckCall(const char* cond, const char* exprList, const char* file, int line, const Args&... args)
{
  auto& opts = get_formatting_options();

  std::string trace_str = formatParameterList(exprList, opts, args...);

  std::string preamble =
      opts.format_style("DEBUG_CHECK", "DEBUG_CHECK") + " at " + opts.format_style(file, "TRACE_FILENAME") +
      opts.format_style(fmt::format(":{}", line), "TRACE_LINE") +
      fmt::format("\n{} is {}!", opts.format_style(cond, "TRACE_EXPR"), opts.format_style("false", "TRACE_VALUE"));

  print("{}{}{}\n", preamble, trace_str.empty() ? "" : "\n : ", trace_str);
  std::fflush(nullptr); // flush all output streams
  std::abort();
}

//------------------------------------------------------------------------------
// CHECK_EQUAL
//------------------------------------------------------------------------------
template <typename... Args>
void CheckEqualCall(const char* a, const char* b, const char* exprList, const char* file, int line, const Args&... args)
{
  auto& opts = get_formatting_options();

  std::string trace_str = formatParameterList(exprList, opts, args...);

  std::string preamble =
      opts.format_style("CHECK_EQUAL", "CHECK") + " at " + opts.format_style(file, "TRACE_FILENAME") +
      opts.format_style(fmt::format(":{}", line), "TRACE_LINE") +
      fmt::format("\n{} is not equal to {}!", opts.format_style(a, "TRACE_EXPR"), opts.format_style(b, "TRACE_EXPR"));

  print("{}{}{}\n", preamble, trace_str.empty() ? "" : "\n : ", trace_str);
  std::fflush(nullptr); // flush all output streams
  std::abort();
}

//------------------------------------------------------------------------------
// DEBUG_CHECK_EQUAL
//------------------------------------------------------------------------------
template <typename... Args>
void DebugCheckEqualCall(const char* a, const char* b, const char* exprList, const char* file, int line,
                         const Args&... args)
{
  auto& opts = get_formatting_options();

  std::string trace_str = formatParameterList(exprList, opts, args...);

  std::string preamble =
      opts.format_style("DEBUG_CHECK_EQUAL", "DEBUG_CHECK") + " at " + opts.format_style(file, "TRACE_FILENAME") +
      opts.format_style(fmt::format(":{}", line), "TRACE_LINE") +
      fmt::format("\n{} is not equal to {}!", opts.format_style(a, "TRACE_EXPR"), opts.format_style(b, "TRACE_EXPR"));

  print("{}{}{}\n", preamble, trace_str.empty() ? "" : "\n : ", trace_str);
  std::fflush(nullptr); // flush all output streams
  std::abort();
}

//------------------------------------------------------------------------------
// PRECONDITION
//------------------------------------------------------------------------------
template <typename... Args>
void PreconditionCall(const char* cond, const char* exprList, const char* file, int line, const Args&... args)
{
  auto& opts = get_formatting_options();

  std::string trace_str = formatParameterList(exprList, opts, args...);

  std::string preamble =
      opts.format_style("PRECONDITION", "PRECONDITION") + " at " + opts.format_style(file, "TRACE_FILENAME") +
      opts.format_style(fmt::format(":{}", line), "TRACE_LINE") +
      fmt::format("\n{} is {}!", opts.format_style(cond, "TRACE_EXPR"), opts.format_style("false", "TRACE_VALUE"));

  print("{}{}{}\n", preamble, trace_str.empty() ? "" : "\n : ", trace_str);
  std::fflush(nullptr); // flush all output streams
  std::abort();
}

//------------------------------------------------------------------------------
// DEBUG_PRECONDITION
//------------------------------------------------------------------------------
template <typename... Args>
void DebugPreconditionCall(const char* cond, const char* exprList, const char* file, int line, const Args&... args)
{
  auto& opts = get_formatting_options();

  std::string trace_str = formatParameterList(exprList, opts, args...);

  std::string preamble =
      opts.format_style("DEBUG_PRECONDITION", "DEBUG_PRECONDITION") + " at " +
      opts.format_style(file, "TRACE_FILENAME") + opts.format_style(fmt::format(":{}", line), "TRACE_LINE") +
      fmt::format("\n{} is {}!", opts.format_style(cond, "TRACE_EXPR"), opts.format_style("false", "TRACE_VALUE"));

  print("{}{}{}\n", preamble, trace_str.empty() ? "" : "\n : ", trace_str);
  std::fflush(nullptr); // flush all output streams
  std::abort();
}

//------------------------------------------------------------------------------
// PRECONDITION_EQUAL
//------------------------------------------------------------------------------
template <typename... Args>
void PreconditionEqualCall(const char* a, const char* b, const char* exprList, const char* file, int line,
                           const Args&... args)
{
  auto& opts = get_formatting_options();

  std::string trace_str = formatParameterList(exprList, opts, args...);

  std::string preamble =
      opts.format_style("PRECONDITION_EQUAL", "PRECONDITION") + " at " + opts.format_style(file, "TRACE_FILENAME") +
      opts.format_style(fmt::format(":{}", line), "TRACE_LINE") +
      fmt::format("\n{} is not equal to {}!", opts.format_style(a, "TRACE_EXPR"), opts.format_style(b, "TRACE_EXPR"));

  print("{}{}{}\n", preamble, trace_str.empty() ? "" : "\n : ", trace_str);
  std::fflush(nullptr); // flush all output streams
  std::abort();
}

//------------------------------------------------------------------------------
// DEBUG_PRECONDITION_EQUAL
//------------------------------------------------------------------------------
template <typename... Args>
void DebugPreconditionEqualCall(const char* a, const char* b, const char* exprList, const char* file, int line,
                                const Args&... args)
{
  auto& opts = get_formatting_options();

  std::string trace_str = formatParameterList(exprList, opts, args...);

  std::string preamble =
      opts.format_style("DEBUG_PRECONDITION_EQUAL", "DEBUG_PRECONDITION") + " at " +
      opts.format_style(file, "TRACE_FILENAME") + opts.format_style(fmt::format(":{}", line), "TRACE_LINE") +
      fmt::format("\n{} is not equal to {}!", opts.format_style(a, "TRACE_EXPR"), opts.format_style(b, "TRACE_EXPR"));

  print("{}{}{}\n", preamble, trace_str.empty() ? "" : "\n : ", trace_str);
  std::fflush(nullptr); // flush all output streams
  std::abort();
}

//------------------------------------------------------------------------------
// PANIC
//------------------------------------------------------------------------------
template <typename... Args> void PanicCall(const char* exprList, const char* file, int line, const Args&... args)
{
  auto& opts = get_formatting_options();

  std::string trace_str = formatParameterList(exprList, opts, args...);

  std::string preamble = opts.format_style("PANIC", "PANIC") + " at " + opts.format_style(file, "TRACE_FILENAME") +
                         opts.format_style(fmt::format(":{}", line), "TRACE_LINE");

  opts.sink(preamble + (trace_str.empty() ? "" : " : " + trace_str) + "\n");

#if TRACE_HAS_STACKTRACE
  opts.sink(fmt::format("{}\n", opts.format_style("Stacktrace:\n", "PANIC")));
  opts.sink(fmt::format("{}\n", std::stacktrace::current()));
#else
  opts.sink("Stacktrace not available (compiler too old)\n");
#endif

  std::fflush(nullptr); // flush all output streams
  std::abort();
}

//------------------------------------------------------------------------------
// ERROR
//------------------------------------------------------------------------------
template <typename... Args> void ErrorCall(const char* exprList, const char* file, int line, const Args&... args)
{
  auto& opts = get_formatting_options();

  std::string trace_str = formatParameterList(exprList, opts, args...);

  std::string preamble = opts.format_style("ERROR", "ERROR") + " at " + opts.format_style(file, "TRACE_FILENAME") +
                         opts.format_style(fmt::format(":{}", line), "TRACE_LINE");

  std::string msg = preamble + (trace_str.empty() ? "" : " : " + trace_str) + "\n";

  if (opts.errors_abort())
  {
    opts.sink(msg);
    std::fflush(nullptr); // flush all output streams
    std::abort();
  }
  throw std::runtime_error(msg);
}

//------------------------------------------------------------------------------
// ERROR_IF
//------------------------------------------------------------------------------
template <typename... Args>
void ErrorIfCall(const char* cond, const char* exprList, const char* file, int line, const Args&... args)
{
  auto& opts = get_formatting_options();

  std::string trace_str = formatParameterList(exprList, opts, args...);

  std::string preamble = opts.format_style("ERROR", "ERROR") + " at " + opts.format_style(file, "TRACE_FILENAME") +
                         opts.format_style(fmt::format(":{}", line), "TRACE_LINE");

  std::string fail_msg =
      fmt::format("\n{} is {}!", opts.format_style(cond, "TRACE_EXPR"), opts.format_style("false", "TRACE_VALUE"));

  std::string msg = preamble + (trace_str.empty() ? "" : " : " + trace_str) + fail_msg + "\n";

  if (opts.errors_abort())
  {
    opts.sink(msg);
    std::fflush(nullptr); // flush all output streams
    std::abort();
  }
  throw std::runtime_error(msg);
}

} // namespace trace
