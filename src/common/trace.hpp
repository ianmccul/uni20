#pragma once

#include "config.hpp"
#include "namedenum.hpp"
#include "terminal.hpp"

#include <algorithm>
#include <cctype>
#include <fmt/core.h>
// #include <iostream>
#include <cstdio>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

// Check for stacktrace support (C++23 and GCC 13+ or Clang+libc++)
#if defined(__cpp_lib_stacktrace) && (__cpp_lib_stacktrace >= 202011L)
#define UNI20_HAS_STACKTRACE 1
#include <stacktrace>
#else
#define UNI20_HAS_STACKTRACE 0
#endif

// TRACE MACROS
// These macros forward both the stringified expression list and the evaluated
// arguments, along with file and line info, to the corresponding trace functions.
// __VA_OPT__ is used to conditionally include the comma when no extra arguments are provided.

#define TRACE(...) trace::TraceCall(#__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));

#define TRACE_IF(cond, ...)                                                                                            \
  do                                                                                                                   \
  {                                                                                                                    \
    if (cond)                                                                                                          \
    {                                                                                                                  \
      trace::TraceCall(#__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));                                    \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

#define TRACE_MODULE(m, ...)                                                                                           \
  do                                                                                                                   \
  {                                                                                                                    \
    if constexpr (ENABLE_TRACE_##m)                                                                                    \
    {                                                                                                                  \
      trace::TraceModuleCall(#m, #__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));                          \
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
        trace::TraceModuleCall(#m, #__VA_ARGS__, __FILE__, __LINE__ __VA_OPT__(, __VA_ARGS__));                        \
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

//-------------------------------------------------------------------------
// Global formatting options

// FormattingOptions: Holds configuration for formatted output.
struct FormattingOptions
{
    template <typename T> inline static int fp_precision = 6;

    inline static int terminal_width =
        terminal::columns(); // Maximum width (in characters) before switching to multi-line.

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

    // The output sink is a function that takes a string and returns void
    // Default version writes to stderr
    using Sink = std::function<void(std::string)>;

    // Change the sink function. This assumes that the output is not a terminal, auto color is disabled
    static void set_sink(Sink s)
    {
      sink = s;
      outputStream = nullptr;
    }

    // set the output stream as a FILE*
    static void set_output_stream(FILE* outputStream);

    // Write str to the sink
    static void emit_string(std::string const& str) { sink(str); }

    static void set_color_output(ColorOptions c);

    static bool should_show_color() { return showColor; }

    static void set_errors_abort(bool b) { errorsAbort = b; }
    static bool errors_abort() { return errorsAbort; }

    static void set_terminal_style(const std::string& Item, const terminal::TerminalStyle& style)
    {
      Styles[Item] = style;
    }

    static terminal::TerminalStyle get_terminal_style(const std::string& Item)
    {
      if (!Styles.count(Item))
        Styles[Item] = terminal::getenv_or_default<terminal::TerminalStyle>("UNI20_COLOR_" + Item, "");
      return Styles[Item];
    }

    static terminal::TerminalStyle get_module_terminal_style(const std::string& Module)
    {
      if (!Styles.count(Module))
        Styles[Module] =
            terminal::getenv_or_default("UNI20_COLOR_MODULE_" + Module, get_terminal_style("TRACE_MODULE"));
      return Styles[Module];
    }

    static std::string format_style(const std::string& Str, const std::string& Style)
    {
      return should_show_color() ? terminal::color_text(Str, get_terminal_style(Style)) : Str;
    }

    static std::string format_module_style(const std::string& Str, const std::string& Style)
    {
      return should_show_color() ? terminal::color_text(Str, get_module_terminal_style(Style)) : Str;
    }

  private:
    inline static FILE* outputStream = stderr;
    inline static Sink sink = [](std::string str) { std::fputs(str.c_str(), stderr); };
    inline static ColorOptions color = terminal::getenv_or_default("UNI20_COLOR", ColorOptions());
    inline static bool showColor = terminal::is_a_terminal(stderr);
    inline static bool errorsAbort = true;

    inline static std::map<std::string, terminal::TerminalStyle> Styles = {
        {"TRACE", terminal::getenv_or_default<terminal::TerminalStyle>("UNI20_COLOR_TRACE", "Cyan")},
        {"DEBUG_TRACE", terminal::getenv_or_default<terminal::TerminalStyle>("UNI20_COLOR_DEBUG_TRACE", "Green")},
        {"TRACE_EXPR", terminal::getenv_or_default<terminal::TerminalStyle>("UNI20_COLOR_TRACE_EXPR", "Blue")},
        {"TRACE_VALUE", terminal::getenv_or_default<terminal::TerminalStyle>("UNI20_COLOR_TRACE_VALUE", "")},
        {"TRACE_MODULE", terminal::getenv_or_default<terminal::TerminalStyle>("UNI20_COLOR_TRACE_MODULE", "LightCyan")},
        {"TRACE_FILENAME", terminal::getenv_or_default<terminal::TerminalStyle>("UNI20_COLOR_TRACE_FILENAME", "Red")},
        {"TRACE_LINE", terminal::getenv_or_default<terminal::TerminalStyle>("UNI20_COLOR_TRACE_LINE", "Bold")},
        {"TRACE_STRING", terminal::getenv_or_default<terminal::TerminalStyle>("UNI20_COLOR_TRACE_STRING", "LightBlue")},
        {"CHECK", terminal::getenv_or_default<terminal::TerminalStyle>("UNI20_COLOR_CHECK", "Red")},
        {"DEBUG_CHECK", terminal::getenv_or_default<terminal::TerminalStyle>("UNI20_COLOR_DEBUG_CHECK", "Red")},
        {"PRECONDITION", terminal::getenv_or_default<terminal::TerminalStyle>("UNI20_COLOR_PRECONDITION", "Red")},
        {"DEBUG_PRECONDITION",
         terminal::getenv_or_default<terminal::TerminalStyle>("UNI20_COLOR_DEBUG_PRECONDITION", "Red")},
        {"PANIC", terminal::getenv_or_default<terminal::TerminalStyle>("UNI20_COLOR_PANIC", "Red")},
        {"ERROR", terminal::getenv_or_default<terminal::TerminalStyle>("UNI20_COLOR_ERROR", "Red")},
    };

    static void updateShowColor();
};

inline void FormattingOptions::set_output_stream(FILE* stream)
{
  FormattingOptions::outputStream = stream;
  FormattingOptions::sink = [stream](std::string str) { std::fputs(str.c_str(), stream); };
  FormattingOptions::updateShowColor();
}

inline void FormattingOptions::set_color_output(ColorOptions c)
{
  color = c;
  FormattingOptions::updateShowColor();
}

inline void FormattingOptions::updateShowColor()
{
  if (color == ColorOptions::yes) showColor = true;
  if (color == ColorOptions::no) showColor = false;
  if (color == ColorOptions::autocolor) showColor = terminal::is_a_terminal(outputStream);
}

template <> inline int FormattingOptions::fp_precision<float> = 6;

template <> inline int FormattingOptions::fp_precision<double> = 15;

template <> inline int FormattingOptions::fp_precision<long double> = 15;

inline FormattingOptions formatting_options;

// Concept for a type that has a fmt::formatter specialization
template <typename T, typename CharT = char>
concept HasFormatter =
    fmt::formattable<T, CharT>; // requires(const T& t) { fmt::is_formattable<T, CharT>::value == true; };

// Formatted output of containers, if they look like a range and have no existing formatter
template <typename T>
concept Container = std::ranges::forward_range<T> && (!HasFormatter<T>);

// formatValue: Converts a value to a string using fmt::format.
// The generic version works for most types.
template <typename T>
std::string formatValue(const T& value, const FormattingOptions& opts)
  requires((!Container<T>) && HasFormatter<T> && !std::floating_point<T>)
{
  return fmt::format("{}", value);
}

template <typename T>
std::string formatValue(const T& value, const FormattingOptions& opts)
  requires(!Container<T> && HasFormatter<T> && std::floating_point<T>)
{
  int precision = FormattingOptions::fp_precision<T>;
  return fmt::format("{:.{}f}", value, precision);
}

template <Container ContainerType>
auto formatValue(const ContainerType& c, const FormattingOptions& opts)
    -> std::vector<decltype(formatValue(*std::begin(c), opts))>
{
  using ElementFormattedType = decltype(formatValue(*std::begin(c), opts));
  std::vector<ElementFormattedType> result;
  for (const auto& elem : c)
  {
    result.push_back(formatValue(elem, opts));
  }
  return result;
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

template <typename... Args> void print(fmt::format_string<Args...> fmt_str, Args&&... args)
{
  // build the string
  auto s = fmt::format(fmt_str, std::forward<Args>(args)...);
  // write it to the sink
  trace::formatting_options.emit_string(s);
}

template <typename... Args> void TraceCall(const char* exprList, const char* file, int line, const Args&... args)
{
  std::string trace_str = formatParameterList(exprList, trace::formatting_options, args...);
  std::string preamble = trace::formatting_options.format_style("TRACE", "TRACE") + " at " +
                         trace::formatting_options.format_style(file, "TRACE_FILENAME") +
                         trace::formatting_options.format_style(fmt::format(":{}", line), "TRACE_LINE");
  print("{}{}{}\n", preamble, trace_str.empty() ? "" : " : ", trace_str);
}

template <typename... Args>
void TraceModuleCall(const char* module, const char* exprList, const char* file, int line, const Args&... args)
{
  std::string trace_str = formatParameterList(exprList, trace::formatting_options, args...);
  std::string preamble = trace::formatting_options.format_module_style("TRACE", module) + " in module " +
                         trace::formatting_options.format_module_style(module, module) + " at " +
                         trace::formatting_options.format_style(file, "TRACE_FILENAME") +
                         trace::formatting_options.format_style(fmt::format(":{}", line), "TRACE_LINE");
  print("{}{}{}\n", preamble, trace_str.empty() ? "" : " : ", trace_str);
}

template <typename... Args> void DebugTraceCall(const char* exprList, const char* file, int line, const Args&... args)
{
  std::string trace_str = formatParameterList(exprList, trace::formatting_options, args...);
  std::string preamble = trace::formatting_options.format_style("DEBUG_TRACE", "DEBUG_TRACE") + " at " +
                         trace::formatting_options.format_style(file, "TRACE_FILENAME") +
                         trace::formatting_options.format_style(fmt::format(":{}", line), "TRACE_LINE");
  print("{}{}{}\n", preamble, trace_str.empty() ? "" : " : ", trace_str);
}

template <typename... Args>
void DebugTraceModuleCall(const char* module, const char* exprList, const char* file, int line, const Args&... args)
{
  std::string trace_str = formatParameterList(exprList, trace::formatting_options, args...);
  std::string preamble = trace::formatting_options.format_module_style("DEBUG_TRACE", module) + " in module " +
                         trace::formatting_options.format_module_style(module, module) + " at " +
                         trace::formatting_options.format_style(file, "TRACE_FILENAME") +
                         trace::formatting_options.format_style(fmt::format(":{}", line), "TRACE_LINE");
  print("{}{}{}\n", preamble, trace_str.empty() ? "" : " : ", trace_str);
}

template <typename... Args>
void CheckCall(const char* cond, const char* exprList, const char* file, int line, const Args&... args)
{
  std::string trace_str = formatParameterList(exprList, trace::formatting_options, args...);
  std::string preamble = trace::formatting_options.format_style("CHECK", "CHECK") + " at " +
                         trace::formatting_options.format_style(file, "TRACE_FILENAME") +
                         trace::formatting_options.format_style(fmt::format(":{}", line), "TRACE_LINE") +
                         fmt::format("\n{} is {}!", trace::formatting_options.format_style(cond, "TRACE_EXPR"),
                                     trace::formatting_options.format_style("false", "TRACE_VALUE"));
  print("{}{}{}\n", preamble, trace_str.empty() ? "" : "\n : ", trace_str);
  std::abort();
}

template <typename... Args>
void DebugCheckCall(const char* cond, const char* exprList, const char* file, int line, const Args&... args)
{
  std::string trace_str = formatParameterList(exprList, trace::formatting_options, args...);
  std::string preamble = trace::formatting_options.format_style("DEBUG_CHECK", "DEBUG_CHECK") + " at " +
                         trace::formatting_options.format_style(file, "TRACE_FILENAME") +
                         trace::formatting_options.format_style(fmt::format(":{}", line), "TRACE_LINE") +
                         fmt::format("\n{} is {}!", trace::formatting_options.format_style(cond, "TRACE_EXPR"),
                                     trace::formatting_options.format_style("false", "TRACE_VALUE"));
  print("{}{}{}\n", preamble, trace_str.empty() ? "" : "\n : ", trace_str);
  std::abort();
}

template <typename... Args>
void CheckEqualCall(const char* a, const char* b, const char* exprList, const char* file, int line, const Args&... args)
{
  std::string trace_str = formatParameterList(exprList, trace::formatting_options, args...);
  std::string preamble =
      trace::formatting_options.format_style("CHECK_EQUAL", "CHECK") + " at " +
      trace::formatting_options.format_style(file, "TRACE_FILENAME") +
      trace::formatting_options.format_style(fmt::format(":{}", line), "TRACE_LINE") +
      fmt::format("\n{} is not equal to {}!", trace::formatting_options.format_style(a, "TRACE_EXPR"),
                  trace::formatting_options.format_style(b, "TRACE_EXPR"));
  print("{}{}{}\n", preamble, trace_str.empty() ? "" : "\n : ", trace_str);
  std::abort();
}

template <typename... Args>
void DebugCheckEqualCall(const char* a, const char* b, const char* exprList, const char* file, int line,
                         const Args&... args)
{
  std::string trace_str = formatParameterList(exprList, trace::formatting_options, args...);
  std::string preamble =
      trace::formatting_options.format_style("DEBUG_CHECK_EQUAL", "DEBUG_CHECK") + " at " +
      trace::formatting_options.format_style(file, "TRACE_FILENAME") +
      trace::formatting_options.format_style(fmt::format(":{}", line), "TRACE_LINE") +
      fmt::format("\n{} is not equal to {}!", trace::formatting_options.format_style(a, "TRACE_EXPR"),
                  trace::formatting_options.format_style(b, "TRACE_EXPR"));
  print("{}{}{}\n", preamble, trace_str.empty() ? "" : "\n : ", trace_str);
  std::abort();
}

template <typename... Args>
void PreconditionCall(const char* cond, const char* exprList, const char* file, int line, const Args&... args)
{
  std::string trace_str = formatParameterList(exprList, trace::formatting_options, args...);
  std::string preamble = trace::formatting_options.format_style("PRECONDITION", "PRECONDITION") + " at " +
                         trace::formatting_options.format_style(file, "TRACE_FILENAME") +
                         trace::formatting_options.format_style(fmt::format(":{}", line), "TRACE_LINE") +
                         fmt::format("\n{} is {}!", trace::formatting_options.format_style(cond, "TRACE_EXPR"),
                                     trace::formatting_options.format_style("false", "TRACE_VALUE"));
  print("{}{}{}\n", preamble, trace_str.empty() ? "" : "\n : ", trace_str);
  std::abort();
}

template <typename... Args>
void DebugPreconditionCall(const char* cond, const char* exprList, const char* file, int line, const Args&... args)
{
  std::string trace_str = formatParameterList(exprList, trace::formatting_options, args...);
  std::string preamble = trace::formatting_options.format_style("DEBUG_PRECONDITION", "DEBUG_PRECONDITION") + " at " +
                         trace::formatting_options.format_style(file, "TRACE_FILENAME") +
                         trace::formatting_options.format_style(fmt::format(":{}", line), "TRACE_LINE") +
                         fmt::format("\n{} is {}!", trace::formatting_options.format_style(cond, "TRACE_EXPR"),
                                     trace::formatting_options.format_style("false", "TRACE_VALUE"));
  print("{}{}{}\n", preamble, trace_str.empty() ? "" : "\n : ", trace_str);
  std::abort();
}

template <typename... Args>
void PreconditionEqualCall(const char* a, const char* b, const char* exprList, const char* file, int line,
                           const Args&... args)
{
  std::string trace_str = formatParameterList(exprList, trace::formatting_options, args...);
  std::string preamble =
      trace::formatting_options.format_style("PRECONDITION_EQUAL", "PRECONDITION") + " at " +
      trace::formatting_options.format_style(file, "TRACE_FILENAME") +
      trace::formatting_options.format_style(fmt::format(":{}", line), "TRACE_LINE") +
      fmt::format("\n{} is not equal to {}!", trace::formatting_options.format_style(a, "TRACE_EXPR"),
                  trace::formatting_options.format_style(b, "TRACE_EXPR"));
  print("{}{}{}\n", preamble, trace_str.empty() ? "" : "\n : ", trace_str);
  std::abort();
}

template <typename... Args>
void DebugPreconditionEqualCall(const char* a, const char* b, const char* exprList, const char* file, int line,
                                const Args&... args)
{
  std::string trace_str = formatParameterList(exprList, trace::formatting_options, args...);
  std::string preamble =
      trace::formatting_options.format_style("DEBUG_PRECONDITION_EQUAL", "DEBUG_PRECONDITION") + " at " +
      trace::formatting_options.format_style(file, "TRACE_FILENAME") +
      trace::formatting_options.format_style(fmt::format(":{}", line), "TRACE_LINE") +
      fmt::format("\n{} is not equal to {}!", trace::formatting_options.format_style(a, "TRACE_EXPR"),
                  trace::formatting_options.format_style(b, "TRACE_EXPR"));
  print("{}{}{}\n", preamble, trace_str.empty() ? "" : "\n : ", trace_str);
  std::abort();
}

template <typename... Args> void PanicCall(const char* exprList, const char* file, int line, const Args&... args)
{
  std::string trace_str = formatParameterList(exprList, trace::formatting_options, args...);
  std::string preamble = trace::formatting_options.format_style("PANIC", "PANIC") + " at " +
                         trace::formatting_options.format_style(file, "TRACE_FILENAME") +
                         trace::formatting_options.format_style(fmt::format(":{}", line), "TRACE_LINE");
  print("{}{}{}\n", preamble, trace_str.empty() ? "" : " : ", trace_str);

#if UNI20_HAS_STACKTRACE
  if (trace::formatting_options.should_show_color())
    print(terminal::color_text("Stacktrace:\n", terminal::TerminalStyle("Bold")));
  else
    print("Stacktrace:\n");
  print("{}", std::stacktrace::current());
#else
  print("Stacktrace not available (compiler is too old)!\n");
#endif

  std::abort();
}

template <typename... Args> void ErrorCall(const char* exprList, const char* file, int line, const Args&... args)
{
  std::string trace_str = formatParameterList(exprList, trace::formatting_options, args...);
  std::string preamble = trace::formatting_options.format_style("ERROR", "ERROR") + " at " +
                         trace::formatting_options.format_style(file, "TRACE_FILENAME") +
                         trace::formatting_options.format_style(fmt::format(":{}", line), "TRACE_LINE");
  std::string Msg = fmt::format("{}{}{}\n", preamble, trace_str.empty() ? "" : " : ", trace_str);
  if (trace::formatting_options.errors_abort())
  {
    print("{}", Msg);
    std::abort();
  }
  throw std::runtime_error(Msg);
}

template <typename... Args>
void ErrorIfCall(const char* cond, const char* exprList, const char* file, int line, const Args&... args)
{
  std::string trace_str = formatParameterList(exprList, trace::formatting_options, args...);
  std::string preamble = trace::formatting_options.format_style("ERROR", "ERROR") + " at " +
                         trace::formatting_options.format_style(file, "TRACE_FILENAME") +
                         trace::formatting_options.format_style(fmt::format(":{}", line), "TRACE_LINE") +
                         fmt::format("\n{} is {}!", trace::formatting_options.format_style(cond, "TRACE_EXPR"),
                                     trace::formatting_options.format_style("false", "TRACE_VALUE"));
  std::string Msg = fmt::format("{}{}{}\n", preamble, trace_str.empty() ? "" : " : ", trace_str);
  if (trace::formatting_options.errors_abort())
  {
    print("{}", Msg);
    std::abort();
  }
  throw std::runtime_error(Msg);
}

} // namespace trace
