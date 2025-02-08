#pragma once

#include <algorithm>
#include <cctype>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "terminal.hpp"
#include <fmt/core.h>

// The TRACE macro: forwards both the string version and evaluated arguments.

#define TRACE(...) trace::OutputTraceCall(#__VA_ARGS__, __FILE__, __LINE__, __VA_ARGS__)

#define TRACE_IF(cond, ...)                                                                                            \
  do                                                                                                                   \
  {                                                                                                                    \
    if (cond)                                                                                                          \
    {                                                                                                                  \
      TRACE(__VA_ARGS__);                                                                                              \
    }                                                                                                                  \
  } while (0)

#define TRACE_MODULE(m, ...)                                                                                           \
  do                                                                                                                   \
  {                                                                                                                    \
    if constexpr (ENABLE_TRACE_##m)                                                                                    \
    {                                                                                                                  \
      trace::OutputTraceModuleCall(#m, #__VA_ARGS__, __FILE__, __LINE__, __VA_ARGS__);                                 \
    }                                                                                                                  \
  } while (0)

#define TRACE_MODULE_IF(m, cond, ...)                                                                                  \
  do                                                                                                                   \
  {                                                                                                                    \
    if constexpr (ENABLE_TRACE_##m)                                                                                    \
    {                                                                                                                  \
      if (cond)                                                                                                        \
      {                                                                                                                \
        trace::OutputTraceModuleCall(#m, #__VA_ARGS__, __FILE__, __LINE__, __VA_ARGS__);                               \
      }                                                                                                                \
    }                                                                                                                  \
  } while (0)

#if defined(NDEBUG)
#define DEBUG_TRACE(...)
#define DEBUG_TRACE_IF(...)
#define DEBUG_TRACE_MODULE(...)
#define DEBUG_TRACE_MODULE_IF(...)

#else

#define DEBUG_TRACE(...) OutputDebugTraceCall(#__VA_ARGS__, __FILE__, __LINE__, __VA_ARGS__)

#define DEBUG_TRACE_IF(cond, ...)                                                                                      \
  do                                                                                                                   \
  {                                                                                                                    \
    if (cond)                                                                                                          \
    {                                                                                                                  \
      DEBUG_TRACE(__VA_ARGS__);                                                                                        \
    }                                                                                                                  \
  } while (0)

#define DEBUG_TRACE_MODULE(m, ...)                                                                                     \
  do                                                                                                                   \
  {                                                                                                                    \
    if constexpr (ENABLE_TRACE_##m)                                                                                    \
    {                                                                                                                  \
      trace::OutputDebugTraceModuleCall(#m, #__VA_ARGS__, __FILE__, __LINE__, __VA_ARGS__);                            \
    }                                                                                                                  \
  } while (0)

#define DEBUG_TRACE_MODULE_IF(m, cond, ...)                                                                            \
  do                                                                                                                   \
  {                                                                                                                    \
    if constexpr (ENABLE_TRACE_##m)                                                                                    \
    {                                                                                                                  \
      if (cond)                                                                                                        \
      {                                                                                                                \
        trace::OutputDebugTraceModuleCall(#m, #__VA_ARGS__, __FILE__, __LINE__, __VA_ARGS__);                          \
      }                                                                                                                \
    }                                                                                                                  \
  } while (0)

#endif

// trace namespace: Global settings for our trace facility.
namespace trace
{

//-------------------------------------------------------------------------
// Global formatting options

// FormattingOptions: Holds configuration for formatted output.
struct FormattingOptions
{
    template <typename T> static inline int fp_precision = 6;

    int terminal_width = terminal::columns(); // Maximum width (in characters) before switching to multi-line.

    static inline FILE* output_stream = stderr;
};

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
inline std::vector<std::pair<std::string, bool>> parseNames(const std::string& s)
{
  std::vector<std::pair<std::string, bool>> tokens;
  std::string current;
  bool tokenHasTopLevelLiteral = false;

  // Counters for grouping symbols.
  int parenCount = 0, squareCount = 0, angleCount = 0, curlyCount = 0;

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
    if ((c == '"' || c == '\'') && parenCount == 0 && squareCount == 0 && angleCount == 0 && curlyCount == 0)
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
    if (c == ',' && parenCount == 0 && squareCount == 0 && angleCount == 0 && curlyCount == 0)
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
    else if (c == '<')
      ++angleCount;
    else if (c == '>')
    {
      if (angleCount > 0) --angleCount;
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
  if (name.second) return value;

  // If we'd spill over onto another line, then insert a line break
  if (isMultiline(value)) // || (getMaxLineWidth(value) + std::ssize(name.first) + 3 > available_width))
  {
    int indent = name.first.size() + 3;
    return "\n" + name.first + " = " + indentMultiline(value, indent);
  }
  return fmt::format("{} = {}", name.first, value);
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
  std::string exprs(exprList);
  auto names = parseNames(exprs);
  return formatParameters(names.begin(), opts, args...);
}

template <typename... Args> void OutputTraceCall(const char* exprList, const char* file, int line, const Args&... args)
{
  std::string trace_str = formatParameterList(exprList, trace::formatting_options, args...);
  fmt::print(trace::formatting_options.output_stream, "TRACE at {}:{} : {}\n", file, line, trace_str);
}

template <typename... Args>
void OutputTraceModuleCall(const char* module, const char* exprList, const char* file, int line, const Args&... args)
{
  std::string trace_str = formatParameterList(exprList, trace::formatting_options, args...);
  fmt::print(trace::formatting_options.output_stream, "TRACE module {} at {}:{} : {}\n", module, file, line, trace_str);
}

template <typename... Args>
void OutputDebugTraceCall(const char* exprList, const char* file, int line, const Args&... args)
{
  std::string trace_str = formatParameterList(exprList, trace::formatting_options, args...);
  fmt::print(trace::formatting_options.output_stream, "DEBUG_TRACE at {}:{} : {}\n", file, line, trace_str);
}

template <typename... Args>
void OutputDebugTraceModuleCall(const char* module, const char* exprList, const char* file, int line,
                                const Args&... args)
{
  std::string trace_str = formatParameterList(exprList, trace::formatting_options, args...);
  fmt::print(trace::formatting_options.output_stream, "DEBUG_TRACE module {} at {}:{} : {}\n", module, file, line,
             trace_str);
}

} // namespace trace
