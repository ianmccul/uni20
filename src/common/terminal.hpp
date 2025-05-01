#pragma once

#include "string_util.hpp"
#include <algorithm>
#include <cstdio>
#include <mutex>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

namespace terminal
{

// return the size of the output terminal, as (rows, cols)
std::pair<int, int> size();

// return the number of rows of the output terminal
int rows();

// return the number of columns in the output terminal
int columns();

// returns true if the given stream is a terminal, false otherwise
bool is_a_terminal(std::FILE* stream);

inline std::string quote_shell(std::string_view s)
{
  // Characters that require quoting/escaping
  constexpr std::string_view special_chars = " \t()[]*\\\"";

  // Check if any character in s is one of the special ones
  bool needs_quote = std::ranges::any_of(s, [&](char c) { return special_chars.find(c) != std::string_view::npos; });

  if (needs_quote)
  {
    std::string result;
    // Reserve extra space (worst-case: every character may need escaping)
    result.reserve(s.size() * 2);
    for (char c : s)
    {
      if (c == '\\')
      {
        result.append("\\\\");
      }
      else if (c == '"')
      {
        result.append("\\\"");
      }
      else
      {
        result.push_back(c);
      }
    }
    return "\"" + result + "\"";
  }
  return std::string(s);
}

// convert argc/argv into a command line string, with shell escapes
inline std::string cmdline(int argc, char** argv)
{
  std::string result;
  if (argc > 0) result = quote_shell(argv[0]);
  for (int i = 1; i < argc; ++i)
    result = result + ' ' + quote_shell(argv[i]);
  return result;
}

// expands components of the form ${XXX} as the corresponding environment string
std::string expand_environment(std::string const& s);

inline bool env_exists(std::string const& str) { return getenv(str.c_str()) != nullptr; }

// getenv_or_default
// Get an environment string, or if the environment string is not defined, return the specified default_value
// template <typename T> T getenv_or_default(const std::string& var, const T& default_value)
// {
//   if (const char* env = std::getenv(var.c_str()))
//   {
//     try
//     {
//       return from_string<T>(env);
//     }
//     catch (...)
//     {
//       // Conversion failed; fall through to return default_value.
//     }
//   }
//   return default_value;
// }

template <typename T, typename U> T getenv_or_default(const std::string& var, const U& default_value)
{
  if (const char* env = std::getenv(var.c_str()))
  {
    try
    {
      return from_string<T>(env);
    }
    catch (...)
    {
      // Conversion failed; fall through to return default_value.
    }
  }
  return T(default_value);
}

inline char const* getenv_or_default(std::string const& str, char const* Default)
{
  char const* Str = getenv(str.c_str());
  return Str ? Str : Default;
}

// color output
enum class ForegroundColor
{
  Default = 39,
  Black = 30,
  Red = 31,
  Green = 32,
  Yellow = 33,
  Blue = 34,
  Magenta = 35,
  Cyan = 36,
  LightGray = 37,
  DarkGray = 90,
  LightRed = 91,
  LightGreen = 92,
  LightYellow = 93,
  LightBlue = 94,
  LightMagenta = 95,
  LightCyan = 96,
  White = 97
};

enum class BackgroundColor
{
  Default = 49,
  Black = 40,
  Red = 41,
  Green = 42,
  Yellow = 43,
  Blue = 44,
  Magenta = 45,
  Cyan = 46,
  LightGray = 47,
  // Extended bright background colors:
  DarkGray = 100, // often used as bright black
  LightRed = 101,
  LightGreen = 102,
  LightYellow = 103,
  LightBlue = 104,
  LightMagenta = 105,
  LightCyan = 106,
  White = 107
};

struct RGBColor
{
    int r, g, b;
};

using FGColor = std::variant<ForegroundColor, RGBColor>;
using BGColor = std::variant<BackgroundColor, RGBColor>;

// --- Bitmask for Attributes (unchanged) ---
enum class ColorAttribute : unsigned
{
  None = 0,
  Bold = 1 << 0,     // ANSI code 1
  Dim = 1 << 1,      // ANSI code 2
  Underline = 1 << 2 // ANSI code 4
};

// bitwise OR operator for ColorAttribute
constexpr ColorAttribute operator|(ColorAttribute lhs, ColorAttribute rhs)
{
  return static_cast<ColorAttribute>(static_cast<unsigned>(lhs) | static_cast<unsigned>(rhs));
}

// compound assignment operator for ColorAttribute
constexpr ColorAttribute& operator|=(ColorAttribute& lhs, ColorAttribute rhs)
{
  lhs = lhs | rhs;
  return lhs;
}

/// \brief A simple yes/no toggle that parses from a string.
///
/// Accepts "yes", "true", "1" (case-insensitive) as true,
/// and "no", "false", "0" as false. An empty string is treated as true.
/// Unrecognized values also default to true.
///
/// Useful for parsing environment variables and optional flags.
struct toggle
{
    bool value;

    toggle(bool value_) : value(value_) {}

    /// \brief Construct from a string (e.g. from an env var).
    /// \param str The string to interpret.
    /// \param default_value Used if the string is empty or not recognized.
    toggle(std::string_view str, bool default_value = true)
    {
      if (str.empty())
      {
        value = default_value;
      }
      else if (iequals(str, "no") || iequals(str, "false") || iequals(str, "0"))
      {
        value = false;
      }
      else if (iequals(str, "yes") || iequals(str, "true") || iequals(str, "1"))
      {
        value = true;
      }
      else
      {
        value = default_value;
      }
    }

    /// \brief Implicit conversion to bool.
    operator bool() const { return value; }
};

class TerminalStyle {
  public:
    // Foreground and background can be either standard colors or RGB
    std::optional<FGColor> fg;
    std::optional<BGColor> bg;
    ColorAttribute attrs = ColorAttribute::None;

    TerminalStyle() = default;

    // Constructors for standard colors or attributes
    TerminalStyle(ForegroundColor f) : fg(f) {}
    TerminalStyle(BackgroundColor b) : bg(b) {}
    TerminalStyle(ColorAttribute a) : attrs(a) {}

    // Constructors for RGB colors
    TerminalStyle(const RGBColor& rgb, bool foreground = true)
    {
      if (foreground)
        fg = rgb;
      else
        bg = rgb;
    }

    // Conversion from a string
    TerminalStyle(const std::string& s);

    TerminalStyle(std::string_view sv) : TerminalStyle(std::string(sv)) {}

    // Combine two TerminalStyle objects
    TerminalStyle operator|(const TerminalStyle& other) const
    {
      TerminalStyle result = *this;
      if (other.fg.has_value()) result.fg = other.fg;
      if (other.bg.has_value()) result.bg = other.bg;
      result.attrs = result.attrs | other.attrs;
      return result;
    }

    // Overloads for combining with a standard color or attribute
    friend TerminalStyle operator|(ForegroundColor f, const TerminalStyle& ts) { return TerminalStyle(f) | ts; }
    friend TerminalStyle operator|(BackgroundColor b, const TerminalStyle& ts) { return TerminalStyle(b) | ts; }
    friend TerminalStyle operator|(ColorAttribute a, const TerminalStyle& ts) { return TerminalStyle(a) | ts; }
    friend TerminalStyle operator|(const RGBColor& rgb, const TerminalStyle& ts) { return TerminalStyle(rgb) | ts; }

    // Helper functions to produce ANSI escape sequences from RGB values
    static std::string rgb_fg_code(const RGBColor& rgb)
    {
      return "38;2;" + std::to_string(rgb.r) + ";" + std::to_string(rgb.g) + ";" + std::to_string(rgb.b);
    }
    static std::string rgb_bg_code(const RGBColor& rgb)
    {
      return "48;2;" + std::to_string(rgb.r) + ";" + std::to_string(rgb.g) + ";" + std::to_string(rgb.b);
    }

    // Convert this TerminalStyle into an ANSI escape sequence.
    std::string to_string() const;
};

// parseTerminalStyle expects a style string made up of one or more style components,
// separated by commas. Each component can optionally specify a target using "fg:" for
// foreground or "bg:" for background (default is foreground), followed by a color spec,
// and then optional attributes separated by semicolons.
//
// Color specifications can be provided as:
//   - A named color (e.g., "Red", "Blue", "lightgray", etc.)
//   - An RGB color in function notation (e.g., "rgb(255,0,0)")
//   - A hexadecimal color (e.g., "#FF0000" or "#F00")
//
// Any tokens following the color (separated by semicolons) are treated as text attributes,
// such as "Bold", "Dim", or "Underline".
//
// Examples:
//   "Red;Bold"                 -> Sets foreground to Red with Bold.
//   "fg:rgb(255,0,0);Bold"       -> Sets foreground to rgb(255,0,0) with Bold.
//   "bg:#00FF00;Dim"            -> Sets background to green (#00FF00) with Dim.
//   "fg:LightGray;Underline, bg:darkgray"
//                              -> Sets foreground to LightGray with Underline,
//                                 and background to darkgray.
TerminalStyle parseTerminalStyle(const std::string& styleStr);

// Convenience function to colorize text.
inline std::string color_text(std::string_view s, const TerminalStyle& ts)
{
  return ts.to_string() + std::string(s) + "\033[0m";
}

inline std::string color_if(std::string_view s, bool b, const TerminalStyle& ts)
{
  return b ? (ts.to_string() + s + "\033[0m") : std::string(s);
}

} // namespace terminal
