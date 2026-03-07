// terminal.cpp

#include "terminal.hpp"

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

// Conditionally include system headers based on macros provided by CMake.
#ifdef HAVE_TERMIOS_H
#include <termios.h>
#endif

#ifdef HAVE_SYS_IOCTL_H
#include <sys/ioctl.h>
#endif

namespace terminal
{

std::string TerminalStyle::to_string() const
{
  // We'll accumulate ANSI codes in a vector.
  std::vector<std::string> codes;

  // Process text attributes.
  if (static_cast<unsigned>(attrs) != 0)
  {
    if (static_cast<unsigned>(attrs) & static_cast<unsigned>(ColorAttribute::Bold)) codes.push_back("1");
    if (static_cast<unsigned>(attrs) & static_cast<unsigned>(ColorAttribute::Dim)) codes.push_back("2");
    if (static_cast<unsigned>(attrs) & static_cast<unsigned>(ColorAttribute::Underline)) codes.push_back("4");
  }

  // Process foreground color.
  if (fg.has_value())
  {
    std::visit(
        [&codes](auto&& arg) {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, ForegroundColor>)
          {
            codes.push_back(fmt::format("{}", static_cast<int>(arg)));
          }
          else if constexpr (std::is_same_v<T, RGBColor>)
          {
            // 38;2;r;g;b for foreground RGB.
            codes.push_back(fmt::format("38;2;{};{};{}", arg.r, arg.g, arg.b));
          }
        },
        fg.value());
  }

  // Process background color.
  if (bg.has_value())
  {
    std::visit(
        [&codes](auto&& arg) {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, BackgroundColor>)
          {
            codes.push_back(fmt::format("{}", static_cast<int>(arg)));
          }
          else if constexpr (std::is_same_v<T, RGBColor>)
          {
            // 48;2;r;g;b for background RGB.
            codes.push_back(fmt::format("48;2;{};{};{}", arg.r, arg.g, arg.b));
          }
        },
        bg.value());
  }

  // Join the codes with semicolons.
  std::string joined = fmt::format("{}", fmt::join(codes, ";"));
  // Construct the full ANSI escape sequence.
  return fmt::format("\033[{}m", joined);
}

// Splits a string by a given delimiter character.
std::vector<std::string> splitString(const std::string& s, char delim)
{
  std::vector<std::string> parts;
  size_t start = 0;
  while (true)
  {
    size_t pos = s.find(delim, start);
    if (pos == std::string::npos)
    {
      parts.push_back(s.substr(start));
      break;
    }
    parts.push_back(s.substr(start, pos - start));
    start = pos + 1;
  }
  return parts;
}

// Splits the input string 's' on the delimiter 'delim', but only at the top level.
// Delimiters inside matching pairs of (), [] or {} are ignored.
std::vector<std::string> splitTopLevel(const std::string& s, char delim = ',')
{
  std::vector<std::string> result;
  std::string current;
  int nestingLevel = 0; // tracks how deep we are in any nested bracket

  for (char c : s)
  {
    // If we see an opening bracket, increase the nesting level.
    if (c == '(' || c == '[' || c == '{')
    {
      ++nestingLevel;
    }
    // If we see a closing bracket, decrease the nesting level.
    else if (c == ')' || c == ']' || c == '}')
    {
      if (nestingLevel > 0)
      {
        --nestingLevel;
      }
    }

    // When we encounter the delimiter and we're not nested, split here.
    if (c == delim && nestingLevel == 0)
    {
      result.push_back(current);
      current.clear();
    }
    else
    {
      current.push_back(c);
    }
  }

  // Push any remaining content as the final token.
  if (!current.empty())
  {
    result.push_back(current);
  }
  return result;
}

// Convert a string to lowercase.
inline std::string to_lower(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
  return s;
}

// Trim leading and trailing whitespace.
inline std::string trim(const std::string& s)
{
  const std::string whitespace = " \t\n\r";
  size_t start = s.find_first_not_of(whitespace);
  if (start == std::string::npos) return "";
  size_t end = s.find_last_not_of(whitespace);
  return s.substr(start, end - start + 1);
}

// Parse a foreground color from a token.
std::optional<ForegroundColor> parseForegroundColor(const std::string& token)
{
  std::string t = to_lower(trim(token));
  if (t == "default") return ForegroundColor::Default;
  if (t == "black") return ForegroundColor::Black;
  if (t == "red") return ForegroundColor::Red;
  if (t == "green") return ForegroundColor::Green;
  if (t == "yellow") return ForegroundColor::Yellow;
  if (t == "blue") return ForegroundColor::Blue;
  if (t == "magenta") return ForegroundColor::Magenta;
  if (t == "cyan") return ForegroundColor::Cyan;
  if (t == "lightgray") return ForegroundColor::LightGray;
  if (t == "darkgray") return ForegroundColor::DarkGray;
  if (t == "lightred") return ForegroundColor::LightRed;
  if (t == "lightgreen") return ForegroundColor::LightGreen;
  if (t == "lightyellow") return ForegroundColor::LightYellow;
  if (t == "lightblue") return ForegroundColor::LightBlue;
  if (t == "lightmagenta") return ForegroundColor::LightMagenta;
  if (t == "lightcyan") return ForegroundColor::LightCyan;
  if (t == "white") return ForegroundColor::White;
  return std::nullopt;
}

// Parse a background color from a token.
std::optional<BackgroundColor> parseBackgroundColor(const std::string& token)
{
  std::string t = to_lower(trim(token));
  if (t == "default") return BackgroundColor::Default;
  if (t == "black") return BackgroundColor::Black;
  if (t == "red") return BackgroundColor::Red;
  if (t == "green") return BackgroundColor::Green;
  if (t == "yellow") return BackgroundColor::Yellow;
  if (t == "blue") return BackgroundColor::Blue;
  if (t == "magenta") return BackgroundColor::Magenta;
  if (t == "cyan") return BackgroundColor::Cyan;
  if (t == "lightgray") return BackgroundColor::LightGray;
  // Extended bright background colors:
  if (t == "darkgray") return BackgroundColor::DarkGray;         // ANSI code 100
  if (t == "lightred") return BackgroundColor::LightRed;         // ANSI code 101
  if (t == "lightgreen") return BackgroundColor::LightGreen;     // ANSI code 102
  if (t == "lightyellow") return BackgroundColor::LightYellow;   // ANSI code 103
  if (t == "lightblue") return BackgroundColor::LightBlue;       // ANSI code 104
  if (t == "lightmagenta") return BackgroundColor::LightMagenta; // ANSI code 105
  if (t == "lightcyan") return BackgroundColor::LightCyan;       // ANSI code 106
  if (t == "white") return BackgroundColor::White;               // ANSI code 107
  return std::nullopt;
}

// Parse a text attribute from a token.
std::optional<ColorAttribute> parseColorAttribute(const std::string& token)
{
  std::string t = to_lower(trim(token));
  if (t == "bold") return ColorAttribute::Bold;
  if (t == "dim") return ColorAttribute::Dim;
  if (t == "underline") return ColorAttribute::Underline;
  return std::nullopt;
}

std::optional<RGBColor> parseRGBColor(const std::string& token)
{
  const std::string prefix = "rgb(";
  // Check that token starts with "rgb(" and ends with ")"
  if (token.size() < prefix.size() + 1 || token.compare(0, prefix.size(), prefix) != 0 || token.back() != ')')
    return std::nullopt;

  // Extract the inner part between "rgb(" and ")"
  std::string inner = token.substr(prefix.size(), token.size() - prefix.size() - 1);

  std::vector<int> values;
  size_t start = 0;
  while (true)
  {
    size_t commaPos = inner.find(',', start);
    std::string part;
    if (commaPos == std::string::npos)
    {
      part = inner.substr(start);
      part = trim(part);
      if (!part.empty())
      {
        try
        {
          int v = std::stoi(part);
          values.push_back(v);
        }
        catch (...)
        {
          return std::nullopt;
        }
      }
      break;
    }
    else
    {
      part = inner.substr(start, commaPos - start);
      part = trim(part);
      if (!part.empty())
      {
        try
        {
          int v = std::stoi(part);
          values.push_back(v);
        }
        catch (...)
        {
          return std::nullopt;
        }
      }
      start = commaPos + 1;
    }
  }

  // There must be exactly 3 values.
  if (values.size() != 3) return std::nullopt;

  // Optionally, verify that each value is in [0,255].
  for (int v : values)
  {
    if (v < 0 || v > 255) return std::nullopt;
  }

  return RGBColor{values[0], values[1], values[2]};
}

std::optional<RGBColor> parseHexColor(const std::string& token)
{
  // Check that the token is non-empty and starts with '#'
  if (token.empty() || token[0] != '#')
  {
    return std::nullopt;
  }

  // Remove the '#' prefix.
  std::string hex = token.substr(1);

  // Depending on the length, handle shorthand (#RGB) or full (#RRGGBB)
  if (hex.size() == 3)
  {
    // Expand each digit to two digits.
    std::string r_str{hex[0], hex[0]};
    std::string g_str{hex[1], hex[1]};
    std::string b_str{hex[2], hex[2]};

    try
    {
      int r = std::stoi(r_str, nullptr, 16);
      int g = std::stoi(g_str, nullptr, 16);
      int b = std::stoi(b_str, nullptr, 16);
      return RGBColor{r, g, b};
    }
    catch (const std::exception&)
    {
      return std::nullopt;
    }
  }
  else if (hex.size() == 6)
  {
    try
    {
      int r = std::stoi(hex.substr(0, 2), nullptr, 16);
      int g = std::stoi(hex.substr(2, 2), nullptr, 16);
      int b = std::stoi(hex.substr(4, 2), nullptr, 16);
      return RGBColor{r, g, b};
    }
    catch (const std::exception&)
    {
      return std::nullopt;
    }
  }

  // If the string is not 3 or 6 hex digits long, it's not valid.
  return std::nullopt;
}

TerminalStyle parseTerminalStyle(const std::string& styleStr)
{
  TerminalStyle result;

  // Split the style string on top-level commas.
  auto parts = splitTopLevel(styleStr, ',');
  for (auto& part : parts)
  {
    part = trim(part);
    if (part.empty()) continue;

    // Determine the target ("fg" or "bg") from the part header if present.
    std::string target = "fg";
    std::string tokenList;
    size_t colonPos = part.find(':');
    if (colonPos != std::string::npos)
    {
      target = to_lower(trim(part.substr(0, colonPos)));
      tokenList = part.substr(colonPos + 1);
    }
    else
    {
      tokenList = part;
    }

    // Split tokenList on semicolons.
    auto tokens = splitString(tokenList, ';');
    for (auto& token : tokens)
    {
      token = trim(token);
      if (token.empty()) continue;

      // Check if the token itself overrides the target (e.g. "bg:White").
      std::string current_target = target;
      size_t innerColon = token.find(':');
      if (innerColon != std::string::npos)
      {
        current_target = to_lower(trim(token.substr(0, innerColon)));
        token = trim(token.substr(innerColon + 1));
        if (token.empty()) continue;
      }

      bool parsedColor = false;
      if (current_target == "fg")
      {
        if (token.rfind("rgb(", 0) == 0)
        {
          if (auto rgbOpt = parseRGBColor(token); rgbOpt.has_value())
          {
            result = result | TerminalStyle(rgbOpt.value(), true);
            parsedColor = true;
          }
        }
        else if (!token.empty() && token.front() == '#')
        {
          if (auto rgbOpt = parseHexColor(token); rgbOpt.has_value())
          {
            result = result | TerminalStyle(rgbOpt.value(), true);
            parsedColor = true;
          }
        }
        else if (auto fgOpt = parseForegroundColor(token); fgOpt.has_value())
        {
          result = result | TerminalStyle(fgOpt.value());
          parsedColor = true;
        }
      }
      else if (current_target == "bg")
      {
        if (token.rfind("rgb(", 0) == 0)
        {
          if (auto rgbOpt = parseRGBColor(token); rgbOpt.has_value())
          {
            result = result | TerminalStyle(rgbOpt.value(), false);
            parsedColor = true;
          }
        }
        else if (!token.empty() && token.front() == '#')
        {
          if (auto rgbOpt = parseHexColor(token); rgbOpt.has_value())
          {
            result = result | TerminalStyle(rgbOpt.value(), false);
            parsedColor = true;
          }
        }
        else if (auto bgOpt = parseBackgroundColor(token); bgOpt.has_value())
        {
          result = result | TerminalStyle(bgOpt.value());
          parsedColor = true;
        }
      }
      // If token didn't parse as a color, try parsing it as an attribute.
      if (!parsedColor)
      {
        if (auto attrOpt = parseColorAttribute(token); attrOpt.has_value())
        {
          result = result | TerminalStyle(attrOpt.value());
        }
      }
    }
  }
  return result;
}

#if 0

TerminalStyle parseTerminalStyle(const std::string& styleStr)
{
  TerminalStyle result;

  // Split the style string on top-level commas.
  auto parts = splitTopLevel(styleStr, ',');
  for (auto& part : parts)
  {
    part = trim(part);
    if (part.empty()) continue;

    // Determine the target ("fg" or "bg"). Default to "fg".
    std::string target = "fg";
    std::string tokenList;
    size_t colonPos = part.find(':');
    if (colonPos != std::string::npos)
    {
      target = to_lower(trim(part.substr(0, colonPos)));
      tokenList = part.substr(colonPos + 1);
    }
    else
    {
      tokenList = part;
    }

    // Split tokenList on semicolons.
    auto tokens = splitString(tokenList, ';');
    bool colorSet = false; // Tracks whether we've successfully parsed a color.
    for (auto& token : tokens)
    {
      token = trim(token);
      if (token.empty()) continue;

      // If a color hasn't been set yet, try to parse a color.
      if (!colorSet)
      {
        bool parsedColor = false;
        if (target == "fg")
        {
          if (token.rfind("rgb(", 0) == 0)
          {
            if (auto rgbOpt = parseRGBColor(token); rgbOpt.has_value())
            {
              result = result | TerminalStyle(rgbOpt.value(), true);
              parsedColor = true;
            }
          }
          else if (!token.empty() && token.front() == '#')
          {
            if (auto rgbOpt = parseHexColor(token); rgbOpt.has_value())
            {
              result = result | TerminalStyle(rgbOpt.value(), true);
              parsedColor = true;
            }
          }
          else if (auto fgOpt = parseForegroundColor(token); fgOpt.has_value())
          {
            result = result | TerminalStyle(fgOpt.value());
            parsedColor = true;
          }
        }
        else if (target == "bg")
        {
          if (token.rfind("rgb(", 0) == 0)
          {
            if (auto rgbOpt = parseRGBColor(token); rgbOpt.has_value())
            {
              result = result | TerminalStyle(rgbOpt.value(), false);
              parsedColor = true;
            }
          }
          else if (!token.empty() && token.front() == '#')
          {
            if (auto rgbOpt = parseHexColor(token); rgbOpt.has_value())
            {
              result = result | TerminalStyle(rgbOpt.value(), false);
              parsedColor = true;
            }
          }
          else if (auto bgOpt = parseBackgroundColor(token); bgOpt.has_value())
          {
            result = result | TerminalStyle(bgOpt.value());
            parsedColor = true;
          }
        }
        // If the token didn't parse as a color, treat it as an attribute.
        if (parsedColor)
        {
          colorSet = true;
          continue; // Color token handled.
        }
        // Fall through: if no color was parsed, attempt to parse as an attribute.
      }
      // Parse token as an attribute.
      if (auto attrOpt = parseColorAttribute(token); attrOpt.has_value())
      {
        result = result | TerminalStyle(attrOpt.value());
      }
    }
  }
  return result;
}

#endif

TerminalStyle::TerminalStyle(const std::string& s) { *this = parseTerminalStyle(s); }

//
// Terminal size routines
//

int rows()
{
  // Primary option: environment variable "LINES"
  if (const char* rows_str = std::getenv("LINES"))
  {
    int n = std::atoi(rows_str);
    if (n > 0) return n;
  }
  // Fall back to ioctl (if available)
#ifdef TIOCGWINSZ
  winsize ws;
  if (!ioctl(1, TIOCGWINSZ, &ws) && ws.ws_row) return ws.ws_row;
#endif
  // Last resort
  return 25;
}

int columns()
{
  // Primary option: environment variable "COLUMNS"
  if (const char* cols_str = std::getenv("COLUMNS"))
  {
    int n = std::atoi(cols_str);
    if (n > 0) return n;
  }
  // Fall back to ioctl (if available)
#ifdef TIOCGWINSZ
  winsize ws;
  if (!ioctl(1, TIOCGWINSZ, &ws) && ws.ws_col) return ws.ws_col;
#endif
  // Last resort
  return 80;
}

std::pair<int, int> size() { return {rows(), columns()}; }

bool is_a_terminal(std::FILE* stream) { return stream && isatty(fileno(stream)); }

std::string expand_environment(std::string const& s)
{
  std::string result;
  std::string::const_iterator I = std::find(s.begin(), s.end(), '$');
  std::string::const_iterator marker = s.begin();
  while (I != s.end())
  {
    result += std::string(marker, I);
    marker = I;

    // see if we have a valid environment string
    if (++I != s.end() && *I == '{')
    {
      ++I;
      std::string::const_iterator EndEnv = std::find(I, s.end(), '}');
      if (EndEnv != s.end())
      {
        std::string EnvString(I, EndEnv);
        char* Replacement = getenv(EnvString.c_str());
        if (Replacement)
        {
          ++EndEnv;
          marker = I = EndEnv;
          result += std::string(Replacement);
        }
      }
    }

    I = std::find(I, s.end(), '$');
  }
  result += std::string(marker, I);
  return result;
}

// Splits a string by a given delimiter.
std::vector<std::string> split_string(const std::string& s, char delimiter)
{
  std::vector<std::string> tokens;
  size_t start = 0;
  while (true)
  {
    size_t pos = s.find(delimiter, start);
    if (pos == std::string::npos)
    {
      tokens.push_back(s.substr(start));
      break;
    }
    else
    {
      tokens.push_back(s.substr(start, pos - start));
      start = pos + 1;
    }
  }
  return tokens;
}

} // namespace terminal
