// terminal.cpp

#include "terminal.hpp"

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

// Map of colors to their string names.
std::map<color, std::string> ColorNames = {{color::Reset, "Reset"},
                                           {color::Bold, "Bold"},
                                           {color::Dim, "Dim"},
                                           {color::Underline, "Underline"},
                                           {color::Black, "Black"},
                                           {color::Red, "Red"},
                                           {color::Green, "Green"},
                                           {color::Yellow, "Yellow"},
                                           {color::Blue, "Blue"},
                                           {color::Magenta, "Magenta"},
                                           {color::Cyan, "Cyan"},
                                           {color::LightGray, "LightGray"},
                                           {color::DarkGray, "DarkGray"},
                                           {color::LightRed, "LightRed"},
                                           {color::LightGreen, "LightGreen"},
                                           {color::LightYellow, "LightYellow"},
                                           {color::LightBlue, "LightBlue"},
                                           {color::LightMagenta, "LightMagenta"},
                                           {color::LightCyan, "LightCyan"},
                                           {color::White, "White"}};

//
// Terminal size routines
//

int rows()
{
  // Primary option: environment variable "LINES"
  if (const char *rows_str = std::getenv("LINES"))
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
  if (const char *cols_str = std::getenv("COLUMNS"))
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

bool is_cout_terminal() { return isatty(1); }

//
// Color and formatting routines
//

// Returns the ANSI escape code for the given color enum.
std::string color_code(color c) { return "\033[" + std::to_string(static_cast<int>(c)) + "m"; }

// Overload: accepts an int instead.
std::string color_code(int c) { return "\033[" + std::to_string(c) + "m"; }

// Returns the string name for a given color.
std::string to_string(color c)
{
  auto it = ColorNames.find(c);
  return (it != ColorNames.end()) ? it->second : "Unknown";
}

// A simple case-insensitive comparison using std::string_view.
bool iequals(std::string_view a, std::string_view b)
{
  return a.size() == b.size() &&
         std::equal(a.begin(), a.end(), b.begin(),
                    [](unsigned char ca, unsigned char cb) { return std::tolower(ca) == std::tolower(cb); });
}

// Splits a string by a given delimiter.
std::vector<std::string> split_string(const std::string &s, char delimiter)
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

// Parses a string to a color value. The string may be numeric (e.g. "31")
// or a name (e.g. "Red"). In case of error, returns color::Reset.
color parse_code(const std::string &s)
{
  if (s.empty())
    return color::Reset;
  else if (std::isdigit(static_cast<unsigned char>(s[0])))
  {
    int value = 0;
    auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), value);
    if (ec == std::errc())
    {
      return color(value);
    }
    else
    {
      return color::Reset;
    }
  }
  else
  {
    // Look for a named color (case-insensitive).
    for (const auto &[col, name] : ColorNames)
    {
      if (iequals(s, name)) return col;
    }
    return color::Reset;
  }
}

// Parses a comma-separated list of color codes (either numeric or named)
// and returns the concatenated ANSI escape sequence.
std::string parse_color_codes(const std::string &s)
{
  std::vector<std::string> codes = split_string(s, ',');
  std::string result;
  for (const auto &code_str : codes)
  {
    result += color_code(parse_code(code_str));
  }
  return result;
}

// Wraps the given string in the specified color (and resets afterward).
std::string color_text(std::string s, color c)
{
  return color_code(c) + s + color_code(static_cast<int>(color::Reset));
}

// Overload: allows combining two color codes.
std::string color_text(std::string s, color c1, color c2)
{
  return color_code(c1) + color_code(c2) + s + color_code(static_cast<int>(color::Reset));
}

}  // namespace terminal
