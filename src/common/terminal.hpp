#pragma once

#include <string>
#include <utility>

namespace terminal
{

// return the size of the output terminal, as (rows, cols)
std::pair<int, int> size();

// return the number of rows of the output terminal
int rows();

// return the number of columns in the output terminal
int columns();

// returns true if std::cout is a terminal, false otherwise
bool is_cout_terminal();

enum class color
{
  Reset = 0,
  Bold = 1,
  Dim = 2,
  Underline = 4,
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

// returns the ANSI escape sequence for the given color
std::string color_code(color c);

// returns the ANSI escape sequence for the given color, expressed as an int
std::string color_code(int c);

// returns the string description of the given color code
std::string to_string(color c);

// parses a string of color codes, and returns the ANSI escape sequence.
// The color codes are a comma-separated list of either numeric values,
// or string values.
std::string parse_color_codes(std::string const& s);

// returns a string with s containing the given color (including a reset suffix)
std::string color_text(std::string s, color c);

// returns a string with s containing the given colors (including a reset suffix)
std::string color_text(std::string s, color c1, color c2);

} // namespace terminal
