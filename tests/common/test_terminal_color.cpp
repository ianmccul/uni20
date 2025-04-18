#include "common/terminal.hpp" // Your header containing TerminalStyle, parseTerminalStyle, etc.
#include "gtest/gtest.h"

using namespace terminal;

// Test parsing a simple named color for foreground.
TEST(TerminalStyleTest, NamedColorOnly)
{
  // "Red" should set the foreground to red.
  TerminalStyle style = parseTerminalStyle("Red");
  // Expect ANSI sequence for foreground red: "\033[31m"
  EXPECT_EQ(style.to_string(), "\033[31m");
}

// Test parsing a named color with an attribute.
TEST(TerminalStyleTest, NamedColorWithAttribute)
{
  // "Red;Bold" should set foreground to red and add Bold (ANSI code 1).
  TerminalStyle style = parseTerminalStyle("Red;Bold");
  // Expected order in our implementation: attributes then foreground.
  // This yields "1;31" so ANSI sequence: "\033[1;31m"
  EXPECT_EQ(style.to_string(), "\033[1;31m");
}

// Test parsing separate foreground and background within one comma‐separated part.
TEST(TerminalStyleTest, ForegroundAndBackgroundInOnePart)
{
  // "fg:Black;bg:White" should set foreground to Black (30) and background to White (107)
  TerminalStyle style = parseTerminalStyle("fg:Black;bg:White");
  // Expected ANSI sequence: "\033[30;107m"
  EXPECT_EQ(style.to_string(), "\033[30;107m");
}

// Test parsing mixed targets in a comma‐separated string.
TEST(TerminalStyleTest, MixedTargets)
{
  // Using two comma-separated components:
  // First: "fg:Blue;Bold" sets foreground to Blue (34) with Bold (1).
  // Second: "bg:Yellow;Underline" sets background to Yellow (43) with Underline (4).
  // The final TerminalStyle should combine these: Attributes Bold and Underline, fg Blue, bg Yellow.
  // Our to_string() implementation prints attributes first, then foreground, then background.
  // So we expect: "1;4;34;43" inside the ANSI sequence.
  TerminalStyle style = parseTerminalStyle("fg:Blue;Bold, bg:Yellow;Underline");
  EXPECT_EQ(style.to_string(), "\033[1;4;34;43m");
}

// Test parsing RGB and hex colors.
TEST(TerminalStyleTest, RGBAndHex)
{
  // For RGB using function notation.
  TerminalStyle styleRGB = parseTerminalStyle("fg:rgb(255,0,0);Bold");
  // Expected ANSI for foreground RGB: "38;2;255;0;0" and Bold adds "1".
  EXPECT_EQ(styleRGB.to_string(), "\033[1;38;2;255;0;0m");

  // For hex notation.
  TerminalStyle styleHex = parseTerminalStyle("fg:#00FF00;Underline");
  // Expected ANSI for foreground hex: "38;2;0;255;0" and Underline is "4".
  // The order from our implementation: attributes, then foreground.
  EXPECT_EQ(styleHex.to_string(), "\033[4;38;2;0;255;0m");
}

// Test a lone attribute should be applied even if no color is given.
TEST(TerminalStyleTest, LoneAttribute)
{
  // If only an attribute is provided, the parser should still add the attribute.
  TerminalStyle style = parseTerminalStyle("Bold");
  // No color provided; so only Bold (ANSI code "1") should be set.
  EXPECT_EQ(style.to_string(), "\033[1m");
}
