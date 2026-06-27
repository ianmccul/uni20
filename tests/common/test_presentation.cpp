#include <uni20/common/mdspan.hpp>
#include <uni20/common/presentation.hpp>
#include <uni20/common/presentation_mdspan.hpp>

#include "env_var_guard.hpp"

#include <fmt/core.h>
#include <gtest/gtest.h>

#include <array>
#include <complex>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
namespace presentation = uni20::presentation;
using uni20::test::EnvVarGuard;

[[nodiscard]] presentation::output_policy base_policy()
{
  auto policy = presentation::plain_policy();
  policy.color = presentation::color_mode::never;
  policy.glyphs = presentation::glyph_set::unicode;
  policy.charset = presentation::text_charset::utf8;
  policy.width = presentation::width_mode::display_cells;
  policy.ambiguous = presentation::ambiguous_width::narrow;
  policy.tab_width = 4;
  return policy;
}

} // namespace

TEST(PresentationPolicies, DefaultPoliciesPreferEmojiGlyphs)
{
  EnvVarGuard glyphs("UNI20_GLYPHS");
  EnvVarGuard charset("UNI20_CHARSET");
  EnvVarGuard color("UNI20_COLOR");
  glyphs.unset();
  charset.unset();
  color.unset();

  EXPECT_EQ(presentation::output_policy{}.glyphs, presentation::glyph_set::emoji);
  EXPECT_EQ(presentation::terminal_policy(stdout).glyphs, presentation::glyph_set::emoji);
  EXPECT_EQ(presentation::plain_policy().glyphs, presentation::glyph_set::emoji);
  EXPECT_EQ(presentation::strict_ascii_policy().glyphs, presentation::glyph_set::ascii);

  presentation::styled_text text;
  text.append(presentation::semantic_glyph::warning);
  EXPECT_EQ(presentation::render(text, presentation::plain_policy()), "\xE2\x9A\xA0\xEF\xB8\x8F");
}

TEST(PresentationPolicies, TerminalPolicyUsesGlobalEnvironment)
{
  EnvVarGuard glyphs("UNI20_GLYPHS");
  EnvVarGuard charset("UNI20_CHARSET");
  EnvVarGuard color("UNI20_COLOR");
  glyphs.set("ascii");
  charset.set("ascii-escape");
  color.set("never");

  auto const policy = presentation::terminal_policy(stdout);

  EXPECT_EQ(policy.glyphs, presentation::glyph_set::ascii);
  EXPECT_EQ(policy.charset, presentation::text_charset::ascii_escape);
  EXPECT_EQ(policy.color, presentation::color_mode::never);
}

TEST(PresentationPolicies, TerminalPolicyAcceptsColorAliases)
{
  EnvVarGuard color("UNI20_COLOR");

  color.set("on");
  EXPECT_EQ(presentation::terminal_policy(stdout).color, presentation::color_mode::always);

  color.set("0");
  EXPECT_EQ(presentation::terminal_policy(stdout).color, presentation::color_mode::never);

  color.set("automatic");
  EXPECT_EQ(presentation::terminal_policy(stdout).color, presentation::color_mode::automatic);
}

TEST(PresentationPolicies, TerminalPolicyIgnoresInvalidGlobalEnvironment)
{
  EnvVarGuard glyphs("UNI20_GLYPHS");
  EnvVarGuard charset("UNI20_CHARSET");
  EnvVarGuard color("UNI20_COLOR");
  glyphs.set("unknown");
  charset.set("unknown");
  color.set("unknown");

  auto const policy = presentation::terminal_policy(stdout);

  EXPECT_EQ(policy.glyphs, presentation::glyph_set::emoji);
  EXPECT_EQ(policy.charset, presentation::text_charset::utf8);
  EXPECT_EQ(policy.color, presentation::color_mode::automatic);
}

TEST(PresentationPolicies, GlobalColorAlwaysOverridesNoColor)
{
  EnvVarGuard color("UNI20_COLOR");
  EnvVarGuard no_color("NO_COLOR");
  color.set("always");
  no_color.set("1");

  EXPECT_TRUE(presentation::should_emit_color(presentation::terminal_policy(stdout)));
}

TEST(PresentationGlyphs, UnicodePolicyUsesSemanticUnicodeGlyphs)
{
  auto policy = base_policy();
  policy.glyphs = presentation::glyph_set::unicode;

  presentation::styled_text text;
  text.append(presentation::semantic_glyph::success)
      .append(" ")
      .append(presentation::semantic_glyph::arrow_right)
      .append(" ")
      .append(presentation::semantic_glyph::fatal)
      .append(" ")
      .append(presentation::semantic_glyph::warning)
      .append(" ")
      .append(presentation::semantic_glyph::box_top_left)
      .append(presentation::semantic_glyph::box_horizontal)
      .append(presentation::semantic_glyph::box_top_right)
      .append(" ")
      .append(presentation::semantic_glyph::box_round_top_left)
      .append(presentation::semantic_glyph::box_horizontal)
      .append(presentation::semantic_glyph::box_round_top_right)
      .append(" ")
      .append(presentation::semantic_glyph::box_diagonal_forward)
      .append(presentation::semantic_glyph::box_diagonal_back);

  EXPECT_EQ(presentation::render(text, policy), "\xE2\x9C\x93 \xE2\x86\x92 \xE2\x80\xBC \xE2\x9A\xA0 "
                                                "\xE2\x94\x8C\xE2\x94\x80\xE2\x94\x90 "
                                                "\xE2\x95\xAD\xE2\x94\x80\xE2\x95\xAE "
                                                "\xE2\x95\xB1\xE2\x95\xB2");
}

TEST(PresentationGlyphs, EmojiPolicyUsesEmojiOnlyForSemanticMappings)
{
  auto policy = base_policy();
  policy.glyphs = presentation::glyph_set::emoji;

  presentation::styled_text text;
  text.append(presentation::semantic_glyph::success)
      .append(" ")
      .append(presentation::semantic_glyph::failure)
      .append(" ")
      .append(presentation::semantic_glyph::fatal)
      .append(" ")
      .append(presentation::semantic_glyph::warning)
      .append(" ")
      .append(presentation::semantic_glyph::arrow_right);

  EXPECT_EQ(presentation::render(text, policy),
            "\xE2\x9C\x85 \xE2\x9D\x8C \xF0\x9F\x9A\xA8 \xE2\x9A\xA0\xEF\xB8\x8F \xE2\x9E\xA1\xEF\xB8\x8F");
}

TEST(PresentationGlyphs, AsciiPolicyUsesCentralFallbackMappings)
{
  auto policy = base_policy();
  policy.glyphs = presentation::glyph_set::ascii;

  presentation::styled_text text;
  text.append(presentation::semantic_glyph::success)
      .append(" ")
      .append(presentation::semantic_glyph::fatal)
      .append(" ")
      .append(presentation::semantic_glyph::warning)
      .append(" ")
      .append(presentation::semantic_glyph::arrow_right)
      .append(" ")
      .append(presentation::semantic_glyph::tree_branch)
      .append(" ")
      .append(presentation::semantic_glyph::box_round_top_left)
      .append(presentation::semantic_glyph::box_diagonal_forward)
      .append(presentation::semantic_glyph::box_diagonal_back);

  EXPECT_EQ(presentation::render(text, policy), "[OK] [FATAL] [WARN] -> |- +/\\");
}

TEST(PresentationTextFallback, RawSymbolFallbackCoversCommonNonLanguageSymbols)
{
  auto policy = base_policy();
  policy.charset = presentation::text_charset::ascii_escape;

  auto const text = std::string("quote \xE2\x80\x9Cword\xE2\x80\x9D \xE2\x80\x94 "
                                "\xE2\x80\xA6 \xE2\x86\x92 \xE2\x89\xA4 "
                                "\xE2\x96\xB2 \xE2\x95\xAD\xE2\x95\xB1\xE2\x95\xB2");

  EXPECT_EQ(presentation::render_text(text, policy), "quote \"word\" - ... -> <= ! +/\\");
}

TEST(PresentationTextFallback, HumanUtf8TextIsPreservedEscapedOrReplaced)
{
  auto policy = base_policy();
  auto const chinese = std::string("\xE4\xB8\xAD\xE6\x96\x87");

  policy.charset = presentation::text_charset::utf8;
  EXPECT_EQ(presentation::render_text(chinese, policy), chinese);

  policy.charset = presentation::text_charset::ascii_escape;
  EXPECT_EQ(presentation::render_text(chinese, policy), "\\u4E2D\\u6587");

  policy.charset = presentation::text_charset::ascii_replace;
  EXPECT_EQ(presentation::render_text(chinese, policy), "??");
}

TEST(PresentationWidth, CoversAsciiStyledCombiningCjkAmbiguousEmojiAndTabs)
{
  auto policy = base_policy();

  EXPECT_EQ(presentation::display_width("abc", policy), 3U);

  presentation::styled_text styled;
  styled.append("abc", terminal::TerminalStyle(std::string("Red;Bold")));
  EXPECT_EQ(presentation::display_width(styled, policy), 3U);

  EXPECT_EQ(presentation::display_width("e\xCC\x81", policy), 1U);
  EXPECT_EQ(presentation::display_width("\xE4\xB8\xAD", policy), 2U);
  EXPECT_EQ(presentation::display_width("\xE2\x9C\x93", policy), 1U);
  EXPECT_EQ(presentation::display_width("\xF0\x9F\x98\x80", policy), 2U);
  EXPECT_EQ(presentation::display_width("\xE2\x9A\xA0\xEF\xB8\x8F", policy), 2U);
  EXPECT_EQ(presentation::display_width("a\tb", policy), 5U);
  EXPECT_EQ(presentation::display_width("a\tb", policy, 2), 3U);

  policy.ambiguous = presentation::ambiguous_width::narrow;
  EXPECT_EQ(presentation::display_width("\xCE\xA9", policy), 1U);
  policy.ambiguous = presentation::ambiguous_width::wide;
  EXPECT_EQ(presentation::display_width("\xCE\xA9", policy), 2U);
}

TEST(PresentationWidth, MeasuresAfterAsciiFallback)
{
  auto policy = base_policy();
  policy.charset = presentation::text_charset::ascii_escape;

  EXPECT_EQ(presentation::render_text("x\xE2\x86\x92\xE4\xB8\xAD", policy), "x->\\u4E2D");
  EXPECT_EQ(presentation::display_width("x\xE2\x86\x92\xE4\xB8\xAD", policy), 9U);
}

TEST(PresentationWidth, BytesModeCountsRenderedBytesWithoutAnsiWidth)
{
  auto policy = base_policy();
  policy.width = presentation::width_mode::bytes;

  EXPECT_EQ(presentation::display_width("\xE4\xB8\xAD", policy), 3U);
  EXPECT_EQ(presentation::display_width("\033[31mX\033[0m", policy), 1U);

  policy.charset = presentation::text_charset::ascii_escape;
  EXPECT_EQ(presentation::display_width("\xE4\xB8\xAD", policy), 6U);
}

TEST(PresentationLayout, PadsClipsTruncatesAndWrapsByDisplayCells)
{
  auto policy = base_policy();

  EXPECT_EQ(presentation::pad_right("\xE4\xB8\xAD", 4, policy), "\xE4\xB8\xAD  ");
  EXPECT_EQ(presentation::pad_left("ab", 4, policy), "  ab");
  EXPECT_EQ(presentation::pad_center("ab", 5, policy), " ab  ");

  policy.charset = presentation::text_charset::ascii_escape;
  EXPECT_EQ(presentation::clip_to_width("a\xE2\x86\x92"
                                        "b",
                                        3, policy),
            "a->");
  EXPECT_EQ(presentation::truncate_to_width("abcdef", 4, policy, "\xE2\x80\xA6"), "a...");
  EXPECT_EQ(presentation::truncate_left_to_width("abcdef", 4, policy, "\xE2\x80\xA6"), "...f");

  auto lines = presentation::wrap_text("abcd", 2, policy);
  ASSERT_EQ(lines.size(), 2U);
  EXPECT_EQ(lines[0], "ab");
  EXPECT_EQ(lines[1], "cd");

  lines = presentation::wrap_text("alpha beta", 6, policy);
  ASSERT_EQ(lines.size(), 2U);
  EXPECT_EQ(lines[0], "alpha");
  EXPECT_EQ(lines[1], "beta");
}

TEST(PresentationLayout, LeftTruncationPreservesDisplayCellSuffix)
{
  auto policy = base_policy();

  EXPECT_EQ(presentation::truncate_left_to_width("abcdef", 4, policy, "\xE2\x80\xA6"), "\xE2\x80\xA6"
                                                                                       "def");
  EXPECT_EQ(presentation::truncate_left_to_width("abc\xE4\xB8\xAD"
                                                 "def",
                                                 5, policy, "\xE2\x80\xA6"),
            "\xE2\x80\xA6"
            "def");

  policy.charset = presentation::text_charset::ascii_replace;
  EXPECT_EQ(presentation::truncate_left_to_width("abc\xE4\xB8\xAD"
                                                 "def",
                                                 5, policy, "..."),
            "...ef");
}

TEST(PresentationLayout, PrefixesAndIndentsRenderedLines)
{
  auto policy = base_policy();

  EXPECT_EQ(presentation::prefix_lines("alpha\n\xE4\xB8\xAD"
                                       "beta",
                                       "| ", policy),
            "| alpha\n| \xE4\xB8\xAD"
            "beta");
  EXPECT_EQ(presentation::prefix_lines("alpha\nbeta\n", "| ", policy, false), "alpha\n| beta\n");
  EXPECT_EQ(presentation::indent_text("alpha\nbeta", 3, policy), "   alpha\n   beta");

  policy.charset = presentation::text_charset::ascii_escape;
  EXPECT_EQ(presentation::prefix_lines("\xE4\xB8\xAD\n\xE2\x86\x92", "\xE2\x94\x82 ", policy), "| \\u4E2D\n| ->");
}

TEST(PresentationNumeric, RealAndComplexFormattingUsesConfiguredDigits)
{
  presentation::numeric_format_options options;
  options.float32_precision = 3;
  options.float64_precision = 4;

  EXPECT_EQ(presentation::format_real(1.0, options), "1");
  EXPECT_EQ(presentation::format_real(3.141592653589793, options), "3.142");
  EXPECT_EQ(presentation::format_real(1.0f / 3.0f, options), "0.333");
  EXPECT_EQ(presentation::format_complex(std::complex<double>{1.25, -3.5}, options), "1.25-3.5i");

  options.notation = presentation::real_notation::fixed;
  options.float64_precision = 2;
  EXPECT_EQ(presentation::format_real(-0.0, options), "0.00");
  EXPECT_EQ(presentation::format_complex(std::complex<double>{1.25, -3.5}, options), "1.25-3.50i");

  options.notation = presentation::real_notation::scientific;
  EXPECT_EQ(presentation::format_real(12.5, options), "1.25e+01");
}

TEST(PresentationNumeric, NonFiniteFormattingIsDeterministic)
{
  presentation::numeric_format_options options;
  options.notation = presentation::real_notation::fixed;
  options.float64_precision = 3;

  auto const inf = std::numeric_limits<double>::infinity();
  auto const nan = std::numeric_limits<double>::quiet_NaN();

  EXPECT_EQ(presentation::format_real(inf, options), "inf");
  EXPECT_EQ(presentation::format_real(-inf, options), "-inf");
  EXPECT_EQ(presentation::format_real(nan, options), "nan");

  options.notation = presentation::real_notation::scientific;
  EXPECT_EQ(presentation::format_real(inf, options), "inf");
  EXPECT_EQ(presentation::format_complex(std::complex<double>{-inf, nan}, options), "-inf+nani");
  EXPECT_EQ(presentation::format_complex(std::complex<double>{nan, -inf}, options), "nan-infi");
}

TEST(PresentationMdspan, DefaultFormatterHandlesRealAndComplexScalars)
{
  auto policy = base_policy();

  presentation::mdspan_format_options options;
  options.numeric.float64_precision = 4;

  std::array<double, 4> real_data{1.0, 2.0 / 3.0, -12.5, 1000.0};
  stdex::mdspan<double, stdex::extents<std::size_t, 2, 2>> real_matrix(real_data.data());

  EXPECT_EQ(presentation::format_mdspan(real_matrix, policy, options), "shape=(2, 2)\n"
                                                                       "\xE2\x8E\xA1     1 0.6667 \xE2\x8E\xA4\n"
                                                                       "\xE2\x8E\xA3 -12.5   1000 \xE2\x8E\xA6");

  options.numeric.notation = presentation::real_notation::fixed;
  options.numeric.float64_precision = 2;
  std::array<std::complex<double>, 2> complex_data{std::complex<double>{1.0, -2.5}, std::complex<double>{3.125, 0.0}};
  stdex::mdspan<std::complex<double>, stdex::extents<std::size_t, 2>> complex_vector(complex_data.data());

  EXPECT_EQ(presentation::format_mdspan(complex_vector, policy, options), "shape=(2)\n[ 1.00-2.50i 3.12+0.00i ]");
}

TEST(PresentationMdspan, UnicodeMatrixUsesDisplayCellAlignedBrackets)
{
  auto policy = base_policy();
  std::array<int, 6> data{1, 200, 30, 4, 5, 6};
  stdex::mdspan<int, stdex::extents<std::size_t, 3, 2>> matrix(data.data());

  auto rendered = presentation::format_mdspan(matrix, policy, [](int value) { return std::to_string(value); });

  EXPECT_EQ(rendered, "shape=(3, 2)\n"
                      "\xE2\x8E\xA1  1 200 \xE2\x8E\xA4\n"
                      "\xE2\x8E\xA2 30   4 \xE2\x8E\xA5\n"
                      "\xE2\x8E\xA3  5   6 \xE2\x8E\xA6");
}

TEST(PresentationMdspan, AsciiPolicyUsesAsciiMatrixArt)
{
  auto policy = base_policy();
  policy.glyphs = presentation::glyph_set::ascii;
  std::array<int, 4> data{1, 20, 3, 4};
  stdex::mdspan<int, stdex::extents<std::size_t, 2, 2>> matrix(data.data());

  auto rendered = presentation::format_mdspan(matrix, policy, [](int value) { return std::to_string(value); });

  EXPECT_EQ(rendered, "shape=(2, 2)\n"
                      "[ 1 20 ]\n"
                      "[ 3  4 ]");
}

TEST(PresentationMdspan, CjkCellsAlignByDisplayWidth)
{
  auto policy = base_policy();
  std::array<std::string, 4> data{"x", "\xE4\xB8\xAD", "long", "y"};
  stdex::mdspan<std::string, stdex::extents<std::size_t, 2, 2>> matrix(data.data());

  auto rendered = presentation::format_mdspan(matrix, policy, [](std::string const& value) { return value; });

  EXPECT_EQ(rendered, "shape=(2, 2)\n"
                      "\xE2\x8E\xA1    x \xE4\xB8\xAD \xE2\x8E\xA4\n"
                      "\xE2\x8E\xA3 long  y \xE2\x8E\xA6");
}

TEST(PresentationMdspan, MatrixAxesCanTransposeRankTwoViews)
{
  auto policy = base_policy();
  std::array<int, 6> data{1, 20, 300, 4, 5, 6};
  stdex::mdspan<int, stdex::extents<std::size_t, 2, 3>> matrix(data.data());

  presentation::mdspan_format_options options;
  options.matrix_axes = presentation::mdspan_matrix_axes{1, 0};

  auto rendered = presentation::format_mdspan(matrix, policy, options);

  EXPECT_EQ(rendered, "shape=(2, 3)\n"
                      "\xE2\x8E\xA1   1 4 \xE2\x8E\xA4\n"
                      "\xE2\x8E\xA2  20 5 \xE2\x8E\xA5\n"
                      "\xE2\x8E\xA3 300 6 \xE2\x8E\xA6");
}

TEST(PresentationMdspan, HigherRankTensorsRenderLabeledMatrixSlices)
{
  auto policy = base_policy();
  std::array<int, 8> data{1, 2, 3, 4, 5, 6, 7, 8};
  stdex::mdspan<int, stdex::extents<std::size_t, 2, 2, 2>> tensor(data.data());

  auto rendered = presentation::format_mdspan(tensor, policy, [](int value) { return std::to_string(value); });

  EXPECT_EQ(rendered, "shape=(2, 2, 2)\n"
                      "slice [0, :, :]\n"
                      "\xE2\x8E\xA1 1 2 \xE2\x8E\xA4\n"
                      "\xE2\x8E\xA3 3 4 \xE2\x8E\xA6\n"
                      "\n"
                      "slice [1, :, :]\n"
                      "\xE2\x8E\xA1 5 6 \xE2\x8E\xA4\n"
                      "\xE2\x8E\xA3 7 8 \xE2\x8E\xA6");
}

TEST(PresentationMdspan, MatrixAxesCanSelectRankThreeView)
{
  auto policy = base_policy();
  std::array<int, 12> data{};
  for (std::size_t i = 0; i < data.size(); ++i)
    data[i] = static_cast<int>(i);

  stdex::mdspan<int, stdex::extents<std::size_t, 2, 3, 2>> tensor(data.data());

  presentation::mdspan_format_options options;
  options.matrix_axes = presentation::mdspan_matrix_axes{0, 2};

  auto rendered = presentation::format_mdspan(tensor, policy, options);

  EXPECT_EQ(rendered, "shape=(2, 3, 2)\n"
                      "slice [:, 0, :]\n"
                      "\xE2\x8E\xA1 0 1 \xE2\x8E\xA4\n"
                      "\xE2\x8E\xA3 6 7 \xE2\x8E\xA6\n"
                      "\n"
                      "slice [:, 1, :]\n"
                      "\xE2\x8E\xA1 2 3 \xE2\x8E\xA4\n"
                      "\xE2\x8E\xA3 8 9 \xE2\x8E\xA6\n"
                      "\n"
                      "slice [:, 2, :]\n"
                      "\xE2\x8E\xA1  4  5 \xE2\x8E\xA4\n"
                      "\xE2\x8E\xA3 10 11 \xE2\x8E\xA6");
}

TEST(PresentationMdspan, RankFourTensorRenderingIsExhaustive)
{
  auto policy = base_policy();
  std::array<int, 16> data{};
  for (std::size_t i = 0; i < data.size(); ++i)
    data[i] = static_cast<int>(i);

  stdex::mdspan<int, stdex::extents<std::size_t, 2, 2, 2, 2>> tensor(data.data());

  auto rendered = presentation::format_mdspan(tensor, policy, [](int value) { return std::to_string(value); });

  EXPECT_EQ(rendered, "shape=(2, 2, 2, 2)\n"
                      "slice [0, 0, :, :]\n"
                      "\xE2\x8E\xA1 0 1 \xE2\x8E\xA4\n"
                      "\xE2\x8E\xA3 2 3 \xE2\x8E\xA6\n"
                      "\n"
                      "slice [0, 1, :, :]\n"
                      "\xE2\x8E\xA1 4 5 \xE2\x8E\xA4\n"
                      "\xE2\x8E\xA3 6 7 \xE2\x8E\xA6\n"
                      "\n"
                      "slice [1, 0, :, :]\n"
                      "\xE2\x8E\xA1  8  9 \xE2\x8E\xA4\n"
                      "\xE2\x8E\xA3 10 11 \xE2\x8E\xA6\n"
                      "\n"
                      "slice [1, 1, :, :]\n"
                      "\xE2\x8E\xA1 12 13 \xE2\x8E\xA4\n"
                      "\xE2\x8E\xA3 14 15 \xE2\x8E\xA6");
}

TEST(PresentationMdspan, MatrixAxesRejectInvalidChoices)
{
  auto policy = base_policy();
  std::array<int, 4> data{1, 2, 3, 4};
  stdex::mdspan<int, stdex::extents<std::size_t, 2, 2>> matrix(data.data());

  presentation::mdspan_format_options options;
  options.matrix_axes = presentation::mdspan_matrix_axes{0, 0};
  EXPECT_THROW((void)presentation::format_mdspan(matrix, policy, options), std::invalid_argument);

  options.matrix_axes = presentation::mdspan_matrix_axes{0, 2};
  EXPECT_THROW((void)presentation::format_mdspan(matrix, policy, options), std::invalid_argument);
}

TEST(PresentationInvalidUtf8, InvalidBytesAreEscapedOrReplacedByPolicy)
{
  auto policy = base_policy();
  std::string invalid;
  invalid.push_back('A');
  invalid.push_back(static_cast<char>(0xFF));
  invalid.push_back('B');

  policy.invalid = presentation::invalid_utf8::escape;
  EXPECT_EQ(presentation::render_text(invalid, policy), "A\\xFFB");
  EXPECT_EQ(presentation::display_width(invalid, policy), 6U);

  policy.invalid = presentation::invalid_utf8::replace;
  policy.charset = presentation::text_charset::ascii_replace;
  EXPECT_EQ(presentation::render_text(invalid, policy), "A?B");
  EXPECT_EQ(presentation::display_width(invalid, policy), 3U);
}

TEST(PresentationRenderers, TerminalColorIsPolicyControlled)
{
  presentation::styled_text text;
  text.append("red", terminal::TerminalStyle(std::string("Red")));

  auto policy = base_policy();
  policy.color = presentation::color_mode::always;
  EXPECT_EQ(presentation::render(text, policy), "\033[31mred\033[0m");

  policy.color = presentation::color_mode::never;
  EXPECT_EQ(presentation::render(text, policy), "red");
  EXPECT_EQ(presentation::render_plain(text, policy), "red");
}

TEST(PresentationReportBuilder, RendersStatusFieldsAndAlignedTables)
{
  auto policy = base_policy();

  presentation::report_builder report("Krylov solve");
  report.status(presentation::semantic_glyph::success, "converged").field("matrix", "demo").field("dimension", 128);

  report.table("Solver Summary")
      .column("solver", presentation::table_alignment::left)
      .column("matvecs")
      .column("residual")
      .row("native", 185, "1.0e-15")
      .row("arpack", 238, "8.0e-13");

  EXPECT_EQ(presentation::render_plain(report, policy), "Krylov solve\n"
                                                        "\xE2\x9C\x93 converged\n"
                                                        "  matrix     demo\n"
                                                        "  dimension  128\n"
                                                        "Solver Summary\n"
                                                        "  solver  matvecs  residual\n"
                                                        "  native      185   1.0e-15\n"
                                                        "  arpack      238   8.0e-13\n");
}

TEST(PresentationReportBuilder, TableReferencesRemainStableWhenAddingTables)
{
  presentation::report_builder report("Stable tables");
  auto& first = report.table("first");

  for (int i = 0; i < 16; ++i)
  {
    report.table(fmt::format("extra {}", i));
  }

  first.column("name", presentation::table_alignment::left).row("alpha");

  auto const& entries = report.tables().front().entries();
  ASSERT_EQ(entries.size(), 1U);
  auto const* row = std::get_if<std::vector<presentation::table_cell>>(&entries.front());
  ASSERT_NE(row, nullptr);
  ASSERT_EQ(row->size(), 1U);
  EXPECT_EQ(row->front().text, "alpha");
}

TEST(PresentationReportBuilder, UsesSemanticGlyphFallbackInAscii)
{
  auto policy = presentation::strict_ascii_policy();

  presentation::report_builder report("Status");
  report.status(presentation::semantic_glyph::success, "converged");

  EXPECT_EQ(presentation::render_plain(report, policy), "Status\n"
                                                        "[OK] converged\n");
}

TEST(PresentationReportBuilder, RendersAsciiGridTableWithIntersections)
{
  auto policy = presentation::strict_ascii_policy();

  presentation::report_builder report("Grid report");
  report.table("Solver Summary")
      .grid()
      .column("solver", presentation::table_alignment::left)
      .column("matvecs")
      .column("residual")
      .row("native", 185, "1.0e-15")
      .row("arpack", 238, "8.0e-13");

  EXPECT_EQ(presentation::render_plain(report, policy), "Grid report\n"
                                                        "Solver Summary\n"
                                                        "  +--------+---------+----------+\n"
                                                        "  | solver | matvecs | residual |\n"
                                                        "  +--------+---------+----------+\n"
                                                        "  | native |     185 |  1.0e-15 |\n"
                                                        "  +--------+---------+----------+\n"
                                                        "  | arpack |     238 |  8.0e-13 |\n"
                                                        "  +--------+---------+----------+\n");
}

TEST(PresentationReportBuilder, RendersCellsSpanningMultipleColumns)
{
  auto policy = presentation::strict_ascii_policy();
  policy.wrap_width = 32;

  presentation::report_builder report("Span report");
  report.table("Spans")
      .grid()
      .column("name", presentation::table_alignment::left)
      .column("detail", presentation::table_alignment::left)
      .column("value")
      .row({{"alpha", 1}, {"spans detail and value", 2}})
      .row("beta", "plain", 7);

  EXPECT_EQ(presentation::render_plain(report, policy), "Span report\n"
                                                        "Spans\n"
                                                        "  +-------+---------+----------+\n"
                                                        "  | name  | detail  |    value |\n"
                                                        "  +-------+---------+----------+\n"
                                                        "  | alpha | spans detail and   |\n"
                                                        "  |       | value              |\n"
                                                        "  +-------+---------+----------+\n"
                                                        "  | beta  | plain   |        7 |\n"
                                                        "  +-------+---------+----------+\n");
}

TEST(PresentationReportBuilder, RendersManualSeparatorsWithoutGlobalRowSeparators)
{
  auto policy = presentation::strict_ascii_policy();

  presentation::report_builder report("Manual rules");
  report.table("Phases")
      .outer_border()
      .column_separators()
      .header_separator()
      .top_separator(presentation::table_rule_style::double_line)
      .column("phase", presentation::table_alignment::left)
      .column("count")
      .row("setup", 2)
      .separator()
      .row("solve", 7)
      .row("cleanup", 1)
      .separator(presentation::table_rule_style::double_line);

  EXPECT_EQ(presentation::render_plain(report, policy), "Manual rules\n"
                                                        "Phases\n"
                                                        "  +---------+-------+\n"
                                                        "  +=========+=======+\n"
                                                        "  | phase   | count |\n"
                                                        "  +---------+-------+\n"
                                                        "  | setup   |     2 |\n"
                                                        "  +---------+-------+\n"
                                                        "  | solve   |     7 |\n"
                                                        "  | cleanup |     1 |\n"
                                                        "  +=========+=======+\n"
                                                        "  +---------+-------+\n");
}

TEST(PresentationReportBuilder, RendersDoubleLineGridWithAsciiFallback)
{
  auto policy = presentation::strict_ascii_policy();

  presentation::report_builder report("Double grid");
  report.table("Summary")
      .grid(presentation::table_rule_style::double_line)
      .column("key", presentation::table_alignment::left)
      .column("value")
      .row("alpha", 12);

  EXPECT_EQ(presentation::render_plain(report, policy), "Double grid\n"
                                                        "Summary\n"
                                                        "  +=======+=======+\n"
                                                        "  | key   | value |\n"
                                                        "  +=======+=======+\n"
                                                        "  | alpha |    12 |\n"
                                                        "  +=======+=======+\n");
}

TEST(PresentationReportBuilder, RendersMixedSingleSeparatorInsideDoubleGrid)
{
  auto policy = base_policy();

  presentation::report_builder report("Mixed rules");
  report.table("Rules")
      .grid(presentation::table_rule_style::double_line)
      .column("name", presentation::table_alignment::left)
      .column("note", presentation::table_alignment::left)
      .column("value")
      .row("alpha", "plain", 1)
      .separator()
      .row({{"notes", 1}, {"spanning text", 2, presentation::table_alignment::left}});

  EXPECT_EQ(presentation::render_plain(report, policy), "Mixed rules\n"
                                                        "Rules\n"
                                                        "  ╔═══════╦═══════╦═══════╗\n"
                                                        "  ║ name  ║ note  ║ value ║\n"
                                                        "  ╠═══════╬═══════╬═══════╣\n"
                                                        "  ║ alpha ║ plain ║     1 ║\n"
                                                        "  ╟───────╫───────╨───────╢\n"
                                                        "  ║ notes ║ spanning text ║\n"
                                                        "  ╚═══════╩═══════════════╝\n");
}

TEST(PresentationReportBuilder, AlignsDecimalColumnsAndCellOverrides)
{
  auto policy = presentation::strict_ascii_policy();

  presentation::report_builder report("Decimal report");
  report.table("Values")
      .outer_border()
      .column_separators()
      .column("name", presentation::table_alignment::left)
      .column("value", presentation::table_alignment::decimal)
      .column("note", presentation::table_alignment::left)
      .row("alpha", "1.25", "base")
      .row("beta", "12", "integer")
      .row({{"gamma", 1}, {"3.5", 1}, {"center", 1, presentation::table_alignment::center}});

  EXPECT_EQ(presentation::render_plain(report, policy), "Decimal report\n"
                                                        "Values\n"
                                                        "  +-------+-------+---------+\n"
                                                        "  | name  | value | note    |\n"
                                                        "  | alpha |  1.25 | base    |\n"
                                                        "  | beta  | 12    | integer |\n"
                                                        "  | gamma |  3.5  | center  |\n"
                                                        "  +-------+-------+---------+\n");
}

TEST(PresentationReportBuilder, DecimalAlignmentExpandsColumnForLeftPadding)
{
  auto policy = presentation::strict_ascii_policy();

  presentation::report_builder report("Decimal width");
  report.table("Values")
      .grid()
      .column("label", presentation::table_alignment::left)
      .column("value", presentation::table_alignment::decimal)
      .row("short", "1.0e-15")
      .row("wide", "123");

  EXPECT_EQ(presentation::render_plain(report, policy), "Decimal width\n"
                                                        "Values\n"
                                                        "  +-------+-----------+\n"
                                                        "  | label | value     |\n"
                                                        "  +-------+-----------+\n"
                                                        "  | short |   1.0e-15 |\n"
                                                        "  +-------+-----------+\n"
                                                        "  | wide  | 123       |\n"
                                                        "  +-------+-----------+\n");
}

TEST(PresentationReportBuilder, DecimalAlignmentCentersNonFiniteValues)
{
  auto policy = presentation::strict_ascii_policy();

  presentation::report_builder report("Nonfinite decimal");
  report.table("Values")
      .grid()
      .column("label", presentation::table_alignment::left)
      .column("value", presentation::table_alignment::decimal)
      .row("finite", "12.5")
      .row("pos", "inf")
      .row("neg", "-inf")
      .row("bad", "nan");

  EXPECT_EQ(presentation::render_plain(report, policy), "Nonfinite decimal\n"
                                                        "Values\n"
                                                        "  +--------+-------+\n"
                                                        "  | label  | value |\n"
                                                        "  +--------+-------+\n"
                                                        "  | finite | 12.5  |\n"
                                                        "  +--------+-------+\n"
                                                        "  | pos    |  inf  |\n"
                                                        "  +--------+-------+\n"
                                                        "  | neg    | -inf  |\n"
                                                        "  +--------+-------+\n"
                                                        "  | bad    |  nan  |\n"
                                                        "  +--------+-------+\n");
}

TEST(PresentationReportBuilder, ExplicitSeparatorsUseRuledWidthBudget)
{
  auto policy = presentation::strict_ascii_policy();
  policy.wrap_width = 18;

  presentation::report_builder report("Manual fit");
  report.table("")
      .column("item", presentation::table_alignment::left)
      .column("note", presentation::table_alignment::left)
      .row("id", "alpha beta gamma")
      .separator()
      .row("tail", "done");

  auto const rendered = presentation::render_plain(report, policy);
  EXPECT_NE(rendered.find("--------"), std::string::npos);

  std::size_t start = 0;
  while (start < rendered.size())
  {
    auto const newline = rendered.find('\n', start);
    auto const line = rendered.substr(start, newline == std::string::npos ? std::string::npos : newline - start);
    EXPECT_LE(presentation::display_width(line, policy), *policy.wrap_width) << line;
    if (newline == std::string::npos) break;
    start = newline + 1;
  }
}

TEST(PresentationReportBuilder, WrapsCellsInsideTableWidth)
{
  auto policy = presentation::strict_ascii_policy();
  policy.wrap_width = 22;

  presentation::report_builder report("Grid report");
  report.table("Wrapped")
      .grid()
      .column("name", presentation::table_alignment::left)
      .column("note", presentation::table_alignment::left)
      .row("alpha", "abcdefghijklmno");

  EXPECT_EQ(presentation::render_plain(report, policy), "Grid report\n"
                                                        "Wrapped\n"
                                                        "  +-------+----------+\n"
                                                        "  | name  | note     |\n"
                                                        "  +-------+----------+\n"
                                                        "  | alpha | abcdefgh |\n"
                                                        "  |       | ijklmno  |\n"
                                                        "  +-------+----------+\n");
}

TEST(PresentationReportBuilder, PrefersWhitespaceWrapsWhenFittingColumns)
{
  auto policy = presentation::strict_ascii_policy();
  policy.wrap_width = 34;

  presentation::report_builder report("Fit report");
  report.table("Prefer soft wraps")
      .grid()
      .column("id", presentation::table_alignment::left)
      .column("note", presentation::table_alignment::left)
      .row("abcdefghijklmno", "alpha beta gamma");

  EXPECT_EQ(presentation::render_plain(report, policy), "Fit report\n"
                                                        "Prefer soft wraps\n"
                                                        "  +-----------------+------------+\n"
                                                        "  | id              | note       |\n"
                                                        "  +-----------------+------------+\n"
                                                        "  | abcdefghijklmno | alpha beta |\n"
                                                        "  |                 | gamma      |\n"
                                                        "  +-----------------+------------+\n");
}
