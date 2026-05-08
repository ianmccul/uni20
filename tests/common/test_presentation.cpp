#include <uni20/common/presentation.hpp>
#include <uni20/common/presentation_mdspan.hpp>
#include <uni20/common/mdspan.hpp>

#include <gtest/gtest.h>

#include <array>
#include <complex>
#include <string>
#include <vector>

namespace
{
namespace presentation = uni20::presentation;

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

TEST(PresentationGlyphs, UnicodePolicyUsesSemanticUnicodeGlyphs)
{
  auto policy = base_policy();
  policy.glyphs = presentation::glyph_set::unicode;

  presentation::styled_text text;
  text.append(presentation::semantic_glyph::success)
      .append(" ")
      .append(presentation::semantic_glyph::arrow_right)
      .append(" ")
      .append(presentation::semantic_glyph::box_top_left)
      .append(presentation::semantic_glyph::box_horizontal)
      .append(presentation::semantic_glyph::box_top_right);

  EXPECT_EQ(presentation::render(text, policy), "\xE2\x9C\x93 \xE2\x86\x92 \xE2\x94\x8C\xE2\x94\x80\xE2\x94\x90");
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
      .append(presentation::semantic_glyph::arrow_right);

  EXPECT_EQ(presentation::render(text, policy), "\xE2\x9C\x85 \xE2\x9D\x8C \xE2\x9E\xA1\xEF\xB8\x8F");
}

TEST(PresentationGlyphs, AsciiPolicyUsesCentralFallbackMappings)
{
  auto policy = base_policy();
  policy.glyphs = presentation::glyph_set::ascii;

  presentation::styled_text text;
  text.append(presentation::semantic_glyph::success)
      .append(" ")
      .append(presentation::semantic_glyph::warning)
      .append(" ")
      .append(presentation::semantic_glyph::arrow_right)
      .append(" ")
      .append(presentation::semantic_glyph::tree_branch);

  EXPECT_EQ(presentation::render(text, policy), "[OK] [WARN] -> |-");
}

TEST(PresentationTextFallback, RawSymbolFallbackCoversCommonNonLanguageSymbols)
{
  auto policy = base_policy();
  policy.charset = presentation::text_charset::ascii_escape;

  auto const text = std::string("quote \xE2\x80\x9Cword\xE2\x80\x9D \xE2\x80\x94 "
                                "\xE2\x80\xA6 \xE2\x86\x92 \xE2\x89\xA4");

  EXPECT_EQ(presentation::render_text(text, policy), "quote \"word\" - ... -> <=");
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
  EXPECT_EQ(presentation::display_width("\xF0\x9F\x98\x80", policy), 2U);
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

  auto lines = presentation::wrap_text("abcd", 2, policy);
  ASSERT_EQ(lines.size(), 2U);
  EXPECT_EQ(lines[0], "ab");
  EXPECT_EQ(lines[1], "cd");
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
  std::array<std::complex<double>, 2> complex_data{std::complex<double>{1.0, -2.5},
                                                  std::complex<double>{3.125, 0.0}};
  stdex::mdspan<std::complex<double>, stdex::extents<std::size_t, 2>> complex_vector(complex_data.data());

  EXPECT_EQ(presentation::format_mdspan(complex_vector, policy, options),
            "shape=(2)\n[ 1.00-2.50i 3.12+0.00i ]");
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
