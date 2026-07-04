#include <uni20/common/display.hpp>
#include <uni20/common/presentation.hpp>

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace
{
namespace display = uni20::display;
namespace presentation = uni20::presentation;

[[nodiscard]] presentation::output_policy ascii_policy()
{
  auto policy = presentation::strict_ascii_policy();
  policy.color = presentation::color_mode::never;
  return policy;
}

[[nodiscard]] std::string render_ascii(display::event const& event)
{
  auto policy = ascii_policy();
  return std::visit([&](auto const& content) { return presentation::render_plain(content, policy); }, event.content);
}

[[nodiscard]] std::vector<std::string> split_lines(std::string const& text)
{
  std::vector<std::string> lines;
  std::string current;
  for (char ch : text)
  {
    if (ch == '\n')
    {
      lines.push_back(current);
      current.clear();
    }
    else
    {
      current.push_back(ch);
    }
  }
  lines.push_back(current);
  return lines;
}

void expect_no_trailing_spaces(std::string const& text)
{
  for (auto const& line : split_lines(text))
  {
    if (!line.empty())
    {
      EXPECT_NE(line.back(), ' ');
    }
  }
}

} // namespace

TEST(DisplaySink, StatusHelpersEmitSemanticStyledText)
{
  std::vector<display::event> events;
  display::scoped_sink capture([&](display::event const& event) { events.push_back(event); });

  display::success("converged in {} sweeps", 6);
  display::warning("bond cap {}", 256);
  display::partial("validated {} of {}", 3, 5);
  display::deferred("policy cleanup");
  display::skipped("CUDA backend unavailable");

  ASSERT_EQ(events.size(), 5U);
  EXPECT_EQ(events[0].destination, display::stream::err);
  EXPECT_TRUE(events[0].newline);
  EXPECT_TRUE(std::holds_alternative<presentation::styled_text>(events[0].content));

  EXPECT_EQ(render_ascii(events[0]), "[OK] converged in 6 sweeps");
  EXPECT_EQ(render_ascii(events[1]), "[WARN] bond cap 256");
  EXPECT_EQ(render_ascii(events[2]), "[PARTIAL] validated 3 of 5");
  EXPECT_EQ(render_ascii(events[3]), "[DEFER] policy cleanup");
  EXPECT_EQ(render_ascii(events[4]), "[SKIP] CUDA backend unavailable");
}

TEST(DisplaySink, StatusHelpersCaptureCallSite)
{
  std::vector<display::event> events;
  display::scoped_sink capture([&](display::event const& event) { events.push_back(event); });

  auto const line = __LINE__ + 1;
  display::info("located");

  ASSERT_EQ(events.size(), 1U);
  EXPECT_EQ(events[0].where.line(), line);
  EXPECT_NE(std::string(events[0].where.file_name()).find("test_display.cpp"), std::string::npos);
}

TEST(DisplaySink, StatusCellDoesNotEmit)
{
  std::vector<display::event> events;
  display::scoped_sink capture([&](display::event const& event) { events.push_back(event); });

  auto cell = display::status_cell(presentation::semantic_glyph::failure, "open finding {}", 10);

  EXPECT_TRUE(events.empty());
  EXPECT_EQ(presentation::render_plain(cell, ascii_policy()), "[FAIL] open finding 10");
}

TEST(DisplaySink, EmitCanRouteToStdoutAndSuppressTrailingNewline)
{
  std::vector<display::event> events;
  display::scoped_sink capture([&](display::event const& event) { events.push_back(event); });

  presentation::styled_text text;
  text.append("plain message");
  display::emit(std::move(text), display::stream::out, false);

  ASSERT_EQ(events.size(), 1U);
  EXPECT_EQ(events[0].destination, display::stream::out);
  EXPECT_FALSE(events[0].newline);
  EXPECT_EQ(render_ascii(events[0]), "plain message");
}

TEST(DisplaySink, EmitReportPreservesReportUntilSinkRendering)
{
  std::vector<display::event> events;
  display::scoped_sink capture([&](display::event const& event) { events.push_back(event); });

  presentation::report_builder report("Display report");
  report.status(presentation::semantic_glyph::info, "routing through sink").field("rows", 2);
  report.table("Summary")
      .grid()
      .column("state", presentation::table_alignment::left)
      .column("count")
      .row("accepted", 3)
      .row("skipped", 1);

  display::emit(std::move(report), display::stream::out);

  ASSERT_EQ(events.size(), 1U);
  EXPECT_EQ(events[0].destination, display::stream::out);
  EXPECT_TRUE(std::holds_alternative<presentation::report_builder>(events[0].content));

  auto const rendered = render_ascii(events[0]);
  EXPECT_NE(rendered.find("Display report"), std::string::npos);
  EXPECT_NE(rendered.find("[INFO] routing through sink"), std::string::npos);
  EXPECT_NE(rendered.find("accepted"), std::string::npos);
  EXPECT_NE(rendered.find("skipped"), std::string::npos);
}

TEST(DisplaySink, ScopedSinkRestoresPreviousSink)
{
  std::vector<display::event> outer_events;
  std::vector<display::event> inner_events;

  {
    display::scoped_sink outer([&](display::event const& event) { outer_events.push_back(event); });
    display::info("outer");

    {
      display::scoped_sink inner([&](display::event const& event) { inner_events.push_back(event); });
      display::info("inner");
    }

    display::info("outer again");
  }

  ASSERT_EQ(outer_events.size(), 2U);
  ASSERT_EQ(inner_events.size(), 1U);
  EXPECT_EQ(render_ascii(outer_events[0]), "[INFO] outer");
  EXPECT_EQ(render_ascii(inner_events[0]), "[INFO] inner");
  EXPECT_EQ(render_ascii(outer_events[1]), "[INFO] outer again");
}

TEST(DisplayStreamingTable, EmitsHeaderOnceAndKeepsLaterRowsCompact)
{
  std::vector<display::event> events;
  display::scoped_sink capture([&](display::event const& event) { events.push_back(event); });

  auto table = display::table("Sweeps");
  table.wrap_width(24)
      .column("sweep", display::width::fixed(5))
      .column("status", display::width::fixed(12), presentation::table_alignment::left);

  table.row(1, "accepted");
  table.row(2, "bond cap reached here");
  table.row(3, "ok");

  ASSERT_EQ(events.size(), 3U);
  auto const first = render_ascii(events[0]);
  EXPECT_NE(first.find("Sweeps"), std::string::npos);
  EXPECT_NE(first.find("sweep"), std::string::npos);
  EXPECT_NE(first.find("status"), std::string::npos);
  EXPECT_NE(first.find("accepted"), std::string::npos);
  expect_no_trailing_spaces(first);

  auto const second_lines = split_lines(render_ascii(events[1]));
  EXPECT_GT(second_lines.size(), 1U);
  EXPECT_NE(second_lines[1].find("->"), std::string::npos);
  for (auto const& line : second_lines)
  {
    EXPECT_LE(presentation::display_width(line, ascii_policy()), 24U);
  }
  expect_no_trailing_spaces(render_ascii(events[1]));

  auto const third_lines = split_lines(render_ascii(events[2]));
  ASSERT_EQ(third_lines.size(), 1U);
  EXPECT_NE(third_lines[0].find("ok"), std::string::npos);
  expect_no_trailing_spaces(render_ascii(events[2]));
}

TEST(DisplayStreamingTable, VeryNarrowWidthUsesVerticalFallback)
{
  std::vector<display::event> events;
  display::scoped_sink capture([&](display::event const& event) { events.push_back(event); });

  auto table = display::table("Tiny");
  table.wrap_width(16)
      .column("sweep", display::width::fixed(5))
      .column("status", display::width::fixed(12), presentation::table_alignment::left);

  table.row(2, "bond cap reached");

  ASSERT_EQ(events.size(), 1U);
  auto const rendered = render_ascii(events[0]);
  EXPECT_NE(rendered.find("Tiny"), std::string::npos);
  EXPECT_NE(rendered.find("sweep:"), std::string::npos);
  EXPECT_NE(rendered.find("status:"), std::string::npos);
  EXPECT_NE(rendered.find("bond"), std::string::npos);
  EXPECT_NE(rendered.find("->"), std::string::npos);
}

TEST(DisplayStreamingTable, WideSchemaUsesVerticalFallbackInsteadOfTerminalWrap)
{
  std::vector<display::event> events;
  display::scoped_sink capture([&](display::event const& event) { events.push_back(event); });

  auto table = display::table("Wide");
  table.wrap_width(28)
      .column("sweep", display::width::fixed(5))
      .column("energy", display::width::fixed(14), presentation::table_alignment::decimal)
      .column("delta", display::width::fixed(10), presentation::table_alignment::decimal)
      .column("bond", display::width::fixed(6))
      .column("status", display::width::fixed(12), presentation::table_alignment::left);

  table.row(2, "-12.456789", "-1.1e-1", 256, "accepted");

  ASSERT_EQ(events.size(), 1U);
  auto const rendered = render_ascii(events[0]);
  EXPECT_NE(rendered.find("sweep:"), std::string::npos);
  EXPECT_NE(rendered.find("energy:"), std::string::npos);
  EXPECT_EQ(rendered.find("sweep  energy"), std::string::npos);
  for (auto const& line : split_lines(rendered))
  {
    EXPECT_LE(presentation::display_width(line, ascii_policy()), 28U);
  }
  expect_no_trailing_spaces(rendered);
}

TEST(DisplayStreamingTable, LongTitleWrapsWithContinuationMarker)
{
  std::vector<display::event> events;
  display::scoped_sink capture([&](display::event const& event) { events.push_back(event); });

  auto table = display::table("Streaming diagnostics for an intentionally narrow terminal");
  table.wrap_width(28).column("sweep", display::width::fixed(5)).column("status", display::width::fixed(12));

  table.row(1, "ok");

  ASSERT_EQ(events.size(), 1U);
  auto const rendered = render_ascii(events[0]);
  auto const lines = split_lines(rendered);
  ASSERT_GE(lines.size(), 2U);
  EXPECT_NE(lines[1].find("->"), std::string::npos);
  for (auto const& line : lines)
  {
    EXPECT_LE(presentation::display_width(line, ascii_policy()), 28U);
  }
  expect_no_trailing_spaces(rendered);
}

TEST(DisplayStreamingTable, MarksWrappedCellEvenWhenFirstColumnWraps)
{
  std::vector<display::event> events;
  display::scoped_sink capture([&](display::event const& event) { events.push_back(event); });

  auto table = display::table("Labels");
  table.wrap_width(24)
      .column("label", display::width::fixed(10), presentation::table_alignment::left)
      .column("status", display::width::fixed(8), presentation::table_alignment::left);

  table.row("alpha beta gamma", "ok");

  ASSERT_EQ(events.size(), 1U);
  auto const lines = split_lines(render_ascii(events[0]));
  ASSERT_GE(lines.size(), 4U);
  EXPECT_NE(lines.back().find("->"), std::string::npos);
  EXPECT_NE(lines.back().find("gamma"), std::string::npos);
  for (auto const& line : lines)
  {
    EXPECT_LE(presentation::display_width(line, ascii_policy()), 24U);
  }
  expect_no_trailing_spaces(render_ascii(events[0]));
}

TEST(DisplayStreamingTable, DecimalAlignmentKeepsLaterShorterFractionsAligned)
{
  std::vector<display::event> events;
  display::scoped_sink capture([&](display::event const& event) { events.push_back(event); });

  auto table = display::table("Energy");
  table.wrap_width(60)
      .column("step", display::width::fixed(4))
      .column("energy", display::width::fixed(18), presentation::table_alignment::decimal, display::format::number())
      .column("state", display::width::fixed(6), presentation::table_alignment::left);

  table.row(1, -12.345678901234, "base");
  table.row(2, -12.467, "short");

  ASSERT_EQ(events.size(), 2U);
  auto const first_lines = split_lines(render_ascii(events[0]));
  ASSERT_FALSE(first_lines.empty());
  auto const first_row = first_lines.back();
  auto const second_lines = split_lines(render_ascii(events[1]));
  ASSERT_FALSE(second_lines.empty());
  auto const second_row = second_lines.front();

  ASSERT_NE(first_row.find('.'), std::string::npos);
  ASSERT_NE(second_row.find('.'), std::string::npos);
  EXPECT_EQ(first_row.find('.'), second_row.find('.'));
  expect_no_trailing_spaces(render_ascii(events[0]));
  expect_no_trailing_spaces(render_ascii(events[1]));
}

TEST(DisplayStreamingTable, ColumnNumericFormatAppliesToTypedValues)
{
  std::vector<display::event> events;
  display::scoped_sink capture([&](display::event const& event) { events.push_back(event); });

  auto table = display::table("Energy");
  table.wrap_width(48)
      .column("step", display::width::fixed(4))
      .column("energy", display::width::fixed(12), display::format::fixed(3));

  table.row(1, 1.25);

  ASSERT_EQ(events.size(), 1U);
  auto const rendered = render_ascii(events[0]);
  EXPECT_NE(rendered.find("1.250"), std::string::npos);
  EXPECT_EQ(rendered.find("1.25  "), std::string::npos);
}

TEST(DisplayStreamingTable, NumericFormatWithoutWidthUsesFitColumn)
{
  std::vector<display::event> events;
  display::scoped_sink capture([&](display::event const& event) { events.push_back(event); });

  auto table = display::table("Energy");
  table.wrap_width(48)
      .column("step", display::width::fixed(4))
      .column("energy", display::format::fixed(2))
      .column("state", display::width::fixed(5), presentation::table_alignment::left);

  table.row(1, 1234.5, "base");
  table.row(2, 1.25, "next");

  ASSERT_EQ(events.size(), 2U);
  auto const first_lines = split_lines(render_ascii(events[0]));
  auto const second_lines = split_lines(render_ascii(events[1]));
  ASSERT_FALSE(first_lines.empty());
  ASSERT_FALSE(second_lines.empty());

  auto const first_row = first_lines.back();
  auto const second_row = second_lines.front();
  ASSERT_NE(first_row.find('.'), std::string::npos);
  ASSERT_NE(second_row.find('.'), std::string::npos);
  EXPECT_EQ(first_row.find('.'), second_row.find('.'));
  EXPECT_NE(first_row.find("1234.50"), std::string::npos);
  EXPECT_NE(second_row.find("1.25"), std::string::npos);

  auto const header = first_lines.size() > 1 ? first_lines[1] : std::string{};
  auto const energy_position = header.find("energy");
  auto const state_position = header.find("state");
  ASSERT_NE(energy_position, std::string::npos);
  ASSERT_NE(state_position, std::string::npos);
  EXPECT_LT(state_position - energy_position, 14U);

  expect_no_trailing_spaces(render_ascii(events[0]));
  expect_no_trailing_spaces(render_ascii(events[1]));
}

TEST(DisplayStreamingTable, FitNumericColumnGrowsForLaterWideValues)
{
  std::vector<display::event> events;
  display::scoped_sink capture([&](display::event const& event) { events.push_back(event); });

  auto table = display::table("Residuals");
  table.wrap_width(44)
      .column("id", display::width::fixed(2))
      .column("value", display::format::fixed(2))
      .column("status", display::width::share(1, 8), presentation::table_alignment::left);

  table.row(1, 1.25, "accepted");
  table.row(2, 123456789.12, "accepted");

  ASSERT_EQ(events.size(), 2U);
  auto const second = render_ascii(events[1]);
  EXPECT_NE(second.find("123456789.12"), std::string::npos);
  EXPECT_EQ(second.find("->"), std::string::npos);
  for (auto const& line : split_lines(second))
  {
    EXPECT_LE(presentation::display_width(line, ascii_policy()), 44U);
  }
  expect_no_trailing_spaces(second);
}

TEST(DisplayStreamingTable, NumericColumnTreatsTextAsExceptionalCell)
{
  std::vector<display::event> events;
  display::scoped_sink capture([&](display::event const& event) { events.push_back(event); });

  auto table = display::table("Energy");
  table.wrap_width(64)
      .column("step", display::width::fixed(4))
      .column("energy", display::width::fixed(18), presentation::table_alignment::decimal, display::format::fixed(6))
      .column("state", display::width::fixed(8), presentation::table_alignment::left);

  table.row(1, -12.5, "base");
  table.row(2, "non-converged", "fail");
  table.row(3, -1.25, "after");

  ASSERT_EQ(events.size(), 3U);
  auto const first_lines = split_lines(render_ascii(events[0]));
  auto const second_lines = split_lines(render_ascii(events[1]));
  auto const third_lines = split_lines(render_ascii(events[2]));
  ASSERT_FALSE(first_lines.empty());
  ASSERT_FALSE(second_lines.empty());
  ASSERT_FALSE(third_lines.empty());

  auto const first_row = first_lines.back();
  auto const second_row = second_lines.front();
  auto const third_row = third_lines.front();

  EXPECT_NE(second_row.find("non-converged"), std::string::npos);
  EXPECT_EQ(second_row.find('.'), std::string::npos);
  ASSERT_NE(first_row.find('.'), std::string::npos);
  ASSERT_NE(third_row.find('.'), std::string::npos);
  EXPECT_EQ(first_row.find('.'), third_row.find('.'));
}

TEST(DisplayStreamingTable, RejectsMismatchedRowsAndLateSchemaChanges)
{
  auto table = display::table("Schema");
  table.wrap_width(40).column("a").column("b");

  EXPECT_THROW(table.row("only one cell"), std::invalid_argument);

  std::vector<display::event> events;
  display::scoped_sink capture([&](display::event const& event) { events.push_back(event); });
  table.row("left", "right");

  EXPECT_THROW(table.column("c"), std::logic_error);
  EXPECT_THROW(table.wrap_width(80), std::logic_error);
  EXPECT_THROW(table.header_separator(false), std::logic_error);
}

TEST(DisplayStreamingTable, PreservesStyledCells)
{
  std::vector<display::event> events;
  display::scoped_sink capture([&](display::event const& event) { events.push_back(event); });

  auto red_bold = presentation::style("Red;Bold");
  auto table = display::table("Styled");
  table.wrap_width(36)
      .column("sweep", display::width::fixed(5))
      .column("status", display::width::share(1), presentation::table_alignment::left);

  table.row(1, red_bold(presentation::semantic_glyph::warning, "stagnated"));

  ASSERT_EQ(events.size(), 1U);

  auto color_policy = ascii_policy();
  color_policy.color = presentation::color_mode::always;
  auto const rendered = std::visit(
      [&](auto const& content) { return presentation::render_terminal(content, color_policy); }, events[0].content);

  EXPECT_NE(rendered.find("\033[1;31m[WARN]\033[0m"), std::string::npos);
  EXPECT_NE(rendered.find("\033[1;31mstagnated\033[0m"), std::string::npos);
  EXPECT_NE(render_ascii(events[0]).find("[WARN] stagnated"), std::string::npos);
}
