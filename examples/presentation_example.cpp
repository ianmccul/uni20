#include <uni20/common/presentation.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <fmt/core.h>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

namespace
{
namespace presentation = uni20::presentation;

[[nodiscard]] presentation::output_policy terminal_demo_policy(std::size_t width)
{
  auto policy = presentation::terminal_policy(stdout);
  policy.width = presentation::width_mode::display_cells;
  policy.wrap_width = width;
  policy.tab_width = 4;
  return policy;
}

[[nodiscard]] presentation::output_policy forced_demo_policy(presentation::glyph_set glyphs, std::size_t width)
{
  auto policy = terminal_demo_policy(width);
  policy.glyphs = glyphs;
  policy.charset = presentation::text_charset::utf8;
  return policy;
}

[[nodiscard]] std::string repeat(char ch, std::size_t count) { return std::string(count, ch); }

[[nodiscard]] terminal::TerminalStyle style(std::string_view spec) { return terminal::TerminalStyle(spec); }

[[nodiscard]] std::string width_ruler(std::size_t width)
{
  std::string ruler;
  ruler.reserve(width);
  for (std::size_t i = 0; i < width; ++i)
  {
    ruler.push_back(static_cast<char>('0' + ((i + 1) % 10)));
  }
  return ruler;
}

[[nodiscard]] std::size_t detected_terminal_width()
{
  int const columns = terminal::columns();
  return columns > 0 ? static_cast<std::size_t>(columns) : 80;
}

void append_glyphs(presentation::styled_text& text, presentation::semantic_glyph glyph, std::size_t count,
                   terminal::TerminalStyle line_style = {})
{
  for (std::size_t i = 0; i < count; ++i)
  {
    text.append(glyph, line_style);
  }
}

[[nodiscard]] presentation::styled_text glyph_line(std::string_view label)
{
  presentation::styled_text line;
  line.append(label, style("Cyan;Bold"))
      .append(": ")
      .append(presentation::semantic_glyph::success, style("Green;Bold"))
      .append(" parsed ")
      .append(presentation::semantic_glyph::arrow_right)
      .append(" ")
      .append("unicode=\xE4\xB8\xAD\xE6\x96\x87 ")
      .append("emoji=\xF0\x9F\x9A\x80");

  return line;
}

[[nodiscard]] presentation::styled_text spacing_table(presentation::output_policy const& policy)
{
  struct row
  {
      std::string label;
      std::string value;
  };

  row const rows[] = {{"ascii", "alpha"},
                      {"cjk", "\xE4\xB8\xAD\xE6\x96\x87"},
                      {"emoji", "\xF0\x9F\x9A\x80 rocket"},
                      {"combining", "e\xCC\x81 = e + combining acute"}};

  presentation::styled_text text;
  text.append("display-cell aligned table\n", style("Yellow;Bold"));
  for (auto const& item : rows)
  {
    text.append("  ")
        .append(presentation::pad_right(item.label, 10, policy))
        .append(" | ")
        .append(presentation::pad_right(item.value, 26, policy))
        .append(" | cells=")
        .append(std::to_string(presentation::display_width(item.value, policy)))
        .append("\n");
  }
  return text;
}

[[nodiscard]] presentation::report_builder solver_report()
{
  presentation::report_builder report("Krylov solve");
  report.status(presentation::semantic_glyph::success, "converged")
      .field("matrix", "demo")
      .field("dimension", 128)
      .field("tolerance", "1.0e-12");

  report.table("Solver Summary")
      .grid(presentation::table_rule_style::double_line)
      .column("solver", presentation::table_alignment::left)
      .column("matvecs")
      .column("ritz value")
      .column("residual")
      .row("native", 185, "-1.732050807568877", "1.0e-15")
      .row("arpack", 238, "-1.732050807568102", "8.0e-13")
      .separator()
      .row({{"notes", 1},
            {"spanning cells are useful for solver diagnostics that need a short label followed by wider prose.", 3,
             presentation::table_alignment::left}});

  return report;
}

[[nodiscard]] presentation::report_builder fixed_width_wrapping_report()
{
  presentation::report_builder report("Fixed-width wrapped table");
  report.status(presentation::semantic_glyph::info, "wrap_width = 44 display cells");

  report.table("Cell wrapping")
      .grid()
      .column("case", presentation::table_alignment::left)
      .column("content", presentation::table_alignment::left)
      .row("ascii", "A long diagnostic message wraps inside this cell instead of after the table is rendered.")
      .row("unicode", "display width counts 中文 and 🚀 before choosing where each cell line breaks.")
      .row("numbers", "right-aligned numeric columns can stay compact while nearby text columns wrap.");

  return report;
}

[[nodiscard]] presentation::report_builder advanced_table_report()
{
  presentation::report_builder report("Advanced table layout");
  report.status(presentation::semantic_glyph::info, "spans, manual rules, mixed borders, and decimal alignment");

  report.table("Phase ledger")
      .outer_border()
      .column_separators()
      .header_separator()
      .top_separator(presentation::table_rule_style::double_line)
      .column("phase", presentation::table_alignment::left)
      .column("description", presentation::table_alignment::left)
      .column("cost", presentation::table_alignment::decimal)
      .column("state", presentation::table_alignment::center)
      .row("parse", "read sparse blocks", "1.250", "ok")
      .row("factor", "symbolic setup", "12", "cached")
      .separator()
      .row({{"solve", 1},
            {"Krylov restart used a spanning note cell; the separator above shows where vertical rules stop.", 2,
             presentation::table_alignment::left},
            {"warn", 1, presentation::table_alignment::center}})
      .separator(presentation::table_rule_style::double_line)
      .row("cleanup", "release temporaries", "0.03125", "done");

  report.table("Decimal alignment")
      .grid()
      .column("quantity", presentation::table_alignment::left)
      .column("estimate", presentation::table_alignment::decimal)
      .column("observed", presentation::table_alignment::decimal)
      .column("note", presentation::table_alignment::left)
      .row("residual", "1.0e-15", "8.0e-13", "scientific strings still align before the decimal")
      .row("runtime", "12", "12.375", "integers align as if the decimal follows the value")
      .row("overflow", presentation::format_real(std::numeric_limits<double>::infinity(), {}),
           presentation::format_real(-std::numeric_limits<double>::infinity(), {}),
           "non-finite real values use deterministic lowercase spelling")
      .row("invalid", presentation::format_real(std::numeric_limits<double>::quiet_NaN(), {}), "nan",
           "NaN renders as nan and decimal alignment treats it like text")
      .row({{"status", 1},
            {"-", 1, presentation::table_alignment::center},
            {"-", 1, presentation::table_alignment::center},
            {"per-cell center overrides", 1}});

  return report;
}

[[nodiscard]] presentation::report_builder terminal_width_wrapping_report(std::size_t terminal_width)
{
  presentation::report_builder report("Terminal-width table");
  report.status(presentation::semantic_glyph::info,
                fmt::format("detected width = {} display cells; resize the terminal and rerun", terminal_width));

  report.table("Resize experiment")
      .grid()
      .column("item", presentation::table_alignment::left)
      .column("observation", presentation::table_alignment::left)
      .column("value")
      .row("policy", "The table consumes output_policy::wrap_width before final rendering, so borders remain intact.",
           "stable")
      .row("wide terminal", "More width is assigned to natural columns, reducing the number of wrapped cell lines.",
           "fewer rows")
      .row("narrow terminal", "The widest cells shrink first; each entry wraps independently inside its own column.",
           "more rows")
      .row("glyphs", "Semantic box drawing still falls back through the selected glyph policy: ✓ → 中文 🚀.", "cells")
      .row("width: check", "isolated checkmark: ✓", "probe")
      .row("width: arrow", "isolated right arrow: →", "probe")
      .row("width: cjk", "isolated CJK text: 中文", "probe")
      .row("width: emoji", "isolated emoji: 🚀", "probe")
      .row("width: mixed", "mixed sequence: ✓ → 中文 🚀", "probe");

  return report;
}

[[nodiscard]] presentation::styled_text indented_wrapping(presentation::output_policy const& policy)
{
  auto const diagnostic_block = "parser note: this continuation block was split intentionally\n"
                                "before adding fixed indentation, so follow-up lines stay\n"
                                "aligned under the diagnostic instead of drifting to column zero.";

  presentation::styled_text text;
  text.append("fixed indentation for nested text blocks\n", style("Yellow;Bold"))
      .append(presentation::indent_text(diagnostic_block, 4, policy))
      .append("\n");
  return text;
}

[[nodiscard]] presentation::styled_text tensor_network_diagram()
{
  auto const line_style = style("LightBlue");
  auto const node_style = style("Cyan;Bold");
  auto const index_style = style("Yellow;Bold");

  presentation::styled_text text;
  text.append("tensor-network sketch\n", style("Yellow;Bold"));

  text.append("       ", line_style).append(presentation::semantic_glyph::box_round_top_left, line_style);
  append_glyphs(text, presentation::semantic_glyph::box_horizontal, 3, line_style);
  text.append(presentation::semantic_glyph::box_round_top_right, line_style)
      .append("       ", line_style)
      .append(presentation::semantic_glyph::box_round_top_left, line_style);
  append_glyphs(text, presentation::semantic_glyph::box_horizontal, 3, line_style);
  text.append(presentation::semantic_glyph::box_round_top_right, line_style).append("\n");

  text.append("i", index_style).append(" ");
  append_glyphs(text, presentation::semantic_glyph::box_horizontal, 4, line_style);
  text.append(presentation::semantic_glyph::arrow_right, line_style)
      .append(presentation::semantic_glyph::box_tee_left, line_style)
      .append(" A ", node_style)
      .append(presentation::semantic_glyph::box_tee_right, line_style);
  append_glyphs(text, presentation::semantic_glyph::box_horizontal, 2, line_style);
  text.append(" α ", index_style);
  append_glyphs(text, presentation::semantic_glyph::box_horizontal, 1, line_style);
  text.append(presentation::semantic_glyph::arrow_right, line_style)
      .append(presentation::semantic_glyph::box_tee_left, line_style)
      .append(" B ", node_style)
      .append(presentation::semantic_glyph::box_tee_right, line_style);
  append_glyphs(text, presentation::semantic_glyph::box_horizontal, 4, line_style);
  text.append(presentation::semantic_glyph::arrow_right, line_style).append(" j", index_style).append("\n");

  text.append("       ", line_style).append(presentation::semantic_glyph::box_round_bottom_left, line_style);
  append_glyphs(text, presentation::semantic_glyph::box_horizontal, 1, line_style);
  text.append(presentation::semantic_glyph::box_tee_down, line_style);
  append_glyphs(text, presentation::semantic_glyph::box_horizontal, 1, line_style);
  text.append(presentation::semantic_glyph::box_round_bottom_right, line_style)
      .append("       ", line_style)
      .append(presentation::semantic_glyph::box_round_bottom_left, line_style);
  append_glyphs(text, presentation::semantic_glyph::box_horizontal, 1, line_style);
  text.append(presentation::semantic_glyph::box_tee_down, line_style);
  append_glyphs(text, presentation::semantic_glyph::box_horizontal, 1, line_style);
  text.append(presentation::semantic_glyph::box_round_bottom_right, line_style).append("\n");

  text.append("         ", line_style)
      .append(presentation::semantic_glyph::box_vertical, line_style)
      .append("           ", line_style)
      .append(presentation::semantic_glyph::box_vertical, line_style)
      .append("\n");
  text.append("         ", line_style)
      .append("β", index_style)
      .append("           ", line_style)
      .append("γ", index_style)
      .append("\n");

  return text;
}

[[nodiscard]] presentation::styled_text tensor_node_style()
{
  auto const line_style = style("LightBlue");
  auto const node_style = style("Cyan;Bold");

  presentation::styled_text text;
  text.append("tensor node style\n", style("Yellow;Bold"));

  text.append("rounded corners are the default box style\n", style("LightGray"));
  text.append(presentation::semantic_glyph::box_round_top_left, line_style);
  append_glyphs(text, presentation::semantic_glyph::box_horizontal, 5, line_style);
  text.append(presentation::semantic_glyph::box_round_top_right, line_style).append("\n");

  text.append(presentation::semantic_glyph::box_vertical, line_style)
      .append("  T  ", node_style)
      .append(presentation::semantic_glyph::box_vertical, line_style)
      .append("\n");

  text.append(presentation::semantic_glyph::box_round_bottom_left, line_style);
  append_glyphs(text, presentation::semantic_glyph::box_horizontal, 5, line_style);
  text.append(presentation::semantic_glyph::box_round_bottom_right, line_style).append("\n");

  return text;
}

[[nodiscard]] presentation::styled_text parser_region_diagnostic(std::string_view line, std::size_t region_begin,
                                                                 std::size_t region_end, std::size_t terminal_width,
                                                                 std::string_view message, std::string_view label,
                                                                 presentation::output_policy const& policy)
{
  static constexpr std::string_view marker = "\xE2\x80\xA6";
  static constexpr std::string_view source_gutter = "  12 | ";
  static constexpr std::string_view annotation_gutter = "     | ";

  std::size_t const gutter_width = presentation::display_width(source_gutter, policy);
  std::size_t const content_width = terminal_width > gutter_width ? terminal_width - gutter_width : 1;
  std::size_t const clamped_begin = std::min(region_begin, line.size());
  std::size_t const clamped_end = std::max(clamped_begin, std::min(region_end, line.size()));

  auto const before = line.substr(0, clamped_begin);
  auto const region = line.substr(clamped_begin, clamped_end - clamped_begin);
  auto const suffix = line.substr(clamped_begin);
  auto const before_width = presentation::display_width(before, policy);
  auto const prefix_budget = std::min(before_width, content_width / 2);

  auto const visible_prefix = presentation::truncate_left_to_width(before, prefix_budget, policy, marker);
  auto const suffix_budget = content_width - presentation::display_width(visible_prefix, policy);
  auto const visible_suffix = presentation::truncate_to_width(suffix, suffix_budget, policy, marker);

  auto const caret_column = presentation::display_width(visible_prefix, policy);
  auto const region_width = std::max<std::size_t>(1, presentation::display_width(region, policy));
  bool const region_continues = region_width > suffix_budget;
  auto const annotation_label = std::string(label) + (region_continues ? " continues" : "");
  auto const annotation_tail = (region_continues ? std::string(marker) : std::string{}) + " " + annotation_label;
  static constexpr std::string_view header_prefix = "parser error: ";
  auto const header_prefix_width = presentation::display_width(header_prefix, policy);
  auto const message_budget = terminal_width > header_prefix_width ? terminal_width - header_prefix_width : 0;
  auto const visible_message =
      message_budget > 0 ? presentation::truncate_to_width(message, message_budget, policy, marker) : std::string{};
  auto const annotation_available =
      terminal_width > gutter_width + caret_column ? terminal_width - gutter_width - caret_column : 1;
  auto const tail_width = presentation::display_width(annotation_tail, policy);
  std::size_t underline_width = std::min(region_width, annotation_available);
  if (tail_width < annotation_available)
  {
    underline_width = std::min(underline_width, annotation_available - tail_width);
  }
  underline_width = std::max<std::size_t>(1, underline_width);
  auto const tail_available = annotation_available > underline_width ? annotation_available - underline_width : 0;
  auto const visible_tail =
      tail_available > 0 ? presentation::truncate_to_width(annotation_tail, tail_available, policy) : std::string{};

  presentation::styled_text out;
  out.append("parser error", style("Red;Bold"))
      .append(": ")
      .append(visible_message)
      .append("\n")
      .append(source_gutter)
      .append(visible_prefix)
      .append(visible_suffix)
      .append("\n")
      .append(annotation_gutter)
      .append(repeat(' ', caret_column))
      .append("^", style("Red;Bold"))
      .append(repeat('~', underline_width > 0 ? underline_width - 1 : 0), style("Red;Bold"))
      .append(visible_tail, style("Red;Bold"))
      .append("\n");
  return out;
}

} // namespace

int main()
{
  auto unicode_policy = forced_demo_policy(presentation::glyph_set::unicode, 80);
  auto emoji_policy = forced_demo_policy(presentation::glyph_set::emoji, 80);
  auto environment_policy = terminal_demo_policy(80);

  fmt::print("forced glyph policy demo\n");
  fmt::print("{}\n", presentation::render(glyph_line("unicode glyphs"), unicode_policy));
  fmt::print("{}\n\n", presentation::render(glyph_line("emoji glyphs"), emoji_policy));

  fmt::print("terminal environment policy demo\n");
  fmt::print("try UNI20_GLYPHS=ascii UNI20_CHARSET=ascii_replace UNI20_COLOR=never\n");
  fmt::print("{}\n\n", presentation::render(glyph_line("terminal policy glyphs"), environment_policy));

  fmt::print("{}\n", presentation::render(spacing_table(unicode_policy), unicode_policy));
  fmt::print("{}\n", presentation::render_plain(solver_report(), unicode_policy));
  fmt::print("{}\n", presentation::render_plain(advanced_table_report(), unicode_policy));

  auto fixed_table_policy = unicode_policy;
  fixed_table_policy.wrap_width = 44;
  fmt::print("{}\n", presentation::render_plain(fixed_width_wrapping_report(), fixed_table_policy));

  auto terminal_table_policy = unicode_policy;
  terminal_table_policy.wrap_width = std::max<std::size_t>(detected_terminal_width(), 24);
  fmt::print("{}\n", presentation::render_plain(terminal_width_wrapping_report(*terminal_table_policy.wrap_width),
                                                terminal_table_policy));

  fmt::print("{}\n", presentation::render(indented_wrapping(unicode_policy), unicode_policy));
  fmt::print("{}\n", presentation::render(tensor_network_diagram(), unicode_policy));
  fmt::print("{}\n", presentation::render(tensor_node_style(), unicode_policy));

  std::string const source = "let result = parse(\xE4\xB8\xAD\xE6\x96\x87_input, rocket_\xF0\x9F\x9A\x80) + "
                             "missing_call(argument_one, argument_two, argument_three)";
  auto const error_byte = source.find("missing_call");
  if (error_byte == std::string::npos)
  {
    return 1;
  }
  auto const region_end = source.size();
  auto constexpr identifier = std::string_view("missing_call");
  auto const identifier_end = error_byte + identifier.size();

  auto narrow = unicode_policy;
  narrow.wrap_width = std::nullopt;

  fmt::print("point diagnostic at width 48\n");
  fmt::print("{}\n", width_ruler(48));
  fmt::print("{}\n",
             presentation::render(parser_region_diagnostic(source, error_byte, error_byte, 48,
                                                           "expected ';' before call expression", "here", narrow),
                                  narrow));

  fmt::print("fitting region diagnostic at width 72\n");
  fmt::print("{}\n", width_ruler(72));
  fmt::print("{}\n", presentation::render(parser_region_diagnostic(source, error_byte, identifier_end, 72,
                                                                   "unknown parser action", "identifier", narrow),
                                          narrow));

  fmt::print("region diagnostic at width 48\n");
  fmt::print("{}\n", width_ruler(48));
  fmt::print("{}\n",
             presentation::render(parser_region_diagnostic(source, error_byte, region_end, 48,
                                                           "selected expression is not valid here", "region", narrow),
                                  narrow));

  fmt::print("region diagnostic at width 72\n");
  fmt::print("{}\n", width_ruler(72));
  fmt::print("{}\n",
             presentation::render(parser_region_diagnostic(source, error_byte, region_end, 72,
                                                           "selected expression is not valid here", "region", narrow),
                                  narrow));
}
