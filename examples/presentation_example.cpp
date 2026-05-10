#include <uni20/common/presentation.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <fmt/core.h>
#include <optional>
#include <string>
#include <string_view>

namespace
{
namespace presentation = uni20::presentation;

[[nodiscard]] presentation::output_policy demo_policy(presentation::glyph_set glyphs, std::size_t width)
{
  auto policy = presentation::terminal_policy(stdout);
  policy.glyphs = glyphs;
  policy.charset = presentation::text_charset::utf8;
  policy.width = presentation::width_mode::display_cells;
  policy.wrap_width = width;
  policy.tab_width = 4;
  return policy;
}

[[nodiscard]] std::string repeat(char ch, std::size_t count) { return std::string(count, ch); }

[[nodiscard]] terminal::TerminalStyle style(std::string_view spec) { return terminal::TerminalStyle(spec); }

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

  std::size_t const gutter_width = 6;
  std::size_t const content_width = terminal_width > gutter_width + 8 ? terminal_width - gutter_width : 24;
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
  std::size_t const marker_width = presentation::display_width(marker, policy);
  std::size_t underline_width = std::min(region_width, suffix_budget);
  if (region_continues && underline_width > marker_width)
  {
    underline_width -= marker_width;
  }

  presentation::styled_text out;
  out.append("parser error", style("Red;Bold"))
      .append(": ")
      .append(message)
      .append("\n")
      .append("  12 | ")
      .append(visible_prefix)
      .append(visible_suffix)
      .append("\n")
      .append("     | ")
      .append(repeat(' ', caret_column))
      .append("^", style("Red;Bold"))
      .append(repeat('~', underline_width > 0 ? underline_width - 1 : 0), style("Red;Bold"))
      .append(region_continues ? std::string(marker) : std::string{}, style("Red;Bold"))
      .append(" ", style("Red;Bold"))
      .append(label, style("Red;Bold"))
      .append(region_continues ? " continues" : "", style("Red;Bold"))
      .append("\n");
  return out;
}

} // namespace

int main()
{
  auto unicode_policy = demo_policy(presentation::glyph_set::unicode, 80);
  auto emoji_policy = demo_policy(presentation::glyph_set::emoji, 80);

  fmt::print("{}\n", presentation::render(glyph_line("unicode glyphs"), unicode_policy));
  fmt::print("{}\n\n", presentation::render(glyph_line("emoji glyphs"), emoji_policy));

  fmt::print("{}\n", presentation::render(spacing_table(unicode_policy), unicode_policy));
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
  fmt::print("{}\n",
             presentation::render(parser_region_diagnostic(source, error_byte, error_byte, 48,
                                                           "expected ';' before call expression", "here", narrow),
                                  narrow));

  fmt::print("fitting region diagnostic at width 72\n");
  fmt::print("{}\n", presentation::render(parser_region_diagnostic(source, error_byte, identifier_end, 72,
                                                                   "unknown parser action", "identifier", narrow),
                                          narrow));

  fmt::print("region diagnostic at width 48\n");
  fmt::print("{}\n",
             presentation::render(parser_region_diagnostic(source, error_byte, region_end, 48,
                                                           "selected expression is not valid here", "region", narrow),
                                  narrow));

  fmt::print("region diagnostic at width 72\n");
  fmt::print("{}\n",
             presentation::render(parser_region_diagnostic(source, error_byte, region_end, 72,
                                                           "selected expression is not valid here", "region", narrow),
                                  narrow));
}
