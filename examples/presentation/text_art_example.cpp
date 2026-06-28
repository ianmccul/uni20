#include "presentation_example_common.hpp"

#include <fmt/core.h>

namespace
{
namespace demo = uni20::examples::presentation_demo;
namespace presentation = uni20::presentation;

[[nodiscard]] presentation::styled_text indented_wrapping(presentation::output_policy const& policy)
{
  auto const diagnostic_block = "parser note: this continuation block was split intentionally\n"
                                "before adding fixed indentation, so follow-up lines stay\n"
                                "aligned under the diagnostic instead of drifting to column zero.";

  presentation::styled_text text;
  text.append("fixed indentation for nested text blocks\n", demo::style("Yellow;Bold"))
      .append(presentation::indent_text(diagnostic_block, 4, policy))
      .append("\n");
  return text;
}

[[nodiscard]] presentation::styled_text tensor_network_diagram()
{
  auto const line_style = demo::style("LightBlue");
  auto const node_style = demo::style("Cyan;Bold");
  auto const index_style = demo::style("Yellow;Bold");

  presentation::styled_text text;
  text.append("tensor-network sketch\n", demo::style("Yellow;Bold"));

  text.append("       ", line_style).append(presentation::semantic_glyph::box_round_top_left, line_style);
  demo::append_glyphs(text, presentation::semantic_glyph::box_horizontal, 3, line_style);
  text.append(presentation::semantic_glyph::box_round_top_right, line_style)
      .append("       ", line_style)
      .append(presentation::semantic_glyph::box_round_top_left, line_style);
  demo::append_glyphs(text, presentation::semantic_glyph::box_horizontal, 3, line_style);
  text.append(presentation::semantic_glyph::box_round_top_right, line_style).append("\n");

  text.append("i", index_style).append(" ");
  demo::append_glyphs(text, presentation::semantic_glyph::box_horizontal, 4, line_style);
  text.append(presentation::semantic_glyph::arrow_right, line_style)
      .append(presentation::semantic_glyph::box_tee_left, line_style)
      .append(" A ", node_style)
      .append(presentation::semantic_glyph::box_tee_right, line_style);
  demo::append_glyphs(text, presentation::semantic_glyph::box_horizontal, 2, line_style);
  text.append(" \xCE\xB1 ", index_style);
  demo::append_glyphs(text, presentation::semantic_glyph::box_horizontal, 1, line_style);
  text.append(presentation::semantic_glyph::arrow_right, line_style)
      .append(presentation::semantic_glyph::box_tee_left, line_style)
      .append(" B ", node_style)
      .append(presentation::semantic_glyph::box_tee_right, line_style);
  demo::append_glyphs(text, presentation::semantic_glyph::box_horizontal, 4, line_style);
  text.append(presentation::semantic_glyph::arrow_right, line_style).append(" j", index_style).append("\n");

  text.append("       ", line_style).append(presentation::semantic_glyph::box_round_bottom_left, line_style);
  demo::append_glyphs(text, presentation::semantic_glyph::box_horizontal, 1, line_style);
  text.append(presentation::semantic_glyph::box_tee_down, line_style);
  demo::append_glyphs(text, presentation::semantic_glyph::box_horizontal, 1, line_style);
  text.append(presentation::semantic_glyph::box_round_bottom_right, line_style)
      .append("       ", line_style)
      .append(presentation::semantic_glyph::box_round_bottom_left, line_style);
  demo::append_glyphs(text, presentation::semantic_glyph::box_horizontal, 1, line_style);
  text.append(presentation::semantic_glyph::box_tee_down, line_style);
  demo::append_glyphs(text, presentation::semantic_glyph::box_horizontal, 1, line_style);
  text.append(presentation::semantic_glyph::box_round_bottom_right, line_style).append("\n");

  text.append("         ", line_style)
      .append(presentation::semantic_glyph::box_vertical, line_style)
      .append("           ", line_style)
      .append(presentation::semantic_glyph::box_vertical, line_style)
      .append("\n");
  text.append("         ", line_style)
      .append("\xCE\xB2", index_style)
      .append("           ", line_style)
      .append("\xCE\xB3", index_style)
      .append("\n");

  return text;
}

[[nodiscard]] presentation::styled_text tensor_node_style()
{
  auto const line_style = demo::style("LightBlue");
  auto const node_style = demo::style("Cyan;Bold");

  presentation::styled_text text;
  text.append("tensor node style\n", demo::style("Yellow;Bold"));

  text.append("rounded corners are the default box style\n", demo::style("LightGray"));
  text.append(presentation::semantic_glyph::box_round_top_left, line_style);
  demo::append_glyphs(text, presentation::semantic_glyph::box_horizontal, 5, line_style);
  text.append(presentation::semantic_glyph::box_round_top_right, line_style).append("\n");

  text.append(presentation::semantic_glyph::box_vertical, line_style)
      .append("  T  ", node_style)
      .append(presentation::semantic_glyph::box_vertical, line_style)
      .append("\n");

  text.append(presentation::semantic_glyph::box_round_bottom_left, line_style);
  demo::append_glyphs(text, presentation::semantic_glyph::box_horizontal, 5, line_style);
  text.append(presentation::semantic_glyph::box_round_bottom_right, line_style).append("\n");

  return text;
}

} // namespace

int main()
{
  auto const unicode_policy = demo::forced_demo_policy(presentation::glyph_set::unicode, 80);

  fmt::print("{}\n", presentation::render(indented_wrapping(unicode_policy), unicode_policy));
  fmt::print("{}\n", presentation::render(tensor_network_diagram(), unicode_policy));
  fmt::print("{}\n", presentation::render(tensor_node_style(), unicode_policy));
}
