#pragma once

#include <uni20/common/presentation.hpp>

#include <cstddef>
#include <cstdio>
#include <string>
#include <string_view>

namespace uni20::examples::presentation_demo
{
namespace presentation = uni20::presentation;

[[nodiscard]] inline presentation::output_policy terminal_demo_policy(std::size_t width)
{
  auto policy = presentation::terminal_policy(stdout);
  policy.width = presentation::width_mode::display_cells;
  policy.wrap_width = width;
  policy.tab_width = 4;
  return policy;
}

[[nodiscard]] inline presentation::output_policy forced_demo_policy(presentation::glyph_set glyphs, std::size_t width)
{
  auto policy = terminal_demo_policy(width);
  policy.glyphs = glyphs;
  policy.charset = presentation::text_charset::utf8;
  return policy;
}

[[nodiscard]] inline terminal::TerminalStyle style(std::string_view spec) { return terminal::TerminalStyle(spec); }

[[nodiscard]] inline std::string repeat(char ch, std::size_t count) { return std::string(count, ch); }

[[nodiscard]] inline std::string width_ruler(std::size_t width)
{
  std::string ruler;
  ruler.reserve(width);
  for (std::size_t i = 0; i < width; ++i)
  {
    ruler.push_back(static_cast<char>('0' + ((i + 1) % 10)));
  }
  return ruler;
}

[[nodiscard]] inline std::size_t detected_terminal_width()
{
  int const columns = terminal::columns();
  return columns > 0 ? static_cast<std::size_t>(columns) : 80;
}

inline void append_glyphs(presentation::styled_text& text, presentation::semantic_glyph glyph, std::size_t count,
                          terminal::TerminalStyle line_style = {})
{
  for (std::size_t i = 0; i < count; ++i)
  {
    text.append(glyph, line_style);
  }
}

} // namespace uni20::examples::presentation_demo
