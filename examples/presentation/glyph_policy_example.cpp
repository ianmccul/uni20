#include "presentation_example_common.hpp"

#include <cstddef>
#include <fmt/core.h>
#include <string>
#include <string_view>

namespace
{
namespace demo = uni20::examples::presentation_demo;
namespace presentation = uni20::presentation;

[[nodiscard]] presentation::styled_text glyph_line(std::string_view label)
{
  presentation::styled_text line;
  line.append(label, demo::style("Cyan;Bold"))
      .append(": ")
      .append(presentation::semantic_glyph::success, demo::style("Green;Bold"))
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
  text.append("terminal-policy display-cell table\n", demo::style("Yellow;Bold"));
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

} // namespace

int main()
{
  auto const unicode_policy = demo::forced_demo_policy(presentation::glyph_set::unicode, 80);
  auto const emoji_policy = demo::forced_demo_policy(presentation::glyph_set::emoji, 80);
  auto const environment_policy = demo::terminal_demo_policy(80);

  fmt::print("forced glyph policy demo\n");
  fmt::print("these two lines use explicit policies and ignore UNI20_* environment variables\n");
  fmt::print("{}\n", presentation::render(glyph_line("unicode glyphs"), unicode_policy));
  fmt::print("{}\n\n", presentation::render(glyph_line("emoji glyphs"), emoji_policy));

  fmt::print("terminal environment policy demo\n");
  fmt::print("terminal_policy() reads these environment variables:\n");
  fmt::print("  UNI20_GLYPHS=emoji|unicode|ascii\n");
  fmt::print("  UNI20_CHARSET=utf8|escape|replace\n");
  fmt::print("    aliases: utf-8, ascii_escape, ascii-escape, ascii_replace, ascii-replace\n");
  fmt::print("  UNI20_COLOR=auto|automatic|yes|always|true|on|1|no|never|false|off|0\n");
  fmt::print("  NO_COLOR disables automatic color when set to a non-empty value\n\n");
  fmt::print("UNI20_GLYPHS changes semantic glyph tokens such as success and arrows.\n");
  fmt::print("UNI20_CHARSET changes raw UTF-8 text such as Chinese characters and emoji.\n");
  fmt::print("try:\n");
  fmt::print("  UNI20_GLYPHS=ascii ./presentation_glyph_policy_example\n");
  fmt::print("  UNI20_GLYPHS=ascii UNI20_CHARSET=replace ./presentation_glyph_policy_example\n");
  fmt::print("  UNI20_GLYPHS=ascii UNI20_CHARSET=escape UNI20_COLOR=never ./presentation_glyph_policy_example\n\n");
  fmt::print("{}\n\n", presentation::render(glyph_line("terminal policy glyphs"), environment_policy));

  fmt::print("the table below also uses terminal_policy(), so UNI20_CHARSET changes its raw text\n");
  fmt::print("{}\n", presentation::render(spacing_table(environment_policy), environment_policy));
}
