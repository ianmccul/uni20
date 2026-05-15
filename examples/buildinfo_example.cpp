#include <uni20/buildinfo.hpp>
#include <uni20/common/presentation.hpp>

#include <algorithm>
#include <cstdio>
#include <fmt/core.h>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace
{
namespace presentation = uni20::presentation;

[[nodiscard]] terminal::TerminalStyle style(std::string_view spec) { return terminal::TerminalStyle(spec); }

[[nodiscard]] presentation::output_policy output_policy()
{
  auto policy = presentation::terminal_policy(stdout);
  policy.glyphs = presentation::glyph_set::unicode;
  policy.charset = presentation::text_charset::utf8;
  policy.width = presentation::width_mode::display_cells;
  policy.tab_width = 4;
  return policy;
}

[[nodiscard]] std::size_t terminal_width()
{
  int const columns = terminal::columns();
  if (columns > 0)
  {
    return static_cast<std::size_t>(columns);
  }
  return 100;
}

void append_rule(presentation::styled_text& text,
                 std::size_t width,
                 presentation::semantic_glyph glyph,
                 terminal::TerminalStyle line_style)
{
  for (std::size_t i = 0; i < width; ++i)
  {
    text.append(glyph, line_style);
  }
  text.append("\n");
}

[[nodiscard]] std::string clipped(std::string_view value, std::size_t width, presentation::output_policy const& policy)
{
  return presentation::truncate_to_width(value, width, policy, "...");
}

[[nodiscard]] std::vector<std::string> wrapped(std::string_view value,
                                               std::size_t width,
                                               presentation::output_policy const& policy)
{
  auto lines = presentation::wrap_text(value, width, policy);
  if (lines.empty())
  {
    lines.emplace_back();
  }
  return lines;
}

void append_field(presentation::styled_text& text,
                  std::string_view key,
                  std::string_view value,
                  std::size_t key_width,
                  std::size_t value_width,
                  presentation::output_policy const& policy,
                  terminal::TerminalStyle value_style = style("White"))
{
  auto const lines = wrapped(value, value_width, policy);
  auto const continuation_prefix = std::string(6, ' ');

  text.append("  ")
      .append(presentation::pad_right(clipped(key, key_width, policy), key_width, policy), style("Cyan;Bold"))
      .append(" ")
      .append(presentation::semantic_glyph::arrow_right, style("LightBlue"))
      .append(" ")
      .append(lines.front(), value_style)
      .append("\n");

  for (std::size_t i = 1; i < lines.size(); ++i)
  {
    text.append(continuation_prefix).append(lines[i], value_style).append("\n");
  }
}

[[nodiscard]] std::size_t key_width_for(std::span<uni20::build_info::Entry const> entries,
                                        std::size_t width,
                                        presentation::output_policy const& policy)
{
  std::size_t longest_key = 14;
  for (auto const& entry : entries)
  {
    longest_key = std::max(longest_key, presentation::display_width(entry.key, policy));
  }

  std::size_t const maximum_key_width = width > 64 ? width / 2 : width / 3;
  return std::min(longest_key, std::max<std::size_t>(14, maximum_key_width));
}

void append_entries(presentation::styled_text& text,
                    std::string_view title,
                    std::span<uni20::build_info::Entry const> entries,
                    std::size_t width,
                    presentation::output_policy const& policy)
{
  std::size_t constexpr minimum_value_width = 24;
  std::size_t const key_width = key_width_for(entries, width, policy);
  std::size_t const value_width =
      width > key_width + 12 ? std::max<std::size_t>(minimum_value_width, width - key_width - 12) : minimum_value_width;

  text.append("\n")
      .append(title, style("Yellow;Bold"))
      .append(" (")
      .append(std::to_string(entries.size()), style("LightGray"))
      .append(")\n", style("LightGray"));

  if (entries.empty())
  {
    text.append("  ")
        .append(presentation::semantic_glyph::info, style("LightBlue;Bold"))
        .append(" no entries\n", style("LightGray"));
    return;
  }

  for (auto const& entry : entries)
  {
    append_field(text, entry.key, entry.value, key_width, value_width, policy);
    if (!entry.help.empty())
    {
      std::size_t constexpr detail_indent = 6;
      std::size_t const help_width = width > detail_indent + 2 ? width - detail_indent - 2 : minimum_value_width;
      auto const help_lines = wrapped(entry.help, help_width, policy);
      auto const help_prefix = std::string(detail_indent, ' ');
      for (auto const& line : help_lines)
      {
        text.append(help_prefix).append(line, style("DarkGray")).append("\n");
      }
    }
  }
}

[[nodiscard]] presentation::styled_text buildinfo_report(presentation::output_policy const& policy)
{
  auto const info = uni20::build_info::current();
  std::size_t const width = std::max<std::size_t>(terminal_width(), 72);

  presentation::styled_text text;
  text.append("Uni20 build information\n", style("Green;Bold"));
  append_rule(text, width, presentation::semantic_glyph::box_horizontal, style("LightBlue"));

  std::size_t constexpr key_width = 22;
  std::size_t const value_width = width > key_width + 8 ? width - key_width - 8 : 40;

  append_field(text, "generator", info.generator, key_width, value_width, policy);
  append_field(text, "build type", info.build_type, key_width, value_width, policy);
  append_field(text, "system", fmt::format("{} {} ({})", info.system_name, info.system_version, info.system_processor),
               key_width, value_width, policy);
  append_field(text, "compiler",
               fmt::format("{} {} ({})", info.cxx_compiler_id, info.cxx_compiler_version, info.cxx_compiler_path),
               key_width, value_width, policy);

  append_entries(text, "Detected environment", info.detected_environment, width, policy);
  append_entries(text, "Build options", info.build_options, width, policy);
  return text;
}

} // namespace

int main()
{
  auto const policy = output_policy();
  fmt::print("{}", presentation::render(buildinfo_report(policy), policy));
}
