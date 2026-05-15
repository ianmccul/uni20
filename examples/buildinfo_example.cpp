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

[[nodiscard]] std::size_t glyph_width(presentation::semantic_glyph glyph, presentation::output_policy const& policy)
{
  return presentation::display_width(presentation::render_glyph(glyph, policy), policy);
}

[[nodiscard]] std::size_t field_prefix_width(std::size_t key_width, presentation::output_policy const& policy)
{
  return 2 + key_width + 1 + glyph_width(presentation::semantic_glyph::arrow_right, policy) + 1;
}

[[nodiscard]] std::string clipped_key(std::string_view value,
                                      std::size_t width,
                                      presentation::output_policy const& policy)
{
  if (presentation::display_width(value, policy) <= width)
  {
    return std::string(value);
  }

  if (value.starts_with("UNI20_"))
  {
    return presentation::truncate_left_to_width(value, width, policy, "...");
  }

  return presentation::truncate_to_width(value, width, policy, "...");
}

[[nodiscard]] bool is_preferred_wrap_boundary(char ch)
{
  switch (ch)
  {
  case ' ':
  case '/':
  case '\\':
  case '-':
  case '_':
  case ',':
  case ';':
  case ':':
    return true;
  default:
    return false;
  }
}

[[nodiscard]] std::string_view trim_right_spaces(std::string_view text)
{
  while (!text.empty() && text.back() == ' ')
  {
    text.remove_suffix(1);
  }
  return text;
}

[[nodiscard]] std::size_t preferred_break_position(std::string_view value,
                                                   std::size_t width,
                                                   presentation::output_policy const& policy)
{
  std::size_t best = 0;
  for (std::size_t i = 0; i < value.size(); ++i)
  {
    if (presentation::display_width(value.substr(0, i + 1), policy) > width)
    {
      break;
    }
    if (is_preferred_wrap_boundary(value[i]))
    {
      best = i + 1;
    }
  }

  std::size_t const minimum_preferred_break = std::min<std::size_t>(8, width);
  return best >= minimum_preferred_break ? best : 0;
}

[[nodiscard]] std::vector<std::string> wrapped(std::string_view value,
                                               std::size_t width,
                                               presentation::output_policy const& policy)
{
  std::vector<std::string> lines;
  if (width == 0)
  {
    lines.emplace_back(value);
    return lines;
  }

  std::string_view remaining = value;
  while (!remaining.empty())
  {
    if (presentation::display_width(remaining, policy) <= width)
    {
      lines.emplace_back(remaining);
      break;
    }

    std::size_t const break_pos = preferred_break_position(remaining, width, policy);
    if (break_pos == 0)
    {
      auto hard_lines = presentation::wrap_text(remaining, width, policy);
      lines.insert(lines.end(), hard_lines.begin(), hard_lines.end());
      break;
    }

    lines.emplace_back(trim_right_spaces(remaining.substr(0, break_pos)));
    remaining.remove_prefix(break_pos);
    while (!remaining.empty() && remaining.front() == ' ')
    {
      remaining.remove_prefix(1);
    }
  }

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
  auto const continuation_prefix = std::string(field_prefix_width(key_width, policy), ' ');

  text.append("  ")
      .append(presentation::pad_right(clipped_key(key, key_width, policy), key_width, policy), style("Cyan;Bold"))
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
  std::size_t constexpr minimum_value_width = 16;
  std::size_t const key_width = key_width_for(entries, width, policy);
  std::size_t const prefix_width = field_prefix_width(key_width, policy);
  std::size_t const value_width =
      width > prefix_width + minimum_value_width ? width - prefix_width : minimum_value_width;

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
      std::size_t const detail_indent = width > prefix_width + minimum_value_width ? prefix_width : 6;
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
  std::size_t const width = std::max<std::size_t>(terminal_width(), 40);

  presentation::styled_text text;
  text.append("Uni20 build information\n", style("Green;Bold"));
  append_rule(text, width, presentation::semantic_glyph::box_horizontal, style("LightBlue"));

  std::size_t constexpr key_width = 22;
  std::size_t constexpr minimum_value_width = 16;
  std::size_t const prefix_width = field_prefix_width(key_width, policy);
  std::size_t const value_width =
      width > prefix_width + minimum_value_width ? width - prefix_width : minimum_value_width;

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
