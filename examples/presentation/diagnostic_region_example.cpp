#include "presentation_example_common.hpp"

#include <algorithm>
#include <cstddef>
#include <fmt/core.h>
#include <optional>
#include <string>
#include <string_view>

namespace
{
namespace demo = uni20::examples::presentation_demo;
namespace presentation = uni20::presentation;

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
  out.append("parser error", demo::style("Red;Bold"))
      .append(": ")
      .append(visible_message)
      .append("\n")
      .append(source_gutter)
      .append(visible_prefix)
      .append(visible_suffix)
      .append("\n")
      .append(annotation_gutter)
      .append(demo::repeat(' ', caret_column))
      .append("^", demo::style("Red;Bold"))
      .append(demo::repeat('~', underline_width > 0 ? underline_width - 1 : 0), demo::style("Red;Bold"))
      .append(visible_tail, demo::style("Red;Bold"))
      .append("\n");
  return out;
}

} // namespace

int main()
{
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

  auto narrow = demo::forced_demo_policy(presentation::glyph_set::unicode, 80);
  narrow.wrap_width = std::nullopt;

  fmt::print("point diagnostic at width 48\n");
  fmt::print("{}\n", demo::width_ruler(48));
  fmt::print("{}\n",
             presentation::render(parser_region_diagnostic(source, error_byte, error_byte, 48,
                                                           "expected ';' before call expression", "here", narrow),
                                  narrow));

  fmt::print("fitting region diagnostic at width 72\n");
  fmt::print("{}\n", demo::width_ruler(72));
  fmt::print("{}\n", presentation::render(parser_region_diagnostic(source, error_byte, identifier_end, 72,
                                                                   "unknown parser action", "identifier", narrow),
                                          narrow));

  fmt::print("region diagnostic at width 48\n");
  fmt::print("{}\n", demo::width_ruler(48));
  fmt::print("{}\n",
             presentation::render(parser_region_diagnostic(source, error_byte, region_end, 48,
                                                           "selected expression is not valid here", "region", narrow),
                                  narrow));

  fmt::print("region diagnostic at width 72\n");
  fmt::print("{}\n", demo::width_ruler(72));
  fmt::print("{}\n",
             presentation::render(parser_region_diagnostic(source, error_byte, region_end, 72,
                                                           "selected expression is not valid here", "region", narrow),
                                  narrow));
}
