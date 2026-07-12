#pragma once

/**
 * \file presentation_stacktrace.hpp
 * \ingroup common
 * \brief Presentation-layer formatting for standard stacktraces.
 */

#include <uni20/common/presentation.hpp>
#include <uni20/config.hpp>

#if UNI20_HAS_STACKTRACE
#include <stacktrace>

#include <algorithm>
#include <fmt/core.h>
#include <string>
#include <vector>

namespace uni20::presentation
{

/// \brief Style and indentation controls for a stacktrace presentation document.
struct stacktrace_format_options
{
    std::size_t indent = 2;
    terminal::TerminalStyle tree_style = terminal::TerminalStyle("LightGray");
    terminal::TerminalStyle index_style = terminal::TerminalStyle("LightGray");
    terminal::TerminalStyle description_style = {};
    terminal::TerminalStyle source_file_style = terminal::TerminalStyle("Cyan");
    terminal::TerminalStyle source_line_style = terminal::TerminalStyle("LightGray");
};

/// \brief Format a standard stacktrace as policy-aware styled text.
/// \details Tree connectors remain semantic glyphs until rendering. When
///          `policy.wrap_width` is set, frame text wraps by display width and
///          continuation lines align beneath the frame description.
/// \param stacktrace Stacktrace frames to format.
/// \param policy Output policy controlling glyph widths and line wrapping.
/// \param options Frame styles and indentation controls.
/// \return Styled stacktrace document suitable for any presentation renderer.
inline styled_text format_stacktrace(std::stacktrace const& stacktrace, output_policy const& policy,
                                     stacktrace_format_options const& options = {})
{
  styled_text text;
  std::size_t frame_index = 0;
  auto const frame_count = stacktrace.size();

  for (auto const& frame : stacktrace)
  {
    bool const is_last = frame_index + 1 == frame_count;
    std::string description = frame.description();
    if (description.empty())
    {
      description = "(unknown)";
    }

    styled_text prefix;
    prefix.append(std::string(options.indent, ' '));
    prefix.append(is_last ? semantic_glyph::tree_last : semantic_glyph::tree_branch, options.tree_style);
    prefix.append(" ");
    prefix.append(fmt::format("#{}", frame_index), options.index_style);
    prefix.append(" ");

    styled_text frame_text;
    frame_text.append(description, options.description_style);
    if (!frame.source_file().empty())
    {
      frame_text.append(" at ");
      frame_text.append(frame.source_file(), options.source_file_style);
      frame_text.append(fmt::format(":{}", frame.source_line()), options.source_line_style);
    }

    std::vector<styled_text> lines{frame_text};
    auto const prefix_width = display_width(prefix, policy);
    if (policy.wrap_width.has_value() && *policy.wrap_width > prefix_width)
    {
      lines = wrap_text(frame_text, std::max<std::size_t>(*policy.wrap_width - prefix_width, 1), policy);
    }
    if (lines.empty())
    {
      lines.emplace_back();
    }

    text.append(prefix).append(lines.front()).append("\n");
    for (std::size_t line = 1; line < lines.size(); ++line)
    {
      text.append(std::string(options.indent, ' '));
      styled_text connector;
      connector.append(is_last ? semantic_glyph::tree_space : semantic_glyph::tree_vertical, options.tree_style);
      text.append(connector);
      auto const connector_width = display_width(connector, policy);
      text.append(std::string(prefix_width - options.indent - connector_width, ' '));
      text.append(lines[line]).append("\n");
    }
    ++frame_index;
  }

  if (frame_index == 0)
  {
    text.append(std::string(options.indent, ' '));
    text.append("(empty stacktrace)", options.description_style);
    text.append("\n");
  }
  return text;
}

} // namespace uni20::presentation
#endif
