#include "display.hpp"

#include "terminal.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace uni20::display
{
namespace
{
using presentation::table_alignment;

[[nodiscard]] std::FILE* file_for_stream(stream destination) { return destination == stream::out ? stdout : stderr; }

[[nodiscard]] presentation::output_policy display_policy(std::FILE* file)
{
  auto policy = presentation::terminal_policy(file);
  if (auto const columns = terminal::columns(); columns > 0)
  {
    policy.wrap_width = static_cast<std::size_t>(columns);
  }
  return policy;
}

void default_sink(event const& item)
{
  auto* file = file_for_stream(item.destination);
  auto policy = display_policy(file);
  auto rendered = std::visit([&](auto const& content) { return presentation::render_terminal(content, policy, file); },
                             item.content);

  std::fputs(rendered.c_str(), file);
  if (item.newline) std::fputc('\n', file);
  std::fflush(file);
}

std::mutex& sink_mutex()
{
  static std::mutex mutex;
  return mutex;
}

sink& sink_storage()
{
  static sink current = default_sink;
  return current;
}

void emit_event(event item)
{
  sink current;
  {
    std::lock_guard lock(sink_mutex());
    current = sink_storage();
  }
  current(item);
}

[[nodiscard]] std::string spaces(std::size_t count) { return std::string(count, ' '); }

[[nodiscard]] std::string repeat(char value, std::size_t count) { return std::string(count, value); }

[[nodiscard]] presentation::styled_text make_plain_cell(std::string_view text)
{
  presentation::styled_text result;
  result.append(text);
  return result;
}

[[nodiscard]] presentation::styled_text align_cell(presentation::styled_text const& text, std::size_t width,
                                                   table_alignment alignment, presentation::output_policy const& policy,
                                                   bool trim_trailing_padding = false)
{
  switch (alignment)
  {
    case table_alignment::left:
      if (trim_trailing_padding) return text;
      return presentation::pad_right(text, width, policy);
    case table_alignment::center:
    {
      auto const used = presentation::display_width(text, policy);
      if (used >= width) return text;
      auto const extra = width - used;
      auto const left = extra / 2;
      presentation::styled_text result;
      result.append(spaces(left)).append(text);
      if (!trim_trailing_padding) result.append(spaces(extra - left));
      return result;
    }
    case table_alignment::right:
    case table_alignment::decimal:
      return presentation::pad_left(text, width, policy);
  }
  if (trim_trailing_padding) return text;
  return presentation::pad_right(text, width, policy);
}

[[nodiscard]] std::vector<presentation::styled_text> wrap_cell(presentation::styled_text const& text, std::size_t width,
                                                               presentation::output_policy const& policy)
{
  auto lines = presentation::wrap_text(text, std::max<std::size_t>(width, 1), policy);
  if (lines.empty()) lines.push_back(presentation::styled_text{});
  return lines;
}

} // namespace

namespace width
{
column_width fixed(std::size_t cells)
{
  return column_width{.mode = column_width_mode::fixed, .value = std::max<std::size_t>(cells, 1), .minimum = 1};
}

column_width share(std::size_t weight, std::size_t minimum)
{
  return column_width{.mode = column_width_mode::share,
                      .value = std::max<std::size_t>(weight, 1),
                      .minimum = std::max<std::size_t>(minimum, 1)};
}
} // namespace width

namespace detail
{
terminal::TerminalStyle status_style(presentation::semantic_glyph glyph)
{
  switch (glyph)
  {
    case presentation::semantic_glyph::success:
      return terminal::TerminalStyle(std::string_view("Green;Bold"));
    case presentation::semantic_glyph::failure:
    case presentation::semantic_glyph::fatal:
      return terminal::TerminalStyle(std::string_view("Red;Bold"));
    case presentation::semantic_glyph::warning:
    case presentation::semantic_glyph::partial:
      return terminal::TerminalStyle(std::string_view("Yellow;Bold"));
    case presentation::semantic_glyph::info:
      return terminal::TerminalStyle(std::string_view("LightBlue;Bold"));
    case presentation::semantic_glyph::deferred:
      return terminal::TerminalStyle(std::string_view("LightMagenta;Bold"));
    case presentation::semantic_glyph::skipped:
      return terminal::TerminalStyle(std::string_view("DarkGray;Bold"));
    default:
      return terminal::TerminalStyle(std::string_view("Bold"));
  }
}
} // namespace detail

streaming_table::streaming_table(std::string title, stream destination)
    : title_(std::move(title)), destination_(destination), policy_(display_policy(file_for_stream(destination)))
{}

streaming_table& streaming_table::column(std::string heading, column_width width_spec, table_alignment alignment)
{
  this->ensure_can_change_schema();
  columns_.push_back(column_spec{.heading = std::move(heading), .width = width_spec, .alignment = alignment});
  return *this;
}

streaming_table& streaming_table::wrap_width(std::size_t width)
{
  this->ensure_can_change_schema();
  policy_.wrap_width = std::max<std::size_t>(width, 1);
  return *this;
}

streaming_table& streaming_table::header_separator(bool enabled)
{
  this->ensure_can_change_schema();
  header_separator_ = enabled;
  return *this;
}

void streaming_table::row(std::vector<presentation::styled_text> cells, std::source_location where)
{
  this->emit_rows(cells, where);
}

void streaming_table::row(std::vector<std::string> cells, std::source_location where)
{
  std::vector<presentation::styled_text> styled_cells;
  styled_cells.reserve(cells.size());
  for (auto& cell : cells)
  {
    styled_cells.push_back(make_plain_cell(cell));
  }
  this->emit_rows(styled_cells, where);
}

void streaming_table::row(std::initializer_list<std::string> cells, std::source_location where)
{
  this->row(std::vector<std::string>(cells), where);
}

void streaming_table::ensure_can_change_schema() const
{
  if (header_emitted_)
  {
    throw std::logic_error("display streaming table schema cannot change after rows have been emitted");
  }
}

void streaming_table::resolve_widths()
{
  if (widths_resolved_) return;
  widths_resolved_ = true;

  if (columns_.empty())
  {
    throw std::logic_error("display streaming table requires at least one column");
  }

  auto const table_width =
      policy_.wrap_width.value_or(terminal::columns() > 0 ? static_cast<std::size_t>(terminal::columns()) : 80U);
  auto const separator_width = columns_.size() > 1 ? (columns_.size() - 1) * 2 : 0;
  if (table_width <= separator_width)
  {
    vertical_fallback_ = true;
    return;
  }

  widths_.assign(columns_.size(), 1);
  std::size_t fixed_total = 0;
  std::size_t share_minimum_total = 0;
  std::size_t share_weight_total = 0;

  for (std::size_t i = 0; i != columns_.size(); ++i)
  {
    auto const& column = columns_[i];
    if (column.width.mode == column_width_mode::fixed)
    {
      widths_[i] = std::max<std::size_t>(column.width.value, 1);
      fixed_total += widths_[i];
    }
    else
    {
      auto const heading_width = presentation::display_width(column.heading, policy_);
      widths_[i] = std::max({std::size_t{1}, column.width.minimum, heading_width});
      share_minimum_total += widths_[i];
      share_weight_total += std::max<std::size_t>(column.width.value, 1);
    }
  }

  if (fixed_total + share_minimum_total + separator_width > table_width)
  {
    vertical_fallback_ = true;
    return;
  }

  if (share_weight_total == 0) return;

  auto remaining = table_width - separator_width - fixed_total - share_minimum_total;
  std::size_t distributed = 0;
  std::size_t last_share = columns_.size();
  for (std::size_t i = 0; i != columns_.size(); ++i)
  {
    if (columns_[i].width.mode != column_width_mode::share) continue;
    last_share = i;
    auto const extra = remaining * columns_[i].width.value / share_weight_total;
    widths_[i] += extra;
    distributed += extra;
  }
  if (last_share != columns_.size()) widths_[last_share] += remaining - distributed;
}

void streaming_table::emit_rows(std::vector<presentation::styled_text> const& cells, std::source_location where)
{
  if (cells.size() != columns_.size())
  {
    throw std::invalid_argument("display streaming table row cell count does not match the column count");
  }

  this->resolve_widths();

  presentation::styled_text text;
  if (!header_emitted_)
  {
    if (!title_.empty()) text.append(title_, terminal::TerminalStyle(std::string_view("Cyan;Bold"))).append("\n");
    if (!vertical_fallback_)
    {
      for (std::size_t i = 0; i != columns_.size(); ++i)
      {
        if (i != 0) text.append("  ");
        auto heading = align_cell(make_plain_cell(columns_[i].heading), widths_[i], table_alignment::left, policy_,
                                  i + 1 == columns_.size());
        text.append(presentation::render_plain(heading, policy_),
                    terminal::TerminalStyle(std::string_view("LightGray;Bold")));
      }
      text.append("\n");
      if (header_separator_)
      {
        for (std::size_t i = 0; i != columns_.size(); ++i)
        {
          if (i != 0) text.append("  ");
          text.append(repeat('-', widths_[i]), terminal::TerminalStyle(std::string_view("LightGray")));
        }
        text.append("\n");
      }
    }
    header_emitted_ = true;
  }

  if (vertical_fallback_)
  {
    std::size_t key_width = 0;
    for (auto const& column : columns_)
      key_width = std::max(key_width, presentation::display_width(column.heading, policy_));

    auto const table_width = policy_.wrap_width.value_or(80U);
    auto const value_width = table_width > key_width + 2 ? table_width - key_width - 2 : std::size_t{1};
    for (std::size_t i = 0; i != columns_.size(); ++i)
    {
      auto const wrapped = wrap_cell(cells[i], value_width, policy_);
      text.append(presentation::pad_right(columns_[i].heading + ":", key_width + 1, policy_),
                  terminal::TerminalStyle(std::string_view("LightGray;Bold")))
          .append(" ")
          .append(wrapped.front())
          .append("\n");
      for (std::size_t line = 1; line != wrapped.size(); ++line)
      {
        text.append(spaces(key_width + 2)).append(wrapped[line]).append("\n");
      }
    }
    emit(std::move(text), destination_, false, where);
    return;
  }

  std::vector<std::vector<presentation::styled_text>> wrapped_cells;
  wrapped_cells.reserve(cells.size());
  std::size_t row_lines = 1;
  for (std::size_t i = 0; i != cells.size(); ++i)
  {
    auto wrapped = wrap_cell(cells[i], widths_[i], policy_);
    row_lines = std::max(row_lines, wrapped.size());
    wrapped_cells.push_back(std::move(wrapped));
  }

  for (std::size_t line = 0; line != row_lines; ++line)
  {
    for (std::size_t column = 0; column != columns_.size(); ++column)
    {
      if (column != 0) text.append("  ");
      auto const cell_line =
          line < wrapped_cells[column].size() ? wrapped_cells[column][line] : presentation::styled_text{};
      auto aligned =
          align_cell(cell_line, widths_[column], columns_[column].alignment, policy_, column + 1 == columns_.size());
      text.append(aligned);
    }
    if (line + 1 != row_lines) text.append("\n");
  }
  emit(std::move(text), destination_, true, where);
}

streaming_table table(std::string title, stream destination) { return streaming_table(std::move(title), destination); }

void set_sink(sink replacement)
{
  std::lock_guard lock(sink_mutex());
  sink_storage() = std::move(replacement);
  if (!sink_storage()) sink_storage() = default_sink;
}

void reset_sink()
{
  std::lock_guard lock(sink_mutex());
  sink_storage() = default_sink;
}

sink current_sink()
{
  std::lock_guard lock(sink_mutex());
  return sink_storage();
}

scoped_sink::scoped_sink(sink replacement) : previous_(current_sink()) { set_sink(std::move(replacement)); }

scoped_sink::scoped_sink(scoped_sink&& other) noexcept
    : previous_(std::move(other.previous_)), active_(std::exchange(other.active_, false))
{}

scoped_sink& scoped_sink::operator=(scoped_sink&& other) noexcept
{
  if (this != &other)
  {
    if (active_) set_sink(std::move(previous_));
    previous_ = std::move(other.previous_);
    active_ = std::exchange(other.active_, false);
  }
  return *this;
}

scoped_sink::~scoped_sink()
{
  if (active_) set_sink(std::move(previous_));
}

void emit(presentation::styled_text text, stream destination, bool newline, std::source_location where)
{
  emit_event(
      event{.destination = destination, .content = std::move(text), .newline = newline, .context = {}, .where = where});
}

void emit(presentation::report_builder report, stream destination, bool newline, std::source_location where)
{
  emit_event(event{
      .destination = destination, .content = std::move(report), .newline = newline, .context = {}, .where = where});
}

} // namespace uni20::display
