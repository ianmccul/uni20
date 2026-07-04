#include "display.hpp"

#include "terminal.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <mutex>
#include <optional>
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

[[nodiscard]] std::size_t effective_wrap_width(presentation::output_policy const& policy)
{
  return policy.wrap_width.value_or(terminal::columns() > 0 ? static_cast<std::size_t>(terminal::columns()) : 80U);
}

[[nodiscard]] table_alignment default_column_alignment(column_format const& cell_format)
{
  switch (cell_format.kind)
  {
    case column_value_kind::automatic:
      return table_alignment::right;
    case column_value_kind::text:
      return table_alignment::left;
    case column_value_kind::number:
      return table_alignment::decimal;
  }
  return table_alignment::right;
}

[[nodiscard]] std::size_t default_numeric_width(column_format const& cell_format)
{
  return cell_format.minimum_width != 0 ? cell_format.minimum_width : 10;
}

[[nodiscard]] column_width default_column_width(column_format const& cell_format)
{
  if (cell_format.kind == column_value_kind::number) return width::fit(default_numeric_width(cell_format));
  return width::share();
}

[[nodiscard]] terminal::TerminalStyle continuation_style()
{
  return terminal::TerminalStyle(std::string_view("LightGray;Bold"));
}

[[nodiscard]] presentation::styled_text make_plain_cell(std::string_view text)
{
  presentation::styled_text result;
  result.append(text);
  return result;
}

[[nodiscard]] std::string render_plain_unwrapped(presentation::styled_text const& text,
                                                 presentation::output_policy const& policy)
{
  auto render_policy = policy;
  render_policy.wrap_width = std::nullopt;
  return presentation::render_plain(text, render_policy);
}

[[nodiscard]] presentation::styled_text continuation_marker()
{
  presentation::styled_text marker;
  marker.append(presentation::semantic_glyph::arrow_right, continuation_style());
  return marker;
}

struct continuation_prefix
{
    presentation::styled_text text;
    std::size_t width = 0;
};

struct wrapped_cell_line
{
    presentation::styled_text text;
    bool continuation = false;
    bool decimal_candidate = false;
    bool decimal_exception = false;
};

[[nodiscard]] continuation_prefix continuation_prefix_for_width(std::size_t width,
                                                                presentation::output_policy const& policy)
{
  auto marker = continuation_marker();
  if (presentation::display_width(marker, policy) + 1 >= width)
  {
    marker = make_plain_cell(">");
  }

  auto const marker_width = presentation::display_width(marker, policy);
  if (marker_width >= width) return {};

  continuation_prefix prefix;
  prefix.text.append(marker);
  prefix.width = marker_width;
  if (marker_width + 1 < width)
  {
    prefix.text.append(" ");
    ++prefix.width;
  }
  return prefix;
}

[[nodiscard]] bool is_nonfinite_decimal_text(std::string_view rendered)
{
  return rendered == "nan" || rendered == "+nan" || rendered == "-nan" || rendered == "inf" || rendered == "+inf" ||
         rendered == "-inf";
}

[[nodiscard]] presentation::styled_text align_cell(presentation::styled_text const& text, std::size_t width,
                                                   table_alignment alignment, presentation::output_policy const& policy,
                                                   bool trim_trailing_padding = false,
                                                   std::optional<std::size_t> decimal_position = std::nullopt)
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
      return presentation::pad_left(text, width, policy);
    case table_alignment::decimal:
    {
      if (!decimal_position.has_value()) return presentation::pad_left(text, width, policy);

      auto const rendered = render_plain_unwrapped(text, policy);
      if (is_nonfinite_decimal_text(rendered))
      {
        if (!trim_trailing_padding) return presentation::pad_center(text, width, policy);

        auto const used = presentation::display_width(text, policy);
        if (used >= width) return text;
        presentation::styled_text result;
        result.append(spaces((width - used) / 2)).append(text);
        return result;
      }

      auto const point = rendered.find('.');
      auto const before_decimal =
          point == std::string::npos ? presentation::display_width(rendered, policy)
                                     : presentation::display_width(std::string_view(rendered).substr(0, point), policy);
      auto const used = presentation::display_width(text, policy);
      auto const desired_left_padding = *decimal_position > before_decimal ? *decimal_position - before_decimal : 0;
      auto const max_left_padding = width > used ? width - used : 0;
      auto const left_padding = std::min(desired_left_padding, max_left_padding);
      auto const padded_width = used + left_padding;

      presentation::styled_text result;
      if (left_padding > 0) result.append(spaces(left_padding));
      result.append(text);
      if (!trim_trailing_padding && padded_width < width) result.append(spaces(width - padded_width));
      return result;
    }
  }
  if (trim_trailing_padding) return text;
  return presentation::pad_right(text, width, policy);
}

[[nodiscard]] std::optional<std::size_t> decimal_position(presentation::styled_text const& text,
                                                          presentation::output_policy const& policy)
{
  auto const rendered = render_plain_unwrapped(text, policy);
  if (is_nonfinite_decimal_text(rendered))
  {
    return std::nullopt;
  }

  auto const point = rendered.find('.');
  if (point == std::string::npos) return presentation::display_width(rendered, policy);
  return presentation::display_width(std::string_view(rendered).substr(0, point), policy);
}

[[nodiscard]] std::vector<presentation::styled_text> wrap_cell(presentation::styled_text const& text, std::size_t width,
                                                               presentation::output_policy const& policy)
{
  auto lines = presentation::wrap_text(text, std::max<std::size_t>(width, 1), policy);
  if (lines.empty()) lines.push_back(presentation::styled_text{});
  return lines;
}

[[nodiscard]] std::vector<wrapped_cell_line> wrap_streaming_cell(detail::streaming_cell const& cell, std::size_t width,
                                                                 presentation::output_policy const& policy)
{
  auto const raw_lines = wrap_cell(cell.text, width, policy);
  std::vector<wrapped_cell_line> lines;
  lines.reserve(raw_lines.size());
  lines.push_back(wrapped_cell_line{.text = raw_lines.front(),
                                    .continuation = false,
                                    .decimal_candidate = cell.decimal_candidate,
                                    .decimal_exception = cell.decimal_exception});

  auto const prefix = continuation_prefix_for_width(width, policy);
  if (prefix.width == 0)
  {
    for (std::size_t i = 1; i != raw_lines.size(); ++i)
    {
      lines.push_back(wrapped_cell_line{.text = raw_lines[i], .continuation = true});
    }
    return lines;
  }

  auto const continuation_width = std::max<std::size_t>(width - prefix.width, 1);
  for (std::size_t i = 1; i != raw_lines.size(); ++i)
  {
    for (auto const& segment : wrap_cell(raw_lines[i], continuation_width, policy))
    {
      presentation::styled_text marked;
      marked.append(prefix.text).append(segment);
      lines.push_back(wrapped_cell_line{.text = std::move(marked), .continuation = true});
    }
  }
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

column_width fit(std::size_t minimum)
{
  return column_width{.mode = column_width_mode::fit, .value = 1, .minimum = std::max<std::size_t>(minimum, 1)};
}
} // namespace width

namespace format
{
column_format text(terminal::TerminalStyle style)
{
  return column_format{.kind = column_value_kind::text, .style = std::move(style)};
}

column_format number(std::string pattern, terminal::TerminalStyle style)
{
  return column_format{.kind = column_value_kind::number,
                       .pattern = std::move(pattern),
                       .style = std::move(style),
                       .exception_style = terminal::TerminalStyle(std::string_view("Yellow;Bold")),
                       .minimum_width = 10};
}

column_format fixed(int precision, terminal::TerminalStyle style)
{
  auto const digits = static_cast<std::size_t>(std::max(precision, 0));
  auto result = number(fmt::format("{{:.{}f}}", digits), std::move(style));
  result.minimum_width = digits + 6;
  return result;
}

column_format scientific(int precision, terminal::TerminalStyle style)
{
  auto const digits = static_cast<std::size_t>(std::max(precision, 0));
  auto result = number(fmt::format("{{:.{}e}}", digits), std::move(style));
  result.minimum_width = digits + 8;
  return result;
}

column_format general(int precision, terminal::TerminalStyle style)
{
  auto const digits = static_cast<std::size_t>(std::max(precision, 0));
  auto result = number(fmt::format("{{:.{}g}}", digits), std::move(style));
  result.minimum_width = digits + 6;
  return result;
}
} // namespace format

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

streaming_table& streaming_table::column(std::string heading, column_width width_spec, table_alignment alignment,
                                         column_format cell_format)
{
  this->ensure_can_change_schema();
  if (cell_format.kind == column_value_kind::automatic && alignment == table_alignment::decimal)
  {
    cell_format = format::number();
  }
  columns_.push_back(column_spec{
      .heading = std::move(heading), .width = width_spec, .alignment = alignment, .format = std::move(cell_format)});
  return *this;
}

streaming_table& streaming_table::column(std::string heading, table_alignment alignment)
{
  auto cell_format = alignment == table_alignment::decimal ? format::number() : column_format{};
  return this->column(std::move(heading), default_column_width(cell_format), alignment, std::move(cell_format));
}

streaming_table& streaming_table::column(std::string heading, column_format cell_format)
{
  auto const alignment = default_column_alignment(cell_format);
  auto const width_spec = default_column_width(cell_format);
  return this->column(std::move(heading), width_spec, alignment, std::move(cell_format));
}

streaming_table& streaming_table::column(std::string heading, column_width width_spec, column_format cell_format)
{
  auto const alignment = default_column_alignment(cell_format);
  return this->column(std::move(heading), width_spec, alignment, std::move(cell_format));
}

streaming_table& streaming_table::column(std::string heading, table_alignment alignment, column_format cell_format)
{
  if (cell_format.kind == column_value_kind::automatic && alignment == table_alignment::decimal)
  {
    cell_format = format::number();
  }
  auto const width_spec = default_column_width(cell_format);
  return this->column(std::move(heading), width_spec, alignment, std::move(cell_format));
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
  std::vector<detail::streaming_cell> formatted_cells;
  formatted_cells.reserve(cells.size());
  for (std::size_t i = 0; i != cells.size(); ++i)
  {
    formatted_cells.push_back(
        detail::format_cell_value(i < columns_.size() ? columns_[i].format : column_format{}, std::move(cells[i])));
  }
  this->emit_rows(formatted_cells, where);
}

void streaming_table::row(std::vector<std::string> cells, std::source_location where)
{
  std::vector<detail::streaming_cell> formatted_cells;
  formatted_cells.reserve(cells.size());
  for (std::size_t i = 0; i != cells.size(); ++i)
  {
    formatted_cells.push_back(
        detail::format_cell_value(i < columns_.size() ? columns_[i].format : column_format{}, cells[i]));
  }
  this->emit_rows(formatted_cells, where);
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

  auto const table_width = effective_wrap_width(policy_);
  auto const separator_width = columns_.size() > 1 ? (columns_.size() - 1) * 2 : 0;
  if (table_width <= separator_width)
  {
    vertical_fallback_ = true;
    return;
  }

  widths_.assign(columns_.size(), 1);
  decimal_positions_.assign(columns_.size(), std::nullopt);
  std::size_t fixed_total = 0;
  std::size_t fit_total = 0;
  std::size_t share_minimum_total = 0;
  std::size_t share_weight_total = 0;

  for (std::size_t i = 0; i != columns_.size(); ++i)
  {
    auto const& column = columns_[i];
    switch (column.width.mode)
    {
      case column_width_mode::fixed:
        widths_[i] = std::max<std::size_t>(column.width.value, 1);
        fixed_total += widths_[i];
        break;
      case column_width_mode::share:
      {
        auto const heading_width = presentation::display_width(column.heading, policy_);
        widths_[i] = std::max({std::size_t{1}, column.width.minimum, heading_width});
        share_minimum_total += widths_[i];
        share_weight_total += std::max<std::size_t>(column.width.value, 1);
        break;
      }
      case column_width_mode::fit:
      {
        auto const heading_width = presentation::display_width(column.heading, policy_);
        widths_[i] = std::max({std::size_t{1}, column.width.minimum, heading_width});
        fit_total += widths_[i];
        break;
      }
    }
  }

  if (fixed_total + fit_total + share_minimum_total + separator_width > table_width)
  {
    vertical_fallback_ = true;
    return;
  }

  if (share_weight_total == 0) return;

  auto remaining = table_width - separator_width - fixed_total - fit_total - share_minimum_total;
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

void streaming_table::expand_fit_columns(std::vector<detail::streaming_cell> const& cells)
{
  if (vertical_fallback_) return;

  auto const separator_width = columns_.size() > 1 ? (columns_.size() - 1) * 2 : 0;
  auto const table_width = effective_wrap_width(policy_);

  auto used_width = [&]() {
    std::size_t total = separator_width;
    for (auto const width : widths_)
      total += width;
    return total;
  };

  auto minimum_width = [&](std::size_t column) {
    auto const& spec = columns_[column];
    if (spec.width.mode == column_width_mode::fixed) return std::max<std::size_t>(spec.width.value, 1);
    auto const heading_width = presentation::display_width(spec.heading, policy_);
    return std::max({std::size_t{1}, spec.width.minimum, heading_width});
  };

  auto shrink_share_columns = [&](std::size_t amount) {
    std::size_t taken = 0;
    while (amount > 0)
    {
      auto best = columns_.size();
      std::size_t best_extra = 0;
      for (std::size_t i = 0; i != columns_.size(); ++i)
      {
        if (columns_[i].width.mode != column_width_mode::share) continue;
        auto const minimum = minimum_width(i);
        auto const extra = widths_[i] > minimum ? widths_[i] - minimum : 0;
        if (extra > best_extra)
        {
          best = i;
          best_extra = extra;
        }
      }

      if (best == columns_.size()) break;

      auto const step = std::min(amount, best_extra);
      widths_[best] -= step;
      amount -= step;
      taken += step;
    }
    return taken;
  };

  for (std::size_t i = 0; i != columns_.size(); ++i)
  {
    if (columns_[i].width.mode != column_width_mode::fit) continue;

    auto const value_width = presentation::display_width(cells[i].text, policy_);
    auto const desired = std::max(value_width, minimum_width(i));
    if (desired <= widths_[i]) continue;

    auto needed = desired - widths_[i];
    auto const used = used_width();
    auto const unused = table_width > used ? table_width - used : std::size_t{0};
    auto const from_unused = std::min(needed, unused);
    widths_[i] += from_unused;
    needed -= from_unused;

    if (needed > 0)
    {
      auto const from_share = shrink_share_columns(needed);
      widths_[i] += from_share;
    }
  }
}

void streaming_table::emit_rows(std::vector<detail::streaming_cell> const& cells, std::source_location where)
{
  if (cells.size() != columns_.size())
  {
    throw std::invalid_argument("display streaming table row cell count does not match the column count");
  }

  this->resolve_widths();
  if (!vertical_fallback_) this->expand_fit_columns(cells);

  presentation::styled_text text;
  if (!header_emitted_)
  {
    if (!title_.empty())
    {
      presentation::styled_text title;
      title.append(title_, terminal::TerminalStyle(std::string_view("Cyan;Bold")));
      for (auto const& line : wrap_streaming_cell(detail::streaming_cell{.text = std::move(title)},
                                                  effective_wrap_width(policy_), policy_))
      {
        text.append(line.text).append("\n");
      }
    }
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

    auto const table_width = effective_wrap_width(policy_);
    auto const marker_width = presentation::display_width(continuation_marker(), policy_);
    auto const value_width =
        table_width > key_width + marker_width + 1 ? table_width - key_width - marker_width - 1 : std::size_t{1};
    for (std::size_t i = 0; i != columns_.size(); ++i)
    {
      auto const wrapped = wrap_cell(cells[i].text, value_width, policy_);
      text.append(presentation::pad_right(columns_[i].heading + ":", key_width + 1, policy_),
                  terminal::TerminalStyle(std::string_view("LightGray;Bold")))
          .append(" ")
          .append(wrapped.front())
          .append("\n");
      for (std::size_t line = 1; line != wrapped.size(); ++line)
      {
        text.append(spaces(key_width)).append(continuation_marker()).append(" ").append(wrapped[line]).append("\n");
      }
    }
    emit(std::move(text), destination_, false, where);
    return;
  }

  std::vector<std::optional<std::size_t>> row_decimal_positions(cells.size());
  for (std::size_t i = 0; i != cells.size(); ++i)
  {
    if (columns_[i].alignment != table_alignment::decimal) continue;
    if (!cells[i].decimal_candidate) continue;
    auto const position = decimal_position(cells[i].text, policy_);
    if (!position.has_value()) continue;
    if (!decimal_positions_[i].has_value() || *decimal_positions_[i] < *position)
    {
      decimal_positions_[i] = position;
    }
    row_decimal_positions[i] = decimal_positions_[i];
  }

  std::vector<std::vector<wrapped_cell_line>> wrapped_cells;
  wrapped_cells.reserve(cells.size());
  std::size_t row_lines = 1;
  for (std::size_t i = 0; i != cells.size(); ++i)
  {
    auto wrapped = wrap_streaming_cell(cells[i], widths_[i], policy_);
    row_lines = std::max(row_lines, wrapped.size());
    wrapped_cells.push_back(std::move(wrapped));
  }

  for (std::size_t line = 0; line != row_lines; ++line)
  {
    auto last_column = columns_.size() - 1;
    if (line > 0)
    {
      for (std::size_t column = columns_.size(); column > 0; --column)
      {
        auto const index = column - 1;
        if (line < wrapped_cells[index].size() && !wrapped_cells[index][line].text.empty())
        {
          last_column = index;
          break;
        }
      }
    }

    for (std::size_t column = 0; column <= last_column; ++column)
    {
      if (column != 0) text.append("  ");
      auto const has_cell_line = line < wrapped_cells[column].size();
      auto const cell_line = has_cell_line ? wrapped_cells[column][line] : wrapped_cell_line{};
      auto const alignment = cell_line.continuation ? table_alignment::left : columns_[column].alignment;
      auto const effective_alignment =
          alignment == table_alignment::decimal && cell_line.decimal_exception ? table_alignment::center : alignment;
      auto aligned = align_cell(cell_line.text, widths_[column], effective_alignment, policy_, column == last_column,
                                row_decimal_positions[column]);
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
