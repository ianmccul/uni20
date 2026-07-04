#pragma once

#include "presentation.hpp"

#include <fmt/core.h>
#include <fmt/format.h>

#include <cstddef>
#include <functional>
#include <initializer_list>
#include <optional>
#include <source_location>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace uni20::display
{

/// \brief Selects the ordinary display stream for a message.
enum class stream
{
  out,
  err
};

/// \brief Width allocation mode for a streaming display table column.
enum class column_width_mode
{
  fixed,
  share,
  fit
};

/// \brief Schema-level width request for a streaming display table column.
struct column_width
{
    column_width_mode mode = column_width_mode::share;
    std::size_t value = 1;
    std::size_t minimum = 1;
};

namespace width
{
/// \brief Request an exact display-cell width for a streaming table column.
[[nodiscard]] column_width fixed(std::size_t cells);

/// \brief Request a flexible streaming table column with the given share weight.
[[nodiscard]] column_width share(std::size_t weight = 1, std::size_t minimum = 1);

/// \brief Request a non-greedy column that starts at a minimum width and grows when row values need it.
///
/// \details Fit columns do not receive terminal slack during initial layout.
///          When a later row needs more room, they grow by using unused space
///          or by shrinking shared-width columns toward their minimum widths.
[[nodiscard]] column_width fit(std::size_t minimum = 1);
} // namespace width

/// \brief Semantic value type expected by a streaming table column.
enum class column_value_kind
{
  automatic,
  text,
  number
};

/// \brief Default formatting and styling for values emitted in a streaming table column.
struct column_format
{
    column_value_kind kind = column_value_kind::automatic;
    std::string pattern = "{}";
    terminal::TerminalStyle style = {};
    terminal::TerminalStyle exception_style = {};
    /// \brief Preferred fit-column minimum width for numeric columns. Zero selects the default for the format.
    std::size_t minimum_width = 0;
};

namespace format
{
/// \brief Treat column values as text, optionally applying a style.
[[nodiscard]] column_format text(terminal::TerminalStyle style = {});

/// \brief Treat column values as numeric and format them with a runtime fmt pattern.
[[nodiscard]] column_format number(std::string pattern = "{}", terminal::TerminalStyle style = {});

/// \brief Format numeric values with fixed-point notation and the requested precision.
[[nodiscard]] column_format fixed(int precision, terminal::TerminalStyle style = {});

/// \brief Format numeric values with scientific notation and the requested precision.
[[nodiscard]] column_format scientific(int precision, terminal::TerminalStyle style = {});

/// \brief Format numeric values with general notation and the requested precision.
[[nodiscard]] column_format general(int precision, terminal::TerminalStyle style = {});
} // namespace format

namespace detail
{
struct streaming_cell
{
    presentation::styled_text text;
    bool decimal_candidate = false;
    bool decimal_exception = false;
};
} // namespace detail

/// \brief Presentation payload carried to a display sink before final rendering.
using event_content = std::variant<presentation::styled_text, presentation::report_builder>;

/// \brief Human-facing display event routed through the active sink.
struct event
{
    stream destination = stream::err;
    event_content content;
    bool newline = true;
    std::string context;
    std::source_location where;
};

/// \brief Callable that receives display events for final routing and rendering.
using sink = std::function<void(event const&)>;

/// \brief Compile-time checked display format string with call-site metadata.
template <typename... Args> struct format_string
{
    fmt::format_string<Args...> format;
    std::source_location where;

    template <typename String>
    consteval format_string(String const& value, std::source_location location = std::source_location::current())
        : format(value), where(location)
    {}
};

/// \brief Non-deduced format-string parameter type used by display helper templates.
template <typename... Args> using checked_format_string = std::type_identity_t<format_string<Args...>>;

/// \brief Replace the process-wide display sink.
void set_sink(sink replacement);

/// \brief Restore the default C++ terminal display sink.
void reset_sink();

/// \brief Return a copy of the active display sink.
[[nodiscard]] sink current_sink();

/// \brief Temporarily replace the display sink and restore it on destruction.
class scoped_sink {
  public:
    explicit scoped_sink(sink replacement);
    scoped_sink(scoped_sink const&) = delete;
    scoped_sink& operator=(scoped_sink const&) = delete;
    scoped_sink(scoped_sink&& other) noexcept;
    scoped_sink& operator=(scoped_sink&& other) noexcept;
    ~scoped_sink();

  private:
    sink previous_;
    bool active_ = true;
};

/// \brief Schema-first table for progress output where rows are emitted immediately.
class streaming_table {
  public:
    explicit streaming_table(std::string title = {}, stream destination = stream::out);

    streaming_table& column(std::string heading, column_width width_spec = width::share(),
                            presentation::table_alignment alignment = presentation::table_alignment::right,
                            column_format format = {});
    /// \brief Add a flexible-width column with an explicit alignment.
    streaming_table& column(std::string heading, presentation::table_alignment alignment);
    /// \brief Add a column whose width and alignment are inferred from its format.
    ///
    /// \details Numeric formats use decimal alignment and non-greedy fit width.
    ///          Text and automatic formats use shared width.
    streaming_table& column(std::string heading, column_format format);
    /// \brief Add a column whose alignment is inferred from its format.
    streaming_table& column(std::string heading, column_width width_spec, column_format format);
    /// \brief Add a flexible-width column with explicit alignment and format.
    streaming_table& column(std::string heading, presentation::table_alignment alignment, column_format format);
    streaming_table& wrap_width(std::size_t width);
    streaming_table& header_separator(bool enabled = true);

    void row(std::vector<presentation::styled_text> cells,
             std::source_location where = std::source_location::current());
    void row(std::vector<std::string> cells, std::source_location where = std::source_location::current());
    void row(std::initializer_list<std::string> cells, std::source_location where = std::source_location::current());

    template <typename... Values> void row(Values&&... values);

  private:
    struct column_spec
    {
        std::string heading;
        column_width width;
        presentation::table_alignment alignment = presentation::table_alignment::right;
        column_format format;
    };

    std::string title_;
    stream destination_ = stream::out;
    presentation::output_policy policy_;
    std::vector<column_spec> columns_;
    std::vector<std::size_t> widths_;
    std::vector<std::optional<std::size_t>> decimal_positions_;
    bool header_separator_ = true;
    bool header_emitted_ = false;
    bool widths_resolved_ = false;
    bool vertical_fallback_ = false;

    void ensure_can_change_schema() const;
    void resolve_widths();
    void expand_fit_columns(std::vector<detail::streaming_cell> const& cells);
    void emit_rows(std::vector<detail::streaming_cell> const& cells, std::source_location where);
};

/// \brief Create a schema-first streaming display table.
[[nodiscard]] streaming_table table(std::string title = {}, stream destination = stream::out);

/// \brief Build styled status text without emitting it.
template <typename... Args>
[[nodiscard]] presentation::styled_text status_cell(presentation::semantic_glyph glyph,
                                                    checked_format_string<Args...> format, Args&&... args);

/// \brief Emit styled text through the active display sink.
void emit(presentation::styled_text text, stream destination = stream::err, bool newline = true,
          std::source_location where = std::source_location::current());

/// \brief Emit a presentation report through the active display sink.
void emit(presentation::report_builder report, stream destination = stream::err, bool newline = true,
          std::source_location where = std::source_location::current());

/// \brief Emit a semantic status message through the active display sink.
template <typename... Args>
void status(presentation::semantic_glyph glyph, checked_format_string<Args...> format, Args&&... args);

/// \brief Emit an informational display message.
template <typename... Args> void info(checked_format_string<Args...> format, Args&&... args);

/// \brief Emit a successful/completed display message.
template <typename... Args> void success(checked_format_string<Args...> format, Args&&... args);

/// \brief Emit a non-fatal warning display message.
template <typename... Args> void warning(checked_format_string<Args...> format, Args&&... args);

/// \brief Emit a failure display message.
template <typename... Args> void failure(checked_format_string<Args...> format, Args&&... args);

/// \brief Emit a fatal-status display message.
template <typename... Args> void fatal(checked_format_string<Args...> format, Args&&... args);

/// \brief Emit a partially-completed display message.
template <typename... Args> void partial(checked_format_string<Args...> format, Args&&... args);

/// \brief Emit an intentionally deferred display message.
template <typename... Args> void deferred(checked_format_string<Args...> format, Args&&... args);

/// \brief Emit a skipped/unavailable display message.
template <typename... Args> void skipped(checked_format_string<Args...> format, Args&&... args);

namespace detail
{
[[nodiscard]] terminal::TerminalStyle status_style(presentation::semantic_glyph glyph);

template <typename T>
inline constexpr bool display_numeric_value =
    std::is_arithmetic_v<std::remove_cvref_t<T>> && !std::is_same_v<std::remove_cvref_t<T>, bool> &&
    !std::is_same_v<std::remove_cvref_t<T>, char> && !std::is_same_v<std::remove_cvref_t<T>, signed char> &&
    !std::is_same_v<std::remove_cvref_t<T>, unsigned char>;

[[nodiscard]] inline bool expects_number(column_format const& format)
{
  return format.kind == column_value_kind::number;
}

[[nodiscard]] inline streaming_cell make_text_cell(presentation::styled_text value, column_format const& format = {})
{
  auto const exception = expects_number(format);
  return streaming_cell{.text = std::move(value), .decimal_candidate = false, .decimal_exception = exception};
}

[[nodiscard]] inline streaming_cell format_cell_value(column_format const& format, presentation::styled_text value)
{
  return make_text_cell(std::move(value), format);
}

[[nodiscard]] inline streaming_cell format_cell_value(column_format const& format,
                                                      presentation::table_cell const& value)
{
  return make_text_cell(value.content, format);
}

[[nodiscard]] inline presentation::styled_text make_plain_cell(std::string_view value,
                                                               terminal::TerminalStyle style = {})
{
  presentation::styled_text text;
  text.append(value, std::move(style));
  return text;
}

[[nodiscard]] inline streaming_cell format_cell_value(column_format const& format, char const* value)
{
  auto const exception = expects_number(format);
  return streaming_cell{.text = make_plain_cell(value != nullptr ? std::string_view(value) : std::string_view{},
                                                exception ? format.exception_style : format.style),
                        .decimal_candidate = false,
                        .decimal_exception = exception};
}

template <typename T> [[nodiscard]] streaming_cell format_cell_value(column_format const& format, T&& value)
{
  using value_type = std::remove_cvref_t<T>;
  if constexpr (display_numeric_value<T>)
  {
    auto const numeric = expects_number(format);
    auto const finite = uni20::isfinite(value);
    auto text = make_plain_cell(numeric ? fmt::format(fmt::runtime(format.pattern), std::forward<T>(value))
                                        : fmt::format("{}", std::forward<T>(value)),
                                finite ? format.style : format.exception_style);
    return streaming_cell{
        .text = std::move(text), .decimal_candidate = numeric && finite, .decimal_exception = numeric && !finite};
  }
  else if constexpr (std::is_same_v<value_type, std::string>)
  {
    auto const exception = expects_number(format);
    return streaming_cell{
        .text = make_plain_cell(std::forward<T>(value), exception ? format.exception_style : format.style),
        .decimal_candidate = false,
        .decimal_exception = exception};
  }
  else if constexpr (std::is_same_v<value_type, std::string_view>)
  {
    auto const exception = expects_number(format);
    return streaming_cell{.text = make_plain_cell(value, exception ? format.exception_style : format.style),
                          .decimal_candidate = false,
                          .decimal_exception = exception};
  }
  else if constexpr (std::is_array_v<value_type>)
  {
    auto const exception = expects_number(format);
    return streaming_cell{
        .text = make_plain_cell(std::string_view(value), exception ? format.exception_style : format.style),
        .decimal_candidate = false,
        .decimal_exception = exception};
  }
  else if constexpr (std::is_convertible_v<T, char const*>)
  {
    return format_cell_value(format, static_cast<char const*>(value));
  }
  else
  {
    auto const exception = expects_number(format);
    return streaming_cell{.text = make_plain_cell(fmt::format("{}", std::forward<T>(value)),
                                                  exception ? format.exception_style : format.style),
                          .decimal_candidate = false,
                          .decimal_exception = exception};
  }
}
} // namespace detail

template <typename... Args>
presentation::styled_text status_cell(presentation::semantic_glyph glyph, checked_format_string<Args...> format,
                                      Args&&... args)
{
  auto style = detail::status_style(glyph);
  presentation::styled_text text;
  text.append(glyph, style).append(" ").append(fmt::format(format.format, std::forward<Args>(args)...), style);
  return text;
}

template <typename... Args>
void status(presentation::semantic_glyph glyph, checked_format_string<Args...> format, Args&&... args)
{
  emit(status_cell(glyph, format, std::forward<Args>(args)...), stream::err, true, format.where);
}

template <typename... Args> void info(checked_format_string<Args...> format, Args&&... args)
{
  status(presentation::semantic_glyph::info, format, std::forward<Args>(args)...);
}

template <typename... Args> void success(checked_format_string<Args...> format, Args&&... args)
{
  status(presentation::semantic_glyph::success, format, std::forward<Args>(args)...);
}

template <typename... Args> void warning(checked_format_string<Args...> format, Args&&... args)
{
  status(presentation::semantic_glyph::warning, format, std::forward<Args>(args)...);
}

template <typename... Args> void failure(checked_format_string<Args...> format, Args&&... args)
{
  status(presentation::semantic_glyph::failure, format, std::forward<Args>(args)...);
}

template <typename... Args> void fatal(checked_format_string<Args...> format, Args&&... args)
{
  status(presentation::semantic_glyph::fatal, format, std::forward<Args>(args)...);
}

template <typename... Args> void partial(checked_format_string<Args...> format, Args&&... args)
{
  status(presentation::semantic_glyph::partial, format, std::forward<Args>(args)...);
}

template <typename... Args> void deferred(checked_format_string<Args...> format, Args&&... args)
{
  status(presentation::semantic_glyph::deferred, format, std::forward<Args>(args)...);
}

template <typename... Args> void skipped(checked_format_string<Args...> format, Args&&... args)
{
  status(presentation::semantic_glyph::skipped, format, std::forward<Args>(args)...);
}

template <typename... Values> void streaming_table::row(Values&&... values)
{
  std::vector<detail::streaming_cell> cells;
  cells.reserve(sizeof...(Values));
  std::size_t column = 0;
  ((cells.push_back(detail::format_cell_value(column < columns_.size() ? columns_[column].format : column_format{},
                                              std::forward<Values>(values))),
    ++column),
   ...);
  this->emit_rows(cells, std::source_location::current());
}

} // namespace uni20::display
