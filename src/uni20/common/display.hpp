#pragma once

#include "presentation.hpp"

#include <fmt/core.h>

#include <cstddef>
#include <functional>
#include <initializer_list>
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
  share
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
} // namespace width

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
                            presentation::table_alignment alignment = presentation::table_alignment::right);
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
    };

    std::string title_;
    stream destination_ = stream::out;
    presentation::output_policy policy_;
    std::vector<column_spec> columns_;
    std::vector<std::size_t> widths_;
    bool header_separator_ = true;
    bool header_emitted_ = false;
    bool widths_resolved_ = false;
    bool vertical_fallback_ = false;

    void ensure_can_change_schema() const;
    void resolve_widths();
    void emit_rows(std::vector<presentation::styled_text> const& cells, std::source_location where);
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

[[nodiscard]] inline presentation::styled_text format_cell_value(presentation::styled_text value) { return value; }

[[nodiscard]] inline presentation::styled_text format_cell_value(presentation::table_cell const& value)
{
  return value.content;
}

[[nodiscard]] inline presentation::styled_text make_plain_cell(std::string_view value)
{
  presentation::styled_text text;
  text.append(value);
  return text;
}

[[nodiscard]] inline presentation::styled_text format_cell_value(char const* value)
{
  return make_plain_cell(value != nullptr ? std::string_view(value) : std::string_view{});
}

template <typename T> [[nodiscard]] presentation::styled_text format_cell_value(T&& value)
{
  using value_type = std::remove_cvref_t<T>;
  if constexpr (std::is_same_v<value_type, std::string>)
  {
    return make_plain_cell(std::forward<T>(value));
  }
  else if constexpr (std::is_same_v<value_type, std::string_view>)
  {
    return make_plain_cell(value);
  }
  else if constexpr (std::is_array_v<value_type>)
  {
    return make_plain_cell(std::string_view(value));
  }
  else if constexpr (std::is_convertible_v<T, char const*>)
  {
    return format_cell_value(static_cast<char const*>(value));
  }
  else
  {
    return make_plain_cell(fmt::format("{}", std::forward<T>(value)));
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
  this->row(std::vector<presentation::styled_text>{detail::format_cell_value(std::forward<Values>(values))...});
}

} // namespace uni20::display
