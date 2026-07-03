#pragma once

#include "terminal.hpp"

#include <uni20/core/math.hpp>
#include <uni20/core/scalar_io.hpp>

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <deque>
#include <fmt/core.h>
#include <initializer_list>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace uni20::presentation
{

/// \brief Selects when terminal color/style control sequences may be emitted.
enum class color_mode
{
  never,
  automatic,
  always
};

/// \brief Selects the preferred glyph family for semantic symbols and ornaments.
enum class glyph_set
{
  unicode,
  emoji,
  ascii
};

/// \brief Selects how non-ASCII text code points are rendered.
enum class text_charset
{
  utf8,
  ascii_escape,
  ascii_replace
};

/// \brief Selects the unit used by alignment and clipping helpers.
enum class width_mode
{
  bytes,
  display_cells
};

/// \brief Selects how East Asian ambiguous-width code points are measured.
enum class ambiguous_width
{
  narrow,
  wide
};

/// \brief Selects how invalid UTF-8 bytes are represented.
enum class invalid_utf8
{
  escape,
  replace
};

/// \brief Selects the richness of decorative line and tree glyphs.
enum class ornament_mode
{
  none,
  minimal,
  rich
};

/// \brief Selects how real floating-point values are formatted.
enum class real_notation
{
  general,
  fixed,
  scientific
};

/// \brief Selects table cell alignment in report-style output.
enum class table_alignment
{
  left,
  right,
  center,
  decimal
};

/// \brief Selects the glyph family used for table rules and borders.
enum class table_rule_style
{
  single,
  double_line
};

/// \brief Numeric scalar formatting controls shared by tensor-style renderers.
struct numeric_format_options
{
    int float32_precision = 6;
    int float64_precision = 15;
    int long_double_precision = 18;
    int float128_precision = 36;
    real_notation notation = real_notation::general;
    bool normalize_negative_zero = true;
    std::string imaginary_unit = "i";
};

/// \brief Names semantic glyphs independently of their rendered spelling.
enum class semantic_glyph
{
  success,
  failure,
  fatal,
  warning,
  info,
  partial,
  deferred,
  skipped,
  arrow_right,
  arrow_left,
  arrow_up,
  arrow_down,
  arrow_left_right,
  ellipsis,
  box_horizontal,
  box_vertical,
  box_top_left,
  box_top_right,
  box_bottom_left,
  box_bottom_right,
  box_round_top_left,
  box_round_top_right,
  box_round_bottom_left,
  box_round_bottom_right,
  box_tee_left,
  box_tee_right,
  box_tee_up,
  box_tee_down,
  box_cross,
  box_double_horizontal,
  box_double_vertical,
  box_double_top_left,
  box_double_top_right,
  box_double_bottom_left,
  box_double_bottom_right,
  box_double_tee_left,
  box_double_tee_right,
  box_double_tee_up,
  box_double_tee_down,
  box_double_cross,
  box_double_vertical_tee_right_single,
  box_double_vertical_tee_left_single,
  box_double_vertical_cross_single_horizontal,
  box_double_down_single_horizontal,
  box_double_up_single_horizontal,
  box_single_vertical_tee_right_double,
  box_single_vertical_tee_left_double,
  box_single_vertical_cross_double_horizontal,
  box_single_down_double_horizontal,
  box_single_up_double_horizontal,
  box_diagonal_forward,
  box_diagonal_back,
  box_diagonal_cross,
  tree_branch,
  tree_last,
  tree_vertical,
  tree_space,
  bullet,
  matrix_top_left,
  matrix_top_right,
  matrix_middle_left,
  matrix_middle_right,
  matrix_bottom_left,
  matrix_bottom_right
};

/// \brief Output policy shared by presentation renderers and layout helpers.
struct output_policy
{
    color_mode color = color_mode::automatic;
    glyph_set glyphs = glyph_set::emoji;
    text_charset charset = text_charset::utf8;
    width_mode width = width_mode::display_cells;
    ambiguous_width ambiguous = ambiguous_width::narrow;
    invalid_utf8 invalid = invalid_utf8::escape;
    std::size_t tab_width = 8;
    std::optional<std::size_t> wrap_width = std::nullopt;
    ornament_mode ornaments = ornament_mode::rich;
    std::FILE* output_stream = stdout;
};

/// \brief A text span carrying style metadata but no terminal escape sequences.
struct styled_text_span
{
    std::string text;
    terminal::TerminalStyle style;
};

/// \brief A semantic glyph token carrying style metadata but no rendered spelling.
struct semantic_glyph_span
{
    semantic_glyph glyph = semantic_glyph::info;
    terminal::TerminalStyle style;
};

using presentation_span = std::variant<styled_text_span, semantic_glyph_span>;

/// \brief A sequence of styled text spans and semantic glyph tokens.
class styled_text {
  public:
    /// \brief Append a text span.
    /// \param text Text payload encoded as UTF-8.
    /// \param style Style metadata associated with the span.
    /// \return Reference to this object for chaining.
    styled_text& append(std::string_view text, terminal::TerminalStyle style = {});

    /// \brief Append a semantic glyph token.
    /// \param glyph Semantic glyph to render according to policy.
    /// \param style Style metadata associated with the glyph.
    /// \return Reference to this object for chaining.
    styled_text& append(semantic_glyph glyph, terminal::TerminalStyle style = {});

    /// \brief Append all spans from another styled text document.
    /// \param other Styled text document to append.
    /// \return Reference to this object for chaining.
    styled_text& append(styled_text const& other);

    /// \brief Return whether this document has no spans.
    /// \return True when no spans have been appended.
    [[nodiscard]] bool empty() const noexcept;

    /// \brief Return the stored presentation spans.
    /// \return Immutable span vector.
    [[nodiscard]] std::vector<presentation_span> const& spans() const noexcept;

  private:
    std::vector<presentation_span> spans_;
};

/// \brief Callable style wrapper that builds styled presentation text.
/// \details `text_style` keeps style metadata attached to spans without rendering
///          ANSI escape sequences. Final renderers still apply color, charset,
///          glyph, and display-width policy.
class text_style {
  public:
    /// \brief Construct an empty style wrapper.
    text_style() = default;

    /// \brief Construct from an existing terminal style.
    /// \param style Terminal style metadata to apply to generated spans.
    explicit text_style(terminal::TerminalStyle style) : style_(std::move(style)) {}

    /// \brief Construct from a terminal style specification string.
    /// \param spec Style specification such as `"Red;Bold"`.
    explicit text_style(std::string_view spec) : style_(spec) {}

    /// \brief Construct from a terminal style specification string.
    /// \param spec Style specification such as `"Red;Bold"`.
    explicit text_style(char const* spec) : text_style(spec != nullptr ? std::string_view(spec) : std::string_view{}) {}

    /// \brief Return the terminal style metadata carried by this wrapper.
    /// \return Terminal style metadata.
    [[nodiscard]] terminal::TerminalStyle const& terminal_style() const noexcept { return style_; }

    /// \brief Build a styled text span from literal text.
    /// \param text Text payload encoded as UTF-8.
    /// \return Styled presentation text.
    [[nodiscard]] styled_text operator()(std::string_view text) const
    {
      styled_text result;
      result.append(text, style_);
      return result;
    }

    /// \brief Build a styled text span from literal text.
    /// \param text Null-terminated text payload encoded as UTF-8.
    /// \return Styled presentation text.
    [[nodiscard]] styled_text operator()(char const* text) const
    {
      return (*this)(text != nullptr ? std::string_view(text) : std::string_view{});
    }

    /// \brief Build a styled semantic glyph span.
    /// \param glyph Semantic glyph to render according to policy.
    /// \return Styled presentation text.
    [[nodiscard]] styled_text operator()(semantic_glyph glyph) const
    {
      styled_text result;
      result.append(glyph, style_);
      return result;
    }

    /// \brief Format one non-string value with `fmt` and apply this style.
    /// \tparam T Value type.
    /// \param value Value to format with the default `{}` formatter.
    /// \return Styled presentation text.
    template <typename T>
      requires(!std::is_convertible_v<T &&, std::string_view> && !std::same_as<std::remove_cvref_t<T>, semantic_glyph>)
    [[nodiscard]] styled_text operator()(T&& value) const
    {
      styled_text result;
      result.append(fmt::format("{}", std::forward<T>(value)), style_);
      return result;
    }

    /// \brief Format text with `fmt` and apply this style.
    /// \tparam Args Format argument types.
    /// \param format Compile-time checked format string.
    /// \param args Format arguments.
    /// \return Styled presentation text.
    template <typename... Args>
      requires(sizeof...(Args) > 0)
    [[nodiscard]] styled_text operator()(fmt::format_string<Args...> format, Args&&... args) const
    {
      styled_text result;
      result.append(fmt::format(format, std::forward<Args>(args)...), style_);
      return result;
    }

    /// \brief Build a styled semantic glyph followed by styled literal text.
    /// \param glyph Semantic glyph to render according to policy.
    /// \param text Text payload encoded as UTF-8.
    /// \return Styled presentation text.
    [[nodiscard]] styled_text operator()(semantic_glyph glyph, std::string_view text) const
    {
      styled_text result;
      result.append(glyph, style_).append(" ").append(text, style_);
      return result;
    }

    /// \brief Build a styled semantic glyph followed by styled literal text.
    /// \param glyph Semantic glyph to render according to policy.
    /// \param text Null-terminated text payload encoded as UTF-8.
    /// \return Styled presentation text.
    [[nodiscard]] styled_text operator()(semantic_glyph glyph, char const* text) const
    {
      return (*this)(glyph, text != nullptr ? std::string_view(text) : std::string_view{});
    }

    /// \brief Build a styled semantic glyph followed by formatted styled text.
    /// \tparam Args Format argument types.
    /// \param glyph Semantic glyph to render according to policy.
    /// \param format Compile-time checked format string.
    /// \param args Format arguments.
    /// \return Styled presentation text.
    template <typename... Args>
      requires(sizeof...(Args) > 0)
    [[nodiscard]] styled_text operator()(semantic_glyph glyph, fmt::format_string<Args...> format, Args&&... args) const
    {
      styled_text result;
      result.append(glyph, style_).append(" ").append(fmt::format(format, std::forward<Args>(args)...), style_);
      return result;
    }

  private:
    terminal::TerminalStyle style_;
};

/// \brief Create a callable style wrapper from terminal style metadata.
/// \param style Terminal style metadata to apply to generated spans.
/// \return Callable style wrapper.
[[nodiscard]] inline text_style style(terminal::TerminalStyle style) { return text_style(std::move(style)); }

/// \brief Create a callable style wrapper from a style specification.
/// \param spec Style specification such as `"Red;Bold"`.
/// \return Callable style wrapper.
[[nodiscard]] inline text_style style(std::string_view spec) { return text_style(spec); }

/// \brief Create a callable style wrapper from a style specification.
/// \param spec Style specification such as `"Red;Bold"`.
/// \return Callable style wrapper.
[[nodiscard]] inline text_style style(char const* spec) { return text_style(spec); }

/// \brief Column specification for report tables.
struct table_column
{
    std::string heading;
    table_alignment alignment = table_alignment::right;
};

/// \brief Rule and border controls for report tables.
struct table_border_options
{
    bool outer = false;
    bool column_separators = false;
    bool row_separators = false;
    bool header_separator = false;
    table_rule_style rule_style = table_rule_style::single;
    std::size_t horizontal_padding = 1;
};

/// \brief Cell specification for table rows, including optional style, column span, and alignment.
struct table_cell
{
    styled_text content;
    std::size_t span = 1;
    std::optional<table_alignment> alignment = std::nullopt;

    table_cell() = default;
    table_cell(std::string_view text, std::size_t span_ = 1, std::optional<table_alignment> alignment_ = std::nullopt)
        : span(span_), alignment(alignment_)
    {
      content.append(text);
    }
    table_cell(char const* text, std::size_t span_ = 1, std::optional<table_alignment> alignment_ = std::nullopt)
        : table_cell(text != nullptr ? std::string_view(text) : std::string_view{}, span_, alignment_)
    {}
    table_cell(std::string text, std::size_t span_ = 1, std::optional<table_alignment> alignment_ = std::nullopt)
        : table_cell(std::string_view(text), span_, alignment_)
    {}
    table_cell(styled_text text, std::size_t span_ = 1, std::optional<table_alignment> alignment_ = std::nullopt)
        : content(std::move(text)), span(span_), alignment(alignment_)
    {}
};

/// \brief Build a table cell from styled presentation text.
/// \param content Styled content for the table cell.
/// \param span Number of table columns spanned by the cell.
/// \param alignment Optional per-cell alignment override.
/// \return Table cell specification.
[[nodiscard]] inline table_cell cell(styled_text content, std::size_t span = 1,
                                     std::optional<table_alignment> alignment = std::nullopt)
{
  return table_cell(std::move(content), span, alignment);
}

/// \brief Build a table cell from literal text.
/// \param text Text payload encoded as UTF-8.
/// \param span Number of table columns spanned by the cell.
/// \param alignment Optional per-cell alignment override.
/// \return Table cell specification.
[[nodiscard]] inline table_cell cell(std::string_view text, std::size_t span = 1,
                                     std::optional<table_alignment> alignment = std::nullopt)
{
  return table_cell(text, span, alignment);
}

/// \brief Build a table cell from literal text.
/// \param text Null-terminated text payload encoded as UTF-8.
/// \param span Number of table columns spanned by the cell.
/// \param alignment Optional per-cell alignment override.
/// \return Table cell specification.
[[nodiscard]] inline table_cell cell(char const* text, std::size_t span = 1,
                                     std::optional<table_alignment> alignment = std::nullopt)
{
  return table_cell(text, span, alignment);
}

namespace detail
{
[[nodiscard]] inline table_cell make_table_cell(table_cell const& value) { return value; }
[[nodiscard]] inline table_cell make_table_cell(styled_text const& value) { return table_cell(value); }
[[nodiscard]] inline table_cell make_table_cell(std::string const& value) { return table_cell(value); }
[[nodiscard]] inline table_cell make_table_cell(std::string_view value) { return table_cell(value); }
[[nodiscard]] inline table_cell make_table_cell(char const* value) { return table_cell(value); }

template <typename T> [[nodiscard]] table_cell make_table_cell(T const& value)
{
  return table_cell(fmt::format("{}", value));
}
} // namespace detail

/// \brief Explicit separator row inserted between table data rows.
struct table_separator
{
    table_rule_style style = table_rule_style::single;
};

using table_entry = std::variant<std::vector<table_cell>, table_separator>;

/// \brief A small report table builder for command-line examples and diagnostics.
class report_table {
  public:
    /// \brief Construct a titled report table.
    /// \param title Human-readable table title.
    explicit report_table(std::string title = {});

    /// \brief Add a column to the table.
    /// \param heading Column heading text.
    /// \param alignment Cell alignment for this column.
    /// \return Reference to this table for chaining.
    report_table& column(std::string heading, table_alignment alignment = table_alignment::right);

    /// \brief Add a row from already formatted cell strings.
    /// \param cells Cell text in column order.
    /// \return Reference to this table for chaining.
    report_table& row(std::vector<std::string> cells);

    /// \brief Add a row from table cell specifications.
    /// \param cells Cell specifications in column order.
    /// \return Reference to this table for chaining.
    report_table& row(std::vector<table_cell> cells);

    /// \brief Add a row from table cell specifications.
    /// \param cells Cell specifications in column order.
    /// \return Reference to this table for chaining.
    report_table& row(std::initializer_list<table_cell> cells);

    /// \brief Insert a separator before the generated heading row.
    /// \param style Rule glyph style to use.
    /// \return Reference to this table for chaining.
    report_table& top_separator(table_rule_style style = table_rule_style::single);

    /// \brief Insert a separator at the current body position.
    /// \param style Rule glyph style to use.
    /// \return Reference to this table for chaining.
    report_table& separator(table_rule_style style = table_rule_style::single);

    /// \brief Set table border and rule options.
    /// \param options Border and rule controls to apply.
    /// \return Reference to this table for chaining.
    report_table& borders(table_border_options options);

    /// \brief Set the rule style used by automatic borders and separators.
    /// \param style Rule glyph style to use.
    /// \return Reference to this table for chaining.
    report_table& border_style(table_rule_style style);

    /// \brief Enable or disable an outer border around the table.
    /// \param enabled Whether to draw the border.
    /// \return Reference to this table for chaining.
    report_table& outer_border(bool enabled = true);

    /// \brief Enable or disable vertical rules between columns.
    /// \param enabled Whether to draw column separators.
    /// \return Reference to this table for chaining.
    report_table& column_separators(bool enabled = true);

    /// \brief Enable or disable horizontal rules between rows.
    /// \param enabled Whether to draw row separators.
    /// \return Reference to this table for chaining.
    report_table& row_separators(bool enabled = true);

    /// \brief Enable or disable a horizontal rule after the heading row.
    /// \param enabled Whether to draw the heading separator.
    /// \return Reference to this table for chaining.
    report_table& header_separator(bool enabled = true);

    /// \brief Enable or disable a full table grid.
    /// \param enabled Whether to draw outer, column, row, and heading rules.
    /// \return Reference to this table for chaining.
    report_table& grid(bool enabled = true);

    /// \brief Enable a full table grid with a specific rule style.
    /// \param style Rule glyph style to use.
    /// \return Reference to this table for chaining.
    report_table& grid(table_rule_style style);

    /// \brief Add a row by formatting each value with `fmt`.
    /// \tparam Values Cell value types.
    /// \param values Values to format into cell text.
    /// \return Reference to this table for chaining.
    template <typename... Values> report_table& row(Values const&... values)
    {
      return this->row(std::vector<table_cell>{detail::make_table_cell(values)...});
    }

    /// \brief Return the table title.
    /// \return Table title text.
    [[nodiscard]] std::string const& title() const noexcept;

    /// \brief Return the column specifications.
    /// \return Immutable column vector.
    [[nodiscard]] std::vector<table_column> const& columns() const noexcept;

    /// \brief Return table entries, including row and separator entries.
    /// \return Immutable table entry vector.
    [[nodiscard]] std::vector<table_entry> const& entries() const noexcept;

    /// \brief Return explicit separators rendered before the heading row.
    /// \return Immutable top separator vector.
    [[nodiscard]] std::vector<table_rule_style> const& top_separators() const noexcept;

    /// \brief Return table border and rule options.
    /// \return Border and rule controls.
    [[nodiscard]] table_border_options const& border_options() const noexcept;

  private:
    std::string title_;
    std::vector<table_column> columns_;
    std::vector<table_entry> entries_;
    std::vector<table_rule_style> top_separators_;
    table_border_options border_options_;
};

/// \brief Builder for simple terminal reports with headings, status lines, fields, and tables.
class report_builder {
  public:
    /// \brief Construct a report with an optional title.
    /// \param title Report title.
    explicit report_builder(std::string title = {});

    /// \brief Add a semantic status line such as `✓ converged`.
    /// \param glyph Semantic status glyph.
    /// \param label Status label text.
    /// \return Reference to this report for chaining.
    report_builder& status(semantic_glyph glyph, std::string label);

    /// \brief Add a labeled field to the report header.
    /// \param key Field label.
    /// \param value Already formatted field value.
    /// \return Reference to this report for chaining.
    report_builder& field(std::string key, std::string value);

    /// \brief Add a labeled field after formatting the value with `fmt`.
    /// \tparam Value Field value type.
    /// \param key Field label.
    /// \param value Field value.
    /// \return Reference to this report for chaining.
    template <typename Value> report_builder& field(std::string key, Value const& value)
    {
      return this->field(std::move(key), fmt::format("{}", value));
    }

    /// \brief Add a titled table and return it for population.
    /// \param title Table title.
    /// \return Mutable table reference.
    report_table& table(std::string title);

    /// \brief Return the report title.
    /// \return Report title text.
    [[nodiscard]] std::string const& title() const noexcept;

    /// \brief Return the status lines.
    /// \return Immutable status vector.
    [[nodiscard]] std::vector<std::pair<semantic_glyph, std::string>> const& statuses() const noexcept;

    /// \brief Return the report fields.
    /// \return Immutable field vector.
    [[nodiscard]] std::vector<std::pair<std::string, std::string>> const& fields() const noexcept;

    /// \brief Return the report tables.
    /// \return Immutable table sequence.
    [[nodiscard]] std::deque<report_table> const& tables() const noexcept;

  private:
    std::string title_;
    std::vector<std::pair<semantic_glyph, std::string>> statuses_;
    std::vector<std::pair<std::string, std::string>> fields_;
    std::deque<report_table> tables_;
};

/// \brief Build a policy for terminal rendering.
/// \param stream Stream used for automatic terminal detection.
/// \return Terminal-oriented output policy.
[[nodiscard]] output_policy terminal_policy(std::FILE* stream = stdout);

/// \brief Build a policy for plain file/log rendering with no ANSI color.
/// \return Plain output policy.
[[nodiscard]] output_policy plain_policy();

/// \brief Build a strict ASCII policy.
/// \param charset ASCII handling mode for non-ASCII text.
/// \return Strict ASCII output policy.
[[nodiscard]] output_policy strict_ascii_policy(text_charset charset = text_charset::ascii_escape);

/// \brief Return whether a policy currently permits ANSI color output.
/// \param policy Output policy to inspect.
/// \return True when color/style sequences should be emitted.
[[nodiscard]] bool should_emit_color(output_policy const& policy);

/// \brief Render a semantic glyph according to an output policy.
/// \param glyph Semantic glyph to render.
/// \param policy Output policy controlling glyph family and ornaments.
/// \return Rendered glyph spelling.
[[nodiscard]] std::string render_glyph(semantic_glyph glyph, output_policy const& policy);

/// \brief Render raw UTF-8 text through charset and ASCII fallback policy.
/// \param text Input text.
/// \param policy Output policy controlling fallback behavior.
/// \return Rendered text with no ANSI styling.
[[nodiscard]] std::string render_text(std::string_view text, output_policy const& policy);

/// \brief Render styled text with color/style sequences only when policy allows.
/// \param text Styled text document.
/// \param policy Output policy controlling glyphs, charset, and color.
/// \return Rendered string.
[[nodiscard]] std::string render(styled_text const& text, output_policy const& policy);

/// \brief Render styled text for a terminal stream.
/// \param text Styled text document.
/// \param policy Output policy to apply.
/// \param stream Stream used for automatic color detection.
/// \return Rendered string.
[[nodiscard]] std::string render_terminal(styled_text const& text, output_policy policy, std::FILE* stream = stdout);

/// \brief Render styled text without ANSI escape sequences.
/// \param text Styled text document.
/// \param policy Output policy to apply apart from forced color suppression.
/// \return Rendered string.
[[nodiscard]] std::string render_plain(styled_text const& text, output_policy policy = plain_policy());

/// \brief Render styled text under strict ASCII fallback.
/// \param text Styled text document.
/// \param charset ASCII handling mode for non-ASCII text.
/// \param policy Base policy to apply apart from forced ASCII settings.
/// \return Rendered strict ASCII string.
[[nodiscard]] std::string render_strict_ascii(styled_text const& text,
                                              text_charset charset = text_charset::ascii_escape,
                                              output_policy policy = strict_ascii_policy());

/// \brief Format a high-level report builder into styled text.
/// \details The returned document may already be width-formatted according to
///          `policy.wrap_width`, with table cell wrapping and column alignment
///          resolved. Prefer the report-specific `render_terminal` and
///          `render_plain` overloads for final output; if this intermediate
///          document is rendered directly, do not apply a smaller final
///          `wrap_width`, because generic whole-string wrapping can split
///          completed table layouts.
/// \param report Report description to render.
/// \param policy Output policy controlling width, glyphs, and fallback.
/// \return Styled text document preserving report styles and semantic glyphs.
[[nodiscard]] styled_text render_report(report_builder const& report, output_policy const& policy);

/// \brief Render a high-level report builder for a terminal stream.
/// \param report Report description to render.
/// \param policy Output policy to apply.
/// \param stream Stream used for automatic color detection.
/// \return Rendered terminal string.
[[nodiscard]] std::string render_terminal(report_builder const& report, output_policy policy,
                                          std::FILE* stream = stdout);

/// \brief Render a high-level report builder without ANSI escape sequences.
/// \param report Report description to render.
/// \param policy Output policy to apply apart from forced color suppression.
/// \return Rendered plain string.
[[nodiscard]] std::string render_plain(report_builder const& report, output_policy policy = plain_policy());

/// \brief Measure rendered text width under the selected policy.
/// \param text Input text.
/// \param policy Output policy controlling charset and width mode.
/// \param initial_column Starting display column used for tab expansion.
/// \return Width in bytes or display cells after fallback.
[[nodiscard]] std::size_t display_width(std::string_view text, output_policy const& policy,
                                        std::size_t initial_column = 0);

/// \brief Measure rendered styled text width under the selected policy.
/// \param text Styled text document.
/// \param policy Output policy controlling glyphs, charset, and width mode.
/// \param initial_column Starting display column used for tab expansion.
/// \return Width in bytes or display cells after fallback.
[[nodiscard]] std::size_t display_width(styled_text const& text, output_policy const& policy,
                                        std::size_t initial_column = 0);

/// \brief Left-pad rendered text to a target width.
/// \param text Input text.
/// \param target_width Target width in the selected width mode.
/// \param policy Output policy controlling fallback and width mode.
/// \return Rendered text with leading spaces when needed.
[[nodiscard]] std::string pad_left(std::string_view text, std::size_t target_width, output_policy const& policy);

/// \brief Left-pad styled text to a target width.
/// \param text Styled text document.
/// \param target_width Target width in the selected width mode.
/// \param policy Output policy controlling fallback and width mode.
/// \return Styled text with leading spaces when needed.
[[nodiscard]] styled_text pad_left(styled_text const& text, std::size_t target_width, output_policy const& policy);

/// \brief Right-pad rendered text to a target width.
/// \param text Input text.
/// \param target_width Target width in the selected width mode.
/// \param policy Output policy controlling fallback and width mode.
/// \return Rendered text with trailing spaces when needed.
[[nodiscard]] std::string pad_right(std::string_view text, std::size_t target_width, output_policy const& policy);

/// \brief Right-pad styled text to a target width.
/// \param text Styled text document.
/// \param target_width Target width in the selected width mode.
/// \param policy Output policy controlling fallback and width mode.
/// \return Styled text with trailing spaces when needed.
[[nodiscard]] styled_text pad_right(styled_text const& text, std::size_t target_width, output_policy const& policy);

/// \brief Center rendered text within a target width.
/// \param text Input text.
/// \param target_width Target width in the selected width mode.
/// \param policy Output policy controlling fallback and width mode.
/// \return Rendered text padded on both sides when needed.
[[nodiscard]] std::string pad_center(std::string_view text, std::size_t target_width, output_policy const& policy);

/// \brief Center styled text within a target width.
/// \param text Styled text document.
/// \param target_width Target width in the selected width mode.
/// \param policy Output policy controlling fallback and width mode.
/// \return Styled text padded on both sides when needed.
[[nodiscard]] styled_text pad_center(styled_text const& text, std::size_t target_width, output_policy const& policy);

/// \brief Clip rendered text to a target width without appending a marker.
/// \param text Input text.
/// \param max_width Maximum width in the selected width mode.
/// \param policy Output policy controlling fallback and width mode.
/// \param initial_column Starting display column used for tab expansion.
/// \return Rendered text clipped to the requested width.
[[nodiscard]] std::string clip_to_width(std::string_view text, std::size_t max_width, output_policy const& policy,
                                        std::size_t initial_column = 0);

/// \brief Truncate rendered text to a target width and optionally append a marker.
/// \param text Input text.
/// \param max_width Maximum width in the selected width mode.
/// \param policy Output policy controlling fallback and width mode.
/// \param marker Marker appended when truncation occurs and space permits.
/// \param initial_column Starting display column used for tab expansion.
/// \return Rendered text truncated to the requested width.
[[nodiscard]] std::string truncate_to_width(std::string_view text, std::size_t max_width, output_policy const& policy,
                                            std::string_view marker = "", std::size_t initial_column = 0);

/// \brief Truncate rendered text from the left to a target width and optionally prepend a marker.
/// \param text Input text.
/// \param max_width Maximum width in the selected width mode.
/// \param policy Output policy controlling fallback and width mode.
/// \param marker Marker prepended when truncation occurs and space permits.
/// \param initial_column Starting display column used for tab expansion.
/// \return Rendered text truncated from the left to the requested width.
[[nodiscard]] std::string truncate_left_to_width(std::string_view text, std::size_t max_width,
                                                 output_policy const& policy, std::string_view marker = "",
                                                 std::size_t initial_column = 0);

/// \brief Prefix each rendered line with fixed text.
/// \param text Input text.
/// \param prefix Prefix inserted before each selected line.
/// \param policy Output policy controlling fallback and width mode.
/// \param prefix_first Whether to prefix the first line as well as following lines.
/// \return Rendered text with line prefixes inserted.
[[nodiscard]] std::string prefix_lines(std::string_view text, std::string_view prefix, output_policy const& policy,
                                       bool prefix_first = true);

/// \brief Indent each rendered line by a fixed number of spaces.
/// \param text Input text.
/// \param spaces Number of spaces to insert before each selected line.
/// \param policy Output policy controlling fallback and width mode.
/// \param indent_first Whether to indent the first line as well as following lines.
/// \return Rendered text with fixed indentation inserted.
[[nodiscard]] std::string indent_text(std::string_view text, std::size_t spaces, output_policy const& policy,
                                      bool indent_first = true);

/// \brief Wrap rendered text by display cells or bytes.
/// \param text Input text.
/// \param max_width Maximum width per line.
/// \param policy Output policy controlling fallback and width mode.
/// \return Wrapped rendered lines.
[[nodiscard]] std::vector<std::string> wrap_text(std::string_view text, std::size_t max_width,
                                                 output_policy const& policy);

/// \brief Wrap styled text by display cells or bytes.
/// \param text Styled text document.
/// \param max_width Maximum width per line.
/// \param policy Output policy controlling glyph, fallback, and width mode.
/// \return Wrapped styled lines.
[[nodiscard]] std::vector<styled_text> wrap_text(styled_text const& text, std::size_t max_width,
                                                 output_policy const& policy);

/// \brief Select the configured precision for a Uni20 real scalar type.
/// \tparam T Real scalar type.
/// \param options Numeric formatting controls.
/// \return Precision value associated with \p T.
template <uni20::Real T> [[nodiscard]] int real_precision(numeric_format_options const& options)
{
  if constexpr (std::is_same_v<std::remove_cv_t<T>, float>)
  {
    return options.float32_precision;
  }
  else if constexpr (std::is_same_v<std::remove_cv_t<T>, double>)
  {
    return options.float64_precision;
  }
#if UNI20_HAS_FLOAT128
  else if constexpr (std::is_same_v<std::remove_cv_t<T>, uni20::float128>)
  {
    return options.float128_precision;
  }
#endif
  else
  {
    return options.long_double_precision;
  }
}

[[nodiscard]] inline uni20::real_format_notation core_notation(real_notation notation)
{
  switch (notation)
  {
    case real_notation::fixed:
      return uni20::real_format_notation::fixed;
    case real_notation::scientific:
      return uni20::real_format_notation::scientific;
    case real_notation::general:
      return uni20::real_format_notation::general;
  }
  return uni20::real_format_notation::general;
}

/// \brief Format a real scalar value according to numeric presentation options.
/// \tparam T Real scalar type.
/// \param value Value to format.
/// \param options Numeric formatting controls.
/// \return Formatted numeric text.
template <uni20::Real T> [[nodiscard]] std::string format_real(T value, numeric_format_options const& options)
{
  return uni20::format_real(value,
                            uni20::scalar_format_options{.precision = real_precision<T>(options),
                                                         .notation = core_notation(options.notation),
                                                         .normalize_negative_zero = options.normalize_negative_zero,
                                                         .imaginary_unit = std::string_view(options.imaginary_unit)});
}

/// \brief Format a complex value as `real+imagi`.
/// \tparam T Real component type.
/// \param value Complex value to format.
/// \param options Numeric formatting controls.
/// \return Formatted complex numeric text.
template <uni20::Real T>
[[nodiscard]] std::string format_complex(uni20::complex<T> const& value, numeric_format_options const& options)
{
  return uni20::format_complex(
      value, uni20::scalar_format_options{.precision = real_precision<T>(options),
                                          .notation = core_notation(options.notation),
                                          .normalize_negative_zero = options.normalize_negative_zero,
                                          .imaginary_unit = std::string_view(options.imaginary_unit)});
}

/// \brief Format an arbitrary scalar-like value for presentation output.
/// \tparam T Value type.
/// \param value Value to format.
/// \param options Numeric formatting controls used for real and complex values.
/// \return Formatted scalar text.
template <typename T> [[nodiscard]] std::string format_scalar(T const& value, numeric_format_options const& options)
{
  using value_type = std::remove_cvref_t<T>;
  if constexpr (uni20::Real<value_type>)
  {
    return format_real(value, options);
  }
  else if constexpr (uni20::Complex<value_type>)
  {
    return format_complex(value, options);
  }
  else
  {
    return fmt::format("{}", value);
  }
}

namespace detail
{
[[nodiscard]] inline terminal::TerminalStyle nonfinite_scalar_style()
{
  return terminal::TerminalStyle(std::string_view("Red;Bold"));
}
} // namespace detail

/// \brief Apply the default non-finite scalar style to already-formatted presentation text.
/// \details Real and complex `nan`, `inf`, and `-inf` values are highlighted when the output policy permits
///          color. Other value types and color-disabled policies leave the text unchanged.
/// \tparam T Scalar value type.
/// \param text Already-formatted scalar text.
/// \param value Scalar value that produced \p text.
/// \param policy Output policy controlling whether ANSI style may be emitted.
/// \return Styled text when \p value is non-finite and color is enabled; otherwise \p text.
template <typename T>
[[nodiscard]] std::string style_nonfinite_scalar(std::string text, T const& value, output_policy const& policy)
{
  using value_type = std::remove_cvref_t<T>;
  if constexpr (uni20::RealOrComplex<value_type>)
  {
    if (!uni20::isfinite(value) && should_emit_color(policy) && !text.empty())
    {
      return terminal::color_text(text, detail::nonfinite_scalar_style());
    }
  }
  return text;
}

/// \brief Format a scalar for policy-aware presentation output.
/// \details This is equivalent to `format_scalar(value, options)` followed by `style_nonfinite_scalar(...)`.
/// \tparam T Value type.
/// \param value Value to format.
/// \param options Numeric formatting controls used for real and complex values.
/// \param policy Output policy controlling whether non-finite values are styled.
/// \return Formatted scalar text, with non-finite real or complex values highlighted when color is enabled.
template <typename T>
[[nodiscard]] std::string format_scalar_for_output(T const& value, numeric_format_options const& options,
                                                   output_policy const& policy)
{
  return style_nonfinite_scalar(format_scalar(value, options), value, policy);
}

} // namespace uni20::presentation
