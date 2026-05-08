#pragma once

#include "terminal.hpp"

#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdio>
#include <fmt/core.h>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
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

/// \brief Numeric scalar formatting controls shared by tensor-style renderers.
struct numeric_format_options
{
    int float32_precision = 6;
    int float64_precision = 15;
    int long_double_precision = 18;
    real_notation notation = real_notation::general;
    bool normalize_negative_zero = true;
    std::string imaginary_unit = "i";
};

/// \brief Names semantic glyphs independently of their rendered spelling.
enum class semantic_glyph
{
  success,
  failure,
  warning,
  info,
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
  box_tee_left,
  box_tee_right,
  box_tee_up,
  box_tee_down,
  box_cross,
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
    glyph_set glyphs = glyph_set::unicode;
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

/// \brief Right-pad rendered text to a target width.
/// \param text Input text.
/// \param target_width Target width in the selected width mode.
/// \param policy Output policy controlling fallback and width mode.
/// \return Rendered text with trailing spaces when needed.
[[nodiscard]] std::string pad_right(std::string_view text, std::size_t target_width, output_policy const& policy);

/// \brief Center rendered text within a target width.
/// \param text Input text.
/// \param target_width Target width in the selected width mode.
/// \param policy Output policy controlling fallback and width mode.
/// \return Rendered text padded on both sides when needed.
[[nodiscard]] std::string pad_center(std::string_view text, std::size_t target_width, output_policy const& policy);

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

/// \brief Wrap rendered text by display cells or bytes.
/// \param text Input text.
/// \param max_width Maximum width per line.
/// \param policy Output policy controlling fallback and width mode.
/// \return Wrapped rendered lines.
[[nodiscard]] std::vector<std::string> wrap_text(std::string_view text, std::size_t max_width,
                                                 output_policy const& policy);

/// \brief Select the configured precision for a floating-point type.
/// \tparam T Floating-point type.
/// \param options Numeric formatting controls.
/// \return Precision value associated with \p T.
template <std::floating_point T> [[nodiscard]] int real_precision(numeric_format_options const& options)
{
  if constexpr (std::is_same_v<std::remove_cv_t<T>, float>)
  {
    return options.float32_precision;
  }
  else if constexpr (std::is_same_v<std::remove_cv_t<T>, double>)
  {
    return options.float64_precision;
  }
  else
  {
    return options.long_double_precision;
  }
}

/// \brief Format a real floating-point value according to numeric presentation options.
/// \tparam T Floating-point type.
/// \param value Value to format.
/// \param options Numeric formatting controls.
/// \return Formatted numeric text.
template <std::floating_point T> [[nodiscard]] std::string format_real(T value, numeric_format_options const& options)
{
  if (options.normalize_negative_zero && value == T{})
  {
    value = T{};
  }

  int const precision = real_precision<T>(options);
  switch (options.notation)
  {
    case real_notation::fixed:
      return fmt::format("{:.{}f}", value, precision);
    case real_notation::scientific:
      return fmt::format("{:.{}e}", value, precision);
    case real_notation::general:
      return fmt::format("{:.{}g}", value, precision);
  }
  return fmt::format("{}", value);
}

/// \brief Format a complex value as `real+imagi`.
/// \tparam T Floating-point component type.
/// \param value Complex value to format.
/// \param options Numeric formatting controls.
/// \return Formatted complex numeric text.
template <std::floating_point T>
[[nodiscard]] std::string format_complex(std::complex<T> const& value, numeric_format_options const& options)
{
  T imag = value.imag();
  bool const negative_imag = std::signbit(imag) && !(options.normalize_negative_zero && imag == T{});
  if (negative_imag)
  {
    imag = -imag;
  }
  else if (options.normalize_negative_zero && imag == T{})
  {
    imag = T{};
  }

  return format_real(value.real(), options) + (negative_imag ? "-" : "+") + format_real(imag, options) +
         options.imaginary_unit;
}

/// \brief Format an arbitrary scalar-like value for presentation output.
/// \tparam T Value type.
/// \param value Value to format.
/// \param options Numeric formatting controls used for real and complex values.
/// \return Formatted scalar text.
template <typename T> [[nodiscard]] std::string format_scalar(T const& value, numeric_format_options const& options)
{
  using value_type = std::remove_cvref_t<T>;
  if constexpr (std::floating_point<value_type>)
  {
    return format_real(value, options);
  }
  else if constexpr (requires { typename value_type::value_type; } &&
                     requires(value_type const& z) {
                       { z.real() } -> std::floating_point;
                       { z.imag() } -> std::floating_point;
                     })
  {
    return format_complex(value, options);
  }
  else
  {
    return fmt::format("{}", value);
  }
}

} // namespace uni20::presentation
