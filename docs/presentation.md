# Presentation Formatting

Uni20 common presentation formatting lives in `uni20::presentation`:

```cpp
#include <uni20/common/presentation.hpp>
```

The layer is intentionally independent of trace, async, tensor, AD, and scheduler semantics. It renders styled text spans and semantic glyph tokens into terminal, plain, or strict ASCII output. Trace uses this layer for diagnostic line assembly, styled/color output, charset fallback, and display-cell container alignment, but presentation does not depend on trace.

## Output Policy

`output_policy` controls rendering:

| Field | Purpose |
|---|---|
| `color` | `never`, `automatic`, or `always` ANSI style emission. |
| `glyphs` | `unicode`, `emoji`, or `ascii` semantic glyph rendering. |
| `charset` | Preserve UTF-8, escape non-ASCII text, or replace it. |
| `width` | Measure by bytes or display cells. |
| `ambiguous` | Treat ambiguous-width code points as narrow or wide. |
| `invalid` | Escape or replace invalid UTF-8 bytes. |
| `tab_width` | Display-cell tab expansion. |
| `wrap_width` | Optional wrapping width. |
| `ornaments` | Disable, minimize, or enrich decorative glyphs. |

Automatic color follows terminal detection and honors `NO_COLOR`. Explicit `color_mode::always` still forces color.

## Semantic Glyphs

Call sites should use `semantic_glyph` tokens instead of hard-coding Unicode and ASCII spellings:

```cpp
namespace presentation = uni20::presentation;

presentation::styled_text text;
text.append(presentation::semantic_glyph::success)
    .append(" operation ")
    .append(presentation::semantic_glyph::arrow_right)
    .append(" complete");

auto policy = presentation::terminal_policy(stderr);
auto rendered = presentation::render(text, policy);
```

Central mappings cover status symbols, arrows, ellipsis, square and rounded box/table drawing, diagonal connector glyphs, and tree drawing. ASCII output uses these mappings automatically.

## Text Fallback

Strict ASCII modes perform a small deterministic fallback pass for common non-language symbols such as smart quotes, dashes, ellipsis, arrows, check/cross/warning symbols, simple math signs, and box drawing. Human-language text is not transliterated. For example, Chinese text is preserved under `text_charset::utf8`, escaped under `text_charset::ascii_escape`, and replaced under `text_charset::ascii_replace`.

## Width And Layout

Layout helpers operate after glyph and charset fallback:

```cpp
auto cells = presentation::display_width(text, policy);
auto padded = presentation::pad_right("label", 12, policy);
auto clipped = presentation::clip_to_width("value", 8, policy);
auto lines = presentation::wrap_text("long value", 10, policy);
auto indented = presentation::indent_text("line 1\nline 2", 4, policy);
```

Display-cell width is best effort and deterministic. It treats ASCII printable characters as width 1, style spans as width 0, combining marks as width 0, common CJK wide characters as width 2, emoji conservatively as width 2, and tabs relative to the current column.

`prefix_lines(...)` and `indent_text(...)` are deliberately simple block-layout helpers. They are intended for diagnostics, nested trace sections, and text-art blocks where callers want fixed indentation after wrapping or clipping has already chosen visible content.

Use `truncate_to_width(...)` when preserving the beginning of a long value, and `truncate_left_to_width(...)` when preserving the end. The latter is useful for diagnostics that need to show the tail of a long prefix before a highlighted token:

```cpp
auto prefix = presentation::truncate_left_to_width(before_error, 24, policy, "…");
auto suffix = presentation::truncate_to_width(error_and_after, 40, policy, "…");
```

See `examples/presentation_example.cpp` for semantic Unicode/emoji output, display-cell table alignment, fixed indentation after wrapping, tensor-network-style connector art, the default rounded tensor-box style, and a parser-style range diagnostic that adapts to different terminal widths.

## Mdspan And Tensor Art

Mdspan-like objects can be rendered through the presentation layer:

```cpp
#include <uni20/common/presentation_mdspan.hpp>

auto policy = uni20::presentation::terminal_policy(stdout);
auto text = uni20::presentation::format_mdspan(matrix, policy, [](auto const& value) {
  return fmt::format("{}", value);
});
```

Rank-1 values render as a row vector, rank-2 values render as aligned matrix art, and higher-rank values render as labeled rank-2 slices over every leading-axis coordinate. The default formatter is exhaustive: printing an actual tensor emits every element. Any future preview, clipping, or elision mode should be an explicit separate policy. Trace uses the same formatter for mdspan-like values and tensor/view-like objects, while still applying trace scalar formatting such as floating-point precision.

Real and complex tensor elements use `numeric_format_options` when no custom element formatter is supplied. The defaults use general notation with 6 significant digits for `float`, 15 significant digits for `double`, normalized negative zero, and algebraic complex form such as `1.25-3.5i`. `mdspan_format_options::numeric` can switch to fixed or scientific notation and adjust the digit counts.

For rank-2-or-higher tensors, `mdspan_format_options::matrix_axes` can choose which axes form the displayed row and column dimensions:

```cpp
uni20::presentation::mdspan_format_options options;
options.matrix_axes = uni20::presentation::mdspan_matrix_axes{0, 2};

auto text = uni20::presentation::format_mdspan(tensor, policy, options);
```

Any remaining axes become exhaustive slice labels. With shape `(2, 3, 2)` and matrix axes `{0, 2}`, labels are `slice [:, 0, :]`, `slice [:, 1, :]`, and `slice [:, 2, :]`.
