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
| `glyphs` | `emoji` by default; select `unicode` or `ascii` for semantic glyph rendering when needed. |
| `charset` | Preserve UTF-8, escape non-ASCII text, or replace it. |
| `width` | Measure by bytes or display cells. |
| `ambiguous` | Treat ambiguous-width code points as narrow or wide. |
| `invalid` | Escape or replace invalid UTF-8 bytes. |
| `tab_width` | Display-cell tab expansion. |
| `wrap_width` | Optional wrapping width. |
| `ornaments` | Disable, minimize, or enrich decorative glyphs. |

`terminal_policy()` reads the global presentation environment:

| Variable | Default | Values | Effect |
|---|---|---|---|
| `UNI20_GLYPHS` | `emoji` | `emoji`, `unicode`, `ascii` | Select semantic glyph spelling. |
| `UNI20_CHARSET` | `utf8` | `utf8`, `ascii_escape`, `ascii_replace` | Preserve UTF-8, escape non-ASCII text, or replace it. Hyphen aliases are accepted. |
| `UNI20_COLOR` | `auto` | `auto`, `yes`, `always`, `no`, `never`, plus boolean aliases | Control ANSI style emission globally. |

Automatic color follows terminal detection and honors `NO_COLOR`. Explicit `color_mode::always`, including
`UNI20_COLOR=yes` or `UNI20_COLOR=always`, still forces color. Terminal width comes from
`terminal::columns()`, which uses `COLUMNS` when it is set to a positive integer.

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

The default terminal and plain policies prefer emoji for semantic status glyphs. Use `glyph_set::unicode` for symbol-only output, or `glyph_set::ascii` when fixed-width terminal behavior matters more than rich status symbols. Central mappings cover status symbols, arrows, ellipsis, square and rounded box/table drawing, diagonal connector glyphs, and tree drawing. ASCII output uses these mappings automatically.

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

## Report Tables

`report_builder` and `report_table` provide a small higher-level API for command-line examples and diagnostics:

```cpp
presentation::report_builder report("Krylov solve");
report.status(presentation::semantic_glyph::success, "converged")
    .field("dimension", 128)
    .field("tolerance", "1.0e-12");

report.table("Solver Summary")
    .grid()
    .column("solver", presentation::table_alignment::left)
    .column("matvecs")
    .column("residual")
    .row("native", 185, "1.0e-15")
    .row("arpack", 238, "8.0e-13");

auto rendered = presentation::render_plain(report, presentation::plain_policy());
```

Tables are compact by default, preserving the existing two-space column layout. Use `outer_border()`,
`column_separators()`, `row_separators()`, and `header_separator()` for individual rules, or `grid()` for a full
outer border with column, header, and row rules. Borders use semantic box glyphs, so Unicode, emoji, and strict ASCII
policies all render through the same fallback path.

When `output_policy::wrap_width` is set, report tables use it as a table-width budget. The renderer first keeps natural
column widths if the table fits, then prefers shrinking columns that can still wrap at whitespace before forcing a hard
split inside an unbreakable token. Report-specific renderers disable final whole-string wrapping after the table has been
formatted, because wrapping a completed table would split borders and column alignment.

## Python And Notebook Display

The presentation layer is intended to be the shared formatting backend for future Python bindings and Jupyter display, but Python should not expose the C++ terminal model directly. Treat the current renderers as separate adapters over the same semantic presentation data:

- terminal text: ANSI color, semantic glyphs, and display-cell width handling;
- plain text: stable output for `repr(...)`, logs, tests, and redirected streams;
- strict ASCII: deterministic fallback for non-Unicode environments;
- future HTML/Jupyter: rich display from the same semantic spans, without ANSI escapes or terminal-width assumptions.

Python bindings should keep `repr(obj)` conservative: plain, stable, and safe to call on large objects. Rich notebook output should use `_repr_html_()` or `_repr_mimebundle_()` and may use color, CSS, and richer layout. User-facing helpers such as `obj.pretty(...)` can expose explicit per-call controls for glyphs, charset, color, width, selected axes, and preview limits.

Do not let notebook display depend on terminal-only facts. A Python policy should derive automatic color and stream behavior from Python streams such as `sys.stdout.isatty()` or notebook display detection, not only from C `FILE*` handles. Environment variables such as `UNI20_GLYPHS`, `UNI20_CHARSET`, `UNI20_COLOR`, `NO_COLOR`, and `COLUMNS` are useful defaults, but Python APIs should allow explicit overrides.

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

Python and Jupyter tensor display must be preview-first rather than exhaustive by default. Before binding tensor `repr`, add a preview policy with explicit limits such as maximum elements, edge items, maximum rows/columns, maximum slices, selected matrix axes, and `full=true` opt-in behavior. This protects notebooks from accidentally rendering very large tensors while preserving an explicit path to exhaustive output when the user requests it.

Real and complex tensor elements use `numeric_format_options` when no custom element formatter is supplied. The defaults use general notation with 6 significant digits for `float`, 15 significant digits for `double`, normalized negative zero, and algebraic complex form such as `1.25-3.5i`. `mdspan_format_options::numeric` can switch to fixed or scientific notation and adjust the digit counts.

For rank-2-or-higher tensors, `mdspan_format_options::matrix_axes` can choose which axes form the displayed row and column dimensions:

```cpp
uni20::presentation::mdspan_format_options options;
options.matrix_axes = uni20::presentation::mdspan_matrix_axes{0, 2};

auto text = uni20::presentation::format_mdspan(tensor, policy, options);
```

Any remaining axes become exhaustive slice labels. With shape `(2, 3, 2)` and matrix axes `{0, 2}`, labels are `slice [:, 0, :]`, `slice [:, 1, :]`, and `slice [:, 2, :]`.
