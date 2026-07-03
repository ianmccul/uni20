#Presentation Formatting

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
| `UNI20_CHARSET` | `utf8` | `utf8`, `escape`, `replace` | Preserve UTF-8, escape non-ASCII text, or replace it. `utf-8`, `ascii_escape`, and `ascii_replace` aliases are accepted. |
| `UNI20_COLOR` | `auto` | `auto`, `yes`, `always`, `no`, `never`, plus boolean aliases | Control ANSI style emission globally. |

Automatic color follows terminal detection and honors `NO_COLOR`. Explicit `color_mode::always`, including
`UNI20_COLOR=yes` or `UNI20_COLOR=always`, still forces color. Terminal width comes from
`terminal::columns()`, which uses `COLUMNS` when it is set to a positive integer.

Callers may deliberately override fields on a policy for a forced demonstration or a
known output destination. For ordinary terminal output, prefer `terminal_policy(...)`
as the starting point so user environment choices win over program defaults.

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

Diagnostic and display code should choose semantic severity or disposition, not literal symbols:

| Semantic glyph | Intended use | Emoji policy | ASCII policy |
|---|---|---|---|
| `success` | Completed, passed, or converged status. | check mark button | `[OK]` |
| `warning` | Non-fatal warning or advisory diagnostic. | warning sign | `[WARN]` |
| `info` | Neutral information or progress status. | information sign | `[INFO]` |
| `failure` | Recoverable error or exception-boundary diagnostic. | cross mark | `[FAIL]` |
| `fatal` | Abort-path diagnostics such as `PANIC`, `CHECK`, and `PRECONDITION`. | siren | `[FATAL]` |
| `partial` | Partly completed report status. | yellow circle | `[PARTIAL]` |
| `deferred` | Intentionally postponed report status. | pause button | `[DEFER]` |
| `skipped` | Not applicable, out of scope, or unavailable report status. | prohibited sign | `[SKIP]` |

Trace diagnostics use the severity split so warnings, recoverable errors, and
aborting assertions remain distinct while still rendering through the active
glyph and charset policy. Human-facing reports and display output can also use
the disposition glyphs when communicating progress or review state rather than
diagnostic severity.

## Styled Text

Use `presentation::style(...)` to define a reusable callable style. The helper
builds `styled_text`, not ANSI strings, so final rendering still respects
`UNI20_COLOR`, `NO_COLOR`, `UNI20_CHARSET`, and `UNI20_GLYPHS`:

```cpp
auto RedBold = presentation::style("Red;Bold");

auto message = RedBold("residual {:.3e}", residual);
auto value = RedBold(residual);
auto warning = RedBold(presentation::semantic_glyph::warning, "stagnated at {:.3e}", residual);
```

`terminal::TerminalStyle` remains the low -
        level style carrier.It can be constructed from style strings such as `"Red;Bold"` or `"fg:#ff0000;Bold"`,
    but ordinary presentation and display code should prefer `presentation::style(...)` so color is
        not rendered too early.

        Styled text can be inserted into table cells without pre
        - rendering it :

```cpp auto Warn = presentation::style("Yellow;Bold");

report.table("Solver")
    .column("quantity", presentation::table_alignment::left)
    .column("value", presentation::table_alignment::left)
    .row("residual", Warn(presentation::semantic_glyph::warning, "{:.3e}", residual));
```

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

See `examples/presentation/` for runnable demonstrations of semantic Unicode/emoji output, display-cell table alignment, fixed indentation after wrapping, tensor-network-style connector art, the default rounded tensor-box style, and parser-style range diagnostics that adapt to different terminal widths.

## Report Tables

`report_builder` and `report_table` provide a small higher-level API for command-line examples and diagnostics:

```cpp
presentation::report_builder report("Krylov solve");
report.status(presentation::semantic_glyph::success, "converged").field("dimension", 128).field("tolerance", "1.0e-12");

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

Rows can contain `table_cell` values when a cell should span multiple columns:

```cpp
report.table("Grouped Summary")
    .grid(presentation::table_rule_style::double_line)
    .column("stage", presentation::table_alignment::left)
    .column("note", presentation::table_alignment::left)
    .column("error", presentation::table_alignment::decimal)
    .row("setup", "read inputs", 4)
    .separator()
    .row({{"solve", 1}, {"Krylov iterations and restart details", 2, presentation::table_alignment::left}});
```

`presentation::cell(...)` builds the same `table_cell` explicitly and can carry styled content, spans,
    and per - cell alignment :

```cpp auto Note = presentation::style("Cyan;Bold");

report.table("Notes")
    .grid()
    .column("label", presentation::table_alignment::left)
    .column("detail", presentation::table_alignment::left)
    .row({presentation::cell("solver"),
          presentation::cell(Note("styled note"), 1, presentation::table_alignment::left)});
```

`separator()` inserts an explicit body rule without enabling global row separators;
`top_separator()` inserts an explicit rule before the generated heading row
    .Automatic borders and separators use `table_border_options::rule_style`,
    set directly with `border_style(...)` or by calling `grid(presentation::table_rule_style::double_line)`
                                                 .

`table_alignment::decimal` aligns finite values on `.`; values without a decimal point align as if the point followed the
rendered value. Non-finite spellings such as `nan`, `inf`, and `-inf` are centered in a decimal-aligned cell. A
`table_cell` can also override alignment for one cell, including non-spanning cells.

When `output_policy::wrap_width` is set, report tables use it as a table-width budget. The renderer first keeps natural
column widths if the table fits, then prefers shrinking columns that can still wrap at whitespace before forcing a hard
split inside an unbreakable token. Report-specific renderers disable final whole-string wrapping after the table has been
formatted, because wrapping a completed table would split borders and column alignment.

`render_report(report, policy)` exposes the styled intermediate representation used by the report renderers. It preserves
semantic glyphs and span styles, but the report layout may already have consumed `policy.wrap_width`. Prefer
`render_plain(report, policy)` or `render_terminal(report, policy)` for final output. If the intermediate styled document
is rendered directly, render it without a smaller final `wrap_width`; a wider output area is harmless, but generic
whole-string wrapping can break a completed table.

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
auto text =
    uni20::presentation::format_mdspan(matrix, policy, [](auto const& value) { return fmt::format("{}", value); });
```

Rank-1 values render as a row vector, rank-2 values render as aligned matrix art, and higher-rank values render as labeled rank-2 slices over every leading-axis coordinate. `format_mdspan(...)` is intentionally exhaustive: printing an actual tensor emits every element.

Use `format_mdspan_preview(...)` for bounded diagnostic output:

```cpp
uni20::presentation::mdspan_preview_options preview;
preview.full_element_limit = 256;
preview.edge_items = 3;
preview.max_slices = 4;

auto result = uni20::presentation::format_mdspan_preview(tensor, policy, preview);
```

    The preview renderer first uses exhaustive output when the element count is small enough and the result
        fits `output_policy::wrap_width`.Otherwise it displays edge rows /
    columns / slices with the active semantic ellipsis glyph and records `mdspan_preview_result::elided =
    true`.Metadata lines are also width - aware;
if the terminal is too narrow for even a one-cell preview, the renderer falls back to a shape/element-count summary plus an elision note.

Trace uses bounded preview by default for mdspan-like values and tensor/view-like objects, while still applying trace scalar formatting such as floating-point precision. Python and Jupyter tensor display should also be preview-first rather than exhaustive by default, with explicit controls for selected axes, preview limits, and `full=true` opt-in behavior.

Real and complex tensor elements use `numeric_format_options` when no custom element formatter is supplied. The defaults use general notation with 6 significant digits for `float`, 15 significant digits for `double`, normalized negative zero, and algebraic complex form such as `1.25-3.5i`. Non-finite real values render deterministically as `nan`, `inf`, and `-inf`; complex values use the same component spelling, for example `-inf+nani`. When color is enabled, mdspan/tensor rendering highlights non-finite real or complex elements with the default red-bold style while preserving display-cell alignment. `mdspan_format_options::numeric` can switch to fixed or scientific notation and adjust the digit counts.

For rank-2-or-higher tensors, `mdspan_format_options::matrix_axes` can choose which axes form the displayed row and column dimensions:

```cpp
uni20::presentation::mdspan_format_options options;
options.matrix_axes = uni20::presentation::mdspan_matrix_axes{0, 2};

auto text = uni20::presentation::format_mdspan(tensor, policy, options);
```

    Any remaining axes become exhaustive slice labels.With shape `(2, 3, 2)` and matrix axes `
{
  0, 2
}
`, labels are `slice[:, 0, :]`, `slice[:, 1, :]`, and `slice[:, 2, :]`.
