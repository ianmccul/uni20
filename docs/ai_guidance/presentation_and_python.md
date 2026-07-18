# Presentation and Python Display: AI Guidance

- **Audience:** remote assistants, coding agents, and reviewers
- **Authority:** non-normative retrieval summary
- **Reviewed against:** `Uni20-dev/uni20` `main`, 2026-07-18
- **Canonical sources:** `docs/diagnostics/presentation.md`,
  `docs/python/bindings.md`, presentation source, Python bindings, and tests

## Implemented presentation layer

- `uni20::presentation` is independent of trace, async, Tensor, AD, and scheduler semantics.
- `styled_text` and semantic glyphs separate formatting intent from final rendering.
- Terminal, plain, and strict-ASCII renderers exist.
- Width-aware wrapping, clipping, indentation, report tables, semantic borders,
  source/stacktrace formatting where enabled, and structured diagnostics exist.
- Trace uses the shared presentation layer.
- Environment defaults include `UNI20_GLYPHS`, `UNI20_CHARSET`, `UNI20_COLOR`,
  `NO_COLOR`, and `COLUMNS`.
- Prefer semantic glyph tokens over hard-coded Unicode.

## Mdspan/Tensor display

Two distinct paths are implemented:

- `format_mdspan(...)` is exhaustive.
- `format_mdspan_preview(...)` is bounded and reports whether data was elided.

The preview path can limit full element count, edge items, and displayed slices;
it is width-aware and can fall back to shape/element-count metadata. Trace uses
bounded preview by default for tensor/mdspan-like values.

Do not state that preview/clipping is unimplemented. Do not use exhaustive
formatting as a safe default for large user-facing objects.

## Current Python boundary

- The `uni20` extension is currently a nanobind smoke module.
- It exposes `greet()` and generated build metadata.
- It does not expose Tensor values, numerical operations, NumPy/DLPack interop,
  native async values, packaging, or notebook-rich Tensor display.
- Import configures recoverable Uni20 error paths to throw exceptions process-wide;
  invariant failures remain aborting failures.
- Python-visible native views must retain the owner needed for data lifetime.
- Python-owned objects retained by async work need interpreter-safe destruction.

## Future Python display rules

- `repr(obj)` must be stable, plain, deterministic, and safe for large values.
- `str(obj)` may be human-facing but should remain preview-first.
- Rich notebook display belongs in `_repr_html_()` or `_repr_mimebundle_()`.
- HTML rendering must not contain ANSI escapes or assume terminal display-cell layout.
- Per-call Python options should override environment defaults.
- A `pretty(...)` API may expose width, glyphs, charset, color, precision,
  selected axes, preview limits, and explicit full output.
- Display elision must be visible and deterministic.
- Rich display is an adapter over presentation data, not a dependency of core compute.

## Renderer boundary

- Producers should emit semantic spans or structured formatting intent.
- Terminal ANSI, plain text, strict ASCII, and future HTML are separate adapters.
- HTML must escape user text and use controlled CSS/classes.
- Terminal width calculations are renderer-specific and must not become HTML invariants.

## Push-back triggers

- Claiming Python Tensor bindings or notebook rendering exist.
- Binding Python `repr` to exhaustive `format_mdspan(...)`.
- Calling preview output a numerical approximation.
- Hiding elision.
- Reusing ANSI output inside HTML.
- Making the presentation layer responsible for Tensor computation semantics.
