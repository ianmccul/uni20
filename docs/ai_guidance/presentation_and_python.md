# Uni20 Presentation And Python Display: AI Guidance

This file is for questions about presentation formatting, Python display, Jupyter notebooks, and tensor repr design.

## File-level answer rule

- Separate implemented presentation behavior from roadmap Python/Jupyter behavior.
- Do not claim tensor Python display exists unless specific binding code has been inspected.
- Treat display formatting as an adapter layer, not as part of Uni20 compute semantics.
- Prefer safe preview output over exhaustive tensor output in Python and notebooks.

## presentation layer

### STATUS

- `uni20::presentation` exists in C++.
- `styled_text` stores semantic text spans and semantic glyph tokens.
- Terminal, plain, strict ASCII, and layout helpers exist.
- Trace output uses the presentation layer.
- Mdspan-like formatting exists and is currently exhaustive by default.

### SAFE CLAIMS

- Presentation is intentionally independent of trace, async, tensor, AD, and scheduler semantics.
- Presentation policy controls glyphs, charset fallback, color, width mode, ambiguous width, invalid UTF-8 handling, tab width, wrapping, and ornament mode.
- Semantic glyphs should be preferred over hard-coded Unicode spellings.
- General presentation environment defaults include `UNI20_GLYPHS`, `UNI20_CHARSET`, `UNI20_COLOR`, `NO_COLOR`, and `COLUMNS`.
- Trace uses the shared presentation policy for diagnostic line assembly, semantic glyphs, charset fallback, color, and display-cell alignment.

### DO NOT CLAIM

- Do not claim that presentation is a terminal-only layer.
- Do not claim that HTML or Jupyter rendering is implemented unless binding/rendering code has been inspected.
- Do not claim that tensor pretty-printing is safe for large tensors by default.
- Do not bind Python `repr` to exhaustive mdspan formatting by default.

## Python display

### STATUS

- Python bindings currently expose only a subset of Uni20 through `nanobind`.
- Current Python binding functionality includes smoke-test helpers and build metadata.
- Python tensor bindings and notebook-rich tensor display are roadmap material unless code proves otherwise.

### DESIGN RULES

- `repr(obj)` must be stable, plain, and safe for large objects.
- `str(obj)` and `print(obj)` may be human-facing, but should still avoid accidental exhaustive tensor output.
- `_repr_html_()` and `_repr_mimebundle_()` are the appropriate hooks for rich Jupyter display.
- `obj.pretty(...)` should expose explicit controls for width, glyphs, charset, color, selected axes, precision, and preview limits.
- Python per-call display options should override environment defaults.
- Python automatic color/terminal policy should account for Python streams such as `sys.stdout.isatty()` and notebook display contexts, not only C `FILE*` handles.

### FAILURE MODES

- Rendering an entire large tensor in `repr(obj)` can freeze or crash a notebook.
- Using ANSI escape sequences in `_repr_html_()` produces poor notebook output.
- Using terminal display-cell width as an HTML layout invariant is incorrect.
- Letting environment variables be the only control makes notebook output hard to reproduce.

## tensor preview policy

### STATUS

- Current mdspan tensor-art formatting is exhaustive.
- Preview, clipping, and elision are future policy work.

### REQUIRED BEFORE PYTHON TENSOR REPR

- Add an explicit tensor/mdspan preview policy before exposing tensor `repr` or rich notebook tensor display.
- Include limits such as maximum elements, edge items, maximum rows, maximum columns, maximum slices, selected matrix axes, and an explicit full-output opt-in.
- Make preview behavior deterministic and testable.
- Preserve an explicit path to exhaustive output for users who request it.

### SAFE CLAIMS

- Preview-first display supports teaching and notebooks without changing Uni20 compute goals.
- Preview policy belongs in presentation/bindings design, not in core tensor arithmetic semantics.

### DO NOT CLAIM

- Do not describe preview output as a numerical approximation.
- Do not hide the fact that displayed tensor data may be elided.
- Do not make rich notebook display a dependency of the C++ core runtime.

## renderer boundary

### INVARIANTS

- Presentation producers should emit semantic spans or structured formatting intent where possible.
- Terminal ANSI rendering, plain text rendering, strict ASCII rendering, and future HTML rendering should be separate adapters.
- Terminal width calculations are backend-specific and should not be required for HTML rendering.
- HTML/Jupyter rendering should escape user text and apply styles through controlled CSS/classes.

### RELATED

- `../presentation.md`
- `../Python.md`
- `../trace_macros.md`
- `architecture_status.md`
