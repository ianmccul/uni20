# Screen Display Layer Plan

**Status:** design and implementation note. The first C++ slice now provides
`uni20::display` status helpers, sink routing, report emission, and schema-first
streaming tables. Python sinks, async queue sinks, display contexts, and richer
streaming-table controls remain planned work.

## Purpose

Uni20 already has a presentation layer that can render styled spans, semantic
glyphs, reports, tables, and tensor previews. That layer is intentionally low
level: it answers "how should this semantic presentation object render under
this output policy?"

The display layer should answer a different question: "how can ordinary Uni20
code emit useful human-facing output without caring whether the caller is a C++
terminal executable, a redirected batch job, a Python script, pytest, or a
Jupyter notebook?"

The intended namespace is:

```cpp
namespace uni20::display;
```

The API should be easier than ad hoc `std::cout` and no harder than
`fmt::print`, while giving callers semantic glyphs, colors, width-aware
formatting, tables, and Python-friendly routing.

## Design Center

The design center is ephemeral user display:

- progress messages while a calculation runs,
- setup summaries at the start of a run,
- DMRG sweep tables,
- Krylov convergence summaries,
- warnings that a user should see while watching the terminal,
- compact reports printed before or after an algorithm.

This is not primarily an archival logging layer. Output may be redirected to a
file, especially on clusters, but the content is still aimed at a human reading
the run transcript. A durable structured logger can be a later sink or a
separate layer.

## Non-Goals

- Do not replace `trace.hpp` correctness-boundary macros such as `CHECK`,
  `PRECONDITION`, `PANIC`, or debug traces with source-location-heavy output.
- Do not turn ordinary progress output into a heavyweight logging framework.
- Do not make Python display depend on C `stdout` or `stderr`.
- Do not require callers to manually choose literal Unicode, emoji, ANSI escape
  codes, or ASCII fallbacks.
- Do not make `repr(obj)` in Python exhaustive or terminal-dependent.

## Layer Boundaries

The design has three explicit layers.

### Semantic Content

Callers create or emit semantic content:

- plain formatted messages,
- `presentation::styled_text`,
- `presentation::semantic_glyph` status markers,
- `presentation::report_builder` reports,
- streaming table rows,
- bounded tensor or mdspan previews.

This layer must not write to `FILE*`, `std::ostream`, or Python objects.

### Rendering

Rendering is owned by `uni20::presentation`.

The display layer should not duplicate glyph fallback, ANSI style generation,
display-cell width calculations, text wrapping, table drawing, or tensor art.
It should build presentation objects and ask presentation renderers to render
them with an appropriate `presentation::output_policy`.

### Sinks

The active sink decides where rendered output goes:

- C++ terminal sink: writes to `stdout` or `stderr`;
- Python sink: writes through `sys.stdout.write` or `sys.stderr.write`;
- vector/string sink: collects output for tests;
- null sink: discards output;
- file sink: writes human-facing transcript output to a file;
- queue sink: receives events from async tasks and drains later;
- composite sink: fans out to several sinks.

Sinks are routing and policy adapters. They should not invent their own glyph
or table formatting rules.

## Basic API Shape

The common path should look like `fmt::print` with a semantic prefix:

```cpp
display::info("loaded {} symmetry sectors", sector_count);
display::success("converged in {} sweeps", sweep_count);
display::warning("bond dimension hit cap {}", max_bond);
display::failure("factorization failed: {}", reason);
display::partial("validated {} of {} checks", passed, total);
display::deferred("complex Schur pair policy left for API cleanup");
display::skipped("CUDA backend unavailable");
```

The same concept should also support explicit semantic glyphs:

```cpp
display::status(presentation::semantic_glyph::warning, "residual stagnated at {:.3e}", residual);
```

A minimal first API could be:

```cpp
namespace uni20::display
{
  enum class stream
  {
    out,
    err
  };

  template <typename... Args> void info(format_string<Args...> format, Args&&... args);

  template <typename... Args> void success(format_string<Args...> format, Args&&... args);

  template <typename... Args> void warning(format_string<Args...> format, Args&&... args);

  template <typename... Args> void failure(format_string<Args...> format, Args&&... args);

  template <typename... Args> void partial(format_string<Args...> format, Args&&... args);

  template <typename... Args> void deferred(format_string<Args...> format, Args&&... args);

  template <typename... Args> void skipped(format_string<Args...> format, Args&&... args);

  template <typename... Args>
  void status(presentation::semantic_glyph glyph, format_string<Args...> format, Args&&... args);

  template <typename... Args>
  presentation::styled_text status_cell(presentation::semantic_glyph glyph, format_string<Args...> format,
                                        Args&&... args);

  void emit(presentation::styled_text const& text, stream destination = stream::err);
  void emit(presentation::report_builder const& report, stream destination = stream::err);
}
```

The exact names can change, but the ergonomics should remain: one line for
ordinary messages, and direct emission for richer presentation objects.

Helpers used inside reports and tables should not emit. For example,
`display::status_cell(...)` or a similar name can return a
`presentation::styled_text` value that combines a semantic glyph with text. This
keeps `display::success(...)` as an action and avoids surprising output while a
row is being assembled.

## Semantic Glyphs

The display layer should use `presentation::semantic_glyph`, not literal
characters. The presentation layer currently has status and disposition glyphs
for:

| Semantic meaning | Glyph name | Typical emoji rendering | ASCII rendering |
|---|---|---|---|
| completed / passed | `success` | `✅` | `[OK]` |
| failed / open problem | `failure` | `❌` | `[FAIL]` |
| aborting / fatal | `fatal` | `🚨` | `[FATAL]` |
| warning / advisory | `warning` | `⚠️` | `[WARN]` |
| neutral information | `info` | `ℹ️` | `[INFO]` |
| partly done / incomplete | `partial` | `🟡` | `[PARTIAL]` |
| intentionally postponed | `deferred` | `⏸️` | `[DEFER]` |
| not applicable / out of scope / not run | `skipped` | `🚫` | `[SKIP]` |

These are not trace severities. In particular, `partial` is not the same as
`warning`, and `skipped` is not the same as `failure`. They are useful when a
report or progress summary communicates disposition rather than error severity.

## Reports And Batch Tables

When all rows are known, display should reuse the existing batch report and
table machinery:

```cpp
presentation::report_builder report("DMRG setup");
report.status(presentation::semantic_glyph::info, "initializing")
    .field("sites", sites)
    .field("target sweeps", sweeps)
    .field("max bond", max_bond)
    .field("backend", backend_name);

report.table("Hamiltonian terms")
    .grid()
    .column("term", presentation::table_alignment::left)
    .column("coefficient", presentation::table_alignment::decimal)
    .column("sites", presentation::table_alignment::left)
    .row("Sz Sz", "1.0", "nearest neighbor")
    .row("S+ S-", "0.5", "nearest neighbor")
    .row("S- S+", "0.5", "nearest neighbor");

display::emit(report);
```

Batch tables may inspect all rows, choose widths from the complete content, and
fit the table to the available width. This is appropriate for setup summaries,
final reports, and compact diagnostics.

## Streaming Tables

Per-step output often does not know future rows. For example, DMRG sweeps,
iterative eigensolvers, benchmark progress, and async task progress are
streaming. A streaming table should be schema-first rather than data-first:

```cpp
auto sweeps = display::table("DMRG sweeps")
  .column("sweep", display::width::fixed(5))
  .column("energy", display::format::fixed(12))
  .column("delta", display::format::scientific(3))
  .column("bond", display::width::fixed(6))
  .column("status", display::width::share(4));

sweeps.row(sweep, energy, delta, bond, "accepted");
sweeps.row(sweep, energy, delta, bond, fmt::format("bond cap reached; truncation error {:.3e}", trunc));
```

The table object owns a fixed schema:

- column names,
- alignment,
- minimum width,
- optional fixed width,
- optional maximum width,
- share weight for flexible columns,
- wrapping policy,
- optional per-cell formatter.

### Streaming Width Rules

The streaming table should choose initial widths from the schema, headers, and
the sink's current display width. It should not wait for future rows.

Recommended behavior:

1. Compute fixed column widths first.
2. Reserve rule/separator/padding width.
3. Allocate remaining width among flexible columns by share weight.
4. Respect each column's minimum width.
5. If all minimum widths do not fit, switch to a vertical fallback rather than
   dropping data.
6. Keep the chosen widths stable for later rows.

Long row values should wrap inside that row only. They should not permanently
expand the column, and they should not force future rows to wrap merely to
preserve alignment with an outlier row.

This means one long status message may produce:

```text
sweep  energy             delta       bond  status
1      -12.345678901234   -           128   accepted
2      -12.456789012345   -1.111e-1   256   bond cap reached;
                                             -> truncation error 3.2e-7
3      -12.467000000000   -1.021e-2   256   accepted
```

The table schema remains stable after row 2. Row 3 does not inherit row 2's
wrapping. Continuation lines should carry a visible marker, rendered from a
semantic arrow glyph, inside the cell that wrapped so progress output is easy
to scan without changing the chosen column widths or relying on a blank leading
column.

Streaming tables should not rely on terminal auto-wrap. The formatter should
emit physical lines that fit the selected terminal width; if the configured
schema cannot fit, it should switch to a vertical key/value fallback instead of
letting a wide row spill into the next terminal line.

Streaming columns can declare default value formatting as part of the schema.
For example, a solver progress table can use fixed-point formatting for an
energy column and scientific formatting for a residual column. Typed numeric
values use the column formatter and participate in decimal alignment. If a
numeric format such as `display::format::fixed(...)`,
`display::format::scientific(...)`, or `display::format::general(...)` is
supplied without an explicit alignment, the column defaults to decimal
alignment. If the width is omitted for a numeric format, the column uses a
non-greedy fit width: it starts from a format-specific minimum, does not absorb
terminal slack, and can grow for later rows by using unused space or by
shrinking shared columns toward their minimum widths. If the width is omitted
for text or automatic columns, the column uses the ordinary shared-width
terminal allocation. Use explicit fixed widths when a field should never change
width; reserve shared-width columns for prose-like fields that should absorb
terminal slack. Text sent to a numeric column, for example `"non-converged"`, is
treated as exceptional text: it is displayed in the column but does not update
the decimal anchor.

Decimal-aligned numeric streaming columns keep a per-column decimal anchor based
on numeric rows seen so far. This aligns later shorter values with earlier
values in the common progress-output case. If a later row has more display cells
before the decimal than any previous row, future rows can use the wider anchor,
but already-emitted rows are not reflowed.

The current streaming-table API accepts values that can be converted to strings
through `fmt::format`, plus `presentation::styled_text` and
`presentation::table_cell` values. Styled cells keep semantic glyphs and style
metadata until the active sink renders them, so ASCII, Unicode, emoji, color,
and charset policy still apply at the output boundary.

### No Data Loss By Default

Streaming display should not drop cell content by default. If a cell is too
wide, it wraps. If a token is wider than the cell and has no whitespace, it is
hard-split. Clipping and ellipsis should be opt-in for progress displays where
the caller explicitly prefers compactness over full content.

For very narrow terminals, a vertical fallback is preferable:

```text
sweep: 2
energy: -12.456789012345
delta: -1.111e-1
bond: 256
status: bond cap reached;
                                             truncation error 3.2e-7
```

This is less pretty, but it preserves information and avoids broken table
rules.

### Terminal Resizing

The first implementation can choose widths when the streaming table is created.
Later, the table may support an explicit `resize()` or an automatic policy:

- `stable`: never recompute widths;
- `on_terminal_change`: recompute at row boundaries when the sink width changes;
- `manual`: recompute only when the caller asks.

The default should be stable. A progress table that changes width while the
user watches it can be distracting.

## Redirected Output

Display output may be redirected to a file, especially on clusters. The default
policy should follow current presentation behavior:

- automatic color emits ANSI only when the destination appears terminal-like;
- `NO_COLOR` disables automatic color;
- `UNI20_COLOR=always` or an explicit policy preserves ANSI escapes;
- `UNI20_GLYPHS` controls emoji, Unicode, or ASCII glyph spelling;
- `UNI20_CHARSET` controls UTF-8 preservation, escaping, or replacement.

Redirected display is still human-facing transcript output. It should not be
treated as structured logging unless the caller installs a structured sink.

## Python Behavior

Python bindings must not rely on C `stdout` or `stderr` for ordinary display.
When Uni20 is imported from Python, the binding layer should be able to install
a Python-aware sink:

- render with a policy derived from Python stream properties;
- write through `sys.stdout.write` or `sys.stderr.write`;
- acquire the GIL before calling Python;
- respect Python redirection and pytest capture;
- avoid direct C++ stdio except for abort-path emergency diagnostics.

Python policy detection should not require a fake `FILE*`. It should be able to
construct a `presentation::output_policy` directly from:

- `sys.stdout.isatty()` / `sys.stderr.isatty()`;
- notebook or rich-display detection when available;
- explicit per-call Python options;
- environment defaults such as `UNI20_COLOR`, `UNI20_GLYPHS`, `UNI20_CHARSET`,
  `NO_COLOR`, and `COLUMNS`.

For long-running C++ calls entered from Python, worker threads should not call
Python directly. They should emit display events into a queue or context. The
owning Python boundary can drain the queue at safe points while holding the GIL.

## Async Behavior

Async tasks should not render directly in the general case. They should emit
events:

```text
worker task -> display event queue -> sink drain -> presentation render -> output
```

This avoids:

- interleaved writes from worker threads,
- unsafe Python calls from non-Python threads,
- inconsistent ordering between C++ stdio and Python streams,
- accidental terminal-width decisions on background threads.

Events should optionally carry task metadata:

- task id,
- task name,
- context name such as `krylov`, `dmrg`, or `tensor`,
- thread id,
- source location for debug builds or verbose modes.

The default screen renderer may ignore most metadata. A debug sink or test sink
can preserve it.

## Relation To trace.hpp

`trace.hpp` remains the emergency and correctness-boundary layer:

- `CHECK`, `PRECONDITION`, `PANIC`: aborting diagnostics;
- `ERROR` / `ERROR_IF`: exception-boundary diagnostics;
- debug traces with source locations and parameter dumps.

`display::` is for ordinary user-facing output:

- progress,
- summaries,
- warnings that do not imply a failed invariant,
- solver statistics,
- formatted tables,
- reports and previews.

Some output may be semantically warning-like in both systems. The distinction is
the caller intent. If a condition indicates a violated invariant, use `trace`.
If it is something the user should know while the computation continues, use
`display`.

## Relation To Logging

Logging can be implemented as a display sink, but it should not drive the first
API.

Display output:

- optimized for a human watching the run;
- may use color, glyphs, spacing, and terminal width;
- may wrap for readability;
- often disappears after the run.

Durable logging:

- optimized for later search, parsing, or audit;
- often disables color and terminal-width assumptions;
- may prefer JSONL or stable key/value records;
- may include timestamps and source locations on every event.

The same event model can support both, but the default API should stay pleasant
for screen output.

## Sink Model

A first implementation can use a small value-type sink interface:

```cpp
namespace uni20::display
{
  using event_content = std::variant<presentation::styled_text, presentation::report_builder>;

  struct event
  {
      stream destination = stream::err;
      event_content content;
      bool newline = true;
      std::string context;
      std::source_location where;
  };

  using sink = std::function<void(event const&)>;

  void set_sink(sink replacement);
  sink const& current_sink();
}
```

This keeps the implementation simple and testable. A later design can replace
or supplement `std::function` with an abstract sink class if flushing,
back-pressure, or queue ownership becomes important.

Public emitting helpers should capture `std::source_location::current()` at the
call site and store it in the event. The first implementation can use a small
`display::format_string<Args...>` wrapper around `fmt::format_string<Args...>`
so calls such as `display::info("sweep {}", sweep)` keep fmt compile-time
checking while also preserving the user's source location. The event type
itself should not use a default member initializer for source location, because
that would capture the line in the header rather than the user's call site.

The default sink should:

1. select `stdout` or `stderr` from `event::destination`;
2. build a terminal output policy for that stream;
3. render the event text with the presentation layer;
4. write and flush conservatively for interactive output.

Tests can install a vector sink:

```cpp
std::vector<display::event> events;
display::scoped_sink capture([&](display::event const& event) { events.push_back(event); });
```

Python can install a sink that renders to Python strings and calls Python stream
objects. It should not need to change call sites.

## Display Contexts

Global convenience functions are important, but algorithms often need local
control. A context object should let a caller redirect or silence one algorithm
without changing global state:

```cpp
auto out = display::context("dmrg");
out.info("starting sweep {}", sweep);
out.warning("truncation error {:.3e}", truncation_error);
```

Possible context controls:

- enabled/disabled;
- minimum importance;
- destination stream;
- sink override;
- output policy override;
- table style defaults;
- context label visibility.

The global functions can delegate to a default context.

## Width Policy

Display should use terminal width only at layout boundaries:

- batch reports use the width while formatting the whole report;
- streaming tables choose schema widths at creation or explicit resize;
- individual message text wraps according to the sink policy;
- tensor previews use bounded preview rules.

Display should not format a table and then rely on generic whole-string
wrapping. Wrapping belongs to cells, messages, and preview metadata before the
final line is emitted.

For plain redirected output, width may come from `COLUMNS`, an explicit policy,
or a conservative default. The caller should be able to override it.

## Examples

### Simple Progress

```cpp
display::info("reading model from {}", path);
display::success("loaded {} sites", sites);
display::warning("using dense reference path for debug validation");
```

### Final Solver Report

```cpp
presentation::report_builder report("Krylov solve");
report.status(presentation::semantic_glyph::success, "converged")
    .field("dimension", n)
    .field("matvecs", matvecs)
    .field("residual", fmt::format("{:.3e}", residual));

report.table("Ritz values")
    .grid()
    .column("index")
    .column("value", presentation::table_alignment::decimal)
    .column("residual", presentation::table_alignment::decimal)
    .row(0, "-1.732050807568877", "1.0e-15")
    .row(1, "-1.414213562373095", "2.0e-14");

display::emit(report);
```

### Streaming DMRG Sweeps

```cpp
auto table = display::table("DMRG sweeps")
                 .column("sweep", display::width::fixed(5))
                 .column("energy", display::format::fixed(12))
                 .column("dE", display::format::scientific(3))
                 .column("m", display::width::fixed(5))
                 .column("status", display::width::share(3));

for (int sweep = 0; sweep != max_sweeps; ++sweep)
{
  auto result = run_sweep(sweep);
  table.row(sweep, result.energy, result.delta_energy, result.max_bond, result.converged ? "converged" : "continuing");
}
```

## Implementation Stages

### Stage 1: Message And Report Emission

- Add `src/uni20/common/display.hpp` and `.cpp`.
- Add `uni20::display` convenience functions for status messages.
- Add `display::emit(styled_text)` and `display::emit(report_builder)`.
- Add default C++ terminal sink.
- Add scoped test sink.
- Add tests for glyph policy, color policy, newline behavior, and sink capture.
- Add examples showing C++ terminal output and redirected/ascii policy behavior.

### Stage 2: Additional Semantic Status Glyphs

- Add `partial`, `deferred`, and `skipped` or `excluded` to
  `presentation::semantic_glyph`.
- Define emoji, Unicode, and ASCII renderings.
- Update tests and presentation examples.
- Document intended use so these do not get confused with trace severities.

### Stage 3: Streaming Tables

- Add schema-first `display::table`.
- Implement stable column width allocation.
- Implement per-row cell wrapping.
- Implement vertical fallback for very narrow widths.
- Add DMRG-style example output.
- Add tests for row wrapping that does not affect later rows.

### Stage 4: Python Sink

- Add a Python-aware sink in `bindings/python`.
- Derive policy from Python streams and explicit Python options.
- Ensure pytest capture sees display output.
- Add Python tests for `sys.stdout` / `sys.stderr` redirection.
- Keep abort-path trace diagnostics separate.

### Stage 5: Async Queue Sink

- Add a queue sink suitable for worker tasks.
- Add explicit drain points.
- Preserve optional task metadata.
- Test deterministic ordering in simple async examples.

## Open Questions

- Should the namespace be exactly `uni20::display`, or should the low-level sink
  API live under `uni20::display::detail` while the user API stays flat?
- Should `skipped` and `excluded` be separate semantic glyphs, or is one concept
  enough?
- Should the default destination for `info` and `success` be stdout while
  `warning` and `failure` go to stderr, or should all display messages default
  to stderr unless explicitly directed?
- Should streaming table rows flush immediately by default?
- Should contexts be global singletons, stack-scoped values, or explicit objects
  passed into algorithms?
- How much source-location metadata should ordinary display events carry by
  default?
- Should display expose progress-line rewriting later, or keep every emitted
  line append-only for cluster transcripts?

## Initial Recommendation

Start small but preserve the architecture:

1. Implement `display::info`, `success`, `warning`, `failure`, `status`, and
   `emit`.
2. Route through a replaceable sink from the beginning.
3. Reuse `presentation::styled_text` and `presentation::report_builder`
   directly.
4. Add `partial`, `deferred`, and `skipped` semantic glyphs in the presentation
   layer.
5. Add streaming tables as the first substantial display-specific formatter.
6. Delay durable logging, Python rich display, and async queue draining until
   the C++ sink and streaming table API are stable.
