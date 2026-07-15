# Diagnostics And Presentation Documentation

This directory covers semantic formatting, user-facing display, correctness
diagnostics, and Graphviz output.

## Current Guides

- [Presentation Formatting](presentation.md) defines styled text, semantic
  glyphs, width-aware layout, reports, tables, and Tensor previews.
- [Trace Macros](trace_macros.md) documents `CHECK`, `PANIC`, `ERROR`, tracing,
  stacktraces, and presentation-layer rendering.
- [Graphviz Basics](graphviz.md) is the developer reference used by async DAG
  diagnostics.

## Design And Forward Work

- [Display Layer](display_layer.md) records implemented display/report slices
  and remaining sink, Python, async-queue, and context work.
- [Diagnostics and Logging](logging_plan.md) is a design note for durable and
  structured diagnostics beyond terminal display.

## Source Navigation

- [Common diagnostics and presentation](../../src/uni20/common/README.md)
- [Async task-registry diagnostics](../../src/uni20/async/README.md)
- [Linalg dispatch diagnostics](../../src/uni20/linalg/README.md)
- [Presentation examples](../../examples/presentation/README.md)
- [Common diagnostic examples](../../examples/common/README.md)
