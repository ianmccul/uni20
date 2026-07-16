# src/uni20/common

This directory contains shared infrastructure used across most Uni20 modules:
diagnostics, presentation, and lightweight containers. Code here should remain
broadly reusable and should avoid depending on high-level tensor, linalg, async,
or symmetry layers.

## Contents

- `trace.hpp`, `trace_impl.hpp`, `trace_format_mdspan.hpp`: checked diagnostics
  and trace formatting.
- `diagnostic_error.hpp`: structured user/runtime error base with source and
  stacktrace context.
- `gtest.hpp`: test helper integration.
- `presentation*`, `display*`, `terminal*`: user-facing formatting and terminal
  helpers, including mdspan previews and policy-aware stacktrace presentation
  when available.
- `aligned_buffer.hpp`, `static_vector.hpp`: small storage utilities.
- `nifty_counter.hpp`: ordered one-time initialization and finalization shared
  by namespace-scope users across translation units.
- `floating_eq.hpp`, `half_int.hpp`, `namedenum.hpp`, `string_util.hpp`,
  `demangle.hpp`: common scalar, enum, string, and type-name helpers.

## Notes

- Keep diagnostics usable from low-level code. Avoid making common utilities
  depend on module-specific types.
- Configured mdspan integration and structural helpers belong in
  [`../mdspan/`](../mdspan/).

## Related Documentation

- [Source tree map](../)
- [Diagnostics and presentation](../../../docs/diagnostics/)
- [Presentation formatting](../../../docs/diagnostics/presentation.md)
- [Trace macros and failure policy](../../../docs/diagnostics/trace_macros.md)
