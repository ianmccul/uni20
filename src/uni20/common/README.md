# `src/uni20/common`

This directory contains shared infrastructure used across most Uni20 modules:
diagnostics, presentation, lightweight containers, and configured mdspan access.
Code here should remain broadly reusable and should avoid depending on
high-level tensor, linalg, async, or symmetry layers.

## Contents

- `trace.hpp`, `trace_impl.hpp`, `trace_format_mdspan.hpp`: checked diagnostics
  and trace formatting.
- `gtest.hpp`: test helper integration.
- `mdspan.hpp`: project include point for the Kokkos reference mdspan
  implementation in the `stdex` namespace.
- `presentation*`, `display*`, `terminal*`: user-facing formatting and terminal
  helpers, including policy-aware stacktrace presentation when available.
- `aligned_buffer.hpp`, `static_vector.hpp`: small storage utilities.
- `floating_eq.hpp`, `half_int.hpp`, `namedenum.hpp`, `string_util.hpp`,
  `demangle.hpp`: common scalar, enum, string, and type-name helpers.

## Notes

- `mdspan.hpp` is the preferred include for mdspan in Uni20 source. It owns the
  local configuration checks for namespace and indexing policy.
- Keep diagnostics usable from low-level code. Avoid making common utilities
  depend on module-specific types.
