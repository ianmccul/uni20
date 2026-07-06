# `src/uni20/mdspan`

This directory contains Uni20's mdspan-facing structural utilities. It works
with the Kokkos reference mdspan implementation through `stdex::` and provides
small helpers used by dense kernels and layout-aware algorithms.

## Contents

- `concepts.hpp`: concepts for mdspan-like objects and related structural
  properties.
- `strides.hpp`: stride inspection and stride utility helpers.
- `iteration_plan.hpp`: iteration planning over extents and layouts.
- `zip_layout.hpp`: helpers for matching or combining view layouts.

## Notes

- Include `uni20/common/mdspan.hpp` when code needs the configured mdspan
  implementation itself.
- This module should describe structure and layout, not ownership or backend
  dispatch policy.
