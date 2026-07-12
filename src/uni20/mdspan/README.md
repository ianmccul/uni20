# `src/uni20/mdspan`

This directory contains Uni20's mdspan-facing structural utilities. It works
with the Kokkos reference mdspan implementation through `stdex::` and provides
small helpers used by dense kernels and layout-aware algorithms.

## Contents

- `concepts.hpp`: concepts for mdspan-like objects and related structural
  properties.
- `conjugate_accessor.hpp`: read-only mdspan accessor adaptor and `conj(...)`
  view helper for lazy complex conjugation.
- `strides.hpp`: stride inspection and stride utility helpers.
- `iteration_plan.hpp`: iteration planning over extents and layouts.
- `zip_layout.hpp`: helpers for matching or combining view layouts.

## Notes

- Include `uni20/common/mdspan.hpp` when code needs the configured mdspan
  implementation itself.
- This module should describe structure and layout, not ownership or backend
  dispatch policy.
- `SpanLike` is the complete readable mdspan protocol used by leaf kernels: its
  descriptor aliases must agree, it exposes rank and extent observers, and its
  rank-dimensional `operator[]` returns the declared `reference` type.
  `MutableSpanLike` additionally proves assignment through that indexed result.
- `StridedMdspan` refines `SpanLike` by requiring both mdspan and mapping stride
  observers. Code constrained by these concepts should not assume additional
  structural operations without adding the corresponding refinement.
- A pointer `data_handle_type` is not enough to prove direct memory semantics.
  Backends that bypass `access(...)` must check for `stdex::default_accessor`
  or an explicitly lowerable accessor such as Uni20's `conjugated_accessor`.
- `uni20::conj(span)` is the user-facing lazy conjugation helper. Its accessor
  follows the C++26 `std::linalg::conjugated_accessor` direction while keeping
  Uni20's value-level `conj` behavior for real scalar types.
