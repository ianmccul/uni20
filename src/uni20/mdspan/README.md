# src/uni20/mdspan

This directory contains Uni20's mdspan-facing structural utilities. It works
with the Kokkos reference mdspan implementation through `stdex::` and provides
small helpers used by dense kernels and layout-aware algorithms.

## Contents

- `mdspan.hpp`: configured gateway to the Kokkos reference implementation in
  namespace `stdex::`.
- `device_mdspan.hpp`: mdspan-shaped mapping and accessor metadata paired with
  a data descriptor for later handle acquisition.
- `format.hpp`: `std::format` and `{fmt}` support for mdspan extents.
- `concepts.hpp`: concepts for mdspan-like objects and related structural
  properties.
- `conjugate_accessor.hpp`: read-only mdspan accessor adaptor and `conj(...)`
  view helper for lazy complex conjugation.
- `generated_accessor.hpp` and `generated_layout.hpp`: read-only generated
  values with synthetic, non-strided logical offset mapping.
- `strides.hpp`: stride inspection and stride utility helpers.
- `iteration_plan.hpp`: backend-neutral iteration planning over strided
  mappings.
- `transform_view.hpp`: lazy read-only unary and variadic elementwise views.
- `zip_layout.hpp`: helpers for matching or combining view layouts.

## Notes

- Include `uni20/mdspan/mdspan.hpp` when code needs the configured mdspan
  implementation itself.
- This module should describe structure and layout, not ownership or backend
  dispatch policy. CPU execution of iteration plans belongs to the linalg CPU
  backend.
- `SpanLike` is the complete readable mdspan protocol used by leaf kernels: its
  descriptor aliases must agree, it exposes rank and extent observers, and its
  rank-dimensional `operator[]` returns the declared `reference` type.
  `MutableSpanLike` additionally proves assignment through that indexed result.
- `DeviceSpanLike` is the broader structural protocol for mdspan metadata whose
  data handle is either immediately present or available through a data
  descriptor. `device_mdspan` is the standard unresolved materialization, but
  independent types may satisfy the concept directly. It preserves the actual
  mapping and accessor while intentionally exposing neither `data_handle()` nor
  element indexing.
- `MutableDeviceSpanLike` refines eventual write capability through an
  assignable accessor reference. It does not add indexing to an unresolved
  descriptor.
- `StridedMdspan` refines `SpanLike` by requiring both mdspan and mapping stride
  observers. Code constrained by these concepts should not assume additional
  structural operations without adding the corresponding refinement.
- A pointer `data_handle_type` is not enough to prove direct memory semantics.
  Backends that bypass `access(...)` must check for `stdex::default_accessor`
  or an explicitly lowerable accessor such as Uni20's `conjugated_accessor`.
- Read-only accessor policies declare const `element_type`, including calculated
  accessors that return values. Ordinary `MutableSpanLike` accessors also prove
  indexed assignment. An opaque accessor without assignable element semantics
  is not mutable merely because its handle is pointer-shaped.
- `uni20::const_access(span, indices...)` performs read-only scalar access
  directly through a span's mapping and const-adapted accessor. Use it when a
  descriptor owner needs const element semantics without constructing a second
  mdspan for each access.
- A tensor descriptor's const `mdspan()` overload resolves a const-element view.
  Mutable tensor access is exposed only by the non-const overload.
- `uni20::conj(span)` is the user-facing lazy conjugation helper. Its accessor
  follows the C++26 `std::linalg::conjugated_accessor` direction while keeping
  Uni20's value-level `conj` behavior for real scalar types.
- `uni20::transform_view(function, spans...)` constructs a read-only expression
  descriptor. Eager overwrite and update use the dispatched tensor operations
  `assign_transform` and `transform_inplace`.

## Related Documentation

- [Source tree map](../)
- [Device mdspan contract](../../../docs/tensor/device_mdspan.md)
- [Tensor dispatch and view semantics](../../../docs/tensor/dispatch_and_view_semantics_draft.md)
- [BLAS/LAPACK mdspan wrappers](../../../docs/linalg/blas_lapack_wrappers.md)
- [Mdspan linear algebra dispatch](../../../docs/linalg/mdspan_dispatch.md)
