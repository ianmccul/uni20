# Generated Tensors and Reshape

For the wider operation taxonomy and current Async support, see
[Tensor Operations, Semantics, and Async Support](tensor_operations.md).

## Generated Values

Uni20 represents uniform and identity-like values as compact readable tensors:

```cpp
auto zero = uni20::zeros<double>(2, 3);
auto one = uni20::ones<double>(2, 3);
auto filled = uni20::full(4.0, 2, 3);
auto identity = uni20::eye<double>(2, 3, 4);
```

The C++ `full` factory puts the fill value first so its scalar type can be
inferred; Python bindings may provide Python-specific argument ordering
independently.

These objects model `TensorView`, but their accessors return generated values
rather than references into a dense element allocation. Consequently they do
not model `MutableTensorView`, `StridedTensorView`, or reusable dense
`OwningTensor`. Their `GeneratedLayout` maps logical indices to synthetic
accessor offsets; it does not claim a row-major or column-major physical order.

Algorithms that need writable storage, such as destructive LAPACK routines,
materialize a work tensor through the normal copy and backend-dispatch path:

```cpp
auto default_owner = uni20::make_tensor(uni20::ones<double>(3, 4, 5));
auto row_owner =
    uni20::make_tensor<uni20::RowMajor>(uni20::ones<double>(3, 4, 5));
```

The layout is an optional compile-time policy because it changes the concrete
return type. Without an explicit policy, `make_tensor` preserves a canonical
`layout_left` or `layout_right` physical source. A generated or otherwise
noncanonical source uses the ordinary `Tensor` default, `ColumnMajor`.
Materialization traverses canonical output storage in its native order, so a
generated source does not impose strided writes on the destination.

`GeneratedStorage` is backend-neutral. When generated and concretely stored
operands participate in one operation, the concrete storage policy selects the
default backend list. Each candidate kernel still decides whether it supports
the generated accessor. With no concrete operand, generated storage provides
the host reference backend as its default.

### Generalized `eye`

For a rank-`R` result with extents `(n0, ..., n{R-1})`, `eye` is defined by

```text
eye[i0, ..., i{R-1}] = 1  if i0 == i1 == ... == i{R-1}
                        0  otherwise.
```

The extents need not be equal. Rank-zero `eye<T>()` is scalar one; rank-one
`eye<T>(n)` is an all-ones vector; rank-two `eye<T>(m, n)` is the ordinary
rectangular identity matrix. This is also the tensor-network copy tensor, so a
separate `copy_tensor` factory is unnecessary.

## Reshape Operations

Uni20 separates aliasing, object mutation, and owning value construction:

```cpp
auto alias = uni20::reshape_view(x, 3, 2);
uni20::reshape_inplace(x, 3, 2);
auto copied = uni20::reshape(x, 3, 2);
auto transferred = uni20::reshape(std::move(x), 3, 2);
```

One requested extent may be `-1`; Uni20 infers it from the source element
count. All forms reject mismatched element counts and ambiguous inference.

### `reshape_view`

`reshape_view` never allocates or copies elements. It preserves the source data
handle and accessor. Automatic order selection is available only when the
source mdspan's static layout type is `layout_left` or `layout_right`; the
result preserves that same layout type.

For a general strided mapping, select the interpretation explicitly:

```cpp
auto column_alias = uni20::reshape_view_left(strided, 3, 2);
auto row_alias = uni20::reshape_view_right(strided, 3, 2);
```

Both explicit forms require a unique, exhaustive, canonical contiguous mapping
in the selected order. Singleton-axis strides are ignored because they do not
affect addressing. A contiguous rank-one source is compatible with either
choice, and the selected result layout resolves the otherwise unavoidable
ambiguity. Non-contiguous mappings are rejected by every no-copy form.

A tensor-level view accepts lvalues only because its descriptor does not extend
the lifetime of addressable source storage. Mutability follows the source
accessor: reshaping a mutable tensor gives a mutable alias, while a const source
remains read-only. Generated tensors are layout-neutral and do not have a
structural `reshape_view`; use owning `reshape` when a value-level reshape is
required.

Repeated tensor-level reshapes preserve order through the result's static
`layout_left` or `layout_right` type. No runtime provenance flag is required,
including when an intermediate shape contains singleton extents.

For asynchronous tensors, `async::reshape_view(parent, ...)` returns an
owner-retaining `Async<TensorView>` alias on the parent's exact epoch queue.
The alias may be created while the parent value is still pending. It resolves
and validates its reshaped mdspan only after the parent epoch becomes readable,
and a parent failure propagates through the alias. Mutable aliases support
write-through assignment; aliases of a const parent are read-only. The async
API also provides `reshape_view_left` and `reshape_view_right` for strided
parents whose order must be selected explicitly.

### `reshape_inplace`

`reshape_inplace` replaces a canonically laid-out owning `BasicTensor` mapping
without moving or reallocating its storage. Because mdspan rank is part of the
C++ type, the new shape must have the same rank. Existing standalone mdspan
descriptors retain their old mappings, as normal for copied view descriptors.

### Owning `reshape`

Plain `reshape` always returns an owning tensor. Its canonical owning
`BasicTensor` overload takes the source by value:

- passing an lvalue creates an independent copy;
- passing an owning rvalue grants permission to transfer its allocation;
- the returned type has the requested compile-time rank and preserves the
  source's canonical layout type.

As with consuming eigensolver overloads, `std::move` grants permission to reuse
storage but is not a separate operation name. There is therefore no
`reshape_move` or `reshape_copy` API.

Canonical non-owning sources are reshaped before materialization so their
contiguous sequence is preserved. Layout-neutral generated inputs are first
materialized in the default or explicitly requested layout, then reshaped by
transferring that allocation:

```cpp
auto column_result = uni20::reshape(uni20::eye<double>(2, 3), 3, 2);
auto row_result =
    uni20::reshape<uni20::RowMajor>(uni20::eye<double>(2, 3), 3, 2);
```

Plain reshape does not guess an order for a `layout_stride` source. Use
`reshape_view_left` or `reshape_view_right`, then materialize the selected view
when an owning result is required. Python bindings can build NumPy-compatible
`C`, `F`, `A`, and copy policies from these explicit primitives without adding
runtime layout ambiguity to the C++ return types.
