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
not model `MutableTensorView` or reusable dense `OwningTensor`. Algorithms that
need writable storage, such as destructive LAPACK routines, materialize a work
tensor through the normal copy and backend-dispatch path.

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
handle and accessor, so default, conjugating, and generated accessor semantics
remain intact. The source mapping must be unique, exhaustive, and canonically
contiguous in row-major or column-major order. The result preserves that order.

A tensor-level `reshape_view` accepts lvalues only because its descriptor does
not extend the lifetime of addressable source storage. Mutability follows the
source accessor: reshaping a mutable tensor gives a mutable alias, while a
const or generated source remains read-only.

### `reshape_inplace`

`reshape_inplace` replaces an owning `BasicTensor` mapping without moving or
reallocating its storage. Because mdspan rank is part of the C++ type, the new
shape must have the same rank. Existing standalone mdspan descriptors retain
their old mappings, as normal for copied view descriptors.

### Owning `reshape`

Plain `reshape` always returns an owning tensor. Its owning `BasicTensor`
overload takes the source by value:

- passing an lvalue creates an independent copy;
- passing an owning rvalue grants permission to transfer its allocation;
- the returned type has the requested compile-time rank.

As with consuming eigensolver overloads, `std::move` grants permission to reuse
storage but is not a separate operation name. There is therefore no
`reshape_move` or `reshape_copy` API. Compact generated and other non-owning
strided tensors are materialized through their `reshape_view` representation.

The initial implementation requires a canonical contiguous source for every
reshape form. A future explicitly ordered materializing path may extend plain
`reshape` to arbitrary indexed or non-contiguous inputs without weakening the
strict no-copy contract of `reshape_view`.
