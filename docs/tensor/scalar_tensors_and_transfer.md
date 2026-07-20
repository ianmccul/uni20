# Scalar Tensors, Host Scalars, and Storage Transfer

**Status:** scalar tensors plus CPU sum, inner-product, and norm reductions are
implemented. Full and axis-selective sums also have all-Async Tensor lowering.
Async inner-product/norm lowering, provider reductions, CUDA, multi-GPU, and
MPI transfer APIs remain future work.

## Summary

Uni20 distinguishes:

- a **C++ scalar**, such as `double` or `uni20::complex<double>`, which is an
  ordinary value available to host control flow;
- a **scalar tensor**, which is a rank-zero `Tensor` and retains storage,
  backend, lifetime, and Async semantics;
- a **storage transfer**, which produces a tensor in another storage domain
  without changing its rank.

These operations are related but must not be conflated. In particular,
accessing a scalar tensor must not silently synchronize or migrate device
storage.

## Scalar Tensor Vocabulary

The owning alias is:

```cpp
template <typename ElementType, typename StoragePolicy = VectorStorage,
          typename AccessorFactory = DefaultAccessorFactory>
using ScalarTensor =
    Tensor<ElementType, 0, StoragePolicy, ColumnMajor, AccessorFactory>;
```

The layout parameter is structurally irrelevant at rank zero. The alias uses
the ordinary `Tensor` default only so that scalar tensors participate in the
same owner, storage-policy, backend-selection, and Async machinery as other
ranks.

The corresponding concepts describe rank-zero tensor-level objects:

```cpp
template <class T>
concept ScalarTensorView = RankedTensorView<T, 0>;

template <class T>
concept MutableScalarTensorView = MutableRankedTensorView<T, 0>;
```

`uni20::Scalar` continues to mean a numeric C++ scalar type. Reusing that name
for a rank-zero tensor would erase a useful compile-time distinction.

## Rank-Zero Indexing

C++23 permits a subscript operator with zero indices. Tensor indexing therefore
extends naturally across all supported ranks:

```cpp
matrix[i, j] // rank two
vector[i]    // rank one
scalar[]     // rank zero
```

`scalar[]` accesses the sole logical element. It is not a transfer operation.
It is available only when the tensor's ordinary accessor can be invoked in the
current compilation and execution context.

For `VectorStorage`, `scalar[]` returns a host-accessible element reference in
the same way as indexing any other host tensor. `CudaAsyncTensor` deliberately
does not provide a host-callable element subscript merely because its rank is
zero. Host extraction from such a tensor requires an explicit host-result or
transfer operation.

There is no separate `item()` operation in the initial design. The ordinary
rank-zero subscript already expresses element access without implying hidden
movement.

## Reduction Result Contracts

A full tensor reduction has two useful result contracts.

### Storage-Preserving Result

The unqualified operation returns a scalar tensor:

```cpp
auto total = uni20::sum(x);
auto overlap = uni20::inner_product(x, y);
auto magnitude = uni20::norm(x);
```

The result remains in the selected storage and execution domain. For example,
a future CUDA reduction may leave one scalar in device storage so that a
subsequent device kernel can consume it without a device-to-host transfer.

The result element type follows the mathematical operation:

- `sum` normally preserves the input element type;
- `inner_product` follows its scalar-field and conjugation contract;
- `norm` returns the corresponding real element type.

Accumulator precision is a separate backend and operation policy. It must not
be inferred solely from the result tensor's element type.

### Host Scalar Result

An explicitly host-returning operation returns a C++ scalar:

```cpp
auto total = uni20::sum_host(x);
auto overlap = uni20::inner_product_host(x, y);
auto magnitude = uni20::norm_host(x);
```

The `_host` suffix is deliberate. For non-host storage, the operation may
require synchronization, communication, or transfer. A backend may still
produce the host value directly when its provider supports that result mode;
the API does not require an intermediate scalar tensor.

The host and storage-preserving forms implement the same mathematical
operation. They differ only in result residency, synchronization, and the
concrete return type.

Return type alone cannot select between these contracts, so they require
different function names.

### Partial Reductions

A reduction over only some axes returns a tensor whose rank is the number of
surviving axes:

```cpp
auto rows = uni20::sum(x, 1);
auto middle = uni20::sum(x, 0, 2);
uni20::sum(output, x, -1);
```

Axis arguments are runtime integral values, while the number of axes is part of
the C++ overload and therefore determines the result rank at compile time.
Negative axes count backward from the input rank. Duplicate and out-of-range
axes are rejected before dispatch. The reduced axes may be supplied in any
order; surviving axes retain their original logical order.

The same naming rule applies:

- an operation returning a tensor preserves storage residency;
- `_host` is reserved for an ordinary C++ scalar and therefore applies only
  when every logical axis has been reduced.

There is no `keepdims` option in the current API. Callers that require singleton
reduced dimensions can add the desired structural reshape explicitly.

## Krylov Algorithms

The matrix-free Krylov interfaces intentionally require host scalar results for
inner products and norms. These values participate immediately in convergence,
restart, and control-flow decisions.

Tensor-based Krylov adapters should therefore lower their existing operations
through:

```cpp
inner_product_host(x, y);
norm_host(x);
```

This does not imply that all tensor reductions should return host values.
Device-resident scalar tensors remain useful in tensor expressions and fused
or asynchronously composed device work.

## Async Semantics

Async sums preserve the same result distinction:

```cpp
sum(Async<Tensor>)       -> Async<ScalarTensor>
sum_host(Async<Tensor>)  -> Async<Element>
```

Axis-selective sums return `Async<Tensor<..., R - N>>` in the storage policy
and canonical layout selected by the synchronous operation. Explicit async
outputs are also supported: owning outputs may be constructed or resized after
the input becomes readable, while mutable async aliases are fixed-shape
write-through destinations.

`sum_host` schedules the host-result operation rather than synchronously
blocking the caller merely because the final value is a C++ scalar. Observing
the returned `Async<Element>` through `get_wait()` or a downstream buffer
performs the normal Async synchronization.

Reduction axes are normalized and validated before submission. Shape
preparation and backend dispatch remain deferred until the input epoch is
readable. An explicit output must not share an epoch queue with its input;
violating that contract is rejected before buffer enrollment.

A storage-preserving scalar tensor can flow directly into another Async tensor
operation. Its owner and epoch queue provide the same lifetime and dependency
tracking as for tensors of higher rank.

## Kernel Lowering

The leaf-kernel interface remains output-first.

A storage-preserving reduction receives a writable result mdspan of rank
`R - N`; a full reduction therefore receives a rank-zero output:

```cpp
dispatch_kernel(selector, sum_reduction_op<R, N>{axes},
                output.mdspan(), input.mdspan());
```

This keeps kernel dispatch consistent with GEMM, GEMV, copy, and elementwise
transforms. The output tensor determines result storage and participates in
backend selection.

A host-scalar result is a distinct lowering contract. It may use an explicit
host scalar output wrapper or another narrowly defined host-output operand. A
plain pointer or rank-zero mdspan is not sufficient to describe memory
residency for future heterogeneous backends.

Dispatch should not gain arbitrary value-returning leaf kernels solely for
reductions. Explicit output operands keep storage, mutation, and multi-output
semantics visible.

## Storage Transfer Is Separate

Moving a tensor between storage domains preserves its logical shape and returns
another tensor. Intended high-level names include:

```cpp
auto host_x = uni20::to_host(x);
auto gpu_x = uni20::to_device(x, device);
auto gpu1_x = uni20::to_device(x, gpu1);
```

An explicit-output form may be useful for allocation reuse and asynchronous
execution:

```cpp
uni20::copy_to(destination, source);
```

These names are provisional until Uni20 has concrete CUDA storage and device
context types. Their semantic requirements are already clear:

- transfer does not change tensor rank;
- transfer does not extract a C++ scalar;
- destination storage and device identity are explicit;
- an owning result has the destination's storage policy;
- cross-device movement may select direct peer transfer or a staged path;
- ordinary tensor indexing never performs transfer implicitly.

`to_host(scalar_tensor)[]` is therefore a valid explicit two-step expression,
but a caller that only needs the value should prefer `sum_host`, `norm_host`,
or the corresponding direct host-result operation.

## Result Allocation

The current front end constructs an owning result by rebinding the input's
static storage policy to the result element type and rank. Backend-neutral
generated inputs materialize into `VectorStorage`. Canonical row-major or
column-major layout is preserved for partial sums; noncanonical inputs use the
default column-major layout. Rank-zero results use `ScalarTensor`.

This static policy mechanism does not yet carry runtime device or context state.
The design must eventually account for:

- changing rank while preserving storage policy;
- changing element type, as for complex input to real norm output;
- preserving or explicitly selecting device/context state;
- generated and other backend-neutral inputs;
- views whose owner type or runtime storage context is intentionally hidden.

Such cases should require an explicit output or a future result-allocation
customization rather than silently selecting unrelated host storage.

## Rank-Zero Construction Invariant

A rank-zero mapping has one logical element. An owning scalar tensor must
therefore allocate storage for one element when it is constructed as a value.

`Tensor<T, 0>` now default-constructs its one logical element, establishing:

```text
constructed rank-zero Tensor
  => mapping requires one element
  => storage contains at least one element
```

Positive-rank default owners remain empty deferred outputs because their
runtime shape is not yet known.

## CPU Implementation

The current host implementation provides:

1. `ScalarTensor`, `ScalarTensorView`, and `MutableScalarTensorView`;
2. const and mutable `scalar[]` indexing;
3. one shared output-first CPU reduction executor for full and partial
   reductions;
4. `sum_reduction_op`, `inner_product_op`, and `norm_op` CPU kernels;
5. same-field compensated accumulation for real and complex sums and inner
   products;
6. scaled sum-of-squares norms that avoid unnecessary overflow and underflow;
7. storage-preserving and `_host` Tensor front ends;
8. direct host-scalar kernel outputs without a one-element allocation;
9. axis normalization, duplicate rejection, negative axes, and zero-extent
   identities;
10. accessor-respecting generated and conjugating input paths;
11. DenseHostVector Krylov operations routed through the dispatched kernels;
12. optional binary128 validation in the MPLAPACK test lane.

The current `sum` domain is real and complex scalar tensors. Integer sum policy
is deferred until overflow and promotion semantics are specified.

CUDA storage, device identity, cross-device transfer, and device-resident
provider reductions remain outside the current implementation.
