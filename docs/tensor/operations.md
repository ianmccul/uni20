# Tensor Operations, Semantics, and Async Support

**Status:** canonical guide for the currently implemented dense Tensor
operation model. This page records behavior that exists in the codebase and
separately identifies operations that do not yet have an Async wrapper.

This guide covers local dense tensors, tensor-level views, and their lowering
to dense kernels. Symmetry-aware `BlockTensor` operations have additional
metadata and selection-rule contracts and must not use an implicit dense
fallback.

The distinction between C++ scalar results, rank-zero Tensor results, and
future storage migration is specified in
[Scalar Tensors, Host Scalars, and Storage Transfer](scalar_tensors_and_transfer.md).

## Header Map

The main entry points are:

```cpp
#include <uni20/tensor/tensor.hpp>       // Tensor, DenseMatrix, generated values, views, reshape
#include <uni20/tensor/copy.hpp>         // copy, make_tensor, materializing reshape
#include <uni20/tensor/transform.hpp>    // elementwise overwrite and update
#include <uni20/tensor/reductions.hpp>   // sums, inner products, and Euclidean norms
#include <uni20/tensor/async.hpp>        // async::conj, async::reshape_view
#include <uni20/linalg/linalg.hpp>       // synchronous dense linalg operations
#include <uni20/linalg/async.hpp>        // implemented Async linalg wrappers
```

Including a narrower operation header is preferred in library implementation
files when it keeps dependencies clear.

## Layer Model

Tensor operations have three distinct layers:

```text
Tensor operation
  -> shape, ownership, and output policy
  -> storage-derived backend selector
  -> fixed-operand MdspecLike normalization
  -> backend dispatch on normalized descriptors
  -> backend-specific execution-domain acquisition
  -> resolved mdspan leaf kernel
```

An Async operation adds scheduling around the same synchronous operation:

```text
Async<Tensor> operands
  -> enroll ReadBuffer and WriteBuffer epochs
  -> scheduled coroutine awaits stored Tensor values
  -> fixed-operand normalization or replaceable-output dispatch
  -> descriptor acquisition and resolved mdspan leaf kernel
```

Leaf kernels do not receive `Tensor` or `Async<Tensor>`. They receive resolved
mdspan-like operands after ownership, storage domain, output policy, and async
ordering have been handled by higher layers.

Operation-tag `kernel_accepts_types(...)` and `try_kernel(...)` overloads are
backend adapters and therefore accept normalized `MdspecLike` fixed operands,
not tensor views or resolved mdspans. A backend calls an ordinary lower-level
function such as `cpu::gemm` or `cpu::gemv`, or a provider adapter such as
`blas::try_gemm` or `blas::try_gemv`, after acquiring the required
execution-domain leases.

### Type Roles

| Type or concept | Role |
|---|---|
| `Tensor<Element, Rank, ...>` | Concrete column-major-default owner with compile-time rank and runtime extents on every axis by default. |
| `BasicTensor<Element, Extents, ...>` | Extents-first alias for a `Tensor` specialization with mixed or static extents. |
| `ColumnMajorTensor`, `RowMajorTensor`, `StridedTensor` | Named runtime-extents owner aliases for an explicit physical layout policy. |
| `DenseMatrix<Element, Layout>` | Rank-two host `Tensor`; column-major by default. |
| `CudaTensor<Element, Rank, Layout>` | Owning CUDA device Tensor. It uses the installed runtime's default device unless explicit device resources are supplied, and its mdspan handle is opaque to host element access. Direct operations may block during resource admission; async matrix products lower through `CudaTask` and cuBLAS. |
| `CudaMatrix<Element, Layout>` | Rank-two `CudaTensor`; column-major by default. |
| `ScalarTensor<Element, StoragePolicy, ...>` | Rank-zero owning Tensor that retains storage, backend, lifetime, and Async semantics. |
| `GeneratedTensor` | Compact, layout-neutral read-only tensor whose accessor computes values without dense element storage. |
| `TensorView` | Readable tensor-level object exposing extents, a normalized `MdspecLike` representation, and a backend selector. It is a concept, not a base class. |
| `MutableTensorView` | `TensorView` whose normalized writable mdspec permits eventual element assignment. |
| `ImmediateTensorView` | `TensorView` whose normalized mdspec is already an mdspan and which exposes `mdspan()` directly. |
| `MutableImmediateTensorView` | `ImmediateTensorView` whose resolved mdspan permits writes. |
| `OwningTensor` | Explicit opt-in classification whose move operation transfers the storage and lifetime exposed by `mdspan()`. |
| `StridedTensorView` | `TensorView` whose normalized mdspec has a strided mapping. |
| `StridedImmediateTensorView` | Immediate tensor view whose resolved mdspan has a strided mapping. Operations such as reshape may require this refinement. |
| Resolved mdspan | Short-lived leaf-kernel operand containing handle, mapping, extents, and accessor semantics. |

Tensor objects deliberately do not model mdspan concepts. Front-end operations
accept tensor concepts, normalize fixed operands to mdspecs after backend
selection, and acquire resolved mdspans inside the selected backend.

## Operation Vocabulary

Names communicate ownership and mutation. New operations should use these
forms unless the underlying mathematical convention makes another name
materially clearer.

| Form | Semantic contract |
|---|---|
| `foo_view(x)` | Return a no-copy alias. The result does not extend an ordinary synchronous source lifetime unless its type explicitly says otherwise. |
| `foo_inplace(x)` | Mutate the existing tensor, its elements, or its descriptor. No independent value is returned. |
| `assign_foo(out, ...)` | Overwrite `out`. A resizable output may replace its shape and allocation. Old output values do not participate. |
| `add_foo(out, ...)` | Compound update. Existing output values participate, so shape is fixed and the output must already exist. |
| `foo(x)` returning an owner | Preserve lvalue inputs and allocate or materialize an owning result. |
| `foo(std::move(x))` | Grant permission to consume an owning input and transfer its allocation. Reuse is not guaranteed. |
| `Tensor(view)` | Eagerly materialize a readable tensor view using CTAD and the inferred physical layout. |
| `make_tensor(view)` | Explicit eager materialization boundary for a view or generated expression. |

There is no general rule that assignment to an arbitrary tensor view means
element assignment. A plain mdspan assignment rebinds a descriptor. A future
write-through slice proxy must define that behavior explicitly.

### Assignment by Object Category

| Object | Meaning of ordinary assignment |
|---|---|
| Owning `Tensor` or `BasicTensor` | Replace the owned value and descriptor state through the concrete type's copy/move assignment. Storage may be reused or replaced. |
| Plain mdspan | Rebind the mdspan descriptor. Elements are not copied. |
| Read-only semantic view | Assignment through the view is unavailable. |
| Future write-through slice/ref | Assign elements into the referenced region according to an explicitly documented contract. |

Elementwise copying remains the named `copy(out, in)` operation. This avoids
making descriptor assignment depend on whether a particular view happens to
own storage.

## Core Semantic Rules

### Readable, Mutable, and Owning Are Independent

A type may own compact descriptor state without owning a writable dense
allocation. `GeneratedTensor` is readable but neither mutable nor an
`OwningTensor`. A lazy conjugating view is also readable and non-owning.

Only a mutable owning rvalue is eligible for storage transfer. Passing a view
by value does not make it consumable.

### Views and Materialization

Structural views change addressing while preserving the direct write law.
Examples include reshape and future slice or component views. They may be
writable when their accessor and source permit it.

Future `real(x)` and `imag(x)` views of non-const complex storage are structural
component slices: they change scalar type, handle, and stride and may be
writable. They are not writable proxy-accessor transformations. Generic mutable
operation outputs should normally expose direct/default-access semantics;
writable proxy accessors require an operation-specific assignment law and
backend lowering.

Semantic transform views change values observed through an accessor. Complex
`conj(x)` is therefore read-only. For real tensors, `conj(x)` returns a const
identity view. Applying `conj` twice returns a read-only view of the original
values rather than a writable left-hand side.

Materialization is explicit:

```cpp
auto lazy = uni20::conj(matrix);
auto owned = uni20::Tensor(lazy);
auto equivalent = uni20::make_tensor(lazy);

uni20::Tensor<uni20::complex<double>, 2> output;
uni20::copy(output, lazy);
```

`make_tensor` accepts an optional compile-time layout policy:

```cpp
auto inferred = uni20::make_tensor(view);
auto row_major = uni20::make_tensor<uni20::RowMajor>(view);
```

The constructor and inferred `make_tensor` form preserve canonical row-major or
column-major physical sources. Generated and other noncanonical sources use
`Tensor`'s default column-major layout. `RowMajorTensor(view)` and
`ColumnMajorTensor(view)` work when that inferred specialization belongs to the
named alias; use `make_tensor<Layout>(view)` when the destination layout must be
changed. A runtime optional layout is intentionally not used because layout is
part of the concrete return type.

`copy` observes accessor semantics. A raw pointer-shaped data handle is not
enough to bypass an accessor; a provider backend may do so only for a default
accessor or a recognized accessor whose semantics it lowers explicitly.
A bare mdspan carries no storage policy, so its `copy` or `make_tensor`
operation requires an explicit backend selector. Tensor-level views supply
their selector through storage policy.

### Output Shape

Output operations use two contracts:

```cpp
uni20::require_output(output, required);            // validate only
uni20::prepare_output(output, required);            // resize owner or validate fixed view
uni20::prepare_output(output, required, placement); // shape plus backend storage requirement
```

Operation and backend code share the public `TensorExtentsLike`,
`tensor_extents_equal`, and `convert_tensor_extents` contracts when inspecting
or converting a proposed output shape. These are tensor output-policy
interfaces; callers should not depend on their implementation helpers in
`uni20::detail`.

An update operation uses `require_output` because old output values participate.
An overwrite backend may use `prepare_output`. A resizable owner retains a
matching allocation and calls `reset_shape` only when the shape differs. A
placement-aware overload additionally retains compatible storage or replaces
it through the Tensor's storage policy. A fixed view can never resize or
replace storage.

Shape preparation must happen before resolving the output mdspan because
resizing may invalidate its handle.

`prepare_output` may invalidate storage and resolved views. For an operation
whose contract declares the output replaceable, a backend may prepare it before
completing all acceptance checks. If that backend declines without writing
result elements or submitting work, a later backend receives the prepared
output and may reuse or replace it. If every backend declines, the output
remains valid but its shape, allocation, placement, and values may reflect
provisional preparation.

Inputs and fixed or update outputs remain unchanged on decline. Once a backend
writes result elements, submits work, or enters a provider operation, later
failure is terminal and must not trigger fallback.

### Move and Storage Reuse

`std::move` grants permission; it does not promise reuse. The operation may
still materialize when layout, accessor, storage policy, scalar type, or backend
requirements are incompatible.

Current examples are:

```cpp
auto reshaped = uni20::reshape(std::move(tensor), 6, 4);
auto eigensystem = uni20::linalg::eigh(std::move(matrix));
```

Existing synchronous views into transferred storage are invalidated under
ordinary C++ moved-from owner rules. Uni20 does not track them dynamically.

### Operand Roles and Aliasing

An operation assigns each tensor operand one semantic role:

| Role | Old values participate? | Shape policy | Aliasing contract |
|---|---|---|---|
| read-only input | yes | fixed | may alias another read-only input |
| overwrite output | no | may resize before mdspan resolution | must not overlap any input |
| update output | yes | fixed | appears once as the operation's read/write operand and must not overlap any other input |

An update output is not modelled as two operands that happen to alias. For
example:

```text
assign_transform(out, generator)      means out = generator()
assign_transform(out, op, lhs, rhs)   means out = op(lhs, rhs)
transform_inplace(out, op, rhs)       means out = op(old(out), rhs)
transform_inplace(out, op)            means out = op(old(out))
fill(out, value)                      means out = value
```

The first form has three physical operands. The second has two, not three: the
old value of `out` is read through the same output descriptor that is later
written. This distinction preserves the real memory-traversal problem for
backend optimization and avoids inventing an alias that lower layers must
rediscover.

Unless an operation explicitly assigns the output an update role, mutable
output storage must not overlap an input. An update output must likewise not
overlap any additional read-only input. `assign_product` and `add_product`
cheaply reject the obvious same-object case, but they do not attempt a complete
strided range-overlap proof. C++ callers remain responsible for less obvious
overlap.

Backend interfaces should receive an update output once and encode
read-modify-write behavior in the operation or kernel contract. A backend may
decline unsupported overlap or traversal cases, but it must not reinterpret an
overwrite operation as an implicitly aliased update.

Async lowering preserves the same roles:

- an overwrite or update output enrolls one `WriteBuffer`;
- an update reads the old output value through that writer;
- every other tensor input enrolls one `ReadBuffer`;
- no input reader may share the output's `EpochQueue`.

Acquiring both a `ReadBuffer` and `WriteBuffer` for the update output would
create two epochs rather than one read/write operand and can self-block.
`A = f(A)` is therefore represented as a unary update, not as an aliased
overwrite. `A = f(A, B)` is represented as an update of `A` with distinct input
`B`. Read-only inputs may share a queue with one another.

Future AD lowering must also preserve the role distinction. The old value of an
update output is a primal dependency, but it is not a second aliased tensor
operand. If reverse propagation needs that value after mutation, the AD layer
must save or reconstruct it explicitly rather than relying on an input/output
alias hidden inside the kernel call.

### Backend Selection

Default backend selection comes from storage policy and operation type. An
explicit-selector overload is available for backend-sensitive code and tests.
Concrete backends may still decline after mdspans reveal unsupported layout or
accessor details.

`GeneratedStorage` is backend-neutral. `GeneratedTensor` resolves a synthetic,
non-strided `GeneratedLayout`; its offset encoding is an accessor implementation
detail rather than a physical row-major claim. In a mixed operation, concrete
storage selects the backend list. With only generated operands, the host
reference backend is the fallback. Each kernel still decides whether it
understands the generated accessor.

## Core Tensor Operation Support

In the tables below, "Not implemented" means there is no public overload that
accepts `Async<Tensor>` operands. A developer can still await values in an
operation-specific Tensor coroutine and call `co_dispatch_kernel`; a backend
without a coroutine implementation runs its ordinary `try_kernel` inline. The
wrapper must still implement the ownership, buffer, aliasing, and failure
contracts described later in this guide.

| Operation | Synchronous semantics | Allocation or alias behavior | Dedicated Async support |
|---|---|---|---|
| `full`, `zeros`, `ones`, `eye` | Create readable generated tensors. | Compact generator state; no dense element allocation. | No Async-specific overload. |
| `conj(x)` | Read-only lazy semantic view. | Aliases the source; no copy. | `async::conj(x)` returns an owner-retaining alias on the same epoch queue. |
| `reshape_view(x, ...)` | No-copy reshape of a static `layout_left` or `layout_right` source. | Aliases an lvalue source and preserves its canonical layout type and accessor. | `async::reshape_view(x, ...)` returns an owner-retaining alias on the same epoch queue. |
| `reshape_view_left`, `reshape_view_right` | Explicitly ordered no-copy reshape of a general strided source. | Requires a unique, exhaustive canonical mapping in the selected order. | Matching async overloads retain the parent and queue. |
| `reshape_inplace(x, ...)` | Replace a canonical owning tensor mapping at the same compile-time rank. | Keeps the allocation; existing copied descriptors keep their old mapping. | Not implemented. |
| `reshape(x, ...)` | Return an owning reshaped value. | Canonical lvalue copies; canonical owning rvalue may transfer; deferred non-owning and generated inputs materialize before owning reshape. | Not implemented. |
| `copy(out, in)` | Overwrite while observing input accessor semantics. Matching contiguous host/CUDA transfers use the CUDA runtime; same-device positive-strided CUDA mappings with compact rank through eight use 32- or 64-bit logical-index elementwise execution. | Resizes a resizable output or validates a fixed output. CUDA strided copy supports jointly compacted differing layouts, padding, and buffer-view offsets; non-strided or unregistered accessor lowerings decline. | Implemented for CUDA-to-CUDA owning tensors. Pageable host transfers remain blocking and have no Async overload. |
| `assign_transform(out, function, inputs...)` | Nullary or variadic elementwise overwrite through backend dispatch. With no input, the callable generates every output value. The CUDA reference backend registers same-element-type unary and binary arithmetic function objects plus stateful `linalg::scale<Factor>` for positive-strided raw CUDA operands. | A nullary overwrite requires fixed output shape. Other forms resize a resizable output to the first input or validate a fixed output. | Implemented for all-Async Tensor operands. The callable is immediate state owned by the coroutine. |
| `fill(out, value)` | Constant nullary overwrite; the old output value is never read. | Requires fixed output shape and preserves storage. | Implemented for synchronous and Async host tensors. Async fill takes one write epoch. |
| `transform_inplace(out, function, inputs...)` | Variadic elementwise update with the old output as the callable's first argument. | Output shape is fixed and the output appears once as a read/write operand. | Implemented for all-Async Tensor operands with one output writer and distinct input readers. |
| `make_tensor(view)` | Materialize a readable tensor or mdspan as an owning host tensor. | Allocates an inferred runtime-extents owner. | Not implemented. |
| `conjugate_inplace(x)` | Eager element mutation. CPU accessors are evaluated directly; positive-strided CUDA `cfloat` and `cdouble` storage uses the CUDA reference elementwise executor. Real and integer values return as a no-op. | Reuses existing storage and takes one exclusive execution-domain access when work is required. | Not implemented. |
| `sum(input)` | Full reduction returning a same-element-type `ScalarTensor`. | Allocates a rank-zero result in the selected storage domain; an explicit scalar output is also supported. | Implemented; returns `Async<ScalarTensor>` or writes an explicit async scalar output. |
| `sum(input, axes...)` | Remove one or more runtime-selected axes; negative axes are accepted. | Allocates rank `R - sizeof...(axes)`, preserves canonical input layout, and retains surviving-axis order. Explicit outputs may resize. | Implemented; returns an async storage-preserving result or writes an explicit async output. |
| `sum_host(input)` | Full sum returning a C++ scalar. | CPU path writes the host result directly without allocating a scalar tensor. | Implemented; returns `Async<Element>` without blocking submission. |
| `inner_product(lhs, rhs)` | Conjugate-linear-left full reduction returning a `ScalarTensor`. | Allocates a rank-zero result in the selected storage domain; an explicit scalar output is also supported. | Not implemented. |
| `inner_product_host(lhs, rhs)` | Same inner product returning a C++ scalar. | CPU path writes the host result directly without allocating a scalar tensor. | Not implemented. |
| `norm(input)` | Stable Euclidean full reduction returning a real `ScalarTensor`. | Uses scaled sum-of-squares in the CPU reference backend. | Not implemented. |
| `norm_host(input)` | Same Euclidean norm returning a real C++ scalar. | CPU path writes the host result directly. | Not implemented. |
| `require_output`, `prepare_output` | Output-policy helpers for operation authors. | Validate only, or construct/resize/replace when the output type permits it. | Replaceable-output preparation may be provisional across backend decline; fixed and update outputs remain unchanged. |

For a rank-`R` result, generalized `eye<T>(n0, ..., n{R-1})` returns
one exactly when all indices agree. It includes scalar one, an all-ones
rank-one tensor, rectangular identity matrices, and higher-rank tensor-network
copy tensors.

The reshape contracts, including `-1` inference and contiguous-order rules,
are detailed in [Generated Tensors and Reshape](creation_and_reshape.md).

Elementwise transform callables are stored by value in
`linalg::transform_op<F>` or `linalg::transform_inplace_op<F>` and invoked as
const objects. Their arguments are element values: an update receives the old
output value first, followed by its read-only input values. Bare mdspan operands
require an explicit backend selector. Tensor operands derive their selector
from storage policy. The CPU reference backend accepts arbitrary rank and input
arity, respects accessor semantics, and uses logical-index traversal when an
operand is not strided. Named callable types may gain optimized backend
implementations without changing these front-end signatures.

Overwrite and update transforms require the output not to overlap any separate
input operand. Backends may share one allocation access for views proven or
assumed disjoint, but they do not make destructive overlap element-order safe.

Uni20-owned tensor function objects intended for execution through an accessor
use `UNI20_HOST_DEVICE`. A CUDA-accessible transform's stored callable must do
the same. This makes the expression device-callable but does not install a
precompiled CUDA backend specialization; the selected backend still needs an
explicit typed lowering or a sufficient type-erased execution plan.
The named unary `negate`, `square`, and `reciprocal`; stateful `scale<Factor>`;
and binary `add`, `subtract`, `multiply`, and `divide` function objects are
registered this way. Ordinary host code uses the same objects, while the CUDA
reference backend lowers supported scalar combinations to precompiled overwrite
kernels. This initial CUDA set preserves the input element type. Comparisons and
complex-to-real functions such as magnitude require a heterogeneous output
contract and are intentionally separate.

The Async overloads keep the same argument order and require every Tensor
operand to be `Async<T>`. The callable is not itself an async operand: it is
decayed into the operation value and retained by the coroutine. Overwrite may
construct an empty async output from the first input's extents; update requires
an existing output. Input queues may coincide with one another, but the output
queue must be distinct from every input queue. Mutable owner-retaining async
aliases are fixed-shape outputs: the coroutine holds their writer, copies the
bound descriptor locally, and dispatches through that copy without retargeting
the alias.

Async sums normalize and validate runtime axes before task submission because
rank and axis validity do not depend on stored values. Output shape
construction, resizing, and backend dispatch occur after the input epoch is
readable. A value-returning sum creates an independent result epoch;
`sum_host` likewise returns `Async<Element>` and does not call `get_wait()`
internally. Explicit outputs must have a distinct epoch queue from the input.
Mutable async aliases are accepted as fixed-shape outputs and write through
their bound descriptor.

## Dense Linear Algebra Support

These are tensor-front-end operations even though their leaf kernels receive
mdspans or spans.

| Operation | Synchronous contract | Output/storage behavior | Async support |
|---|---|---|---|
| `contract` | Fixed-output pairwise contraction `C = alpha * contract(A, B) + beta * C` over explicit normalized axis pairs. | Caller supplies a shape-compatible output; no resize. Surviving left axes precede surviving right axes. | Not implemented. |
| `gemm` | Fixed-output `C = alpha * A * B + beta * C`. | Caller supplies compatible output; no resize. | Implemented for all-Async tensor operands; `alpha` and `beta` may be immediate or Async. |
| `gemv` | Fixed-output `y = alpha * A * x + beta * y`. | Caller supplies compatible output; no resize. | Not implemented. |
| `assign_product` | Overwrite matrix product. | Output may resize; old values are ignored. | Implemented for all-Async tensor operands; `alpha` may be immediate or Async. |
| `add_product` | Accumulate `output += alpha * lhs * rhs`. | Output must exist and have the required shape. | Implemented as the `beta = 1` forwarding form of async `gemm`; `alpha` may be immediate or Async. |
| `set_matrix` | Set diagonal/off-diagonal values in a selected matrix region. | In-place mutation. | Not implemented. |
| `matrix_norm`, `matrix_norm_host` | Maximum-entry, induced one/infinity, or Frobenius matrix norm selected by `MatrixNorm`. | Returns a storage-preserving real rank-zero Tensor or a host scalar. Complex norms use mathematical magnitude. | Implemented through LAPACK and accessor-respecting CPU backends. |
| `solve_inplace` | Solve `A * X = B` using destructive coefficient and RHS workspaces. | `B` contains `X` on return; `A` contains backend factorization data. Singular provider results are terminal errors. | Implemented through LAPACK and accessor-respecting CPU backends. |
| `solve` | Preserve `A` and `B` and return `X`. | Materializes owning column-major host work matrices before destructive dispatch. | Implemented through `solve_inplace`. |
| `qr_factorization`, `lq_factorization` | Reduced real QR or LQ using a destructive matrix workspace. | Factor outputs may resize; the matrix workspace is overwritten. | Not implemented. |
| `qr(matrix)`, `lq(matrix)` | Preserve a real matrix and return owning reduced factors. | Materializes column-major host work. QR returns `Q: m x k`, `R: k x n`; LQ returns `L: m x k`, `Q: k x n`, where `k = min(m,n)`. | Not implemented. |
| `matrix_exponential` | Compute into a fixed rank-two output. | Caller supplies compatible output. | Not implemented. |
| `self_adjoint_eigh` | Destructive LAPACK-style workspace operation. | Matrix workspace is overwritten; eigenvalue output may resize. | No direct wrapper. |
| `eigh(matrix)` | Preserving value operation returning eigenvalues and eigenvectors. | Materializes work storage and returns two owners. | Implemented; returns two independent Async outputs. |
| `eigh(std::move(matrix))` | Consuming value operation. | May transfer a compatible owning allocation to eigenvectors. | Implemented for `Async<OwningTensor>&&`; consumes the stored value on success. |
| `singular_value_decomposition` | Destructive exact SVD workspace operation. | Matrix workspace is overwritten; `U`, `s`, and `Vh` outputs may resize. | Not implemented. |
| `singular_values(matrix)` | Exact singular values only. | Preserving form materializes work storage; consuming form may destroy a compatible owning input in place. | Preserving and consuming forms return one independent Async output. |
| `svd_left(matrix)` | Exact left singular vectors and singular values. | Reduced by default; full left extent is optional. A consuming reduced call may adopt the input allocation through `JOBU='O'`. | Preserving and consuming forms return two independent Async outputs. |
| `svd_right(matrix)` | Exact singular values and `Vh`. | Reduced by default; full right extent is optional. A consuming reduced call may adopt the input allocation through `JOBVT='O'`. | Preserving and consuming forms return two independent Async outputs. |
| `svd(matrix)` | Exact `U`, `s`, and `Vh`. | Preserving form materializes work storage. Consuming form may adopt one reduced factor, preserving a padded leading dimension; left/right full extents remain independent. | Preserving and consuming forms return three independent Async outputs. |
| `truncated_svd(matrix, policy)` | Truncated reduced `U`, `s`, and `Vh` plus `SvdTruncationInfo`. | Applies minimum/maximum rank, absolute singular-value, normalized squared singular-value, and discarded-weight criteria. Rank zero is valid. Consuming exact-SVD workspace reuse is permitted, but returned factors are right-sized owners. | Preserving and consuming forms return four independent Async outputs. |
| `nonsymmetric_eigen` | LAPACK-style nonsymmetric eigensystem. | Destructive matrix workspace and caller-provided outputs. | Not implemented. |
| `schur`, `hessenberg_schur` | LAPACK-style decomposition. | Destructive matrix workspace and caller-provided outputs. | Not implemented. |
| `reorder_schur` | Reorder an existing Schur form. | In-place mutation of Schur form and optionally vectors. | Not implemented. |
| `symmetric_tridiagonal_eigen` | Tridiagonal eigensystem over caller-provided spans and tensor output. | LAPACK-style mutable work/output buffers. | Not implemented. |

Preserving Async `eigh`, exact SVD, and truncated SVD overloads accept
`Async<Tensor>` whenever `Tensor` models the appropriate ranked `TensorView`.
This includes deferred views: the preserving synchronous operation materializes
its work tensor through ordinary backend-dispatched copy. Consuming overloads
require a mutable owning `TensorView`, but not immediate access. Taking the
stored async value is the semantic consumption; compatible immediate owners may
additionally reuse or transfer their allocation as an optimization.

Async tensor wrappers constrain fixed operands at the `TensorView` or
`MutableTensorView` level. They must not require `ImmediateTensorView` merely
because the currently selected backend eventually needs an mdspan. After the
stored values become readable, the wrapper preserves their mdspecs through
dispatch and the selected backend acquires the execution-domain leases.
Whether a backend implementation exists for a particular storage domain is a
separate kernel-acceptance question.

The fixed-output `gemm` and `gemv` forms are low-level tensor front ends. New
ordinary application code should prefer operation-specific overwrite, update,
or value-returning APIs where those exist because their output policy is
explicit.

`assign_product` and `gemm` use distinct dispatch operations:

```cpp
auto lhs_span = mdspec_of(std::as_const(lhs));
auto rhs_span = mdspec_of(std::as_const(rhs));
dispatch_kernel(
    selector, assign_product_op{}, output, alpha, lhs_span, rhs_span);

auto output_span = mdspec_of(output);
dispatch_kernel(
    selector, gemm_op{}, output_span, alpha, lhs_span, rhs_span, beta);
```

`assign_product_op` is a replaceable-output overwrite. The previous output value
is irrelevant, so an output type that supports replacement may change its shape
or storage. `gemm_op` is a fixed-output update: its output must already exist and
its storage is never rebound. This remains true when `beta` is numerically zero.
Dispatch does not inspect `beta` to infer overwrite permission because a backend
may represent scalar parameters with opaque or device-resident handles.

Backends may lower both operation tags to the same provider GEMM implementation.
The `assign_product_op` adapter supplies zero in the representation required by
that backend; `gemm_op` forwards the caller's `beta`. Shape and storage
preparation occur in the `assign_product_op` backend adapter, where the backend
has enough information to state its storage requirements. The synchronous form
receives a concrete Tensor output. The async form receives the output's
potentially unconstructed `shared_storage<Tensor>`.

The frontends select their backend lists while tensor policy is still
available, then normalize fixed operands exactly once before dispatch.
`gemm_op` backend customizations therefore accept
`MutableMdspecLike` and `MdspecLike` refinements, not tensor views.
`assign_product_op` receives normalized readable inputs but retains its
tensor-level output because that output may not exist until the selected
backend supplies placement and storage requirements.

Host backends require only the product shape. The cuBLAS backend also requires
the output to use the operands' CUDA device, after first rejecting operands that
reside on different devices. This requirement is a `cuda::Device`, not a stream
pool or `DeviceResources`; `CudaStorage` resolves the device through the
process-wide CUDA runtime when it must allocate a new buffer. A matching
allocation is retained, a shape mismatch on the same device preserves its
existing storage resources, and a device mismatch replaces the allocation.
Operands that reside on different CUDA devices produce
`KernelAttempt::incompatible_devices` before output preparation.

For `assign_product_op`, the BLAS backend resolves the readable input mdspans
and probes the prospective output metadata before calling `prepare_output`.
`unsupported_instance`, `unsupported_layout`, and `unsupported_transform`
therefore leave a wrong-shaped concrete output unchanged and leave deferred
output storage unconstructed. After the probe succeeds, BLAS prepares the real
output and is committed to execution.

The cuBLAS `assign_product_op` adapter uses the provisional-preparation
contract instead: it may prepare the actual CUDA output and then decline an
unsupported layout or transform. A later CUDA backend receives that prepared
output and may reuse or replace it. Within one cuBLAS attempt, the readable
input descriptors have already been normalized by the frontend and the writable
output is normalized once after preparation. The adapter then invokes the
private cuBLAS mdspan implementation directly rather than redispatching those
operands as `gemm_op`.

For `CudaTensor`, `gemm`, `assign_product`, and `add_product` retain their
Tensor epoch buffers while awaiting `co_dispatch_kernel`. The cuBLAS backend's
`try_make_kernel_task` returns a nested `CudaTask` bound to the operand device. It
inherits the compatible unified scheduler, awaits an idle cuBLAS handle and
stream, and publishes CUDA buffer completion records before returning. Column-
and row-major outputs are supported. A direct non-Async Tensor `gemm` uses the
same operand preparation and provider leaf but may block during resource
admission. Async `gemm` and `add_product` require an existing compatible output;
`assign_product` uses its distinct replaceable-output dispatch operation and may
prepare an unconstructed or resizable output.

## Async Tensor Contract

### All Tensor Operands Are Async

An Async overload does not mix synchronous and asynchronous tensor operands:

```cpp
uni20::async::Async<uni20::DenseMatrix<double>> lhs = /* ... */;
uni20::async::Async<uni20::DenseMatrix<double>> rhs = /* ... */;
uni20::async::Async<uni20::DenseMatrix<double>> output;

uni20::linalg::assign_product(output, lhs, rhs);
```

Ordinary scalar/configuration values are passed by value. An operation may
explicitly allow an `Async<Scalar>`; matrix-product `alpha` does. Borrowed
objects such as spans and string views remain borrowed even when their
descriptor is passed by value and require separate lifetime analysis.

### Async Is a Scheduling Wrapper

The wrapper resolves the immutable selector from tensor/storage types before
scheduling, enrolls buffers, and moves all coroutine state into a named
coroutine or captureless static coroutine lambda. The coroutine awaits stored
Tensor values and invokes backend dispatch. Runtime backend declines still
happen after the Tensor values are available.

Do not make `Async<Tensor>` model `TensorView`, and do not add Async awareness
to leaf backends.

### Overwrite and Update Outputs

For async `assign_product`, backend dispatch receives the output's
`shared_storage<Tensor>`. The selected backend may construct an unconstructed
output or prepare an existing one through `prepare_output`.

For async `add_product`, output is both input and output. It uses one
`WriteBuffer`, must already be constructed, and cannot resize. Taking separate
read and write buffers for the same timeline is incorrect.

Independent-value async write-proxy assignment follows the stored type. The
first write constructs uninitialized storage when possible; later writes invoke
the stored type's assignment operator. Alias proxies instead expose their
descriptor read-only and route supported assignment through `assign_through`.
Use explicit `emplace` for value reconstruction and named tensor operations for
backend-dispatched element evaluation.

Direct assignment to the outer `Async<T>` handle is a separate decision.
Independent owning tensors detach onto fresh storage and a fresh queue. Mutable
async tensor aliases retain the parent's owner and queue and obtain
write-through assignment from `MutableTensorView` plus the tensor
`assign_through` customization. Const and conjugating async views are not
assignable. Exact `Async<View>` assignment follows the same rule: mutable views
write through and read-only views reject the expression.

### Async Views and Lifetime

`async::conj(parent)` returns `Async<ConjugatedTensorView<...>>` for complex
tensors and an async const identity view for real tensors. The alias owns its
descriptor, retains the parent storage control block, and shares the parent's
exact `EpochQueue`.

`async::reshape_view(parent, extents...)` returns an owner-retaining structural
alias for a statically canonical layout. The explicit `reshape_view_left` and
`reshape_view_right` overloads select an order for a general strided parent. An
alias can be created before the parent value is constructed; shape and layout
validation occurs when the shared parent epoch is first readable. A mutable
parent produces a mutable write-through alias, while a const parent produces a
read-only alias.

Nested async views retain the complete descriptor-owner chain and use one
shared timeline. Reshape descriptors preserve the selected order in their
static `layout_left` or `layout_right` result type. No runtime order metadata
needs to be forwarded through semantic views such as conjugation.

Future structural aliases such as slices must use the same owner-retaining
mechanism. Wrapping a raw synchronous view in a fresh `Async` would create an
independent queue for aliased bytes and is incorrect.

### Multiple Outputs and Consuming Inputs

Async `eigh` returns independent handles:

```cpp
auto [eigenvalues, eigenvectors] = uni20::linalg::eigh(matrix);
```

This avoids an extraction coroutine and allows either result to flow directly
into downstream work. Every output writer is an exception sink.

The consuming overload accepts `Async<OwningTensor>&&`, enrolls the input as a
writer, and removes its stored value with the consuming buffer API. On success,
the input timeline no longer contains a readable matrix. The writer remains
gated until both outputs are committed so a failure can reach the input and
both output epochs consistently.

### Failures

An early backend decline is not a failure and must have no externally visible
side effects. Once a synchronous operation fails to produce its result, the
async task fails rather than trying another backend.

Unhandled coroutine exceptions propagate to every output `WriteBuffer` passed
to the task. Reads of those output epochs observe the failure. Synchronous
`CHECK` and `PANIC` still identify internal logic errors and may terminate the
process; bindings must validate user input before entering such paths.

## Adding an Async Operation

Before adding an Async overload:

1. Implement and test the synchronous Tensor operation first.
2. Classify every output as overwrite, update, allocating, or consuming.
3. Decide whether outputs are one `Async<Result>` or several independent Async
   values. Prefer independent handles when downstream operations use results
   separately.
4. Resolve the default selector from Tensor/storage types before scheduling.
5. Reject obvious output/input queue aliasing before buffer enrollment.
6. Use one `WriteBuffer` for each mutated timeline and `ReadBuffer` for each
   distinct read-only Tensor input. An update output obtains its old value
   through that writer and is never enrolled again as an input.
7. Pass selectors, options, and other ordinary state into the coroutine by
   value. Never use a capturing coroutine lambda; scheduled coroutine lambdas
   must be `static`.
8. Await Tensor values before resolving mdspans, then call the synchronous
   Tensor operation.
9. Ensure every output writer receives task failures.
10. Test pending-input lifetime, unconstructed outputs, aliases, numerical
    behavior, and exception propagation.

Keep wrappers operation-specific until several more implementations establish
a genuinely reusable output and lifetime pattern.

## Current Gaps

The following are intentionally not implied by the current API:

- no Async `copy` or `make_tensor`
- no Async general slice alias
- no Async `gemv`, matrix exponential, or general LAPACK workspace operation
- no allocating value API for most destructive LAPACK front ends
- no general concrete synchronous `TensorRef` slice proxy
- no subrange epoch tracking; async aliases conservatively share a whole parent
  queue
- no implicit dense fallback for symmetry-aware tensors

These are implementation targets, not compatibility promises. New work should
preserve the semantic categories in this document while choosing the clearest
API for the operation.

## Related Documents

- [Generated Tensors and Reshape](creation_and_reshape.md)
- [Kernel Dispatch Design](../architecture/kernel_dispatch.md)
- [Backend Dispatch Design](../architecture/backend_dispatch.md)
- [Async Tensor Kernel Authoring](../async/kernel_authoring.md)
- [Async Runtime Model](../async/runtime_model.md)
- [Async Buffers and Awaiters](../async/buffers_and_awaiters.md)
- [Async Tensor Lifetime and Dispatch Draft](../async/tensor_lifetime_and_dispatch_draft.md)
- [Tensor Dispatch and View Semantics Draft](dispatch_and_view_semantics_draft.md)
