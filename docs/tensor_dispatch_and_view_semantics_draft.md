# Tensor Dispatch And View Semantics Draft

**Status:** design notes for review. This is not a finalized API contract.

This note summarizes the current discussion about dense tensor/view semantics,
output assignment, and how front-end tensor operations relate to backend kernel
dispatch. The goal is to make the tentative design reviewable before changing
the core tensor API.

Related notes:

- [`kernel_dispatch.md`](kernel_dispatch.md) - backend lists, operation tags,
  and async/lowering layers.
- [`backend_dispatch.md`](backend_dispatch.md) - compile-time capability,
  runtime attempt, and fallback behavior.
- [`async_storage.md`](async_storage.md) - async write assignment semantics.
- [`async_tensor_lifetime_and_dispatch_draft.md`](async_tensor_lifetime_and_dispatch_draft.md)
  - async tensor aliases, lifetimes, and async dispatch lowering.
- [`block_tensor.md`](block_tensor.md) - block tensor storage and per-block
  view/lowering model.

## Core Distinction

The design is converging on a strict layer split:

1. **Front-end tensor operations** accept actual Uni20 tensor-like objects. They
   need storage policy, output policy, synchronous metadata, async/block
   structure, and a default backend selector.
2. **Backend dispatch** selects a backend from an operation tag plus a backend
   selector derived from tensor storage policy.
3. **Leaf kernels** operate on resolved mdspan-like views. At this level the
   backend has already been selected, and the only remaining job is to run the
   kernel on handles, extents, strides, and accessors.

A plain `stdex::mdspan` or structural `SpanLike` is sufficient for a leaf kernel
but is not sufficient for top-level Uni20 dispatch. It does not carry the
storage policy needed to derive the default backend stack.

For local dense tensors, a useful model is:

```cpp
dispatchable dense tensor = SpanLike + backend selector
```

The backend selector may be derived from owning storage (`BasicTensor`) or
carried by a non-owning adaptor. A bare mdspan remains only `SpanLike`; it
becomes dispatchable only when an explicit backend selector is supplied by the
call or by an adapter.

## Tentative Type Roles

The current semantic split is:

```cpp
BasicTensor  // owning dense tensor value
TensorRef    // non-owning tensor lvalue proxy, such as a slice or block view
mdspan       // shallow non-owning descriptor; assignment rebinds descriptor
adaptors     // unnamed concept-modeling wrappers, e.g. as_tensor(vector)
```

### `BasicTensor`

`BasicTensor` is the owning dense tensor type. Assignment has value/replace
semantics:

```cpp
BasicTensor C;
C = rhs; // replace C's logical tensor value
```

An implementation may reuse existing storage when shape/layout are compatible,
or reallocate/rebuild the owned storage when needed. This is value assignment,
not strict write-through assignment.

### Resolved Views

A resolved view is a shallow descriptor: data handle plus
mapping/extents/accessor. This can often be a plain `stdex::mdspan` or a small
mdspan-like object. Assignment of a resolved view should have mdspan semantics:

```cpp
auto v = tensor_view(A);
v = tensor_view(B); // rebind v's descriptor; no element assignment
```

The design may not need a central C++ `TensorView` class at all. For C++ leaf
kernels, a copied mdspan-like descriptor plus a separate backend selector/domain
may be enough. A `TensorView` class may still be useful as a named adaptor type,
for Python bindings, or for cases where mdspan alone cannot carry enough
metadata ergonomically.

For dense CUDA data, the tensor/adaptor object must carry enough dispatch
information to identify device memory. At minimum that means a storage/backend
domain and likely a device ordinal or equivalent placement record. Dense MPI
distribution is not a near-term requirement for dense tensors; dense tensors are
expected to be local to one node, while block/MPI distribution belongs to
`BlockTensor` storage policy.

### `TensorRef`

`TensorRef` is the proposed write-through non-owning tensor lvalue proxy:

```cpp
auto r = C.slice(...);
r = rhs; // write through into C's slice
```

It should expose or produce a resolved mdspan-like view for kernels, but its own
assignment operator writes elements into the referenced region. Shape must
already be compatible. `TensorRef` is the likely type for slices, block mutation
targets, and other fixed non-owning outputs.

An explicit adaptor such as `as_tensor(std::vector<T>&)` does not need to be a
`TensorRef`. Since dispatch uses concepts, user code does not name the adaptor
type. The adaptor can have semantics tailored to the wrapped object.

## Assignment Semantics

The async assignment trait currently distinguishes rebind and write-through.
The tensor design suggests three semantic contracts:

```cpp
BasicTensor -> value/replace
mdspan/view  -> rebind descriptor
TensorRef    -> write-through
```

For async write proxies, this likely means:

- `BasicTensor`: if unconstructed, emplace from the right-hand side; if
  constructed, call `BasicTensor::operator=`, which may reuse or reallocate.
- A bare mdspan-like descriptor: if stored async, assignment/emplace copies or
  rebinds the descriptor; it does not copy elements.
- `TensorRef`: require an already-constructed reference/proxy target; assignment
  writes through to the existing tensor region.
- Explicit adaptor types choose the semantic contract of the wrapped object.
  `as_tensor(std::vector<T>&)` may support resizing and element assignment,
  while a slice adaptor should remain fixed/write-through.

Open naming question: whether `assignment_semantics` should gain a third value
such as `value`, or whether the async storage policy should split "what
assignment means" from "what to do when storage is unconstructed".

## Output Parameters

There are two broad output modes.

Both can be expressed through one customization point:

```cpp
ensure_shape(out, shape);
```

For an owning/resizable tensor such as `BasicTensor`, `ensure_shape` may reuse
existing storage or reallocate/rebuild the tensor to match `shape`. For a
non-reallocatable output such as `TensorRef`, a block view, or a resolved
mdspan-like output, `ensure_shape` validates that the current shape already
matches and otherwise throws/asserts. For an adaptor such as
`as_tensor(std::vector<T>&)`, the adaptor decides: it may resize the vector, or
it may expose fixed/write-through semantics. This lets front-end kernels use one
output-preparation step while the output type controls whether resizing is legal.

`ensure_shape` must run before materializing the output write view. For a
resizable tensor it may reallocate and invalidate any previous mdspan-like
object referring to the old storage.

### Fixed Output

A fixed output cannot reallocate parent storage. It validates shape and then
produces a mutable view:

```cpp
ensure_shape(C.view(), shape);      // validate only
ensure_shape(C.slice(...), shape);  // validate only
gemm_into(A, B, C.view());      // C must already have the right shape
gemm_into(A, B, C.slice(...));  // slice shape must already match
```

This is the natural mode for `TensorRef`, fixed adaptors, block views, and BLAS
update forms where the output is also an input.

### Resizable Output

A resizable output can prepare the requested shape by reuse or allocation:

```cpp
ensure_shape(C, shape); // may resize BasicTensor
matmul(A, B, C);   // C is a BasicTensor and may be resized
auto C = matmul(A, B);
```

Classic BLAS-style `gemm(alpha, A, B, beta, C)` is more subtle. When `beta != 0`,
`C` is both input and output, so a fixed-output/update API is the clearer
default. Reallocation is natural for overwrite/value-producing operations such
as `C = A * B`, `matmul(A, B, C)`, or `beta == 0` APIs that explicitly define a
fresh output.

## Examples Driving Concepts

The concept names should be derived from call sites. These examples are the
current target surface.

### Owning Assignment

```cpp
BasicTensor C;
C = A + B;
C = tensor_view(rhs);
```

Needed behavior:

- infer the right-hand-side shape
- call `ensure_shape(C, shape)`
- evaluate into `tensor_write_view(C)`

Concept pressure:

- `TensorOutput<C, Shape>` for `ensure_shape` plus writable view
- optional `tensor_can_resize_v<C>` for code that needs to know whether resize is
  legal

### Value-Producing Operation

```cpp
matmul(A, B, C);   // C may be resized
auto C = matmul(A, B);
```

Needed behavior:

- `A` and `B` provide synchronous metadata and read views
- `C` provides `ensure_shape` and a write view
- backend selector is computed from all participating tensor operands

Concept pressure:

- `TensorReadable<A>`
- `TensorReadable<B>`
- `TensorOutput<C, Shape>`
- `TensorBackendSelectable<T>` or equivalent requirement on all dispatch
  operands

### Fixed Update Operation

```cpp
gemm(alpha, A, B, beta, C.view());
gemm(alpha, A, B, beta, C.slice(...));
```

Needed behavior:

- `C` is both input and output when `beta != 0`
- `ensure_shape(C, shape)` validates only
- write view must be materialized after `ensure_shape`

Concept pressure:

- `TensorReadWrite<C, Shape>` or `TensorOutput<C, Shape> + TensorReadable<C>`
- fixedness may be a trait such as `!tensor_can_resize_v<C>` rather than a
  separate concept

### Explicit Backend On Mdspan

```cpp
gemm(alpha, a_mdspan, b_mdspan, beta, c_mdspan,
     backend_list{BlasBackend{}, CpuGenericBackend{}});
```

Needed behavior:

- mdspans provide leaf-kernel view information
- the explicit backend selector supplies the dispatch information missing from
  bare mdspan

Concept pressure:

- `SpanLike` / `StridedMdspan` remains the leaf-kernel concept
- top-level dispatch may accept `SpanLike` operands only when a backend selector
  is explicit

### Vector Adaptor

```cpp
std::vector<double> x;
auto X = as_tensor(x);              // rank-1 host tensor adaptor
auto X = as_tensor(x, HostDomain{}); // explicit domain/backend selector
```

Needed behavior:

- `X` is an unnamed concept-modeling adaptor
- `tensor_view(X)` returns a rank-1 mdspan-like view
- `tensor_backend_selector(X)` defaults to a CPU/host selector unless overridden
- `ensure_shape(X, shape)` may resize the vector because this adaptor explicitly
  owns that policy

Concept pressure:

- adaptor types can choose fixed or resizable behavior
- raw `std::vector<T>` should not automatically model the tensor concept

### Slice Or Block Output

```cpp
auto S = C.slice(...);
S = rhs;
gemm_into(A, B, S);
```

Needed behavior:

- `S` is a non-owning fixed output
- assignment writes through
- `ensure_shape(S, shape)` validates only
- `tensor_write_view(S)` exposes a resolved mdspan-like view

Concept pressure:

- `TensorRef` remains useful as a named non-owning tensor lvalue type
- fixed write-through behavior is distinct from vector-adaptor behavior

### Async Or Block Tensor

```cpp
block_gemm(C, A, B);
```

Needed behavior:

- planner reads synchronous descriptor metadata
- per-block read/write handles resolve to mdspan-like views after awaiting
- backend selector may be container-level (`Mpi`) and recurse into local dense
  selectors

Concept pressure:

- descriptor and view access may be two-level for async/block storage
- dense local concepts should not assume all tensor structure is a single
  mdspan

### Backend-Compatible Temporary

```cpp
auto tmp = make_temporary_like(A, shape);
copy(A, tmp);
```

Needed behavior:

- allocate scratch/output storage that is compatible with the selected backend
- preserve value type, rank, layout requirements, memory domain, and device
  placement
- support the same path when `A` is only an mdspan-like view plus explicit
  backend/domain information
- allow the backend selector value to carry composed runtime state, such as
  CUDA device, workspace allocator, stream policy, or advisory hints
- treat population of the temporary as an ordinary copy kernel

Concept pressure:

- backend selection alone is not enough to allocate storage; the dispatch layer
  also needs a memory domain or storage factory
- the backend selector must be a value, not only a backend tag list, when
  placement or allocation policy is runtime state
- `TensorBackendSelectable` should stay separate from a possible
  `TensorStorageDomain` / `TensorTemporaryFactory` concept
- an explicit mdspan adapter may need to carry both backend selector and memory
  domain if temporaries are required

## Candidate Concept Names

The naming should distinguish dispatch participation from mdspan-like leaf
views.

```cpp
template <class T>
concept TensorBackendSelectable =
  requires(T const& t) {
    tensor_backend_selector(t);
  };

template <class T>
concept TensorStorageDomain =
  requires(T const& t) {
    tensor_storage_domain(t); // memory kind, device ordinal, allocator/resource
  };

template <class T>
concept TensorDescribed =
  requires(T const& t) {
    tensor_descriptor(t); // shape/layout/domain metadata, sync
  };

template <class T>
concept TensorReadable =
  TensorDescribed<T> &&
  TensorBackendSelectable<T> &&
  TensorStorageDomain<T> &&
  requires(T const& t) {
    tensor_read_view(t); // mdspan-like view or awaitable handle
  };

template <class T, class Shape>
concept TensorOutput =
  TensorDescribed<T> &&
  TensorBackendSelectable<T> &&
  TensorStorageDomain<T> &&
  requires(T&& t, Shape const& shape) {
    ensure_shape(t, shape);
    tensor_write_view(t); // materialized after ensure_shape
  };

template <class T, class Shape>
concept TensorReadWrite =
  TensorReadable<T> && TensorOutput<T, Shape>;
```

For local dense tensors, a narrower concept may be useful:

```cpp
template <class T>
concept DenseTensor =
  SpanLike<T> &&
  TensorBackendSelectable<T>;
```

Naming alternatives:

| Candidate | Meaning | Concern |
|---|---|---|
| `TensorOperand` | any top-level dispatch operand | too vague unless split into read/write |
| `TensorReadable` | dispatchable tensor read input | clear for inputs |
| `TensorOutput` | output that supports `ensure_shape` and write view | hides fixed vs resizable behind behavior |
| `TensorReadWrite` | update operand, both readable and writable | useful for BLAS `beta != 0` |
| `DenseTensor` | local dense `SpanLike + backend selector` | should not imply block/MPI support |
| `TensorStorageDomain` | memory domain for allocation and placement | name may sound too much like ownership |
| `TensorTemporaryFactory` | can allocate compatible scratch/output storage | may be a CPO/trait rather than a concept |
| `TensorLike` | broad umbrella | probably too imprecise for constraints |

## Dispatch Interface And Adaptors

The top-level dispatch API should use concepts plus an adaptor/customization
layer rather than base-class conversion to `TensorView`.

The customization layer can be implemented as free CPOs that call member
functions when present. Uni20's own tensor and adaptor classes can therefore use
ordinary members for locality and documentation, while external types can still
opt in through wrappers or ADL/customization.

For example:

```cpp
tensor_view(x);              // may call x.view()
tensor_backend_selector(x);  // may call x.backend_selector()
tensor_descriptor(x);        // may call x.descriptor()
```

The important point is that top-level dispatch is written against the concept,
not against inheritance from `TensorView`.

Possible CPO shape:

```cpp
tensor_descriptor(x);
tensor_backend_selector(x);
tensor_storage_domain(x);
tensor_read_view(x);
ensure_shape(out, shape);
tensor_write_view(out);
make_temporary_tensor(selector, domain, descriptor);
```

Whether an output is fixed or resizable may be a separate semantic trait, for
example `tensor_can_resize_v<T>`, rather than a separate concept. Most front-end
kernels can simply call `ensure_shape`; APIs that specifically forbid
reallocation can constrain on the fixed-output trait or pass a fixed view/ref.

Then a value-producing operation can be structured as:

```cpp
template <class A, class B, class C>
void matmul(A const& a, B const& b, C& c)
{
  auto selector = common_backend_selector(a, b, c);
  auto shape = matmul_shape(tensor_descriptor(a), tensor_descriptor(b));

  ensure_shape(c, shape);
  dispatch_kernel(backend_list_t<decltype(selector)>{}, matmul_op{}, a, b, c);
}
```

The selected backend eventually lowers operands to views:

```cpp
auto av = tensor_read_view(a);
auto bv = tensor_read_view(b);
auto cv = tensor_write_view(c);
try_kernel(Backend{}, matmul_op{}, av, bv, cv);
```

This keeps `SpanLike` at the leaf-kernel level and keeps backend selection at
the tensor/storage level.

For local dense tensors, the concept may collapse to:

```cpp
dispatchable_dense_tensor = SpanLike + tensor_backend_selector
```

In that case `tensor_view(x)` can simply return `x` or a shallow wrapper around
`x`, provided the backend selector is also available. This lets a `BasicTensor`
or explicit tensor adaptor satisfy mdspan-like algorithms directly without
making `SpanLike` alone sufficient for Uni20 dispatch.

## Temporaries

Kernel front ends need to create temporaries in all modes: synchronous dense
tensors, async tensors, block tensors, and direct mdspan entry points. This is
not special assignment semantics; it is allocation plus a copy/evaluation
kernel.

The temporary type should be selected from the backend selector/list value and
its storage domain, not from the C++ name of the operand:

```cpp
auto selector = common_backend_selector(a, b);
auto domain = temporary_storage_domain(selector, a, b);
auto tmp = make_temporary_tensor(selector, domain, descriptor);
copy(a, tmp); // ordinary backend-dispatched copy kernel
```

For Uni20-owned tensors, the domain comes from the storage policy:

```cpp
BasicTensor<T, HostStorage> -> HostDomain{allocator/resource}
BasicTensor<T, CudaStorage> -> CudaDomain{device, allocator/resource}
```

For non-owning views/adaptors, the domain must be carried explicitly or be
recoverable from the adaptor:

```cpp
auto A = as_tensor_view(a_mdspan, HostDomain{});
auto B = as_tensor_view(b_mdspan, CudaDomain{device});
```

The resulting concrete temporary type can be internal and unnamed by user code.
It might be `BasicTensor<T, HostStorage>` for host domains, a future
`BasicTensor<T, CudaStorage>` for CUDA domains, or an async/block scratch handle
when the caller is already in an async/block path. The front-end algorithm
should only rely on the returned object satisfying the tensor output/read-write
concepts.

Backend selector state and storage domain are related but not identical. A
backend list says which kernels may run. A storage domain says where bytes live
and how to allocate more of them. In simple cases `[Blas, CpuGeneric]` implies
host memory and `[Cublas, DeviceGeneric]` implies CUDA device memory, but
relying on that equivalence loses information:

- a raw `double*` mdspan plus a stateless `CublasBackend{}` does not identify
  the CUDA device unless the pointer/accessor type, adaptor, or backend
  selector state carries it
- pinned host, pageable host, unified memory, and device memory may all have
  different allocation policies even when the same fallback CPU kernel could
  read them
- an explicit backend override is a compute policy, not necessarily permission
  to move storage domains silently

Therefore direct mdspan calls that need temporaries should either use an
adapter carrying a storage domain or pass an explicit domain/factory:

```cpp
gemm(a_mdspan, b_mdspan, c_mdspan,
     backend_list{BlasBackend{}, CpuGenericBackend{}},
     HostDomain{});

auto A = as_tensor_view(a_mdspan, CudaDomain{device});
auto tmp = make_temporary_like(A, descriptor);
```

The copy into or out of such a temporary is itself a backend-dispatched kernel.
That lets staging be explicit in the dispatch model: a backend may decline
because operands are in incompatible domains, a staging backend may allocate a
temporary in the target domain and call copy kernels, and then the ordinary
compute backend can run on the staged views.

Temporary allocation should normally happen after a backend has accepted the
operation, or inside a staging backend that has accepted responsibility for the
operation. That keeps the fallback contract clean: a backend that has not yet
committed should not allocate scratch for a path it may later decline. Runtime
feasibility checks should happen before allocation where possible; once a
backend allocates and starts copies or writes, it is committed and later failure
is an error rather than a fallback.

## Backend Selection

Backend selection belongs to tensor/storage operands:

- Dense host tensor: likely `[Blas, CpuGeneric]`.
- Dense device tensor: likely `[Cublas, DeviceGeneric]`.
- Block/MPI tensor: likely a container-level backend such as `[Mpi]`, which
  recurses into local dense backends per block or coalesced group.

For dense local tensors, `SpanLike + backend selector` may be sufficient. The
selector is the extra information that distinguishes a host span from a CUDA
device span. A raw mdspan has no such selector; a `TensorView` over CUDA memory
must carry it explicitly.

Storage and backend selection should remain separate concepts. A storage policy
owns or describes how bytes are stored, and one of its traits is the default
backend selector. A non-owning adaptor may have no storage object at all, but it
still needs a backend selector or storage-domain descriptor. For example:

```cpp
BasicTensor<T, HostStorage> -> backend_list{BlasBackend{}, CpuGenericBackend{}}
TensorAdaptor<T, HostDomain> -> backend_list{BlasBackend{}, CpuGenericBackend{}}
TensorAdaptor<T, CudaDomain> -> backend_list{CublasBackend{device}, CudaGenericBackend{device}}
```

For CUDA views, the selector/domain must distinguish device memory from host
memory and carry or recover the device ordinal. The view cannot rely on owning
storage to provide that information.

The selector can be overridden with either an ordered backend-list value or a
singleton backend value:

```cpp
gemm(alpha, A, B, beta, C);                                  // storage default
gemm(alpha, A, B, beta, C, CpuGenericBackend{});              // one backend only
gemm(alpha, A, B, beta, C,
     backend_list{BlasBackend{}, CpuGenericBackend{}});       // ordered list
gemm(alpha, A, B, beta, C,
     make_backend_selector<backend_list<CublasBackend, CudaGenericBackend>>(
       CublasConfig{.device = {1}, .stream = {stream}, .math_mode = {tf32_allowed}}));
```

A singleton backend is normalized to a one-entry backend list value. It does
not get an implicit fallback.

Plain mdspan entry points should require an explicit backend selector:

```cpp
gemm(alpha, a_mdspan, b_mdspan, beta, c_mdspan,
     backend_list{BlasBackend{}, CpuGenericBackend{}});
```

or an explicit backend helper:

```cpp
BlasBackend::gemm_or_throw(alpha, a_mdspan, b_mdspan, beta, c_mdspan);
```

The reason is that mdspans do not carry Uni20 storage policy.

The term `backend_list<...>` should be read as a shorthand for the static
ordering of backend entries. The actual selector passed through dispatch should
be a value, but backend entries can be stateless tags. Runtime state can be
composed from type-level state requirements declared by those tags:

```cpp
struct Device { int value; };
struct Stream { cudaStream_t value; };
struct CublasMathMode { math_mode_t value; };
struct CpuThreads { thread_pool_t* value; };

struct CublasBackend {
  using state = std::tuple<Device, Stream, CublasMathMode>;
};

struct CudaGenericBackend {
  using state = std::tuple<Device, Stream>;
};

struct CpuGenericBackend {
  using state = std::tuple<CpuThreads>;
};
```

For a selector such as:

```cpp
backend_list<CublasBackend, CudaGenericBackend>
```

the selector state is the unique concatenation of all backend state lists:

```cpp
using state_t =
  unique_tuple_cat_t<CublasBackend::state, CudaGenericBackend::state>;
// std::tuple<Device, Stream, CublasMathMode>

state_t state;
```

Each runtime state component is stored once. The backend tags remain the ordered
candidate list, and dispatch obtains state by tag:

```cpp
auto device = std::get<Device>(state).value;
auto stream = std::get<Stream>(state).value;
auto math_mode = std::get<CublasMathMode>(state).value;
```

`std::get<T>(tuple)` requires `T` to appear exactly once in the tuple. This is
a useful constraint: duplicate state tags are intentionally collapsed by
`unique_tuple_cat_t`, and two semantically different state values must use
different tag types.

For a hypothetical dense MPI-striped tensor, the structural context could be
more detailed than a CUDA device: communicator, rank-local ownership, global
shape, and a section map describing which tensor regions live on which ranks.
Those pieces can be represented as global state tags as well, for example
`MpiCommunicator`, `RankOwnership`, and `TensorSectionMap`.

The tags are global names, so they need deliberate namespacing and semantics:
`cuda::Device`, `cuda::Stream`, `cublas::MathMode`,
`cutensornetwork::WorkspaceLimit`, and so on. This is manageable because
backends in one fallback chain are already mutually aware at the integration
boundary.

Inheritance is still possible, but it should only help declare or share
type-level requirements. It should not define the backend order. Simple
inheritance such as `CuSolverBackend : CudaBackend` is too strong: it says a
cuSOLVER backend *is* the CUDA generic backend, and it risks coupling capability
inheritance to fallback ordering. The ordered backend list should remain the
only thing that says "try cuTensorNetwork, then cuBLAS, then generic CUDA."

An empty base can be a spelling for state requirements:

```cpp
template <class... State>
struct requires_state {
  using state = std::tuple<State...>;
};

struct CublasBackend : requires_state<cuda::Device, cuda::Stream, cublas::MathMode> {};
struct CudaGenericBackend : requires_state<cuda::Device, cuda::Stream> {};
```

A CRTP family base is another possible spelling:

```cpp
template <class BackendTag, class Derived>
struct Backend;

template <class Derived>
struct Backend<CublasTag, Derived>
    : requires_state<cuda::Device, cuda::Stream, cublas::MathMode> {};
```

That can factor common type-level traits or helper APIs across related backends,
but it should still be used only to express requirements/capabilities. It should
not imply that `CublasBackend` inherits from `CudaGenericBackend` or that
`CudaGenericBackend` is the next fallback.

A construction convenience may still be useful for the initial implementation.
For example, a user could provide a richer backend configuration object, and the
selector can decompose it into state tags:

```cpp
struct CublasConfig {
  cuda::Device device;
  cuda::Stream stream;
  cublas::MathMode math_mode;
};
```

Then:

```cpp
auto selector = make_backend_selector<backend_list<CublasBackend, CudaGenericBackend>>(
    CublasConfig{.device = {0}, .stream = {stream}, .math_mode = {tf32_allowed}});
```

can populate the shared `backend_state_store<Device, Stream, CublasMathMode>`.
This keeps the core model tag-based while allowing ergonomic construction from
domain-specific config objects.

If the automatically composed state store is not sufficient, it can still be a
customization point:

```cpp
template <class BackendList>
struct backend_state;

template <>
struct backend_state<backend_list<CuSolverBackend, CublasBackend, CudaGenericBackend>> {
  using type = std::tuple<cuda::Device, cuda::Stream,
                          cusolver::Options, cublas::MathMode>;
};
```

The default `backend_state<BackendList>` can be computed with ordinary C++23
tuple-type concatenation and uniqueness:

```cpp
template <class... Backends>
struct backend_state<backend_list<Backends...>> {
  using type = unique_tuple_cat_t<typename Backends::state...>;
};
```

`unique_tuple_cat_t` is not a standard metafunction. It would be a small Uni20
helper equivalent to "concatenate these `std::tuple<...>` types, then remove
duplicate element types while preserving first occurrence order." C++ has
`std::tuple_cat` for values, but not a standard type-level unique tuple
concatenation helper. No reflection is required.

Advisory state may vary by backend entry and may be ignored by a backend that
does not understand it.

Whether structural state may vary across fallback entries is an operation-level
policy question. Different CUDA kernel implementations for the same device
should share the same device/domain state. A staging backend may deliberately
change domains by allocating temporaries and copying. A fallback from CUDA to
CPU is therefore not just "next backend in the list"; it is either a declared
staging path or an error if the operands cannot be read by the CPU backend.

## Inheritance Question

The main open design question is whether `BasicTensor` should continue to
inherit from `TensorView`, or whether `TensorView` should stop being a central
C++ abstraction.

### Keeping Inheritance

Benefits:

- Convenient base-class conversion to a view-like object.
- Some existing code can call APIs that name `TensorView` parameters.
- The owning tensor can reuse view implementation details.

Costs:

- `BasicTensor` and `TensorView` have different assignment semantics.
- A mutable `TensorView` base subobject can be rebound independently unless the
  inherited `TensorView` is specially configured to delete assignment.
- Top-level dispatch no longer wants "is a view"; it wants "has Uni20 storage
  policy and backend selector".
- Structural concepts may accidentally treat `BasicTensor` as `SpanLike` when
  that is not the intended dispatch layer.

If inheritance is kept temporarily, the embedded view should probably be a
non-rebindable `TensorView` specialization controlled by traits. `BasicTensor`
would need protected descriptor-reset hooks for its own copy/move/resize logic,
while external code could not assign through the base view.

### Abandoning Inheritance

Benefits:

- Cleaner semantic split: owner, view, and reference are separate types.
- Resolved view assignment can simply be mdspan-like without threatening
  `BasicTensor` invariants.
- Top-level tensor dispatch becomes pluggable through CPOs and concepts.
- External tensor-like objects can opt into Uni20 dispatch without inheriting
  from Uni20 classes.
- Non-owning rebinding views can be represented by copying an mdspan-like
  descriptor plus a backend selector/domain adaptor.

Costs:

- Existing APIs that name `TensorView` parameters need an adapter/lowering layer
  or should be generalized to resolved `SpanLike`/mdspan-like views.
- More explicit CPO surface must be designed and documented.
- Some temporary mdspan-like descriptors may be materialized at the backend
  lowering boundary.

The current tentative recommendation is to avoid relying on inheritance for the
dispatch contract. `BasicTensor` can keep inheritance as a short-term internal
implementation detail if useful, but front-end dispatch should be expressed in
terms of tensor operand concepts/CPOs.

If adaptor/concept dispatch is adopted, the long-term design probably does not
need `BasicTensor : TensorView`. It may not need a central C++ `TensorView`
class at all. A cleaner concrete layout is:

```cpp
class BasicTensor {
  storage_type data_;
  descriptor_type descriptor_;

public:
  auto view() &;
  auto view() const&;
  auto backend_selector() const;
};
```

`BasicTensor` owns storage with value/replace assignment. Dispatch obtains an
mdspan-like resolved view through the adaptor layer when it needs one. A named
`TensorView` type remains optional: it may be useful as an adaptor object that
combines an mdspan-like descriptor with a backend selector/domain, but C++ code
does not necessarily need it as the canonical non-owning view.

## External And Pluggable Operands

A concept/CPO design would let non-Uni20 types opt in:

```cpp
std::vector<double> x;
```

could be treated as a rank-1 host tensor if we provide:

```cpp
tensor_descriptor(x);
tensor_backend_selector(x);
tensor_read_view(x);
tensor_write_view(x);
```

For higher-rank external buffers, an explicit adapter is probably better:

```cpp
auto A = as_tensor_view(ptr, extents, strides, host_storage{});
```

The adapter can carry the backend selector that a plain mdspan lacks.

There is no need for raw `std::vector<T>` to automatically satisfy the tensor
concept globally. That would make generic tensor overloads surprisingly accept
ordinary vectors. A safer path is explicit adaptation:

```cpp
std::vector<double> x;
auto X = as_tensor(x);             // rank-1 host tensor adaptor
auto X = as_tensor(x, HostDomain{}); // explicit domain/backend selector
```

The adaptor object may have member functions (`view()`, `backend_selector()`,
`descriptor()`), and the dispatch CPOs can forward to those members. That gives
the pluggability benefits without making every standard container implicitly a
Uni20 tensor.

The adaptor's concrete type can encode semantics specific to the wrapped object,
and user code normally does not need to name that type. For example,
`as_tensor(std::vector<T>&)` can expose the existing vector storage as a rank-1
host tensor and also allow `ensure_shape` to resize the vector. A slice adaptor
or block-view adaptor would instead validate shape and write through fixed
storage. Both can satisfy the same `TensorOutput` concept because the behavior is
owned by the adaptor's `ensure_shape` and assignment operations.

## Async And Block Tensor Notes

The async/block design keeps descriptor and data separate:

- Descriptor metadata is synchronous: shape, strides, block structure, location,
  and operation state.
- Bytes are async: read/write data handles may need to be awaited.
- Resolved views are synthesized from synchronous descriptor plus awaited data
  handle.

For a block tensor, the planner reads block descriptors synchronously. Each block
handle then produces either an immediate mdspan-like resolved view or an
awaitable read/write handle that resolves to such a view.

For slices/views of async or owning tensors, `TensorRef` may need to carry a
lifetime or epoch token in addition to the shallow view descriptor. It should
still expose a plain resolved view to leaf kernels.

## Python Semantics

The proposed roles align with Python-style behavior:

```python
x = a[:, 0]   # x is a view object/name binding
x = b         # rebinding x does not mutate a
a[:, 0] = b   # slice assignment writes through
x[...] = b    # explicit write-through through the view
```

In C++, mdspan-like descriptor assignment corresponds to rebinding the view
object. `TensorRef::operator=` corresponds to slice/adaptor assignment.
`BasicTensor::operator=` corresponds to replacing the owning tensor value.

Python bindings may still choose to expose NumPy/nanobind arrays at the
boundary, with Uni20 tensor/adaptor objects used internally for dispatch. A
named `TensorView` type may still be valuable in the bindings even if C++ leaf
kernels mostly use mdspan-like descriptors directly.

## Open Questions

1. What are the exact concept names? Current candidates are
   `TensorDescribed`, `TensorBackendSelectable`, `TensorReadable`,
   `TensorOutput`, `TensorReadWrite`, and possibly `DenseTensor`.
2. What are the exact CPO names and signatures?
3. Does `assignment_semantics` gain a `value` semantic, or does async storage
   split semantic meaning from unconstructed-storage behavior?
4. Should `BasicTensor` drop public inheritance immediately, or keep a
   non-rebindable embedded view during migration?
5. What are the public API names for fixed-output versus resizable-output
   operations?
6. Is backend compatibility "same backend selector" or "common dispatch domain"?
   Block/MPI paths likely need the latter.
7. Do we need a central C++ `TensorView` type at all, or is an mdspan-like view
   plus a backend-selector adaptor enough? For CUDA, any adaptor/view must at
   least distinguish device memory from host memory and carry or recover the
   device ordinal.
8. What lifetime/epoch token does `TensorRef` need for slices of async or shared
   owning tensors?
9. Which external types should get built-in CPO adapters (`std::vector`,
   `stdex::mdspan`, nanobind arrays), and which should require explicit
   adapters?
10. Should the dispatch customization layer be implemented as named CPO objects,
   ADL free functions, or a small trait class that forwards to members?
11. Should fixed/resizable output be represented as separate concepts, or as one
    `TensorOutput` concept plus a trait controlling `ensure_shape` behavior?
12. Which explicit adaptors should expose resizable behavior through
    `ensure_shape`? `as_tensor(std::vector<T>&)` is a plausible resizable
    adaptor, while slices and block views should remain fixed/write-through.
13. What is the exact storage-domain/factory API for temporaries? Current
    candidates are `tensor_storage_domain(x)`,
    `make_temporary_tensor(selector, domain, descriptor)`, and
    `make_temporary_like(x, descriptor)`.
14. When only mdspan-like information is available, which combinations are
    acceptable without an explicit storage domain? Host mdspan plus host
    backend list may be safe; raw pointer mdspan plus CUDA backend probably
    needs an explicit `CudaDomain{device}` or a device-aware accessor type.
15. How much backend state may vary down an ordered backend list? Structural
    state such as CUDA device and memory domain should likely be invariant
    across compatible CUDA candidates, while advisory hints such as tile size,
    algorithm id, math mode, or workspace limit can vary by backend entry.
16. Is temporary allocation driven by the full backend selector value, by the
    selected backend tag after it accepts, or by the composed backend state tuple
    containing structural state?
17. What is the exact boundary between backend selector state and tensor
    storage-domain metadata? CUDA device can plausibly live in both; a
    hypothetical MPI-striped dense tensor would need a placement map that may
    belong more naturally to the tensor domain while still being visible to
    backend dispatch.
18. How should backend entries declare state requirements? Current options
    are a direct `using state = std::tuple<...>`, an empty base such as
    `requires_state<cuda::Device, cuda::Stream>`, or a CRTP backend-family base.
    This mechanism must not imply fallback order or duplicate runtime state.
19. Should `unique_tuple_cat_t` live in a backend-specific detail namespace or
    as a more general Uni20 detail helper? It is generic, but the expected use
    is composing short backend state tuples.

## Tentative Conclusions

- Top-level kernel dispatch should require tensor/storage operands, not bare
  mdspan-like views.
- For local dense tensors, `SpanLike + backend selector` is a plausible minimal
  dispatchable tensor concept.
- The likely dispatch concept set is `TensorDescribed`,
  `TensorBackendSelectable`, `TensorReadable`, `TensorOutput`, and
  `TensorReadWrite`, with `DenseTensor` as an optional local dense refinement.
- Leaf kernels should operate on resolved mdspan-like views.
- Storage policy and backend selector should be split: storage provides a
  default backend selector, but non-owning views can carry a selector/domain
  without owning storage.
- Temporary creation is a storage-domain/factory operation. Copying into or out
  of the temporary is an ordinary backend-dispatched copy kernel.
- Backend selectors are values, not only type lists. Their type gives the
  candidate order; stateless backend tags declare runtime state requirements
  with `using state = std::tuple<...>`.
- Shared structural state should live once in a composed backend state tuple or
  storage-domain descriptor. Backend inheritance, if used, should declare
  required state/capabilities only; it should not encode backend ordering.
- `unique_tuple_cat_t` is the intended implementation helper for composing
  backend state: concatenate backend `std::tuple<...>` state lists and remove
  duplicate tag types so `std::get<Tag>` remains well-formed.
- Backend selectors and storage domains are related but distinct: the selector
  chooses compute backends and can carry state, while the storage domain chooses
  memory placement, allocator/resource, and CUDA device information.
- `BasicTensor`, mdspan-like resolved views, and `TensorRef` should have
  distinct assignment semantics: value/replace, rebind descriptor, and
  write-through.
- Explicit adaptors can choose semantics appropriate to the wrapped object. For
  example, `as_tensor(std::vector<T>&)` may be resizable even though slice and
  block-view adaptors are fixed/write-through.
- Backend dispatch should use operation tags plus generic `try_kernel` /
  optional `kernel_maybe_can` customization points.
- Backend selectors should accept both singleton backend values and ordered
  backend-list values.
- Concepts and CPOs are the preferred long-term dispatch abstraction. They make
  Uni20 tensor dispatch pluggable and avoid making inheritance from a view type
  part of the API contract.
- If adaptor/concept dispatch is adopted, `BasicTensor : TensorView` is likely
  unnecessary in the long-term design, and a central C++ `TensorView` type may be
  optional rather than fundamental.
