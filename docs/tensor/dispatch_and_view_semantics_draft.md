# Tensor Dispatch And View Semantics Draft

**Status:** background design note. The canonical implemented operation
contracts and current Async support matrix are in
[`operations.md`](operations.md). This draft retains rationale
and future design discussion; where the documents differ, the canonical guide
and code are authoritative.

This note records the implemented dense tensor/view boundary and the remaining
design questions around output assignment and higher-level dispatch.

Related notes:

- [`../architecture/kernel_dispatch.md`](../architecture/kernel_dispatch.md) - backend lists, operation tags,
  and async/lowering layers.
- [`../architecture/backend_dispatch.md`](../architecture/backend_dispatch.md) - compile-time capability,
  runtime attempt, and fallback behavior.
- [`../async/storage.md`](../async/storage.md) - async value identity, timeline rebinding, and write-through assignment.
- [`../async/tensor_lifetime_and_dispatch_draft.md`](../async/tensor_lifetime_and_dispatch_draft.md)
  - async tensor aliases, lifetimes, and async dispatch lowering.
- [`../symmetry/block_tensor.md`](../symmetry/block_tensor.md) - block tensor storage and per-block
  view/lowering model.

## Core Distinction

The implemented dense GEMM and GEMV paths use a strict layer split:

1. **Front-end tensor operations** accept actual Uni20 tensor-like objects. They
   need storage policy, output policy, synchronous metadata, async/block
   structure, and a default backend selector.
2. **Backend dispatch** selects a backend from an operation tag plus a backend
   selector derived from tensor storage policy.
3. **Backend lowering** acquires or interprets those tensor views in the selected
   execution domain and produces resolved mdspan-like views.
4. **Leaf kernels** are provider calls or lower-level Uni20 module functions.
   They run on handles, extents, strides, accessors, or provider descriptors and
   do not participate in the operation-tag dispatch walk.

A plain `stdex::mdspan` or structural `MdspanLike` is sufficient for a lower-level
kernel but is not an operation-dispatch operand. It does not carry the tensor-level
acquisition and backend-selection contract.

For local dense tensors, the tensor-level concepts are intentionally not
mdspan-like:

```cpp
TensorView        = backend selector + readable mdspan() result
MutableTensorView = TensorView + writable mdspan() result
```

The tensor object itself does not model `MdspanLike`. This keeps policy selection
at the tensor layer and makes the lowering boundary explicit. A public
bare-mdspan convenience overload may take an explicit selector, but it adapts the
mdspan to a tensor view before operation dispatch.

## Type Roles

The current semantic split is:

```cpp
BasicTensor      // configurable owner parameterized by an mdspan extents type
Tensor           // ordinary owner with runtime extents and a static rank
TensorView       // readable tensor-level concept, not an mdspan-like type
MutableTensorView // writable refinement of TensorView
mdspan           // resolved leaf-kernel operand
TensorRef        // future non-owning tensor lvalue proxy
```

### Tensor and BasicTensor

`Tensor<Element, Rank, StoragePolicy, LayoutPolicy, AccessorFactory>` is the
concrete owning dense tensor class; its default mdspan extents are dynamic on
every axis. `BasicTensor<Element, Extents, ...>` is an extents-first alias for
the same class when static or mixed extents belong in the type. Most code should
use `Tensor<T, Rank>`, deduction through `Tensor(view)` or `make_tensor(view)`,
or a domain alias such as `DenseMatrix<T>` rather than spelling an extents type
or every policy.
Assignment has value/replace semantics:

```cpp
Tensor<double, 2> C;
C = rhs; // replace C's logical tensor value
```

`make_tensor(view)` deliberately returns the runtime-extents `Tensor` form,
even when the source mdspan has static or mixed extents. Code that intends to
carry extent values in the C++ type names `BasicTensor` explicitly.

Both forms retain compile-time rank because their resolved descriptor is an
mdspan. A future runtime-rank tensor or `dynamic_mdspan` is a separate type and
kernel-interface design; it is not another mode of `Tensor`.

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

`TensorView` is a concept, not a central base class. There is currently no
general concrete non-owning tensor adaptor. Leaf kernels receive the mdspan
resolved by an owning tensor or a future adaptor with explicit storage-policy
and lifetime semantics.

For dense CUDA data, the tensor/adaptor object must carry enough dispatch
information to identify device memory. At minimum that means a storage/backend
domain and likely a device ordinal or equivalent placement record. Dense MPI
distribution is not a near-term requirement for dense tensors; dense tensors are
expected to be local to one node, while block/MPI distribution belongs to
`BlockTensor` storage policy.

### TensorRef

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

## View Algebra Policy

Uni20 should distinguish **structural views** from **semantic transform views**.
This distinction controls which views may appear on the left-hand side of
assignment and which views are only readable expression operands.

Structural views change how storage is addressed without changing the write
law. Examples include slices, block views, reshape/reindex views, and component
views such as `real(x)` or `imag(x)` for complex storage. These views may change
the element type, data handle, extents, and strides, but writes remain local:

```cpp
real(z)[i] = a; // writes the real component of z[i]
```

`real(x)` and `imag(x)` are therefore slice-like component views, not proxy
accessor adaptors. They should be represented by adjusted handles and strides
over the original complex storage when the scalar/storage policy guarantees that
component layout. They can be writable because assignment writes one addressed
component directly.

Semantic transform views change the value read from storage. Examples include
`conj(x)`, scaling views, and arbitrary elementwise `transform_view(...)`
expressions. These are read-only by default. A conjugating view is not an
ordinary lvalue because assigning through it has a contravariant law:

```cpp
conj(x) = y; // would mean x = conj(y)
```

If Uni20 ever supports such a write, it should be introduced as a deliberate
operation-specific rewrite or a separately named internal adaptor, not as the
default behavior of `conj(x)`.

Generic writable outputs should therefore require ordinary raw/default-style
accessors over a structural view. Writable proxy accessors are special cases
that must document their assignment law and backend lowering explicitly. The
implemented mdspan and Tensor `conj(...)` helpers follow this rule: complex
inputs become read-only conjugating views, double conjugation cancels to a
read-only identity view of the original storage, and non-complex inputs become
const identity views. The Tensor adaptor resolves the existing mdspan
`conjugated_accessor`; it does not allocate or run a kernel.

Eager evaluation is explicit:

```cpp
copy(out, conj(in));             // copy into an existing/resizable output
auto owned = make_tensor(conj(in)); // allocate, then dispatch copy_op
```

`copy_op` is backend-dispatched. Its reference CPU backend reads through
accessor semantics. A future BLAS backend may lower rank-two layout and
conjugation metadata to provider matrix-copy extensions such as `omatcopy`;
this does not change the lazy semantics of `conj(in)` itself.

Do not treat a pointer `data_handle_type` as proof that a backend may read or
write the storage directly. The data handle identifies storage; the accessor
defines the value observed at that storage. A semantic accessor can keep the
same pointer while changing the value returned by `access(...)`. Direct
provider paths that bypass `access(...)` must require `stdex::default_accessor`
or a specifically recognized accessor whose semantics are lowered into provider
metadata. Uni20's conjugating mdspan accessor follows the C++26
`std::linalg::conjugated_accessor` direction described in WG21 P3050R3, but
Uni20 keeps `uni20::conj` as the operation that fixes real-scalar conjugation
semantics instead of using standard `std::conj` behavior.

## Assignment Semantics

Tensor types still have three useful semantic contracts:

```cpp
Tensor -> value/replace
mdspan/view  -> rebind descriptor
TensorRef    -> write-through
```

Async write proxies do not add a second buffer category. For independent values,
proxy assignment constructs empty storage or invokes `T::operator=`. For aliases,
the descriptor is read-only and assignment invokes ADL `assign_through`.
Consequently:

- `Tensor`: if unconstructed, emplace from the right-hand side; if
  constructed, call `Tensor::operator=`, which may reuse or reallocate.
- A bare mdspan-like descriptor is not an async alias by default; storing one
  asynchronously is unsafe unless an external owner establishes its lifetime.
- `TensorRef`: when explicitly classified as an async alias, assignment writes
  through to the existing tensor region via `assign_through`.
- Explicit adaptor types choose the semantic contract of the wrapped object.
  `as_tensor(std::vector<T>&)` may support resizing and element assignment,
  while a slice adaptor should remain fixed/write-through.

At the outer handle level, `is_async_alias_v<T>` distinguishes independent
values from owner-bound aliases. Direct assignment detaches an independent
value onto fresh storage and a fresh `EpochQueue`. Alias assignment retains its
identity and is available only when ADL finds `assign_through`; exact and
heterogeneous sources follow the same capability rule.

For independent values, explicit `proxy.emplace(...)` reconstructs the stored
object inside the current async storage and queue. It is not a handle rebind.
Alias proxies do not expose `emplace`.
Element copying that requires shape preparation or backend dispatch remains a
named tensor operation rather than hidden proxy-assignment behavior.

## Output Parameters

There are two broad output modes.

Both use the implemented shape helpers:

```cpp
require_shape(out, shape); // fixed update
ensure_shape(out, shape);  // overwrite may resize
```

For an owning/resizable `Tensor`, `ensure_shape` retains matching storage and
otherwise calls `reset_shape`, which rebuilds the tensor using its default
mapping. For a
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
require_shape(C, shape);            // validate only
gemm(C, alpha, A, B, beta);         // fixed-output BLAS form
add_product(C, A, B);               // reads and updates existing C
```

This is the natural mode for `TensorRef`, fixed adaptors, block views, and BLAS
update forms where the output is also an input.

### Resizable Output

A resizable output can prepare the requested shape by reuse or allocation:

```cpp
assign_product(C, A, B);       // overwrite; C may be resized
add_product(C, A, B);          // update; C must already match
auto C = make_tensor(conj(A)); // explicit view materialization
```

Uni20 uses output-first mutable parameters for linalg APIs, so the update form is
`gemm(C, alpha, A, B, beta)`. When `beta != 0`,
`C` is both input and output, so a fixed-output/update API is the clearer
default. Reallocation is natural for overwrite/value-producing operations such
as `assign_product(C, A, B)` and explicit materialization through
`make_tensor(view)`.

## Examples Driving Concepts

The concept names should be derived from call sites. These examples are the
current target surface.

### Owning Assignment

```cpp
auto C = make_tensor(rhs);
C = same_concrete_tensor;
```

Needed behavior:

- infer the right-hand-side shape
- call `ensure_shape(C, shape)`
- evaluate into `tensor_write_view(C)`

Concept pressure:

- `ResizableTensorOutput<C>` for `ensure_shape` plus writable view

### Value-Producing Operation

```cpp
assign_product(C, A, B); // C may be resized
```

Needed behavior:

- `A` and `B` provide synchronous metadata and read views
- `C` provides `ensure_shape` and a write view
- backend selector is computed from all participating tensor operands

Concept pressure:

- `TensorView<A>`
- `TensorView<B>`
- `ResizableTensorOutput<C>` for a shape-changing output
- compatible backend selectors on all dispatch operands

### Fixed Update Operation

```cpp
gemm(C, alpha, A, B, beta);
gemm(C.slice(...), alpha, A, B, beta);
```

Needed behavior:

- `C` is both input and output when `beta != 0`
- shape is already fixed and validated by GEMM
- the writable `mdspan()` is resolved only after tensor-level dispatch policy is known

Concept pressure:

- `MutableTensorView<C>`
- `TensorView<A>` and `TensorView<B>`

### Explicit Backend On Mdspan

```cpp
dispatch_kernel(backend_list{BlasBackend{}, CpuReferenceBackend{}}, gemm_op{},
                c_mdspan, alpha, a_mdspan, b_mdspan, beta);
```

Needed behavior:

- mdspans provide leaf-kernel view information
- the explicit backend selector supplies the dispatch information missing from
  bare mdspan

Concept pressure:

- `MdspanLike` / `StridedMdspanLike` remains the leaf-kernel concept
- top-level dispatch may accept `MdspanLike` operands only when a backend selector
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
copy(tmp, A);
```

Needed behavior:

- allocate scratch/output storage that is compatible with the selected backend
- preserve value type, rank, layout requirements, memory domain, and device
  placement
- support the same path when `A` is only an mdspan-like view plus explicit
  backend/domain information
- allow an explicit backend selector value to carry immutable runtime state, such as
  CUDA device, workspace allocator, stream policy, or advisory hints
- treat population of the temporary as an ordinary copy kernel

Concept pressure:

- backend selection alone is not enough to allocate storage; the dispatch layer
  also needs a memory domain or storage factory
- an explicit backend selector must be a value, not only a backend tag list, when
  placement or allocation policy is runtime state
- `TensorBackendSelectable` should stay separate from a possible
  `TensorStorageDomain` / `TensorTemporaryFactory` concept
- an explicit mdspan adapter may need to carry both backend selector and memory
  domain if temporaries are required

## Implemented Dense Concepts

The dense checkpoint distinguishes tensor dispatch participation from
mdspan-like leaf views directly:

```cpp
template <class T>
concept TensorView = requires(T const& tensor) {
  tensor.backend_selector();
  tensor.extents();
  tensor.extent(size_t{});
  { tensor.mdspan() } -> MdspanLike;
};

template <class T>
concept MutableTensorView = TensorView<T> &&
  MutableMdspanLike<decltype(std::declval<T&>().mdspan())>;

template <class T>
concept StridedTensorView = TensorView<T> &&
  StridedMdspanLike<decltype(std::declval<T const&>().mdspan())>;

template <class T, size_t Rank>
concept RankedTensorView = TensorView<T> &&
  RankedMdspanLike<decltype(std::declval<T const&>().mdspan()), Rank>;

template <class T, size_t Rank>
concept MutableRankedTensorView = MutableTensorView<T> && RankedTensorView<T, Rank>;
```

`ResizableTensorOutput` detects `reset_shape(extents)`. `ensure_shape` uses it
for overwrite operations; it is not part of fixed-output GEMM, where allocation
policy has already been decided by the caller.

## Dispatch Interface And Adaptors

The top-level dispatch API uses concepts plus ordinary members rather than
base-class conversion. External customization remains future work.

The customization layer can be implemented as free CPOs that call member
functions when present. Uni20's own tensor and adaptor classes can therefore use
ordinary members for locality and documentation, while external types can still
opt in through wrappers or ADL/customization.

The implemented `select_backend(operation, operands...)` uses a global
`backend_selector_override<Operation, StoragePolicy>` customization trait. Its
fallback requires a common storage policy and returns the first operand's
storage-provided static selector. Operand values are present only for type
deduction and front-end ergonomics. A specialization may define
`select(operation)` to globally replace the backend list for an
operation/storage combination. Explicit-selector operation overloads bypass
this default-selection step and remain the route for caller-specific immutable
context.

The default path rejects mixed storage policies at compile time. In particular,
a host tensor and a CUDA tensor do not acquire an implicit transfer merely
because some backend appears in both selector lists. The caller must request an
explicit transfer or use an operation-specific higher-level path whose global
policy explicitly defines the mixed-domain semantics. Mixed-policy global
selection is intentionally left open rather than inferred by the current
single-policy override.

For example:

```cpp
tensor_mdspan(x);            // may call x.mdspan()
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
void matmul(C& c, A const& a, B const& b)
{
  auto selector = select_backend(matmul_op{}, a, b, c);
  auto shape = matmul_shape(tensor_descriptor(a), tensor_descriptor(b));

  ensure_shape(c, shape);
  dispatch_kernel(backend_list_t<decltype(selector)>{}, matmul_op{}, c, a, b);
}
```

The selected backend eventually lowers operands to views:

```cpp
auto av = tensor_read_view(a);
auto bv = tensor_read_view(b);
auto cv = tensor_write_view(c);
try_kernel(Backend{}, matmul_op{}, cv, av, bv);
```

This keeps `MdspanLike` at the leaf-kernel level and keeps backend selection at
the tensor/storage level.

For local dense tensors, `TensorView` remains backend selector plus synchronous
metadata plus a readable `mdspan()` result. `Tensor` itself deliberately does
not satisfy `MdspanLike`; the resolved result of `mdspan()` does.

## Temporaries

Kernel front ends need to create temporaries in all modes: synchronous dense
tensors, async tensors, block tensors, and direct mdspan entry points. This is
not special assignment semantics; it is allocation plus a copy/evaluation
kernel.

The current host implementation provides the first explicit form:

```cpp
auto tmp = make_tensor(conj(a));
auto tmp_column_major = make_tensor<ColumnMajor>(conj(a));
auto tmp_from_span = make_tensor<ColumnMajor>(CpuReferenceBackend{}, span);
```

The bare-mdspan overload requires an explicit selector because a mdspan does
not provide storage policy. The more general temporary type should eventually
be selected from the backend selector/list value and its storage domain, not
from the C++ name of the operand:

```cpp
auto selector = select_backend(copy_op{}, a, b);
auto domain = temporary_storage_domain(selector, a, b);
auto tmp = make_temporary_tensor(selector, domain, descriptor);
copy(tmp, a); // ordinary backend-dispatched copy kernel
```

For Uni20-owned tensors, the domain comes from the storage policy:

```cpp
Tensor<T, Rank, HostStorage> -> HostDomain{allocator/resource}
Tensor<T, Rank, CudaStorage> -> CudaDomain{device, allocator/resource}
```

For non-owning views/adaptors, the domain must be carried explicitly or be
recoverable from the adaptor:

```cpp
auto A = as_tensor_view(a_mdspan, HostDomain{});
auto B = as_tensor_view(b_mdspan, CudaDomain{device});
```

The resulting concrete temporary type can be internal and unnamed by user code.
It might be `Tensor<T, Rank, HostStorage>` for host domains, a future
`Tensor<T, Rank, CudaStorage>` for CUDA domains, or an async/block scratch handle
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
dispatch_kernel(make_host_selector(HostDomain{}), gemm_op{},
                c_mdspan, alpha, a_mdspan, b_mdspan, beta);

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

For dense local tensors, `TensorView` requires a backend selector and a readable
mdspan result; `MutableTensorView` additionally requires a writable mdspan
result. The tensor object is not `MdspanLike`. Memory kind and runtime device
placement are properties of the returned mdspan's accessor-defined data handle,
not of the default selector.

Storage and backend selection should remain separate concepts. A storage policy
owns or describes how bytes are stored, and one of its traits is the default
backend selector. A non-owning adaptor may have no storage object at all, but it
still needs a backend selector or storage-domain descriptor. For example:

```cpp
Tensor<T, Rank, HostStorage> -> backend_list{BlasBackend{}, CpuReferenceBackend{}}
TensorAdaptor<T, HostDomain> -> backend_list{BlasBackend{}, CpuReferenceBackend{}}
TensorAdaptor<T, CudaDomain> -> backend_list{CublasBackend{}, CudaGenericBackend{}}
```

For CUDA views, the selector/domain must distinguish device memory from host
memory and carry or recover the device ordinal. The view cannot rely on owning
storage to provide that information.

The selector can be overridden with either an ordered backend-list value or a
singleton backend value:

```cpp
gemm(C, alpha, A, B, beta);                                  // storage default
gemm(CpuReferenceBackend{}, C, alpha, A, B, beta);            // one backend only
gemm(backend_list{BlasBackend{}, CpuReferenceBackend{}},
     C, alpha, A, B, beta);                                  // ordered list
gemm(make_backend_selector<backend_list<CublasBackend, CudaGenericBackend>>(
       CublasConfig{.device = {1}, .stream = {stream}, .math_mode = {tf32_allowed}}),
     C, alpha, A, B, beta);
```

A singleton backend is normalized to a one-entry backend list value. It does
not get an implicit fallback.

Plain mdspan entry points should require an explicit backend selector:

```cpp
dispatch_kernel(backend_list{BlasBackend{}, CpuReferenceBackend{}}, gemm_op{},
                c_mdspan, alpha, a_mdspan, b_mdspan, beta);
```

or an explicit backend helper:

```cpp
BlasBackend::gemm(c_mdspan, alpha, a_mdspan, b_mdspan, beta);
```

The reason is that mdspans do not carry Uni20 storage policy.

The term `backend_list<...>` should be read as a shorthand for the static
ordering of backend entries. The actual selector passed through dispatch should
be a value. Default backend entries should normally be stateless tags. An
explicit override may carry operation context:

```cpp
struct CublasBackend {
  cudaStream_t stream;
  math_mode_t math_mode;
};

struct CudaGenericBackend {
  cudaStream_t stream;
};

struct CpuReferenceBackend {};
```

For a selector such as:

```cpp
backend_list{CublasBackend{stream, math_mode},
             CudaGenericBackend{stream}}
```

the list value supplies both ordering and the per-backend runtime state. A
separate composed selector-state tuple was an earlier design option, inherited
from the cytnx dispatch experiment. It may still be useful internally for a
future selector implementation, but the leaf-kernel customization points should
not require a separate `State&` parameter.

For distributed block tensors, communicator and operation policy may belong in
an explicit backend value, while rank-local ownership and section maps belong to
the container's placement metadata.

Inheritance is still possible, but it should only help declare or share
type-level requirements. It should not define the backend order. Simple
inheritance such as `CuSolverBackend : CudaBackend` is too strong: it says a
cuSOLVER backend *is* the CUDA generic backend, and it risks coupling capability
inheritance to fallback ordering. The ordered backend list should remain the
only thing that says "try cuTensorNetwork, then cuBLAS, then generic CUDA."

An empty base can still be a spelling for shared type-level capabilities:

```cpp
template <class... Capabilities>
struct backend_capabilities {};

struct CublasBackend : backend_capabilities<cuda_memory, blas_scalar_types> {};
struct CudaGenericBackend : backend_capabilities<cuda_memory> {};
```

A CRTP family base is another possible spelling:

```cpp
template <class BackendTag, class Derived>
struct Backend;

template <class Derived>
struct Backend<CublasTag, Derived>
    : backend_capabilities<cuda_memory, blas_scalar_types> {};
```

That can factor common type-level traits or helper APIs across related backends,
but it should still be used only to express requirements/capabilities. It should
not imply that `CublasBackend` inherits from `CudaGenericBackend` or that
`CudaGenericBackend` is the next fallback.

A construction convenience may still be useful for the initial implementation.
For example, a user could provide a richer backend configuration object, and the
selector can build ordered backend values from it:

```cpp
struct CublasConfig {
  cudaStream_t stream;
  math_mode_t math_mode;
};
```

Then:

```cpp
auto selector = make_backend_selector<backend_list<CublasBackend, CudaGenericBackend>>(
    CublasConfig{.stream = {stream}, .math_mode = {tf32_allowed}});
```

can produce a selector equivalent to:

```cpp
backend_list{CublasBackend{stream, math_mode},
             CudaGenericBackend{stream}};
```

An implementation may still store shared state once inside the selector and hand
each CPO a lightweight backend view. That is an optimization or implementation
detail. The leaf CPO shape remains `kernel_accepts_types(backend, op, args...)`
as a `consteval` tri-state function over ordinary reference parameters, and
`try_kernel(backend, op, args...)` as the runtime attempt.

Advisory state may vary by backend entry and may be ignored by a backend that
does not understand it.

Operand placement does not vary across fallback entries because it comes from
the operand accessors and data handles. Operation context may vary by candidate.
A staging backend may deliberately change domains by allocating temporaries and
copying. A fallback from CUDA to CPU is therefore not just "next backend in the
list"; it is either a declared staging path or an error if the operands cannot
be read by the CPU backend.

## Owning Tensor Composition

The `Tensor` implementation owns storage, mapping, and accessor-factory state
directly, and constructs resolved mdspans on demand. `BasicTensor` aliases a
specialization with an explicit extents type:

```cpp
class Tensor {
  storage_type data_;
  descriptor_type descriptor_;

public:
  auto mdspan() &;
  auto mdspan() const&;
  auto backend_selector() const;
};
```

This keeps owner copy/move semantics independent of mdspan descriptor rebinding.
`Tensor` satisfies the tensor-level concepts but not `MdspanLike`; its
returned mdspans satisfy the leaf concepts. A future slice or external-storage
adaptor must carry storage/execution policy and explicit lifetime semantics.

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
`Tensor::operator=` corresponds to replacing the owning tensor value.

Python bindings may still choose to expose NumPy/nanobind arrays at the
boundary, with Uni20 tensor/adaptor objects used internally for dispatch. Such
adaptors must retain their Python owner and storage/execution policy rather than
being raw mdspan descriptors.

## Open Questions

1. What additional concepts are needed for allocating/resizable operations
   beyond the implemented `TensorView` and `MutableTensorView` concepts?
2. What external-adaptor CPO names and signatures are needed, if member-based
   adaptation is insufficient?
3. What are the public API names for fixed-output versus resizable-output
   operations?
4. Is backend compatibility "same backend selector" or "common dispatch domain"?
   Block/MPI paths likely need the latter.
5. What lifetime/epoch token does `TensorRef` need for slices of async or shared
   owning tensors?
6. Which external types should get built-in CPO adapters (`std::vector`,
   `stdex::mdspan`, nanobind arrays), and which should require explicit
   adapters?
7. Should the dispatch customization layer be implemented as named CPO objects,
   ADL free functions, or a small trait class that forwards to members?
8. Should fixed/resizable output be represented as separate concepts, or as one
    `TensorOutput` concept plus a trait controlling `ensure_shape` behavior?
9. Which explicit adaptors should expose resizable behavior through
    `ensure_shape`? `as_tensor(std::vector<T>&)` is a plausible resizable
    adaptor, while slices and block views should remain fixed/write-through.
10. What is the exact storage-domain/factory API for temporaries? Current
    candidates are `tensor_storage_domain(x)`,
    `make_temporary_tensor(selector, domain, descriptor)`, and
    `make_temporary_like(x, descriptor)`.
11. When only mdspan-like information is available, which combinations are
    acceptable without an explicit storage domain? Host mdspan plus host
    backend list may be safe; raw pointer mdspan plus CUDA backend probably
    needs an explicit `CudaDomain{device}` or a device-aware accessor type.
12. How much operation context may vary down an ordered backend list? Hints such
    as tile size, algorithm id, math mode, or workspace limit can vary by backend
    entry, while operand placement remains in the operands.
13. Is temporary allocation driven by the full backend selector value, by the
    selected backend after it accepts, or by a storage-domain descriptor
    containing structural state?
14. How should distributed placement metadata be exposed to backend dispatch
    without duplicating it in selector state?
15. Should backend entries carry all small runtime fields directly, or should a
    richer selector object share some state internally while presenting the same
    backend-first CPO shape to kernels?

## Tentative Conclusions

- Operation-tag dispatch requires tensor-level operands. Bare-mdspan convenience
  overloads use an explicit selector and adapt their operands to tensor views
  before dispatch.
- `TensorView` and `MutableTensorView` are tensor-level concepts. Their objects
  deliberately do not model `MdspanLike`; they produce resolved mdspans.
- Provider APIs and lower-level Uni20 module kernels may operate directly on
  resolved mdspan-like views; they are below operation-tag dispatch.
- Storage policy and backend selector should be split: storage provides a
  default backend selector, but non-owning views can carry a selector/domain
  without owning storage.
- Temporary creation is a storage-domain/factory operation. Copying into or out
  of the temporary is an ordinary backend-dispatched copy kernel.
- Backend selectors are values, not only type lists. Their type gives the
  candidate order; backend values may be stateless tags or may carry small
  runtime fields directly.
- Shared operation context, if needed, may live in a selector implementation
  detail. Backend inheritance, if used, should declare capabilities only; it
  should not encode backend ordering.
- Backend selectors and storage domains are related but distinct: the selector
  chooses compute backends and may carry explicit operation context, while the
  storage/accessor-defined data handle carries memory placement and device
  information.
- `Tensor`, mdspan-like resolved views, and `TensorRef` should have
  distinct assignment semantics: value/replace, rebind descriptor, and
  write-through. Async storage defers to those underlying semantics rather than
  classifying them with another trait.
- Explicit adaptors can choose semantics appropriate to the wrapped object. For
  example, `as_tensor(std::vector<T>&)` may be resizable even though slice and
  block-view adaptors are fixed/write-through.
- Backend dispatch should use operation tags plus generic runtime-attempt and
  optional type-level capability customization points. The current CPO names are
  `try_kernel` and `kernel_accepts_types`.
- Backend selectors should accept both singleton backend values and ordered
  backend-list values.
- Concepts are the tensor dispatch abstraction. `Tensor` uses composition;
  future non-owning adaptors must define lifetime and assignment semantics
  explicitly rather than relying on inheritance.
