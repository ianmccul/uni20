# Mdspec

**Status:** current guide for the implemented class, structural concepts, and
initial tensor acquisition layer.

`uni20::mdspec` describes a multidimensional array whose shape, layout,
and element-access semantics are already known, but whose usable data pointer
or handle is not yet available.

The object stores the actual layout mapping and AccessorPolicy that will be
used to access the array. In place of a data-handle value, it stores a data
descriptor through which an access layer can lease a compatible handle. That
lease is RAII-scoped: it retains the access state that makes the handle usable
and releases that access when the lease ends.

This separates two facts that an ordinary mdspan holds together:

1. how multidimensional indices map to elements and how those elements are
   accessed;
2. the currently usable handle to the storage.

Code can therefore inspect and transform the array's extents, mapping, and
accessor before the resources needed for actual data access have been acquired.

An ordinary mdspan contains:

```text
mdspan = (data_handle, mapping, accessor)
```

A `mdspec` contains:

```text
mdspec = (data_descriptor, mapping, accessor)
```

The mapping and accessor are the actual objects intended for a later resolved
mdspan. The data descriptor identifies the logical storage region from which a
compatible data handle can be leased.

```text
mdspec + acquisition operation
        |
        v
RAII access lease containing a usable mdspan
```

Acquisition is a free-function operation over either tensor-level objects or
normalized mdspec descriptors. Tensor overloads return a move-only RAII
object that models `ImmediateTensorView`, retains the source storage and
backend selector, and exposes the resolved mdspan through `.mdspan()`.
Descriptor overloads return a policy-free mdspan lease for use inside an
already selected backend.

## Execution Domains

An mdspec may describe data intended for a particular execution domain. The
domain does not necessarily describe where the authoritative data currently
resides.

Examples include:

- host memory;
- CUDA memory;
- another accelerator or addressable memory space;
- mapped or staged storage;
- data demand-loaded from host, disk, or a network service into an execution
  domain.

The accessor describes where its eventual mdspan may be evaluated. The mdspec
name itself is domain-neutral: it says only that the multidimensional metadata
is complete while a usable handle may still require acquisition.

## Current Header and Template

Include:

```cpp
#include <uni20/mdspan/mdspec.hpp>
```

The implemented class template is:

```cpp
template<class ElementType,
         class Extents,
         class LayoutPolicy,
         AccessorPolicy Accessor,
         class DataDescriptor>
  requires std::same_as<ElementType, typename Accessor::element_type>
class mdspec;
```

The template deliberately follows the mdspan type vocabulary and appends the
data descriptor as the unresolved component. It currently has no default
template arguments.

Construction stores all three components by value:

```cpp
mdspec(
    data_descriptor_type descriptor,
    mapping_type mapping,
    accessor_type accessor);
```

Copy and move behavior follow the stored component types. In particular, a
move-only data descriptor produces a move-only `mdspec`.

## Type and Observer Contract

`mdspec` exposes the mdspan-compatible type aliases needed before handle
acquisition:

- `element_type` and `value_type`;
- `extents_type`, `index_type`, `size_type`, and `rank_type`;
- `layout_type` and `mapping_type`;
- `accessor_type`, `data_handle_type`, and `reference`;
- `data_descriptor_type`.

The `data_handle_type` alias is known from the accessor even though no handle
value is stored.

The available observers are:

| Category | Observers |
|---|---|
| Descriptor | mutable and const `data_descriptor()` |
| Mapping and accessor | `mapping()`, `accessor()` |
| Shape | `rank()`, `rank_dynamic()`, `static_extent(axis)`, `extents()`, `extent(axis)` |
| Mapping properties | `is_always_unique()`, `is_always_exhaustive()`, `is_always_strided()` |
| Runtime mapping properties | `is_unique()`, `is_exhaustive()`, `is_strided()` |
| Strides | `stride(axis)` when the mapping provides it |

An unresolved `mdspec` intentionally does not expose:

- `data_handle()`;
- multidimensional `operator[]`;
- host or device dereference;
- implicit conversion to `stdex::mdspan`;
- allocation, migration, synchronization, or access acquisition.

It therefore does not model `MdspanLike`.

## Structural Concepts

The concepts are the primary generic interface. Code should not require the
concrete `mdspec` class when a structural constraint is sufficient.

### `MdspecLike`

`MdspecLike<S>` accepts either:

1. an ordinary `MdspanLike<S>`, as the immediate-handle case; or
2. an unresolved structural model exposing mdspan metadata plus
   `data_descriptor_type` and `data_descriptor()`.

For Uni20's concrete `Tensor`, immediate accessibility takes precedence over
descriptor availability. Its mutable and const `mdspec()` overloads
therefore return ordinary mdspans whenever the corresponding writable or
readable handle is immediately available, even if the storage policy also
provides a descriptor. Only a missing immediate handle selects the
metadata-only `uni20::mdspec` representation. Read, write, immediate,
and deferred capabilities are detected independently.

The unresolved model must expose compatible element, extent, layout, mapping,
accessor, handle, and reference aliases. Its accessor must satisfy
`AccessorPolicy`.

An independent type can therefore model `MdspecLike` without inheriting
from or converting to `mdspec`.

The concept provides a common metadata vocabulary, but not one common handle
operation. Generic code can distinguish the two cases explicitly:

```cpp
template<uni20::MdspecLike Span>
void inspect(Span const& span)
{
  inspect_mapping(span.mapping());
  inspect_accessor(span.accessor());

  if constexpr (uni20::MdspanLike<Span>)
    inspect_immediate_handle(span.data_handle());
  else
    inspect_descriptor(span.data_descriptor());
}
```

### Rank and Stride Refinements

The implemented refinements are:

```cpp
MutableMdspecLike<Span>
RankedMdspecLike<Span, Rank>
StridedMdspecLike<Span>
MutableRankedMdspecLike<Span, Rank>
MutableStridedMdspecLike<Span>
RankedStridedMdspecLike<Span, Rank>
MutableRankedStridedMdspecLike<Span, Rank>
```

Like `MdspecLike`, these accept both immediate mdspans and independent
descriptor-backed models.

The mutable forms additionally require non-const element semantics and an
assignable accessor reference. They describe eventual mutation after
acquisition and do not add indexing to an unresolved descriptor.

Metadata-only helpers use the mdspec concepts directly. For example,
`uni20::strides()` accepts `StridedMdspecLike`, and its tensor overload
accepts `StridedTensorView`. Both immediate and descriptor-backed inputs
therefore expose the same mapping information without acquiring a handle.
Contiguous-mapping analysis used by reshape order validation follows the same
rule; the view-producing reshape operations remain mdspan-only because they
must retain an immediately usable handle.

### Tensor-Level Concepts

`TensorView<T>` is the tensor-level counterpart. A model exposes:

- multidimensional metadata obtainable through `mdspec_of(tensor)`;
- `backend_selector()`;
- tensor extents and extent observers.

`mdspec_of(tensor)` calls `tensor.mdspec()` when that member is
available and otherwise calls `tensor.mdspan()`. It does not acquire, copy, or
transform data. The fallback result is the actual mdspan, not a wrapper:
`MdspanLike` is already the immediate case of `MdspecLike`.

The returned result must model `MdspecLike`.
`MutableTensorView<T>` applies the corresponding mutable mdspec
requirement. `ImmediateTensorView<T>` additionally requires `.mdspan()` and an
immediate normalized representation. This gives the direct refinement
relationships:

```text
MdspanLike refines MdspecLike
ImmediateTensorView refines TensorView
```

The object categories remain distinct: `ImmediateTensorView` does not itself model
`MdspanLike`, and `TensorView` does not itself model
`MdspecLike`. `mdspec_of(tensor)` retrieves the corresponding
descriptor without requiring every immediate tensor view to add a redundant
`mdspec()` member.

The public type aliases `tensor_mdspec_t<T>` and
`mutable_tensor_mdspec_t<T>` name the normalized readable and writable
descriptor types. `tensor_storage_policy_t<T>` names a tensor's storage policy,
or `void` when no policy is exposed. Backend and tensor modules use these
aliases rather than depending on the normalization traits in
`uni20::detail`.

Rank, stride, and mutability use the same Cartesian naming for immediate and
tensor views. Rank-zero convenience concepts are
`ScalarImmediateTensorView`, `MutableScalarImmediateTensorView`, `ScalarTensorView`, and
`MutableScalarTensorView`.

The acquisition result concepts are:

```cpp
uni20::ReadMdspanLease<T>
uni20::WriteMdspanLease<T>
uni20::HostReadMdspanLease<T>
uni20::HostWriteMdspanLease<T>

uni20::ReadTensorLease<T>
uni20::WriteTensorLease<T>

uni20::HostReadTensorLease<T>
uni20::HostWriteTensorLease<T>
uni20::CudaReadTensorLease<T>
uni20::CudaWriteTensorLease<T>
```

Mdspan leases are move-only, policy-free access lifetimes with `.mdspan()` and
`release()`. They deliberately do not model `ImmediateTensorView`; a
fixed-operation backend no longer needs a backend selector or storage policy
after dispatch has selected that backend. Tensor leases additionally model
`ImmediateTensorView` and expose `.backend_selector()`. A write tensor lease is
a `MutableImmediateTensorView`, while its const interface returns a
const-element mdspan. Write leases deliberately expose no storage observer:
element mutation goes through `.mdspan()`, while structural mutation of the
owner remains unavailable during the lease.

Uni20 provides the generic `read_mdspan_lease`, `write_mdspan_lease`,
`read_tensor_lease`, and `write_tensor_lease` class templates, but an
acquisition backend may return any type satisfying the relevant concepts.
Generic lease components must be nothrow move-constructible, and their move
assignment participates only when every transferred component is nothrow
move-assignable. This prevents a failed partial move from separating a resolved
mdspan from the access state that keeps its handle valid.

### Execution-Domain Concepts

The shape of `data_handle_type` does not determine where an mdspan can be used.
In particular, a CUDA device pointer and a host pointer may both be spelled
`T*`. The accessor explicitly opts into the execution domains in which its
`access(...)` operation is valid.

The mdspan concepts are:

```cpp
uni20::HostAccessibleAccessor<Accessor>
uni20::CudaAccessibleAccessor<Accessor>

uni20::HostAccessibleMdspan<Span>
uni20::CudaAccessibleMdspan<Span>

uni20::HostAccessibleMdspec<Span>
uni20::CudaAccessibleMdspec<Span>
```

The accessor concepts classify accessor policies directly. The first mdspan
pair requires a resolved `MdspanLike`; the second mdspan pair also accepts
descriptor-backed `MdspecLike` metadata targeting that domain.
`stdex::default_accessor<T>` is host-accessible.
`uni20::cuda::CudaPointerAccessor<T>` is CUDA-accessible. Const adaptation,
conjugation, and elementwise transform accessors preserve the domains supported
by their wrapped accessors.

Domain registration is a semantic contract, not reflection over every
expression in `access(...)`. A function object evaluated by a CUDA accessor
must make the invoked call operator device-callable with
`UNI20_HOST_DEVICE`. Missing annotations are diagnosed when an actual CUDA
kernel instantiates the accessor.

Custom accessors opt in through:

```cpp
template<>
inline constexpr bool
    uni20::enable_accessor_in_domain<MyAccessor,
                                     uni20::host_access_domain> = true;
```

This is a semantic declaration about where the accessor may be evaluated. It
must not be inferred from the handle type. An accessor for unified or mapped
memory may opt into both domains; its adapted forms then inherit both
registrations.

### Execution-Descriptor Callability

Mappings and accessors form the value-semantic execution descriptor that a
backend may eventually pass to a leaf kernel. Uni20-owned execution surfaces
are therefore host/device-callable:

- mapping construction, observation, and multidimensional offset calculation;
- accessor construction, `access(...)`, `offset(...)`, and accessor-state
  observation;
- proxy-reference conversion and assignment used by an accessor;
- stored generator and transform function objects invoked by those accessors.

These functions use `UNI20_HOST_DEVICE`. A stored function object must apply
the same annotation to the call operator that the accessor invokes. Uni20 does
not add a function-domain registration trait: an omitted annotation is a code
generation error, and an actual CUDA compilation probe is the authoritative
diagnostic.

Device-callability and memory-domain accessibility remain independent. A
mapping performs index arithmetic and needs no execution-domain registration.
An accessor still opts into `host_access_domain`, `cuda_access_domain`, or both,
because that declaration says where its acquired handle may be evaluated.
Annotating `access(...)` does not make a host pointer CUDA-accessible or a CUDA
pointer host-accessible.

Execution-descriptor state must be safely copyable into the selected leaf
kernel and must not retain host-only temporary state. The data descriptor is
different: it belongs to backend acquisition and may retain buffers, files,
communicators, or other host-side control state. The backend resolves or lowers
that descriptor before launch; it does not pass the descriptor itself to a
kernel merely because the mapping and accessor are device-callable.

Uni20 keeps the same C++ descriptor types in host and CUDA translation units.
CUDA builds enable nvcc's relaxed constexpr mode so value-semantic
`std::array` and `std::tuple` members remain usable without conditionally
changing public types to `cuda::std` equivalents.

A device-callable descriptor is not automatically accepted by every
precompiled backend. Ordinary C++ callers cannot instantiate an arbitrary new
CUDA kernel specialization inside an already-built Uni20 library. A backend
that ships precompiled CUDA code must either:

1. use a type-erased execution plan with sufficient semantics;
2. register and explicitly instantiate a typed lowering; or
3. cleanly decline the descriptor.

This distinction lets external mappings and accessors participate in CUDA
compilation without pretending that every installed backend already contains
their execution code.

### Access-State Lifetime Contract

An acquired lease and the access state retained inside it follow the same
one-owner lifetime rules:

- `release()` is `noexcept` and idempotent. Its first call ends active access;
  later calls have no effect.
- Moving an active lease or access state transfers responsibility for the
  access lifetime and leaves the source inactive.
- Releasing or destroying a moved-from object has no effect.
- Move assignment, when supported, releases the destination's previous access
  before transferring the source access.

The generic leases use the presence of their resolved mdspan as their active
marker. They invalidate every resolved mdspan before releasing the retained
access state. The access state must still implement idempotent release because
it may itself be an RAII guard whose destructor calls `release()`.

## Acquisition

Include the tensor acquisition API through:

```cpp
#include <uni20/tensor/tensor.hpp>
```

Acquisition names the execution domain of the resulting mdspan:

```cpp
auto host_read = uni20::acquire_host_read_access_sync(host_tensor);
auto host_write = uni20::acquire_host_write_access_sync(host_tensor);

auto cuda_read =
    co_await uni20::acquire_cuda_read_access_async(cuda_tensor, stream);
auto cuda_write =
    co_await uni20::acquire_cuda_write_access_async(cuda_tensor, stream);

auto descriptor = cuda_tensor.mdspec();
auto cuda_descriptor_write =
    co_await uni20::acquire_cuda_write_access_async(descriptor, stream);
```

The suffix names the completion model consistently across execution domains.
`_sync` returns a lease directly after completing any caller-side
synchronization. `_async` returns an awaitable; for CUDA, acquisition is
ordered on the supplied stream and need not block the host. There are no
unsuffixed acquisition aliases.

These are constrained overloads, not required members of the data descriptor.
This keeps storage and resource policy in the acquisition layer. A descriptor
may identify a buffer that can be mapped, migrated, prefetched, or admitted to
a device without imposing those operations on every descriptor type.

Backend type probes use the public acquisition contracts:

```cpp
uni20::HostReadableMdspec<Input>
uni20::HostWritableMdspec<Output>
uni20::host_read_mdspan_t<Input>
uni20::host_write_mdspan_t<Output>
```

The concepts require the corresponding acquisition expression and lease
contract; they do not expose the acquisition implementation. Access-state
implementations satisfy the public `uni20::TensorAccessState` concept.
Moving a `TensorAccessState` is required not to throw; moving an active state
transfers its release responsibility and leaves the source inactive.
CUDA-buffer-specific acquisition uses
`uni20::cuda::BufferMdspec`, which identifies a CUDA-accessible
descriptor backed by `cuda::CudaBufferView`. These are shared backend-author
contracts and are intentionally not members of `uni20::detail`.

Acquisition never performs an implicit cross-domain transfer. It resolves
synchronization, resource admission, and handle lifetime within the named
domain. Host-to-CUDA, CUDA-to-host, and CUDA-to-CUDA movement use explicit
`copy` operations. A future managed or mapped representation may explicitly
model accessibility in more than one domain, but ordinary host and CUDA
storage do not.

For an immediately host-accessible tensor view, host acquisition is a borrowed
no-op lifetime guard. It depends only on the source `ImmediateTensorView`, not
on a public storage observer. The concrete immediate leases store only a
pointer to the source view and forward `.mdspan()`, extents, and backend
selection. Read leases may also forward a read-only storage observer. They do
not copy the mdspan or backend selector:

```cpp
uni20::Tensor<float, 2> tensor(4, 8);

{
  auto lease = uni20::acquire_host_write_access_sync(tensor);
  static_assert(uni20::HostWriteTensorLease<decltype(lease)>);
  lease.mdspan()[2, 3] = 1.0F;
}
```

`acquire_host_read_access_async(tensor)` and
`acquire_host_write_access_async(tensor)` provide always-ready awaitables for
the same immediate case.

The borrowed overloads accept lvalues only because their leases retain a
reference to the source tensor or view. Descriptor-backed acquisition instead
retains whatever backend-specific state makes the resolved handle usable. CUDA
also accepts an owning tensor rvalue for read acquisition: it captures the
mdspec metadata, moves the `CudaBuffer` into a distinct owning access
state, and resolves the mdspan against that owned buffer. Non-owning
descriptor-backed views remain lvalue-only.

CUDA buffer descriptors also support direct acquisition. These overloads return
policy-free `CudaReadMdspanLease` or `CudaWriteMdspanLease` models rather than
tensor leases:

```cpp
auto descriptor = std::as_const(tensor).mdspec();
auto lease = uni20::acquire_cuda_read_access_sync(descriptor);

static_assert(uni20::CudaReadMdspanLease<decltype(lease)>);
static_assert(!uni20::ImmediateTensorView<decltype(lease)>);
```

The descriptor and its access state refer to the original `CudaBuffer`; they do
not take ownership of it. The buffer owner must outlive the descriptor lease.
The tensor overloads remain useful when the caller needs a lease that models
`ImmediateTensorView`, retains backend selection, or consumes an owning tensor
rvalue.

### Host Backend Lowering

After tensor-level selector resolution, a fixed operation normalizes its
operands to `MdspecLike` descriptors. A host backend accepts those
descriptors without distinguishing immediate from deferred host storage. It
acquires the required host read and write leases, then invokes its
backend-specific lower-level mdspan implementation:

```cpp
auto output_access = acquire_host_write_access_sync(output);
auto input_access = acquire_host_read_access_sync(input);

return cpu_reference::copy(
    output_access.mdspan(),
    input_access.mdspan());
```

The operation-tag `try_kernel` customization does not re-enter operation
dispatch or call another operation-tag overload with the resolved mdspans.
Instead it calls a function whose namespace or name identifies the selected
backend, such as `cpu_reference::copy`, `blas::try_gemm`, or
`cuda_reference::copy`.

The synchronous CPU reference kernels and direct BLAS adapters use this
lowering pattern for fixed GEMM, GEMV, and copy. For an ordinary host
`ImmediateTensorView`, the normalized descriptor is an mdspan and acquisition creates a
small no-op lease; the forwarding optimizes to the direct mdspan path. A
descriptor-backed implementation may block while producing the same
mdspan-lease interface.

The descriptor-level `kernel_accepts_types` overload derives the exact resolved
mdspan types returned by acquisition and delegates to the backend-specific
lower-level acceptance helper. That helper is the single source of truth for
accessor, scalar, rank, layout, and assignment eligibility; it is not another
operation-tag customization point.

Descriptor-native backends do not use this host lowering when their
execution model requires more context. For example, cuBLAS inspects CUDA
descriptors and acquires buffer access against its selected stream.

Selector resolution precedes device-view normalization. Fixed-operation
frontends use tensor storage policy to choose the backend list, then pass each
normalized `MdspecLike` representation into the backend walk. A backend
either acquires a lease or interprets the descriptor directly. `CudaStorage`
may provide the default backend list, but it does not make a tensor a
`TensorView` and does not by itself make a backend overload participate.
CUDA-specific lowering identifies a `CudaBufferView` from the normalized
descriptor and checks the accessor independently.

### CUDA Vertical Slice

`CudaTensor::mdspec()` stores:

```text
(CudaBufferView descriptor, tensor mapping, CudaPointerAccessor)
```

Its accessor declares the eventual `T*` or `T const*` data-handle type. Ordinary
real scalars resolve to `T&` or `T const&`. Persistent
`uni20::complex<Real>` storage resolves through a mutable proxy or a
`cuda::std::complex<Real>` execution value so device code never treats the
standard and CUDA complex class types as alias-compatible objects. The
unresolved object contains no pointer and provides no indexed access. CUDA
acquisition resolves the descriptor through `CudaBuffer` access state:

```cpp
auto stream = co_await uni20::cuda::acquire_stream(
    tensor.storage().resources().streams());
auto lease = co_await uni20::acquire_cuda_write_access_async(tensor, stream);

static_assert(uni20::CudaWriteTensorLease<decltype(lease)>);
launch_kernel(stream, lease.mdspan());
```

Acquiring the stream is separate from acquiring tensor data. Once a stream is
available, `acquire_cuda_write_access_async(tensor, stream)` installs predecessor
waits through `CudaBuffer` and returns an immediately-ready task awaitable.
Destroying the lease records the writer completion at the stream tail. A later
synchronized or stream-ordered CUDA access observes that completion.

Host-synchronized CUDA-domain acquisition is also available:

```cpp
auto lease = uni20::acquire_cuda_read_access_sync(std::as_const(tensor));
copy_to_host_synchronously(lease.mdspan().data_handle());
```

The synchronization waits on the host, but the resolved mdspan remains
CUDA-accessible and is not host-dereferenceable. The explicit copy in the
example is the domain crossing.

An owning rvalue transfers its CUDA buffer into the read lease:

```cpp
auto lease = uni20::acquire_cuda_read_access_sync(std::move(tensor));
```

The owning CUDA access state does not retain a guard pointer into the original
tensor. It owns the moved buffer directly, so the lease may move without
invalidating its data handle.

All device operations using a synchronized lease must complete before the
lease is destroyed. Stream-ordered work should use the stream overloads so
lease release can publish the completion event.

CUDA copy and fixed GEMM accept descriptor-backed metadata without making
`CudaTensor` an `ImmediateTensorView`. Their tensor frontends select the backend, normalize
the operands to CUDA `mdspec()` descriptors, and pass those descriptors
through `dispatch_kernel` or `co_dispatch_kernel`. CPU reference GEMM uses the
host descriptor lease interface. The cuBLAS synchronous path blocks for stream
and provider resources, while its async path awaits them.

Raw contiguous CUDA copies with matching physical order retain the
`cudaMemcpyAsync` fast path. The CUDA reference elementwise executor handles
same-device positive-strided mappings with compact rank through eight, including
differing input/output strides, padding, nonzero buffer-view offsets, and the
compiled raw or conjugating accessor lowerings. A backend-neutral host plan
orders dimensions from the output mapping and coalesces only when every
operand's strides are jointly adjacent. CUDA lowering selects 32-bit indices
only when the logical count and every reachable operand offset fit, otherwise
using a 64-bit payload. The kernel decodes the compact plan into independent
input and output offsets before evaluating the accessors. The operation
publishes read/write completion through the same stream-ordered buffer access.

Non-strided mappings and unregistered stateful accessor compositions remain
valid device-callable descriptors but are not yet part of this precompiled copy
executor. They cleanly decline until a typed lowering registry or a sufficiently
general execution plan is available. Same-buffer copies with distinct
descriptor offsets use one exclusive access state and rely on the public C++
precondition that input and output do not destructively overlap. A transformed
copy at the same descriptor offset is proven to overlap and declines.

The same affine executor also provides registered `transform_op` lowerings for
raw CUDA input and output accessors: unary `negate`, `square`, and `reciprocal`;
stateful `scale<Factor>`; and binary `add`, `subtract`, `multiply`, and `divide`.
These are explicit typed registrations, not evidence that arbitrary callable
state can cross the precompiled library boundary.

## Data Descriptor Boundary

A data descriptor is an opaque description of the logical region associated
with the missing handle. The current structural concept requires only:

- a `data_descriptor_type` alias;
- a `data_descriptor()` observer.

The concept deliberately prescribes no other descriptor operations or internal
representation.

Owning a data descriptor does not necessarily own the underlying storage. A
descriptor-backed view remains subject to the lifetime contract of its
descriptor implementation.

Operation frontends copy fixed descriptor values before dispatch. That value
copy retains the mapping, accessor, allocation identity, and element offset
needed by a backend, but it does not by itself extend storage lifetime. A
synchronous tensor call keeps its operands alive through the backend walk. An
async tensor call must retain the applicable epoch storage for as long as a
copied descriptor or acquired access state refers to it.

Prefetch is likewise descriptor- or storage-specific. It can improve later
acquisition without changing the `mdspec` structural contract.

## Accessor Boundary

The stored accessor is an actual AccessorPolicy object, not an accessor recipe
or factory. It defines:

- the eventual `data_handle_type`;
- the presented `element_type` and `reference`;
- the semantics of `access(handle, offset)`;
- handle offsetting behavior.

The class preserves the accessor object exactly as supplied and does not
automatically change its type, state, or constness.

## Current Invariants

- A concrete `mdspec` stores a descriptor, mapping, and actual accessor.
- It contains no data-handle value.
- It cannot be indexed before acquisition.
- Its mapping and accessor are the objects intended for the resolved mdspan.
- `MdspecLike` is structural and does not require the concrete class.
- `TensorView` and the lease concepts are structural and do not require
  Uni20's concrete materializations.
- Ordinary `MdspanLike` values satisfy `MdspecLike`.
- Tensor read and write leases model `ImmediateTensorView`, retain backend
  selection, and control handle validity through RAII. Read-only storage inspection is
  optional for tensor read leases; tensor write leases expose no storage
  observer.
- Mdspan read and write leases are policy-free and do not model `ImmediateTensorView`.
  They expose only the resolved mdspan and the RAII access lifetime.
- Lease and access-state release is idempotent, and moves transfer the active
  access lifetime while leaving the source inactive.
- Construction and metadata observation perform no handle acquisition.

## Source and Tests

- [`mdspec.hpp`](../../src/uni20/mdspan/mdspec.hpp)
- [`concepts.hpp`](../../src/uni20/mdspan/concepts.hpp)
- [`tensor/access.hpp`](../../src/uni20/tensor/access.hpp)
- [`tensor/cuda_access.hpp`](../../src/uni20/tensor/cuda_access.hpp)
- [`test_mdspec.cpp`](../../tests/mdspan/test_mdspec.cpp)
- [`test_tensor.cpp`](../../tests/tensor/test_tensor.cpp)
- [`test_cuda_tensor.cpp`](../../tests/backend/cuda/test_cuda_tensor.cpp)
- [`test_cuda_descriptor_compile.cu`](../../tests/linalg/test_cuda_descriptor_compile.cu)
- [`test_cuda_copy.cpp`](../../tests/linalg/test_cuda_copy.cpp)

The tests cover concrete storage of stateful descriptor/mapping/accessor
objects, independent structural models, ordinary-mdspan compatibility, ranked
and strided refinements, move-only descriptors, the absence of premature handle
access, immediate tensor leases, idempotent generic release, access-state
transfer across lease moves, and CUDA pointer resolution through synchronized
and stream-ordered leases. CUDA compilation probes cover canonical, generated,
zipped, const-adapted, conjugated, and stateful transform descriptor
compositions. CUDA copy tests cover physical-order conversion, padded strides,
and nonzero buffer offsets in addition to contiguous and conjugating transfers.
