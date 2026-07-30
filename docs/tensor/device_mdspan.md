# Device Mdspan

**Status:** current guide for the implemented class, structural concepts, and
initial tensor acquisition layer.

`uni20::device_mdspan` describes a multidimensional array whose shape, layout,
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

A `device_mdspan` contains:

```text
device_mdspan = (data_descriptor, mapping, accessor)
```

The mapping and accessor are the actual objects intended for a later resolved
mdspan. The data descriptor identifies the logical storage region from which a
compatible data handle can be leased.

```text
device_mdspan + acquisition operation
        |
        v
RAII access lease containing a usable mdspan
```

Acquisition is a tensor-level operation. It returns a move-only RAII object that
models `TensorView`, retains the source storage and backend selector, and exposes
the resolved mdspan through `.mdspan()`.

## Meaning of Device

Here, a device is a target data-access domain. It is not limited to a CUDA GPU
and does not describe where the authoritative data currently resides.

Examples include:

- host memory;
- CUDA memory;
- another accelerator or addressable memory space;
- mapped or staged storage;
- data demand-loaded from host, disk, or a network service into an execution
  domain.

The name describes the domain in which the eventual mdspan handle will be
used. It does not imply that the handle is already available, or that the
authoritative copy already resides in that domain.

## Current Header and Template

Include:

```cpp
#include <uni20/mdspan/device_mdspan.hpp>
```

The implemented class template is:

```cpp
template<class ElementType,
         class Extents,
         class LayoutPolicy,
         AccessorPolicy Accessor,
         class DataDescriptor>
  requires std::same_as<ElementType, typename Accessor::element_type>
class device_mdspan;
```

The template deliberately follows the mdspan type vocabulary and appends the
data descriptor as the unresolved component. It currently has no default
template arguments.

Construction stores all three components by value:

```cpp
device_mdspan(
    data_descriptor_type descriptor,
    mapping_type mapping,
    accessor_type accessor);
```

Copy and move behavior follow the stored component types. In particular, a
move-only data descriptor produces a move-only `device_mdspan`.

## Type and Observer Contract

`device_mdspan` exposes the mdspan-compatible type aliases needed before handle
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

An unresolved `device_mdspan` intentionally does not expose:

- `data_handle()`;
- multidimensional `operator[]`;
- host or device dereference;
- implicit conversion to `stdex::mdspan`;
- allocation, migration, synchronization, or access acquisition.

It therefore does not model `MdspanLike`.

## Structural Concepts

The concepts are the primary generic interface. Code should not require the
concrete `device_mdspan` class when a structural constraint is sufficient.

### `DeviceMdspanLike`

`DeviceMdspanLike<S>` accepts either:

1. an ordinary `MdspanLike<S>`, as the immediate-handle case; or
2. an unresolved structural model exposing mdspan metadata plus
   `data_descriptor_type` and `data_descriptor()`.

For Uni20's concrete `Tensor`, immediate accessibility takes precedence over
descriptor availability. Its mutable and const `device_mdspan()` overloads
therefore return ordinary mdspans whenever the corresponding writable or
readable handle is immediately available, even if the storage policy also
provides a descriptor. Only a missing immediate handle selects the
metadata-only `uni20::device_mdspan` representation. Read, write, immediate,
and deferred capabilities are detected independently.

The unresolved model must expose compatible element, extent, layout, mapping,
accessor, handle, and reference aliases. Its accessor must satisfy
`AccessorPolicy`.

An independent type can therefore model `DeviceMdspanLike` without inheriting
from or converting to `device_mdspan`.

The concept provides a common metadata vocabulary, but not one common handle
operation. Generic code can distinguish the two cases explicitly:

```cpp
template<uni20::DeviceMdspanLike Span>
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
MutableDeviceMdspanLike<Span>
RankedDeviceMdspanLike<Span, Rank>
StridedDeviceMdspanLike<Span>
MutableRankedDeviceMdspanLike<Span, Rank>
MutableStridedDeviceMdspanLike<Span>
RankedStridedDeviceMdspanLike<Span, Rank>
MutableRankedStridedDeviceMdspanLike<Span, Rank>
```

Like `DeviceMdspanLike`, these accept both immediate mdspans and independent
descriptor-backed models.

The mutable forms additionally require non-const element semantics and an
assignable accessor reference. They describe eventual mutation after
acquisition and do not add indexing to an unresolved descriptor.

Metadata-only helpers use the device concepts directly. For example,
`uni20::strides()` accepts `StridedDeviceMdspanLike`, and its tensor overload
accepts `StridedDeviceTensorView`. Both immediate and descriptor-backed inputs
therefore expose the same mapping information without acquiring a handle.
Contiguous-mapping analysis used by reshape order validation follows the same
rule; the view-producing reshape operations remain mdspan-only because they
must retain an immediately usable handle.

### Tensor-Level Concepts

`DeviceTensorView<T>` is the tensor-level counterpart. A model exposes:

- multidimensional metadata obtainable through `device_mdspan_of(tensor)`;
- `backend_selector()`;
- tensor extents and extent observers.

`device_mdspan_of(tensor)` calls `tensor.device_mdspan()` when that member is
available and otherwise calls `tensor.mdspan()`. It does not acquire, copy, or
transform data. The fallback result is the actual mdspan, not a wrapper:
`MdspanLike` is already the immediate case of `DeviceMdspanLike`.

The returned result must model `DeviceMdspanLike`.
`MutableDeviceTensorView<T>` applies the corresponding mutable device-mdspan
requirement. This gives the direct refinement relationships:

```text
TensorView   is an immediate DeviceTensorView
MdspanLike   is an immediate DeviceMdspanLike
```

The object categories remain distinct: `TensorView` does not itself model
`MdspanLike`, and `DeviceTensorView` does not itself model
`DeviceMdspanLike`. `device_mdspan_of(tensor)` retrieves the corresponding
descriptor without requiring every immediate tensor view to add a redundant
`device_mdspan()` member.

Rank, stride, and mutability use the same Cartesian naming for immediate and
device tensor views. Rank-zero convenience concepts are
`ScalarTensorView`, `MutableScalarTensorView`, `ScalarDeviceTensorView`, and
`MutableScalarDeviceTensorView`.

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
`release()`. They deliberately do not model `TensorView`; a fixed-operation
backend no longer needs a backend selector or storage policy after dispatch has
selected that backend. Tensor leases additionally model `TensorView` and expose
`.backend_selector()`. A write tensor lease is a `MutableTensorView`, while its
const interface returns a const-element mdspan. Write leases deliberately
expose no storage observer: element mutation goes through `.mdspan()`, while
structural mutation of the owner remains unavailable during the lease.

Uni20 provides the generic `read_mdspan_lease`, `write_mdspan_lease`,
`read_tensor_lease`, and `write_tensor_lease` class templates, but an
acquisition backend may return any type satisfying the relevant concepts.

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

uni20::HostAccessibleDeviceMdspan<Span>
uni20::CudaAccessibleDeviceMdspan<Span>
```

The accessor concepts classify accessor policies directly. The first mdspan
pair requires a resolved `MdspanLike`; the second mdspan pair also accepts
descriptor-backed `DeviceMdspanLike` metadata targeting that domain.
`stdex::default_accessor<T>` is host-accessible.
`uni20::cuda::CudaPointerAccessor<T>` is CUDA-accessible. Const adaptation,
conjugation, and elementwise transform accessors preserve the domains supported
by their wrapped accessors.

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
auto host_read = uni20::acquire_host_read_access(host_tensor);
auto host_write = uni20::acquire_host_write_access(host_tensor);

auto cuda_read =
    co_await uni20::acquire_cuda_read_access(cuda_tensor, stream);
auto cuda_write =
    co_await uni20::acquire_cuda_write_access(cuda_tensor, stream);

auto descriptor = cuda_tensor.device_mdspan();
auto cuda_descriptor_write =
    co_await uni20::acquire_cuda_write_access(descriptor, stream);
```

These are constrained overloads, not required members of the data descriptor.
This keeps storage and resource policy in the acquisition layer. A descriptor
may identify a buffer that can be mapped, migrated, prefetched, or admitted to
a device without imposing those operations on every descriptor type.

Acquisition never performs an implicit cross-domain transfer. It resolves
synchronization, resource admission, and handle lifetime within the named
domain. Host-to-CUDA, CUDA-to-host, and CUDA-to-CUDA movement use explicit
`copy` operations. A future managed or mapped representation may explicitly
model accessibility in more than one domain, but ordinary host and CUDA
storage do not.

For an immediately host-accessible tensor view, host acquisition is a borrowed
no-op lifetime guard. It depends only on the source `TensorView`, not on a
public storage observer. The concrete immediate leases store only a pointer to
the source view and forward `.mdspan()`, extents, and backend selection. Read
leases may also forward a read-only storage observer. They do not copy the
mdspan or backend selector:

```cpp
uni20::Tensor<float, 2> tensor(4, 8);

{
  auto lease = uni20::acquire_host_write_access(tensor);
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
device-mdspan metadata, moves the `CudaBuffer` into a distinct owning access
state, and resolves the mdspan against that owned buffer. Non-owning deferred
views remain lvalue-only.

CUDA buffer descriptors also support direct acquisition. These overloads return
policy-free `CudaReadMdspanLease` or `CudaWriteMdspanLease` models rather than
tensor leases:

```cpp
auto descriptor = std::as_const(tensor).device_mdspan();
auto lease = uni20::acquire_cuda_read_access_sync(descriptor);

static_assert(uni20::CudaReadMdspanLease<decltype(lease)>);
static_assert(!uni20::TensorView<decltype(lease)>);
```

The descriptor and its access state refer to the original `CudaBuffer`; they do
not take ownership of it. The buffer owner must outlive the descriptor lease.
The tensor overloads remain useful when the caller needs a lease that models
`TensorView`, retains backend selection, or consumes an owning tensor rvalue.

### Host Backend Lowering

After tensor-level selector resolution, a fixed operation normalizes its
operands to `DeviceMdspanLike` descriptors. A host backend accepts those
descriptors without distinguishing immediate from deferred host storage. It
acquires the required host read and write leases, then invokes its
backend-specific lower-level mdspan implementation:

```cpp
auto output_access = acquire_host_write_access(output);
auto input_access = acquire_host_read_access(input);

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
`TensorView`, the normalized descriptor is an mdspan and acquisition creates a
small no-op lease; the forwarding optimizes to the direct mdspan path. A
deferred descriptor implementation may block while producing the same
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
normalized `DeviceMdspanLike` representation into the backend walk. A backend
either acquires a lease or interprets the descriptor directly. `CudaStorage`
may provide the default backend list, but it does not make a tensor a
`DeviceTensorView` and does not by itself make a backend overload participate.
CUDA-specific lowering identifies a `CudaBufferView` from the normalized
descriptor and checks the accessor independently.

### CUDA Vertical Slice

`CudaTensor::device_mdspan()` stores:

```text
(CudaBufferView descriptor, tensor mapping, CudaPointerAccessor)
```

Its accessor declares the eventual `T*` or `T const*` data-handle type and
returns `T&` or `T const&` from indexed access. The unresolved object contains
no pointer and provides no indexed access. CUDA acquisition resolves the
descriptor through `CudaBuffer` access state:

```cpp
auto stream = co_await uni20::cuda::acquire_stream(
    tensor.storage().resources().streams());
auto lease = co_await uni20::acquire_cuda_write_access(tensor, stream);

static_assert(uni20::CudaWriteTensorLease<decltype(lease)>);
launch_kernel(stream, lease.mdspan());
```

Acquiring the stream is separate from acquiring tensor data. Once a stream is
available, `acquire_cuda_write_access(tensor, stream)` installs predecessor
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

CUDA copy and fixed GEMM accept deferred metadata without making `CudaTensor`
an immediate `TensorView`. Their tensor frontends select the backend, normalize
the operands to CUDA `device_mdspan()` descriptors, and pass those descriptors
through `dispatch_kernel` or `co_dispatch_kernel`. CPU reference GEMM uses the
host descriptor lease interface. The cuBLAS synchronous path blocks for stream
and provider resources, while its async path awaits them.

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
acquisition without changing the `device_mdspan` structural contract.

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

- A concrete `device_mdspan` stores a descriptor, mapping, and actual accessor.
- It contains no data-handle value.
- It cannot be indexed before acquisition.
- Its mapping and accessor are the objects intended for the resolved mdspan.
- `DeviceMdspanLike` is structural and does not require the concrete class.
- `DeviceTensorView` and the lease concepts are structural and do not require
  Uni20's concrete materializations.
- Ordinary `MdspanLike` values satisfy `DeviceMdspanLike`.
- Tensor read and write leases model `TensorView`, retain backend selection,
  and control handle validity through RAII. Read-only storage inspection is
  optional for tensor read leases; tensor write leases expose no storage
  observer.
- Mdspan read and write leases are policy-free and do not model `TensorView`.
  They expose only the resolved mdspan and the RAII access lifetime.
- Lease and access-state release is idempotent, and moves transfer the active
  access lifetime while leaving the source inactive.
- Construction and metadata observation perform no handle acquisition.

## Source and Tests

- [`device_mdspan.hpp`](../../src/uni20/mdspan/device_mdspan.hpp)
- [`concepts.hpp`](../../src/uni20/mdspan/concepts.hpp)
- [`tensor/access.hpp`](../../src/uni20/tensor/access.hpp)
- [`tensor/cuda_access.hpp`](../../src/uni20/tensor/cuda_access.hpp)
- [`test_device_mdspan.cpp`](../../tests/mdspan/test_device_mdspan.cpp)
- [`test_tensor.cpp`](../../tests/tensor/test_tensor.cpp)
- [`test_cuda_tensor.cpp`](../../tests/backend/cuda/test_cuda_tensor.cpp)

The tests cover concrete storage of stateful descriptor/mapping/accessor
objects, independent structural models, ordinary-mdspan compatibility, ranked
and strided refinements, move-only descriptors, the absence of premature handle
access, immediate tensor leases, idempotent generic release, access-state
transfer across lease moves, and CUDA pointer resolution through synchronized
and stream-ordered leases.
