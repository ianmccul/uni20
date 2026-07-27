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

It therefore does not model `SpanLike`.

## Structural Concepts

The concepts are the primary generic interface. Code should not require the
concrete `device_mdspan` class when a structural constraint is sufficient.

### `DeviceSpanLike`

`DeviceSpanLike<S>` accepts either:

1. an ordinary `SpanLike<S>`, as the immediate-handle case; or
2. an unresolved structural model exposing mdspan metadata plus
   `data_descriptor_type` and `data_descriptor()`.

The unresolved model must expose compatible element, extent, layout, mapping,
accessor, handle, and reference aliases. Its accessor must satisfy
`AccessorPolicy`.

An independent type can therefore model `DeviceSpanLike` without inheriting
from or converting to `device_mdspan`.

The concept provides a common metadata vocabulary, but not one common handle
operation. Generic code can distinguish the two cases explicitly:

```cpp
template<uni20::DeviceSpanLike Span>
void inspect(Span const& span)
{
  inspect_mapping(span.mapping());
  inspect_accessor(span.accessor());

  if constexpr (uni20::SpanLike<Span>)
    inspect_immediate_handle(span.data_handle());
  else
    inspect_descriptor(span.data_descriptor());
}
```

### Rank and Stride Refinements

The implemented refinements are:

```cpp
RankedDeviceSpanLike<Span, Rank>
StridedDeviceSpanLike<Span>
RankedStridedDeviceSpanLike<Span, Rank>
```

Like `DeviceSpanLike`, these accept both immediate mdspans and independent
descriptor-backed models.

`MutableDeviceSpanLike<Span>` additionally requires non-const element semantics
and either an assignable accessor reference or an explicit backend-writable
accessor opt-in.

### Tensor-Level Concepts

`DeviceTensorView<T>` is the tensor-level counterpart. A model exposes:

- a `device_mdspan()` whose result models `DeviceSpanLike`, or an ordinary
  `mdspan()` for the immediate case;
- `backend_selector()`;
- tensor extents and extent observers.

When a type explicitly provides `device_mdspan()`, that representation governs
the concept. `MutableDeviceTensorView<T>` applies the corresponding mutable
device-span requirement.

The acquisition result concepts are:

```cpp
uni20::ReadTensorLease<T>
uni20::WriteTensorLease<T>
```

Both are move-only `TensorView` models with `.storage()`, `.backend_selector()`,
`.mdspan()`, and `release()`. A write lease is a `MutableTensorView`, while its
const interface returns a const-element mdspan and const storage.

Uni20 provides the generic `read_tensor_lease` and `write_tensor_lease` class
templates, but an acquisition backend may return any type satisfying the
concepts.

## Acquisition

Include the tensor acquisition API through:

```cpp
#include <uni20/tensor/tensor.hpp>
```

The common free-function vocabulary is:

```cpp
auto read_lease = uni20::blocking_read_access(tensor);
auto write_lease = uni20::blocking_write_access(tensor);

auto read_awaitable = uni20::read_access(tensor, acquisition_arguments...);
auto write_awaitable = uni20::write_access(tensor, acquisition_arguments...);
```

These are constrained overloads, not required members of the data descriptor.
This keeps storage and resource policy in the acquisition layer. A descriptor
may identify a buffer that can be mapped, migrated, prefetched, or admitted to
a device without imposing those operations on every descriptor type.

For an immediately accessible host tensor, blocking acquisition is a no-op
lifetime guard:

```cpp
uni20::Tensor<float, 2> tensor(4, 8);

{
  auto lease = uni20::blocking_write_access(tensor);
  static_assert(uni20::WriteTensorLease<decltype(lease)>);
  lease.mdspan()[2, 3] = 1.0F;
}
```

`read_access(tensor)` and `write_access(tensor)` provide always-ready awaitables
for the same immediate case.

### CUDA Vertical Slice

`CudaTensor::device_mdspan()` stores:

```text
(CudaBufferView descriptor, tensor mapping, CudaPointerAccessor)
```

Its accessor declares the eventual `T*` or `T const*` data-handle type, but the
unresolved object contains no pointer. CUDA acquisition resolves the descriptor
through `CudaBuffer` access state:

```cpp
auto stream = co_await uni20::cuda::acquire_stream(
    tensor.storage().resources().streams());
auto lease = co_await uni20::write_access(tensor, stream);

static_assert(uni20::WriteTensorLease<decltype(lease)>);
launch_kernel(stream, lease.mdspan());
```

Acquiring the stream is separate from acquiring tensor data. Once a stream is
available, `write_access(tensor, stream)` installs predecessor waits through
`CudaBuffer` and returns an immediately-ready task awaitable. Destroying the
lease records the writer completion at the stream tail. A later blocking or
stream-ordered access observes that completion.

Blocking CUDA acquisition is also available:

```cpp
auto lease = uni20::blocking_read_access(std::as_const(tensor));
copy_to_host_synchronously(lease.mdspan().data_handle());
```

All device operations using a blocking lease must complete before the lease is
destroyed. Stream-ordered work should use the stream overloads so lease release
can publish the completion event.

This is the first acquisition vertical slice. Existing CUDA tensor operations
have not all been migrated to consume `DeviceTensorView` leases yet.

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
- `DeviceSpanLike` is structural and does not require the concrete class.
- `DeviceTensorView` and the lease concepts are structural and do not require
  Uni20's concrete materializations.
- Ordinary `SpanLike` values satisfy `DeviceSpanLike`.
- A read or write lease models `TensorView`, retains storage and backend
  selection, and controls handle validity through RAII.
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
access, immediate tensor leases, and CUDA pointer resolution through blocking
and stream-ordered leases.
