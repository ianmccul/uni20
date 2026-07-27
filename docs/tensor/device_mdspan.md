# Device Mdspan

**Status:** current guide for the implemented class and structural concepts.

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

The concrete lease and acquisition APIs are separate from `device_mdspan` and
are not yet implemented. The descriptor/mapping/accessor split establishes the
structural contract they will consume.

## Meaning of Device

Here, a device is a target data-access domain. It is not limited to a CUDA GPU
and does not describe where the authoritative data currently resides.

Examples include:

- host memory;
- CUDA memory;
- another accelerator or addressable memory space;
- mapped or staged storage.

The name describes the domain in which the eventual mdspan handle will be
used. It does not imply that the handle is already available.

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
| Descriptor | `data_descriptor()` |
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

There is not currently a mutable device-span refinement.

## Example: Buffer Requiring a Lease

Suppose a buffer cannot return a pointer directly. It must first perform an
operation that establishes read access and returns an RAII object:

```cpp
class leased_buffer
{
  public:
    using data_handle_type = float const*;

    class read_lease
    {
      public:
        [[nodiscard]] data_handle_type data() const noexcept;
        ~read_lease(); // Releases the buffer access.
    };

    // May wait, map, transfer, or reserve resources before returning.
    [[nodiscard]] read_lease lease_read();
};

struct buffer_region
{
    leased_buffer* buffer = nullptr;
    std::size_t element_offset = 0;
};

using extents_type = stdex::dextents<std::size_t, 2>;
using accessor_type = stdex::default_accessor<float const>;
using span_type = uni20::device_mdspan<
    float const,
    extents_type,
    stdex::layout_left,
    accessor_type,
    buffer_region>;

leased_buffer buffer;
extents_type extents{4, 8};
span_type::mapping_type mapping{extents};

span_type span{
    buffer_region{.buffer = &buffer, .element_offset = 16},
    mapping,
    accessor_type{}};
```

At this point `span` describes the array, but it still has no usable pointer
and cannot be indexed. The buffer-specific acquisition can resolve it:

```cpp
auto const& region = span.data_descriptor();
auto buffer_lease = region.buffer->lease_read();

auto handle =
    span.accessor().offset(buffer_lease.data(), region.element_offset);

using resolved_type = stdex::mdspan<
    float const,
    extents_type,
    stdex::layout_left,
    accessor_type>;

resolved_type resolved{
    handle,
    span.mapping(),
    span.accessor()};

consume(resolved);
```

`buffer_lease` remains alive while `resolved` is used. Its destructor releases
the access state after `resolved` leaves scope. The `leased_buffer` API above is
illustrative; Uni20's common acquisition and lease API has not yet been
implemented.

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
- Ordinary `SpanLike` values satisfy `DeviceSpanLike`.
- Construction and metadata observation perform no handle acquisition.

## Source and Tests

- [`device_mdspan.hpp`](../../src/uni20/mdspan/device_mdspan.hpp)
- [`concepts.hpp`](../../src/uni20/mdspan/concepts.hpp)
- [`test_device_mdspan.cpp`](../../tests/mdspan/test_device_mdspan.cpp)

The tests cover concrete storage of stateful descriptor/mapping/accessor
objects, independent structural models, ordinary-mdspan compatibility, ranked
and strided refinements, move-only descriptors, and the absence of handle and
indexing operations.
