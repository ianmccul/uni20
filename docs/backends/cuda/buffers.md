# CUDA Buffers

**Status:** current guide for the implemented low-level CUDA buffer API.

This guide introduces `uni20::cuda::CudaBuffer<T>` for developers writing CUDA
kernel and provider backends. Application Tensor code normally reaches these
buffers through `CudaTensor` rather than constructing them directly.

## The Mental Model

Treat a CUDA buffer like an ordinary mutable value whose device work may finish
later:

- any number of reads may overlap;
- a write cannot overlap another write or any read;
- a buffer cannot be moved, destroyed, or synchronized while an access object
  refers to it;
- releasing access submits a completion marker; it does not wait for the GPU.

`CudaBuffer<T>` keeps a completion ledger behind this value-like API. A
later operation can be submitted immediately on another stream. The buffer
inserts CUDA event waits so that device execution still follows the established
read/write order.

## Creating A Buffer

A scoped CUDA runtime owns one canonical `DeviceResources` for each enrolled
device. Each resource set owns that device's stream pool and lazily constructed
provider-resource pools such as the cuBLAS execution pool. The runtime must
outlive every buffer, stream, and provider lease acquired from it.

```cpp
#include <uni20/backend/cuda/buffer.hpp>

using namespace uni20::cuda;

auto cuda_lifetime = initialize({
    .device_ordinals = {0},
    .default_device = 0,
    .streams_per_device = 4,
});

CudaBuffer<float> values(1024);
```

The allocation uses CUDA's stream-ordered memory pool when the device supports
it, with `cudaMalloc` as the fallback. `CudaBuffer<T>` is move-only. Direct
construction from an explicit `DeviceResources&` selects another enrolled
device or an isolated resource set used by tests.

## Tensor Storage

`CudaTensor<T, Rank>` is the owning Tensor form for CUDA device storage:

```cpp
#include <uni20/tensor/tensor.hpp>

uni20::CudaTensor<float, 2> matrix(32, 48);
```

The Tensor owns a `CudaBuffer<float>` and preserves ordinary extents and layout
metadata. Its unresolved device mdspan contains a non-owning
`cuda::CudaBufferView<float>` descriptor with buffer identity and an element
offset; it provides no indexed access. A CUDA operation must lower the view
through `read_synchronized_with(stream)` or
`write_synchronized_with(stream)` before a leaf backend receives a raw pointer.

The resolved `CudaPointerAccessor<T>` uses `T*` as its data handle and `T&` as
its reference type. Its indexed access follows ordinary mdspan semantics, but
must be evaluated only in an execution domain where that CUDA pointer is
directly accessible.

For GEMM this lowering is reached through the Async Tensor API:

```cpp
uni20::linalg::assign_product(output, lhs, rhs, 1.0F);
```

The Async front end schedules a `CudaTask`, resolves the Tensor device after
its epochs are ready, and awaits a cuBLAS handle-plus-stream lease. The backend
then validates layouts and devices, opens one write access and two read
accesses, and calls cuBLAS. Resource exhaustion suspends the CUDA task rather
than blocking a scheduler participant.

## Submitting Work

Raw device pointers are available only through scoped access objects:

```cpp
auto stream = device_resources().streams().acquire();

{
  auto output = values.write_synchronized_with(stream);

  check(cudaMemsetAsync(output.data(), 0, output.size_bytes(),
                        stream.native_handle()),
        "cudaMemsetAsync values", values.device().ordinal());
}
```

`WriteAccess<T>` exposes `T*`. Its destructor records a completion at the
current stream tail and publishes that completion to the buffer. The access
object retains the stream lease, so the stream cannot return to the pool too
early.

Read access works the same way and exposes `T const*`:

```cpp
auto input = values.read_synchronized_with(stream);
launch_read_only_kernel(input.data(), input.size(), stream.native_handle());
```

Keep an access object alive until every operation using its pointer has been
submitted to that object's stream. Do not retain the pointer after releasing
the access.

## Several Operands

A typical kernel submission acquires every operand on one chosen stream:

```cpp
auto stream = device_resources().streams().acquire();
auto out = output.write_synchronized_with(stream);
auto a = lhs.read_synchronized_with(stream);
auto b = rhs.read_synchronized_with(stream);

launch_product(out.data(), a.data(), b.data(), stream.native_handle());

a.release();
b.release();
out.release();
```

Explicit `release()` is an idempotent early cleanup operation. Lexical
destruction has the same effect and is usually simpler. Early release is useful
when the caller wants to submit a causally later operation before leaving the
current scope.

An in-place operation uses one `WriteAccess<T>` for the mutated buffer. Do not
acquire a separate read access to the same buffer.

## Access Rules

The host-side access objects enforce ordinary read/write rules immediately:

| Existing access | New read | New write |
|---|---:|---:|
| none | allowed | allowed |
| one or more readers | allowed | contract violation |
| writer | contract violation | contract violation |

A conflicting acquisition does not queue, block until the other object is
released, or suspend a coroutine. It is a programming error and fails through
Uni20's contract diagnostics.

This is different from `Async<T>` and its `ReadBuffer<T>`/`WriteBuffer<T>`
epochs. The async layer may represent work whose producer has not run yet. CUDA
buffer access only lowers an order that ordinary C++ control flow or the async
layer has already established.

## What Release Means

Releasing an access object:

1. records a CUDA event at the retained stream's current tail;
2. publishes that completion to the buffer;
3. returns the live host-access token;
4. leaves the access object inert.

Successful completion publication does not call `cudaStreamSynchronize`. A
later writer can be acquired as soon as all earlier read objects have released,
even when their GPU work is still running. The writer's stream waits on the
published reader completions. If event recording or publication fails during
cleanup, the implementation synchronizes the retained stream before returning
the live token rather than publishing a false dependency.

Use `buffer.synchronize()` only when the host genuinely needs all currently
published work involving that buffer to finish. No access object may be live
when it is called.

## Blocking Host Access

Pageable host transfers use `blocking_read_access()` and
`blocking_write_access()` rather than acquiring a stream. Construction claims
the same reader or writer token described above and waits on the host for the
relevant completion ledger. The guarded CUDA runtime operation must finish
before the access object is released. Release returns the live token but records
no event because there is no outstanding device work to publish.

These guards are intended for synchronous calls such as pageable-host
`cudaMemcpy`. CUDA issues synchronous copies through the default stream and may
block or synchronize for reasons beyond the dependencies recorded in one
buffer's ledger. Callers must therefore treat pageable host transfer as a broad
blocking boundary rather than rely on ledger-local concurrency. It is not a
CUDA stream-capture path. Pageable host-to-device `cudaMemcpy` may return after
host staging but before its device DMA completes. That path records the default
stream tail and passes the completion to
`BlockingWriteAccess::release_with_completion()`, preserving the outstanding
destination dependency without synchronizing the device. Other blocking guard
uses must finish their device access before release. Pinned host storage and a
completion awaiter will provide a separate genuinely non-blocking host-transfer
path later.

## Streams On Another Device

An access object may be synchronized with a stream from another device when the
CUDA operation itself permits that combination. This supports a peer copy using
one destination-device stream:

```cpp
auto source_access = source.read_synchronized_with(destination_stream);
auto destination_access =
    destination.write_synchronized_with(destination_stream);

check(cudaMemcpyPeerAsync(
          destination_access.data(), destination.device().ordinal(),
          source_access.data(), source.device().ordinal(),
          source_access.size_bytes(), destination_stream.native_handle()),
      "cudaMemcpyPeerAsync", destination.device().ordinal());
```

Synchronization does not make arbitrary foreign-device pointer access legal.
The kernel or provider API must independently support the devices and pointers
being supplied.

## Async Integration

CUDA buffer access itself has no awaiter. Once a stream is available, acquiring
access performs only bounded host work: contract checks, completion snapshots,
and CUDA event-wait submission.

Non-blocking CUDA operations await resources that may be temporarily
unavailable, such as streams or provider handles. Tensor-level lowering will
then use the same `read_synchronized_with()` and
`write_synchronized_with()` API.

## Current Boundary

The implemented boundary includes low-level allocation, stream, completion,
and access semantics; CUDA Tensor storage descriptors; contiguous host/device,
device/host, device/device, and peer transfer; and both direct and
coroutine-aware Tensor-to-cuBLAS matrix-product lowering. GEMM is currently
provider-backed rather than a native Uni20 CUDA kernel. Direct dispatch may
block during resource admission; coroutine dispatch awaits the same resources
before entering the non-suspending backend leaf.

General CUDA Tensor operation coverage, native Uni20 CUDA kernels, and automatic
storage-driven CUDA scheduler selection are not yet implemented.

For the exact completion-ledger and failure contract, see [CUDA Buffer
Completion Lowering](epoch_design_draft.md). For stream ownership and error
handling, see the [CUDA Runtime Model](runtime.md).
