# CUDA Buffers

**Status:** current guide for the implemented low-level CUDA buffer API.

This guide introduces `uni20::cuda::CudaBuffer<T>` for developers writing CUDA
kernel and provider backends. Application Tensor code normally reaches these
buffers through `CudaAsyncTensor` rather than constructing them directly.

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

A `DeviceContext` owns the device's stream pool, the shared state used by its
buffers, and lazily constructed provider-resource pools such as the cuBLAS
execution pool. The context must outlive every buffer, stream, and provider
lease acquired from it.

```cpp
#include <uni20/backend/cuda/buffer.hpp>

using namespace uni20::cuda;

DeviceContext context({
    .device = Device::get(0),
    .stream_count = 4,
});

CudaBuffer<float> values(context, 1024);
```

The allocation uses CUDA's stream-ordered memory pool when the device supports
it, with `cudaMalloc` as the fallback. `CudaBuffer<T>` is move-only.

## Tensor Storage

`CudaAsyncTensor<T, Rank>` is the owning Tensor form for the non-blocking CUDA
submission channel:

```cpp
#include <uni20/tensor/tensor.hpp>

uni20::CudaAsyncTensor<float, 2> matrix(context, 32, 48);
```

The Tensor owns a `CudaBuffer<float>` and preserves ordinary extents and layout
metadata. Its mdspan data handle is a non-owning
`cuda::CudaBufferView<float>` containing buffer identity and an element offset.
Mdspan indexing computes another view offset; it does not dereference device
memory, perform a transfer, acquire a stream, or wait.

The non-const CUDA accessor opts into backend-mediated writes so the owning
Tensor can satisfy mutable Tensor output concepts. Its opaque reference remains
non-assignable on the host. A CUDA operation must lower the view through
`read_synchronized_with(stream)` or `write_synchronized_with(stream)` before a
leaf backend receives a raw pointer.

For GEMM this lowering is reached through the ordinary Tensor API:

```cpp
uni20::linalg::gemm(output, 1.0F, lhs, rhs, 0.0F);
```

The Tensor front end resolves the three mdspans and selects `CublasBackend`.
The backend validates layouts and devices before acquiring a handle and stream,
then opens one write access and two read accesses and calls cuBLAS. The direct
API may block while waiting for a resource-pool slot; it does not synchronize
the submitted GEMM with the host.

## Submitting Work

Raw device pointers are available only through scoped access objects:

```cpp
auto stream = context.streams().acquire();

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
auto stream = context.streams().acquire();
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

The implemented API provides low-level allocation, stream, completion, access,
and CUDA Tensor storage-descriptor semantics. It does not yet provide:

- CUDA Tensor kernels;
- top-down CUDA Tensor-to-provider lowering;
- automatic storage-driven CUDA scheduler selection.

For the exact completion-ledger and failure contract, see [CUDA Buffer
Completion Lowering](epoch_design_draft.md). For stream ownership and error
handling, see the [CUDA Runtime Model](runtime.md).
