# CUDA Memory Allocation: Pools vs Custom Sub-Allocation

**Status:** active design note for future CUDA storage and scheduling.

The current bring-up `cuda::CudaBuffer` uses `cudaMallocAsync`/`cudaFreeAsync`
when stream-ordered memory pools are available, and falls back to
`cudaMalloc`/`cudaFree` otherwise. It waits for retained writer and reader
completions before destruction. That implementation establishes ownership and
synchronization semantics, but the broader allocator policy described below
still needs pool configuration, retention control, and tensor-storage
integration.

This is a draft design note. It records the intended direction for GPU memory
allocation in uni20 and why. It is design direction beyond the current
`CudaBuffer` bring-up behavior.

Related notes:

- `docs/backends/cuda/runtime.md` — stream ownership and the idle-stream pool.
- `docs/architecture/execution.md` — movement/allocation overhead as the real lever.
- `docs/symmetry/block_coalescing.md` — the single-buffer packing that doubles as allocation.
- `docs/tensor_network/contraction_integration_findings.md` — the execution profile that motivated this.

## Motivating observation

Watching `nvidia-smi` during a large-`m` DMRG run, each two-site solve shows the
same cycle: GPU memory **rises slowly at ~0% SM utilization** (host→device staging
of the A/C environments and center vector, plus allocation), then a **short 100%
compute burst** (the GEMMs / SVD), then a **slow free** as memory is released. The
compute is a small fraction of wall time; the run is **data-movement / allocation
bound, not compute bound**. The allocation half of that is what this note addresses
(staging/residency is the device-first/resident-data story in
`../../architecture/execution.md`).

## Why custom arenas existed historically

Classic `cudaMalloc` / `cudaFree` are **device-synchronizing**: `cudaMalloc`
serializes against prior GPU work and `cudaFree` synchronizes the whole device. In
a hot loop they stall everything, so the classic remedy was to **allocate all GPU
memory once at startup and sub-allocate it yourself** (an arena), eliminating
per-operation allocation entirely.

## What the stream-ordered allocator changed

`cudaMallocAsync` / `cudaFreeAsync` (CUDA 11.2+) are **pool-backed and
stream-ordered**, not device-synchronizing:

- alloc/free order only within a stream — no device-wide sync;
- `cudaFreeAsync` returns memory **to a pool**, not the OS, so the next
  `cudaMallocAsync` reuses it cheaply.

So the driver pool **is** the "own it and sub-allocate" pattern, implemented and
maintained by the driver (and it gets fragmentation, multi-stream, multi-device,
and IPC right, which a naive custom arena does not). The historical reason to
hand-roll an arena — avoiding the sync — is essentially gone.

### The release-threshold gotcha (the slow free)

The pool's `cudaMemPoolAttrReleaseThreshold` defaults to **0**, so the pool hands
freed memory **back to the OS at each sync point** — that is the "slow free, memory
drops in `nvidia-smi`" phase above. Set the threshold high (e.g. `UINT64_MAX`) and
optionally **prime** the pool once (the TC bridge already has `preallocatePool`),
and you get the retained-arena behavior — fast reuse, no OS round-trips, no sync —
**without a custom allocator**. The bridge also keeps a `CachedPoolAllocation`
layer sized by `UNI20_TENSORCONTRACTION_POOL_CACHE_BYTES` (default 256 MB) on top
of the pool.

## Where custom sub-allocation still wins

The sync was one cost; the other is **per-call overhead**. Even `cudaMallocAsync`
is a driver call (~ hundreds of ns to ~ µs) — fine for big/medium buffers, but it is
the *same per-op overhead pathology as the small-block tail*: thousands of tiny
allocations per matvec × that cost is real time. There, sub-allocating from a
buffer you already own is a **pointer bump (~ns)**, which the pool cannot match.

Crucially, that sub-allocation is **the same buffer wanted for coalescing**
(`../../symmetry/block_coalescing.md`): packing the tail's many tiny blocks into one strided
buffer serves *three* things at once — one batched GEMM, one MPI message, and **one
allocation** instead of thousands. "Sub-allocate the tail" is not a separate
allocator; it is the coalescing buffer.

## Direction for uni20: two layers, not a general arena

1. **Configured driver pool** as the backing allocator: high release threshold
   (retain) + prime once. Handles big/medium allocations with no sync and good
   reuse, and removes the slow-free phase.
2. **Pack the tail**: sub-allocate the many tiny blocks from a single pooled buffer
   (the coalescing buffer), so the allocator is called once per group, not per
   block.

Do **not** hand-roll a general-purpose custom arena: it is mostly redundant now and
gets fragmentation / multi-stream / multi-device wrong.

### The retain-vs-headroom tradeoff

Retaining the pool (layer 1) consumes headroom. For **maximum-`m`** runs (pushing
until the device OOMs) the aggressive-free default actually helps reach higher `m`,
because the peak working set needs the headroom. For **converged runs at a fixed
`m`**, raise the release threshold and the per-solve cycle is much faster. So the
retention policy should be a runtime choice, not a build-time default.

## Decisions made

- Use the stream-ordered allocator (`cudaMallocAsync`/`cudaFreeAsync`) with a
  configured `cudaMemPool` as the backing allocator; do not rely on synchronous
  `cudaMalloc`/`cudaFree` on hot paths.
- Treat the driver pool as the arena: set a high release threshold and prime it to
  retain memory, instead of writing a general custom sub-allocator.
- Sub-allocate the small-block tail from a single pooled buffer — the same buffer
  used for GEMM batching and MPI message aggregation.
- Make the pool retention policy a runtime choice (retain for fixed-`m` speed;
  aggressive-free for maximum-`m` headroom).

## Open questions

- Default release threshold and prime size per device class, and how they interact
  with the stream-pool backpressure in `runtime.md`.
- Whether pool fragmentation at large `m` ever forces the **CUDA VMM API**
  (`cuMemCreate` / `cuMemMap` — reserve a virtual range, map physical pages on
  demand) for a growable, fragmentation-free arena. Treat as a fallback only if the
  pool fragments in practice.
- Allocation lifetime under deferred (buffer-access) sync: the
  token-pins-storage rule (`../../architecture/ordering_and_backend_lowering.md`) must drive
  `cudaFreeAsync` ordering so memory is not recycled before the producing kernel
  completes.
