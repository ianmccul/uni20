# Kernel Dispatch Design

**Status: checkpoint of current thinking, not implemented.** This note captures
a worked design direction for how Uni20 selects and runs a backend kernel for a
tensor operation. It generalizes the three-stage pattern in
[`backend_dispatch.md`](backend_dispatch.md) into a single configurable
mechanism that also covers device and distributed execution. Treat the code
below as schematic — the type and helper names illustrate the shape, they are
not a frozen API.

Related notes:

- [`backend_dispatch.md`](backend_dispatch.md) — the `maybe_can_* / try_* /
  generic` three-stage pattern this note generalizes.
- [`ordering_and_backend_lowering.md`](ordering_and_backend_lowering.md) — the
  async scheduler owns all ordering; backends emit work, they do not order it.
- [`execution_architecture.md`](execution_architecture.md) — mechanism/policy
  split and scheduler shapes.
- [`storage_kind_and_location.md`](storage_kind_and_location.md) — memory kind
  vs location, the axes dispatch keys on.
- [`block_coalescing.md`](block_coalescing.md) — coalesced/batched kernels as a
  backend capability.

## The model in one paragraph

A **backend** is a uniform thing with two ways to decline an operation —
`maybe_can_*` (compile-time facts about the C++ types; prunes the candidate at
zero runtime cost) and `try_*` (runtime facts plus the actual attempt) — and one
way to succeed: return `true` with the work done or correctly emitted. (For
deferred device work, "correctly emitted" means the completion token — the CUDA
event recorded at submission, per `ordering_and_backend_lowering.md` — is
attached to the buffers' epochs before the backend returns. The token is
buffer-lifetime machinery owned by the storage policy; it never travels through
the dispatch interface.)
Dispatch is a walk over an **ordered list of backends**: the first backend that
does not decline wins. The old `generic` fallback is simply the last entry in
the list, the correctness oracle. Backends **nest** — a backend may be
implemented in terms of other backends — which is how distributed and RPC
execution layer on without a class hierarchy. This replaces the original
tagged-dispatch-with-inheritance plan, which was too rigid.

The two configuration rules:

1. The backend list is a **parameter of the kernel function**, defaulting to one
   derived from the tensor storage type — for a dense tensor its `TensorStorage`
   (memory kind), for a `BlockTensor` the container's `BlockTensorStorage` policy
   (`block_tensor.md` §1/§5). A caller can override it to pin a specific backend
   for benchmarking, debugging, or correctness checks.
2. There is **no runtime backend selection in the first pass.** Python bindings
   are compiled against a fixed target list (for example `[Blas, CpuGeneric]`,
   or `[Cublas, Blas, CpuGeneric]`) baked in at build time. A `RuntimeSelectable`
   backend or a `std::variant` of backends can be layered on later if needed; it
   is just another list entry.

## Worked example: `gemm`

The contraction primitive `C <- alpha * A * B + beta * C` exercises every part of
the model. The entry point derives a default list from storage and forwards to
the walk:

```cpp
// gemm.hpp
template <class C, class A, class B,
          class Backends = default_backends_t<C, A, B>>   // (1) default from storage
void gemm(scalar_t<C> alpha, A const& a, B const& b,
          scalar_t<C> beta, C& c, Backends = {})           // (2) overridable parameter
{
  // The whole-list eligibility check lives here, at the entry point. It must
  // not live in the recursive walk: there it would see only the remaining
  // *tail* of the list, and a list like [Blas, Cublas] on host tensors would
  // static_assert when the walk instantiates the [Cublas] tail, even though
  // Blas is eligible.
  static_assert(any_maybe_can_gemm_v<Backends, C, A, B>,
                "no backend in this list can ever service this gemm");
  dispatch_gemm(Backends{}, alpha, a, b, beta, c);
}

template <class First, class... Rest, class A, class B, class C>
void dispatch_gemm(backend_list<First, Rest...>,
                   scalar_t<C> alpha, A const& a, B const& b, scalar_t<C> beta, C& c)
{
  if constexpr (First::template maybe_can_gemm<C, A, B>) {   // compile-time prune
    if (First::try_gemm(alpha, a, b, beta, c)) return;       // runtime attempt
  }
  if constexpr (sizeof...(Rest) > 0)
    dispatch_gemm(backend_list<Rest...>{}, alpha, a, b, beta, c);
  else
    throw no_available_backend{"all eligible gemm backends declined at runtime"};
}
```

Two distinct failure modes fall out, which is exactly the behaviour an override
should have:

- Forcing `backend_list<Cublas>` on **host** tensors makes `maybe_can_gemm`
  false for every element, so the entry-point `static_assert` fires —
  a **compile error**.
- Forcing `backend_list<Blas>` when libBLAS is not loaded at runtime makes
  `try_gemm` return false, the list is exhausted, and the kernel **throws**. The
  default lists always end in the infallible oracle, so the default path never
  throws.

### Three local backends

```cpp
struct CpuGenericBackend {                                  // the correctness oracle
  template <class C, class A, class B>
  static constexpr bool maybe_can_gemm =
      is_host_v<C> && is_host_v<A> && is_host_v<B>;          // any scalar, any layout

  // a, b, c are resolved TensorViews — run the kernel immediately, no scheduler.
  template <class C, class A, class B>
  static bool try_gemm(scalar_t<C> alpha, A const& a, B const& b, scalar_t<C> beta, C& c) {
    using T = scalar_t<C>;
    for (size_t i = 0; i < c.extent(0); ++i)
      for (size_t j = 0; j < c.extent(1); ++j) {
        // BLAS rule: beta == 0 means C is *not read* (a fresh output block may
        // hold uninitialized data; 0 * NaN must not poison the result).
        T acc = (beta == T{}) ? T{} : beta * c[i, j];
        for (size_t k = 0; k < a.extent(1); ++k) acc += alpha * a[i, k] * b[k, j];
        c[i, j] = acc;
      }
    return true;                                             // infallible
  }
};

struct BlasBackend {
  template <class C, class A, class B>
  static constexpr bool maybe_can_gemm =
      is_host_v<C> && is_host_v<A> && is_host_v<B> &&
      same_scalar_v<C, A, B> && is_blas_scalar_v<scalar_t<C>>;   // f / d / cf / cd only

  // a, b, c are resolved TensorViews — call BLAS immediately, no scheduler.
  template <class C, class A, class B>
  static bool try_gemm(scalar_t<C> alpha, A const& a, B const& b, scalar_t<C> beta, C& c) {
    if (!blas::available()) return false;                   // runtime capability
    auto plan = blas::as_gemm(a.layout(), b.layout(), c.layout());  // metadata only
    if (!plan) return false;                                // strides not BLAS-expressible
    blas::gemm(*plan, alpha, a, b, beta, c);                // synchronous call
    return true;
  }
};

struct CublasBackend {
  template <class C, class A, class B>
  static constexpr bool maybe_can_gemm =
      is_device_v<C> && is_device_v<A> && is_device_v<B> &&
      same_scalar_v<C, A, B> && is_blas_scalar_v<scalar_t<C>>;

  // Unlike the CPU backends, this *does* go through a scheduler: the CudaScheduler
  // builds sequencing into its streams/events, so the submit is correctly ordered.
  // (A blocking variant on the default stream is also possible, but the scheduler
  // path is typical.)
  template <class C, class A, class B>
  static bool try_gemm(scalar_t<C> alpha, A const& a, B const& b, scalar_t<C> beta, C& c) {
    int dev = c.device();
    for (int other : {a.device(), b.device()})              // operands on another GPU?
      if (other != dev && !cuda::peer_access(dev, other))
        return false;                                       // let a staging backend handle it
    gpu_scheduler(dev).submit(cublas_gemm_task<C,A,B>, alpha, beta,  // CudaScheduler: stream/event
                              a.read(), b.read(), c.write());        //   sync is built in; per-GPU
    return true;                                                     //   thread holds the cuBLAS handle
  }
};
```

Three invariants this surfaces:

- **Dispatch reads only metadata.** `layout()`, `device()`, dtype, and extents
  are all available synchronously, so `try_gemm` can decide before any data is
  read. The kernel then runs on the resolved `TensorView`s.
- **CPU backends are synchronous; they do not touch a scheduler.** `BlasBackend`
  and `CpuGenericBackend` call the kernel immediately on resolved views. Async
  execution of CPU work comes from wrapping the tensors in `Async<T>`, whose epoch
  queues provide the sequencing — submitting a bare CPU task to a scheduler with no
  buffer await would carry no ordering and never be correct. There is no point in a
  separate "CPU async backend": `Async<T>` already is it.
- **CUDA is the exception.** `CublasBackend` does go through a scheduler, because
  the `CudaScheduler` builds sequencing into its streams and events (see
  `ordering_and_backend_lowering.md`). Its per-GPU threads hold the thread-local
  cuBLAS/cuSOLVER context. A blocking variant on the default stream is possible,
  but the scheduler path is typical. Note the division of labour: *ordering* is
  already guaranteed by dispatch order from the async layer; events exist for
  *synchronization* (completion detection, buffer lifetime) and for cross-stream
  concurrency. A zero-event, default-stream debug mode is therefore correct.

### MPI falls out as nesting

A distributed backend handles only the distribution layer and recurses into the
local kernel for the per-MPI-rank compute:

```cpp
struct MpiBackend {                                         // distribution layer only
  // is_distributed_v / is_replicated_v are traits of the container's
  // BlockTensorStorage policy (block_tensor.md §1/§5) — distribution is a
  // container capability, never a dense TensorStorage. Replicated × distributed
  // is the *common* case (a replicated MPO contracted with a distributed state),
  // so the predicate must not demand distribution of every operand.
  template <class C, class A, class B>
  static constexpr bool maybe_can_gemm =
      (is_distributed_v<C> || is_distributed_v<A> || is_distributed_v<B>) &&
      all_distributed_or_replicated_v<C, A, B>;

  template <class C, class A, class B>
  static bool try_gemm(scalar_t<C> alpha, A const& a, B const& b, scalar_t<C> beta, C& c) {
    auto plan = mpi::plan_gemm(a.distribution(), b.distribution(), c.distribution());
    if (!plan) return false;        // decline *before* any side effect (see below)
    for (auto const& edge : plan->exchanges)
      mpi::post(edge);              // Isend/Irecv first; unique tag per edge (ordering_and_backend_lowering.md)
    for (auto const& t : plan->local_tiles)                  // overlap compute with the exchanges
      gemm(alpha, t.a_local, t.b_local, t.beta, t.c_local);  // recursion -> local default list
    for (auto const& t : plan->remote_tiles) {               // contributions needing received data
      mpi::wait(t.recv);            // schematic: a real impl schedules these as tasks gated on the receives
      gemm(alpha, t.a_recv, t.b_recv, scalar_t<C>{1}, t.c_local);  // accumulate into the local tile
    }
    return true;
  }
};
```

One contract the composite backend makes explicit: **`try_*` declines before any
side effect, or not at all.** Once a backend has posted a message or executed a
tile, returning `false` would hand the next backend in the list a partially
updated `c`. So all feasibility checks (here `mpi::plan_gemm`) come first, and
past the first side effect the backend is committed — a later failure is an
error to report, not a decline. The simple backends satisfy this trivially
(their checks are reads of metadata and library availability); any backend that
composes other work must be written to it deliberately.

Because `MpiBackend::try_gemm` recurses into `gemm` on the **local** tensors,
which re-derive *their own* default list, **MPI/CPU versus MPI/CUDA is just
whether the node-local storage — the `X` in `Mpi<X>` — is host or device** — the
same `MpiBackend` covers both and nesting supplies the local kernel. This is also how an optional
MPI_RPC backend would layer on: a front-of-list backend implemented on top of
the collective MPI backend, one more level of the same nesting. None of it
requires core changes, so the first pass ships with the local backends only.

### Default list per storage mode, and overrides

| storage mode | `default_backends_t` | effective stack |
|---|---|---|
| `Tensor<T, Host>` | `[Blas, CpuGeneric]` | BLAS then oracle |
| `Tensor<T, Device>` | `[Cublas, DeviceGeneric]` | cuBLAS then device oracle |
| `BlockTensor<…, Mpi<Host>>` | `[Mpi]` | Mpi composed with `[Blas, CpuGeneric]` per MPI rank |
| `BlockTensor<…, Mpi<Cuda>>` | `[Mpi]` | Mpi composed with `[Cublas, DeviceGeneric]` per MPI rank |

`Mpi<…>` never appears on a dense `Tensor`: distribution is a capability of the
block-sparse *container*'s `BlockTensorStorage` policy, not a dense memory kind
(`block_tensor.md` §1/§5 — a dense tensor striped over MPI ranks is conceivable but
off-roadmap). The dense rows dispatch on the leaf `TensorStorage` only.

```cpp
gemm(alpha, a, b, beta, c);                                  // default for the storage
gemm(alpha, a, b, beta, c, backend_list<CpuGeneric>{});      // force the oracle (correctness debugging)
gemm(alpha, a, b, beta, c, backend_list<Blas, CpuGeneric>{});// pin BLAS-or-oracle (benchmark a path)
gemm(alpha, a, b, beta, c, backend_list<Cublas>{});          // GPU only, no fallback - fail loudly
```

## What this pins down

- The dispatch mechanism is `backend_list` plus per-backend `maybe_can_*`
  (compile-time) and `try_*` (runtime). No inheritance, no runtime tags.
- The **dispatch-to-async seam is a separate layer above the backends, not the
  backends themselves.** When operands are `Async<T>`, that layer selects the
  backend (on synchronous metadata, before anything is scheduled) and schedules a
  coroutine which `co_await`s the operand buffers (the epoch queues, not the
  scheduler, are what order the work), resolves them to `TensorView`s, and then
  runs the already-selected synchronous backend — exactly the shape of the
  existing `Async<T>` operations in `async_ops.hpp`. The CPU *backends* never name a scheduler; the
  CUDA path is the exception, submitting to the `CudaScheduler` whose streams and
  events carry the ordering. There the coroutine's *return type* selects the
  scheduler (`AsyncTask` → CPU, `CudaTask` → GPU). See *Scheduler and Async
  integration* below.
- Distributed and RPC execution are front-of-list backends that recurse, so they
  need zero core changes. The first pass can ship `[Blas, CpuGeneric]` with the
  seam being only "the list is a parameter."

## Scheduler and Async integration

This section records how kernel dispatch sits on top of the async runtime
(`src/uni20/async/`) — the part the worked `gemm` above leaves implicit.

### Three layers

Dispatch separates into three layers, and the bottom layer knows nothing about
the async runtime:

1. **Kernels** — plain functions over `TensorView`s (an mdspan: pointer +
   extents + strides). `cblas_dgemm`, a cuBLAS call on a given stream, a generic
   triple loop. CPU kernels are entirely async-unaware; they compute and return.
   A CUDA kernel takes a stream or handle but does not know about the async
   runtime.
2. **Async dispatch wrappers** — run backend selection at submission time (it
   reads only metadata, which is synchronous), then schedule a coroutine that
   `co_await`s the operand buffers, resolves them to `TensorView`s, and calls the
   already-selected kernel. By the time a coroutine is scheduled, dispatch has
   succeeded: the coroutine is a thin wrapper around the raw backend kernel call.
   This is the shape of the existing `Async<T>` operations in `async_ops.hpp`
   (`async_binary_op`, `async_compound_op`). Synchronization lives here, in the
   epoch-queue awaits; the kernel never sees it.
3. **Backend selection** — the `backend_list` walk (`maybe_can_* / try_*`) and,
   for a block tensor, the planner that fans out per block. A CPU backend's
   `try_*` calls a layer-1 kernel directly and synchronously; the CUDA backend
   submits to the `CudaScheduler`. Layer 2 sits *above* this walk in the
   layering, but the walk runs *first* in time: selection happens at submission,
   on synchronous metadata, and the scheduled coroutine then awaits the buffers
   and runs the kernel that was selected.

The consequence: one layer-1 kernel is reused by every storage mode; only the
wrapper changes.

### Metadata synchronous, data async

Dispatch reads only metadata — extents, strides, device id, block structure — and
all of it must be readable *without awaiting*, or the planner serializes behind
the producer of every operand. So the boundary is:

> Descriptor synchronous, bytes async. The kernel dispatch point is a
> `TensorView` synthesized from a synchronous descriptor plus an awaited pointer.

Three things stay distinct:

- **Descriptor** — per-block dims/strides/coord; lives on the container; always
  synchronous.
- **Data handle** — the per-block async-like object; `co_await` it for ready data.
- **`TensorView`** — descriptor + awaited pointer; what every kernel sees.

### The async-block concept

The async dispatch path is written against a concept, not a concrete type, so a
single wrapper serves both a whole-value `Async<T>` and one element of an
`AsyncArray`:

```cpp
template <class H>
concept async_block = requires(H h) {
  { h.read()  };   // co_await -> TensorView<S> const   (ready data)
  { h.write() };   // co_await -> TensorView<S>          (ready, writable)
};
```

`AsyncArray<S>` is deliberately lightweight: one shared backing allocation, a
synchronous descriptor table, and a per-element epoch queue for independent
ordering. Its element handle models `async_block` and shares a common public
interface with `Async<T>`; it does not allocate a full `Async` per block. (For
code that genuinely wants an `Async<TensorView>` from an element, the existing
deferred/aliasing `Async` constructors — `async.hpp:216`, `:239` — alias the
shared backing with the element's queue.)

### BlockTensor: two async axes

"Async or not" is two independent axes:

- **Container/structure** — is the block structure (which blocks exist, their
  dims) known synchronously, or itself an async result? Wrapping a `BlockTensor`
  as `Async<BlockTensor>` makes the structure async (e.g. a truncation output).
- **Per-block data** — each block's data is immediate, async (host), or async
  (device). This is the async facet of the container's `BlockTensorStorage`
  policy (`block_tensor.md` §5): a `TensorView` itself has no async-ness, so
  whether block data sits behind async handles is the container policy's
  decision, not a property of the dense leaf.

| container structure | per-block data | when |
|---|---|---|
| synchronous | immediate | basic CPU path — planner runs, kernels run inline, no coroutines |
| synchronous | async (host/device) | common contraction — structure known up front, blocks scheduled for fine-grained parallelism |
| async | async | truncation / SVD output — shape and data are both results |

The bare `BlockTensor` always carries its structure synchronously; the container
`Storage` policy chooses the per-block mode:

```cpp
template <class Storage>                    // BlockTensorStorage policy (block_tensor.md §5)
class BlockTensor {
    BlockSpace             legs_[Rank];     // structure — synchronous
    std::vector<Coord>     coords_;         // which blocks exist — synchronous
    block_store_t<Storage> blocks_;         // immediate views | AsyncArray | AsyncArray<device>
  public:
    size_t          num_blocks()         const;          // sync
    BlockDescriptor descriptor(size_t i) const;          // sync: dims/strides/coord
    auto            block(size_t i);                      // -> async_block (or TensorView)
};
```

The planner reads only descriptors, so it is synchronous; `block_gemm` is written
once against `async_block`, with `if constexpr` keeping the immediate path
coroutine-free:

```cpp
template <class C, class A, class B>
void block_gemm(C& c, A const& a, B const& b) {
  for (auto r : output_blocks(c))                          // descriptors only — synchronous
    for (auto [ai, bi] : contributing(c, a, b, r)) {
      if constexpr (immediate_blocks_v<C>)
        gemm_kernel(c.block(r), a.block(ai), b.block(bi)); // no coroutine
      else
        schedule_block_gemm(c.block(r), a.block(ai), b.block(bi));
    }
}
```

(The sketch suppresses the per-block GEMM parameters: for a symmetry-typed
`BlockTensor`, each contribution also carries the operands' `op` state and a
per-block `alpha = s_A · s_B / s_C` derived from the view op-state and coupling
factors — `block_tensor.md` §7. So `contributing()` yields
`(ai, bi, op_a, op_b, alpha)` records, not bare block indices.)

**Who selects the per-block backend.** Ownership sits with the block's storage
type, which the container's `BlockTensorStorage` policy determines — but the
*mechanism* is the kernel's choice. A generic linear-algebra op typically
re-enters the ordinary `backend_list` walk per block (or per coalesced group),
keeping its runtime fallbacks. A specialized planner — an optimized R/A/B/C
Hamiltonian-apply (`rabc_contraction_scheduling.md`), say — may instead bind a
whole list of CUDA sub-kernels at plan time, potentially as a CUDA graph, and
never consult a walk per block. Both are legitimate shapes of layer 3. The one
constraint is the order of decisions: **dispatch happens before scheduling**. By
the time a coroutine is scheduled, dispatch has already succeeded — the coroutine
is a thin wrapper around the chosen backend's raw kernel call plus its buffer
awaits — so a scheduled kernel-coroutine never re-enters the walk.

The host-async and device-async block kernels differ only in the coroutine return
type, which is what routes them to the CPU or GPU scheduler:

```cpp
// host-async block kernel
schedule([](auto a_, auto b_, auto c_) static -> AsyncTask {       // -> CPU scheduler
  auto A_ = co_await a_;  auto B_ = co_await b_;  auto C_ = co_await c_;
  gemm_kernel(C_, A_, B_);                                          // layer-1 kernel
}(a.block(ai).read(), b.block(bi).read(), c.block(r).write()));

// device-async block kernel — identical but for the return type
schedule([](auto a_, auto b_, auto c_) static -> CudaTask {        // -> GPU scheduler
  auto A_ = co_await a_;  auto B_ = co_await b_;  auto C_ = co_await c_;
  cublas_gemm(C_, A_, B_);                                          // thread-local handle + stream
}(a.block(ai).read(), b.block(bi).read(), c.block(r).write()));
```

The accumulation `r += sum a*b` over multiple contributing `(a, b)` needs no
explicit reduction lock: every contribution writes block `r`, so block `r`'s
epoch queue serializes them. The queue only *orders*, though — it does not elect
an initializer. Exactly one contribution must run with the caller's `beta` (or
`beta = 0` for a fresh output block, which per the BLAS rule must not read the
uninitialized data), and every other contribution accumulates with `beta = 1`.
The planner designates that first contribution when it enumerates the work.

### Structure-async: the two-level await

An operation on `Async<BlockTensor>` `co_await`s the container first to obtain
structure and layout, then runs `block_gemm`, whose per-block awaits handle the
data:

```cpp
// gemm on Async<BlockTensor>: await structure, then dispatch per block
schedule([](auto a_, auto b_, auto c_) static -> AsyncTask {
  auto const& A = co_await a_;     // structure + layout known
  auto const& B = co_await b_;
  auto&        C = co_await c_;
  block_gemm(C, A, B);             // inner per-block dispatch
}(A.read(), B.read(), C.write()));
```

Awaiting the container guarantees the *structure*, not the block data. Block
validity is governed by the container `Storage` policy: immediate blocks are
valid after the outer await; async blocks are not, and the inner `block(i).read()`
is the second await level. The two await levels map onto the two async axes. This
is what "async until output" means in practice — nothing forces a thread to block
until a real materialization boundary (a scalar read, a print, a save, a branch on
a value).

### Scheduler routing by promise type

Where work *is* scheduled — the layer-2 async wrappers — the coroutine's return
type selects the scheduler: `-> AsyncTask` routes to the CPU scheduler, `-> CudaTask`
(its own `CudaTaskPromise`, `cuda_task.hpp`) to a GPU scheduler, via an overloaded
`schedule()`. `myScheduler->schedule(task)` remains available directly. For
multi-GPU, the device ordinal rides in `CudaTaskPromise` (so `schedule(CudaTask)`
fans out to `gpu_scheduler(dev)`), or the CUDA backend calls
`gpu_scheduler(c.device())->schedule(...)` itself. The synchronous CPU backends,
having no coroutine, sit outside this routing entirely.

CUDA and MPI kernels admit several wrapper strategies — run basically
synchronously on the default stream, or schedule a `CudaTask` on the GPU
scheduler — which appear as distinct backends sharing one layer-1 kernel. Both
are legitimate and can coexist in a backend list; the `CudaTask`-on-scheduler
form is expected to dominate, with the synchronous default-stream form remaining
useful as the simplest correct path and for debugging. The first pass does not coalesce the small-block tail on
CPU: the tail is a GPU launch-bound problem (see
`tensorcontraction_integration_findings.md`), not a CPU one, so CPU ships one task
per block and coalescing is added to the GPU path later if a profile demands it.

## Open questions

1. **The backend interface.** Exact signatures for `maybe_can_*` / `try_*` /
   submit, and how a backend advertises which scheduler(s) it targets.
2. **List configuration.** The default travels with the tensor (storage type),
   but is the override a compile-time `backend_list` only, or also a runtime
   value? The `RuntimeSelectable` case erases types; the C++ path stays a
   compile-time tuple.
3. **Scheduler shapes.** `gemm` submits and returns; a `cuSOLVER` factorize holds
   its thread for the duration of a blocking call. The per-GPU scheduler must
   support both the async-submission and sync-blocking shapes
   (see `execution_architecture.md`), with deterministic acquisition under
   `DebugScheduler` so differential tests hold.
4. **The oracle for device and distributed storage.** Every list must end in a
   runnable correctness oracle. For device tensors that is a naive device kernel;
   for distributed tensors a gather-to-one-MPI-rank-then-generic path is a candidate,
   useful mainly for tests.
5. **Coalesced and batched kernels.** A coalesced single-axis GEMM group should
   be expressible as one backend capability (`maybe_can_batched_*`) emitting one
   task, per `block_coalescing.md`.
6. **Testing the oracle.** Guaranteeing the generic path stays exercised even
   when an optimized backend is always eligible — the forced-fallback override
   above is the mechanism, per `backend_dispatch.md`.
7. **`AsyncArray` queue representation.** Striping per-block epoch queues over the
   shared backing cheaply enough for thousands of small blocks, while keeping
   independent per-block ordering and a public interface common with `Async<T>`.
   This is where the small-block tail cost lands on CPU. This is the same
   problem as the hazard-granularity question in `block_tensor.md` §10, seen
   from the queue side: a *coalesced* operand (`block_coalescing.md`) spans many
   elements of one backing buffer, so either it awaits every element queue it
   covers (and writers do, symmetrically) or there is a buffer-level queue the
   element queues nest under. Per-element queues alone do not cover the dual
   block-view / wide-view access that coalescing requires.
8. **CUDA wrapper strategies.** Whether the synchronous default-stream backend
   remains a maintained sibling of the `CudaTask`-on-scheduler form (the two can
   coexist in a backend list) or is retired once the scheduler path is proven.
9. **MPI user-facing default.** This note presents the collective SPMD
   `MpiBackend` as the base, with RPC as an optional front-of-list layer;
   `uni20_mpi_persistent_dispatch_design.md` proposes the root-controller /
   worker-runtime model as the *default* user-facing mode, with expert SPMD
   remaining possible. The two are mechanism vs. default policy and can
   coexist, but which mode ordinary user code gets by default is an open
   decision that both notes should record the same way.
