# Kernel Dispatch Design

**Status: checkpoint of current thinking, not implemented.** This note captures
a worked design direction for how Uni20 selects and runs a backend kernel for a
tensor operation. It generalizes the three-stage pattern in
[`backend_dispatch.md`](backend_dispatch.md) into a single configurable
mechanism that also covers device and distributed execution. Treat the code
below as schematic — the type and helper names illustrate the shape, they are
not a frozen API.

The static capability CPO name used here is `kernel_accepts_types(...)`. The
runtime attempt CPO is `try_kernel(...)`. Both put the backend value first, then
the operation tag, then the operation arguments.

Related notes:

- [`backend_dispatch.md`](backend_dispatch.md) — the compile-time capability /
  runtime attempt / generic fallback pattern this note generalizes.
- [`mdspan_linalg_dispatch_plan.md`](mdspan_linalg_dispatch_plan.md) — the
  planned first concrete implementation slice of this model for dense linalg
  leaf kernels over strided mdspan views.
- [`ordering_and_backend_lowering.md`](ordering_and_backend_lowering.md) — the
  async scheduler owns all ordering; backends emit work, they do not order it.
- [`execution_architecture.md`](execution_architecture.md) — mechanism/policy
  split and scheduler shapes.
- [`storage_kind_and_location.md`](storage_kind_and_location.md) — memory kind
  vs location, the axes dispatch keys on.
- [`block_coalescing.md`](block_coalescing.md) — coalesced/batched kernels as a
  backend capability.

## The model in one paragraph

A **backend** is a uniform thing with two ways to decline an operation. The
dispatch mechanism is generic over an **operation tag** (`gemm_op`, `scale_op`,
`assign_op`, ...), and discovers backend support with customization points:

- `kernel_accepts_types(Backend const&, Op const&, Args&...)` — optional
  `consteval` tri-state function for compile-time facts about the C++ types;
  prunes the candidate at zero runtime cost. It is written as an ordinary
  function with ordinary reference parameters, so this CPO does not need a tuple
  or explicit type-pack token.
- `try_kernel(Backend, Op, args...)` — required for an implemented operation;
  receives the backend value, checks runtime facts, and either performs/emits
  the work or returns `false`.

A backend that provides `kernel_accepts_types(...)` should put its complete
type-level eligibility test there. Its `try_kernel(...)` overload can then be
deliberately unconstrained: dispatch never instantiates it for rejected types,
and a local `static_assert` can diagnose accidental direct misuse. This keeps
the runtime CPO focused on instance facts rather than duplicating a long
constraint expression. A backend without a static CPO may instead use a
constrained `try_kernel(...)`; detection then treats that overload as
`KernelTypeAcceptance::maybe`.

Success means return `true` with the work done or correctly emitted. (For
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

The ordered list has a static type order but can hold backend values, not only
backend types. Backend entries can be stateless tags in the first pass, or
values carrying context such as CUDA device and stream later:

```cpp
struct BlasBackend {};

struct CublasBackend {
  int device;
  cudaStream_t stream;
  math_mode_t math_mode;
};
```

The CPO signature remains backend-first:

```cpp
try_kernel(selector.first(), op, args...);
```

If several entries need shared structural state, the selector can store that
state once and hand each CPO an appropriate backend value or lightweight backend
view. Shared state is a selector implementation detail, not a required separate
`State&` parameter in every leaf-kernel CPO.

`normalize_backend_selector(selector)` is where a compact user override becomes
the concrete selector used by the dispatch walk:

```cpp
auto selector =
  make_backend_selector<backend_list<CublasBackend, CudaGenericBackend>>(
      CublasConfig{.device = {0}, .stream = {stream}, .math_mode = {tf32_allowed}});
```

The config object is construction sugar. Internally the selector may store
deduplicated shared state, but that does not change the backend-first CPO shape.

The two configuration rules:

1. The backend selector is a **parameter of the kernel function**, defaulting to
   a list derived from the tensor storage type — for a dense tensor its
   `TensorStorage` (memory kind), for a `BlockTensor` the container's
   `BlockTensorStorage` policy (`block_tensor.md` §1/§5). A caller can override
   it with either an ordered backend-list value or a single backend value. A
   single backend is normalized to a one-element list, which is convenient for
   forcing a backend during benchmarking, debugging, or correctness checks.
2. There is **no runtime type-erased backend selection in the first pass.**
   Python bindings are compiled against a fixed target list (for example
   `[Blas, CpuGeneric]`, or `[Cublas, Blas, CpuGeneric]`) baked in at build time.
   Backend entries may still carry runtime state such as CUDA device or
   workspace pool. A `RuntimeSelectable` backend or a `std::variant` of backends
   can be layered on later if needed; it is just another list entry.

New Uni20 kernel and linalg APIs put API tags and explicit backend selectors
first, then mutable outputs, then inputs and scalar operands. For example,
write `gemm(selector, c, alpha, a, b, beta)` for an explicit override and
`gemm(c, alpha, a, b, beta)` for the storage-default overload, rather than the
BLAS/LAPACK ABI order `gemm(alpha, a, b, beta, c)`.

## Dispatch operands versus resolved spans

The main dispatch entry points operate on Uni20 tensor/storage operands, not on
bare mdspans. Backend lists come from tensor storage policy: host/device memory,
block storage, MPI replication/distribution, async data handles, and scheduler
requirements. A plain `stdex::mdspan` or generic `SpanLike` view has enough
information to run a leaf kernel, but it does not carry the Uni20 storage policy
needed to select the default backend stack.

The layers are therefore:

1. **Tensor operation wrapper** — accepts tensor-like operands, output policy,
   and an optional backend list override. It computes the default `Backends` from
   storage policy and prepares resizable or fixed outputs.
2. **Backend dispatch** — walks the backend list for the operation tag. It reads
   synchronous metadata and, for async/block tensors, schedules the await/lowering
   wrapper selected by the backend.
3. **Leaf kernel** — receives resolved mdspan-like views (`TensorView`,
   `stdex::mdspan`, transform views, etc.) after backend compatibility has
   already been decided.

Passing mdspans directly is a lower-level escape hatch. The caller must provide a
backend list explicitly, or call a concrete backend helper such as
`BlasBackend::gemm(...)` if such a convenience wrapper exists. In that path the
caller has already taken responsibility for storage compatibility.

## Worked example: gemm

The contraction primitive `C <- alpha * A * B + beta * C` exercises every part of
the model. The entry point derives a default list from storage and forwards to
the walk:

The first end-to-end mdspan implementation slice exists for explicit backend
selectors. The low-level mdspan leaf is
`uni20::linalg::blas::try_gemm(c, alpha, a, b, beta)`;
`try_kernel(BlasBackend, gemm_op, ...)` wraps that leaf, and
`CpuGenericBackend` provides an independently tested fallback when BLAS
declines. This proves the operation-tag walk without first solving the broader
LAPACK surface, workspace policy, or vector descriptor layer.

```cpp
struct gemm_op
{
    static constexpr char const* name = "gemm";
};

enum class KernelTypeAcceptance {
  no,     // invalid for these types; do not form try_kernel
  maybe, // runtime values decide
  yes    // runtime attempt is expected to succeed for every instance
};

template <class Backend, class Op, class... Args>
consteval bool backend_has_try_kernel()
{
  return requires(Backend backend, Op op, Args&&... args) {
    { try_kernel(backend, op, std::forward<Args>(args)...) } -> std::same_as<bool>;
  };
}

namespace detail {
template <class T>
extern std::remove_reference_t<T> kernel_type_probe_object;

template <class T>
constexpr std::remove_reference_t<T>& kernel_type_probe_arg() noexcept
{
  return kernel_type_probe_object<T>;
}
} // namespace detail

template <class Backend, class Op, class... Args>
consteval KernelTypeAcceptance kernel_type_acceptance()
{
  if constexpr (requires {
             kernel_accepts_types(detail::kernel_type_probe_arg<Backend const&>(),
                                  detail::kernel_type_probe_arg<Op const&>(),
                                  detail::kernel_type_probe_arg<Args>()...);
           }) {
    constexpr auto acceptance =
        kernel_accepts_types(detail::kernel_type_probe_arg<Backend const&>(),
                             detail::kernel_type_probe_arg<Op const&>(),
                             detail::kernel_type_probe_arg<Args>()...);
    if constexpr (acceptance == KernelTypeAcceptance::no) {
      return KernelTypeAcceptance::no;
    } else {
      static_assert(backend_has_try_kernel<Backend, Op, Args...>(),
                    "kernel_accepts_types accepted these types, but try_kernel is not available");
      return acceptance;
    }
  }

  if constexpr (backend_has_try_kernel<Backend, Op, Args...>())
    return KernelTypeAcceptance::maybe; // constrained try_kernel is the type test
  else
    return KernelTypeAcceptance::no;
}

template <class Backends, class Op, class... Args>
inline constexpr bool any_kernel_type_eligible_v = false;

template <class... Backends, class Op, class... Args>
inline constexpr bool any_kernel_type_eligible_v<backend_list<Backends...>, Op, Args...> =
    ((kernel_type_acceptance<Backends, Op, Args...>() != KernelTypeAcceptance::no) || ...);

template <class Op, class First, class... Rest, class... Args>
bool dispatch_kernel(backend_list<First, Rest...> backends, Op op, Args&&... args)
{
  constexpr auto acceptance = kernel_type_acceptance<First, Op, Args&&...>();
  if constexpr (acceptance == KernelTypeAcceptance::yes) {
    bool const success = try_kernel(backends.first(), op, std::forward<Args>(args)...);
    UNI20_ASSERT(success);
    return true;
  } else if constexpr (acceptance == KernelTypeAcceptance::maybe) {
    if (try_kernel(backends.first(), op, std::forward<Args>(args)...))
      return true;
  }

  if constexpr (sizeof...(Rest) > 0)
    return dispatch_kernel(backends.rest(), op, std::forward<Args>(args)...);
  else
    return false;
}

// gemm.hpp
template <class C, class A, class B>
void gemm(C& c, scalar_t<C> alpha, A const& a, B const& b, scalar_t<C> beta)
{
  auto selector = default_backend_selector(c, a, b);
  gemm(selector, c, alpha, a, b, beta);
}

template <class BackendSelector, class C, class A, class B>
  requires BackendSelectorLike<BackendSelector>
void gemm(BackendSelector selector, C& c, scalar_t<C> alpha, A const& a,
          B const& b, scalar_t<C> beta)
{
  auto backends = normalize_backend_selector(selector);
  using Backends = std::remove_cvref_t<decltype(backends)>;

  // The whole-list eligibility check lives here, at the entry point. It must
  // not live in the recursive walk: there it would see only the remaining
  // *tail* of the list, and a list like [Blas, Cublas] on host tensors would
  // static_assert when the walk instantiates the [Cublas] tail, even though
  // Blas is eligible.
  static_assert(any_kernel_type_eligible_v<Backends, gemm_op, C&, scalar_t<C>, A const&, B const&, scalar_t<C>>,
                "no backend in this list can ever service gemm");
  if (!dispatch_kernel(backends, gemm_op{}, c, alpha, a, b, beta))
    throw no_available_backend{"all eligible gemm backends declined at runtime"};
}
```

Two distinct failure modes fall out, which is exactly the behaviour an override
should have:

- Forcing `make_backend_selector<backend_list<CublasBackend>>(
  CublasConfig{.device = {0}, .stream = {stream}, .math_mode = {tf32_allowed}})`
  on **host** tensors makes
  `kernel_type_acceptance<CublasBackend, gemm_op, ...>()` is
  `KernelTypeAcceptance::no`, so the entry-point `static_assert` fires — a
  **compile error**.
- Forcing `backend_list{BlasBackend{}}` when libBLAS is not loaded at runtime
  makes `try_kernel(blas_entry, gemm_op{}, ...)` return false, the list is
  exhausted, and the kernel **throws**. The default lists always end in the
  infallible oracle, so the default path never throws.

### Three local backends

```cpp
struct CpuGenericBackend {};                                // the correctness oracle

template <class T>
using operand_t = std::remove_cvref_t<T>;

template <class C, class Alpha, class A, class B, class Beta>
concept cpu_gemm_types_supported =
    is_host_v<operand_t<C>> && is_host_v<operand_t<A>> &&
    is_host_v<operand_t<B>>;

template <class Alpha, class A, class B, class Beta, class C>
consteval KernelTypeAcceptance
kernel_accepts_types(CpuGenericBackend const&, gemm_op const&, C&,
                     Alpha const&, A const&, B const&, Beta const&)
{
  if constexpr (cpu_gemm_types_supported<C, Alpha, A, B, Beta>) {
    return KernelTypeAcceptance::yes;                          // any scalar, any layout
  } else {
    return KernelTypeAcceptance::no;
  }
}

// a, b, c are resolved TensorViews — run the kernel immediately, no scheduler.
template <class C, class A, class B>
bool try_kernel(CpuGenericBackend, gemm_op, C& c, scalar_t<C> alpha,
                A const& a, B const& b, scalar_t<C> beta)
{
  static_assert(cpu_gemm_types_supported<C, scalar_t<C>, A, B, scalar_t<C>>);
  using T = scalar_t<C>;
  for (size_t i = 0; i < c.extent(0); ++i)
    for (size_t j = 0; j < c.extent(1); ++j) {
      // BLAS rule: beta == 0 means C is *not read* (a fresh output block may
      // hold uninitialized data; 0 * NaN must not poison the result).
      T acc = (beta == T{}) ? T{} : beta * c[i, j];
      for (size_t k = 0; k < a.extent(1); ++k) acc += alpha * a[i, k] * b[k, j];
      c[i, j] = acc;
    }
  return true;                                               // infallible
}

struct BlasBackend {};

template <class C, class Alpha, class A, class B, class Beta>
concept blas_gemm_types_supported =
    is_host_v<operand_t<C>> && is_host_v<operand_t<A>> &&
    is_host_v<operand_t<B>> &&
    same_scalar_v<operand_t<C>, operand_t<A>, operand_t<B>> &&
    is_blas_scalar_v<scalar_t<operand_t<C>>>;

template <class Alpha, class A, class B, class Beta, class C>
consteval KernelTypeAcceptance
kernel_accepts_types(BlasBackend const&, gemm_op const&, C&, Alpha const&,
                     A const&, B const&, Beta const&)
{
  if constexpr (blas_gemm_types_supported<C, Alpha, A, B, Beta>) {
    return KernelTypeAcceptance::maybe;                       // strides/library are runtime facts
  } else {
    return KernelTypeAcceptance::no;
  }
}

// a, b, c are resolved mdspan-like views or TensorViews lowered to such views:
// call BLAS immediately, no scheduler.
template <class C, class A, class B>
bool try_kernel(BlasBackend, gemm_op, C& c, scalar_t<C> alpha,
                A const& a, B const& b, scalar_t<C> beta)
{
  static_assert(blas_gemm_types_supported<C, scalar_t<C>, A, B, scalar_t<C>>);
  return uni20::linalg::blas::try_gemm(c, alpha, a, b, beta);
}

struct CublasBackend {
  int device;
  cudaStream_t stream;
  math_mode_t math_mode;
};

template <class Alpha, class A, class B, class Beta, class C>
consteval KernelTypeAcceptance
kernel_accepts_types(CublasBackend const&, gemm_op const&, C&, Alpha const&,
                     A const&, B const&, Beta const&)
{
  if constexpr (is_device_v<operand_t<C>> &&
                is_device_v<operand_t<A>> &&
                is_device_v<operand_t<B>> &&
                same_scalar_v<operand_t<C>, operand_t<A>, operand_t<B>> &&
                is_blas_scalar_v<scalar_t<operand_t<C>>>) {
    return KernelTypeAcceptance::maybe;                       // device placement is runtime
  } else {
    return KernelTypeAcceptance::no;
  }
}

// Unlike the CPU backends, this *does* go through a scheduler: the CudaScheduler
// builds sequencing into its streams/events, so the submit is correctly ordered.
// (A blocking variant on the default stream is also possible, but the scheduler
// path is typical.)
template <class C, class A, class B>
bool try_kernel(CublasBackend backend, gemm_op, C& c, scalar_t<C> alpha,
                A const& a, B const& b, scalar_t<C> beta)
{
  int dev = backend.device;
  cublas::set_math_mode(backend.math_mode);
  if (c.device() != dev) return false;
  for (int other : {a.device(), b.device()})                // operands on another GPU?
    if (other != dev && !cuda::peer_access(dev, other))
      return false;                                         // let a staging backend handle it
  gpu_scheduler(dev).submit(cublas_gemm_task<C,A,B>, c.write(),    // CudaScheduler: stream/event
                            alpha, a.read(), b.read(), beta);      //   sync is built in; per-GPU
  return true;                                                     //   thread holds the cuBLAS handle
}
```

Three invariants this surfaces:

- **Dispatch reads only metadata.** `layout()`, `device()`, dtype, and extents
  are all available synchronously, so `try_kernel` can decide before any data is
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
struct MpiBackend {
  mpi_comm_t comm;
  placement_map_t placement;
};                                                          // distribution layer only

// is_distributed_v / is_replicated_v are traits of the container's
// BlockTensorStorage policy (block_tensor.md §1/§5) — distribution is a
// container capability, never a dense TensorStorage. Replicated × distributed
// is the *common* case (a replicated MPO contracted with a distributed state),
// so the predicate must not demand distribution of every operand.
template <class Alpha, class A, class B, class Beta, class C>
consteval KernelTypeAcceptance
kernel_accepts_types(MpiBackend const&, gemm_op const&, C&, Alpha const&,
                     A const&, B const&, Beta const&)
{
  if constexpr ((is_distributed_v<operand_t<C>> ||
                 is_distributed_v<operand_t<A>> ||
                 is_distributed_v<operand_t<B>>) &&
                all_distributed_or_replicated_v<operand_t<C>,
                                                operand_t<A>,
                                                operand_t<B>>) {
    return KernelTypeAcceptance::maybe;                       // plan construction is runtime
  } else {
    return KernelTypeAcceptance::no;
  }
}

template <class C, class A, class B>
bool try_kernel(MpiBackend backend, gemm_op, C& c, scalar_t<C> alpha,
                A const& a, B const& b, scalar_t<C> beta)
{
  auto& comm = backend.comm;
  auto& placement = backend.placement;
  auto plan = mpi::plan_gemm(a.distribution(), b.distribution(), c.distribution());
  (void)comm;
  (void)placement;
  if (!plan) return false;        // decline *before* any side effect (see below)
  for (auto const& edge : plan->exchanges)
    mpi::post(edge);              // Isend/Irecv first; unique tag per edge (ordering_and_backend_lowering.md)
  for (auto const& t : plan->local_tiles)                    // overlap compute with the exchanges
    gemm(t.c_local, alpha, t.a_local, t.b_local, t.beta);    // recursion -> local default list
  for (auto const& t : plan->remote_tiles) {                 // contributions needing received data
    mpi::wait(t.recv);              // schematic: a real impl schedules these as tasks gated on the receives
    gemm(t.c_local, alpha, t.a_recv, t.b_recv, scalar_t<C>{1});  // accumulate into the local tile
  }
  return true;
}
```

One contract the composite backend makes explicit: **`try_kernel` declines
before any side effect, or not at all.** Once a backend has posted a message or
executed a tile, returning `false` would hand the next backend in the list a
partially updated `c`. So all feasibility checks (here `mpi::plan_gemm`) come
first, and past the first side effect the backend is committed — a later failure
is an error to report, not a decline. The simple backends satisfy this trivially
(their checks are reads of metadata and library availability); any backend that
composes other work must be written to it deliberately.

Because `try_kernel(mpi_entry, gemm_op{}, ...)` recurses into `gemm` on the
**local** tensors, which re-derive *their own* default list, **MPI/CPU versus
MPI/CUDA is just whether the node-local storage — the `X` in `Mpi<X>` — is host
or device** — the same `MpiBackend` covers both and nesting supplies the local
kernel. This is also how an optional MPI_RPC backend would layer on: a
front-of-list backend implemented on top of the collective MPI backend, one more
level of the same nesting. None of it requires core changes, so the first pass
ships with the local backends only.

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
gemm(c, alpha, a, b, beta);                                  // default for the storage
gemm(CpuGenericBackend{}, c, alpha, a, b, beta);             // force one backend, no fallback
gemm(backend_list{BlasBackend{}, CpuGenericBackend{}},
     c, alpha, a, b, beta);                                  // pin BLAS-or-oracle
gemm(BackendChain{BlasBackend{}, CpuGenericBackend{}},
     c, alpha, a, b, beta);                                  // same idea, chain spelling
gemm(make_backend_selector<backend_list<CublasBackend>>(
       CublasConfig{.device = {0}, .stream = {stream}, .math_mode = {tf32_allowed}}),
     c, alpha, a, b, beta);                                  // GPU only, no fallback - fail loudly
```

Singleton backend overrides are syntax for a one-entry backend list. They do not
get an implicit oracle fallback; if the backend declines at runtime, the
operation throws just like an exhausted one-element list.

`BackendChain<...>` or `BackendChain{...}` should be treated as selector
spelling, not a new leaf-kernel protocol. Its acceptance is `no` only when every
entry is `no`; it is `yes` only when the chain can guarantee success without an
outer runtime decline; otherwise it is `maybe`.

## What this pins down

- The dispatch mechanism is a backend-list value plus an operation tag and
  detected backend customization points: optional `kernel_accepts_types(...)`
  for type-level no/maybe/yes pruning and required `try_kernel(...)` for runtime
  attempts. The list's type gives static ordering; the list entries carry any
  runtime backend state they need. No inheritance, no public runtime tags, and
  no per-kernel dispatcher boilerplate.
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

1. **Kernels** — plain functions over resolved mdspan-like views (`TensorView`,
   `stdex::mdspan`, transform views). `cblas_dgemm`, a cuBLAS call on a given
   stream, a generic triple loop. CPU kernels are entirely async-unaware; they
   compute and return. A CUDA kernel takes a stream or handle but does not know
   about the async runtime.
2. **Async dispatch wrappers** — run backend selection at submission time (it
   reads only metadata, which is synchronous), then schedule a coroutine that
   `co_await`s the operand buffers, resolves them to `TensorView`s, and calls the
   already-selected kernel. By the time a coroutine is scheduled, dispatch has
   succeeded: the coroutine is a thin wrapper around the raw backend kernel call.
   This is the shape of the existing `Async<T>` operations in `async_ops.hpp`
   (`async_binary_op`, `async_compound_op`). Synchronization lives here, in the
   epoch-queue awaits; the kernel never sees it.
3. **Backend selection** — the `backend_list` walk for an operation tag
   (`kernel_accepts_types` / `try_kernel`) and, for a block tensor, the planner
   that fans out per block. A CPU backend's `try_kernel` calls a layer-1 kernel
   directly and synchronously; the CUDA backend submits to the `CudaScheduler`.
   Layer 2 sits *above* this walk in the layering, but the walk runs *first* in
   time: selection happens at submission, on synchronous metadata, and the
   scheduled coroutine then awaits the buffers and runs the kernel that was
   selected.

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
schedule([](auto c_, auto a_, auto b_) static -> AsyncTask {         // -> CPU scheduler
  auto C_ = co_await c_;  auto A_ = co_await a_;  auto B_ = co_await b_;
  gemm_kernel(C_, A_, B_);                                          // layer-1 kernel
}(c.block(r).write(), a.block(ai).read(), b.block(bi).read()));

// device-async block kernel — identical but for the return type
schedule([](auto c_, auto a_, auto b_) static -> CudaTask {          // -> GPU scheduler
  auto C_ = co_await c_;  auto A_ = co_await a_;  auto B_ = co_await b_;
  cublas_gemm(C_, A_, B_);                                          // thread-local handle + stream
}(c.block(r).write(), a.block(ai).read(), b.block(bi).read()));
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
schedule([](auto c_, auto a_, auto b_) static -> AsyncTask {
  auto&        C = co_await c_;     // structure + layout known
  auto const& A = co_await a_;
  auto const& B = co_await b_;
  block_gemm(C, A, B);             // inner per-block dispatch
}(C.write(), A.read(), B.read()));
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

1. **The backend interface.** Exact CPO names, the `std::tuple` type-pack representation,
   and how a backend advertises which scheduler(s) it targets.
2. **List configuration.** The default travels with the tensor (storage type),
   and C++ overrides may be either a singleton backend value or an ordered
   backend-list value. The list has static type order, and the selector carries
   a composed state tuple. The open question is which state tags should be
   considered structural (CUDA device, scheduler target, allocator/resource)
   versus advisory (algorithm id, tile shape, math mode, workspace limit). A
   fully `RuntimeSelectable` case can still be layered on later by erasing
   backend-entry types.
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
   be expressible as one operation tag / backend capability emitting one task,
   per `block_coalescing.md`.
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
