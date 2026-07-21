# Kernel Dispatch Design

**Status: implemented dense BLAS, CPU-reference, and initial LAPACK slices plus
forward design.** Uni20 has an operation-tag dispatch walk, structured failure
reporting, opt-in runtime dispatch diagnostics, Tensor-to-mdspan forwarding,
output-shape preparation, accessor-respecting copy/materialization, direct
BLAS/reference CPU backends, and LAPACK adapters used by the native Krylov
projected problems. The first all-async Tensor matrix-product wrappers await
their values and enter the same operation-tag backend walk through
`co_dispatch_kernel`. The CUDA, distributed, prepared-operand, and broader
BLAS/LAPACK portions remain design work. This note
captures both the implemented contract and that direction. It generalizes the
three-stage pattern in
[`backend_dispatch.md`](backend_dispatch.md) into a single configurable
mechanism that also covers device and distributed execution. Treat the code
below as schematic where it discusses unimplemented backends; the current CPO
and dispatch names match the implementation but are not a frozen API.

The static capability CPO name used here is `kernel_accepts_types(...)`. The
runtime attempt CPO is `try_kernel(...)`. Both put the backend value first, then
the operation tag, then the operation arguments.

Related notes:

- [`../async/kernel_authoring.md`](../async/kernel_authoring.md) — implemented
  all-async Tensor wrapper contract, lifetime rules, and output semantics.
- [`backend_dispatch.md`](backend_dispatch.md) — the compile-time capability /
  runtime attempt / generic fallback pattern this note generalizes.
- [`../linalg/mdspan_dispatch.md`](../linalg/mdspan_dispatch.md) — the
  planned first concrete implementation slice of this model for dense linalg
  leaf kernels over strided mdspan views.
- [`ordering_and_backend_lowering.md`](ordering_and_backend_lowering.md) — the
  async scheduler owns all ordering; backends emit work, they do not order it.
- [`execution.md`](execution.md) — mechanism/policy
  split and scheduler shapes.
- [`storage_kind_and_location.md`](storage_kind_and_location.md) — memory kind
  vs location, the axes dispatch keys on.
- [`../symmetry/block_coalescing.md`](../symmetry/block_coalescing.md) — coalesced/batched kernels as a
  backend capability.

## The model in one paragraph

A **backend** is a uniform thing with two ways to decline an operation. The
dispatch mechanism is generic over an **operation value** (`copy_op`, `gemm_op`,
`gemv_op`, ...), and discovers backend support with customization points. Most
operations are empty tags, but an operation value may carry immutable options
or callable state. Concrete dense-linalg operation types and their diagnostic
names live in the central `src/uni20/linalg/operation_tags.hpp` catalogue;
backend headers use that catalogue rather than redeclaring an operation
locally:

- `kernel_accepts_types(Backend const&, Op const&, Args&...)` — required
  `consteval` tri-state function for compile-time facts about the C++ types;
  prunes the candidate at zero runtime cost. It returns `kernel_types_no`,
  `kernel_types_maybe`, or `kernel_types_yes`, whose distinct types let the
  dispatcher inspect the result through `decltype` without manufacturing
  operand objects. The CPO keeps ordinary reference parameters, so it does not
  need a tuple or explicit type-pack token.
- `try_kernel(Backend, Op, args...)` — required for an implemented operation;
  receives the backend value, checks runtime facts, and either performs/emits
  the work or returns a structured clean-decline `KernelAttempt`.

A non-success `KernelAttempt` has a strong decline guarantee. The backend must
not consume or move from an argument, semantically mutate an input or output,
commit an allocation, submit work, or produce another externally visible side
effect. Dispatch invokes candidates with the same stable lvalue arguments and
does not copy mdspan descriptors or operand objects to conceal contract
violations. Once a backend changes externally visible state, later failure is
an operation error rather than a dispatch decline. A backend whose type gate
returns `kernel_types_yes` must return `KernelAttempt::success`; dispatch treats
any decline as a backend logic error. Terminal execution failures throw, abort
through a logic check, or use an operation-specific result and never trigger
fallback.

The gate may be narrowly constrained. Type probing uses unevaluated lvalue
expressions for the underlying deduced argument types; it does not inspect
runtime values. Runtime `try_kernel(...)` calls use the corresponding stable
lvalue arguments.
If the gate is not callable for the backend, operation, and argument types,
acceptance is a hard `no`; a callable gate returns the corresponding acceptance
constant. Its `try_kernel(...)` overload may remain deliberately unconstrained
because dispatch never instantiates it for rejected types. This keeps the
runtime CPO focused on instance facts rather than duplicating type constraints.

Generic callers should normally use the safe value-shaped front end:

```cpp
probe_dispatch_kernel(backends, op, args...);
```

It deduces but does not inspect argument values. The result is `yes` if any
candidate is `yes`, otherwise `maybe` if any candidate is `maybe`, otherwise
`no`. A backend whose `kernel_accepts_types(...)` customization is not callable
contributes `no`.

Conformance tests and other compile-time introspection can retain the complete
ordered candidate set instead:

```cpp
auto candidates = kernel_type_candidates(backends, op, args...);
```

The returned `backend_list` contains every backend whose type acceptance is
`yes` or `maybe`, preserving selector order and backend object state. It does
not inspect runtime values, layouts, dimensions, or provider availability, so a
retained `maybe` backend can still decline when tested. The corresponding
`kernel_type_candidates_t<BackendSelector, Op, Args...>` alias exposes the
filtered list type without constructing operands.

Runtime dispatch has separate trial and checked front ends:

- `try_dispatch_kernel(backends, op, args...)` walks the eligible backends and
  returns `false` when every runtime candidate declines before side effects.
  It is constrained out when the aggregate type probe is `no`.
- `dispatch_kernel(backends, op, args...)` performs the same walk and reports an
  exhausted list by raising `KernelDispatchError`. Native C++ therefore renders
  its presentation report and aborts with a stacktrace, while importing the
  Python extension selects exception mode for recoverable errors. Like the
  trial form, it is not callable when the aggregate type probe is `no`.
- `dynamic_dispatch_kernel(backends, op, args...)` is the language-binding and
  runtime-erased boundary. It remains callable when the type probe is `no` and
  converts that rejection into `KernelDispatchError`, so Python receives an
  exception when a kernel was not compiled into the active backend list.

Operation values and backend values define stable `static constexpr
std::string_view name` members. These names are diagnostic metadata only; C++
types remain the dispatch identity and the names are not serialization or ABI
keys.

Generic elementwise dispatch uses callable-carrying `transform_op<F>` and
`transform_inplace_op<F>` values. The callable is stored by value and invoked
as const. A named callable type can therefore select an optimized backend while
an arbitrary lambda remains eligible for the CPU reference backend. Input
elements and the old update value are passed as values, so mutable mdspan
accessors do not turn semantic inputs into writable callable arguments.

Tensor `conj(...)` is a lazy read-only view whose resolved mdspan uses the
conjugating accessor. Explicit `copy(...)` and `make_tensor(...)` dispatch
`copy_op`; the current CPU reference backend reads through accessor semantics.
This leaves rank-two copy open to a future BLAS `omatcopy`-style backend without
turning view construction into eager work.

The rendered diagnostic begins with a failure-styled line naming the operation,
followed by the failure category and ordered backend candidate table. It then
includes the `trace::raise(...)` source location and, in builds with
`std::stacktrace` support, the captured stacktrace.

## Runtime dispatch diagnostics

Successful and trial dispatches can expose the same ordered backend information
through `uni20::linalg::dispatch_diagnostics`. Diagnostics are disabled by
default. A disabled dispatch performs one inline relaxed atomic flag load and
branch; it does not construct a record, allocate, lock, or invoke a sink.

Install a sink for structured `dispatch_diagnostics::event` values:

```cpp
uni20::linalg::dispatch_diagnostics::scoped_sink diagnostics(
    [](uni20::linalg::dispatch_diagnostics::event const& event) {
      uni20::display::emit(uni20::linalg::diagnostic_report(event),
                           uni20::display::stream::out);
    });
```

Each event contains the operation name and every ordered backend candidate.
The candidate records distinguish:

- a type-level hard `no`, rendered as `not eligible`;
- an attempted runtime decline and its `KernelAttempt` reason;
- the successful selected backend;
- an eligible fallback after the selected backend, rendered as `not attempted`.

`set_sink(...)` and `reset_sink()` support longer-lived application or binding
configuration; `scoped_sink` is convenient for tests and local tools. The sink
is process-wide and is invoked synchronously after the backend walk. Concurrent
dispatches may invoke it concurrently, so a collecting sink must synchronize
its own state. A sink can render through `display`, append structured data to a
file, or hand the event to a Python-aware boundary. Diagnostic sinks must not
throw: an enabled sink runs after a backend may already have completed and
mutated its output.

Runnable examples under `examples/linalg/` cover the main dispatch behaviors:

- `kernel_dispatch_example.cpp` defines a small operation and two backends,
  demonstrating type probing, clean runtime decline, and ordered fallback.
- `gemm_dispatch_example.cpp` performs tensor GEMM through the selector supplied
  by `VectorStorage`, renders all matrices before and after the operation, shows
  the runtime backend walk, and supports `fp32`, `fp64`, plus `fp128` when the
  build enables MPLAPACK binary128 support.
- `kernel_dispatch_error_example.cpp` catches and renders structured errors for
  both runtime exhaustion and a type-level hard `no` at a dynamic boundary.

`KernelAttempt::success` means the work is done or correctly emitted. (For
deferred device work, "correctly emitted" means the completion token — the CUDA
event recorded at submission, per `ordering_and_backend_lowering.md` — is
attached to the buffers' epochs before the backend returns. The token is
buffer-lifetime machinery owned by the storage policy; it never travels through
the dispatch interface.)
Dispatch is a walk over an **ordered list of backends**: the first backend that
does not decline wins. A reference implementation may be the last entry in a
list and act as its correctness oracle over the types it accepts. Backends
**nest** — a backend may be implemented in terms of other backends — which is
how distributed and RPC execution layer on without a class hierarchy. This
replaces the original tagged-dispatch-with-inheritance plan, which was too rigid.

Default backend lists remain within one storage and execution domain. Host
storage may select host BLAS followed by the host CPU reference backend; future
CUDA storage selects CUDA-device backends and does not fall through to a host
CPU backend. An ordinary runtime decline never transfers operands or changes
execution domain. A deliberate emergency device-to-host implementation, if one
is needed for an individual operation, must be an explicit composite kernel or
higher-level route whose transfer and synchronization costs are visible in its
policy. It is not a general fallback backend.

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
   `BlockTensorStorage` policy (`../symmetry/block_tensor.md` §1/§5). A caller can override
   it with either an ordered backend-list value or a single backend value. A
   single backend is normalized to a one-element list, which is convenient for
   forcing a backend during benchmarking, debugging, or correctness checks.
2. There is **no runtime type-erased backend selection in the first pass.**
   Python bindings are compiled against a fixed target list (for example
   `[Blas, CpuGeneric]`, or `[Cublas, Blas, CpuGeneric]`) baked in at build time.
   Explicit overrides may still carry operation context such as a stream,
   workspace pool, communicator, or precision setting. Operand placement stays
   in storage/accessor-defined data handles. A `RuntimeSelectable` backend or a `std::variant` of backends
   can be layered on later if needed; it is just another list entry.

New Uni20 kernel and linalg APIs put API tags and explicit backend selectors
first, then mutable outputs, then inputs and scalar operands. Tensor front ends
use `gemm(selector, c, alpha, a, b, beta)` or
`gemv(selector, y, alpha, a, x, beta)` for an explicit override and omit the
selector for the storage default. Bare mdspans use `dispatch_kernel` directly
with `gemm_op` or `gemv_op` rather than adding operation-specific dispatch
aliases.

## Dispatch operands versus resolved spans

Tensor operations derive backend candidates from Uni20 Tensor-view operands.
They then resolve `mdspan()` before calling the backend
walk. A plain `stdex::mdspan` has enough information to run a leaf kernel but
does not carry the storage policy needed to choose a default backend stack.

The layers are therefore:

1. **Tensor operation wrapper** — accepts `TensorView` / `MutableTensorView`
   concept operands, derives or receives a selector, decides fixed versus owning
   output policy, and resolves mdspans.
2. **Backend dispatch** — walks the backend list for an operation tag over the
   resolved mdspan operands.
3. **Leaf kernel** — receives rank-constrained mdspans and either calls a
   provider or runs a generic implementation.

Passing mdspans directly is the lower-level interface. The caller provides an
explicit backend selector and operation tag to `dispatch_kernel` or
`try_dispatch_kernel`; Tensor-view wrappers themselves are not mdspan-like.

## Worked example: gemm

The contraction primitive `C <- alpha * A * B + beta * C` exercises every part of
the model. The entry point derives a default list from storage and forwards to
the walk:

The first end-to-end mdspan implementation slice exists for explicit backend
selectors. The low-level mdspan leaf is
`uni20::linalg::blas::try_gemm(c, alpha, a, b, beta)`;
`try_kernel(BlasBackend, gemm_op, ...)` wraps that leaf, and
`CpuReferenceBackend` provides an independently tested fallback when BLAS
declines. GEMV applies the same layering to rank-one vector descriptors and
BLAS increments, including fallback for conjugating input accessors. The first
`LapackBackend` operations extend the same walk to symmetric tridiagonal and
nonsymmetric eigensystems, Schur decomposition, Hessenberg Schur reduction, and
Schur reordering. Their workspace is backend-owned LAPACK work storage rather
than hidden operand materialization.

```cpp
#include <uni20/linalg/operation_tags.hpp>

enum class KernelTypeAcceptance {
  no,     // invalid for these types; do not form try_kernel
  maybe, // runtime values decide
  yes    // runtime attempt is expected to succeed for every instance
};

enum class KernelAttempt {
  success,
  unsupported_instance,
  unsupported_shape,
  unsupported_layout,
  unsupported_accessor,
  unsupported_transform,
  unavailable,
  insufficient_resources
};

template <class Backend, class Op, class... Args>
consteval bool backend_has_try_kernel()
{
  return requires(Backend backend, Op op, Args&&... args) {
    { try_kernel(backend, op, args...) } -> std::same_as<KernelAttempt>;
  };
}

template <class T>
using kernel_type_probe_arg_t = std::remove_reference_t<T>&;

template <class Backend, class Op, class... Args>
consteval KernelTypeAcceptance backend_type_acceptance()
{
  using backend_type = std::remove_cvref_t<Backend>;
  using op_type = std::remove_cvref_t<Op>;
  if constexpr (requires {
             kernel_accepts_types(std::declval<backend_type const&>(),
                                  std::declval<op_type const&>(),
                                  std::declval<kernel_type_probe_arg_t<Args>>()...);
           }) {
    using result_type = std::remove_cvref_t<decltype(kernel_accepts_types(
        std::declval<backend_type const&>(), std::declval<op_type const&>(),
        std::declval<kernel_type_probe_arg_t<Args>>()...))>;
    constexpr auto acceptance = result_type::value;
    if constexpr (acceptance == KernelTypeAcceptance::no) {
      return KernelTypeAcceptance::no;
    } else {
      static_assert(backend_has_try_kernel<backend_type, op_type, Args...>(),
                    "kernel_accepts_types accepted these types, but try_kernel is not available");
      return acceptance;
    }
  }

  return KernelTypeAcceptance::no; // a non-callable type gate is a hard rejection
}

template <class... Backends, class Op, class... Args>
constexpr KernelTypeAcceptance
probe_dispatch_kernel(backend_list<Backends...> const&, Op const&, Args&&...)
{
  // Query detail::backend_type_acceptance for every candidate using probe
  // lvalues. Return yes if any candidate is yes, then maybe, otherwise no.
}

template <class Op, class First, class... Rest, class... Args>
  requires detail::KernelDispatchTypesAccepted<backend_list<First, Rest...>, Op, Args...>
bool try_dispatch_kernel(backend_list<First, Rest...> backends, Op op, Args&&... args)
{
  constexpr auto acceptance = detail::backend_type_acceptance(
      detail::kernel_type_probe_arg<First const&>(),
      detail::kernel_type_probe_arg<Op const&>(),
      detail::kernel_type_probe_arg<Args>()...);
  if constexpr (acceptance == KernelTypeAcceptance::yes) {
    KernelAttempt const attempt = try_kernel(backends.first(), op, args...);
    CHECK(kernel_attempt_succeeded(attempt));
    return true;
  } else if constexpr (acceptance == KernelTypeAcceptance::maybe) {
    if (kernel_attempt_succeeded(try_kernel(backends.first(), op, args...)))
      return true;
  }

  if constexpr (sizeof...(Rest) > 0)
    return try_dispatch_kernel(backends.rest(), op, args...);
  else
    return false;
}

template <class... Backends, class Op, class... Args>
  requires detail::KernelDispatchTypesAccepted<backend_list<Backends...>, Op, Args...>
void dispatch_kernel(backend_list<Backends...> backends, Op op, Args&&... args)
{
  ERROR_IF(!try_dispatch_kernel(backends, op, std::forward<Args>(args)...),
           "every eligible backend declined the kernel operation");
}

template <class... Backends, class Op, class... Args>
void dynamic_dispatch_kernel(backend_list<Backends...> backends, Op op, Args&&... args)
{
  if constexpr (!detail::KernelDispatchTypesAccepted<backend_list<Backends...>, Op, Args...>)
    trace::raise(detail::make_kernel_dispatch_error<KernelDispatchFailure::no_eligible_backend>(
        backends, op, args...));
  else
    dispatch_kernel(backends, op, std::forward<Args>(args)...);
}

// gemm.hpp: Tensor-view convenience overload
template <MutableRankedTensorView<2> C, RankedTensorView<2> A,
          RankedTensorView<2> B>
void gemm(C&& c, scalar_t<C> alpha, A const& a, B const& b, scalar_t<C> beta)
{
  auto selector = select_backend(gemm_op{}, c, a, b);
  gemm(selector, c, alpha, a, b, beta);
}

// Tensor adapter: resolve views before entering kernel dispatch.
template <class BackendSelector, MutableRankedTensorView<2> C,
          RankedTensorView<2> A, RankedTensorView<2> B>
void gemm(BackendSelector selector, C& c, scalar_t<C> alpha, A const& a,
          B const& b, scalar_t<C> beta)
{
  dispatch_kernel(selector, gemm_op{}, c.mdspan(), alpha,
                  a.mdspan(), b.mdspan(), beta);
}
```

`select_backend(operation, operands...)` first requires a common tensor storage
policy. Operand values are accepted for front-end ergonomics but are not
inspected. The default returns the storage policy's static selector. A global
`backend_selector_override<Operation, StoragePolicy>` specialization may define
`select(operation)` to replace that default for a particular operation/storage
combination. Passing an explicit selector bypasses this default-selection
customization point.

Two distinct failure modes fall out, which is exactly the behaviour an override
should have:

- Forcing `make_backend_selector<backend_list<CublasBackend>>(
  CublasConfig{.device = {0}, .stream = {stream}, .math_mode = {tf32_allowed}})`
  on **host** tensors makes
  `probe_dispatch_kernel(cublas_only, gemm_op{}, host_operands...)` is
  `KernelTypeAcceptance::no`, so checked dispatch is constrained out — a
  **compile error**. A Python binding calls `dynamic_dispatch_kernel` for its
  concrete binding instantiation instead, converting the same static rejection
  into a Python exception.
- Forcing `backend_list{BlasBackend{}}` for an mdspan layout or accessor that
  the direct BLAS adapter cannot represent makes `try_kernel(...)` return
  `false`, the list is
  exhausted, and checked dispatch raises `KernelDispatchError`: native C++
  renders the structured report and aborts with a stacktrace, while Python
  receives the concrete exception. The current host default ends in the CPU
  reference backend, which is total only over its accepted semantic type
  domain; no default list is required to accept arbitrary argument types.

### Three local backends

```cpp
struct CpuReferenceBackend {};                              // the correctness oracle

template <class T>
using operand_t = std::remove_cvref_t<T>;

template <class Alpha, class A, class B, class Beta, class C>
consteval auto
kernel_accepts_types(CpuReferenceBackend const&, gemm_op const&, C&,
                     Alpha const&, A const&, B const&, Beta const&)
{
  if constexpr (/* rank-2 addressable operands with compatible scalar types */) {
    return kernel_types_yes;                                  // any scalar, any layout
  } else {
    return kernel_types_no;
  }
}

// a, b, c are resolved mdspans — run the kernel immediately, no scheduler.
template <class C, class A, class B>
KernelAttempt try_kernel(CpuReferenceBackend, gemm_op, C& c,
                          scalar_t<C> alpha, A const& a, B const& b,
                          scalar_t<C> beta)
{
  using T = scalar_t<C>;
  for (size_t i = 0; i < c.extent(0); ++i)
    for (size_t j = 0; j < c.extent(1); ++j) {
      // BLAS rule: beta == 0 means C is *not read* (a fresh output block may
      // hold uninitialized data; 0 * NaN must not poison the result).
      T acc = (beta == T{}) ? T{} : beta * c[i, j];
      for (size_t k = 0; k < a.extent(1); ++k) acc += alpha * a[i, k] * b[k, j];
      c[i, j] = acc;
    }
  return KernelAttempt::success;                             // infallible
}

struct BlasBackend {};

template <class Alpha, class A, class B, class Beta, class C>
consteval auto
kernel_accepts_types(BlasBackend const&, gemm_op const&, C&, Alpha const&,
                     A const&, B const&, Beta const&)
{
  if constexpr (/* rank-2 strided BLAS-compatible operands */) {
    return kernel_types_maybe;                               // strides/library are runtime facts
  } else {
    return kernel_types_no;
  }
}

// a, b, c are resolved mdspan-like views:
// call BLAS immediately, no scheduler.
template <class C, class A, class B>
KernelAttempt try_kernel(BlasBackend, gemm_op, C& c, scalar_t<C> alpha,
                          A const& a, B const& b, scalar_t<C> beta)
{
  return uni20::linalg::blas::try_gemm(c, alpha, a, b, beta);
}

template <class Alpha, class A, class B, class Beta, class C>
consteval auto
kernel_accepts_types(CublasBackend const&, gemm_op const&,
                     C&, Alpha const&, A const&, B const&, Beta const&)
{
  if constexpr (/* rank-2 opaque CUDA mdspans with cuBLAS scalar types */) {
    return kernel_types_maybe;                 // layout/device are runtime facts
  } else {
    return kernel_types_no;
  }
}

template <class C, class A, class B>
KernelAttempt try_kernel(CublasBackend, gemm_op, C& c,
                         scalar_t<C> alpha, A const& a, B const& b,
                         scalar_t<C> beta)
{
  return uni20::linalg::cublas::try_gemm(c, alpha, a, b, beta);
}
```

Three invariants this surfaces:

- **Dispatch reads only metadata.** `layout()`, `device()`, dtype, and extents
  are all available synchronously, so `try_kernel` can decide before any data is
  read. The kernel then runs on the resolved mdspans.
- **CPU backends are synchronous; they do not touch a scheduler.** `BlasBackend`
  and `CpuReferenceBackend` call the kernel immediately on resolved views. Async
  execution of CPU work comes from wrapping the tensors in `Async<T>`, whose epoch
  queues provide the sequencing — submitting a bare CPU task to a scheduler with no
  buffer await would carry no ordering and never be correct. There is no point in a
  separate "CPU async backend": `Async<T>` already is it.
- **CUDA admission follows the operation entry point.** Ordinary
  `CublasBackend` preparation is followed by blocking resource admission.
  Async fixed-output GEMM and matrix-product lowering use
  `co_dispatch_kernel`, whose optional
  cuBLAS `try_kernel_task` customization prepares before admission and returns
  a task that awaits resources before invoking the prepared provider leaf.
  Note the division of labour: *ordering* is already guaranteed by the async
  layer; events provide device *synchronization* and buffer lifetime across
  streams.

### MPI falls out as nesting

A distributed backend handles only the distribution layer and recurses into the
local kernel for the per-MPI-rank compute:

```cpp
struct MpiBackend {
  mpi_comm_t comm;
  placement_map_t placement;
};                                                          // distribution layer only

// is_distributed_v / is_replicated_v are traits of the container's
// BlockTensorStorage policy (../symmetry/block_tensor.md §1/§5) — distribution is a
// container capability, never a dense TensorStorage. Replicated × distributed
// is the *common* case (a replicated MPO contracted with a distributed state),
// so the predicate must not demand distribution of every operand.
template <class Alpha, class A, class B, class Beta, class C>
consteval auto
kernel_accepts_types(MpiBackend const&, gemm_op const&, C&, Alpha const&,
                     A const&, B const&, Beta const&)
{
  if constexpr ((is_distributed_v<operand_t<C>> ||
                 is_distributed_v<operand_t<A>> ||
                 is_distributed_v<operand_t<B>>) &&
                all_distributed_or_replicated_v<operand_t<C>,
                                                operand_t<A>,
                                                operand_t<B>>) {
    return kernel_types_maybe;                               // plan construction is runtime
  } else {
    return kernel_types_no;
  }
}

template <class C, class A, class B>
KernelAttempt try_kernel(MpiBackend backend, gemm_op, C& c,
                          scalar_t<C> alpha, A const& a, B const& b,
                          scalar_t<C> beta)
{
  auto& comm = backend.comm;
  auto& placement = backend.placement;
  auto plan = mpi::plan_gemm(a.distribution(), b.distribution(), c.distribution());
  (void)comm;
  (void)placement;
  if (!plan) return KernelAttempt::unsupported_instance; // before side effects
  for (auto const& edge : plan->exchanges)
    mpi::post(edge);              // Isend/Irecv first; unique tag per edge (ordering_and_backend_lowering.md)
  for (auto const& t : plan->local_tiles)                    // overlap compute with the exchanges
    gemm(t.c_local, alpha, t.a_local, t.b_local, t.beta);    // recursion -> local default list
  for (auto const& t : plan->remote_tiles) {                 // contributions needing received data
    mpi::wait(t.recv);              // schematic: a real impl schedules these as tasks gated on the receives
    gemm(t.c_local, alpha, t.a_recv, t.b_recv, scalar_t<C>{1});  // accumulate into the local tile
  }
  return KernelAttempt::success;
}
```

One contract the composite backend makes explicit: **`try_kernel` declines
before any side effect, or not at all.** Once a backend has posted a message or
executed a tile, returning a decline result would hand the next backend in the
list a partially updated `c`. So all feasibility checks (here
`mpi::plan_gemm`) come first, and past the first side effect the backend is
committed — a later failure is an error to report, not a decline. The simple backends satisfy this trivially
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
| `Tensor<T, Rank, HostStorage>` | `[Blas, CpuGeneric]` | BLAS then oracle |
| `Tensor<T, Rank, DeviceStorage>` | `[Cublas, DeviceGeneric]` | cuBLAS then device oracle |
| `BlockTensor<…, Mpi<Host>>` | `[Mpi]` | Mpi composed with `[Blas, CpuGeneric]` per MPI rank |
| `BlockTensor<…, Mpi<Cuda>>` | `[Mpi]` | Mpi composed with `[Cublas, DeviceGeneric]` per MPI rank |

`Mpi<…>` never appears on a dense `Tensor`: distribution is a capability of the
block-sparse *container*'s `BlockTensorStorage` policy, not a dense memory kind
(`../symmetry/block_tensor.md` §1/§5 — a dense tensor striped over MPI ranks is conceivable but
off-roadmap). The dense rows dispatch on the leaf `TensorStorage` only.

```cpp
gemm(c, alpha, a, b, beta);                                  // Tensor storage default
gemm(CpuReferenceBackend{}, c, alpha, a, b, beta);           // Tensor override
dispatch_kernel(backend_list{BlasBackend{}, CpuReferenceBackend{}},
                gemm_op{}, c_span, alpha, a_span, b_span,
                beta);                                       // bare mdspans
gemm(BackendChain{BlasBackend{}, CpuReferenceBackend{}},
     c, alpha, a, b, beta);                                  // Tensor override
gemm(make_backend_selector<backend_list<CublasBackend>>(
       CublasConfig{.device = {0}, .stream = {stream}, .math_mode = {tf32_allowed}}),
     c, alpha, a, b, beta);                                  // GPU only, no fallback - fail loudly
```

Singleton backend overrides are syntax for a one-entry backend list. They do not
get an implicit oracle fallback; if the backend declines at runtime, the
checked operation raises `KernelDispatchError` just like any exhausted backend
list.

`BackendChain<...>` or `BackendChain{...}` should be treated as selector
spelling, not a new leaf-kernel protocol. Its acceptance is `no` only when every
entry is `no`; it is `yes` only when the chain can guarantee success without an
outer runtime decline; otherwise it is `maybe`.

## What this pins down

- The dispatch mechanism is a backend-list value plus an operation tag and
  detected backend customization points: required `kernel_accepts_types(...)`
  for type-level no/maybe/yes eligibility and required `try_kernel(...)` for
  runtime attempts. The list's type gives static ordering; the list entries carry any
  runtime backend state they need. No inheritance, no public runtime tags, and
  no per-kernel dispatcher boilerplate.
- The **dispatch-to-async seam is a generic coroutine walk over the same backend
  list.** The implemented whole-value `Async<Tensor>` wrappers resolve the
  static storage selector, then schedule a coroutine which `co_await`s the
  operand buffers and calls `co_dispatch_kernel`. A backend/operation pair may
  provide `try_kernel_task` when resource admission can suspend; otherwise the
  coroutine invokes its ordinary `try_kernel` implementation directly. Shape
  and mdspan descriptors remain part of the awaited Tensor value, so runtime
  preparation occurs inside the coroutine.
  In both cases the epoch queues, not the scheduler, order the work. The CPU
  and ordinary backend leaves never choose a scheduler. Global `schedule()` and nested
  task routing may select different schedulers for different concrete task
  types. `AsyncTask` and `CudaTask` have distinct initial-admission interfaces
  and distinct promises over one common promise implementation; suspended work
  shares the promise-neutral `BasicTask` representation. An async
  wrapper may create a device-bound `CudaTask` with no scheduler route. Nested
  await inherits the parent's unified scheduler only after that scheduler
  accepts the CUDA route; explicit live-task scheduler migration remains a
  separate capability.
  See *Scheduler and Async integration* below and
  `../async/scheduler_migration.md`.
- Distributed and RPC execution are front-of-list backends that recurse, so they
  need zero core changes. The first pass can ship `[Blas, CpuGeneric]` with the
  seam being only "the list is a parameter."

## Scheduler and Async integration

This section records how kernel dispatch sits on top of the async runtime
(`src/uni20/async/`) — the part the worked `gemm` above leaves implicit.

### Three layers

Dispatch separates into three layers, and the bottom layer knows nothing about
the async runtime:

1. **Kernels** — plain functions over resolved mdspan-like views
   (`stdex::mdspan`, transform views). `cblas_dgemm`, a cuBLAS call on a given
   stream, a generic triple loop. CPU kernels are entirely async-unaware; they
   compute and return. A CUDA kernel takes a stream or handle but does not know
   about the async runtime.
2. **Async Tensor wrappers** — enroll operand buffers and schedule a coroutine
   that awaits stored Tensor values before entering coroutine-aware kernel
   dispatch. The first implementation provides fixed-output `gemm`, resizing
   `assign_product`, and the `beta = 1` `add_product` forwarding form in
   `src/uni20/linalg/async/`. Its static selector is resolved from the Tensor
   and storage types before scheduling; mdspan resolution and the runtime
   backend walk occur after the await. Synchronization lives here, in the
   epoch-queue awaits; the kernel never sees it.
3. **Backend selection** — the `backend_list` walk for an operation tag
   (`kernel_accepts_types` / `try_kernel`) and, for a block tensor, the planner
   that fans out per block. A CPU backend's `try_kernel` calls a layer-1 kernel
   directly and synchronously; a CUDA backend consumes an already-selected
   device context and leased resources. Layer 2 performs scheduler routing and
   sits above this walk. Static selector resolution precedes scheduling, but
   whole-value `Async<Tensor>` dispatches after awaiting. A future
   descriptor-synchronous block planner may perform more of the runtime choice
   before scheduling.

The consequence: one layer-1 kernel is reused by every storage mode; only the
wrapper changes.

### Metadata synchronous, data async

Block contraction planning should read only metadata — extents, strides, device
id, and block structure — and that metadata should be available without
awaiting. Whole `Async<Tensor>` deliberately has a different boundary because
the descriptor is part of the stored value:

> Descriptor synchronous, bytes async. The kernel dispatch point is a
> resolved mdspan synthesized from a synchronous descriptor plus an awaited pointer.

This rule describes the intended `AsyncArray`/block-handle path, not the first
whole-value `Async<Tensor>` wrapper. That wrapper awaits the Tensor and then
uses its descriptor and bytes together through the synchronous operation.

Three things stay distinct:

- **Descriptor** — per-block dims/strides/coord; lives on the container; always
  synchronous.
- **Data handle** — the per-block async-like object; `co_await` it for ready data.
- **Resolved mdspan** — descriptor + awaited pointer; what every kernel sees.

### Future async-block concept

The first whole-value Tensor wrappers use exact `Async<T>` overloads. Once
`AsyncArray` exists, its element handles may justify a shared access concept:

```cpp
template <class H>
concept async_block = requires(H h) {
  { h.read()  };   // co_await -> readable Tensor/block value
  { h.write() };   // co_await -> mutable Tensor/block value
};
```

`AsyncArray<S>` is deliberately lightweight: one shared backing allocation, a
synchronous descriptor table, and a per-element epoch queue for independent
ordering. Its element handle models `async_block` and shares a common public
access shape with `Async<T>`; it does not allocate a full `Async` per block.
This access shape alone does not promise identical descriptor timing or justify
one generic operation overload. (For
code that genuinely wants an async tensor view, `make_async_alias(...)` owns a
local view descriptor, retains the backing storage, and reuses the exact queue
that orders access to those bytes.)

### BlockTensor: two async axes

"Async or not" is two independent axes:

- **Container/structure** — is the block structure (which blocks exist, their
  dims) known synchronously, or itself an async result? Wrapping a `BlockTensor`
  as `Async<BlockTensor>` makes the structure async (e.g. a truncation output).
- **Per-block data** — each block's data is immediate, async (host), or async
  (device). This is the async facet of the container's `BlockTensorStorage`
  policy (`../symmetry/block_tensor.md` §5): a resolved mdspan itself has no async-ness, so
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
template <class Storage>                    // BlockTensorStorage policy (../symmetry/block_tensor.md §5)
class BlockTensor {
    BlockSpace             legs_[Rank];     // structure — synchronous
    std::vector<Coord>     coords_;         // which blocks exist — synchronous
    block_store_t<Storage> blocks_;         // immediate views | AsyncArray | AsyncArray<device>
  public:
    size_t          num_blocks()         const;          // sync
    BlockDescriptor descriptor(size_t i) const;          // sync: dims/strides/coord
    auto            block(size_t i);                      // -> async_block (or resolved mdspan)
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
factors — `../symmetry/block_tensor.md` §7. So `contributing()` yields
`(ai, bi, op_a, op_b, alpha)` records, not bare block indices.)

**Who selects the per-block backend.** Ownership sits with the block's storage
type, which the container's `BlockTensorStorage` policy determines — but the
*mechanism* is the kernel's choice. A generic linear-algebra op typically
re-enters the ordinary `backend_list` walk per block (or per coalesced group),
keeping its runtime fallbacks. A specialized planner — an optimized R/A/B/C
Hamiltonian-apply (`../tensor_network/rabc_contraction_scheduling.md`), say — may instead bind a
whole list of CUDA sub-kernels at plan time, potentially as a CUDA graph, and
never consult a walk per block. Both are legitimate shapes of layer 3. The one
constraint for this descriptor-synchronous block path is the order of
decisions: **dispatch happens before scheduling**. By the time one of these
block coroutines is scheduled, dispatch has already succeeded, so it is a thin
wrapper around the chosen backend's raw kernel call plus its buffer awaits.
This does not apply to whole-value `Async<Tensor>` wrappers, which await the
Tensor before entering ordinary synchronous dispatch.

Scheduler routing is explicit async-wrapper behavior. Distinct task types may
select different initial schedulers, while the numerical backend remains
unaware of that choice:

```cpp
// host-async block kernel
schedule([](auto c_, auto a_, auto b_) static -> AsyncTask {         // -> CPU scheduler
  auto C_ = co_await c_;  auto A_ = co_await a_;  auto B_ = co_await b_;
  gemm_kernel(C_, A_, B_);                                          // layer-1 kernel
}(c.block(r).write(), a.block(ai).read(), b.block(bi).read()));

// Device block kernel: select a device before device-sensitive work.
schedule([](auto c_, auto a_, auto b_, cuda::DeviceResources* resources) static -> CudaTask {
  auto C_ = co_await c_;  auto A_ = co_await a_;  auto B_ = co_await b_;
  co_await cuda::set_device(resources->device());
  auto execution = co_await cublas::acquire_execution(cublas::execution_pool(*resources));
  cublas_gemm(execution, C_, A_, B_);                                // no suspension in leaf call
}(c.block(r).write(), a.block(ai).read(), b.block(bi).read(), &device_resources));
```

The nested routing shown by the CUDA snippet is implemented: `CudaTask` has a
CUDA-specific initial-admission interface and promise, and an `AsyncTask` can
`co_await` a CUDA child and resume on its own scheduler. Optional affinity lives
in `CudaTaskPromise`; the scheduler supplies a default activation device while
it is empty. The same routing now supports async Tensor GEMM: the task returned
by `CublasBackend::try_kernel_task` binds operand affinity, awaits cuBLAS
execution resources, and invokes the prepared provider leaf before returning
to its host parent.

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

### Scheduler routing around non-suspending dispatch

CUDA leaf dispatch remains an ordinary non-coroutine call. Lightweight CUDA
submission and host-intensive provider calls initially run on one logical
scheduler per CUDA device. A oneTBB arena supplies concurrency slots rather
than fixed workers; an arena observer establishes the device on every
participant. Streams, provider handles, and workspaces are device-local leased
resources rather than worker-owned state.

Typed initial admission or nested `co_await` routes a `CudaTask` to a scheduler
that accepts its CUDA domain and device. A scheduler-unbound child may inherit a
compatible unified scheduler from its parent. The CUDA task may suspend while
waiting for a composite resource request. Once admitted, its prepared provider
leaf runs without suspension, records and publishes completion state, and
releases resources according to their provider-specific contract. A separate
provider lane is a later profiling-driven option.

Blocking and fully synchronized CUDA execution remain useful API/debug adapters
over the same leaf kernels; they are not separate numerical backends. See
`../backends/cuda/kernel_dispatch.md`.

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
3. **Scheduler routing.** Type-directed scheduling and heterogeneous nested
   coroutine entry/return require defined activation ownership,
   wait/quiescence, cancellation, and lifetime semantics. Same-task-type
   migration is a separate extension. The provider job itself remains
   non-suspending, and deterministic resource acquisition under
   `DebugScheduler` remains required for differential tests.
4. **The oracle for device and distributed storage.** Every list must end in a
   runnable correctness oracle. For device tensors that is a naive device kernel;
   for distributed tensors a gather-to-one-MPI-rank-then-generic path is a candidate,
   useful mainly for tests.
5. **Coalesced and batched kernels.** A coalesced single-axis GEMM group should
   be expressible as one operation tag / backend capability emitting one task,
   per `../symmetry/block_coalescing.md`.
6. **Testing the oracle.** Guaranteeing the generic path stays exercised even
   when an optimized backend is always eligible — the forced-fallback override
   above is the mechanism, per `backend_dispatch.md`.
7. **`AsyncArray` queue representation.** Striping per-block epoch queues over the
   shared backing cheaply enough for thousands of small blocks, while keeping
   independent per-block ordering and a public interface common with `Async<T>`.
   This is where the small-block tail cost lands on CPU. This is the same
   problem as the hazard-granularity question in `../symmetry/block_tensor.md` §10, seen
   from the queue side: a *coalesced* operand (`../symmetry/block_coalescing.md`) spans many
   elements of one backing buffer, so either it awaits every element queue it
   covers (and writers do, symmetrically) or there is a buffer-level queue the
   element queues nest under. Per-element queues alone do not cover the dual
   block-view / wide-view access that coalescing requires.
8. **CUDA blocking adapters.** Which blocking and default-stream debug adapters
   remain maintained once provider scheduling is implemented. They should wrap
   the same leaf backends rather than appear as sibling numerical backends.
9. **MPI user-facing default.** This note presents the collective SPMD
   `MpiBackend` as the base, with RPC as an optional front-of-list layer;
   `../backends/mpi/persistent_dispatch.md` proposes the root-controller /
   worker-runtime model as the *default* user-facing mode, with expert SPMD
   remaining possible. The two are mechanism vs. default policy and can
   coexist, but which mode ordinary user code gets by default is an open
   decision that both notes should record the same way.
