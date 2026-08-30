# CUDA/cuSOLVER Architecture Notes

**Status:** the first native Tensor backend is implemented. It provides a
blocking real `float`/`double` exact SVD for tall column-major CUDA matrices and
a device-resident BlockTensor sector bridge used by the first CUDA two-site
DMRG sweep path.

The generic host-execution and dispatch contract is defined in
[CUDA Kernel Dispatch and Device Scheduling](kernel_dispatch.md). This note
keeps the cuSOLVER-specific consequences.

## Primary References

- NVIDIA cuSOLVER documentation:
  [Thread Safety](https://docs.nvidia.com/cuda/archive/12.9.0/cusolver/index.html#thread-safety),
  [Parallelism with Streams](https://docs.nvidia.com/cuda/archive/12.9.0/cusolver/index.html#parallelism-with-streams),
  and
  [cusolverDnSetStream](https://docs.nvidia.com/cuda/archive/12.9.0/cusolver/index.html#cusolverdnsetstream).
- NVIDIA CUDA Programming Guide:
  [CUDA Streams](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-streams)
  and the surrounding asynchronous execution section.

The key interpretation for uni20 is:

- CUDA streams are ordered work queues.  Work enqueued into one stream executes
  in stream order; multiple streams can expose independent work to the runtime.
- cuSOLVER is thread-safe, so independent host scheduler threads may use
  independent cuSOLVER handles.
- A cuSOLVER handle has an associated stream.  That stream association is part
  of the execution context for the solver call.
- If a uni20 operation immediately copies results to host and synchronizes, the
  operation is effectively blocking at the API boundary even if cuSOLVER queued
  device work asynchronously internally.
- Synchronous host/device `cudaMemcpy` calls are blocking; asynchronous copies
  still become blocking if the API boundary immediately waits for the stream.

## Resource Model

Uni20 has one canonical `DeviceResources` set per enrolled CUDA device. It owns
the actually-idle stream pool and lazily constructs a bounded cuSOLVER handle
pool:

```text
CUDA device context
  -> device scheduler arena
  -> actually-idle stream pool
  -> exclusive cuSOLVER handle pool
  -> per-operation workspace and completion resources
```

The handle pool belongs to the device rather than to scheduler threads. Host
submission concurrency, stream capacity, and cuSOLVER handle capacity are
separate tunables. The scheduler must retain the capability to use multiple
host submitters for a device, but one is the default for general fine-grained
CUDA work. The initial target for cuSOLVER is two handles per device, independent
of that default submitter count. The canonical pool now implements that default,
capped by stream capacity. `cusolver::execution_pool(resources, count)` may set a
different count before the first provider use.

A solver request acquires an exclusive cuSOLVER handle first and then an
actually-idle stream. The first blocking backend allocates workspace with
`cudaMallocAsync`, binds the handle to the leased stream, calls the provider,
copies `devInfo` to host, and synchronizes that stream before returning. Keeping
handles independent from streams lets a handle rotate onto whichever pool
stream becomes idle next.

Do not create and destroy cuSOLVER handles per operation. Handle creation is
runtime state setup, not useful numerical work. Handles are device-local leased
resources, not permanent properties of particular oneTBB worker threads.

An exclusive handle lease may move between physical workers in the same device
arena because the arena observer establishes the same CUDA device on every
participant. Different operations must not mutate one handle concurrently.
Provider-specific state that remains live after host-call return stays retained
until the relevant completion boundary.

The operation may rotate through the bounded stream pool. Workspace and
completion tracking follow the submitted operation and selected stream, not the
handle alone.

Multiple handle leases allow independent solver work with host-side concurrency.
A cuSOLVER call such as SVD may occupy one device-scheduler participant while
launching and coordinating many device kernels, even though device work may
remain pending after the API returns. Scheduler concurrency and handle-pool
capacity therefore provide separate bounds.

Multiple cuSOLVER leases are already useful at the blocking boundary. A
scheduler batch can run independent symmetry sectors from multiple application
or oneTBB worker threads; each participant blocks only on its own provider call
and operation stream. A future asynchronous backend can retain the same pool
while publishing GPU completion tokens instead of synchronizing in the call.

Raw cuSOLVER handles remain internal to RAII leases and backend adapters. An
internal coroutine may await ordered resource admission, but the actual
backend walk and cuSOLVER call are ordinary non-coroutine code. No further
`co_await` occurs while the provider call is in progress.

The first implementation runs cuSOLVER directly from the active scheduler batch
participant. A separate provider scheduler or lane is justified only if
profiling shows that host-intensive calls starve lightweight submission or
unrelated device continuations.

## Host Blocking Boundaries

The main numerical cuSOLVER calls should be treated as potentially asynchronous
device-work submissions.  They can still perform nontrivial CPU-side setup, and
the amount of host work is implementation- and routine-dependent, so profiling
with Nsight Systems remains the authority for performance decisions.

Known host-side boundaries are:

- handle creation and destruction;
- workspace-size queries and other metadata/setup calls;
- `cusolverDnSetStream`, which mutates the handle state for the next call;
- synchronous `cudaMemcpy` between host and device;
- explicit synchronization such as `cudaStreamSynchronize`,
  `cudaDeviceSynchronize`, or `cudaEventSynchronize`;
- copying device outputs such as `devInfo`, singular values, or factor data back
  to host and then waiting for those copies.

For example, `cusolverDnDgesvd` writes `devInfo` on the device.  A wrapper that
needs `devInfo` on the CPU before returning must enqueue a D2H copy and wait, so
the wrapper is host-blocking even if the solver call itself was submitted
asynchronously.

### Host API return versus solver completion

Calling a cuSOLVER routine and completing the solver operation are different
boundaries. NVIDIA states that cuSOLVER keeps execution asynchronous as much as
possible, but it does not promise that every provider call is a negligible or
strictly nonblocking host operation. A call can perform substantial host-side
planning, launch many kernels, or contain implementation-specific
synchronization before it returns. The selected stream remains the authoritative
completion boundary for the device outputs.

This permits one host submission lane to maintain multiple in-flight solver
operations when the provider calls return before their device work completes:

```text
host lane
  -> acquire handle A and stream A
  -> submit sector A SVD; retain its resources until stream A completes
  -> acquire handle B and stream B
  -> submit sector B SVD; retain its resources until stream B completes
  -> ...
  -> wait for or compose all sector completions
```

The provider calls are serial on that host lane, but work already submitted to
different streams may execute concurrently on the GPU. Each in-flight operation
needs an exclusive handle because binding a stream mutates handle state. It also
needs retained workspace, input/output access state, device `devInfo`, and any
host destination used for an asynchronous result copy.

The current exact SVD wrapper does not use this model. It enqueues the
factorization, copies `devInfo` asynchronously into a stack variable, and
immediately synchronizes the stream. The stack destination and immediate error
check therefore make the whole wrapper host-blocking. An asynchronous wrapper
would instead retain a pinned host `devInfo` destination in the pending
operation, record or publish the stream completion, and check the value only
after completion. Singular values needed for global truncation can use the same
submit-all-then-join structure.

One host lane is not an unconditional design requirement. If profiling shows
that a particular solver routine spends enough time inside the host API call to
underfeed the GPU, a small provider-specific submission lane may be useful. That
limit must remain separate from general CUDA block-operation submission:
allowing every fine-grained GEMM, copy, event, and callback operation to use the
same host parallelism creates driver contention without increasing stream
capacity.

### Asynchronous SVD implementation consequences

Removing the explicit stream synchronization is simple at the provider call,
but the retained operation must replace several assumptions currently supplied
by that synchronization:

- `devInfo` cannot be copied into a stack variable. A pending SVD must retain a
  pinned host destination, or retain device status storage, until its completion
  is collected.
- The input/output buffer accesses must publish one shared operation-tail
  completion instead of using `release_after_synchronization()`. The existing
  CUDA buffer completion machinery already supports this.
- The workspace and device-status allocations use stream-ordered allocation and
  can enqueue their frees after the solver work. They do not require the host to
  retain their C++ wrappers until device completion.
- Owning CUDA tensor temporaries may leave scope once every access guard has
  published its completion. `CudaBuffer` transfers those completions to its
  shared allocation and queues `cudaFreeAsync` on the device-local reclamation
  stream, so ordinary tensor destruction does not restore host blocking.
- The cuSOLVER handle must remain unavailable until the operation stream reaches
  the conservative handle-return boundary. The existing execution-lease tail
  callback already provides that behavior.
- Provider submission errors remain immediate and terminal after work has been
  enqueued. Numerical failure reported through `devInfo` is deferred and must be
  checked when the pending result is collected.

The current block-SVD path has an additional ordering constraint. Each sector
function immediately copies its singular values to host after the SVD. That copy
would wait for the same sector and preserve serialization even if the provider
wrapper returned early. The useful one-submitter structure is therefore:

```text
phase 1: for each sector in cost order
  assemble device matrix
  submit SVD on a leased handle and stream
  enqueue pinned host copies of devInfo and compact singular values
  retain a pending sector result

phase 2: for each pending sector
  wait for or compose completion
  validate devInfo
  publish host singular values and device-resident factors
```

With a two-handle pool, the third submission may wait for either earlier handle
to return. That is intentional admission control: one host lane can keep at most
two solver operations in flight without creating unbounded workspace pressure.
If any collection reports failure, all already-submitted sector operations must
still be drained before the batch propagates the error. Performance measurement
must also attribute completion rather than the return of the submission call;
the current lightweight-batch item timer would otherwise report only enqueue
time as sector SVD time.

## SVD Boundary

The exact Tensor backend currently accepts four mutable CUDA-buffer mdspecs:
the destructive input matrix, singular values, left vectors, and right-adjoint
vectors. It supports real `float` and `double`, reduced or full vectors, tall
column-major matrices, valid nonzero buffer offsets, and one CUDA device across
all operands. Unsupported shapes or layouts decline before resource admission.
Complex scalars, wide matrices, row-major lowering, singular-values-only forms,
and an asynchronous Tensor entry point remain future work.

The BlockTensor bridge keeps bulk factorization data resident:

```text
packed CUDA BlockTensor with one logical buffer per block
  -> assemble one dense matrix per charge sector on CUDA
  -> transpose wide sectors on-device when required
  -> run independent cuSOLVER SVDs from a scheduler batch
  -> retain U, S, and Vt in CUDA tensors
  -> copy only compact singular values to host for global selection
  -> gather selected factors directly into CUDA BlockTensor storage
```

Each packed CUDA block retains an independent completion ledger even though all
blocks share one physical allocation. This permits concurrent block transfers
and multiple blocking cuSOLVER calls from different scheduler participants.
Measured and unmeasured BlockTensor entry points share this same bridge; the
measured form attributes the per-sector scheduler batch to its caller-supplied
event. Repartitioned centers obtain device placement from their retained block
descriptors rather than requiring access to an owning storage object. The
current bridge requires one dense axis on each side of the SVD boundary and one
CUDA device across all blocks. A full-vector sector whose charge occurs on only
one boundary has no provider factorization to perform; the bridge constructs
its side-specific identity null-space factor directly on CUDA.

## Scheduler Boundary

The CUDA/cuSOLVER layer should not become a second dependency DAG. CPU-side
epochs establish causal readiness: a GPU operation is submitted only once its
metadata and object lifetime are valid. The current GPU runtime lowers this
through scoped `buffer.read_synchronized_with(stream)` /
`buffer.write_synchronized_with(stream)` guards and
retained writer/reader completions, without reconstructing an independent epoch
graph.

This keeps the split clear:

- async epoch model: task ordering, object lifetime, and high-level dependency
  causality;
- unified scheduler: execution of `CudaTask` activations in the arena for their
  effective device, selected from explicit affinity or the scheduler default,
  with bounded host participation;
- GPU storage/runtime: buffer access completions, stream/event synchronization,
  resource leasing, and device work submission;
- cuSOLVER backend: a non-suspending solver call that consumes leased resources
  and either publishes completion state or synchronizes at a documented
  blocking boundary.

The current exact SVD entry point is ordinary blocking code rather than a
`CudaTask`. It acquires resources inside the selected backend, executes without
operation-tag redispatch, and returns after stream synchronization. Future
coroutine entry points may await resource admission, but the provider call
itself remains an ordinary non-coroutine leaf.

Temporary cuSOLVER workspace follows the device allocation capability. Devices
with stream-ordered memory-pool support use `cudaMallocAsync` and
`cudaFreeAsync` on the leased execution stream. Other devices use the blocking
`cudaMalloc` and `cudaFree` fallback; the exact SVD boundary already
synchronizes the execution stream before those allocations are released.

## Open Questions

- Whether profiling eventually justifies a separate bounded cuSOLVER execution
  lane rather than direct execution on the device scheduler.
- Which routine families return early enough for one host lane to keep multiple
  streams supplied.
- Whether a provider-specific retained handle can be safely reused after host
  API return; the conservative implementation retains it until stream completion.
- Whether block-sparse SVD should use one cuSOLVER call per sector, batched
  Jacobi SVD for many small sectors, or a hybrid policy.
- How much cuSOLVER workspace should be cached per handle or operation class,
  and how workspace pressure should interact with the broader CUDA allocator.
- How MPI client-server scheduling should assign symmetry-sector SVD blocks to
  remote CUDA workers.
