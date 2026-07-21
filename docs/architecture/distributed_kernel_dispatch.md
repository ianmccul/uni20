# Distributed kernel dispatch: exploratory notes

> **Status:** speculative and non-normative.
>
> This document does not describe an implemented MPI, RPC, or distributed backend. It
> records constraints and possible layering for future design work. The current local
> kernel-dispatch contract is defined by
> [`kernel_dispatch.md`](kernel_dispatch.md).

## Purpose

Distributed execution raises questions that are larger than choosing a local BLAS,
LAPACK, CPU, or device kernel. It may require:

- global shape and ownership reasoning;
- communication planning;
- collective participation;
- rank- or worker-consistent failure behavior;
- placement-aware scheduling;
- local kernel selection after data reaches its execution domain.

The useful idea from local dispatch is composability: a distributed plan may invoke the
ordinary local dispatcher for each local leaf. It does not follow that MPI or RPC
should simply be inserted as another backend in every local `backend_list`.

## Constraints that already apply

Any future distributed design must preserve the existing Uni20 invariants.

### Communication is an explicit cost

Communication, replication, redistribution, and materialization must be represented by
the operation or plan. They must not appear as invisible fallback from a local kernel.

### Symmetry metadata is mathematical state

Distributed planning must preserve quantum-number, block-space, local-space, and
leg-orientation metadata. A distributed symmetry-aware path may not silently project
the operation to one dense matrix.

Logical block identifiers and placement metadata must survive plan construction and
local execution.

### Async legality comes from causality

Rank arrival order, network timing, or scheduler timing cannot establish legality.
Dependencies must be represented by epochs, tasks, or another explicit causal model.

### Selection and execution have separate commitment boundaries

A distributed selection protocol may communicate in order to reach one consistent
decision. For example, it may use a dedicated collective agreement to accept or reject
a candidate. That protocol is not a rank-local `KernelAttempt`: every participant must
remain in the selection phase and observe the same outcome. Failure or disagreement
inside the agreement is a terminal plan-construction error, not permission for one rank
to continue its own backend walk.

The selected plan becomes committed no later than the first of:

- successful collective plan agreement;
- distribution of an immutable command or plan descriptor;
- algorithm-specific communication;
- globally visible ownership or output-placement changes;
- local or remote work submission.

After commitment, failure is terminal for the distributed operation. Falling back to a
different distributed algorithm or local backend would otherwise risk duplicate
communication or partial output.

### Local accessor semantics remain binding

A distributed plan eventually hands local descriptors to a local backend. Those
descriptors retain their mdspan accessor semantics. Distribution does not make an
opaque device buffer host-readable or erase a transform accessor.

## Possible layering

A plausible architecture separates global planning from local kernel selection:

```text
distributed Tensor operation
    -> validate global structure and ownership
    -> choose or build a distributed plan
    -> schedule communication and placement tasks
    -> invoke local dispatch on resolved local operands
    -> publish distributed completion
```

The local leaf remains the existing contract:

```text
local Tensor/mdspan descriptors
    -> dispatch_kernel or co_dispatch_kernel
    -> local provider/reference backend
```

This keeps local backend lists focused on implementations that can legally operate on
the same local arguments.

## Distributed planner versus distributed backend

Two broad designs are possible.

### Planner above local dispatch

A distributed operation wrapper chooses a plan before invoking local kernels.

Advantages:

- communication and placement are explicit;
- collective safety can be validated at plan level;
- local backend lists remain unchanged;
- global structure and symmetry metadata remain available;
- different local storage domains may choose different local selectors.

Risks:

- more operation-specific planning interfaces;
- the planner must expose enough structure for diagnostics and testing;
- plan caching and invalidation need a clear contract.

This is the safer default direction.

### Relationship to the persistent MPI proposal

[`../backends/mpi/persistent_dispatch.md`](../backends/mpi/persistent_dispatch.md)
develops one concrete form of the planner-above-dispatch direction: a root controller
validates the distributed operation, chooses or builds an immutable command descriptor,
and submits it to worker runtimes.

In that model, workers do not independently walk distributed candidates. They execute
the command selected by the controller. A worker may still invoke ordinary local kernel
dispatch for a resolved local work item, but a local provider failure after command
submission fails the distributed operation; it does not reopen distributed selection.

The collective-acceptance discussion below mainly constrains an expert SPMD mode or a
future collectively planned operation. It is not required for every decision in the
root-controller model.

### Distributed entry in a backend selector

A distributed backend could participate in an outer dispatch walk and, after accepting
the operation, recursively invoke local dispatch.

Advantages:

- one selection vocabulary for local and distributed alternatives;
- a distributed algorithm could cleanly decline before communication.

Risks:

- the same `KernelAttempt` vocabulary may be too small for collective planning;
- distinguishing local descriptors from globally distributed values becomes critical;
- recursion can obscure where placement and communication costs are introduced;
- a decline on one rank but acceptance on another can deadlock;
- fallback after partial collective participation is invalid.

This design should be considered only with a precise distributed argument type and a
collectively consistent acceptance protocol.

## Collective acceptance

A local `kernel_accepts_types` result is compile-time and therefore consistent for the
same program and argument types. Runtime acceptance is more difficult.

A distributed candidate must not allow:

```text
rank 0: accept and enter collective
rank 1: decline and try another candidate
```

Possible approaches include:

1. Restrict runtime distributed acceptance to properties guaranteed identical on all
   participants.
2. Perform a dedicated collective selection protocol before any
   algorithm-specific communication.
3. Build a globally validated plan on one authority and distribute an immutable plan.
4. Treat disagreement as a terminal plan-construction error rather than fallback.

A collectively agreed rejection may continue to the next distributed candidate. A
successful agreement to accept commits the selected plan. A selection-protocol failure
or disagreement is a terminal plan-construction error. None of these outcomes is an
ordinary rank-local backend decline.

## Placement and ownership

A distributed value needs an explicit model for:

- global logical indices or blocks;
- owning rank or worker;
- local storage kind and device;
- replicated versus uniquely owned data;
- output placement;
- migration and redistribution.

These properties are not mdspan layout. An mdspan describes one resolved local view;
it does not describe the global ownership graph.

A future distributed operation should lower global values to local work items carrying
both logical identity and placement. Local kernel dispatch should see only the resolved
local descriptors needed for that work item.

## Async integration

Distributed execution is naturally asynchronous, but the design must still distinguish
three levels:

1. **Epoch dependencies:** which logical values may be read or written.
2. **Distributed plan dependencies:** communication and placement order.
3. **Local kernel tasks:** CPU, device, or provider execution.

A possible task graph is:

```text
await input epochs
    -> validate/build distributed plan
    -> schedule receives, sends, or collectives
    -> await local operand readiness
    -> co_dispatch_kernel(local selector, local operation, ...)
    -> publish local completion
    -> publish distributed output epoch
```

Raw threads and scheduler timing are not substitutes for these dependencies.

## Error and cancellation model

Before implementation, the design needs explicit answers for:

- how one participant reports a terminal failure to all others;
- whether outstanding communication can be cancelled;
- how output epochs become failed consistently;
- whether a failed collective leaves the communicator usable;
- which failures are recoverable at a higher operation boundary;
- how diagnostics identify rank, worker, device, and local backend attempts.

A local `KernelDispatchError` may be one component of a distributed diagnostic, but it
is not sufficient by itself.

## Diagnostics

Useful distributed diagnostics would likely include:

- operation and distributed algorithm name;
- participant set or communicator identity;
- global shape and block metadata;
- placement decision;
- communication phase;
- local selector and backend attempts per failed work item;
- first terminal failure and propagated failures.

Diagnostics must not require extra collectives that change execution semantics unless
diagnostic collection is itself explicitly enabled and safe.

## Testing requirements for a future implementation

A distributed implementation would need deterministic focused tests for:

### Planning

- ownership and placement;
- empty and degenerate global shapes;
- symmetry-preserving block assignment;
- plan agreement across participants;
- explicit redistribution.

### Fallback and commitment

- clean decline before communication;
- no fallback after successful plan agreement or algorithm-specific communication;
- no duplicate local work after terminal failure;
- consistent output failure propagation.

### Local composition

- host local kernels;
- device local kernels;
- different local selector choices on different placements where legal;
- accessor-preserving local lowering.

### Failure behavior

- one participant fails before commitment;
- one participant fails after commitment;
- provider failure inside a local task;
- communication failure;
- scheduler shutdown or cancellation.

### Diagnostics

- rank/worker attribution;
- global versus local failure distinction;
- ordered local backend attempts;
- bounded diagnostic collection.

## Open design questions

The following questions are intentionally unresolved:

1. What is the canonical distributed value and placement type?
2. Is plan construction centralized, collective, or hierarchical?
3. Which operations require global planning rather than block-local scheduling?
4. How are communicators or RPC execution contexts represented and owned?
5. What is the commitment point for each communication primitive?
6. Can any distributed decline remain purely local?
7. How are distributed tasks integrated with `Async<T>` epochs?
8. How are output placement and redistribution costs exposed?
9. How is symmetry-aware block placement represented?
10. What failure semantics are guaranteed after a collective has started?
11. Should distributed algorithms use an outer selector, a planner registry, or
    operation-specific policy?
12. How are plan caches keyed and invalidated?

## Promotion criteria

Material from this document should move into normative subsystem documentation only
after the repository contains:

- an agreed distributed value and placement contract;
- concrete operation APIs;
- an implemented planning and execution path;
- focused tests for collective consistency and commitment;
- documented failure and cancellation behavior.

Until then, examples here are architectural sketches rather than promised APIs or
roadmap commitments.
