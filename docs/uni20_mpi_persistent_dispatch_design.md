# Uni20 MPI Persistent Object Store and Kernel Dispatch Design

## Status

Design note / planning document.

This document records a proposed direction for Uni20 MPI support, persistent immutable object storage, and kernel dispatch. It is not an implementation specification yet. Some names are provisional.

The main design premise is:

> Uni20 should make ordinary user code MPI-capable through library/runtime dispatch, not by requiring users to write explicit SPMD rank guards around every operation.

The proposed default MPI mode is therefore a root-controller / worker-runtime model. Expert SPMD mode should remain possible, but it should not be the default user-facing programming model.

## Goals

- Let rank 0 execute normal user control flow.
- Let worker ranks run a Uni20 runtime service loop.
- Let Uni20 operations on MPI-backed tensors become distributed backend operations.
- Avoid requiring users to manually write `if (rank == 0)` guards and collective broadcasts for ordinary serial-looking code.
- Support persistent immutable objects by global ID.
- Allow persistent objects to be freely copied, cached, loaded from disk, fetched over MPI, or served by storage ranks.
- Keep `Async<T>` and persistent storage semantically separate.
- Provide explicit async-safe materialization through `co_load()`.
- Provide explicit persistence bridges from async values: non-destructive copy and destructive take.
- Support Linux/ELF dynamic registration of MPI worker handlers through `dladdr()` / `dlsym()`.
- Keep kernel dispatch compatible with future Python bindings and runtime descriptor dispatch.

## Non-goals

- Do not make arbitrary C++ code transparently distributed.
- Do not require all user code to be written SPMD-style.
- Do not make the persistent layer a raw CUDA allocation store.
- Do not make `persistent_ptr<T>` dereference perform hidden I/O.
- Do not solve general recovery from disk, MPI, or persistent-store failure in the first version.
- Do not make mutable distributed objects the default tensor-data model.
- Do not rely on C++ providing a magic “run code for every template instantiation” mechanism.

## Terminology

### Root rank / controller rank

The rank that executes user code in default MPI mode. Initially this is usually rank 0.

The root rank:
- Runs the user program.
- Constructs distributed operation descriptors.
- Submits commands to worker ranks.
- Makes non-reproducible global decisions.
- Publishes manifests and decision records.

The root rank should not be the main tensor payload bottleneck.

### Worker ranks

Ranks that do not execute arbitrary user control flow in default mode.

Worker ranks:
- Enter a Uni20 worker loop after initialization.
- Receive command envelopes from root.
- Resolve or register worker handlers.
- Execute distributed operation handlers.
- Materialize persistent objects as needed.
- Produce and persist output objects.
- Report completion and output IDs/manifests.

### Storage ranks

Optional ranks dedicated to serving immutable persistent objects.

Storage ranks are object servers, not algorithm masters.

### `Async<T>`

A causal, mutable epoch channel in Uni20’s async runtime.

An `Async<T>` object is not itself persistent. A value produced at a specific completed epoch may be persisted.

### `persistent_ptr<T>`

A small typed handle naming an immutable committed value.

A `persistent_ptr<T>`:
- Is serializable by global object ID.
- May be sent through MPI cheaply.
- Does not imply local residency.
- Does not allow mutation.
- Should not provide `operator*` or `operator->` if those could load.

### `pvalue_ptr<T>`

A resident immutable materialization of a persistent object.

Proposed implementation:

```cpp
template<class T>
using pvalue_ptr = std::shared_ptr<T const>;
```

The pointer should pin a local cache entry or resident materialization while it exists. Destroying the last `pvalue_ptr<T>` releases the pin and returns the resident object to cache policy.

Alternative names considered:
- `loaded_ptr<T>`
- `resident_ptr<T>`
- `persistent_value_ptr<T>`
- plain `std::shared_ptr<T const>`

Current preference: keep `pvalue_ptr<T>` as the public alias or vocabulary type if continuity with the Matrix Product Toolkit model is useful.

### `MpiStorage<LocalStorage>`

A distributed tensor storage type.

Examples:

```cpp
Tensor<double, MpiStorage<Cpu>> A;
Tensor<double, MpiStorage<Gpu>> B;
```

`MpiStorage` is distribution. `Cpu` / `Gpu` are local placement/execution backends. In a later naming pass this may become something like:

```cpp
DistributedStorage<Cpu, MpiBackend>
DistributedStorage<Gpu, MpiBackend>
```

## High-level architecture

Default MPI mode:

```text
rank 0:
  uni20::mpi::init()
  execute user_main()
  library operations on MpiStorage tensors submit distributed commands

worker ranks:
  uni20::mpi::init()
  enter uni20::mpi::worker_loop()
  execute registered worker handlers on demand
```

Conceptually:

```cpp
int main(int argc, char** argv) {
    uni20::mpi::init(argc, argv);

    if (uni20::mpi::is_worker_rank()) {
        return uni20::mpi::worker_loop();
    }

    return user_main(argc, argv); // root rank only
}
```

Ordinary user code:

```cpp
using MpiTensor = Tensor<double, MpiStorage<Cpu>>;

MpiTensor A, B, C;
// initialize A and B
contract(A, B, C);
```

Expected internal flow:

```text
contract(A, B, C)
  -> kernel dispatcher sees MpiStorage<Cpu>
  -> dispatches to mpi_backend::contract(A, B, C)
  -> root builds ContractDescriptorV1
  -> root broadcasts/submits command envelope
  -> workers receive descriptor
  -> workers ensure worker handler is registered
  -> workers execute mpi_contract_worker
  -> workers materialize inputs, compute local blocks, persist outputs
  -> root installs output manifest into C
```

## Persistent immutable object model

### Core invariant

```text
A persistent object ID names an immutable committed value.
It does not name a mutable slot.
It does not name an Async<T> channel.
```

Once committed, the object value must not change. Updates produce new persistent object IDs.

This is the semantic foundation that makes MPI replication easy.

### Replication

Persistent objects may be copied freely:

```text
rank 2 loads object X
rank 5 loads object X
rank 9 loads object X
all receive the same immutable logical value
```

There is no lock, no migration, no read invalidation, and no coherence protocol for read-only objects.

The remaining problems are:
- object identity,
- serialization,
- materialization,
- cache policy,
- garbage collection,
- storage failure handling.

These are much easier than mutable distributed coherence.

### Loading

Use `load`, not `lock`.

`lock()` implies mutual exclusion. No mutual exclusion is involved. The operation materializes an immutable value and pins it resident.

Proposed API:

```cpp
persistent_ptr<T> p = ...;

std::optional<pvalue_ptr<T>> local = p.try_load_local();
pvalue_ptr<T> a = p.load_wait();
pvalue_ptr<T> b = co_await p.co_load();
```

`co_load()` should:
- complete synchronously on a local cache hit,
- suspend on disk/MPI/storage-rank materialization,
- return a `pvalue_ptr<T>` that keeps the value resident.

`persistent_ptr<T>::operator*` and `persistent_ptr<T>::operator->` should not perform I/O. They should probably not exist.

### Cache residency

`pvalue_ptr<T>` pins a resident cache entry.

Destroying the final `pvalue_ptr<T>` should:
- release the pin,
- return the entry to cache policy,
- not perform blocking I/O,
- not call `co_await`.

The cache may later:
- retain the entry,
- evict it,
- spill it,
- write it to disk,
- keep it in memory for reuse,
- destroy the resident materialization.

A likely implementation is an aliasing `shared_ptr`:

```cpp
struct cache_entry {
    object_id id;
    T value;
    // cache metadata, pin state, placement, store backpointer, etc.
};

std::shared_ptr<cache_entry> entry = cache.lookup(id);
pvalue_ptr<T> value(entry, &entry->value);
```

This avoids a custom deleter on each `T const*`, though a custom deleter is also viable if it only releases pins or enqueues background work.

### Construction / commit

A persistent object can be created from an ordinary value:

```cpp
persistent_ptr<T> p = co_await store.co_commit(std::move(t));
```

After commit succeeds:
- the store owns an immutable value,
- a `persistent_ptr<T>` is returned,
- the object is servable by ID.

A global ID must not escape until the object is committed and servable.

### `persist_copy`

Non-destructive persistence from an async epoch.

```cpp
persistent_ptr<T> p = co_await store.persist_copy(reader);
```

Semantics:
- read the current completed value,
- copy / clone / serialize it into the persistent store,
- leave the source `Async<T>` value intact,
- return a persistent ID after commit.

This may be expensive, but it is semantically simple.

### `persist_take`

Destructive persistence from an async epoch.

```cpp
persistent_ptr<T> p = co_await store.persist_take(writer);
```

Semantics:
- acquire destructive access to one completed value,
- move the `T` out of the async storage,
- leave the async storage unconstructed / consumed / failed according to the specific bridge contract,
- commit the moved value into the persistent store,
- return a persistent ID after commit.

This operation may avoid copy. It should use move construction or store-specific direct construction, not move assignment.

The persistent store probably cannot directly reuse the `Async<T>` control block because the async storage control block and persistent-cache control block are different structures. Therefore the first version should explicitly move the `T`.

### Failure policy

For `persist_take`, recovery is not a goal.

If disk, MPI, or persistent-store commit fails:
- no `persistent_ptr<T>` is returned,
- no persistent ID may be published,
- the operation logs a fatal persistent-store error,
- the async branch fails or the process/communicator aborts.

This is a fail-stop HPC model. Silent corruption and dangling IDs are not acceptable.

### Durability

The first implementation may distinguish:
- committed and servable by the persistent store,
- durably flushed to disk.

A checkpoint operation may require an explicit durability barrier:

```cpp
co_await store.flush(p);
co_await store.flush_all();
```

But a `persistent_ptr<T>` must not be published until the store can serve it somehow.

## Relationship between persistent storage and `Async<T>`

`Async<T>` and persistent storage are logically separate.

```text
Async<T>:
  causal mutable epoch channel

persistent_ptr<T>:
  immutable persistent value identity
```

The bridge is:

```text
Async<T> epoch -> persist_copy -> persistent_ptr<T>
Async<T> epoch -> persist_take -> persistent_ptr<T>
persistent_ptr<T> -> co_load -> pvalue_ptr<T>
```

A persistent ID attaches to a value at an epoch, not to the `Async<T>` object.

Do not expose a long-lived invariant like:

```text
this Async<T> has persistent ID X
```

Use:

```text
the value produced at epoch n has persistent ID X
```

## Tensor and CUDA boundary

The persistent store should store CPU-level logical objects.

Examples:
- `Tensor`
- `A_Matrix`
- tensor block objects
- wavefunction manifests
- AD checkpoint records
- metadata objects

It should not store raw CUDA allocations as persistent objects.

A `Tensor` may internally manage:
- host memory,
- pinned memory,
- device buffers,
- lazy host/device materializations,
- storage handles.

Serialization and persistence of tensor payloads should be delegated to tensor storage internals.

## Wavefunction structural sharing

The motivating Matrix Product Toolkit model:

```cpp
std::vector<pheap_ptr<A_Matrix>>
```

Generalized Uni20 shape:

```cpp
std::vector<persistent_ptr<A_Matrix>>
```

or a manifest:

```cpp
struct WavefunctionManifest {
    std::vector<persistent_ptr<A_Matrix>> sites;
    QuantumNumberMetadata qn;
    LayoutMetadata layout;
};
```

A DMRG update can produce:

```text
old: [A0, A1, A2, A3, A4]
new: [A0, A1, B2, B3, A4]
```

Only changed tensors get new persistent IDs. Checkpoints become small manifests of object IDs.

## MPI workflow model

### Default: root-controller / worker-runtime

Default Uni20 MPI mode should be client-server-like at the runtime boundary.

Root:
- runs user code,
- submits Uni20 distributed operations,
- makes global decisions,
- owns manifests and decision records.

Workers:
- execute registered handlers,
- compute local pieces,
- materialize/persist objects,
- participate in collectives under library control.

This is not the same as a naive master bottleneck. Tensor payloads should not flow through root by default.

### Expert: SPMD

SPMD should remain available for expert users and internal distributed kernels.

But ordinary user code should not need to write:

```cpp
if (rank == 0) { ... }
MPI_Bcast(...);
```

for every local-looking operation.

### Storage ranks

Storage ranks can use a true RPC/server model:

```text
get(object_id)
put(object_id, payload)
pin(object_id)
release(object_id)
has(object_id)
```

Storage ranks serve immutable objects. They are not algorithm masters.

## Kernel dispatch mechanism

The initial idea of selecting kernels purely by C++ overloads or simple storage tags is too inflexible.

Uni20 needs an explicit kernel dispatch mechanism.

A kernel needs metadata:
- debug name,
- wire/RPC name,
- descriptor schema,
- supported dtypes,
- supported storage/execution backends,
- local worker handler,
- MPI marshalling logic,
- DAG visualization hooks,
- Python binding entry point.

Storage type is an input to dispatch, not the whole dispatch system.

Example dispatch tree:

```text
contract(A, B, C)
  -> collect argument metadata
  -> resolve storage/execution contexts
  -> choose backend operation
  -> build descriptor / DAG node
  -> execute locally or submit distributed command
```

For `MpiStorage<Cpu>`:

```text
contract(A, B, C)
  -> mpi_backend::contract(A, B, C)
```

## MPI command granularity

The RPC command should identify a coarse operation family, not every local template instantiation.

Good RPC command names:
- `uni20.contract.v1`
- `uni20.svd.v1`
- `uni20.eigh.v1`
- `uni20.truncate.v1`
- `uni20.apply_mpo.v1`
- `uni20.dmrg_two_site_update.v1`

Avoid registering every combination of:
- scalar type,
- storage type,
- layout,
- symmetry,
- contraction pattern,
- backend tuning option.

Those details belong in the descriptor and local worker dispatch.

## Descriptor-driven dispatch

Example descriptor:

```cpp
struct ContractDescriptorV1 {
    OpId op_id;
    DType dtype;
    LocalBackend local_backend;

    TensorManifestRef A;
    TensorManifestRef B;
    TensorOutputRef C;

    ContractionPattern pattern;
    LayoutDescriptor layout;
    DistributionDescriptor distribution;
    SymmetryDescriptor symmetry;
    OptionsBlob options;
};
```

Descriptor data should contain:
- object IDs,
- manifests,
- layout metadata,
- options,
- small scalar values,
- distribution plans.

It must not contain:
- raw C++ pointers,
- references into rank-0 memory,
- lambda closures,
- compiler-dependent type names as semantic data.

## Command identity

Separate these concepts:

```text
debug_name:
  human-readable DAG/log name, e.g. "contract"

wire_name:
  versioned operation name, e.g. "uni20.contract.v1"

command_id:
  stable hash or numeric ID derived from the wire name

descriptor schema:
  versioned serialization format, e.g. ContractDescriptorV1

registration_symbol:
  Linux/ELF symbol used to locate a registration function on worker ranks
```

The registration symbol is a code locator, not the semantic ABI.

## Linux/ELF dynamic registrar model

### Motivation

In default client-server MPI mode, only root executes user code. Worker ranks enter a service loop. Therefore registration side effects that happen only when root calls a templated function will not automatically occur on workers.

A Linux/ELF solution:

1. Root instantiates or selects a kernel registration function.
2. Root obtains that function’s symbol name with `dladdr()`.
3. Root sends the symbol name with the command envelope.
4. Worker calls `dlsym()` on its own process image.
5. Worker calls the registration function.
6. Registration function installs deserializer and worker handler in the local registry.
7. Worker dispatches the command.

This registers only what is actually used at runtime, without requiring a pre-registered matrix of every possible template combination.

### Assumptions

This mode assumes:
- Linux/ELF.
- MPI ranks run the same executable, or at least the same relevant shared libraries.
- The registration function symbol exists in each rank.
- The symbol is visible to `dlsym()`.
- Root can recover an exact symbol with `dladdr()`.
- The registration function has the expected signature.
- Missing symbols or mismatched versions are fatal.

C++ mangled names are acceptable for this internal same-binary mechanism. `extern "C"` is not semantically required, though it may be useful for plugin ABIs or diagnostics.

### Registrar function

A registration function should:
- be a real function, not inlined away,
- have default visibility,
- be retained by the linker,
- register an expected command ID,
- install a descriptor deserializer and static worker handler.

Example shape:

```cpp
template<class KernelSpec>
struct mpi_kernel_registration {
    static void registrar(Registry& r)
        __attribute__((visibility("default")))
        __attribute__((used))
        __attribute__((noinline));

    static void registrar(Registry& r) {
        KernelSpec::register_into(r);
    }

    static std::string symbol_name() {
        void* ptr = reinterpret_cast<void*>(&registrar);

        Dl_info info{};
        if (!dladdr(ptr, &info) || !info.dli_sname) {
            throw registration_error("dladdr failed for registrar");
        }

        if (info.dli_saddr != ptr) {
            throw registration_error(
                "dladdr returned nearest symbol, not exact registrar symbol"
            );
        }

        return info.dli_sname;
    }
};
```

The attribute syntax above is illustrative. Exact portable wrappers should be defined in one Uni20 platform header.

### Build requirements

The executable or shared library must export registrar symbols.

Likely requirements:
- `-rdynamic` or `-Wl,--export-dynamic` for main-executable symbols.
- Default symbol visibility for registrar functions.
- `used` / equivalent attribute to prevent dead stripping.
- `noinline` to ensure a real symbol address.
- Careful handling of LTO and identical-code folding.
- Optional explicit export lists for production builds.

If symbols live in a plugin/shared library:
- root may need to send library identity plus symbol,
- workers may need to `dlopen()` the same plugin,
- paths may differ across cluster nodes, so raw path identity is fragile.

First version can require the same executable and relevant DSOs on all ranks.

### Command envelope

```cpp
struct CommandEnvelope {
    OpId op_id;

    CommandId command_id;
    std::string wire_name;
    std::string debug_name;

    std::string registration_symbol;

    std::vector<std::byte> descriptor_bytes;
};
```

Worker logic:

```cpp
void handle_command(CommandEnvelope const& env) {
    if (!registry.contains(env.command_id)) {
        void* sym = dlsym(RTLD_DEFAULT, env.registration_symbol.c_str());
        if (!sym) {
            fatal_with_dlerror(env);
        }

        using registration_fn = void (*)(Registry&);
        auto fn = reinterpret_cast<registration_fn>(sym);
        fn(registry);

        if (!registry.contains(env.command_id)) {
            fatal("registrar did not install expected command");
        }
    }

    registry.lookup(env.command_id).run(ctx, env.descriptor_bytes);
}
```

### Important distinction

This scheme does not enumerate every template specialization in the program.

It gives:

```text
automatic lazy registration for each kernel specialization actually submitted
to the MPI backend at runtime
```

That is the useful subset.

## Worker handlers

Registered worker handlers should be static or free functions that take owned parameters.

Preferred:

```cpp
static task<void> mpi_contract_worker(
    WorkerContext ctx,
    ContractDescriptorV1 desc
);
```

Avoid captured coroutine lambdas for worker tasks.

Descriptor and context lifetimes must be explicit. No worker handler should hold references into temporary descriptor buffers or root-rank memory.

## Runtime type dispatch and Python compatibility

The MPI descriptor layer should be compatible with Python bindings.

Python cannot instantiate arbitrary C++ templates at runtime. Therefore the MPI front door should support runtime descriptors and variants.

Example runtime metadata:
- dtype enum,
- local backend enum,
- layout descriptor,
- distribution descriptor,
- symmetry descriptor,
- options blob.

The worker handler can dispatch internally:

```cpp
switch (desc.dtype) {
case DType::f64:
    return mpi_contract_typed<double>(ctx, desc);
case DType::c128:
    return mpi_contract_typed<uni20::complex<double>>(ctx, desc);
}
```

Then:

```cpp
switch (desc.local_backend) {
case LocalBackend::Cpu:
    return contract_blocks<double, CpuBackend>(ctx, desc);
case LocalBackend::Cuda:
    return contract_blocks<double, CudaBackend>(ctx, desc);
}
```

The RPC registry remains small. Local templated kernels remain available where performance matters.

## Operand compatibility

Default rule:

```text
If a main tensor operand is MpiStorage<X>, all main tensor operands must have
compatible distributed storage unless the operation explicitly permits otherwise.
```

Valid:

```cpp
Tensor<double, MpiStorage<Cpu>> A, B, C;
contract(A, B, C);
```

Suspicious / rejected by default:

```cpp
Tensor<double, MpiStorage<Cpu>> A, C;
Tensor<double, Cpu> B;
contract(A, B, C);
```

Make mixed behavior explicit:

```cpp
contract(A, replicate(B), C);
```

or:

```cpp
auto Bm = distribute(B, A.layout());
contract(A, Bm, C);
```

Do not silently broadcast a large root-local tensor.

## Output tensors

For MPI storage, the output tensor on root is a manifest/handle, not rank-0-owned bytes.

`C` should contain:
- distributed context,
- layout epoch,
- block ownership map,
- persistent output object IDs,
- async readiness/failure state.

Workers produce output blocks, persist them, and report IDs or manifest updates. Root installs the resulting manifest into `C`.

## Async integration

MPI operations must integrate with Uni20 async causality.

A distributed operation should:
- acquire logical read epochs for inputs,
- acquire logical write epoch for outputs,
- allocate an `op_id`,
- submit a command envelope,
- complete or fail output async state when the distributed operation completes or fails.

Message arrival order and worker scheduler timing are not the legality model.

The distributed analogue of Uni20 local causality is:

```text
DistributedOpQueue / op_id order
+ tensor read/write dependencies
+ explicit descriptor submission
+ explicit completion/failure
```

## Error handling

Initial policy is fail-stop.

Fatal conditions:
- missing registrar symbol,
- `dladdr()` exact-symbol failure,
- `dlsym()` failure,
- registrar fails to install expected command,
- descriptor schema mismatch,
- unsupported dtype/backend/layout combination,
- MPI transport error,
- persistent-store I/O error,
- storage-rank fetch failure.

Diagnostics should include:
- rank,
- op_id,
- debug name,
- wire name,
- command ID,
- registration symbol,
- `dlerror()` if applicable,
- Uni20 ABI/schema version,
- object ID or manifest ID if relevant.

## Invariants

1. `persistent_ptr<T>` names an immutable committed value.
2. `pvalue_ptr<T>` pins a resident immutable materialization.
3. `co_load()` is explicit and awaitable; dereference of `persistent_ptr<T>` must not hide I/O.
4. `persist_copy` is non-destructive.
5. `persist_take` is destructive and may move the value out of an async epoch.
6. No persistent ID may escape before the object is committed and servable.
7. `Async<T>` is a causal epoch channel; persistence attaches to a value at an epoch, not to the channel.
8. Default MPI mode runs user code on root and worker loops elsewhere.
9. Distributed tensor operations are submitted by the MPI backend as command descriptors.
10. RPC commands identify coarse operation families, not every local template instantiation.
11. Registration symbols are code locators only; command IDs and descriptor schemas define the operation ABI.
12. Worker handlers must own their descriptors and avoid unsafe captured coroutine state.
13. Mixed distributed/local operands require explicit operation support.
14. Persistent objects may be freely replicated because they are immutable.
15. Storage failure, MPI failure, or kernel registration failure is initially fatal.

## Open questions

- Exact public names: `persistent_ptr`, `pvalue_ptr`, `persistent_store`, `co_load`, `persist_copy`, `persist_take`.
- Whether `pvalue_ptr<T>` should be a type alias or a thin wrapper around `std::shared_ptr<T const>`.
- Exact descriptor serialization format.
- Whether `command_id` is a hash, integer enum, UUID, or build-generated ID.
- How to handle plugins/shared libraries across cluster nodes.
- How to represent operation availability and worker capability negotiation.
- Whether workers should lazily register on first use or root should preflight all required registrations.
- How storage ranks integrate with the same command loop or use a separate protocol.
- What part of tensor layout/symmetry should be descriptor runtime data versus template specialization.
- How much of the Python-facing dynamic dispatch layer should be shared with the MPI descriptor layer.
- Whether durability is immediate commit or explicit `flush()` / checkpoint barrier.
- Garbage collection strategy for persistent object stores and checkpoint manifests.

## Suggested implementation stages

### Stage 1: Documentation and mock descriptors

- Add this design note.
- Define terminology and invariants.
- Create mock descriptor structs for a small number of operations.
- No MPI execution yet.

### Stage 2: Local registry and descriptor dispatch

- Implement an in-process registry.
- Register `contract.v1` handler manually.
- Serialize/deserialize descriptors locally.
- Run a local worker handler through the registry.

### Stage 3: Linux dynamic registrar prototype

- Implement `dladdr()` symbol discovery for registrar functions.
- Implement `dlsym()` on the same process.
- Validate exact-symbol recovery.
- Add fatal diagnostics for missing or non-exact symbols.

### Stage 4: MPI worker loop prototype

- Root sends command envelope.
- Workers resolve registrar symbol.
- Workers run no-op or simple descriptor handler.
- Root gathers completion.

### Stage 5: Persistent store integration

- Implement `persistent_ptr<T>` and `pvalue_ptr<T>` for local disk-backed object store.
- Implement `co_load()` with cache-hit fast path.
- Implement `co_commit`, `persist_copy`, and `persist_take`.

### Stage 6: First real distributed tensor operation

- Implement `contract` over `Tensor<double, MpiStorage<Cpu>>`.
- Use persistent IDs / manifests for inputs and outputs.
- Validate root user code does not require rank guards.

### Stage 7: Storage ranks and GPU backend

- Add storage-rank serving by object ID.
- Add `MpiStorage<Gpu>` once local GPU tensor storage is mature enough.
- Keep persistent store at CPU-level logical object boundary.

## Summary

The proposed model is:

```text
root user program
+ worker runtime loops
+ storage-type dispatch to MPI backend
+ descriptor-driven distributed operations
+ Linux/ELF lazy registrar discovery via dladdr/dlsym
+ persistent immutable object IDs
+ explicit async co_load materialization
+ copy/take persistence bridges from Async<T> epochs
```

This keeps user-facing code mostly ordinary:

```cpp
Tensor<double, MpiStorage<Cpu>> A, B, C;
contract(A, B, C);
```

while moving MPI complexity into the Uni20 dispatch/runtime layer.

The central semantic choice is to prefer immutable persistent values and manifests over mutable migratable distributed objects. That makes replicated read-only tensor data cheap, safe, and natural for tensor-network workflows.
