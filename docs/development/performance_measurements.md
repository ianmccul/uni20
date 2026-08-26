# Performance Measurements

Uni20 performance measurements are explicit, type-selected instrumentation.
Ordinary operations do not consult a global profiler, a thread-local sink, an
environment variable, or a nullable callback in their hot paths.

The shared vocabulary is defined in
[`performance_measurements.hpp`](../../src/uni20/common/performance_measurements.hpp).

## Measurement Levels

`MeasurementLevel` defines three compile-time levels:

| Level | Cost model | Intended use |
|---|---|---|
| `none` | Direct invocation only | Ordinary algorithms and benchmarks |
| `coarse` | Two host clock reads per measured phase | Operation and phase attribution |
| `detailed` | Coarse timing plus per-item storage and clock reads | Load balance, overlap, and scheduler-tail investigations |

`NoMeasurements` is the disabled policy. Instrumented algorithms select it with
`if constexpr`, so an optimized ordinary instantiation contains no clock read,
counter, atomic operation, virtual call, nullable-recorder branch, or profiling
state in a hot object.

`DurationMeasurements<Event, EventCount>` records count, total, minimum,
maximum, and mean inclusive host wall duration for a fixed event enum. It is not
thread-safe: coarse phase events are recorded by the calling thread.

`DetailedMeasurements<Event, EventCount>` additionally retains individual
`BatchMeasurement` records. Detailed item functions write disjoint slots, and
the calling thread aggregates those slots after the synchronous batch joins.
The measured item path therefore needs no profiling atomic, but it does perform
two clock reads and ordinary slot writes per item.

## Algorithm Integration

An algorithm with an ordinary and an instrumented overload follows this shape:

```cpp
template <class Measurements>
  requires performance::DurationMeasurementPolicy<Measurements, event>
auto operation(arguments..., Measurements& measurements)
{
  return performance::measure_duration(
      measurements,
      event::phase,
      [&] { return implementation(arguments...); });
}

auto operation(arguments...)
{
  performance::NoMeasurements measurements;
  return operation(arguments..., measurements);
}
```

Do not replace the policy with a recorder pointer in the ordinary overload. A
null check still changes the hot path and makes the disabled cost depend on
branch prediction and compiler interprocedural analysis.

Measurement collectors are separate from mathematical result objects. This
keeps profiling lifetime, storage, and reporting out of tensor and solver
semantics. A caller creates a collector, passes it explicitly, and reports or
resets it after the operation.

## Synchronous Batch Measurements

`measure_batch()` accepts the item count, a synchronous batch executor, and an
indexed item function. Its levels behave as follows:

- `none` passes the original function directly to the executor;
- `coarse` measures only the executor's inclusive wall duration;
- `detailed` wraps each item, retains disjoint start/finish slots, aggregates
  after join, records the batch, and rethrows any executor exception.

One detailed batch record contains:

- requested, started, and completed item counts;
- inclusive batch wall duration;
- summed and maximum completed-item durations;
- first-item start delay and item-start spread;
- item-finish spread and return-after-last-finish duration;
- peak overlap of measured item intervals.

The executor contract must guarantee that each started index is invoked at most
once and that no item remains active after return. Uni20's lightweight scheduler
batch interface already provides that synchronous join boundary.

These measurements deliberately perturb fine-grained work. Use them to locate
imbalance or scheduler overhead, then repeat the final benchmark with
`NoMeasurements`.

## Clock And Execution Domain

`WallClock` is `std::chrono::steady_clock`; durations are host wall durations in
nanoseconds. A host duration around a CUDA or other asynchronous provider call
measures submission unless the operation synchronizes before returning. It must
not be presented as device execution time.

Future CUDA timing should use CUDA events retained by the appropriate lease or
task frame. MPI timing may need communicator-aware aggregation. Both can use the
same compile-time policy principle while supplying domain-specific records.

## Two-Site DMRG

`TwoSiteDmrgPerformanceMeasurements` records inclusive run, sweep, bond, center
construction, local eigensolver, individual effective-Hamiltonian application,
block SVD, state selection, selected-factor materialization, and
environment-update phases. The detailed alias
`DetailedTwoSiteDmrgPerformanceMeasurements` also records Krylov vector
allocation, vector-update, and reduction calls, plus every per-charge block-SVD
batch:

```cpp
DetailedTwoSiteDmrgPerformanceMeasurements measurements;

auto result = run_two_site_dmrg(
    mps,
    mpo,
    environments,
    options,
    measurements,
    ParallelPackedSparseBlockStorage<>{});
```

The phase hierarchy is inclusive: run contains sweeps, sweeps contain bond
updates, and bond updates contain their component phases. Do not sum parent and
child durations as if they were exclusive.

The current detailed checkpoint groups Krylov copy, AXPY, scale, and zeroing as
vector updates. It does not yet time each contraction group or selected-factor
construction item. Those require explicit lower-level event propagation. They
must not be approximated by adding clocks to every generic tensor operation in
ordinary builds.

The registered spin-half Heisenberg DMRG example exposes these policies as
`--measurements=off`, `--measurements=coarse`, and
`--measurements=detailed`. The switch selects separate template instantiations
before entering the DMRG run.
