# DMRG Performance Baselines

**Status:** developmental CPU baseline recorded on 2026-08-26.

This document records measurements used to guide the first symmetry-aware
finite-DMRG implementation. They are regression and design evidence, not
performance claims. In particular, timings from different programs or papers
must not be compared as if they used identical models, arithmetic, convergence
criteria, initial states, hardware, or update algorithms.

The durable comparison rule is:

> Record end-to-end cost together with the physical problem, symmetry,
> arithmetic, update algorithm, solver policy, truncation policy, and achieved
> solution quality.

The 2026
[DMRG software performance benchmarking study](https://arxiv.org/abs/2607.28369)
is used here as a source of concrete external workloads and result trajectories.
Its package comparisons and parameter experiments are not general user-tuning
advice: they establish how the tested implementations behave on those fixed
configurations. For Uni20, their value is as reproducible library-development
targets whose cost and achieved energy can be compared while tuning the same
workload.

## 1. Local Environment

The measurements below were made on the `exciton` development laptop:

| Component | Configuration |
|---|---|
| CPU | AMD Ryzen AI 7 350, 8 cores / 16 hardware threads, one NUMA node |
| Memory | 61 GiB |
| Compiler | GCC 15.2 |
| Build | CMake `Release`, LTO enabled |
| Source | Feature-branch revision that first introduces this document |
| Scheduler | oneTBB 2022.3 |
| BLAS/LAPACK | OpenBLAS 0.3.32 pthread build |
| Scalar | `double`, except where `uni20::complex<double>` is stated |
| Symmetry | Bosonic U(1) total spin projection |

Unless stated otherwise, nested dense-library threading was disabled:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

This is important. The measured parallel path schedules independent symmetry
blocks, so allowing every block-level task to create another BLAS team causes
oversubscription and obscures scheduler scaling.

The laptop remained an interactive machine. Tables based on one run should be
treated as orientation only; tables explicitly reporting medians or means use
the stated repeated trials.

## 2. Current Parallel Work

At this baseline, one directional sweep visits bonds serially. Within one bond:

```text
two-site center construction
    parallel over output symmetry blocks

Lanczos iterations
    serial across iterations
    effective-Hamiltonian apply
        parallel over output symmetry blocks
        serial over contributions to one output block
        serial dense BLAS within each contribution
    vector copy/axpy/scale/zero
        parallel over symmetry blocks
    norm and inner product
        parallel block partials, serial deterministic final reduction

block SVD
    parallel over charge sectors, estimated-expensive sectors first
    serial LAPACK call within each sector

selected-factor materialization
    serial construction and population of output blocks

environment update
    parallel over output symmetry blocks
    serial over contributions to one output block
```

Each parallel operation is currently a synchronous lightweight-task batch. A
batch completes before the next operation begins. Output-block ownership avoids
locks and reduction temporaries, but the largest output block determines the
tail when too little independent block work remains.

## 3. End-to-End Block-Thread Scaling

The original scaling tables in Sections 3 and 3.1 predate the DMRG-specific
fixed-step Lanczos solver. They are retained because they diagnose the former
generic-solver path. The current fixed-work checkpoint is in Section 3.3.

The first larger scaling fixture is the open spin-half Heisenberg chain with
`L=80`, a Néel product-state start, real arithmetic, U(1) symmetry, and the
current two-site growth trajectory. `m=512` uses nine directional traversals;
`m=1024` uses ten. oneTBB concurrency is selected with `--block-threads` while
OpenBLAS remains single-threaded.

These historical fixtures stop on the first traversal that can reach the
requested cap. They measure the complete rank-growth trajectory, not repeated
work at steady `m`. In particular, the `m=512` run spends almost the entire
fixture below 512 and first reaches that cap during its ninth traversal. Keep
these results as end-to-end growth baselines rather than relabelling them as
steady-dimension measurements.

The example's `--steady-sweeps=N` mode implements the complementary protocol.
It verifies the retained dimension after every alternating growth traversal,
excludes those traversals from the steady timer, and then measures `N`
additional traversals. For a CPU/CUDA comparison at `m=512`, use the same
optimized binary and request at least one measured traversal in each direction:

```bash
spin_half_heisenberg_dmrg_example \
    --execution=cpu --sites=100 --max-states=512 --steady-sweeps=2 \
    --local-matvecs=4 --block-threads=8

spin_half_heisenberg_dmrg_example \
    --execution=cuda --cuda-device=0 --cuda-streams=8 \
    --sites=100 --max-states=512 --steady-sweeps=2 \
    --local-matvecs=4 --block-threads=8
```

Set the linked BLAS implementation to one thread for both commands. CUDA
runtime setup and initial placement are outside the reported growth and steady
timers; each traversal timer includes an execution-domain synchronization.

Representative commands are:

```bash
spin_half_heisenberg_dmrg_example \
    --sites=80 --max-states=512 --max-sweeps=9 \
    --scalar=real --block-threads=8

spin_half_heisenberg_dmrg_example \
    --sites=80 --max-states=1024 --max-sweeps=10 \
    --scalar=real --block-threads=8
```

The initial one-run scaling series was:

| Participants | `m=512` time | `m=512` speedup | `m=1024` time | `m=1024` speedup |
|---:|---:|---:|---:|---:|
| 1 | 23.28 s | 1.00x | 81.27 s | 1.00x |
| 2 | 14.39 s | 1.62x | 47.07 s | 1.73x |
| 4 | 10.34 s | 2.25x | 32.26 s | 2.52x |
| 8 | 9.47 s | 2.46x | 28.05 s | 2.90x |
| 16 | 11.74 s | 1.98x | 32.50 s | 2.50x |

Larger blocks improve scheduler amortization, but eight participants remain the
best point on this eight-core CPU. The current run shows no benefit from using
both SMT threads. Scheduler, allocation, and memory-system pressure are likely
contributors, while the implementation exposes no additional work axis at 16
participants.

At the smaller `L=160`, `m=64`, seven-traversal fixture, three-run medians were
5.93 s, 4.81 s, 4.84 s, and 5.44 s for 1, 2, 4, and 8 participants. This is a
useful overhead fixture: block parallelism saturates early when the individual
dense blocks are small.

### 3.1 Effective serial fraction and counter evidence

A later profiled `L=80`, `m=512`, nine-traversal run pinned the process to the
eight physical cores. `perf stat` adds some overhead, so only values within this
table should be compared:

| Participants | Wall time | Speedup | Efficiency | Average occupied cores | Karp-Flatt fraction |
|---:|---:|---:|---:|---:|---:|
| 1 | 59.24 s | 1.00x | 100% | 1.00 | n/a |
| 2 | 34.53 s | 1.72x | 86% | 1.99 | 16.6% |
| 4 | 23.21 s | 2.55x | 64% | 3.96 | 18.9% |
| 8 | 20.84 s | 2.84x | 36% | 7.42 | 25.9% |

A least-squares Amdahl fit gives an apparent serial fraction of about 22%, but
it does not fit the curve well: it predicts 18.78 s at eight participants,
where the measured time is 20.84 s. More importantly, the Karp-Flatt fraction
increases with participant count. A literal fixed serial fraction should remain
approximately constant. The table therefore measures an *effective* serial
fraction that includes contention, imbalance, and parallel overhead.

Selected counter changes from one to eight participants were:

| Counter | 1 participant | 8 participants |
|---|---:|---:|
| Instructions per cycle | 2.85 | 1.57 |
| Retired instructions | 478 billion | 659 billion |
| Generic cache-event miss rate | 2.1% | 6.4% |
| AMD backend-memory-bound slots | 14% | 23% |
| Context switches | 7,458 | 671,487 |
| System CPU time | 1.13 s | 18.85 s |

Nearly every core remains occupied, so a large idle single-threaded region is
not the primary limitation. Scaling instead loses both locality and execution
efficiency: IPC falls by about 45%, cache misses rise, retired work grows by
about 38%, and the eight-participant run incurs exceptional context-switch and
system-time cost. The exposed laptop PMUs do not provide working UMC/data-fabric
bandwidth counters, so these measurements do not yet distinguish DRAM bandwidth
saturation from memory-latency and cache-locality effects. They do establish
that a fixed-fraction Amdahl model is insufficient and that scheduler/batch
overhead is also material.

The first library measurement checkpoint now records inclusive run, sweep,
bond, center-construction, local-eigensolver, effective-Hamiltonian-application,
block-SVD, state-selection, selected-factor-materialization, and
environment-update wall durations. Its
detailed mode also records per-charge SVD batch counts, requested/started/
completed items, summed and maximum item time, start and finish spread, peak
overlap, and return-after-last-finish time. The API and overhead contract are in
[Performance Measurements](../development/performance_measurements.md).

Detailed mode now also groups Krylov vector allocations, copy/AXPY/scale/zero
updates, and norm/inner-product reductions. Remaining lower-level checkpoints
are contraction-group and selected-factor-construction item timing. Those
records will further separate truly serial wall time from busy but inefficient
parallel execution.

### 3.2 Generic-solver phase attribution

This historical attribution used the generic convergence-seeking Hermitian
Lanczos solver before the DMRG-specific fixed-step solver was introduced. One
run used the same `L=80`, `m=512`, nine-traversal,
eight-participant fixture, single-threaded BLAS, physical CPUs `0-7`, and the
GCC 15 release build. This is one noisy diagnostic sample, not a replacement
for the median scaling table.

Coarse measurement reported:

| Inclusive phase | Wall time | Fraction of run |
|---|---:|---:|
| Complete run | 26.73 s | 100% |
| Local eigensolver | 25.34 s | 94.8% |
| Effective-Hamiltonian applications | 7.74 s | 29.0% |
| Remaining eigensolver work | 17.60 s | 65.8% |
| Block SVD | 0.77 s | 2.9% |
| Environment update | 0.27 s | 1.0% |
| Selected-factor materialization | 0.15 s | 0.6% |
| Center construction | 0.08 s | 0.3% |

The 26,507 effective-Hamiltonian calls consumed only about 31% of eigensolver
time. The inferred remainder includes Krylov vector allocation, updates,
reductions, orthogonalization control, and projected dense solves. It is not an
inferred serial fraction: the BlockTensor primitives may themselves execute
parallel batches.

A detailed run then measured the Krylov operation adapter directly:

| Eigensolver component | Calls | Inclusive time | Fraction of eigensolver |
|---|---:|---:|---:|
| Effective-Hamiltonian application | 26,507 | 7.87 s | 30.1% |
| Vector allocation | 48,767 | 1.82 s | 7.0% |
| Vector update: copy/AXPY/scale/zero | 739,632 | 7.71 s | 29.5% |
| Reduction: norm/inner product | 459,560 | 8.15 s | 31.2% |
| Remaining solver control/projected work | - | 0.60 s | 2.3% |

The detailed run took 27.51 s. Its millions of clock reads intentionally
perturb the fine-grained path, and the observed difference from the coarse run
is within the variability seen on this laptop. These numbers should therefore
be used for category attribution, not precise overhead claims.

The call counts exposed the more important result. Symmetric Lanczos used full
basis reorthogonalization with a DGKS-style optional second pass. Each
basis vector produces an ordered inner product and AXPY, and every BlockTensor
primitive is a separate synchronous operation. Vector updates and reductions
together account for about 58% of end-to-end wall time and create more than one
million fine-grained operation boundaries in this run. This is consistent with
the earlier context-switch and system-time growth.

For a general convergence-seeking BlockTensor eigensolver, the corresponding
high-leverage design remains a batched orthogonalization surface: compute
multiple basis inner products with fewer reductions, then apply a fused linear
combination/update over each output block. DMRG should not pay for that generic
solver machinery in the first place.

The detailed SVD record reinforces that ordering. Across 711 batches, 4,026
sector items supplied 1.90 s of summed item work in 0.73 s of batch wall time,
or about 2.60 effective concurrent items; peak measured overlap was eight. The
sum of return-after-last-finish tails was only 1.9 ms. Sector imbalance exists,
but scheduler return tail is not a material fraction of this fixture.

### 3.3 Fixed-step DMRG Lanczos

The DMRG path now uses a separate three-term Lanczos projection with four
effective-Hamiltonian applications by default. It does not test local Ritz
convergence, fully reorthogonalize, or restart. This encodes the finite-DMRG
work policy directly: advancing the sweep and improving the environments is
more valuable than tightly converging an intermediate local problem defined by
poor environments.

A coarse run of the same `L=80`, `m=512`, nine-traversal, eight-participant
fixture reported:

| Inclusive phase | Generic solver | Fixed four-step solver |
|---|---:|---:|
| Complete run | 26.73 s | 2.31 s |
| Local solver | 25.34 s | 0.98 s |
| Effective-Hamiltonian applications | 7.74 s, 26,507 calls | 0.59 s, 2,761 calls |
| Block SVD | 0.77 s | 0.83 s |
| Environment update | 0.27 s | 0.27 s |
| Terminal energy | -35.26523705466045 | -35.26523704495348 |

The wall-time ratio is about 11.6x on this diagnostic pair. The terminal
energies differ by about `9.7e-9`, as expected when comparing a tightly solved
local trajectory with fixed local work after the same number of traversals. A
tenth fixed-step traversal reached `-35.26523705414721` in 6.71 s total, within
about `5.1e-10` of the nine-traversal generic result while remaining roughly
four times faster. These are trajectory observations, not matched convergence
claims.

Detailed fixed-step measurement recorded 11,755 vector-update and 6,233
reduction calls, down from 739,632 and 459,560 respectively. Their combined
measured time fell from 15.86 s to 0.17 s. Block SVD is consequently about 36%
of the new nine-traversal run and becomes a materially larger optimization
target. The result also demonstrates why optimizing generic full
reorthogonalization would have addressed the wrong DMRG work policy.

One uninstrumented fixed-step scaling pass, pinned to the corresponding first
physical CPUs, gave:

| Participants | Wall time | Speedup | Terminal energy |
|---:|---:|---:|---:|
| 1 | 5.10 s | 1.00x | -35.26523704495348 |
| 2 | 3.66 s | 1.39x | -35.26523704495348 |
| 4 | 2.62 s | 1.95x | -35.26523704495348 |
| 8 | 2.40 s | 2.13x | -35.26523704495348 |

Removing hundreds of thousands of small parallel vector operations improves
absolute time dramatically but does not make the remaining workload strongly
scalable. Per-sector SVD, environment updates, factor materialization, and the
remaining short block batches now dominate a much shorter run. This table is a
single interactive-laptop pass and should be replaced by repeated medians when
the next parallel checkpoint is evaluated.

### 3.4 First resident CUDA steady-dimension orientation

A first uninstrumented resident-CUDA pass on `polaron` used the same GCC 13
Release executable for both domains, `L=100`, a Néel product state, real
arithmetic, four local matvecs, eight block participants, and two measured
traversals after the requested bond cap was observed. CPU BLAS was restricted
to one MKL thread. CUDA used Quadro GV100 device 0 and eight streams. Each value
below is the mean of the left-to-right and right-to-left steady traversals from
one run, not a repeated-run median:

| Maximum states | CPU steady traversal | CUDA steady traversal | CUDA / CPU |
|---:|---:|---:|---:|
| 128 | 0.225 s | 18.063 s | 80.27x |
| 256 | 0.956 s | 20.742 s | 21.70x |
| 512 | 2.492 s | 28.705 s | 11.52x |
| 1024 | 16.477 s | 38.133 s | 2.31x |
| 2048 | 96.569 s | 59.330 s | 0.61x |
| 4096 | 744.433 s | 143.790 s | 0.19x |

CPU and CUDA terminal energies and discarded weights agreed at every point to
the expected floating-point accuracy. These timings establish an optimization
baseline, not a general hardware-performance claim. In U(1), the global bond
dimension is divided among charge sectors, so `m=512` still produces dense
provider calls much smaller than 512-by-512. The weak CUDA growth from `m=128`
to `m=256`, compared with the CPU growth, is consistent with stream, lease,
provider, and kernel submission costs dominating these small blocks. On this
fixture the first observed crossover lies between `m=1024` and `m=2048`; at
`m=2048`, CUDA is about 1.63 times faster than CPU. The two `m=2048` CPU
traversals took 74.735 and 118.403 seconds, while the CUDA traversals took
56.702 and 61.958 seconds. The reported mean therefore covers a real
directional imbalance rather than repeated measurements of one direction.

At `m=4096`, CUDA is about 5.18 times faster than CPU. The CPU traversals took
649.913 and 838.952 seconds; the CUDA traversals took 122.874 and 164.707
seconds, giving direction-matched speedups of 5.29 and 5.09. Complete growth
plus steady measurement took 1,652.816 seconds on CPU and 548.402 seconds on
CUDA, a 3.01 times end-to-end speedup. Terminal energies differed by less than
`4e-13` between domains.

Peak observed device allocation during `m=4096` growth was about 7.0 GiB, well
below the GV100's 32 GiB capacity. The CPU process reached about 14.9 GB
resident memory; a point observation during a steady traversal reported CPU
use equivalent to roughly 3.8 occupied cores. Higher bond dimensions are
currently limited by benchmark duration before device memory.

A detailed `L=100` attribution pass on 2026-08-28 used the same steady-state
protocol. The top-level phase means and fractions were:

| Phase | `m=128` time | `m=128` fraction | `m=512` time | `m=512` fraction |
|---|---:|---:|---:|---:|
| Complete traversal | 17.146 s | 100.0% | 26.465 s | 100.0% |
| Center construction | 0.172 s | 1.0% | 0.189 s | 0.7% |
| Local eigensolver | 11.731 s | 68.4% | 14.075 s | 53.2% |
| Block SVD | 2.912 s | 17.0% | 8.402 s | 31.7% |
| Factor materialization | 1.058 s | 6.2% | 2.415 s | 9.1% |
| Environment update | 1.204 s | 7.0% | 1.326 s | 5.0% |

The local-eigensolver rows contain these inclusive subphases:

| Local-solver subphase | `m=128` time | Traversal fraction | `m=512` time | Traversal fraction |
|---|---:|---:|---:|---:|
| Effective-Hamiltonian application | 7.639 s | 44.5% | 9.385 s | 35.5% |
| Krylov vector update | 2.042 s | 11.9% | 2.323 s | 8.8% |
| Krylov reduction | 1.542 s | 9.0% | 1.793 s | 6.8% |

The sector SVDs already overlap: summed sector-item time versus batch wall time
was 33.211 versus 5.802 seconds at `m=128`, and 113.333 versus 16.772 seconds
at `m=512`. Those ratios correspond to about 5.7 and 6.8 effective concurrent
sector factorizations. Increasing host parallelism alone is therefore not the
next SVD optimization; reducing per-sector provider overhead or batching
compatible small sectors is more promising.

An Nsight Systems trace of the shorter `L=20`, `m=128` fixture used the required
`--cuda-event-trace=false` setting. Profiling increased complete runtime from
about 13.8 to 30.4 seconds, so its API durations are not timing baselines. Its
counts expose the execution granularity: 325,820 kernel launches, 520,590 CUDA
event creations, 520,614 event destructions, 540,175 event records, and 383,503
host-function submissions across growth and two steady traversals. The GPU
executed 379,950 kernels and memory operations with 1.492 seconds of summed
device activity. The 54,130 memory copies moved only 5.609 MB in total.

An unprofiled run reported 29.01 seconds user time and 24.34 seconds system time
over 14.62 seconds wall time, with 2,054,606 voluntary context switches. Active
one-second `nvidia-smi dmon` samples generally showed 10--25% SM utilization and
zero reported memory-engine utilization. Together these measurements identify
host orchestration, short provider calls, and fine-grained dependency tracking
as the small-dimension limit, rather than PCIe volume or device memory
bandwidth.

A stream-count orientation on the same short fixture first measured one pass at
each count. Three additional runs at two, four, and eight streams confirmed
that the two-stream result was not just one favorable sample:

| CUDA streams | Initial mean | Four-sample median where repeated |
|---:|---:|---:|
| 1 | 5.348 s | - |
| 2 | 2.164 s | 2.056 s |
| 4 | 2.241 s | 2.370 s |
| 8 | 2.385 s | 2.257 s |
| 16 | 2.395 s | - |

One stream is also structurally disadvantaged by the current resource lease:
the stream returns to the idle pool only after its tail completion, so the host
cannot pipeline the next independent submission on that sole stream. Two
streams restore useful provider overlap. More than two did not repay their
extra cross-stream dependency and orchestration cost on this small fixture,
although the `L=100` SVD measurements above demonstrate useful concurrency well
beyond two streams. Stream count should eventually follow the available work
in each phase rather than remain one global constant.

Packed CUDA BlockTensor storage subsequently gained allocation-wide linear
operations. Packed padding is initialized to zero, and whole-allocation
`set_zero`, scale, exact-layout copy, AXPY, inner product, and norm operations
preserve that invariant. A repeated two-stream run of the same short fixture
then gave steady-traversal means of 1.848, 1.667, and 1.698 seconds, with a
median of 1.698 seconds. This is 17.4% below the earlier two-stream median of
2.056 seconds. Terminal energy remained within
`7.1e-15` of the reference value.

A second Nsight trace confirms that the improvement comes from coarser
execution rather than a provider-algorithm change. Relative to the earlier
trace, GPU kernel and memory-operation instances fell from 379,950 to 287,962,
kernel launches from 325,820 to 252,660, host-function submissions from 383,503
to 310,355, and memory copies from 54,130 to 35,302. Most directly, cuBLAS norm
kernels fell from 22,576 to 1,664 because each packed vector now uses one norm
call instead of one call per stored block. Summed device activity fell from
1.492 to 1.061 seconds. The earlier trace used eight streams and the new trace
used two, and profiler overhead makes their API durations unsuitable as wall
time comparisons; operation counts are the relevant evidence here.

## 4. Cost-Ordered Effective-Hamiltonian Groups

The effective-Hamiltonian plan originally submitted output groups in canonical
block-key order. The revised plan estimates the two dense contraction costs of
every contribution, sums them per output block, and stores a descending-cost
permutation once at plan construction. Contributions within one output block
retain their deterministic order.

Two runs of each variant gave these means:

| Fixture | Canonical order, `auto_partitioner` | Cost order, `auto_partitioner` | Cost order, `simple_partitioner` |
|---|---:|---:|---:|
| `m=512`, 8 participants | 16.49 s | 16.41 s | 16.54 s |
| `m=512`, 16 participants | 19.61 s | 18.98 s | 19.00 s |
| `m=1024`, 8 participants | 51.16 s | 49.54 s | 49.24 s |

The raw trial pairs, in the same column order, were:

```text
m=512,  8 participants: (16.65, 16.33) (16.72, 16.11) (17.13, 15.96)
m=512, 16 participants: (19.71, 19.52) (18.96, 19.00) (19.12, 18.88)
m=1024, 8 participants: (50.35, 51.97) (49.71, 49.37) (48.89, 49.59)
```

Absolute values differ from the earlier scaling series because this was a later
interactive-laptop run. Only back-to-back variants within this table should be
compared. Cost ordering improved the largest comparison by about 3.2 percent.

oneTBB's default `auto_partitioner` may combine adjacent loop iterations into
larger chunks. A `simple_partitioner` with the compact integer range's grain
size of one makes every group independently stealable, but it did not produce a
repeatable improvement here. The retained design therefore uses cost ordering
with `auto_partitioner`. See oneTBB's
[chunking guidance](https://uxlfoundation.github.io/oneTBB/main/tbb_userguide/Controlling_Chunking_os.html).

A separate prototype replaced `parallel_for` with a manual atomic-index worker
loop. At `m=256`, the existing `parallel_for` took 5.75 s and 6.04 s with four
and eight participants; the manual loop took 5.94 s and 6.18 s. There is no
evidence for replacing the scheduler primitive.

Later CUDA host-worker profiling separated arena concurrency from CUDA
submission concurrency. `TbbScheduler` retains `auto_partitioner` for uncapped
CPU batches, while `TbbSchedulerBatchOptions::maximum_concurrency` can run a
bounded number of dynamic batch runners. The DMRG example exposes that cap as
`--cuda-submitters=N`; it is independent of `--cuda-streams=N` and bounded by
`--block-threads=N`. This prevents CUDA submission tuning from reducing the
arena capacity available to ordinary host scheduler work.

On the `L=20`, `m=128`, four-stream fixture with eight arena participants,
three runs at each submitter count gave these steady-traversal medians:

| CUDA submitters | Median steady traversal |
|---:|---:|
| 1 | 0.666 s |
| 2 | 0.819 s |
| 4 | 1.156 s |
| 8 | 1.516 s |

The same one-versus-two ordering remained at larger dimensions. Single runs at
`L=20`, `m=512` took 0.742 versus 0.928 seconds, and single runs at `L=100`,
`m=512` took 11.706 versus 13.032 seconds. One submitter can issue independent
operations onto successive streams; it does not restrict execution to one
stream.

Holding the submitter count at one then isolated stream capacity on the short
fixture:

| CUDA streams | Three-run median steady traversal |
|---:|---:|
| 1 | 1.397 s |
| 2 | 0.851 s |
| 4 | 0.558 s |
| 8 | 0.558 s |

The example therefore defaults to one CUDA submitter and four streams. At that
stream capacity, the provider defaults are four cuBLAS handles and two cuSOLVER
handles. `--cuda-cublas-handles` and `--cuda-cusolver-handles` may tune those
pool sizes independently. All four controls remain explicit because larger
provider operations and future batched kernels may shift the optimum.

After CUDA buffer destruction was changed to enqueue completion-dependent
`cudaFreeAsync` on a device-local reclamation stream, the same four-stream,
one-submitter short fixture produced steady-traversal means of 0.553, 0.517,
and 0.709 seconds. Their 0.553-second median is effectively unchanged from the
earlier 0.558-second checkpoint. Two `L=100`, `m=512` repeats averaged 12.752
and 11.778 seconds; the latter matches the earlier 11.706-second orientation
within ordinary run variation. Energies and discarded weights were unchanged.
Asynchronous reclamation therefore removes a host-blocking lifetime boundary
without a measurable standalone end-to-end effect while the cuSOLVER SVD
wrapper still synchronizes each operation.

### 4.1 Provider handle pools and the single-submitter default

A 2026-08-29 follow-up measured the implemented per-device cuBLAS and cuSOLVER
handle pools. The unprofiled release fixture used device 0 on polaron, one CUDA
submitter, four streams, eight block threads, single-threaded host BLAS, and two
steady traversals after growth. Every configuration was repeated three times.
No competing compute process was using either GV100. The NVIDIA kernel/userspace
driver version mismatch prevented `nvidia-smi` telemetry, so these measurements
do not claim controlled clocks or power state. The command shape was:

```bash
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
spin_half_heisenberg_dmrg_example \
    --execution=cuda --cuda-device=0 --cuda-streams=4 --cuda-submitters=1 \
    --sites=20 --max-states=128 --steady-sweeps=2 \
    --local-matvecs=4 --block-threads=8
```

The provider comparisons added `--cuda-cublas-handles=N` or
`--cuda-cusolver-handles=N`; the `m=512` comparison changed only
`--max-states`.

The `L=20`, `m=128` handle-count comparison was:

| Provider configuration | Median steady traversal | Three-run range | Relative to default |
|---|---:|---:|---:|
| cuBLAS 4, cuSOLVER 2 (default) | **0.544 s** | 0.529--0.553 s | 1.00x |
| cuBLAS 2, cuSOLVER 2 | 0.577 s | 0.576--0.597 s | 1.06x |
| cuBLAS 1, cuSOLVER 2 | 1.025 s | 1.013--1.029 s | 1.88x |
| cuBLAS 4, cuSOLVER 1 | 0.538 s | 0.536--0.540 s | 0.99x |
| cuBLAS 4, cuSOLVER 4 | 0.536 s | 0.534--0.538 s | 0.98x |

The cuBLAS result remained at `m=512`:

| cuBLAS handles | Median steady traversal | Three-run range | Relative to four handles |
|---:|---:|---:|---:|
| 4 | **0.657 s** | 0.645--0.663 s | 1.00x |
| 2 | 0.693 s | 0.684--0.710 s | 1.05x |
| 1 | 1.099 s | 1.079--1.113 s | 1.67x |

This demonstrates why provider capacity is independent of host submission
concurrency. One host lane can submit onto several handle/stream leases and keep
them in flight until their stream-tail callbacks return the handles. Restricting
cuBLAS to one handle serializes that useful device queueing even though there is
still only one submitting host thread. The default of one cuBLAS handle per
stream is therefore materially useful rather than merely excess capacity.

The cuSOLVER count is currently unobservable in steady runtime because each SVD
wrapper synchronizes its stream before returning. One submitter can consequently
use only one cuSOLVER lease at a time. The one-, two-, and four-handle results are
within ordinary run variation. Two handles remain a bounded default for the
planned asynchronous SVD boundary, where one submitter will be able to retain
several independent solver operations.

Repeating the submitter comparison with the new default pools gave:

| CUDA submitters | Median steady traversal | Three-run range | Relative to one submitter |
|---:|---:|---:|---:|
| 1 | **0.544 s** | 0.529--0.553 s | 1.00x |
| 2 | 0.760 s | 0.611--0.839 s | 1.40x |
| 4 | 1.126 s | 1.068--1.130 s | 2.07x |
| 8 | 1.423 s | 1.235--1.442 s | 2.61x |

All configurations reproduced the same final energy to approximately
`1e-14`. The default result is also consistent with the immediately preceding
0.553-second asynchronous-reclamation checkpoint. These results support the
implemented defaults: one CUDA submitter per device, cuBLAS handle capacity
equal to stream capacity, and two cuSOLVER handles.

### 4.2 Why multiple CUDA submitters regress

This is not a CUDA correctness limitation or a universal recommendation to use
one host thread per GPU. CUDA supports calls from multiple host threads, and
multithreaded submission can be useful when host-side providers block, when
different software producers cannot be cheaply serialized, or when separate
threads drive separate devices. NVIDIA also documents the corresponding cost:
the CPU wrapper time for a kernel launch includes driver mutex contention during
multithreaded launching. CUDA 11.4 specifically improved interthread locking for
parallel CUDA Graph launches, which further demonstrates that this is a known
implementation cost rather than a prohibited execution model.

The CUDA Runtime contract is deliberately broader: any API call may block or
synchronize because an internal resource is contended or unavailable, and the
specific behavior may change. A long launch call alone does not prove lock
contention. A sufficiently fast stream of small launches may instead fill the
GPU command queue and block until submission space becomes available. OS Runtime
lock traces, CUDA API attribution, GPU queue state, and unprofiled wall time must
therefore be considered together.

The Uni20 recommendation is narrower: default to one general CUDA submission
lane for the current fine-grained, single-context, single-device block workload
when that lane can already keep the available streams supplied. Add
provider-specific host concurrency only when measurements show that a blocking
or host-intensive provider call otherwise underfeeds the device.

Relevant NVIDIA references are:

- [CUDA API synchronization behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html),
  including blocking on contended or unavailable internal resources;
- [Understanding overhead and latency in Nsight Systems](https://developer.nvidia.com/blog/?p=20916),
  including driver mutex contention during multithreaded launches;
- [CUDA 11.4 graph-launch improvements](https://developer.nvidia.com/blog/discovering-new-features-in-cuda-11-4/),
  including interthread-lock contention in parallel graph launch;
- [CUDA asynchronous kernel launches](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html),
  which permit one host lane to enqueue work without waiting for device
  completion.

An Nsight Systems comparison on 2026-08-29 used the same `L=20`, `m=128`,
four-stream workload with one and two CUDA submitters. CUDA event tracing was
disabled as required. The profiler substantially inflates absolute runtime, so
the durations below diagnose where the relative regression occurs rather than
serving as benchmark timings. Both traces performed the same 199,138 kernel
launches.

| CUDA host API | One submitter | Two submitters | Ratio |
|---|---:|---:|---:|
| `cudaLaunchKernel` | 0.934 s | 1.705 s | 1.83x |
| `cudaLaunchHostFunc` | 0.613 s | 1.444 s | 2.36x |
| `cudaEventCreateWithFlags` | 0.122 s | 0.944 s | 7.74x |
| `cudaMemcpyAsync` | 0.147 s | 0.313 s | 2.13x |
| `cudaMallocAsync` | 0.071 s | 0.119 s | 1.68x |
| `cudaStreamSynchronize` | 0.047 s | 0.069 s | 1.47x |

Driver call chains exposed read/write-lock contention in `libcuda.so`, including
paths through kernel launch and the host callbacks that return provider handles
and streams. The ordinary cuBLAS path acquires an actually-idle stream and
handle, enqueues dependency waits and the provider operation, records one shared
access-completion event, and enqueues tail callbacks that return the handle and
stream. One host submitter can therefore fill every stream-pool slot before it
waits for another slot. A second submitter does not increase the four-stream
in-flight limit; it primarily races the first through the same driver, event,
callback, and allocation paths.

The GPU trace confirms that the added host concurrency produced little useful
device concurrency. With one submitter, the kernels occupied 587.480 ms and did
not overlap. With two submitters, only 46.395 ms had two kernels active, while
summed kernel duration grew to 772.784 ms and the union of GPU-busy intervals
grew to 726.390 ms. Kernel overlap was therefore too small to compensate for
host contention and slower concurrent device execution.

Packed CUDA BlockTensor storage is not serializing all blocks through one
completion ledger. Its blocks share one physical allocation, but each partition
has its own `CudaBuffer` access state. Disjoint sector operations can therefore
be in flight independently.

cuSOLVER exposes a distinct issue. The current exact-SVD wrapper submits the
factorization, copies device `devInfo` toward the host, and then synchronizes the
operation stream before returning. A single scheduler submitter consequently
cannot overlap separate sector SVD wrappers. Two submitters did produce a peak
of two concurrent sector tasks, but those small sector operations became slower:

| CUDA submitters | Median SVD batch wall | Median summed item time | Peak item overlap |
|---:|---:|---:|---:|
| 1 | 0.125 s | 0.125 s | 1 |
| 2 | 0.157 s | 0.307 s | 2 |

The item interval includes sector assembly, copies, provider work, and the final
stream synchronization; it is not a pure factorization measurement. Provider
tracing nevertheless showed that the 662 `cusolverDnDgesvd` host calls themselves
grew from 0.521 seconds total with one submitter to 1.029 seconds with two.

This does not imply that independent cuSOLVER operations cannot overlap. The
intended design separates general CUDA submission from solver concurrency. One
general submission lane remains appropriate for fine-grained block operations.
An asynchronous SVD boundary can submit independent sectors sequentially from
one host lane onto separate handles and streams, retain every operation's
workspace and access state, and join the completions only after all eligible
sectors have been submitted. The provider calls remain serial on the host, while
their pending device work may overlap. If a routine's host call is itself too
long to supply multiple streams, a small provider-specific lane can be measured
without exposing all ordinary CUDA work to multiple submitters. See
[CUDA/cuSOLVER Architecture Notes](../backends/cuda/cusolver.md).

## 5. Local Matrix Product Toolkit Orientation

A controlled local comparison used the same open Heisenberg model, U(1)
symmetry, deterministic Néel product MPS, maximum bond dimension 64, seven
directional traversals, and physical CPU 0. Matrix Product Toolkit used
`mp-dmrg-2site` and its complex scalar implementation. Every MPTK trial started
from a fresh copy because the tool updates its input wavefunction.

Two MPTK local-solver policies were measured:

- **adaptive default:** `miniter=4`, `maxiter=20`, and `maxtol=4e-4`; the
  active tolerance is
  `min(sqrt(moving_average_fidelity_loss), maxtol)`, subject to its internal
  lower bound;
- **fixed four:** `--miniter=4 --maxiter=4`, so every non-breaking local solve
  uses four Hamiltonian applications regardless of the adaptive tolerance.

Uni20's default and explicitly selected policies are both fixed four. Its
solver also caps the Krylov basis by the known local vector-space dimension,
so small edge problems may use fewer applications before any numerical
breakdown test is needed.

Three-run median end-to-end wall times were:

| Length | Uni20 real, fixed four | Uni20 complex, fixed four | MPTK complex, fixed four | MPTK complex, adaptive default |
|---:|---:|---:|---:|---:|
| 40 | 0.07 s | 0.12 s | 0.21 s | 0.24 s |
| 80 | 0.17 s | 0.28 s | 0.47 s | 0.53 s |
| 160 | 0.34 s | 0.62 s | 1.02 s | 1.14 s |

The corresponding terminal energies after seven traversals were deterministic
across the three timing trials:

| Length | Uni20 real, fixed four | Uni20 complex, fixed four | MPTK complex, fixed four | MPTK complex, adaptive default |
|---:|---:|---:|---:|---:|
| 40 | -17.54147329019870 | -17.54147329019864 | -17.54147329019872 | -17.54147329549254 |
| 80 | -35.26522672317349 | -35.26522672317333 | -35.26522672317359 | -35.26523131616734 |
| 160 | -70.71525958236920 | -70.71525958236889 | -70.71525958236924 | -70.71559797850429 |

The fixed-four trajectories agree to roundoff across Uni20 real, Uni20
complex, and MPTK complex. This is useful cross-implementation validation of
the current local-work policy. MPTK's adaptive default spends progressively
more local work and obtains a lower energy after the same seven traversals:

| Length | Local updates | Uni20 fixed-four Hamiltonian applications | MPTK fixed-four reported iterations | MPTK adaptive-default reported iterations |
|---:|---:|---:|---:|---:|
| 40 | 273 | 1,049 | 1,083 | 1,359 |
| 80 | 553 | 2,129 | 2,203 | 2,868 |
| 160 | 1,113 | 4,289 | 4,446 | 5,859 |

MPTK averaged about 5.0, 5.2, and 5.3 iterations per local update under its
adaptive default. Fixed four reduced its median wall time by approximately
12%, 11%, and 11% at lengths 40, 80, and 160 respectively. The energy
difference does not establish that either policy is better: a useful DMRG
comparison must follow the complete sweep trajectory to a common energy or
variance target. In particular, a fidelity-derived local tolerance may spend
too much or too little work when environment quality and local-state change do
not track one another reliably.

The individual wall-time trials were:

```text
L=40:  Uni20 real fixed4     0.07, 0.10, 0.07 s
       Uni20 complex fixed4  0.13, 0.12, 0.12 s
       MPTK complex fixed4   0.22, 0.21, 0.21 s
       MPTK complex default  0.23, 0.24, 0.24 s
L=80:  Uni20 real fixed4     0.17, 0.17, 0.16 s
       Uni20 complex fixed4  0.28, 0.28, 0.28 s
       MPTK complex fixed4   0.44, 0.48, 0.47 s
       MPTK complex default  0.53, 0.53, 0.53 s
L=160: Uni20 real fixed4     0.34, 0.34, 0.35 s
       Uni20 complex fixed4  0.62, 0.61, 0.62 s
       MPTK complex fixed4   1.00, 1.03, 1.02 s
       MPTK complex default  1.14, 1.15, 1.13 s
```

Representative reproduction commands are:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

length=80
spinchain-u1 -S 0.5 -o spinchain-u1.lattice
mp-construct --finite --repeat $((length / 2)) \
    --lattice spinchain-u1.lattice --output neel-${length}-base.psi \
    '0.5:-0.5'

# MPTK adaptive defaults: miniter=4, maxiter=20, maxtol=4e-4.
cp neel-${length}-base.psi neel-${length}-default.psi
taskset -c 0 mp-dmrg-2site -w neel-${length}-default.psi \
    -H spinchain-u1.lattice:H_J1 -m 64 -s 7

# MPTK fixed four.
cp neel-${length}-base.psi neel-${length}-fixed4.psi
taskset -c 0 mp-dmrg-2site -w neel-${length}-fixed4.psi \
    -H spinchain-u1.lattice:H_J1 -m 64 -s 7 \
    --miniter 4 --maxiter 4

# Uni20 fixed four; --local-matvecs=4 is also the default.
taskset -c 0 spin_half_heisenberg_dmrg_example \
    --sites=${length} --max-states=64 --max-sweeps=7 \
    --energy-tol=0 --local-matvecs=4 --block-threads=1
```

The MPTK executable identified itself as
`dmrg-ee:780d908f-dirty`, built with GCC 15.2 and OpenBLAS 0.3.32. The dirty
revision and interactive laptop make these orientation measurements rather
than portable performance claims. The fixed-four energy agreement is the more
durable result. The shared Néel start also removes the large convergence bias
of a random U(1) MPS, whose bond charges follow a random walk.

## 6. Published Orientation Points

Published numbers below identify useful scaling targets, not comparable Uni20
benchmarks:

| System | Reported result | Relevance to Uni20 |
|---|---|---|
| [ITensor block-sparse DMRG](https://scipost.org/SciPostPhysCodeb.4) | About 1.5x to 2x from block-sparse threading in the reported 2D DMRG cases; dense BLAS threading was less effective for small symmetry blocks. | Closest published analogue of the current output-block batches and single-threaded BLAS policy. |
| [DMRG++ OpenMP tasking](https://doi.org/10.1007/978-3-030-28596-8_20) | Hamiltonian mini-application speedup of 8.0x with 8 threads and 20.5x with 40 threads on Power9. | Demonstrates the additional concurrency available from finer Hamiltonian tasks, dependencies, priorities, and reductions. It is not an end-to-end sweep result. |
| [Distributed Cyclops DMRG](https://arxiv.org/abs/2007.05540) | Up to 5.9x runtime improvement and 99x processing-rate improvement over the paper's ITensor baseline at roughly comparable resource use. | Long-term MPI reference: distributed block contraction and distributed per-sector SVD become important at bond dimensions from thousands to tens of thousands. |
| [block2](https://doi.org/10.1063/5.0180424) | Combines parallelism over sub-Hamiltonians, sites, operators, symmetry sectors, and dense multiplication. | Confirms that large-scale DMRG needs several composable parallel axes rather than one universal thread policy. |
| [Eight-package 2026 study](https://arxiv.org/abs/2607.28369) | Reports package trajectories on three fixed single-core, length-100 fixtures. | Supplies concrete external developer benchmarks. It is not used as a package ranking or a source of generally optimal DMRG settings. |

The official
[ITensor benchmark repository](https://github.com/ITensor/ITensorBenchmarks.jl)
separates BLAS-thread and block-sparse-thread runs. A future direct Uni20/ITensor
comparison should reuse that separation and add the same model to both suites.

### 6.1 DMRG-Benchmarks developer targets

The source configuration accompanying arXiv:2607.28369 is archived in the
[DMRG-Benchmarks repository](https://github.com/PerSehlstedt/DMRG-benchmarks/tree/8ca54b77386ea04cedb903c3f550ed1e1bdc72c3).
The paper configurations define the following targets:

| Fixture | Symmetries | Maximum bond | Initial random bond | Full sweeps | Local Krylov dimension | Reference energy |
|---|---|---:|---:|---:|---:|---:|
| Critical transverse-field Ising, `L=100`, `h/J=1` | trivial, Z2 | 100 | 10 | 10 | 6 | -126.961876739681 |
| Antiferromagnetic spin-1 Heisenberg, `L=100`, `J=-1` | trivial, U(1), SU(2) | 400 | 40 | 5 | 6 | -138.940086166525 |
| Half-filled Fermi-Hubbard, `L=100`, `U/t=8` | U(1)xU(1), U(1)xSU(2), SU(2)xU(1), SO(4) | 800 | 80 | 22 | 10 | -32.545776173096 |

All use open boundaries, real double-precision arithmetic, two-site updates,
truncation and local-solver tolerances of `1e-14`, and ten independently
initialized runs. Only the DMRG call is timed. The published runs use one core
of an Intel Xeon Gold 6132. The adapter configuration supplies Lanczos
`maxiter=1` or Davidson `restart=3`, according to package support; those are
benchmark controls interpreted by different implementations, not identical
eigensolver algorithms or equivalent iteration limits.

The paper calls a complete forward and backward pass one sweep. Uni20 currently
counts one directional traversal as a sweep, so reproducing the fixtures
requires 20, 10, and 44 Uni20 traversals respectively. The benchmark harness
must report energy and cumulative DMRG wall time after each complete pair of
traversals. A fixed recorded random seed is required for a regression target;
the ten-run ensemble should retain every trial rather than reporting only a
best time.

These targets should be added incrementally as the corresponding mathematical
support becomes available:

1. The trivial-symmetry Ising case tests the dense/no-symmetry path with low
   bond dimension and high framework overhead sensitivity.
2. The U(1) spin-1 Heisenberg case is the first directly relevant block-sparse
   target. It requires a spin-1 model builder and a random fixed-sector MPS that
   is canonicalized before the first directional update.
3. Z2 and SU(2) variants follow their symmetry implementations.
4. The Fermi-Hubbard variants require the appropriate product symmetries and a
   fermionically correct MPO representation before their timings are meaningful.

For cross-package orientation, reproduce the single-core configuration exactly.
For Uni20 optimization, retain the same physical and algorithmic fixture and
add separate block-thread and accelerator scaling series. Do not substitute a
Néel or other product state in the named external fixture: that is a better
initial state for many user calculations, but it changes this particular
benchmark trajectory.

## 7. Block-SVD Parallelization

These changes retain the current execution rule: independent
symmetry work uses lightweight scheduler batches, while nested LAPACK and BLAS
remain single-threaded. The execution choice must be propagated from the
transient center or DMRG operation policy. It must not accidentally require the
persistent MPS site storage to select parallel execution: selected factors are
currently built in intermediate packed storage before being copied into that
site type.

### 7.1 Per-sector block SVD: implemented

Charge-sector SVDs are mathematically independent. The implementation:

1. Build the canonical list of matched domain/codomain sectors.
2. Preallocate one result slot per canonical sector.
3. Estimate cost with a cubic proxy such as `m * n * min(m, n)` and submit
   sectors largest first.
4. In each task, assemble the sector matrix and call `linalg::svd` into that
   sector's unique result slot.
5. Join the batch, construct the decomposition in canonical sector order, and
   perform the small global singular-spectrum sort serially.

This needs no locking, and factorization order cannot change the mathematical
ordering of sectors or the deterministic singular-value tie-break. The
decomposition type retains the source block execution policy alongside the
factors and boundary metadata, so later materialization can use the same policy.

A controlled eight-participant A/B comparison used the `L=80`, `m=512`,
nine-traversal fixture with every other parallel path unchanged. Serial sector
SVDs took 20.08, 19.92, and 19.65 s; parallel sector SVDs took 19.19, 19.34, and
19.34 s. The median improved from 19.92 to 19.34 s, or about 2.9 percent. Every
run ended at energy `-35.26523705466045`. This confirms that sector SVD is a
modest part of the current workload, while still providing measurable useful
work at the existing batch granularity.

### 7.2 Post-truncation block construction: next

Selected left vectors, right vectors, and diagonal singular values are
currently materialized by serial nested loops. Construction should be split
into a serial structural phase and a parallel payload phase:

1. Build the selected bond space, canonical output keys, block extents, packed
   offsets, and one owning output allocation.
2. Build work items for distinct output blocks or disjoint packed ranges.
3. Sort by the number of copied/scaled values and execute largest first.
4. Scatter selected columns, rows, or diagonal values into each task's unique
   output block.
5. Join before factor absorption or MPS replacement.

The selection plan and provider factors are immutable during this batch. Packed
storage is safe because tasks write disjoint logical block ranges even though
they share one allocation. No temporary dense projection or loss of charge
metadata is permitted.

## 8. Future Baseline Requirements

Every retained performance table should add:

- source revision and build configuration;
- CPU/GPU model, memory, compiler, BLAS/LAPACK, and scheduler versions;
- model, length, boundary conditions, symmetry, and scalar type;
- initial-state construction and initial energy;
- one-site or two-site update mode and complete eigensolver/truncation policy;
- wall time, user time, peak memory, sweep trajectory, final energy error, and
  preferably energy variance;
- warmup policy and individual trial values, not only the best result.

Microbenchmarks for contraction, SVD, and materialization remain useful for
optimization, but end-to-end DMRG comparisons must report solution quality.
