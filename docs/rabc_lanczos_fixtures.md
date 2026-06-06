# R/A/B/C Lanczos Fixtures

The TensorContraction bridge can capture one real two-site DMRG effective-Hamiltonian solve and replay it without the
surrounding MPS sweep, SVD, or environment update. This is intended for benchmarking the resident R/A/B/C contraction
path and block placement policies.

## Capture

Set `UNI20_RABC_DUMP_PATH` while running `spin_half_heisenberg_u1_dmrg`. The dump hook runs after the strict U(1)
block-sparse MPS/MPO path has built the legal TensorContraction worklist, then writes the static `A` and `C`
environments, active `B` input vector, output block shapes, and `F` terms.

Useful selectors:

| Variable | Meaning |
| --- | --- |
| `UNI20_RABC_DUMP_PATH` | Output binary fixture path. Enables capture. |
| `UNI20_RABC_DUMP_LEFT_SITE` | Optional two-site left index filter. |
| `UNI20_RABC_DUMP_MIN_BOND_DIM` | Optional minimum left and right external bond dimension. |
| `UNI20_RABC_DUMP_MATCH_INDEX` | Optional zero-based index among matching solves. |
| `UNI20_RABC_DUMP_EXIT` | If truthy, exit immediately after writing the selected fixture. |

Example central `m=2048` capture:

```bash
env HWLOC_HIDE_ERRORS=2 \
  UNI20_HEISENBERG_LENGTH=40 \
  UNI20_HEISENBERG_MAX_RANK=2048 \
  UNI20_HEISENBERG_SWEEPS=7 \
  UNI20_RABC_DUMP_PATH=/tmp/uni20_l40_m2048_central.rabc \
  UNI20_RABC_DUMP_LEFT_SITE=19 \
  UNI20_RABC_DUMP_MIN_BOND_DIM=2048 \
  UNI20_RABC_DUMP_MATCH_INDEX=0 \
  UNI20_RABC_DUMP_EXIT=1 \
  ./build_codex/tensorcontraction-polaron-release-fresh/examples/spin_half_heisenberg_u1_dmrg
```

The fixture is a compact local binary format. It is not intended as a portable archival wavefunction format.

For Hubbard U(1)xU(1) placement work, use `fermi_hubbard_u1u1_dmrg`. The example
fixes the physical sector to half filling and total spin zero by seeding an
even-length chain with alternating `|up>, |down>` states, giving final boundary
charge `(N=L, Sz=0)`. This is the target sector for serious layout benchmarks;
the existing `L=20, m=16` Hubbard fixture is only a smoke/topology probe and is
pure GPU overhead for performance work. Do not use it to tune production layout
policy. Serious layout fixtures should be captured from central bonds in the
same sector at bond dimensions high enough to put nontrivial time inside cuBLAS
DGEMM. In practice, that starts around `m=4000`; `m=5000` to `m=10000` is the
immediate target scale, and larger fixtures are useful when the system size,
memory, and runtime budget allow.

## Replay

Replay the fixture with a fixed Lanczos iteration count:

```bash
env HWLOC_HIDE_ERRORS=2 \
  UNI20_RABC_LANCZOS_ITERS=24 \
  UNI20_RABC_LANCZOS_MIN_ITERS=24 \
  MP_BENCHFILE=/tmp/uni20_rabc_replay.bench \
  ./build_codex/tensorcontraction-polaron-release-fresh/examples/tensorcontraction_rabc_lanczos_benchmark \
  /tmp/uni20_l40_m2048_central.rabc
```

The replay benchmark uploads the captured initial vector once, keeps host synchronization disabled, and resets each
repeat with resident vector algebra. `UNI20_TENSORCONTRACTION_DEVICES=all` exercises one-process multi-GPU placement;
`mpirun -np 2` with `CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK` exercises one GPU per MPI rank.

## Placement Experiments

The default resident bridge places active `MatrixFamily` blocks in coalesced byte-balanced slabs.  This is still the
best default for end-to-end Lanczos because the vector-algebra operations can use slab kernels.

Set `UNI20_TENSORCONTRACTION_RABC_PLACEMENT=cost` to enable the experimental R/A/B/C output-placement policy.  This
policy uses a small dynamic program to choose contiguous `R` block ranges for each local device while minimizing the
estimated maximum per-device matvec cost for the current right-first schedule:

```text
Y = B * C
R += A * Y
```

The default model scores GEMM flops and central-vector `B` movement only.  It deliberately ignores the setup cost of
materializing `A` and `C` environment blocks on the selected devices, because the fixture replay is intended to time the
resident Lanczos loop after environment placement has been arranged.  Add
`UNI20_TENSORCONTRACTION_RABC_MODEL_ENV_BYTES=1` only when profiling a staging policy rather than a Lanczos matvec
policy.

Useful controls:

| Variable | Meaning |
| --- | --- |
| `UNI20_TENSORCONTRACTION_RABC_PLACEMENT` | `cost`, `greedy`, or `cost-greedy` enables contiguous cost placement. |
| `UNI20_TENSORCONTRACTION_RABC_PLACEMENT=cost-block` | Enables the more aggressive arbitrary block-ownership policy. Blocks are still stored as one coalesced slab per device. |
| `UNI20_TENSORCONTRACTION_RABC_PLACEMENT=empirical-contiguous` | Enables a two-device contiguous policy scored by fitted no-trace benchmark coefficients. Aliases: `empirical`, `bench-contiguous`. |
| `UNI20_TENSORCONTRACTION_RABC_PLACEMENT=stripe` | Places center blocks round-robin over local devices, equivalent to an alternating `0,1,0,1,...` layout on two GPUs. Aliases: `striped`, `round-robin`, `alternating`. |
| `UNI20_TENSORCONTRACTION_RABC_PLACEMENT=manual` | Uses the explicit center-block device list supplied by `UNI20_TENSORCONTRACTION_RABC_PLACEMENT_LAYOUT`. |
| `UNI20_TENSORCONTRACTION_RABC_PLACEMENT_LAYOUT` | Comma-separated CUDA device id per center block, for example `0,1,0,1`. The list length must equal the center-vector block count. |
| `UNI20_TENSORCONTRACTION_RABC_PLACEMENT_LOG` | Set to `1`, `true`, or `on` to print the selected output-block distribution. |
| `UNI20_TENSORCONTRACTION_RABC_MODEL_GFLOPS` | Assumed per-device GEMM throughput. Default: `1000`. |
| `UNI20_TENSORCONTRACTION_RABC_MODEL_CENTRAL_GBPS` | Assumed central-vector transfer bandwidth. Default: `32`. |
| `UNI20_TENSORCONTRACTION_RABC_MODEL_ENV_BYTES` | If set, include environment staging bytes in the model. Default: unset. |
| `UNI20_TENSORCONTRACTION_RABC_MODEL_ENV_GBPS` | Assumed environment transfer bandwidth when environment bytes are enabled. Default: central bandwidth. |
| `UNI20_TENSORCONTRACTION_RABC_MODEL_CONTIGUOUS_MIN_SPEEDUP` | Minimum predicted speedup required before `cost` overrides byte-balanced contiguous ranges. Default: `1.05`. |
| `UNI20_TENSORCONTRACTION_RABC_MODEL_ARBITRARY_MIN_SPEEDUP` | Minimum predicted speedup required before `cost-block` overrides byte-balanced slab layout. Default: `1.25`. |
| `UNI20_TENSORCONTRACTION_RABC_EMPIRICAL_COEFFICIENTS` | Comma-separated fitted coefficients for `empirical-contiguous`, in the `bench-fit --model device` runtime order. |
| `UNI20_TENSORCONTRACTION_RABC_EMPIRICAL_COEFFICIENTS_FILE` | Text file containing coefficient stanzas. The runtime accepts a raw comma list, `runtime_coefficients=...`, or `UNI20_TENSORCONTRACTION_RABC_EMPIRICAL_COEFFICIENTS=...`. Generated stanzas also include `runtime_supported_output_blocks=...`; if present, the runtime uses the first stanza matching the current output block count and falls back to byte-balanced placement when none match. |
| `UNI20_TENSORCONTRACTION_RABC_EMPIRICAL_MIN_SPEEDUP` | Minimum fitted-score improvement required before `empirical-contiguous` overrides byte-balanced contiguous ranges. Default: `1.0`. |
| `UNI20_TENSORCONTRACTION_RABC_TRACE_PATH` | Appends one JSONL record per deterministic resident matvec with layout, feature, and timing data for empirical model fitting. |
| `UNI20_TENSORCONTRACTION_RABC_TRACE_TERMS` | If set, include the full term list and selected device for each term in each JSONL record. |

The cost policies are intentionally opt-in.  The contiguous policy compares its
model-selected split with the default byte-balanced split and falls back unless
the predicted speedup clears `UNI20_TENSORCONTRACTION_RABC_MODEL_CONTIGUOUS_MIN_SPEEDUP`.
The arbitrary block policy can still lose to the default layout when the R/A/B/C
model underestimates vector-layout or relayout costs.
The `stripe` policy is also opt-in, but is deterministic rather than fitted.  It is useful as an empirical baseline
because it exercises the measured alternating block layout directly, while avoiding local-search extrapolation outside
measured layouts.  It is not a default: validate it against the byte-balanced layout with tracing disabled before using
it for production benchmarks.
The `empirical-contiguous` policy is also opt-in and currently limited to two
local CUDA devices.  It evaluates every nonempty ordered split of the center
blocks and scores each split with the device-aware no-trace benchmark feature
order:

```text
intercept,
d0_right_flops,d0_b_peer_bytes,d0_terms,d0_unique_bc,d0_output_bytes,
d1_right_flops,d1_b_peer_bytes,d1_terms,d1_unique_bc,d1_output_bytes,
layout_transitions,layout_segments,active_devices,
max_output_block_fraction,max_output_byte_fraction
```

Generate this coefficient vector with `bench-fit --model device` or
`bench-suggest --model device`; both commands print the runtime
`UNI20_TENSORCONTRACTION_RABC_EMPIRICAL_COEFFICIENTS` value.  Use
`--output-runtime-coefficients <path>` to write a reusable coefficient file for
`UNI20_TENSORCONTRACTION_RABC_EMPIRICAL_COEFFICIENTS_FILE`; the generated file
includes the coefficient order as comments, a `runtime_supported_output_blocks`
guard for the fitted fixture block count, and a `runtime_coefficients=...`
line.  Do not use the policy for cold-start/environment-staging fits: the
device-aware runtime model targets steady-state resident Lanczos matvec timing
only.
Multiple generated stanzas may be concatenated into one coefficient bundle when
separate fixture shapes have been fitted.  Shape guards are deliberately strict:
`m=16` runs are useful smoke and overhead regression probes, but their fitted
coefficients must not silently drive larger DMRG or fixture layouts.

Device-aware fits with `--graph-features` use a 28-value runtime coefficient
order.  The graph-augmented order keeps the same intercept and layout features,
and inserts these six per-device counters immediately after each device's
`output_bytes` coefficient:

```text
b_cut_terms,b_peer_blocks,right_duplicate_groups,
mixed_duplicate_groups,mixed_left_groups,mixed_right_groups
```

The C++ runtime accepts both the 16-value base order and this 28-value
graph-augmented order.  The graph features target cut, peer-block,
duplicate-group, and mixed-order effects in the `f` hypergraph; validate them
against held-out benchmark layouts before using them as the active
`empirical-contiguous` placement model.
The current cost model is a variable-middle prototype: the Krylov input blocks `B_i` and output blocks `R_i` are forced
onto one canonical layout so vector algebra can use the result as the next Lanczos input without an implicit relayout.
Benchmark both `#LanczosMatvecS` and total wall time before treating a lower contraction model score as a faster
Lanczos solve.

The placement plan is cached inside the resident operator after the first resident apply for a given policy and device
count.  Use `UNI20_RABC_WARMUP=1` when comparing matvec timings so plan construction and environment materialization
are treated as setup rather than Lanczos-loop cost.

## Empirical Cost Tracing

For model fitting, run the fixture benchmark with an explicit layout and a trace path:

```bash
env \
  UNI20_TENSORCONTRACTION_DEVICES=2 \
  UNI20_TENSORCONTRACTION_RABC_PLACEMENT=manual \
  UNI20_TENSORCONTRACTION_RABC_PLACEMENT_LAYOUT=<one-device-id-per-center-block> \
  UNI20_TENSORCONTRACTION_RABC_TRACE_PATH=/tmp/uni20_rabc_trace.jsonl \
  UNI20_RABC_WARMUP=1 \
  UNI20_RABC_REPEATS=4 \
  ./build_codex/tensorcontraction-polaron-release-fresh/examples/tensorcontraction_rabc_lanczos_benchmark \
  /tmp/uni20_l40_m2048_central.rabc
```

Each trace record contains the input/output block layout, per-device feature aggregates, host enqueue time, host wait
time, and CUDA-event elapsed time.  The CUDA events are recorded on the legacy stream to create a diagnostic boundary
around the blocking work streams on each device.  This makes tracing heavier than the default benchmark path but much
lighter than Nsight Systems, and it produces directly usable feature rows for fitting bandwidth and GEMM-throughput
parameters.

Trace rows are model-fitting diagnostics, not final benchmark results.  CUDA-event tracing, term tracing, and trace
file writes can change the relative cost of layouts.  Use traces to choose candidate layouts, then rerun the fixture
benchmark with `UNI20_TENSORCONTRACTION_RABC_TRACE_PATH` unset before claiming a performance improvement.

The current feature rows describe the right-first executor:

```text
Y = B * C
R += A * Y
```

Per-device fields include `bc_flops`, `accumulate_flops`, local versus peer `B` bytes, `A`/`C` environment bytes,
output bytes, intermediate bytes, term count, and unique block counts.  These are intentionally aggregate features for
the first empirical model.  Set `UNI20_TENSORCONTRACTION_RABC_TRACE_TERMS=1` when debugging individual block placement
or when a later optimizer needs term-level training data.  Term tracing also records the matrix dimensions and per-term
right-first flops, which allows offline scoring of candidate layouts that were not directly measured.

Use `order-summary` on a term trace to compare the current right-first schedule
with the left-first alternative and to inspect per-center-block mixed-order
pressure from the sparse `f` hypergraph:

```bash
scripts/rabc-trace-model.py order-summary /tmp/uni20_rabc_trace.jsonl --devices --blocks --top-blocks 12
```

Use `graph-summary` on a term trace to inspect the layout-dependent graph cuts:
unique peer `B` blocks implied by the canonical center-vector layout,
duplicated first-stage `(B,C)` groups for right-first execution, duplicated
`(A,B)` groups for left-first execution, and the corresponding metrics for a
simple mixed per-`(device,B)` order choice:

```bash
scripts/rabc-trace-model.py graph-summary /tmp/uni20_rabc_trace.jsonl --devices
scripts/rabc-trace-model.py graph-summary /tmp/uni20_rabc_trace.jsonl \
  --layout <one-device-id-per-center-block>
```

## Hypergraph Placement Model

The sparse coefficient tensor `f` is the ground-truth hypergraph for placement.
Each nonzero term connects one output block `R_r`, one input/Krylov block `B_b`,
and one left/right environment pair `(A_a, C_c)`.  A resident Lanczos layout is
therefore a partitioning problem with at least two coupled decisions:

| Decision | Current prototype | Future generalization |
| --- | --- | --- |
| Center-vector owner | One device id per `R/B` block. | Arbitrary per-block owner over all local or MPI-visible devices. |
| First contraction order | Global right-first executor, with offline mixed-order diagnostics. | Per-term or per-group choice of left-first versus right-first. |
| Intermediate placement | Kept local to the first executor device. | Optional migration before second-stage accumulation. |
| Accumulation placement | Canonical `R` owner. | Optional partial accumulation followed by reduction/migration. |

For two GPUs and a fixed canonical `B/R` layout, this already looks like a
four-way split: GPU 0/right-first, GPU 0/left-first, GPU 1/right-first, and
GPU 1/left-first.  Allowing explicit intermediate moves or alternate
accumulation sites adds a small finite number of labels.  Once each label has a
measured cost model, layout selection becomes a weighted hypergraph partition
or min-cut problem.  Contiguous quantum-number ranges are only a useful
restricted search family; the traced `f` connectivity is the actual objective.

The current implementation is intentionally narrower.  It keeps one canonical
Krylov-vector layout so successive Lanczos matvecs and vector algebra do not
need implicit relayouts, and it fits aggregate feature costs rather than solving
the full labeled hypergraph.  This is enough to reject bad layouts such as
round-robin striping when they create excessive peer-`B` cuts and duplicated
first-stage groups, while leaving the mixed left/right and migration choices as
explicit future scheduling variables.

Use `scripts/rabc-trace-model.py` to inspect and fit these traces:

```bash
scripts/rabc-trace-model.py summary /tmp/uni20_rabc_trace.jsonl
scripts/rabc-trace-model.py fit /tmp/uni20_rabc_trace.jsonl
scripts/rabc-trace-model.py suggest /tmp/uni20_rabc_trace.jsonl
scripts/rabc-trace-model.py order-summary /tmp/uni20_rabc_trace.jsonl --devices --blocks --top-blocks 12
scripts/rabc-trace-model.py graph-summary /tmp/uni20_rabc_trace.jsonl --devices
```

The `fit` subcommand performs a small ridge least-squares fit to per-device CUDA-event timings.  The `suggest`
subcommand fits the same model, clamps negative coefficients by default, and performs deterministic single-block
local search over candidate center-vector layouts.  This is a first empirical optimizer, not a proof of global
optimality.  It is intended to generate candidate manual layouts that should then be rerun through the fixture replay
and compared against the measured `gpu_s` trace field.

For term traces, add `--graph-features` to `fit`, `validate`, `tune`, or `suggest`
to include graph counters in the fitted empirical model.  The current graph
features are peer-`B` term count, unique peer-`B` block count, and duplicated
right-first `(B,C)` group count per device.  These target the observed
small-block overheads that are not fully represented by byte and flop features.
Graph counters are correlated with the base features, so tune the ridge and
validate held-out layouts before using them for suggestions.  Do not assume that
coefficient clamping remains valid for graph-feature fits; compare clamped and
unclamped models with `tune --include-clamped`.

When timing rows do not include repeated term dumps, pass a companion term trace
to supply the static `f` hypergraph:

```bash
scripts/rabc-trace-model.py tune /tmp/uni20_rabc_timing_trace.jsonl \
  --term-trace /tmp/uni20_rabc_term_trace.jsonl \
  --graph-features \
  --drop-first-per-layout=1
scripts/rabc-trace-model.py suggest /tmp/uni20_rabc_timing_trace.jsonl \
  --term-trace /tmp/uni20_rabc_term_trace.jsonl \
  --graph-features \
  --observed-only
```

The companion term trace may come from a single-device replay. For no-trace
benchmark fitting, the script treats the `f` tensor as device-independent and
infers the active device count from the measured manual layouts in the
benchmark JSONL. This allows a small term trace to be reused when comparing
single-node two-GPU candidate layouts.

Even lightweight CUDA-event tracing can perturb the resident matvec path because
it inserts timing events and synchronization boundaries.  The `gpu_s` trace
target is useful for structural model development, but final placement decisions
must be confirmed by rerunning the fixture replay with
`UNI20_TENSORCONTRACTION_RABC_TRACE_PATH` unset.

Prefer the scripted sweep wrapper for repeated no-trace layout measurements:

```bash
scripts/run-rabc-layout-sweep.sh \
  --fixture /home/ian/sync/fixtures/tensorcontraction/uni20_l40_m4608_central.rabc \
  --output-dir /tmp/uni20_rabc_l40_m4608_sweep \
  --labels cut3,cut4,cut5,cut6,cut7,cut8,cut9,cut10,cut12,cut14,cut16,cut21,cut36,alternating \
  --repeats 3 \
  --iters 24 \
  --timeout 240
```

The script writes one `MP_BENCHFILE` table per layout and rebuilds a combined
`benchmarks.jsonl` dataset with `bench-record`.  Use `--resume` to continue an
interrupted sweep without rerunning layouts that already have nonempty bench
files.  Use `--trace-path auto --trace-terms` only for a companion term trace or
other structural diagnostics; do not compare final performance from traced rows.
Use `--policy cost-block --labels cost-block` or another non-manual policy name
to benchmark an automatic placement path through the same wrapper.  Non-manual
policy runs enable placement diagnostics automatically and infer the selected
layout when the diagnostic line contains either a contiguous `cut=N` or explicit
`deviceK={blocks=[begin,end)}` ranges.  Inferred layouts are written to the same
`benchmarks.jsonl` dataset as manual layouts, so automatic policies such as
`empirical-contiguous`, `cost`, and `stripe` can be compared with
`bench-summary`.  If a future policy does not expose a parseable layout, the
wrapper still keeps the raw bench table but reports `inferred_layout=unavailable`.
Use `bench-rank` when comparing automatic policies with manual cuts, because it
groups rows by the actual recovered layout rather than the run name:

```bash
scripts/run-rabc-layout-sweep.sh \
  --fixture /home/ian/sync/fixtures/tensorcontraction/uni20_hubbard_l40_m5000_central.rabc \
  --output-dir /tmp/uni20_rabc_empirical_replay \
  --policy empirical-contiguous \
  --labels empirical-contiguous \
  --empirical-coefficients-file /tmp/uni20_rabc_empirical_coefficients.txt
scripts/rabc-trace-model.py bench-rank /tmp/uni20_rabc_empirical_replay/benchmarks.jsonl \
  --compact-layouts \
  --selected-name empirical-contiguous
```

To benchmark segmented candidates directly, generate explicit layout strings
with the same bounded segmented generator and pass selected rows to
`run-rabc-layout-sweep.sh --layout NAME=LIST`.  For long generated layouts,
prefer `--layout-file NAME=PATH`; the wrapper accepts either a raw comma list
or a `bench-suggest` output file containing a `layout=` line, which avoids
truncating a hundreds-of-block layout at the shell:

```bash
scripts/rabc-trace-model.py layouts \
  --block-count 730 \
  --device-count 2 \
  --segmented-cuts \
  --max-segments 3 \
  --segment-cut-stride 20 \
  --max-segment-layouts 20000
```

The first bounded segmented probe on the local Hubbard `L=40, m=5000` fixture
did not beat the contiguous basin.  The best measured segmented candidate,
`seg3_start0_cuts300_340`, placed 690 blocks on GPU 0 and 40 blocks on GPU 1
in three segments and measured mean `#LanczosMatvecS = 0.923069940s` over
three repeats.  This is slower than the best focused contiguous comparison
around `cut326` at about `0.888s`.  A later graph-feature `bench-suggest`
diagnostic extrapolated to a three-segment `200/280` candidate with 650 blocks
on GPU 0 and 80 on GPU 1, but direct replay measured mean
`#LanczosMatvecS = 0.942111177s` over three repeats.  After constraining
segmented suggestions to measured shape support by segment count, the guarded
top-three candidates were also replayed directly.  The best of those placed
670 blocks on GPU 0 and 60 blocks on GPU 1 and measured mean
`#LanczosMatvecS = 0.898845987s` over three repeats.  That is competitive with
nearby contiguous cuts but still slower than the `cut326`/`empirical-contiguous`
comparison at mean `0.888880086s` over nine rows.  This confirms that the
current useful automatic search space remains contiguous layouts unless a
fixture supplies measured evidence for non-contiguous support.  Static
structural diagnostics explain the failure modes: segmented candidates can
lower right-first and mixed critical-path flops, but they tend to increase
peer-`B` traffic and heavily skew terms/unique `B,C` groups onto one device.

For the local `L=40, m=4608` central fixture on Polaron, the repeated no-trace
dual-GPU sweep over all contiguous cuts found `cut6` as the current best
measured contiguous layout: mean `#LanczosMatvecS = 0.169051257s` over three
repeats.  The best model-suggested middle cut was `cut15` at `0.185988945s`;
the right-heavy tail was slower, with cuts beyond `cut23` generally above
`0.21s`.  Treat this as a fixture-local reference point, not a portable
placement rule.

For the local Hubbard `L=40, m=5000` U(1)xU(1) central fixture on Polaron, the
best measured two-GPU contiguous range is currently `cut325`: mean
`#LanczosMatvecS = 0.888566815s` over nine repeats.  The neighboring cuts
`cut323` and `cut326` measured `0.893969526s` and `0.893149359s` over nine
repeats, respectively, so treat this as a shallow placement basin around
`cut323` to `cut326` rather than a sharply universal split.  The one-GPU
baseline for the same fixture was about `0.963s`, so the best observed
two-GPU layout is only an approximately eight-percent improvement at this
problem size.

For this fixture, the graph-augmented device model is a materially better
runtime placement generator than the base two-device model.  With
`--layout-filter contiguous`, leave-one-layout-out tuning over the collected
contiguous rows selected an unclamped graph-feature fit at ridge `1e-3`
with RMSE about `0.0159s` and `R^2` about `0.947`.  The comparable base
16-feature fit selected ridge `1e-2` with RMSE about `0.0177s` and `R^2`
about `0.934`.  More importantly, the graph fit suggests `cut326`, adjacent
to the measured best, while the base fit suggests `cut365`, which measured
about `0.907s`.  Do not clamp coefficients for this benchmark: clamped
graph-feature fits fail validation because the correlated structural counters
need signed compensating weights.

A no-trace coefficient-file replay using the graph-augmented
`empirical-contiguous` runtime path selected `cut326` and measured mean
`#LanczosMatvecS = 0.892266440s` over nine repeats.  This is consistent with
the shallow `cut323` to `cut326` basin and proves the fitted graph feature
coefficient file can drive the C++ runtime to a measured near-best two-GPU
layout.

This replay result does not yet make the same coefficient file a safe live
DMRG default.  A live Hubbard `L=30, max_rank=512, sweeps=4` comparison on
Polaron used `UNI20_DMRG_PROFILE_SOLVER=1` and filtered the resulting
`MP_BENCHFILE` rows to `States >= 512`.  The default byte-balanced placement
measured `#LanczosMatvecS = 94.5838372s` over `1977` Hamiltonian applications,
or `0.0478421028s` per application.  The graph-augmented
`empirical-contiguous` policy using the Hubbard `L=40, m=5000` replay
coefficient file measured `#LanczosMatvecS = 158.236277s` over `1982`
applications, or `0.0798366683s` per application.  Treat replay-fitted
coefficients as fixture-local resident Lanczos policies until the live
effective-Hamiltonian shape family has its own validation.  Coefficient files
generated by `--output-runtime-coefficients` now carry a
`runtime_supported_output_blocks` guard, so this particular fixture-local file
will fall back to byte-balanced placement for the smaller live sweep shapes.
Raw `UNI20_TENSORCONTRACTION_RABC_EMPIRICAL_COEFFICIENTS` strings intentionally
remain unrestricted for diagnostics.

Unconstrained non-contiguous suggestions from the current linear benchmark fit
also require direct measurement.  Two top-ranked local-search candidates from
the full contiguous training set measured at roughly `0.214s` and `0.215s`,
despite much lower predicted times.  This reinforces that fitted coefficients
over correlated aggregate features are candidate generators, not a partitioning
proof.  Use `hypergraph-summary` and `bench-struct-summary` to inspect the
underlying `f` connectivity before promoting any non-contiguous layout.

To record final no-trace replay timings as a separate empirical dataset, save
the benchmark stdout or `MP_BENCHFILE` table and convert it to JSONL:

```bash
scripts/rabc-trace-model.py bench-record /tmp/uni20_rabc_default.out \
  --name default \
  --layout <layout-used-by-run> \
  --output /tmp/uni20_rabc_benchmark.jsonl
scripts/rabc-trace-model.py bench-record /tmp/uni20_rabc_candidate.out \
  --name candidate \
  --layout <layout-used-by-run> \
  --output /tmp/uni20_rabc_benchmark.jsonl \
  --append
scripts/rabc-trace-model.py bench-record /tmp/uni20_rabc_candidate.bench \
  --name candidate \
  --layout <layout-used-by-run> \
  --input-format benchfile \
  --output /tmp/uni20_rabc_benchmark.jsonl \
  --append
scripts/rabc-trace-model.py bench-summary /tmp/uni20_rabc_benchmark.jsonl
```

These `rabc_replay_benchmark` rows use the benchmark's printed `matvec=` field
and are deliberately separate from CUDA-event trace rows.  Use them for final
layout comparisons; use trace rows to explain or propose candidates.

For live DMRG `MP_BENCHFILE` tables, use `dmrg-summary` instead of the replay
benchmark commands.  It groups rows by kept-rank threshold and can also split
the comparison by half sweep:

```bash
scripts/rabc-trace-model.py dmrg-summary \
  /tmp/uni20_live_hubbard_l30_m512_default_profile.bench \
  /tmp/uni20_live_hubbard_l30_m512_empirical_profile.bench \
  --min-states 512 \
  --half-sweeps
```

This command requires `UNI20_DMRG_PROFILE_SOLVER=1` if
`#LanczosMatvecS` and per-application matvec timing are needed.  Without that
profile switch it can still summarize top-level solve, split, and environment
timings.

Use `bench-struct-summary` to join no-trace benchmark rows with static
term-trace features.  This is a diagnostic for correlated or misleading fitted
coefficients: for example, a layout can have low critical-path flops but still
be slow because peer `B` traffic, cut terms, or term imbalance dominate.

```bash
scripts/rabc-trace-model.py bench-struct-summary /tmp/uni20_rabc_benchmark.jsonl \
  --term-trace /tmp/uni20_rabc_term_trace.jsonl
scripts/rabc-trace-model.py bench-struct-summary /tmp/uni20_rabc_benchmark.jsonl \
  --term-trace /tmp/uni20_rabc_term_trace.jsonl \
  --sort right-flops \
  --compact-layouts
```

The structural summary prints timing rows together with critical-path flops,
peer-`B` traffic, cut terms, layout segments/transitions, output-byte skew, and
duplicated first-stage groups.  Sort by `peer-bytes`, `peer-blocks`,
`segments`, `transitions`, `right-duplicates`, or `mixed-duplicates` to inspect
why a segmented or otherwise fragmented layout loses even when the fitted model
predicts low flops.

The same benchmark rows can be fitted against static term-trace features.  This
uses the companion term trace for the fixed `f` hypergraph and reduces each
candidate layout to a critical-path feature vector by taking the maximum
per-device feature value.  This matches the resident replay target more closely
than fitting per-device CUDA-event timings when profiler instrumentation
perturbs the workload:

```bash
scripts/rabc-trace-model.py bench-fit /tmp/uni20_rabc_benchmark.jsonl \
  --term-trace /tmp/uni20_rabc_term_trace.jsonl \
  --graph-features
scripts/rabc-trace-model.py bench-validate /tmp/uni20_rabc_benchmark.jsonl \
  --term-trace /tmp/uni20_rabc_term_trace.jsonl \
  --graph-features
scripts/rabc-trace-model.py bench-tune /tmp/uni20_rabc_benchmark.jsonl \
  --term-trace /tmp/uni20_rabc_term_trace.jsonl \
  --graph-features \
  --ridges 1e-9,1e-7,1e-5,1e-3,1e-2,1e-1
scripts/rabc-trace-model.py bench-suggest /tmp/uni20_rabc_benchmark.jsonl \
  --term-trace /tmp/uni20_rabc_term_trace.jsonl \
  --graph-features \
  --ridge <selected-ridge> \
  --contiguous-only \
  --top 8
```

Use `--model device` when device identity matters for steady-state matvec
timing.  The anonymous critical-path model is unable to distinguish layouts
that put the dominant work on GPU 0 from layouts that put it on GPU 1.  On
Polaron this mattered for the Hubbard fixture: edge cuts with the same
anonymous structure measured differently on the two devices.  For
contiguous-placement studies, add
`--layout-filter contiguous` to `bench-fit`, `bench-validate`, `bench-tune`, and
`bench-suggest` so one round-robin or otherwise fragmented sample does not
dominate held-out validation for the ordered range family:

```bash
scripts/rabc-trace-model.py bench-validate /tmp/uni20_rabc_benchmark.jsonl \
  --term-trace /tmp/uni20_rabc_term_trace.jsonl \
  --graph-features \
  --model device \
  --layout-filter contiguous
scripts/rabc-trace-model.py bench-suggest /tmp/uni20_rabc_benchmark.jsonl \
  --term-trace /tmp/uni20_rabc_term_trace.jsonl \
  --model device \
  --layout-filter contiguous \
  --contiguous-only \
  --top 8 \
  --output-runtime-coefficients /tmp/uni20_rabc_empirical_coefficients.txt
```

Use `--graph-features` with `--output-runtime-coefficients` when the fitted
model should drive the graph-augmented runtime policy.  Omit it when writing a
base 16-value coefficient file.  In both cases, keep `--layout-filter
contiguous` for the current `empirical-contiguous` runtime policy unless the
runtime placement family has been extended to match the measured layouts.
Use `--append-runtime-coefficients` when adding another fitted shape to an
existing coefficient bundle:

```bash
scripts/rabc-trace-model.py bench-suggest /tmp/uni20_rabc_shape2_benchmark.jsonl \
  --term-trace /tmp/uni20_rabc_shape2_term_trace.jsonl \
  --model device \
  --layout-filter contiguous \
  --graph-features \
  --ridge <selected-ridge> \
  --contiguous-only \
  --output-runtime-coefficients /tmp/uni20_rabc_empirical_bundle.txt \
  --append-runtime-coefficients
```

The appended stanza carries its own `runtime_supported_output_blocks` guard, so
the runtime can select the first matching shape and fall back to byte-balanced
placement when the live DMRG solve has no fitted stanza.

Treat `bench-validate` as meaningful only after measuring several distinct
layouts.  With one default layout and one deliberately bad striped layout it is
a smoke test for ranking and command wiring, not evidence that the fitted model
generalizes.

Unlike trace `suggest`, benchmark `bench-suggest` does not clamp negative
coefficients by default.  The benchmark target is an end-to-end critical-path
timing, and correlated graph features can legitimately need compensating
coefficients in a small empirical fit.  Use `bench-tune --include-clamped` if
you want to test clamping explicitly, and prefer a ridge selected by
leave-one-layout-out validation before extrapolating to unmeasured layouts.
The tune output reports best-by-top1, best-by-RMSE, and best-by-R2 summaries;
do not choose a top1-only fit if its RMSE/R2 shows that the timing model is
numerically unstable.

For symmetry-local Hamiltonians, also test the constrained contiguous-range
family.  The traced sparse `f` tensor remains the ground truth for connectivity:
contiguous ranges are only a candidate restriction that is useful when the
block ordering clusters neighboring quantum-number sectors.

`bench-suggest --segmented-only` is a controlled next step between one
contiguous cut and arbitrary per-block local search.  It enumerates alternating
two-device segmented layouts with bounded cut positions:

```bash
scripts/rabc-trace-model.py bench-suggest /tmp/uni20_rabc_benchmark.jsonl \
  --term-trace /tmp/uni20_rabc_term_trace.jsonl \
  --model device \
  --segmented-only \
  --max-segments 3 \
  --segment-cut-stride 20 \
  --top 8 \
  --compact-layouts \
  --show-structure
```

By default, segmented search only keeps candidates whose layout-shape features
were observed in benchmark rows from the same bounded segmented candidate
family.  Contiguous cuts, alternating layouts, and segmented layouts outside the
requested `--max-segments`/`--segment-cut-stride` family do not widen the
support envelope, and each candidate is checked against measured rows with the
same segment count.  This prevents a model trained mostly on contiguous cuts
from assigning unsupported negative or otherwise nonsensical scores to
three-segment layouts.  Use `--allow-shape-extrapolation` only as an explicit
diagnostic, then benchmark any suggested segmented layout directly before
drawing conclusions.

Use `--show-structure` when inspecting candidates from a fitted model.  It adds
graph-derived counters for each ranked candidate, including right-first and
mixed critical-path flops, peer-`B` traffic, cut-term counts, maximum per-device
term pressure, unique `(B, C)` pressure, output-byte skew, and duplicate-group
counts.  These columns make it easier to spot candidates whose fitted scalar
score is attractive only because the regression has assigned compensating
weights to correlated communication or load-skew features.

`--candidate-score monotonic-structure` replaces the candidate ranking score
with a non-negative fit over the same structural counters.  Each counter is
centered at its training-set minimum, so the fit penalizes excess structural
cost rather than the absolute value of a counter.  This is a diagnostic step
toward a calibrated graph cost function: communication, term-pressure, skew,
and duplication counters cannot receive negative weights.  On the current
Hubbard `L=40, m=5000` dataset it is intentionally conservative and does not
recover the best contiguous basin, so do not use it as a runtime policy.  Its
main value is checking whether a proposed layout still looks good when obvious
structural costs are forced to be monotonic penalties.

Validate the monotonic structural score before using it to rank a new candidate
family:

```bash
scripts/rabc-trace-model.py bench-validate /tmp/uni20_rabc_benchmark.jsonl \
  --term-trace /tmp/uni20_rabc_term_trace.jsonl \
  --model device \
  --graph-features \
  --candidate-score monotonic-structure \
  --structure-feature-set all \
  --compact-layouts
```

This performs leave-one-layout-out validation with the same structural counters
used by `bench-suggest --candidate-score monotonic-structure`.  A poor top-1
match or poor `R^2` means the monotonic model is still only a diagnostic for
that benchmark family; benchmark suggested layouts directly before interpreting
them as improvements.

Use `--structure-feature-set` to restrict the monotonic fit to a more
interpretable subset of counters.  `all` keeps every structural counter.
`execution-pressure` focuses on mixed-order critical-path flops, term pressure,
unique `(B, C)` pressure, peer-`B` traffic, and duplicate mixed groups.
`launch-pressure` removes flop counters and focuses on term/group counts.
`no-output` removes output-slab size and output-skew counters.
`typed-hypergraph` adds weighted split counters for `B` fanout, direct `(R,B)`
edges, and right/left first-stage reuse hyperedges.  These subsets are meant to
test hypotheses about the graph cost function; they are not separate runtime
policies.  With `--show-structure`, monotonic scoring also prints a
`score_feature_rank` table containing exactly the selected feature columns.

Current Hubbard `L=40, m=5000` validation confirms that limitation.  With the
mixed contiguous/segmented benchmark rows, leave-one-layout-out validation gives
`R^2 = 0.172333708`, `RMSE = 0.0649954413 s`, and a top-1 mismatch.  Restricting
the same check to contiguous layouts gives `R^2 = 0.401401351`,
`RMSE = 0.0534626698 s`, and still a top-1 mismatch.  The best observed
contiguous basin remains around cuts `323`-`326`, while the structural score
overpredicts those rows by about `0.03`-`0.04 s`; this is useful evidence that the
next step is a better graph/hypergraph cost function rather than a stronger
monotonic least-squares fit over the current counters.

The narrower structural subsets show the same conclusion on the Hubbard
fixture.  `execution-pressure` gives contiguous-layout validation
`R^2 = 0.399415376`, `RMSE = 0.0535512831 s`, with a top-1 mismatch, and
mixed-layout validation `R^2 = 0.152480984`, `RMSE = 0.0657703251 s`.
`launch-pressure` and `no-output` are similar on the contiguous subset
(`R^2` about `0.40`, still top-1 mismatch).  This shows that simply dropping
output-skew counters or emphasizing launch pressure does not recover the best
cut; the next model needs typed hypergraph structure beyond aggregate counters.
The first `typed-hypergraph` feature set is better on the small overhead
fixture (`R^2 = 0.832593018`, top-1 match), but on the Hubbard `L=40, m=5000`
fixture it remains similar to the other monotonic structural models:
contiguous-layout validation gives `R^2 = 0.396808955`,
`RMSE = 0.0536673584 s`, with a top-1 mismatch, and mixed-layout validation
gives `R^2 = 0.152373035`, `RMSE = 0.0657745136 s`.  It ranks the contiguous
`cut358` basin, not the measured `cut323`-`cut326` basin.

```bash
scripts/rabc-trace-model.py layouts --block-count 42 --device-count 2 --contiguous-cuts
scripts/rabc-trace-model.py suggest /tmp/uni20_rabc_trace.jsonl \
  --drop-first-per-layout=1 \
  --contiguous-only
```

The fitting commands default to `--timing-objective=steady-state`, which matches the resident Lanczos comparison:
`A` and `C` environment byte features are still reported in trace rows, but they are ignored as transfer regressors
because environment staging is expected to happen before the repeated Krylov matvec timing.  Use
`--timing-objective=cold-start` only when fitting setup/materialization traces, and do not combine that objective with
`--drop-first-per-layout=1` unless the remaining rows really include environment staging.

Short replay sweeps usually include one slow setup row per process/layout before the steady-state matvec timing.  Use
`--drop-first-per-layout=1` for exploratory fits when each layout was measured with at least two trace rows:

```bash
scripts/rabc-trace-model.py fit /tmp/uni20_rabc_trace.jsonl --drop-first-per-layout=1
scripts/rabc-trace-model.py suggest /tmp/uni20_rabc_trace.jsonl --drop-first-per-layout=1
```

For longer sweeps, group repeated rows by layout before comparing timings:

```bash
scripts/rabc-trace-model.py summary /tmp/uni20_rabc_trace.jsonl --drop-first-per-layout=1 --group-layouts
```

Use leave-one-layout-out validation before trusting a fitted model for unseen layouts:

```bash
scripts/rabc-trace-model.py validate /tmp/uni20_rabc_trace.jsonl --drop-first-per-layout=1
```

The validation output reports held-out layout error and whether the predicted best layout matches the measured best
layout.  A high in-sample `fit` score is not enough to justify `suggest`; the validation score is the relevant
diagnostic for extrapolating to new layouts.

Scan ridge parameters with the same held-out validation before choosing the model used for extrapolation:

```bash
scripts/rabc-trace-model.py tune /tmp/uni20_rabc_trace.jsonl --drop-first-per-layout=1
```

The `tune` subcommand ranks models by held-out top-1 layout match, then held-out `r2`, then held-out RMSE.  Use the
reported ridge explicitly when running `fit`, `validate`, or `suggest`.
To make a tuned model reproducible, export it and then use the saved JSON for
layout suggestions:

```bash
scripts/rabc-trace-model.py fit /tmp/uni20_rabc_trace.jsonl \
  --drop-first-per-layout=1 \
  --graph-features \
  --ridge <validated-ridge> \
  --output-model /tmp/uni20_rabc_model.json
scripts/rabc-trace-model.py suggest /tmp/uni20_rabc_trace.jsonl \
  --model /tmp/uni20_rabc_model.json \
  --contiguous-only \
  --top 8
```

The model file records the timing objective, feature set, ridge, clamp policy,
feature names, coefficients, and fit statistics.  `suggest --model` uses those
coefficients directly; it does not refit the trace or apply `--allow-negative`.
Suggestion output labels whether each ranked layout was observed in the trace,
is an ordered contiguous range split, and matches the byte-balanced default.
Treat `observed=false` layouts as hypotheses only: benchmark them before using
them to update runtime policy, especially when local search is extrapolating
from a trace that only measured contiguous layouts.

Until the held-out validation set is broad enough to trust extrapolated layouts, keep suggestions inside the measured
layout set:

```bash
scripts/rabc-trace-model.py suggest /tmp/uni20_rabc_trace.jsonl \
  --drop-first-per-layout=1 \
  --ridge <validated-ridge> \
  --allow-negative \
  --observed-only
```

To generate a few explicit layouts without fitting:

```bash
scripts/rabc-trace-model.py layouts --block-count 40 --device-count 2 --random 8
```

The fixture preserves the exact TensorContraction block worklist emitted by the symmetry-aware DMRG path, but it does
not store the higher-level `LocalSpace`, `BlockSpace`, or MPO metadata. Do not feed fixture data back into U(1) MPS
state; use it only as a terminal benchmark artifact.

For synthetic fixtures, initialize the center vector with non-zero random data before capture or upload.  A zero center
vector makes every Hamiltonian application zero and can hide movement, accumulation, or validity bugs in the resident
path.
