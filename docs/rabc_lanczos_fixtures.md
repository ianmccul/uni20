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
  ./build_codex/tensorcontraction-polaron-release-fresh/examples/spin_half_heisenberg_u1_dmrg
```

The fixture is a compact local binary format. It is not intended as a portable archival wavefunction format.

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

For symmetry-local Hamiltonians, also test the constrained contiguous-range
family.  The traced sparse `f` tensor remains the ground truth for connectivity:
contiguous ranges are only a candidate restriction that is useful when the
block ordering clusters neighboring quantum-number sectors.

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
  --contiguous-only
```

The model file records the timing objective, feature set, ridge, clamp policy,
feature names, coefficients, and fit statistics.  `suggest --model` uses those
coefficients directly; it does not refit the trace or apply `--allow-negative`.

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
