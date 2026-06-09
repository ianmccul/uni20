# TensorContraction Integration: Findings and Lessons

This note consolidates the empirical findings, calibration methodology, and
operational gotchas from the temporary, quarantined **TensorContraction** bridge
integration (on the `tensorcontraction-integration` branch). The *design*
conclusions it informed live in the execution-architecture notes; this note keeps
the measurements and lessons immediately accessible without digging through that
branch.

Related notes:

- `docs/execution_architecture.md` — the design this exercise informed.
- `docs/block_sparse_tensor.md`, `docs/block_coalescing.md` — tensor + coalescing design.
- `docs/rabc_contraction_scheduling.md` — the R/A/B/C apply and its cost model.
- `docs/ordering_and_backend_lowering.md` — ordering ownership and lowering.
- `docs/rabc_lanczos_fixtures.md` — replay fixture format.

## Purpose and status

The branch vendored an external TensorContraction prototype (gated by
`UNI20_ENABLE_TENSORCONTRACTION`) to get a minimal real-valued U(1) DMRG
Hamiltonian-apply working on GPU via the R/A/B/C model (`R_i += α·A_j·B_k·C_l`;
B = input center vector, R = output, A/C = environments) and to **inform uni20 core
design**. It deliberately bypasses the real async runtime (it has its own
deterministic right-first scheduler) and will be retired once a core GPU device
backend exists. The GPU placement optimization workflow has served its informing
purpose; the conclusions below are the durable output.

## Cost model

The resident apply uses a right-first bridge: `Y = B·C` on the owner of input block
`b`, then `R += A·Y` on the owner of output block `r`. The placement cost model is
an **input-anchored proxy**: per-device additive load (combined GEMM flops,
peer-copy bytes, kernel launches) plus a segment-transition penalty, reduced by a
critical-path `max` over devices.

The right-first (right-dominant) order is a deterministic *simplification*, not the
original intent of the R/A/B/C form. Choosing a strictly right-first or left-first
order makes each term decompose into one of two pairwise contraction sets
(right-first `Y = B·C` then `A·Y`; left-first `X = A·B` then `X·C`). The actual
benefit of keeping the four-way R/A/B/C form is the possibility of **splitting
blocks between the left and right paths** per term — routing some blocks right-first
and others left-first — which the cost model could exploit. That mixed-path regime
remains mostly untested so far; the resident bridge currently uses the uniform
right-first path. Its single most useful property: the
**flop/launch/byte crossover is one knob behind three decisions** — placement
(which device), coalescing (merge blocks or not), and lowering granularity (stream
chain vs CPU batch vs graph).

## Calibrated constants

Untraced fit, GV100 (sm_70), m4608 Heisenberg U(1), 18 diverse layouts. Seed the
core planner's cost model with these:

| constant | value | note |
|---|---|---|
| effective throughput | ~57 TFLOP/s | proxy `gflops`; cross-checks an independent cut-only fit (~52) |
| kernel launch | ~5 µs | the traced fit's 29 µs was mostly event overhead |
| segment transition | ~166 µs | real; ~30 % of the traced 242 µs was a tracing artifact |
| GPU↔GPU peer | ~28 GB/s | NVLink |
| CPU↔GPU | PCIe ~12 GB/s, asymmetric | separate device-class link constant; not measured here |

## Model sophistication is not the lever

A 4-parameter input-anchored proxy ranks placements about as well as elaborate
many-feature models: LOO Kendall-τ ≈ 0.71–0.79 with ~3 % top-1 regret for the
proxy, a typed-hypergraph structural model, and a mono-structure model alike. More
features **overfit** at the available sample size (~18 layouts). The predictive win
came from *diverse calibration data*, not model complexity — so do not invest in
fancier cost models; invest in better data.

## Calibration methodology (the most reusable lesson)

1. **Calibrate on diverse (non-cut) layouts.** Cut-only data hides the
   segment/transition signal (all contiguous cuts have ~2 segments → zero variance),
   which pins LOO τ at ~0.49 and forces all variance into a bogus high flop rate.
   Diverse (random/scattered/proxy-suggested) layouts lift τ to ~0.71–0.79 and
   surface the real transition penalty.
2. **Time untraced; trace once for connectivity.** Term/event-timing tracing
   inflates matvec time ~5× (measured A/B, same layout/session: 0.149 s untraced vs
   0.764 s traced). The overhead is per-kernel-launch, so it scales with segment
   count and can be misattributed to the transition term. Term connectivity is
   layout-independent, so trace a single run for structure and time the timing
   sweep untraced. (Same family of effect as the Nsight `--cuda-event-trace=false`
   rule in `AGENTS.md` §5.)
3. **The "cut" (contiguous) family is an implementation artifact** of BlockSpace
   ordering, not a meaningful placement family. It happens to contain the optimum
   for Heisenberg U(1) (block order ≈ size ≈ connectivity) but does not generalize
   (e.g. Hubbard U(1)×U(1), with a long-range `|r−b|` tail).

## The small-block/high-connectivity tail

A large fraction of the contraction is tiny launch-bound GEMMs:

| problem | tail blocks | % of launches | % of flops | % of bytes |
|---|---|---|---|---|
| Heisenberg m4608 | 16/42 | ~35 % | ~0.03 % | ~0.4 % |
| Hubbard m5000 | 344/730 | ~45 % | ~0.05 % | ~1.2 % |

Tail-term environments (A/C) are also KB-scale. Critically, as bond dimension `m`
grows the tail's *flop* fraction → 0 but its *launch count* stays ~45 % — a
**permanent GPU worst case** (launch-bound). The remedies (single-axis coalescing
at the source, CPU offload, batched/graph submission) all attack per-op overhead;
the CPU is a required device regardless, so tail placement falls out of the
heterogeneous cost model with no bespoke code.

## Placement findings

Best/worst per-matvec spread is only ~1.5×, and the optimum on Heisenberg is
**near-single-device** — a 2-GPU split barely pays because peer-copy plus transition
overhead roughly cancels the parallelism gain. So GPU-only placement headroom is
small for this problem. Hubbard (larger tail) may reward spreading more; not yet
measured at real scale (m5000/m8000 fixtures exist).

## CUDA graphs

CUDA graphs exist in 12.9 (including conditional nodes), but amortization for DMRG
is limited: a Lanczos solve runs only ~4 matvecs per site (not the benchmark's 24),
the environment pass runs once, and the per-site block structure differs so the
graph must be re-instantiated per site. The only viable form is a **per-site graph
cache reused across sweeps** (parameter update once the block structure stabilizes),
and only if a cheap instantiation-cost microbench shows it pays. Amortization-free
levers (CPU offload, coalescing, host-submission parallelism) help every invocation
— including the once-run environment pass — so they are the more robust lever.

## Reproduction

- Sweep driver: `scripts/run-rabc-layout-sweep.sh` (manual per-block layouts).
- Cost-model tooling: `scripts/rabc-trace-model.py` (`proxy-calibrate`,
  `bench-validate`, `layouts`); tests in `scripts/test_rabc_trace_model.py`.
- Fixture format: `docs/rabc_lanczos_fixtures.md`.
- Calibrate: `rabc-trace-model.py proxy-calibrate <benchmarks.jsonl> --term-trace
  <trace.jsonl>`, pairing a sweep with a **matching block-count** term trace, on
  **diverse** layouts timed **untraced**.

## Sharing benchmark structure with collaborators

The contraction *performance* problem is fully specified by the f-hypergraph (which
`(r,a,b,c)` terms exist, with coefficients) plus the per-block dimensions; the
matrix-element values do not affect contraction timing, data movement, or placement.
The term trace (`*_term_trace.jsonl`) is exactly this value-free representation —
`r,a,b,c,coefficient`, every block `*_rows/*_cols`, and the derived `bc_flops`,
`accumulate_flops`, and `intermediate_bytes`. So the artifact to share is the trace
`.jsonl` itself plus a small standalone reader that loads it into data structures;
collaborators run their own analysis/partitioning code rather than this branch's
(heavily modified) tooling. A documented JSONL schema and a minimal header-only
C++ reader (nlohmann/json) live on the `tensorcontraction-integration` branch under
`tools/rabc_trace_reader/` (and a standalone repo), so collaborators can load the
value-free structure without depending on the prototype's tooling.

**What to generate — typical iterations, not whole runs.** The representative unit is
the **central-bond iteration**: maximal bond dimension, largest contraction, most
terms. Edge bonds are boundary-truncated and unrepresentative, and the rest of a run
just interpolates. The suite is a grid `{U(1) Heisenberg, U(1)×U(1) Hubbard} ×
{bond-dimension ladder up to the maximum constructible}`, dumping the central-bond
term trace per point. Lattice size and sweep count are tuned per target `m` so the
central bond actually reaches that dimension (the sectors must be able to hold it,
roughly `d^{L/2} ≥ m`, with enough sweeps; running at `max_rank = m` to convergence
lets the centre saturate). The existing `uni20_l40_m<m>_central` fixtures are
instances of this grid.

**Connectivity is analytic; block dimensions are physics.** The connectivity (which
terms exist) follows combinatorially from the model and the selection rule, so it can
be generated at any `m`. The *block dimensions* cannot be freely synthesized: the
distribution of bond dimension across charge sectors is set by the entanglement
structure of the ground state (the Schmidt/entanglement spectrum), which depends on
the model, couplings, filling, and bipartition. Representative dimensions must
therefore be **extracted** from real central-bond dumps; reaching `m` beyond what can
be constructed would require modeling the entanglement-spectrum scaling (e.g. CFT for
critical chains), which is a research problem, not a free knob.

## Conclusion

The GPU placement optimization workflow is **done informing**. The constants above
seed the core planner's cost model; the methodology (diverse data, untraced timing,
trace-once) transfers to validating the core apply. Next development proceeds on the
core async-DAG apply (CPU kernels first), per the build order in
`docs/execution_architecture.md`.
