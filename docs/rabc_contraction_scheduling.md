# R/A/B/C Contraction Scheduling

This note records the intended Uni20 replacement for the temporary
TensorContraction `Arranger`/`Swapper` scheduling model.

The Hamiltonian apply used by two-site DMRG has terms of the form:

```text
R_i += alpha_t * A_j * B_k * C_l
```

where:

- `B` is the input two-site center vector.
- `R` is the output two-site center vector.
- `B` and `R` have the same block spaces and the same block layout.
- `A` is the left environment.
- `C` is the right environment.

The important scheduling problem is not only the local matrix multiply order.
It is deciding where each block lives, which intermediates are worth
materializing, and whether communication should happen before or after a local
contraction.

## Current Bridge Behavior

The resident `EffectiveHamiltonianOperator` now bypasses the legacy
`Arranger::analyzeComputation` / `compileWorklists` / `doContraction` path by
default.

The current deterministic bridge still uses a right-first strategy:

```text
Y = B * C
R += A * Y
```

The executor:

- uses each resident `R` block's current device as the owner for work that
  contributes to that block;
- initializes each `R` block with `beta = 0` on the first contributing GEMM and
  uses `beta = 1` for later accumulation;
- warns on `stderr` and zeros only `R` blocks that receive no contributions;
- stages `A`, `B`, and `C` blocks to the `R` owner device when needed;
- reuses `B*C` intermediates keyed by `(device, B block, C block)`;
- accumulates `A*(B*C)` into `R`;
- frees temporary intermediates through the resident buffer dependency tracker.

This is not the final Uni20 storage model. It still borrows TensorContraction
buffer bookkeeping while replacing the R/A/B/C worklist scheduling seam.  The
resident matvec path no longer exposes the old worklist planner as a runtime
selection; profiling and cost-model work should use the deterministic bridge.

## Legacy TensorContraction Behavior

The legacy `Arranger` does not compare left-first and right-first plans.

It effectively implements a right-first strategy:

```text
intermediate = B * C
R += A * intermediate
```

The code detects repeated `(B, C)` pairs and materializes reusable
intermediates. It can also combine intermediates for repeated `(R, A)` groups
before the final `A * intermediate` multiply.

This was enough to exercise CUDA/NCCL kernels, but it is not a robust Uni20
design. In particular:

- block placement is not a first-class part of tensor construction;
- blocks may be allocated first and rearranged later;
- the planner does not score the alternative `A * B` route;
- communication is an after-the-fact consequence of the chosen placement;
- scheduling decisions are tied to TensorContraction `Matrix`/`MatrixFamily`
  handles rather than Uni20 tensor storage.

## Placement-First Model

Uni20 should allocate blocks according to the `BlockSpace` and `LocalSpace`
metadata before constructing a contraction schedule.

Each active dense block should have explicit placement:

```text
block id
  -> MPI rank
  -> CUDA device
  -> device buffer
  -> GPU epoch/dependency state
```

For the first U(1) DMRG workflow, the block index comes from the bond
`BlockSpace` and the local physical `LocalSpace`. A placement policy maps those
block indexes to devices/ranks deterministically.

This makes placement part of the logical tensor layout, not a rearrangement pass.

## First Planner

The first serious planner should score two global candidate plans:

```text
left-first:
  X = A * B
  R += X * C

right-first:
  Y = B * C
  R += A * Y
```

For this first version, choose one of these two plan families for the apply.
Do not yet mix left-first and right-first on different blocks. A later planner
can make that choice per `R` block or per group of terms.

The cost model should include:

- GEMM flops for the first multiply.
- GEMM flops for the second multiply.
- intermediate size in bytes;
- number of terms that reuse the same intermediate;
- whether `A`, `B`, `C`, and `R` blocks are local or remote;
- communication bytes if an input must be staged remotely;
- communication bytes if a partial `R` must be sent after local contraction;
- temporary memory pressure and intermediate lifetime.

For a term `A * B * C`, the local dense costs are:

```text
left-first flops  ~= rows(A) * cols(B) * cols(A)
                  + rows(A) * cols(C) * cols(B)

right-first flops ~= rows(B) * cols(C) * cols(B)
                  + rows(A) * cols(C) * cols(A)
```

The actual planner should use exact block dimensions from the term.

## Communication Choices

There are at least two valid communication strategies.

The first prototype can place environment blocks where the input lives:

```text
place A/C near B
compute local contribution
communicate partial R to the R owner
accumulate on the R owner
```

This is simple and keeps most local Hamiltonian work near the active Krylov
vector block. It also makes the final communication explicit: after contraction,
send partial `R` blocks to their owning rank/device.

The alternative is target-owned contraction:

```text
place or stage A/B/C on the R owner
compute directly into R
```

This avoids communicating partial outputs but may require more input staging.

A future planner should score both choices. It may choose a mixture, especially
when environment blocks are much smaller than `B`/`R` blocks or when one side of
the environment has high reuse.

## Environment Placement

The left and right environments are cold host/cache tensors when they are not
active. During one local solve, active environment blocks should be materialized
on GPU according to a placement policy.

For the first placement-first implementation:

- assign `B` and `R` blocks from the two-site center `BlockSpace`;
- materialize `A` and `C` environment blocks on the same rank/device as the
  corresponding input-side `B` blocks when possible;
- compute local partial outputs;
- communicate and accumulate into the owner of each `R` block.

This deliberately favors a simple first implementation over a globally optimal
communication schedule.

The temporary TensorContraction bridge currently implements only the first
local-device placement slice of this idea for resident vector algebra:
complete `MatrixFamily` storage is split into contiguous byte-balanced ranges,
with one coalesced device slab per active local GPU.  That policy removes the
worst per-block `Arranger`/`Swapper` overhead in forced all-GPU diagnostics, but
it is not yet keyed by `BlockSpace`, environment ownership, or R/A/B/C term
structure.

The resident bridge now keeps only a narrow explicit placement hook while the
real planner is being designed.  With
`UNI20_TENSORCONTRACTION_RABC_PLACEMENT=manual`, Hamiltonian matvecs use
`UNI20_TENSORCONTRACTION_RABC_PLACEMENT_LAYOUT` as the shared owner layout for
the matching `B` and `R` center spaces.  Environment-style contractions may use
separate `UNI20_TENSORCONTRACTION_RABC_B_LAYOUT` and
`UNI20_TENSORCONTRACTION_RABC_R_LAYOUT` lists because `B` and `R` are different
spaces.  The executor forms each right-first `B_b C_c` product on owner(`B_b`)
and then accumulates `A_a (B_b C_c)` on owner(`R_r`).

The older `cost`, `cost-block`, `stripe`, and `empirical-contiguous` runtime
paths were ordered-range or replay-fit diagnostics.  They should not be used as
the implementation target: an ordered range only refers to the current block
construction order, while the real objective is the sparse `f` hypergraph.

## Optimization Formulation

The placement problem is a graph/hypergraph partitioning problem with execution
choices attached to each hyperedge.  The sparse TensorContraction `f` tensor is
the ground-truth connectivity object.  Its nonzero entries are terms of the
form

```text
t = (R_r, A_a, B_b, C_c, alpha)
```

Each term connects one output block, three input blocks, and one scalar
coefficient.  Symmetry and fusion rules explain why this sparse `f` tensor has
locality, but they should not replace the `f` tensor in the planner.  For the
U(1) Heisenberg Hamiltonian, local terms change charge by `0`, `+1`, or `-1`.
Since the total Hamiltonian is a scalar, a charge-changing left/environment
operation must be compensated by the opposite change on the other side.  This
clusters strongly connected center blocks in neighboring quantum-number
sectors.  The empirical placement model should therefore read locality from
the actual `f` hypergraph.  A one-dimensional ordered-range search is only a
legacy diagnostic for checking whether the current construction order happens
to cluster connected sectors; it is not a general placement domain.

Use the fixture trace tooling to inspect this connectivity directly:

```bash
scripts/rabc-trace-model.py hypergraph-summary /tmp/uni20_rabc_term_trace.jsonl
```

The summary reports center-block fanout from each `B_b`, direct `(R_r, B_b)`
connectivity, right-first reuse hyperedges keyed by `(B_b, C_c)`, and
left-first reuse hyperedges keyed by `(A_a, B_b)`.  These are the graph objects
that should feed a real partitioner.  A contiguous `cutK` layout is only a
restricted diagnostic over the current block ordering; it is not the general
model, especially once multiple symmetries or less one-dimensional fusion
structure are present. More strongly, contiguity is a storage property, not a
logical invariant: the bond and center block spaces can be permuted, and any
chosen partition can be packed into contiguous per-device slabs after the
placement decision.

The LaTeX design note `docs/latex/rabc_hypergraph_partitioning.tex` gives the
restricted Hamiltonian-domain formulation in terms of a center owner map,
exact left/right temporary factorizations of the sparse `f` tensor, temporary
construction devices, and second-stage execution devices. Under explicit
canonical rules, those variables induce the B-block copies, temporary
migrations, per-device partial outputs, and final reductions. The same note
also records the environment-construction variant, where input and output block
owners do not need to coincide.

The more implementation-oriented note
`docs/latex/rabc_input_anchored_model.tex` records the first serious restricted
model: `B` and `R` share one layout `h`, the first GEMM for a term runs on
`h(b)`, the second GEMM runs on `h(r)`, and the only remaining term-level
choice is left-first versus right-first. That anchoring makes the temporary
copy schedule and environment staging costs deterministic for a chosen layout
and path assignment.

The U(1) Heisenberg chain is useful for low-overhead benchmarking, but it is a
weak stress test for the placement model because the dominant sector
connectivity is close to one-dimensional in the chosen block ordering.  The
nearest-neighbor Fermi-Hubbard chain with U(1)xU(1) symmetry is the next
reference benchmark: center blocks are labelled by two charges, hopping terms
move through distinct up/down charge directions, and the sparse `f` tensor is a
more faithful proxy for the multi-charge hypergraph that the final planner must
partition.  The default byte-balanced layout or a legacy ordered-range
diagnostic may still win for small fixtures, but that should be interpreted as
an empirical result for the measured `f` tensor, not as evidence that a
one-dimensional ordering is the general optimization model.

In the final Uni20 model, decision variables should include:

- block placement `p_F(i) = (rank, device)` for each block `i` in each family
  `F in {A, B, C, R}`;
- multiplication order `o_t in {left-first, right-first}` for each term or term
  group;
- execution location `e_g` for each materialized intermediate group, for
  example `(B_b, C_c)` in a right-first plan;
- reduction location for partial `R_r` contributions when work is computed away
  from the final `R` owner;
- whether to replicate selected environment or center blocks when replication
  is cheaper than repeated communication.

The objective should minimize the critical path, not the sum of all work:

```text
minimize max_device load(device)
       + memory_pressure_penalty
       + layout_mismatch_penalty
```

where `load(device)` contains:

- GEMM time, including a launch/handle overhead term for tiny blocks;
- device-to-device, host-to-device, or rank-to-rank transfer time, including a
  per-copy overhead term;
- allocation/scratch overhead for materialized intermediates;
- reduction or accumulation cost for partial outputs;
- optional setup terms for materializing static environments.

For the current resident Lanczos fixture benchmark, setup terms are deliberately
excluded from the primary objective.  The static `A` and `C` environments should
be placed before timing the Krylov loop.  The timed objective is the repeated
matrix-free matvec plus the vector-algebra constraints it imposes.

The fully generic term workflow is richer than a single target-owned GEMM pair.
For each logical contribution, the planner may need to decide:

1. Move the center block, if the first contraction should run on another device.
2. Apply either the left environment or the right environment first.
3. Accumulate reusable intermediate tensors, when several terms share the same
   first-stage product.
4. Move the intermediate, if the second contraction is cheaper elsewhere, and
   optionally accumulate common intermediates on the new device.
5. Apply the remaining environment tensor.
6. Move the partial result, if necessary, to the canonical owner of `R_r` and
   accumulate into the output block.

The multiplication order is therefore a per-term or per-group decision, not a
global flag.  Some moves in this chain may be unnecessary for a specific
geometry or cost model, but the planner should include them as possible edges so
benchmark feedback can decide which ones are worth scheduling.

The interesting mixed-order case is when different connected regions of the
same sparse `f` hypergraph prefer different first-stage contractions.  The
planner should eventually compare at least these grouped choices:

- right-first groups keyed by `(B_b, C_c)`, computing `Y = B_b * C_c`;
- left-first groups keyed by `(A_a, B_b)`, computing `X = A_a * B_b`;
- per-group execution location and any transfer needed before reducing into
  the canonical `R_r` owner.

This is where symmetry locality enters constructively.  A scalar Hamiltonian
term can move charge on one side only if the opposite side compensates it, so
the useful groups are determined by the nonzero `f` entries and the fusion
algebra that generated them.  The final planner should score mixed left/right
groups from that graph rather than choosing one global order for every block.

The local rectangular-block heuristic is simpler: for an isolated term, multiply
through the long dimension of the center block `B` first.  If `B` is tall, the
left-first path `A * B` contracts the long row dimension; if `B` is wide, the
right-first path `B * C` contracts the long column dimension.  This is only a
local heuristic.  Reuse encoded in the sparse `f` tensor, communication costs,
and shared intermediates can override it, especially when many neighboring
sectors share the same `(A, B)` or `(B, C)` group.

The right-first, target-owned prototype is the reduced problem:

```text
choose p(i) for center-vector blocks B_i/R_i
run every term contributing to R_r on device p(r)
materialize one Y_{d,b,c} = B_b * C_c per device d and pair (b,c)
accumulate R_r += A_a * Y_{d,b,c}
```

Its per-device model is:

```text
load(d) =
  sum_unique_(b,c used on d) cost_gemm(B_b, C_c)
  + sum_terms_with_p(r)=d cost_gemm(A_a, Y_{d,b,c})
  + sum_unique_b_needed_on_d_and_p(b)!=d cost_transfer(B_b)
  + launch/copy/allocation overheads
```

The former `cost`, `cost-block`, `stripe`, and `empirical-contiguous` runtime
diagnostics have been removed from the C++ bridge.  They were useful for
learning, but they were runtime policies over ordered block ranges or replay-fit
coefficients rather than solvers for the graph problem above.  Current
experiments should generate candidate layouts offline and replay them as explicit
`manual` layouts.  The operator still applies one canonical center-vector layout
to both Krylov input and output blocks; if a contraction produces a block away
from that canonical owner, the generic planner must account for the relayout
before the result becomes the next Lanczos input.

## Current Empirical Status

Replay benchmarks still provide useful evidence even though the automatic
runtime policies have been removed.  On the local Hubbard `L=40, m=5000`
U(1)xU(1) fixture, a graph-augmented fit over ordered splits selects the
measured-good basin historically labelled `cut323` to `cut326`, giving a real
two-GPU speedup over the one-GPU baseline.  This validates the tracing and
replay loop, and it shows that graph-derived features are the right source of
placement information.

The same evidence does not yet justify using an unconstrained non-contiguous
layout as runtime policy.  Direct replays of segmented candidates have not
beaten the contiguous basin.  The better segmented candidates reduce the
right-first or mixed critical-path flops, but they increase peer `B` traffic and
skew terms and unique `(B, C)` groups onto one device.  Unconstrained linear
fits can also assign negative weights to correlated communication features, so
they are useful for diagnostics and candidate generation but are not a
principled graph optimizer.

The next planner target should therefore be a calibrated graph/hypergraph cost
function, not simply a larger feature vector.  The cost function should use the
nonzero `f` tensor to construct typed edges for compute, communication,
duplication, and reduction pressure.  Benchmark data should calibrate the
weights and overhead terms, while the structural model should preserve
monotonic penalties for obviously costly events such as peer traffic,
duplicated first-stage groups, output-layout mismatch, and severe device-load
skew.  Candidate layouts from that cost function must still be replayed before
being promoted to a runtime policy.

Current monotonic structural diagnostics support this conclusion.  Narrow
feature subsets such as execution-pressure, launch-pressure, and no-output
produce similar leave-one-layout-out errors and still miss the best observed
Hubbard ordered-range basin.  This suggests that the missing information is not
just which aggregate counters are included, but how typed hyperedges share
work, traffic, and launch pressure across candidate partitions.

A first typed-hypergraph diagnostic adds weighted split counters for `B`
fanout, direct `(R,B)` edges, and right/left first-stage reuse hyperedges.  It
improves the small overhead fixture, but still misses the Hubbard `L=40,
m=5000` best ordered-range basin.  The next model therefore needs a more
explicit execution-state cost over typed hyperedges, not just monotonic
penalties on typed split summaries.

Live DMRG validation also shows that fixture-local replay fits are not portable
enough to use as a default sweep policy.  A graph-augmented fit that selects the
near-best historical `cut326` layout for the Hubbard `L=40, m=5000` central
replay fixture loses badly on a live `L=30, max_rank=512` sweep: after filtering
to `States >= 512`, matvec time rose from `0.0478421028s` to `0.0798366683s` per
Hamiltonian application.  The planner therefore needs either live-shape training
data or a structural execution model that generalizes across the changing
effective-Hamiltonian block graphs seen during a sweep.

## Long-Term Planner

The long-term scheduler should operate on a term graph:

```text
term t:
  output R_i
  inputs A_j, B_k, C_l
  coefficient alpha_t
```

Equivalently, the sparse `f` tensor defines a finite-state hypergraph
partitioning problem.  For two GPUs and a fixed canonical center-vector layout,
the first split is the `B/R` owner, giving two parts.  Each part can then be
subdivided by the first contraction choice, left-first `A * B` or right-first
`B * C`.  Additional binary choices, such as whether to migrate the input
center block first, whether to migrate an intermediate, and where to reduce a
partial `R`, multiply the number of execution labels but keep it finite.  Once
those labels have costs, the planner is a graph or hypergraph partitioner over
the nonzero `f` entries with typed execution states.

It should group terms by reusable pairs:

```text
(A, B) groups for left-first intermediates
(B, C) groups for right-first intermediates
(R, A) or (R, C) groups for accumulation opportunities
```

A dynamic-programming or graph-optimization algorithm can then choose a mixed
plan over groups, subject to memory and communication constraints.

The important architectural constraint is that this planner should consume
Uni20 device/block abstractions directly. It should not depend on
TensorContraction `Arranger`/`Swapper`, and it should not rely on implicit host
copies or late global rearrangement.
