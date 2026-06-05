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

Set `UNI20_TENSORCONTRACTION_RABC_PLANNER=arranger` to use the old worklist
planner for profiling or regression comparison.

This is not the final Uni20 storage model. It still borrows TensorContraction
buffer bookkeeping while replacing the R/A/B/C worklist scheduling seam.

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

The fixture replay benchmark also has an opt-in cost-based R/A/B/C placement
prototype, enabled with `UNI20_TENSORCONTRACTION_RABC_PLACEMENT=cost`.  It
assigns contiguous fresh `R` output-block ranges to local devices using the
current right-first term graph.  The default model scores GEMM work and
central-vector movement; it intentionally ignores `A`/`C` environment staging
cost so the captured-fixture benchmark measures resident Lanczos matvec
behavior rather than setup time.  `cost-block` enables the more aggressive
per-block greedy variant, but that can disable the bridge's current
coalesced-slab vector algebra fast path.  Both policies are diagnostics, not the
final storage layout.

## Optimization Formulation

The placement problem is a graph/hypergraph partitioning problem with execution
choices attached to each hyperedge.  A term

```text
t = (R_r, A_a, B_b, C_c, alpha)
```

connects one output block, three input blocks, and one scalar coefficient.  In
the final Uni20 model, decision variables should include:

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

The current `cost` policy restricts `p(i)` to contiguous block ranges.  The
`cost-block` diagnostic allows arbitrary `p(i)` and therefore approximates the
true graph partitioning problem.  Arbitrary placement still stores each device's
chosen block set as one coalesced slab; contiguity is a storage property, not a
requirement that logical block indices form intervals.  The operator applies one
canonical center-vector layout to both Krylov input and output blocks; if a
contraction produces a block away from that canonical owner, the generic planner
must account for the relayout before the result becomes the next Lanczos input.

For the current TensorContraction bridge, `cost-block` starts from the
byte-balanced slab layout and applies local-search single-block moves.  Because
non-contiguous logical layouts still have vector-layout and relayout overhead
that is not yet represented perfectly in the R/A/B/C term model, the bridge
requires a configurable minimum predicted speedup before accepting an arbitrary
layout.  Otherwise it delegates back to the default byte-balanced coalesced
layout.

## Long-Term Planner

The long-term scheduler should operate on a term graph:

```text
term t:
  output R_i
  inputs A_j, B_k, C_l
  coefficient alpha_t
```

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
