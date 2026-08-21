# Block Coalescing and GEMM Grouping

**Status:** active design note for future block-sparse lowering and placement.

This is a draft design note. It records the intended design for coalescing
block-sparse sub-blocks into larger kernel operations, to reduce kernel/launch
count without introducing structural zeros and without dropping symmetry
metadata. It is design direction, not current implemented behavior.

Related notes:

- `docs/symmetry/block_sparse_tensor.md` — the tensor and layout this operates over.
- `docs/architecture/ordering_and_backend_lowering.md` — stream/event lowering; per-launch cost.
- `docs/architecture/execution.md` — coalescing as one use of the cost-model knob.
- `docs/architecture/backend_dispatch.md` — batched/coalesced kernels as backend capabilities.
- `docs/tensor_network/rabc_contraction_scheduling.md` — the R/A/B/C apply this most affects.

## Summary

Block-sparse contractions decompose into many small GEMMs, one per sector
combination. A large fraction of these are tiny — in the calibrated R/A/B/C data,
roughly **half the kernel launches carry near-zero flops** (the small-block,
high-connectivity "tail"), and that fraction does not shrink as bond dimension `m`
grows. Tiny GEMMs are **launch-bound**, so the most fundamental remedy is to issue
fewer, larger kernels. Coalescing along a **single axis** does this with **no
structural zeros** and full preservation of symmetry: it is a kernel-execution
lowering expressed as wider `mdspan` views over the same buffer storage, with the
individual blocks still tracked so results scatter back to the correct sectors.
The decision *whether* to coalesce a group is the same flop/launch crossover the
R/A/B/C cost model quantifies, applied to the data structure instead of to
placement.

## Why single-axis coalescing is hole-free

For a block GEMM `C[m,n] = Σ_k A[m,k]·B[k,n]` over block-multi-indices, a coalesced
dense operand `A'` over a chosen `(m-set, k-set)` rectangle is dense only if every
`(m,k)` pair in the rectangle is a populated sector. Coalesce along **one** axis
(fix `k`, take a set of `m`; or fix `m`, take a set of `k`) and you can choose
exactly the populated blocks → no holes. Coalesce along **two** axes and the 2-D
set `m-set × k-set` generally has missing combinations → structural zeros. So
"single axis, no zeros" is the general statement, not a heuristic:

- **single-axis** selection is hole-free by construction;
- **two-axis** selection generally forces a block-diagonal-with-holes layout
  (structural zeros), which wastes flops.

The two single-axis directions are both available and are distinct planning
choices: **free-axis** grouping (stack the valid output rows for a shared
contracted index → one taller GEMM) and **contraction-axis** grouping (concatenate
along `k` → one wider-`K` GEMM). You may take one or the other, not both at once.

## Expressed as strided views over shared storage

A logical block is an `mdspan<T, extents, layout_stride>` with its `(offset,
strides)` into a shared buffer; a coalesced operand is a wider view over the same
buffer with the group's leading dimension. The same bytes are simultaneously "N
individual blocks" (carrying sector identity, so symmetry is preserved) and "one
matrix" (handed to a single GEMM). Coalescing therefore never produces a dense
blob the tensor type can see — it satisfies the no-drop-symmetry rule in
`block_sparse_tensor.md` by construction. After the GEMM, results scatter back to
the correct sectors through the same per-block views.

## Interleaved vs contiguous — the real tradeoff is a copy

The difference between the two memory arrangements is whether a gather is needed:

- **Interleaved — zero-copy, if the layout cooperates.** Issue the coalesced GEMM
  on the existing buffer with the right leading dimension; no gather. But a single
  leading dimension means a **constant stride**, which requires the coalesced
  blocks to be stride-compatible (regular sizes along the non-coalesced dims). The
  small, regular **LocalSpace** grouping dimension is ideal here. Individual
  blocks select a LocalSpace state through their key and have no corresponding
  dense axis; the execution layout interleaves several such key values and
  exposes them as a coalesced axis. Varying-dimension BlockSpace sectors may not
  yield a constant stride.
- **Contiguous (packed) — flexible, may cost a pack.** Concatenating varying-size
  blocks always works, but if the source is not already contiguous it costs a
  gather (one pack kernel — still far cheaper than N separate launches).

Which to use is adjudicated by the cost model: pack cost vs launch savings. The
consequence for the tensor is that the **layout's memory arrangement should be
chosen so the expected coalescing groups are stride-addressable** — interleave the
axis you will coalesce. Layout is therefore a contraction-pattern-aware decision,
not arbitrary.

## Two coalescing modes

- **Batched / strided GEMM** (e.g. cuBLAS strided-batched): same-shape blocks, one
  launch covers many, **zero flop waste**. Many tail sectors are similar-shape, so
  this is the clean, low-risk first win — no structural-zero compromise at all.
- **Concatenated GEMM**: different blocks sharing an operand packed into one larger
  matrix — more general, but reintroduces structural zeros if pushed to two axes.
  Use single-axis, and only below the cost-model crossover.

## The crossover decides whether to coalesce

Coalescing trades flops for launches. Below a block size it wins (launch-bound);
above it, coalescing structural zeros is pure waste (flop-bound). The crossover is
roughly where `launch_us × N ≈ wasted_flops / gflops`, i.e. the same calibrated
constants used for placement (see `../architecture/execution.md` for the numbers).
For pure batched GEMM there is no flop waste, so the crossover is simply where the
launch savings exceed any pack cost.

## One mechanism, three payoffs

A single-axis coalescing group is simultaneously:

1. **one GEMM** instead of N (fewer launches — attacks the tail at the source);
2. **one hazard-tracked sub-range** (buffer-with-subviews granularity);
3. **one MPI message** instead of N (the same stride-view that makes one GEMM makes
   one `Isend`, directly reducing the per-message / finite-tag overhead from
   `../architecture/ordering_and_backend_lowering.md`).

## Coalescing is a plan-time transform

The planner chooses groups and emits a *single* DAG node per coalesced group
(plus its scatter map), rather than per-block nodes. This is why coalescing mostly
avoids the aliasing-hazard problem: the coalesced op is the node, so individual
blocks are not separate concurrent ops on the shared region. It also means
coalescing is part of placement planning, using the same cost model — not a
runtime decision inside the kernel.

## Decisions made

- Coalesce along a single axis only; this is hole-free (no structural zeros) by
  construction.
- Express coalescing as wider `mdspan` strided views over shared buffer storage,
  with per-block views retained so symmetry/sector identity is preserved; it is a
  lowering, never a representation change.
- Prefer batched/strided GEMM for same-shape tail groups (zero flop waste);
  concatenated GEMM only single-axis and below the crossover.
- Coalescing is a plan-time transform that emits one DAG node per group; the
  whether-to-coalesce decision uses the placement cost model.
- The same grouping serves GEMM, hazard granularity, and MPI message aggregation.

## Open questions

- **Axis selection policy.** When both free-axis and contraction-axis grouping are
  available for a contraction, which does the planner prefer, and can it mix per
  contraction stage (e.g. free-axis for the `B·C` stage, contraction-axis for the
  `A·Y` accumulate in R/A/B/C)?
- **Interleave vs pack thresholds.** The crossover between zero-copy interleave
  (needs stride regularity) and pack-then-coalesce, especially across
  varying-dimension BlockSpace sectors.
- **Batched-GEMM grouping across shapes.** How aggressively to bucket
  near-same-shape sectors into batched calls vs padding to a common shape (padding
  reintroduces some waste).
- **Interaction with deferred sync.** Confirm that a coalesced tail group pinned to
  one stream with deferred sync and a single terminal synchronization is the
  intended steady-state shape, vs spreading a group across streams for device-level
  overlap (which would force events back in).
- **Whether coalescing groups should be cached** across DMRG sweeps once the block
  structure at a site stabilizes (mirrors the per-site CUDA-graph reuse question).
