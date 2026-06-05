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
| `UNI20_TENSORCONTRACTION_RABC_PLACEMENT_LOG` | Set to `1`, `true`, or `on` to print the selected output-block distribution. |
| `UNI20_TENSORCONTRACTION_RABC_MODEL_GFLOPS` | Assumed per-device GEMM throughput. Default: `1000`. |
| `UNI20_TENSORCONTRACTION_RABC_MODEL_CENTRAL_GBPS` | Assumed central-vector transfer bandwidth. Default: `32`. |
| `UNI20_TENSORCONTRACTION_RABC_MODEL_ENV_BYTES` | If set, include environment staging bytes in the model. Default: unset. |
| `UNI20_TENSORCONTRACTION_RABC_MODEL_ENV_GBPS` | Assumed environment transfer bandwidth when environment bytes are enabled. Default: central bandwidth. |
| `UNI20_TENSORCONTRACTION_RABC_MODEL_ARBITRARY_MIN_SPEEDUP` | Minimum predicted speedup required before `cost-block` overrides byte-balanced slab layout. Default: `1.25`. |

The cost policies are intentionally opt-in.  Even the contiguous policy can choose a partition that is worse for the
current executor than the default byte-balanced slabs; the arbitrary block policy can still lose to the default layout
when the R/A/B/C model underestimates vector-layout or relayout costs.
The current cost model is a variable-middle prototype: the Krylov input blocks `B_i` and output blocks `R_i` are forced
onto one canonical layout so vector algebra can use the result as the next Lanczos input without an implicit relayout.
Benchmark both `#LanczosMatvecS` and total wall time before treating a lower contraction model score as a faster
Lanczos solve.

The placement plan is cached inside the resident operator after the first resident apply for a given policy and device
count.  Use `UNI20_RABC_WARMUP=1` when comparing matvec timings so plan construction and environment materialization
are treated as setup rather than Lanczos-loop cost.

The fixture preserves the exact TensorContraction block worklist emitted by the symmetry-aware DMRG path, but it does
not store the higher-level `LocalSpace`, `BlockSpace`, or MPO metadata. Do not feed fixture data back into U(1) MPS
state; use it only as a terminal benchmark artifact.

For synthetic fixtures, initialize the center vector with non-zero random data before capture or upload.  A zero center
vector makes every Hamiltonian application zero and can hide movement, accumulation, or validity bugs in the resident
path.
