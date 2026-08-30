# Model Examples

`spin_half_heisenberg_dmrg_example.cpp` constructs a U(1) spin-half Néel
product MPS and reduced-boundary Heisenberg MPO, then runs alternating two-site
DMRG sweeps. It prints one summary per directional traversal, elapsed time, and
the difference from a registered reference energy when one is available.

The default four-site calculation can verify the exact open-chain energy

```text
-(3 + 2 sqrt(3)) / 4
```

with

```bash
spin_half_heisenberg_dmrg_example --check
```

The larger reference case is

```bash
spin_half_heisenberg_dmrg_example \
    --sites=20 --max-states=64 --max-sweeps=8 \
    --energy-tol=1e-10 --local-matvecs=4 \
    --scalar=real --precision=fp64 --block-threads=1 --check
```

`--local-matvecs=N` selects the fixed number of effective-Hamiltonian
applications requested for each local update. It defaults to four. This is a
local work budget, not a convergence tolerance; sweep-level environment and
energy convergence remain the controlling DMRG criteria.

`--scalar=complex` runs the same calculation with `uni20::complex<Real>`
storage and arithmetic. This is useful for controlled comparisons with
implementations that do not provide a real-scalar path.
`--precision=fp32|fp64` selects `Real` independently; the default is `fp64`.

In an MPLAPACK-enabled build, `--precision=fp128` runs the same U(1)
BlockTensor DMRG path with `uni20::float128` or
`uni20::complex<uni20::float128>` storage. Precision selection does not change
the symmetry or block-sparse model.

`--block-threads=N` installs a `TbbScheduler` with concurrency `N`. The
environment and transient two-site center use
`ParallelPackedSparseBlockStorage`, so independent output blocks execute as
synchronous lightweight batch items. Dense BLAS should normally remain
single-threaded when block-level concurrency is greater than one.

`--measurements=coarse` records and prints inclusive DMRG phase wall times.
`--measurements=detailed` additionally times every per-charge block-SVD item
and reports aggregate overlap and scheduler-tail fields. The default is
`--measurements=off`, which selects the compile-time disabled library path.
Detailed measurements intentionally perturb fine-grained work and should be
used for attribution, not final benchmark numbers.

The first end-to-end scaling tables, local Matrix Product Toolkit orientation,
and exact current parallel boundary are recorded in
[DMRG Performance Baselines](../../docs/tensor_network/dmrg_performance_baselines.md).

Its comparison energy, `-8.682473334398985`, was calculated with Matrix Product
Toolkit `mp-dmrg-2site` from a Néel product state using `m=128`. The reference is
reproducible with:

```bash
spinchain-u1 -S 0.5 -o spinchain-u1.lattice
mp-construct --finite --repeat 10 -l spinchain-u1.lattice \
    -o spinchain-u1-l20-neel.psi '0.5:-0.5'
mp-dmrg-2site -w spinchain-u1-l20-neel.psi \
    -H spinchain-u1.lattice:H_J1 -m 128 -s 12 \
    --maxiter 40 --miniter 4 --maxtol 1e-12 --flush
```

The example remains entirely symmetry-aware. The reference energy is a terminal
assertion and does not feed a dense state back into the DMRG path.
