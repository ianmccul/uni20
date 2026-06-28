# Matrix Market NEP Fixtures

These fixtures come from the NIST Matrix Market Non-Hermitian Eigenvalue
Problem (NEP) Collection. The collection is explicitly intended as a testbed
for nonsymmetric eigenvalue algorithms.

The original Matrix Market comments embedded in each `.mtx` file are preserved.

| Local file | Source page | Download URL | Type | Purpose |
| --- | --- | --- | --- | --- |
| `brussel/rdb200.mtx` | `https://math.nist.gov/MatrixMarket/data/NEP/brussel/rdb200.html` | `https://math.nist.gov/pub/MatrixMarket2/NEP/brussel/rdb200.mtx.gz` | real unsymmetric, 200 x 200, 1120 entries | Brusselator reaction-diffusion eigenproblem; compact real nonsymmetric application fixture. |
| `dwave/dwa512.mtx` | `https://math.nist.gov/MatrixMarket/data/NEP/dwave/dwa512.html` | `https://math.nist.gov/pub/MatrixMarket2/NEP/dwave/dwa512.mtx.gz` | real unsymmetric, 512 x 512, 2480 entries | Dielectric waveguide problem; medium-sized real nonsymmetric example fixture. |
| `dwave/dw2048.mtx` | `https://math.nist.gov/MatrixMarket/data/NEP/dwave/dw2048.html` | `https://math.nist.gov/pub/MatrixMarket2/NEP/dwave/dw2048.mtx.gz` | real unsymmetric, 2048 x 2048, 10114 entries | Larger dielectric waveguide problem; useful manual example for sparse Arnoldi behavior. |
| `mhd/mhd416a.mtx` | `https://math.nist.gov/MatrixMarket/data/NEP/mhd/mhd416a.html` | `https://math.nist.gov/pub/MatrixMarket2/NEP/mhd/mhd416a.mtx.gz` | real unsymmetric, 416 x 416, 8562 entries | Small magnetohydrodynamics application matrix; harder real nonsymmetric comparison case. |
| `olmstead/olm100.mtx` | `https://math.nist.gov/MatrixMarket/data/NEP/olmstead/olm100.html` | `https://math.nist.gov/pub/MatrixMarket2/NEP/olmstead/olm100.mtx.gz` | real unsymmetric, 100 x 100, 396 entries | Olmstead flow model; compact hydrodynamics eigenproblem for nonsymmetric Arnoldi comparisons. |
| `olmstead/olm500.mtx` | `https://math.nist.gov/MatrixMarket/data/NEP/olmstead/olm500.html` | `https://math.nist.gov/pub/MatrixMarket2/NEP/olmstead/olm500.mtx.gz` | real unsymmetric, 500 x 500, 1996 entries | Larger Olmstead flow model; useful manual example without excessive fixture size. |
| `stoch/lop163.mtx` | `https://math.nist.gov/MatrixMarket/data/NEP/stoch/lop163.html` | `https://math.nist.gov/pub/MatrixMarket2/NEP/stoch/lop163.mtx.gz` | real unsymmetric, 163 x 163, 935 entries | Small stochastic Markov-model eigenproblem; starter real Arnoldi fixture. |
| `tubular/tub1000.mtx` | `https://math.nist.gov/MatrixMarket/data/NEP/tubular/tub1000.html` | `https://math.nist.gov/pub/MatrixMarket2/NEP/tubular/tub1000.mtx.gz` | real unsymmetric, 1000 x 1000, 3996 entries | Tubular reactor model; larger sparse real nonsymmetric example fixture. |

Potential future NEP sources to inspect/import:

- `MVMGRC`: Grcar generator, useful for nonnormal Arnoldi stress tests.
- `AIRFOIL`, `BFWAVE`, `DWAVE`, `GEDNEY`, `MHD`, `TUBULAR`: application
  matrices likely useful once restarted nonsymmetric solvers are implemented.
- `STOCH`: stochastic matrices with dominant-eigenvalue behavior relevant to
  transfer-operator tests.

Do not bulk-import large NEP files into the core repository without checking
size and provenance. Prefer a curated fixture subset here, and keep larger
comparison corpora for the eventual linalg-benchmark repository.

## Suggested Manual Examples

These are useful with `examples/krylov_nonsymmetric_matrix_market_example.cpp`
after building the examples target:

```bash
./build_codex/build_gcc13_debug/examples/krylov_nonsymmetric_matrix_market_example \
  tests/krylov/matrix_market/nep/dwave/dwa512.mtx \
  --nev=2 --ncv=32 --max-iters=500 --real-policy=real-schur
```

```bash
./build_codex/build_gcc13_debug/examples/krylov_nonsymmetric_matrix_market_example \
  tests/krylov/matrix_market/nep/dwave/dw2048.mtx \
  --nev=2 --ncv=40 --max-iters=1000 --real-policy=real-schur
```

`olmstead/olm500.mtx` and `tubular/tub1000.mtx` are useful stress cases for
rightmost eigenvalue searches. `tubular/tub1000.mtx` is deliberately hard and
benefits strongly from guard vectors (`nkeep > nev`) and a larger search space.
The same fixtures are also good candidates for external ARPACK comparison
programs in the validation and benchmarking repository.
