# Tensor-Network Linear Algebra API Survey

This note is a planning document for Uni20 dense linear algebra wrappers. It
compares the practical API surface used by broad array libraries and
tensor-network libraries, then maps that onto Uni20's current LAPACK/MPLAPACK
wrapper work.

The conclusion is deliberately not "wrap all of LAPACK". Uni20 should add dense
wrappers when they support a higher-level tensor-network or Krylov operation,
and should keep low-level wrappers behind scalar, layout, and backend dispatch
boundaries.

## Source Survey

| project | relevant signal | implication for Uni20 |
| --- | --- | --- |
| NumPy | `numpy.linalg` exposes a broad linear algebra surface: products and tensor contractions, norms, Cholesky, QR, SVD, Hermitian and general eigenproblems, condition estimates, determinants, solves, least squares, inverse, pseudo-inverse, and tensor solve/inverse helpers. | Useful as a baseline for common names, but too broad as a coverage target. Uni20 should not chase the full general-purpose array API. |
| PyTorch | `torch.linalg` is similarly broad and groups operations into matrix properties, decompositions, solvers, inverses, matrix functions, matrix products, tensor operations, and experimental variants. | Useful for the division between decompositions, solvers, products, and diagnostics. GPU/batched behavior is important later, but not a reason to expose every LAPACK routine now. |
| ITensors.jl | The tensor interface hides memory layout, supports dense and QN block-sparse storage, and makes contractions, tensor exponentials, null spaces, SVD, eigen, and `factorize` central tensor operations. `factorize` chooses among SVD, eigen, and QR depending on truncation and orthogonality needs. | High-level tensor operations should own the public API. Dense wrappers should support contraction, truncation, canonicalization, and validation while preserving symmetry metadata. |
| TensorKit.jl | Tensor maps have vector operations, index permutations, contractions, and tensor factorizations. Factorizations act on matrix blocks associated with symmetry sectors, with truncation strategies layered above the dense kernels. | This is close to Uni20's intended shape: blockwise dispatch plus dense kernels, not dense fallback. Low-level wrappers must be easy to call per block. |
| quimb | Tensor decompositions center on truncated SVD, SVD via Hermitian eigensystems, randomized/iterative SVD, truncated Hermitian eigensystems, stabilized QR/LQ, regularized Cholesky, polar decompositions, isometrization, and similarity compression. | Truncation and isometry construction are first-class tensor-network operations. The dense layer needs enough SVD/eigh/QR/LQ/polar support to implement those policies cleanly. |
| YASTN | Symmetric tensor algebra exposes contractions, `vdot`, norms, SVD and Hermitian eigensystem decompositions with truncation, QR, and matrix-free Krylov `expmv`/`eigs` on generalized vectors with `norm`, `vdot`, `add`, and Krylov-space expansion. | This strongly supports Uni20's matrix-free Krylov boundary. It also reinforces that high-level truncation masks and blockwise policies matter as much as raw LAPACK calls. |

## Current Uni20 Fit

Uni20 currently has a strong real dense projected-kernel prototype:

- real BLAS-like local vector and matrix operations;
- real dense, banded, tridiagonal, SPD, and symmetric-indefinite solves with
  condition and refinement variants;
- real QR/LQ/QL/RQ, SVD, selected SVD, Hermitian eigensystems, generalized
  Hermitian eigensystems, Schur, QZ, and condition diagnostics;
- real binary128 coverage for selected Krylov eigensolvers, Krylov exponential
  actions, dense matrix exponentials, and precision-validation probes when
  `UNI20_ENABLE_MPLAPACK=ON`.

The complex dense surface is intentionally narrower today:

- complex BLAS-like operations exist for ordinary `c`/`z` paths and selected
  binary128 probes;
- complex nonsymmetric projected eigen and Schur helpers exist for Arnoldi
  (`geev`, `gees`, `trexc`);
- complex Hermitian dense eigensystems, complex SVD, complex QR/LQ, complex
  solves, and complex conditioning/refinement wrappers are the main missing
  pieces for tensor-network work.

The native Krylov matrix-free boundary already matches the tensor-network
direction: vectors are opaque, and solvers require allocation, copy, `axpy`,
scale/zero, `norm`, inner products returning host scalars, and `matvec`.

## Priority Groups

### Needed Soon

| group | examples | why |
| --- | --- | --- |
| Complex Hermitian and generalized Hermitian eigensystems | `heev`, `heevd`, `heevr`, `hegv`, `hegvd`, `hegvx` | Needed for complex Hermitian dense references, canonicalization checks, blockwise Hermitian tensor factorizations, metric problems, tangent-space experiments, and validation of complex Krylov paths. |
| Complex SVD | `gesvd`, `gesdd`, and eventually selected SVD | Core tensor truncation operation for complex tensors and transfer-matrix workflows. Return truncation-ready singular values and reconstruction diagnostics above the raw wrapper. |
| Complex QR/LQ | `geqrf`/`ungqr`, `gelqf`/`unglq`, plus apply-unitary helpers | Needed for MPS canonical forms, orthogonalization, gauge moves, and stable factorization paths when no truncation is requested. |
| Complex dense solves | `gesv`, `getrf`/`getrs`, `gesvx`, `gerfs`, `gecon`, triangular solves | Needed for tensor-network local linear solves, gauge fixing, implicit methods, and diagnostics. Include condition/refinement data where available. |
| Complex positive-definite and Hermitian-indefinite solves | `potrf`/`potrs`/`pocon`; `hetrf`/`hetrs`/`hecon` | Needed once metric problems, normal equations, generalized Hermitian paths, or tangent-space methods need robust complex solves. |
| Complex matrix norms and equilibration | `lange`, `lanhe`, `lantr`, `geequ` family where available | Needed to scale tolerances, diagnose ill conditioning, and keep precision-aware stopping rules honest. |
| Tensor truncation policy layer | cutoff, max dimension, per-sector limits, discarded weight, multiplet-aware hooks | This should sit above SVD/eigh wrappers. It is the actual tensor-network API, and it prevents exposing raw LAPACK choices as user policy. |

### Already Useful

| group | status |
| --- | --- |
| Real Krylov projected kernels | Broad enough for current real symmetric and real nonsymmetric eigensolver work. |
| Real linear solvers | Enough to prototype dense projected solves and condition diagnostics, including binary128 probes. |
| Real SVD/eigh/QR/LQ | Enough to prototype real tensor factorizations and validation cases. |
| Complex Hermitian and generalized Hermitian eigensystems | Enough to try dense complex Hermitian and type-1 generalized Hermitian problems in `c`/`z` arithmetic. |
| Complex nonsymmetric Schur/eigen helpers | Enough for current complex Arnoldi restart and Ritz extraction work. |
| Matrix-free Krylov interface | The right boundary for opaque CPU/GPU/MPI vectors; no workspace-array or reverse-communication API should leak into higher layers. |

### Likely Later

| group | examples | trigger |
| --- | --- | --- |
| Generalized nonsymmetric complex pencils | `ggev`, `gges`, QZ reordering and condition estimates | Maybe later. Needed only if non-Hermitian generalized transfer problems become common. |
| Polar and isometry-specific decompositions | polar via SVD or direct algorithms, QR-based isometrization | Useful for gauge fixing and tensor isometry projection once concrete workflows demand it. |
| Rank-revealing and randomized factorizations | pivoted QR, selected SVD, randomized SVD | Useful for compression and diagnostics, but should follow high-level truncation policy design. |
| Matrix functions beyond exponential | logarithm, square root, sign, fractional powers | Only add when an algorithm needs them. Dense `expm` and matrix-free `expmv` are the current priority. |
| Batched or GPU-specialized dense kernels | backend-specific batched SVD/eigh/QR | Important later, but should enter through the Uni20 backend dispatch layer rather than ad hoc Krylov wrappers. |

### Probably Not Worth Wrapping For Coverage

| group | reason |
| --- | --- |
| Packed, RFP, and obscure storage-format LAPACK variants | Uni20's tensor/block storage will define its own layout boundary. Copying small projected matrices is acceptable when needed. |
| Unblocked helper routines and internal LAPACK building blocks | Add only if a higher-level algorithm needs direct control. |
| Full NumPy/PyTorch surface | General-purpose helpers such as tensor inverse, matrix cross products, Vandermonde helpers, and broad determinant APIs are not tensor-network priorities. |
| Public dense inverse as a recommended operation | A wrapper can exist for diagnostics, but high-level algorithms should prefer solves, factorizations, or condition-aware APIs. |

## Suggested Wrapper Order

1. Add complex matrix norms and general complex LU solve/factor/condition
   wrappers. These are small, useful, and validate the complex wrapper pattern.
2. Keep the completed complex Hermitian and generalized Hermitian eigensystem
   wrappers covered by direct `c`/`z` tests and binary128-complex MPLAPACK
   probes.
3. Add complex SVD wrappers, with tests that compare reconstruction residuals,
   singular-value ordering, and complex phase handling.
4. Add complex QR/LQ wrappers and unitary application/materialization helpers.
5. Add complex SPD/Hermitian-indefinite solve wrappers with refinement and
   reciprocal-condition diagnostics.
6. Build a tensor-facing truncation result type that records kept dimension,
   discarded weight, per-sector decisions, and reconstruction diagnostics.

Each new wrapper should have at least one direct dense test and, when it exists
to support a higher-level Krylov or tensor operation, one test through that
higher-level path. Binary128 tests should not only compile; they should include
at least one case where `double` cannot represent the tested gap or tolerance.

## References

- NumPy linear algebra reference:
  <https://numpy.org/doc/stable/reference/routines.linalg.html>
- PyTorch `torch.linalg` reference:
  <https://docs.pytorch.org/docs/stable/linalg.html>
- ITensors.jl tensor and decomposition reference:
  <https://itensor.github.io/ITensors.jl/stable/ITensorType.html>
- TensorKit.jl tensor reference:
  <https://quantumkithub.github.io/TensorKit.jl/stable/lib/tensors/>
- quimb tensor decomposition reference:
  <https://quimb.readthedocs.io/en/latest/autoapi/quimb/tensor/decomp/index.html>
- YASTN tensor algebra and Krylov references:
  <https://yastn.github.io/yastn/tensor/algebra.html>,
  <https://yastn.github.io/yastn/tensor/krylov.html>
