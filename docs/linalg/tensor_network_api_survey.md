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

The complex dense surface is still narrower than the real survey, but the
checked provider layer now includes the main prerequisites for tensor
factorizations:

- complex BLAS-like operations exist for ordinary `c`/`z` paths and selected
  binary128 probes;
- checked matrix norm, core LU solve/factor/inverse/condition, and SVD wrappers
  exist for ordinary `c`/`z` and optional binary128-complex paths;
- complex nonsymmetric projected eigen and Schur helpers exist for Arnoldi
  (`geev`, `gees`, `trexc`);
- standard, divide-and-conquer, selected, and generalized complex Hermitian
  eigensystem wrappers are available;
- the active `self_adjoint_eigh`/`eigh` path exposes the standard Hermitian
  eigensystem operation;
- the active exact SVD path exposes `singular_values`, `svd_left`, `svd_right`,
  and `svd` through `gesvd`, with reduced factors by default and independent
  full-left/full-right extent options; preserving and consuming Async lowering
  returns independent output epochs;
- the active `truncated_svd` policy layer selects and copies reduced factors,
  permits rank zero, reports stable same-precision truncation statistics, and
  has preserving and consuming Async forms with four independent outputs;
- QR/LQ, expert/refined solves, and positive-definite or Hermitian-indefinite
  solves remain future work.

The native Krylov matrix-free boundary already matches the tensor-network
direction: vectors are opaque, and solvers require allocation, copy, `axpy`,
scale/zero, `norm`, inner products returning host scalars, and `matvec`.

## Priority Groups

### Needed Soon

| group | examples | why |
| --- | --- | --- |
| Tensor-facing SVD truncation and reuse | dispatched `gesvd` plus `truncated_svd`; checked `gesdd` and `gesvdx` provider wrappers | Exact real and complex Tensor SVD includes partial-output APIs and consuming reuse of compatible input allocations for reduced factors. Truncation is a separate policy/result layer; selected or divide-and-conquer backend choices remain separate future work. |
| Complex QR/LQ | `geqrf`/`ungqr`, `gelqf`/`unglq`, plus apply-unitary helpers | Needed for MPS canonical forms, orthogonalization, gauge moves, and stable factorization paths when no truncation is requested. |
| Complex expert and refined dense solves | checked `gesv`, `getrf`/`getrs`, `getri`, and `gecon`; future `gesvx`, `gerfs`, and triangular solves | Core LU operations are implemented. Add expert/refinement data and specialized solve families when tensor-network workflows require them. |
| Complex positive-definite and Hermitian-indefinite solves | `potrf`/`potrs`/`pocon`; `hetrf`/`hetrs`/`hecon` | Needed once metric problems, normal equations, generalized Hermitian paths, or tangent-space methods need robust complex solves. |
| Complex equilibration | checked `lange`, `lanhe`, and `lantr`; future `geequ` family where available | Core norms are implemented. Equilibration remains useful for scaling and condition-aware solve paths. |
| BlockTensor truncation policy layer | per-sector limits, global discarded weight, multiplet-aware hooks | Dense SVD truncation is implemented. Symmetry-aware policy must sit above per-block SVD/eigh calls so global and multiplet constraints do not leak into raw LAPACK choices. |

### Tensor-Facing SVD Semantics

The exact and truncating Tensor SVD operations are separate.
`singular_values(matrix)` returns only `s`; `svd_left(matrix)`
returns `U` and `s`; `svd_right(matrix)` returns `s` and `Vh`; and
`svd(matrix)` returns all three. For an `m x n` input, the reduced shapes are
`m x k`, `k`, and `k x n`, where `k = min(m,n)`. One-sided calls accept one
`SvdVectorExtent`; `SvdOptions` independently requests full left or right
extents for the complete decomposition. The right factor is always `Vh`:
transpose for real inputs and conjugate transpose for complex inputs. A
zero-rank decomposition returns empty reduced factors; a requested
unconstrained full factor is the corresponding identity matrix.

Every value operation has a preserving lvalue form and a consuming mutable
owning-rvalue form. A preserving call materializes LAPACK-compatible work
storage. A consuming call may use the input allocation directly as destructive
workspace. For reduced factors, compatible column-major host storage can be
returned as `U` through `JOBU='O'` or as `Vh` through `JOBVT='O'`; the adopted
strided owner preserves a padded leading dimension and unused storage tail.
Full factors cannot use the corresponding overwrite job. Incompatible layouts
materialize instead, so passing an rvalue permits but does not guarantee
returned-allocation reuse.

`truncated_svd(matrix, policy)` always starts from the reduced exact SVD and
returns right-sized `U`, `s`, and `Vh` factors. Its policy is:

```cpp
template <uni20::Real Real>
struct SvdTruncationPolicy
{
    std::size_t minimum_retained_extent = 0;
    std::size_t maximum_retained_extent =
        uni20::numeric_limits<std::size_t>::max();
    std::optional<Real> singular_value_cutoff = std::nullopt;
    std::optional<Real> normalized_squared_singular_value_cutoff =
        std::nullopt;
    std::optional<Real> maximum_discarded_weight = std::nullopt;
};
```

The absolute cutoff keeps every singular value greater than or equal to the
cutoff. The normalized cutoff compares each `s_i * s_i` with
`cutoff * sum_j(s_j * s_j)`. The discarded-weight criterion chooses the
smallest rank whose normalized discarded squared norm is no greater than the
requested maximum. Active accuracy criteria each impose a minimum rank; the
largest such rank wins, then the explicit minimum floor and maximum hard cap
are applied. A conflicting maximum may therefore prevent an accuracy criterion
from being satisfied. With no active criterion and no restrictive maximum, the
default policy returns the exact reduced decomposition.

The truncation result reports enough information for a caller that also has
the requested policy to determine which constraints were binding:

```cpp
template <uni20::Real StatisticsReal>
struct SvdTruncationInfo
{
    std::size_t available_rank;
    std::size_t retained_rank;
    StatisticsReal original_squared_norm;
    StatisticsReal discarded_weight;
    std::optional<StatisticsReal> smallest_retained_singular_value;
    std::optional<StatisticsReal> largest_discarded_singular_value;
};
```

`original_squared_norm` is the pre-truncation Frobenius norm squared,
`sum_i(s_i * s_i)`. `discarded_weight` is the normalized discarded squared
norm,
`sum_discarded(s_i * s_i) / original_squared_norm`, and is defined as zero when
the original squared norm is zero. These public statistics use the singular
value's real scalar type. The implementation scales by the largest singular
value and accumulates normalized squares from smallest to largest with
same-precision compensation. This prevents avoidable intermediate overflow
while preserving binary128 policy and result types. `original_squared_norm`
may still be infinity when the mathematically correct squared norm is not
representable in that real type. The retained rank identifies minimum- or
maximum-extent limits, the two boundary values identify per-value cutoffs, and
the original norm and discarded weight identify the cumulative-error boundary.
Several constraints may be simultaneously binding; the API does not invent a
unique winner in that case.

Preserving and consuming synchronous overloads mirror exact `svd`. The
consuming form may reuse compatible input storage while computing the exact
factors, but the final truncated factors are right-sized owners and do not
promise to retain that allocation. Async overloads return four independent
values:

```cpp
auto [u, s, vh, info] = truncated_svd(async_matrix, policy);
```

Unhandled task failure reaches all four outputs; a consuming call also fails
the consumed input epoch.

Singular-value normalization or partitioning is a separate output policy and
must not change the meaning of these pre-truncation statistics. If a future
backend supports alternative truncation metrics, expose and report them under
separate explicit names rather than changing `original_squared_norm` or
`discarded_weight`.

For an unconstrained unitary completion, deterministic completion remains the
default. TODO: add an explicit randomized-completion policy, carrying an
explicit random source or seed, for algorithms that intentionally require a
random basis in the unconstrained subspace. Provider-generated or hidden global
randomness must not select this behavior implicitly.

### Zero-Rank Truncation and CUDA

Uni20 truncation must permit a retained extent of zero. The default minimum
retained extent is zero, and a positive minimum is an explicit algorithmic
policy. In particular, a block whose singular values all fail the active
cutoffs must be removable from a `BlockTensor`; retaining an artificial
zero-weight direction can corrupt tangent-space dimensions and introduce
singular gauge or metric problems.

Before implementing the CUDA SVD backend, verify the zero-extent behavior of
each supported cuTensorNet version. cuTensorNet's `max_extent` is an upper bound
on the retained extent, not a configurable minimum. Independently, its
truncated-SVD documentation states that at least one singular value is retained
even when the truncation parameters would remove every singular value. Treat
that forced rank-one result as a provider limitation, not as Uni20 truncation
semantics. If necessary, the CUDA adapter must use an internal nonzero
placeholder extent, post-process the returned spectrum, expose a logical
zero-rank result, and allow the `BlockTensor` layer to omit the block. Add
provider-version tests before choosing the exact workaround, because the
documented behavior may change between releases.

### Already Useful

| group | status |
| --- | --- |
| Real Krylov projected kernels | Broad enough for current real symmetric and real nonsymmetric eigensolver work. |
| Real linear solvers | Enough to prototype dense projected solves and condition diagnostics, including binary128 probes. |
| Real SVD/eigh/QR/LQ | Enough to prototype real tensor factorizations and validation cases. |
| Complex Hermitian eigensystems | Checked `heev`, `heevd`, `heevr`, `hegv`, `hegvd`, and `hegvx` wrappers are available; active Tensor dispatch currently uses standard `heev`. |
| Complex norms, core LU, and SVD provider wrappers | Direct `c`/`z` and optional binary128-complex tests cover norm values, LU solve/inverse/condition behavior, and SVD reconstruction. |
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

1. Keep the completed complex norm, core LU, and SVD provider wrappers covered
   by direct `c`/`z` and binary128-complex tests.
2. Keep the completed complex Hermitian and generalized Hermitian eigensystem
   wrappers covered by direct `c`/`z` tests and binary128-complex MPLAPACK
   probes.
3. Keep exact dispatched Tensor SVD covered for real, complex, rectangular,
   full-factor, zero-extent, and binary128 cases.
4. Keep the completed dense truncation policy/result layer covered for rank
   bounds, both per-value cutoffs, discarded weight, zero rank, complex input,
   Async failure propagation, and binary128 statistics.
5. Add complex QR/LQ wrappers and unitary application/materialization helpers.
6. Add complex expert/refined general solves and
   positive-definite/Hermitian-indefinite solve wrappers with reciprocal
   condition diagnostics.
7. Build the BlockTensor truncation layer that combines per-sector spectra,
   global discarded weight, symmetry selection rules, and multiplet policy.

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
