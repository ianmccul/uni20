# Krylov Algorithms

This document is the implementation inventory for Uni20's native Krylov
solvers. Keep it synchronized with solver entry points, supported scalar types,
public parameters, default policies, and internal convergence or restart tuning.

LAPACK-style scalar tags are used throughout. Complex scalar spellings follow
the project scalar policy in [scalar_policy.md](scalar_policy.md):

| tag | scalar type |
| --- | --- |
| `s` | `float` |
| `d` | `double` |
| `c` | `uni20::complex<float>` |
| `z` | `uni20::complex<double>` |
| `f128` | optional `uni20::float128` |
| `cf128` | optional `uni20::complex<uni20::float128>` |

With `UNI20_ENABLE_MPLAPACK=ON`, Uni20 enables optional experimental binary128
probes for scalar-generic dense projected kernels, matrix-free eigensolvers, and
exponential actions. MPLAPACK is an external package dependency in this
configuration; Uni20 does not download or build it. The
ordinary typed Krylov tests remain focused on the stable `s`, `d`, `c`, and `z`
paths; dedicated `MplapackBinary128*` targets cover binary128-specific stress
cases. The main inventory tables below therefore list the ordinary `s`, `d`,
`c`, and `z` coverage, followed by a separate optional binary128 inventory. See
[krylov_precision_validation.md](krylov_precision_validation.md) for the
test-level `f128` and `cf128` validation matrix.

## Matrix-Free Boundary

Native Krylov solvers operate through a matrix-free interface. The Krylov layer
does not inspect vector storage and assumes only vector allocation, copy,
`axpy`, scaling, zeroing, norm, inner product, and `matvec` operations. The
problem dimension comes from the operation object and is checked against the
prototype or initial vector.

Generalized symmetric paths may additionally use `metric_inner_product(x, y)`.
When it is absent, the current wrapper applies `B*y` into backend-owned scratch
and then calls the ordinary inner product.

The matrix-free boundary does not remove the local dense projected problem.
Entry points that solve projected eigensystems, Schur forms, or dense projected
exponentials therefore use LAPACK-oriented scalar concepts: symmetric/Hermitian
eigensolvers need real LAPACK coverage for the underlying real precision,
complex nonsymmetric eigensolvers need dense complex LAPACK coverage, and
exponential actions need dense coverage for the scalar used by the projected
exponential.

## Native Algorithm Inventory

### Dense Projected Problems

These are local host-side kernels for small Krylov subspace matrices.

| component | `s` | `d` | `c` | `z` | role |
| --- | --- | --- | --- | --- | --- |
| Dense vector and matrix primitives | yes | yes | yes | yes | Local BLAS-like operations over Krylov subspace data. |
| Dense real matrix norms | yes | yes | n/a | n/a | Dense projected full, symmetric, and triangular/trapezoidal norms through LAPACK `lange`, `lansy`, and `lantr`. |
| Dense real linear solve | yes | yes | n/a | n/a | Dense projected utility solve through LAPACK `gesv`. |
| Dense real refined linear solve | yes | yes | n/a | n/a | Dense projected utility solve through LAPACK `getrf`/`getrs` followed by `gerfs`, returning forward/backward error estimates. |
| Dense real expert linear solve | yes | yes | n/a | n/a | Dense projected utility solve through LAPACK `gesvx`, returning condition estimates, forward/backward error bounds, equilibration, and condition diagnostics. |
| Dense real equilibration | yes | yes | n/a | n/a | Dense projected row/column scaling diagnostics through LAPACK `geequ`. |
| Dense real LU factorization and solve | yes | yes | n/a | n/a | Dense projected reusable LU solve through LAPACK `getrf`/`getrs`. |
| Dense real general-band factorization and solve | yes | yes | n/a | n/a | Dense projected reusable general-band LU solve through LAPACK `gbtrf`/`gbtrs`. |
| Dense real general-band refined solve | yes | yes | n/a | n/a | Dense projected general-band solve through LAPACK `gbtrf`/`gbtrs` followed by `gbrfs`, returning forward/backward error estimates. |
| Dense real general-band expert solve | yes | yes | n/a | n/a | Dense projected general-band expert solve with condition and forward/backward error diagnostics through LAPACK `gbsvx`. |
| Dense real general-band equilibration | yes | yes | n/a | n/a | Dense projected general-band row/column scaling diagnostics through LAPACK `gbequ`/`gbequb`. |
| Dense real general-band reciprocal condition estimate | yes | yes | n/a | n/a | Dense projected general-band one-norm reciprocal condition estimate through LAPACK `gbtrf`/`gbcon`. |
| Dense real general tridiagonal solve, condition, and refinement | yes | yes | n/a | n/a | Dense projected reusable general tridiagonal LU solve plus diagnostics through LAPACK `gtsv`/`gttrf`/`gttrs`/`gtcon`/`gtrfs`/`gtsvx`. |
| Dense real reciprocal condition estimate | yes | yes | n/a | n/a | Dense projected one-norm reciprocal condition estimate through LAPACK `gecon`. |
| Dense real triangular solve | yes | yes | n/a | n/a | Dense projected triangular solve through LAPACK `trtrs`. |
| Dense real triangular refined solve | yes | yes | n/a | n/a | Dense projected triangular solve through LAPACK `trtrs` followed by `trrfs`, returning forward/backward error estimates. |
| Dense real triangular inverse | yes | yes | n/a | n/a | Dense projected triangular inverse through LAPACK `trtri`. |
| Dense real triangular reciprocal condition estimate | yes | yes | n/a | n/a | Dense projected triangular one-norm reciprocal condition estimate through LAPACK `trcon`. |
| Dense real Sylvester equation solve | yes | yes | n/a | n/a | Dense projected Sylvester equation solve through LAPACK `trsyl`, returning LAPACK's scale factor and perturbation diagnostic. |
| Dense real inverse | yes | yes | n/a | n/a | Dense projected utility inverse through LAPACK `getrf`/`getri`. |
| Dense real least-squares solve | yes | yes | n/a | n/a | Dense projected least-squares/minimum-norm solve through LAPACK `gels`. |
| Dense real SVD least-squares solve | yes | yes | n/a | n/a | Dense projected least-squares solve with singular values and rank through LAPACK `gelss`; default rank threshold is `100 * uni20::numeric_limits<Scalar>::epsilon()`. |
| Dense real divide-and-conquer SVD least-squares solve | yes | yes | n/a | n/a | Dense projected least-squares solve with singular values and rank through LAPACK `gelsd`; default rank threshold is `100 * uni20::numeric_limits<Scalar>::epsilon()`. |
| Dense real rank-revealing least-squares solve | yes | yes | n/a | n/a | Dense projected least-squares solve with column pivoting through LAPACK `gelsy`; default rank threshold is `100 * uni20::numeric_limits<Scalar>::epsilon()`. |
| Dense real symmetric positive definite solve | yes | yes | n/a | n/a | Dense projected SPD solve through LAPACK `potrf`/`potrs`. |
| Dense real symmetric positive definite factorization and solve | yes | yes | n/a | n/a | Dense projected reusable SPD Cholesky solve through LAPACK `potrf`/`potrs`. |
| Dense real symmetric positive definite band solve, condition, and refinement | yes | yes | n/a | n/a | Dense projected reusable SPD band Cholesky solve plus diagnostics through LAPACK `pbsv`/`pbtrf`/`pbtrs`/`pbcon`/`pbrfs`/`pbsvx`. |
| Dense real symmetric positive definite tridiagonal solve, condition, and refinement | yes | yes | n/a | n/a | Dense projected reusable SPD tridiagonal solve plus diagnostics through LAPACK `ptsv`/`pttrf`/`pttrs`/`ptcon`/`ptrfs`/`ptsvx`. |
| Dense real symmetric positive definite refined solve | yes | yes | n/a | n/a | Dense projected SPD solve through LAPACK `potrf`/`potrs` followed by `porfs`, returning forward/backward error estimates. |
| Dense real symmetric positive definite expert solve | yes | yes | n/a | n/a | Dense projected SPD expert solve with condition and forward/backward error diagnostics through LAPACK `posvx`. |
| Dense real symmetric positive definite equilibration | yes | yes | n/a | n/a | Dense projected SPD symmetric scaling diagnostics through LAPACK `poequ`. |
| Dense real symmetric positive definite reciprocal condition estimate | yes | yes | n/a | n/a | Dense projected SPD one-norm reciprocal condition estimate through LAPACK `potrf`/`pocon`. |
| Dense real symmetric positive definite factorized reciprocal condition estimate | yes | yes | n/a | n/a | Dense projected SPD one-norm reciprocal condition estimate from reusable Cholesky factors through LAPACK `pocon`. |
| Dense real symmetric positive definite inverse | yes | yes | n/a | n/a | Dense projected SPD inverse through LAPACK `potrf`/`potri`. |
| Dense real pivoted Cholesky factorization | yes | yes | n/a | n/a | Dense projected positive-semidefinite rank-revealing Cholesky factorization through LAPACK `pstrf`. |
| Dense real symmetric indefinite solve | yes | yes | n/a | n/a | Dense projected symmetric indefinite solve through LAPACK `sytrf`/`sytrs`. |
| Dense real symmetric indefinite factorization and solve | yes | yes | n/a | n/a | Dense projected reusable Bunch-Kaufman solve through LAPACK `sytrf`/`sytrs`. |
| Dense real symmetric indefinite refined solve | yes | yes | n/a | n/a | Dense projected symmetric indefinite solve through LAPACK `sytrf`/`sytrs` followed by `syrfs`, returning forward/backward error estimates. |
| Dense real symmetric indefinite inverse | yes | yes | n/a | n/a | Dense projected symmetric indefinite inverse through LAPACK `sytrf`/`sytri`. |
| Dense real symmetric indefinite expert solve | yes | yes | n/a | n/a | Dense projected symmetric indefinite expert solve with condition and forward/backward error diagnostics through LAPACK `sysvx`. |
| Dense real symmetric indefinite reciprocal condition estimate | yes | yes | n/a | n/a | Dense projected symmetric indefinite one-norm reciprocal condition estimate through LAPACK `sytrf`/`sycon`. |
| Dense real symmetric indefinite factorized reciprocal condition estimate | yes | yes | n/a | n/a | Dense projected symmetric indefinite one-norm reciprocal condition estimate from reusable Bunch-Kaufman factors through LAPACK `sycon`. |
| Dense real compact QR factor application | yes | yes | n/a | n/a | Dense projected Householder QR factor application through LAPACK `ormqr`. |
| Dense real QR factorization | yes | yes | n/a | n/a | Dense projected reduced QR through LAPACK `geqrf`/`orgqr`. |
| Dense real compact LQ factor application | yes | yes | n/a | n/a | Dense projected Householder LQ factor application through LAPACK `ormlq`. |
| Dense real LQ factorization | yes | yes | n/a | n/a | Dense projected reduced LQ through LAPACK `gelqf`/`orglq`. |
| Dense real compact QL factor application | yes | yes | n/a | n/a | Dense projected Householder QL factor application through LAPACK `ormql`. |
| Dense real QL factorization | yes | yes | n/a | n/a | Dense projected reduced QL through LAPACK `geqlf`/`orgql`. |
| Dense real compact RQ factor application | yes | yes | n/a | n/a | Dense projected Householder RQ factor application through LAPACK `ormrq`. |
| Dense real RQ factorization | yes | yes | n/a | n/a | Dense projected reduced RQ through LAPACK `gerqf`/`orgrq`. |
| Dense real bidiagonal reduction | yes | yes | n/a | n/a | Dense projected bidiagonal reduction and orthogonal-factor materialization through LAPACK `gebrd`/`orgbr`. |
| Dense real bidiagonal orthogonal-factor application | yes | yes | n/a | n/a | Dense projected bidiagonal Householder factor application through LAPACK `ormbr`. |
| Dense real bidiagonal singular values | yes | yes | n/a | n/a | Dense projected bidiagonal singular values through LAPACK `bdsqr`. |
| Dense real bidiagonal divide-and-conquer singular value decomposition | yes | yes | n/a | n/a | Dense projected bidiagonal SVD through LAPACK `bdsdc`, with optional explicit singular vectors. |
| Dense selected real bidiagonal singular value decomposition | yes | yes | n/a | n/a | Dense projected selected bidiagonal SVD through LAPACK `bdsvdx`, using 0-based inclusive decreasing singular-value index ranges. |
| Dense real pivoted QR factorization | yes | yes | n/a | n/a | Dense projected reduced QR with column pivoting through LAPACK `geqp3` and `orgqr`. |
| Dense real Hessenberg reduction | yes | yes | n/a | n/a | Dense projected orthogonal Hessenberg reduction through LAPACK `gehrd`/`orghr`. |
| Dense real Hessenberg orthogonal-factor application | yes | yes | n/a | n/a | Dense projected Hessenberg Householder factor application through LAPACK `ormhr`. |
| Dense real symmetric tridiagonal reduction | yes | yes | n/a | n/a | Dense projected symmetric tridiagonal reduction and Householder factor application through LAPACK `sytrd`/`orgtr`/`ormtr`. |
| Dense symmetric tridiagonal eigenvalues | yes | yes | n/a | n/a | Dense projected symmetric tridiagonal eigenvalues through LAPACK `sterf`. |
| Dense symmetric tridiagonal divide-and-conquer eigensystem | yes | yes | n/a | n/a | Dense projected symmetric tridiagonal eigensystem through LAPACK `stevd`. |
| Dense selected symmetric tridiagonal eigensystem | yes | yes | n/a | n/a | Dense projected symmetric tridiagonal eigensystem through LAPACK `stevr` using 0-based inclusive index ranges. |
| Dense real singular value decomposition | yes | yes | n/a | n/a | Dense projected utility SVD through LAPACK `gesvd`. |
| Dense real divide-and-conquer singular value decomposition | yes | yes | n/a | n/a | Dense projected utility SVD through LAPACK `gesdd`. |
| Dense selected real singular value decomposition | yes | yes | n/a | n/a | Dense projected selected SVD through LAPACK `gesvdx`, using 0-based inclusive decreasing singular-value index ranges. |
| Dense real symmetric eigensystem | yes | yes | n/a | n/a | Dense projected Hermitian reference or utility problem. |
| Dense real divide-and-conquer symmetric eigensystem | yes | yes | n/a | n/a | Dense projected Hermitian reference or utility problem through LAPACK `syevd`. |
| Dense real selected symmetric eigensystem | yes | yes | n/a | n/a | Dense projected Hermitian reference or utility problem through LAPACK `syevr` using 0-based inclusive index ranges. |
| Dense generalized real symmetric eigensystem | yes | yes | n/a | n/a | Dense type-1 generalized Hermitian utility problem through LAPACK `sygv`. |
| Dense generalized real divide-and-conquer symmetric eigensystem | yes | yes | n/a | n/a | Dense type-1 generalized Hermitian utility problem through LAPACK `sygvd`. |
| Dense generalized real selected symmetric eigensystem | yes | yes | n/a | n/a | Dense type-1 generalized Hermitian utility problem through LAPACK `sygvx` using 0-based inclusive index ranges. |
| Dense complex Hermitian eigensystem | n/a | n/a | yes | yes | Dense projected Hermitian reference or utility problem through LAPACK `heev`. |
| Dense complex divide-and-conquer Hermitian eigensystem | n/a | n/a | yes | yes | Dense projected Hermitian reference or utility problem through LAPACK `heevd`. |
| Dense complex selected Hermitian eigensystem | n/a | n/a | yes | yes | Dense projected Hermitian reference or utility problem through LAPACK `heevr` using 0-based inclusive index ranges. |
| Dense generalized complex Hermitian eigensystem | n/a | n/a | yes | yes | Dense type-1 generalized Hermitian utility problem through LAPACK `hegv`. |
| Dense generalized complex divide-and-conquer Hermitian eigensystem | n/a | n/a | yes | yes | Dense type-1 generalized Hermitian utility problem through LAPACK `hegvd`. |
| Dense generalized complex selected Hermitian eigensystem | n/a | n/a | yes | yes | Dense type-1 generalized Hermitian utility problem through LAPACK `hegvx` using 0-based inclusive index ranges. |
| Symmetric tridiagonal eigensystem | yes | yes | n/a | n/a | Lanczos projected eigensystem through LAPACK `sterf` for eigenvalues-only and `steqr` for eigenvectors. |
| Real nonsymmetric eigensystem | yes | yes | n/a | n/a | Arnoldi Ritz extraction in real arithmetic. |
| Real nonsymmetric expert eigensystem | yes | yes | n/a | n/a | Dense projected real nonsymmetric eigensystem through LAPACK `geevx`, returning balancing data and reciprocal eigenvalue/eigenvector condition estimates. |
| Real nonsymmetric balancing and right-vector backtransform | yes | yes | n/a | n/a | Dense projected real nonsymmetric balancing helpers through LAPACK `gebal` and `gebak`. |
| Generalized real nonsymmetric eigensystem | yes | yes | n/a | n/a | Dense projected real nonsymmetric matrix pencil through LAPACK `ggev`, preserving `alpha`/`beta` data and finite ratios. |
| Generalized real nonsymmetric expert eigensystem | yes | yes | n/a | n/a | Dense projected real nonsymmetric matrix pencil through LAPACK `ggevx`, preserving `alpha`/`beta` data and returning balancing data and reciprocal condition estimates. |
| Generalized real nonsymmetric balancing and right-vector backtransform | yes | yes | n/a | n/a | Dense projected real nonsymmetric pencil balancing helpers through LAPACK `ggbal` and `ggbak`. |
| Generalized real Hessenberg reduction | yes | yes | n/a | n/a | Dense projected real QZ building block through LAPACK `gghrd`, assuming the second matrix is already upper triangular. |
| Generalized real Hessenberg Schur factorization | yes | yes | n/a | n/a | Dense projected real QZ iteration through LAPACK `hgeqz`, assuming generalized Hessenberg/upper-triangular input. |
| Generalized real Schur factorization | yes | yes | n/a | n/a | Dense projected real generalized Schur/QZ factorization through LAPACK `gges`, preserving `alpha`/`beta` data and finite ratios. |
| Generalized real Schur reordering | yes | yes | n/a | n/a | Dense projected real generalized Schur/QZ block reordering through LAPACK `tgexc`, preserving aligned `alpha`/`beta` data. |
| Generalized real Schur selected subspace condition estimates | yes | yes | n/a | n/a | Dense projected real generalized Schur/QZ selected deflating subspace through LAPACK `tgsen`, returning `PL`, `PR`, and F-norm `DIF` estimates. |
| Generalized real Schur right eigenvectors | yes | yes | n/a | n/a | Dense projected real generalized Schur/QZ eigenvector extraction through LAPACK `tgevc`, unpacked into complex columns. |
| Generalized real Schur eigenpair condition estimates | yes | yes | n/a | n/a | Dense projected real generalized Schur/QZ eigenvalue/eigenvector conditioning through LAPACK `tgsna`. |
| Real Schur factorization and reordering | yes | yes | n/a | n/a | Real nonsymmetric implicit restart. |
| Real Hessenberg Schur factorization | yes | yes | n/a | n/a | Dense projected upper-Hessenberg-to-real-Schur factorization through LAPACK `hseqr`. |
| Real Schur selected subspace condition estimates | yes | yes | n/a | n/a | Dense projected real Schur block selection through LAPACK `trsen`, returning reciprocal eigenvalue-cluster and invariant-subspace condition estimates. |
| Real Schur right eigenvectors | yes | yes | n/a | n/a | Dense projected real Schur eigenvector extraction through LAPACK `trevc`, unpacked into complex columns. |
| Real Schur eigenpair condition estimates | yes | yes | n/a | n/a | Dense projected real Schur eigenvalue/eigenvector conditioning through LAPACK `trsna`. |
| Complex nonsymmetric eigensystem | n/a | n/a | yes | yes | Arnoldi Ritz extraction in complex arithmetic. |
| Complex Schur factorization and reordering | n/a | n/a | yes | yes | Complex nonsymmetric implicit restart. |

### Optional MPLAPACK Binary128 Inventory

These paths are available only when `UNI20_ENABLE_MPLAPACK=ON` and the
configured MPLAPACK package provides a real binary128 type. This table records
the intended scalar coverage of the optional binary128 surface; the separate
precision-validation matrix records which rows have dedicated stress tests.

At the native solver entry-point level, binary128 mostly mirrors the ordinary
`s`, `d`, `c`, and `z` coverage: real paths have `f128` analogues, complex
paths have `cf128` analogues, and Hermitian paths support both. The broad dense
projected helper inventory is intentionally split from the default-facing
headers where possible, but it remains in-tree and is tested by the gated
`MplapackBinary128DenseSubspaceTest` target. These wrappers are early
infrastructure for the future mdspan/rank-2 tensor layer, not a stable public
API commitment.

| component or entry point | `f128` | `cf128` | notes |
| --- | --- | --- | --- |
| Dense vector and matrix primitives | yes | yes | Scalar-generic Krylov host-side helpers. |
| MPBLAS wrapper surface | yes | yes | Current wrapper surface covers projected `gemm`, `gemv`, rank-update, symmetric-rank, and Hermitian-rank operations used by active paths. |
| Tensor/linalg CPU helper probes | yes | n/a | Current probes cover real one-norm accumulation and real dense solve. |
| Broad dense projected real helper inventory | yes | n/a | Gated tests cover norms, dense/band/tridiagonal solves and diagnostics, SPD and symmetric-indefinite helpers, QR/LQ/QL/RQ, bidiagonal/SVD helpers, real symmetric/generalized symmetric eigensystems, and real nonsymmetric/Schur/QZ helpers. |
| Dense projected complex eigensystem and Schur helper inventory | n/a | yes | Gated tests cover complex Hermitian/generalized Hermitian eigensystems, complex nonsymmetric eigensystems, and complex Schur/reordering. |
| Symmetric tridiagonal projected eigensystem | yes | n/a | Uses MPLAPACK `Rsterf`/`Rsteqr`; this is the projected problem behind real and complex Hermitian Lanczos. |
| Real nonsymmetric projected eigensystem and Schur kernels | yes | n/a | Active wrapper surface covers `Rgeev`, `Rgees`, `Rhseqr`, and `Rtrexc`. |
| Complex nonsymmetric projected eigensystem and Schur kernels | n/a | yes | Active wrapper surface covers `Cgeev`, `Cgees`, and `Ctrexc`. |
| `symmetric_lanczos_standard` | yes | yes | `f128` is directly probed; `cf128` follows the same real tridiagonal projected eigensystem path but does not yet have a dedicated binary128 stress test. |
| `symmetric_lanczos_restarted_standard` | yes | yes | Templated path; no dedicated binary128 restart stress test yet. |
| `symmetric_lanczos_restarted_transformed` | yes | yes | Templated transformed path; no dedicated binary128 stress test yet. |
| `symmetric_lanczos_restarted_generalized_transformed` | yes | yes | Templated generalized transformed path; no dedicated binary128 stress test yet. |
| `real_nonsymmetric_arnoldi_standard` | yes | n/a | Full-projection real binary128 Arnoldi is directly probed. |
| `real_nonsymmetric_arnoldi_restarted_standard` | yes | n/a | Restarted path uses the active real Schur wrapper surface; no dedicated binary128 restart stress test yet. |
| `complex_nonsymmetric_arnoldi_standard` | n/a | yes | Full-projection complex binary128 Arnoldi is directly probed; restart uses the active complex Schur wrapper surface. |
| `hermitian_krylov_exponential_action` | yes | yes | `f128` is directly probed; `cf128` uses the binary128 complex dense exponential instantiations but does not yet have a dedicated stress test. |
| `nonsymmetric_krylov_exponential_action` | yes | yes | Templated path backed by binary128 real and complex dense exponential instantiations; no dedicated binary128 stress test yet. |
| `taylor_exponential_action` | yes | yes | Scalar-generic validation path; no dedicated binary128 stress test yet. |

### Symmetric And Hermitian Lanczos

The native symmetric/Hermitian solver supports real symmetric and complex
Hermitian vector spaces. The Lanczos basis uses the vector-field scalar, while
the projected tridiagonal problem, Ritz values, residual bounds, tolerances, and
spectral-transform metadata use the corresponding real scalar.

| entry point | `s` | `d` | `c` | `z` | description |
| --- | --- | --- | --- | --- | --- |
| `symmetric_lanczos_standard` | yes | yes | yes | yes | Full-projection Lanczos solve without restart. |
| `symmetric_lanczos_restarted_standard` | yes | yes | yes | yes | Restarted Lanczos in regular standard mode. |
| `symmetric_lanczos_restarted_transformed` | yes | yes | yes | yes | Restarted Lanczos on caller-supplied transformed operator. |
| `symmetric_lanczos_restarted_generalized_transformed` | yes | yes | yes | yes | Restarted generalized path using a `B` metric. |

The transformed path supports the following eigenvalue maps:

| transform | projected eigenvalue `theta` maps to physical eigenvalue |
| --- | --- |
| `Regular` | `lambda = theta` |
| `ShiftInvert` | `lambda = sigma + 1 / theta` |
| `Buckling` | `lambda = sigma * theta / (theta - 1)` |
| `Cayley` | `lambda = sigma * (theta + 1) / (theta - 1)` |
| `GeneralizedInverse` | `lambda = theta` |

### Nonsymmetric Arnoldi

| entry point | `s` | `d` | `c` | `z` | description |
| --- | --- | --- | --- | --- | --- |
| `real_nonsymmetric_arnoldi_standard` | yes | yes | n/a | n/a | Full-projection real Arnoldi solve. |
| `real_nonsymmetric_arnoldi_restarted_standard` | yes | yes | n/a | n/a | Restarted real Arnoldi using real Schur data. |
| `complex_nonsymmetric_arnoldi_standard` | n/a | n/a | yes | yes | Complex Arnoldi solve; uses complex Schur restart when restart is needed. |

Real nonsymmetric solves classify selected Ritz values with
`RealNonsymmetricPolicy`. The default policy is `RequireRealEigenpairs`, which
accepts only numerically real selected eigenpairs. Complex-pair handling in real
vector output remains an interface design point; complex arithmetic is the
available path when complex eigenpairs are required.

### Krylov Exponential Actions

Matrix-free exponential actions approximate `exp(t A) v` without exposing
application-space vector storage. They are not ARPACK replacement routines, but
they reuse the same opaque vector-operation boundary and local dense projected
matrix infrastructure.

| entry point | `s` | `d` | `c` | `z` | description |
| --- | --- | --- | --- | --- | --- |
| `hermitian_krylov_exponential_action` | yes | yes | yes | yes | Fixed-subspace Lanczos approximation `||v|| V_m exp(t T_m) e_1`. |
| `nonsymmetric_krylov_exponential_action` | yes | yes | yes | yes | Fixed-subspace Arnoldi approximation `||v|| V_m exp(t H_m) e_1`. |
| `taylor_exponential_action` | yes | yes | yes | yes | Validation-oriented scaled Taylor action using a caller-supplied operator norm bound. |

The current projected exponential backend is the CPU dense
scaling-and-squaring Padé implementation. It accepts real multipliers and
complex multipliers; a real projected matrix with a complex multiplier promotes
to a complex projected exponential. The Taylor action is an independent
reference/emergency path, not an automatic fallback from Lanczos or Arnoldi. It
uses scaled Taylor series steps with a geometric tail estimate and an explicit
caller-supplied bound for `||A||`. Taylor scaling-step selection and
tail-error amplification are computed in the solver real scalar type, so
extended-precision real types are not silently rounded through `long double`.
`examples/krylov/krylov_exponential_probe_example.cpp` compares Lanczos, Taylor, and
exact diagonal exponential actions on deliberately awkward spectra. It reports
both an example-local full-reorthogonalized Lanczos recurrence, mirroring the
production path while exposing projected data, and an example-local legacy
three-term recurrence so precision-floor and loss-of-orthogonality effects can
be inspected side by side. The probe distinguishes the raw last
projected exponential coefficient, matching the historical Cytnx
`abs(B_mat(i,0))` stopping indicator, from the residual-scaled Krylov estimate
used by the native result. It also reports Hermite-quadrature, Saad/Jia-Lv
`phi_1`, and Hochbruck-Lubich/Jawecki-style leading-bound estimates; see
[krylov_exponential_estimators.md](krylov_exponential_estimators.md). Its
rebound diagnostics are intended to expose cases where asking for an
unrealistically small tolerance keeps increasing the Krylov dimension after the
true action error has already reached the scalar precision floor. The probe
also reports the final basis off-diagonal Gram defect and the largest
reorthogonalization correction ratio, which help distinguish estimator
underflow from loss of useful Lanczos orthogonality. Full Al-Mohy/Higham-style
norm estimation, block right-hand sides, adaptive Lanczos time stepping,
restart/error control, and broader benchmark examples are still future work.

`examples/krylov/krylov_exponential_orthogonality_example.cpp` is a smaller teaching
example focused on this failure mode. Its default float run requests
`tol=1e-8`, below the `sqrt(eps(float))` warning scale that motivated the old
TDVP/Cytnx policy for non-reorthogonalized or lightly reorthogonalized Lanczos.
It shows a case where the residual estimate falls below the requested tolerance
while the exact diagonal reference error has already saturated. The same table
contrasts the full-reorthogonalized production path with a legacy three-term
recurrence, making the Gram defect and reorthogonalization correction pressure
visible. This warning scale is not a hard limit for full-reorthogonalized double
precision; it is a conservative default for Lanczos exponential actions that do
not maintain or monitor basis orthogonality strongly enough to certify tighter
tolerances.

`examples/krylov/krylov_exponential_matrix_market_probe_example.cpp` is the real
Matrix Market stress case for this issue. Its default input is the copied TDVP
harmonic-oscillator fixture in `tests/krylov/matrix_market/tdvp_lanczos/`, with
the initial vector from the same directory and the old problematic coefficient
`t = -0.1968473663975394 i`. In `complex<double>`, the default run shows the
old raw projected-tail indicator falling below `1e-8` while the independent
Taylor reference action error is still above `1e-8`. This example is a
convergence diagnostic and benchmark aid; it is not a production fallback
policy.

Real vector spaces accept real time/coefficient values. Complex vector spaces
also accept complex time values, so unitary actions such as `exp(-i t H) v` can
use the Hermitian/Lanczos path without hiding the phase in the matrix-free
operator.

### Relationship To ARPACK

The native eigensolvers intentionally follow ARPACK's core IRLM/IRAM ideas:
Lanczos or Arnoldi expansion, Rayleigh-Ritz or Schur extraction, implicit QR
shift filtering, and restart from the compressed Krylov factorization. The
implementation is not a wrapper around ARPACK. It uses Uni20's matrix-free
interface, native result/diagnostic types, full reentrancy, and no Fortran work
arrays or reverse communication.

ARPACK remains useful as an external behavior and performance oracle. Vendored
ARPACK adapters, imported upstream tests, broad Matrix Market comparison
suites, and convergence dashboards should live in a separate validation and
benchmarking repository rather than in core Uni20.

## Public Parameters

### SymmetricEigenParams<Scalar>

| parameter | default | applies to | meaning |
| --- | --- | --- | --- |
| `eigenvalue_count` | `1` | all symmetric paths | Number of requested converged eigenpairs, `nev`. |
| `retained_ritz_count` | `0` | restarted native paths | Restart dimension, `nkeep`; zero selects the default policy. |
| `krylov_dimension` | `0` | all symmetric paths | Lanczos subspace dimension, `ncv`; zero selects the default policy. |
| `max_iterations` | `300` | restarted native paths | Restart-cycle budget. |
| `tolerance` | `0` | all symmetric paths | Ritz convergence tolerance; zero selects `100 * epsilon`. |
| `breakdown_tolerance` | `0` | native symmetric paths | Internal invariant-subspace threshold; zero selects `10 * epsilon` scaled by the local action/recurrence scale. |
| `spectrum` | `LargestMagnitude` | all symmetric paths | Wanted part of the spectrum. |
| `compute_eigenvectors` | `true` | all symmetric paths | Whether to return Ritz vector approximations. |
| `diagnostics` | `None` | all symmetric paths | Optional projected-problem and restart diagnostics. |

Supported symmetric selectors are `LargestMagnitude`, `SmallestMagnitude`,
`LargestAlgebraic`, `SmallestAlgebraic`, and `BothEnds`.

### SymmetricTransformOptions<Scalar>

| parameter | default | meaning |
| --- | --- | --- |
| `transform` | `Regular` | Spectral transform used to map projected Ritz values back to physical eigenvalues. |
| `sigma` | `0` | Shift for shift-invert, buckling, and Cayley transforms. |

The caller supplies the transformed operation. The options only describe how to
map converged Ritz values and residual diagnostics back to the original problem.

### NonsymmetricEigenParams<Scalar>

| parameter | default | applies to | meaning |
| --- | --- | --- | --- |
| `eigenvalue_count` | `1` | all nonsymmetric paths | Number of requested converged eigenpairs, `nev`. |
| `retained_ritz_count` | `0` | restarted native paths | Restart dimension, `nkeep`; zero selects the default policy. |
| `krylov_dimension` | `0` | all nonsymmetric paths | Arnoldi subspace dimension, `ncv`; zero selects the default policy. |
| `max_iterations` | `300` | restarted native paths | Matvec budget for native nonsymmetric restarted solves. |
| `tolerance` | `0` | all nonsymmetric paths | Ritz convergence tolerance; zero selects `100 * epsilon`. |
| `complex_pair_tolerance` | `0` | real nonsymmetric paths | Reality classification tolerance; zero selects `sqrt(epsilon)`. |
| `spectrum` | `LargestMagnitude` | all nonsymmetric paths | Wanted part of the spectrum. |
| `compute_eigenvectors` | `true` | all nonsymmetric paths | Whether to return right Ritz vector approximations. |
| `real_policy` | `RequireRealEigenpairs` | real nonsymmetric paths | Policy for selected complex Ritz values in real arithmetic. |
| `diagnostics` | `None` | all nonsymmetric paths | Optional projected-problem and restart diagnostics. |

Supported nonsymmetric selectors are `LargestMagnitude`, `SmallestMagnitude`,
`LargestReal`, `SmallestReal`, `LargestImaginary`, and `SmallestImaginary`.
Algebraic and both-ends selectors are symmetric-only.

### KrylovExponentialParams<Scalar>

| parameter | default | applies to | meaning |
| --- | --- | --- | --- |
| `krylov_dimension` | `0` | Hermitian and nonsymmetric exponential actions | Projection dimension; zero selects the default policy. |
| `breakdown_tolerance` | `0` | Hermitian and nonsymmetric exponential actions | Internal invariant-subspace threshold; zero selects `10 * epsilon`. |
| `diagnostics` | `None` | Hermitian and nonsymmetric exponential actions | Optional projected-dimension, residual, and matvec diagnostics. |

### TaylorExponentialParams<Scalar>

| parameter | default | applies to | meaning |
| --- | --- | --- | --- |
| `tolerance` | `0` | Taylor exponential action | Absolute action tolerance; zero selects `100 * epsilon`. |
| `step_norm_limit` | `0.5` | Taylor exponential action | Target upper bound for `||h A||` in each scaled Taylor step. |
| `max_taylor_degree` | `200` | Taylor exponential action | Maximum Taylor degree within one scaled step. |
| `max_scaling_steps` | `100000` | Taylor exponential action | Maximum number of scaled Taylor steps. |
| `throw_on_nonconvergence` | `true` | Taylor exponential action | Throw if any scaled step cannot satisfy the tail bound. |
| `diagnostics` | `None` | Taylor exponential action | Optional scaling, Taylor-degree, norm-bound, and tail-estimate diagnostics. |

## Default Dimension Policy

Explicit user parameters always win. Defaults are used only when
`krylov_dimension == 0` or `retained_ritz_count == 0`.

| family | default `ncv` | default `nkeep` |
| --- | --- | --- |
| Symmetric Lanczos | `min(problem_dimension, max(20, 2*nev + 1))` | `nev` |
| Nonsymmetric Arnoldi | `min(problem_dimension, max(20, 6*nev + 8))` | `min(ncv - 1, max(2*nev, nev + 4))` |
| Exponential action | `min(problem_dimension, 30)` | n/a |
| Taylor exponential action | n/a | n/a |

Restarted native solvers require:

```text
nev <= nkeep < ncv <= problem_dimension
```

## Internal Tuning Parameters

These values are implementation details, but they affect convergence behavior
and should be changed deliberately.

| tuning value | current value | applies to | purpose |
| --- | --- | --- | --- |
| Effective Ritz tolerance when user tolerance is zero | `100 * epsilon` | symmetric and nonsymmetric | ARPACK-style automatic tolerance floor. |
| Residual convergence scale | `tol * max(epsilon^(2/3), abs(theta))` | symmetric and nonsymmetric | Avoids impossible absolute accuracy near zero Ritz values. |
| Symmetric/Lanczos happy-breakdown threshold | `breakdown_tolerance * max(1, local action/recurrence scale)`; default `10 * epsilon` | symmetric Lanczos | Treats a tiny residual expansion vector as invariant-subspace breakdown independently of Ritz convergence tolerance. |
| Arnoldi happy-breakdown threshold | `breakdown_tolerance * max(1, local Arnoldi relation scale)`; default `10 * epsilon` | nonsymmetric Arnoldi | Treats a tiny residual expansion vector as invariant-subspace breakdown. |
| Arnoldi orthogonalization passes | one pass, with a second DGKS-style pass after a `0.717` shrink test | nonsymmetric Arnoldi | Keeps full reorthogonalization when needed without unconditionally doubling the orthogonalization work. |
| Lanczos residual reorthogonalization threshold | `0.717` shrink test | symmetric Lanczos | Controls whether an additional reorthogonalization pass is taken. |
| Symmetric implicit QR deflation threshold | `epsilon * (abs(d_i) + abs(d_{i+1}))` | symmetric restart | Splits negligible tridiagonal subdiagonal entries. |
| Default complex-pair tolerance | `sqrt(epsilon)` | real nonsymmetric classification | Classifies nearly real Ritz values. |
| Ambiguous complex-pair band | `10 * complex_pair_tolerance` | real nonsymmetric classification | Separates numerically real, ambiguous, and complex Ritz values. |
| Symmetric restart shift count | `order - retained_count` | symmetric restart | Number of unwanted Ritz values used as implicit shifts. |
| Exponential action breakdown threshold | `10 * epsilon` | fixed-subspace exponential actions | Treats a tiny expansion residual as invariant-subspace breakdown. |
| Taylor action step norm limit | `0.5` | Taylor exponential action | Keeps each scaled Taylor step in the monotone tail-bound regime. |
| Taylor action default tolerance | `100 * epsilon` | Taylor exponential action | Absolute tail tolerance when the user does not request one. |

The symmetric restart planner also protects numerically exact invariant
directions: unwanted Ritz values with zero residual bounds may force a larger
retained count rather than being used as shifts. When too few requested Ritz
values have converged, it may retain extra guard vectors to avoid an overly
aggressive restart.

## Results And Diagnostics

Symmetric results report eigenvalues, residual bounds, optional eigenvectors,
converged count, iteration count, matvec count, status, and optional
diagnostics. `matvec_count` is the stable operator-work counter. The exact
meaning of `iteration_count` is solver-family specific; for restarted symmetric
Lanczos it is the number of restart cycles completed.

Nonsymmetric results report complex eigenvalues, residual bounds, optional right
eigenvectors, real/ambiguous/complex classification for real arithmetic solves,
converged count, iteration count, matvec count, status, and optional
diagnostics.

Diagnostics can include projected eigenvalues, Ritz residual bounds, final
Lanczos or Arnoldi residual norm, restart counts, and projected departure from
normality where implemented.

Exponential action results report the action vector, projected dimension,
matvec count, initial norm, final Krylov residual norm, an inexpensive residual
estimate, happy-breakdown state, and optional summary diagnostics. Hermitian
exponential diagnostics additionally include final-basis Gram defects and
reorthogonalization correction metrics when diagnostics are enabled.

Taylor exponential action results report the action vector, scaling step count,
maximum Taylor degree used, matvec count, estimated tail error, convergence
state, and optional diagnostics containing the supplied operator-norm bound and
final step tail estimate.

## Known Gaps

- Real nonsymmetric complex-pair output through real Schur two-planes is not a
  finished user-facing result path.
- Krylov exponential actions are currently fixed-subspace only. A conservative
  Taylor validation path and diagonal probe example exist, but adaptive
  Krylov-exponential time stepping, restart/error control,
  Al-Mohy/Higham-style norm estimation, and broader benchmark examples remain
  future work.
- Type-specific stress hardening for `s`, `d`, `c`, and `z` remains ongoing.
