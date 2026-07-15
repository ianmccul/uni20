# Dense BLAS/LAPACK Wrapper Coverage

This document inventories dense BLAS/LAPACK components that were developed for
projected Krylov problems and broader dense linear-algebra experiments. It is
a provider/wrapper coverage survey, not the Krylov algorithm contract.

Most broad utility rows below are retained in
[`dense_subspace_unused.hpp`](../../src/uni20/krylov/dense_subspace_unused.hpp).
That header is quarantined and excluded from the active Krylov target, so those
rows do not define a supported linalg API. Rows explicitly described as
dispatched are active dependencies of the maintained Krylov path and lower
through the [dense linalg layer](../../src/uni20/linalg/).

Keep this inventory synchronized with provider declarations and with any
helpers promoted from the quarantined survey into operation-tag linalg
backends.

## Scalar Tags

LAPACK-style scalar tags are used throughout. Complex scalar spellings follow
the project [Scalar Policy](../tensor/scalar_policy.md):

| tag | scalar type |
| --- | --- |
| `s` | `float` |
| `d` | `double` |
| `c` | `uni20::complex<float>` |
| `z` | `uni20::complex<double>` |
| `f128` | optional `uni20::float128` |
| `cf128` | optional `uni20::complex<uni20::float128>` |

## Ordinary Precision Inventory

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
| Symmetric tridiagonal eigensystem | yes | yes | n/a | n/a | Dispatched Lanczos projected eigensystem through LAPACK `sterf` for eigenvalues-only and `steqr` for eigenvectors. |
| Real nonsymmetric eigensystem | yes | yes | n/a | n/a | Dispatched Arnoldi Ritz extraction through LAPACK `geev`, including complex-pair unpacking. |
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
| Real Schur factorization and reordering | yes | yes | n/a | n/a | Dispatched real nonsymmetric implicit restart through LAPACK `gees` and `trexc`. |
| Real Hessenberg Schur factorization | yes | yes | n/a | n/a | Dispatched projected upper-Hessenberg-to-real-Schur factorization through LAPACK `hseqr`. |
| Real Schur selected subspace condition estimates | yes | yes | n/a | n/a | Dense projected real Schur block selection through LAPACK `trsen`, returning reciprocal eigenvalue-cluster and invariant-subspace condition estimates. |
| Real Schur right eigenvectors | yes | yes | n/a | n/a | Dense projected real Schur eigenvector extraction through LAPACK `trevc`, unpacked into complex columns. |
| Real Schur eigenpair condition estimates | yes | yes | n/a | n/a | Dense projected real Schur eigenvalue/eigenvector conditioning through LAPACK `trsna`. |
| Complex nonsymmetric eigensystem | n/a | n/a | yes | yes | Dispatched Arnoldi Ritz extraction through LAPACK `geev`. |
| Complex Schur factorization and reordering | n/a | n/a | yes | yes | Dispatched complex nonsymmetric implicit restart through LAPACK `gees` and `trexc`. |

## Optional MPLAPACK Binary128 Dense Inventory

These rows are available only when `UNI20_ENABLE_MPLAPACK=ON` and the configured
MPLAPACK package provides a real binary128 type. This table covers dense
provider/helper support; algorithm-level binary128 coverage remains in
[Krylov Algorithms](../krylov/algorithms.md).

| component or entry point | `f128` | `cf128` | notes |
| --- | --- | --- | --- |
| Dense vector and matrix primitives | yes | yes | Scalar-generic Krylov host-side helpers. |
| MPBLAS wrapper surface | yes | yes | Current wrapper surface covers projected `gemm`, `gemv`, rank-update, symmetric-rank, and Hermitian-rank operations used by active paths. |
| Tensor/linalg CPU helper probes | yes | n/a | Current probes cover real one-norm accumulation and real dense solve. |
| Broad dense projected real helper inventory | not active | n/a | Quarantined source inventory; excluded from maintained Krylov targets. |
| Dense projected complex eigensystem and Schur helper inventory | n/a | not active | Quarantined source inventory; excluded from maintained Krylov targets. |
| Symmetric tridiagonal projected eigensystem | yes | n/a | Uses MPLAPACK `Rsterf`/`Rsteqr`; this is the projected problem behind real and complex Hermitian Lanczos. |
| Real nonsymmetric projected eigensystem and Schur kernels | yes | n/a | Active wrapper surface covers `Rgeev`, `Rgees`, `Rhseqr`, and `Rtrexc`. |
| Complex nonsymmetric projected eigensystem and Schur kernels | n/a | yes | Active wrapper surface covers `Cgeev`, `Cgees`, and `Ctrexc`. |

## Related Documentation

- [BLAS/LAPACK mdspan wrappers](blas_lapack_wrappers.md)
- [Mdspan linear algebra dispatch](mdspan_dispatch.md)
- [MPLAPACK binary128 setup](mplapack_binary128.md)
- [Krylov precision validation](../krylov/precision_validation.md)
- [Dense linalg source map](../../src/uni20/linalg/)
- [LAPACK provider source map](../../src/uni20/backend/lapack/)
