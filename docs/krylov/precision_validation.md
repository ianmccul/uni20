# Krylov Precision Validation Matrix

This note records which native Krylov precision paths are implemented and
covered by tests. LAPACK-style precision names are used. Complex scalar
spellings follow the project scalar policy in
[Scalar Policy](../tensor/scalar_policy.md):

For the full algorithm inventory, public parameters, defaults, and internal
tuning values, see [Krylov Algorithms](algorithms.md).

| tag | scalar |
| --- | --- |
| `s` | `float` |
| `d` | `double` |
| `c` | `uni20::complex<float>` |
| `z` | `uni20::complex<double>` |
| `f128` | `uni20::float128`, when `UNI20_HAS_FLOAT128=1` |
| `cf128` | `uni20::complex<uni20::float128>`, also spelled `uni20::complex256`, when `UNI20_HAS_FLOAT128=1` |

Experimental binary128 validation is gated by `UNI20_ENABLE_MPLAPACK=ON` and
requires an external MPLAPACK package configured with the binary128 backend.
Uni20 does not download or build MPLAPACK. The optional validation currently
covers scalar aliases and I/O, BLAS/LAPACK wrapper surfaces, tensor linear
algebra probes, broad dense projected helper coverage, real and complex
matrix-free eigensolvers, and projected exponential action. The gated probes
also check that selected scalar-generic helpers use Uni20's real scalar traits
rather than relying only on `std::floating_point`.

Most ordinary Krylov tests are typed through
`tests/krylov/krylov_test_types.hpp`. In a normal build those aliases expand to
the `s`, `d`, `c`, and `z` paths. The MPLAPACK-enabled binary128 coverage is
kept in separate `MplapackBinary128*` targets. The broad dense-subspace wrappers
are retained and tested there, but are not pulled into the ordinary typed solver
matrix.

## Native Solver Coverage

This table records the ordinary default-build coverage for `s`, `d`, `c`, and
`z`. Optional binary128 probes are listed separately below.

| algorithm | `s` | `d` | `c` | `z` | tests |
| --- | --- | --- | --- | --- | --- |
| Dense vector/matrix primitives | yes | yes | yes | yes | `KrylovDenseLinalgTypedTest` |
| Dense real matrix norms | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.ComputesDenseReal*MatrixNorms` |
| Dense matrix exponential, including complex multipliers | yes | yes | yes | yes | `MatrixExponentialTypedTest`, `MatrixExponentialTest.RealMatrixWithComplexMultiplierPromotesToComplex`, `MatrixExponentialTest.ComplexMatrixWithComplexMultiplierUsesComplexTime` |
| Dense real linear solve | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.SolvesDenseRealLinearSystemWithPivoting` |
| Dense real refined linear solve | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.RefinesDenseRealLinearSystemWithDiagnostics` |
| Dense real expert linear solve | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.SolvesDenseRealExpertLinearSystemWithDiagnostics` |
| Dense real equilibration | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.ComputesDenseRealEquilibration` |
| Dense real LU factorization and solve | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.SolvesFromDenseRealLuFactorization` |
| Dense real general-band norm, equilibration, solve, refined/expert solve, and condition estimate | yes | yes | n/a | n/a | ordinary `s`/`d` dense-subspace tests; binary128 coverage is listed in the optional table below |
| Dense real general tridiagonal solve, condition, and refinement | yes | yes | n/a | n/a | ordinary `s`/`d` dense-subspace tests; binary128 coverage is listed in the optional table below |
| Dense real reciprocal condition estimate | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.EstimatesDenseRealReciprocalConditionNumber` |
| Dense real triangular solve | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.SolvesDenseRealTriangularSystem` |
| Dense real triangular refined solve | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.RefinesDenseRealTriangularSystemWithDiagnostics` |
| Dense real triangular inverse | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.InvertsDenseRealTriangularMatrix` |
| Dense real triangular reciprocal condition estimate | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.EstimatesDenseRealTriangularReciprocalConditionNumber` |
| Dense real Sylvester equation solve | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.SolvesDenseRealSylvesterEquation` |
| Dense real inverse | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.InvertsDenseRealMatrix` |
| Dense real least-squares solve | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.SolvesDenseRealLeastSquaresProblem` |
| Dense real SVD least-squares solve | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.SolvesDenseSvdLeastSquaresProblem` |
| Dense real divide-and-conquer SVD least-squares solve | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.SolvesDenseDivideAndConquerSvdLeastSquaresProblem` |
| Dense real rank-revealing least-squares solve | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.SolvesDenseRankRevealingLeastSquaresProblem` |
| Dense real symmetric positive definite solve | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.SolvesDenseSymmetricPositiveDefiniteSystem` |
| Dense real symmetric positive definite factorization and solve | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.SolvesDenseSymmetricPositiveDefiniteSystemFromFactorization` |
| Dense real symmetric positive definite band solve, condition, and refinement | yes | yes | n/a | n/a | ordinary `s`/`d` dense-subspace tests; binary128 coverage is listed in the optional table below |
| Dense real symmetric positive definite tridiagonal solve, condition, and refinement | yes | yes | n/a | n/a | ordinary `s`/`d` dense-subspace tests; binary128 coverage is listed in the optional table below |
| Dense real symmetric positive definite refined solve | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.RefinesDenseSymmetricPositiveDefiniteSystemWithDiagnostics` |
| Dense real symmetric positive definite expert solve | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.SolvesDenseSymmetricPositiveDefiniteExpertSystemWithDiagnostics` |
| Dense real symmetric positive definite equilibration | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.ComputesDenseSymmetricPositiveDefiniteEquilibration` |
| Dense real symmetric positive definite reciprocal condition estimate | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.EstimatesDenseSymmetricPositiveDefiniteReciprocalConditionNumber` |
| Dense real symmetric positive definite factorized reciprocal condition estimate | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.EstimatesDenseSymmetricPositiveDefiniteFactorizedReciprocalConditionNumber` |
| Dense real symmetric positive definite inverse | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.InvertsDenseSymmetricPositiveDefiniteMatrix` |
| Dense real pivoted Cholesky factorization | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.ComputesDensePivotedCholeskyFactorization`, `ReportsRankDeficientPivotedCholeskyFactorization` |
| Dense real symmetric indefinite solve | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.SolvesDenseSymmetricIndefiniteSystem` |
| Dense real symmetric indefinite factorization and solve | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.SolvesDenseSymmetricIndefiniteSystemFromFactorization` |
| Dense real symmetric indefinite refined solve | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.RefinesDenseSymmetricIndefiniteSystemWithDiagnostics` |
| Dense real symmetric indefinite inverse | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.InvertsDenseSymmetricIndefiniteMatrix` |
| Dense real symmetric indefinite expert solve | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.SolvesDenseSymmetricIndefiniteExpertSystemWithDiagnostics` |
| Dense real symmetric indefinite reciprocal condition estimate | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.EstimatesDenseSymmetricIndefiniteReciprocalConditionNumber` |
| Dense real symmetric indefinite factorized reciprocal condition estimate | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.EstimatesDenseSymmetricIndefiniteFactorizedReciprocalConditionNumber` |
| Dense real compact QR factor application | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.AppliesCompactRealQrFactorization` |
| Dense real QR factorization | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.ComputesReducedRealQrFactorization` |
| Dense real compact LQ factor application | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.AppliesCompactRealLqFactorization` |
| Dense real LQ factorization | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.ComputesReducedRealLqFactorization` |
| Dense real compact QL factor application | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.AppliesCompactRealQlFactorization` |
| Dense real QL factorization | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.ComputesReducedRealQlFactorization` |
| Dense real compact RQ factor application | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.AppliesCompactRealRqFactorization` |
| Dense real RQ factorization | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.ComputesReducedRealRqFactorization` |
| Dense real bidiagonal reduction | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.ComputesRealBidiagonalReduction*` |
| Dense real bidiagonal orthogonal-factor application | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.AppliesRealBidiagonalOrthogonalFactors` |
| Dense real bidiagonal singular values | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.ComputesRealBidiagonalSingularValues` |
| Dense real bidiagonal divide-and-conquer singular value decomposition | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.ComputesRealBidiagonalDivideAndConquerSvd` |
| Dense selected real bidiagonal singular value decomposition | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.ComputesSelectedRealBidiagonalSvd` |
| Dense real pivoted QR factorization | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.ComputesReducedRealPivotedQrFactorization` |
| Dense real Hessenberg reduction | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.ComputesRealHessenbergReductionAndOrthogonalFactor` |
| Dense real Hessenberg orthogonal-factor application | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.AppliesRealHessenbergOrthogonalFactor` |
| Dense real symmetric tridiagonal reduction | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.ComputesRealSymmetricTridiagonalReductionAndOrthogonalFactor` |
| Dense symmetric tridiagonal eigenvalues | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.SolvesSymmetricTridiagonalEigenvalues` |
| Dense symmetric tridiagonal divide-and-conquer eigensystem | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.SolvesSymmetricTridiagonalDivideAndConquerEigensystem` |
| Dense selected symmetric tridiagonal eigensystem | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.SolvesSelectedSymmetricTridiagonalEigenvectors` |
| Dense real singular value decomposition | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.ComputesDenseRealSingularValueDecomposition` |
| Dense real divide-and-conquer singular value decomposition | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.ComputesDenseRealDivideAndConquerSingularValueDecomposition` |
| Dense selected real singular value decomposition | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.ComputesSelectedDenseRealSingularValueDecomposition` |
| Dense real symmetric eigensystem | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.SolvesDenseRealSymmetric*` |
| Dense real divide-and-conquer symmetric eigensystem | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.SolvesDenseRealSymmetricDivideAndConquerEigenvectors` |
| Dense real selected symmetric eigensystem | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.SolvesSelectedDenseRealSymmetricEigenvectors` |
| Dense generalized real symmetric eigensystem | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.SolvesDenseGeneralizedRealSymmetricEigenvectors` |
| Dense generalized real divide-and-conquer symmetric eigensystem | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.SolvesDenseGeneralizedRealSymmetricDivideAndConquerEigenvectors` |
| Dense generalized real selected symmetric eigensystem | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.SolvesSelectedDenseGeneralizedRealSymmetricEigenvectors` |
| Dense complex Hermitian eigensystem | n/a | n/a | yes | yes | `KrylovDenseSubspaceComplexTypedTest.SolvesDenseComplexHermitianEigenvectors` |
| Dense complex divide-and-conquer Hermitian eigensystem | n/a | n/a | yes | yes | `KrylovDenseSubspaceComplexTypedTest.SolvesDenseComplexHermitianDivideAndConquerEigenvectors` |
| Dense complex selected Hermitian eigensystem | n/a | n/a | yes | yes | `KrylovDenseSubspaceComplexTypedTest.SolvesSelectedDenseComplexHermitianEigenvectors` |
| Dense generalized complex Hermitian eigensystem | n/a | n/a | yes | yes | `KrylovDenseSubspaceComplexTypedTest.SolvesDenseGeneralizedComplexHermitianEigenvectors` |
| Dense generalized complex divide-and-conquer Hermitian eigensystem | n/a | n/a | yes | yes | `KrylovDenseSubspaceComplexTypedTest.SolvesDenseGeneralizedComplexHermitianDivideAndConquerEigenvectors` |
| Dense generalized complex selected Hermitian eigensystem | n/a | n/a | yes | yes | `KrylovDenseSubspaceComplexTypedTest.SolvesSelectedDenseGeneralizedComplexHermitianEigenvectors` |
| Dense symmetric tridiagonal eigensystem | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.SolvesSymmetricTridiagonal*` |
| Dense real nonsymmetric eigensystem | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.SolvesRealNonsymmetric*` |
| Dense real nonsymmetric expert eigensystem | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.SolvesRealNonsymmetricExpertEigenvectorsWithDiagnostics` |
| Dense real nonsymmetric balancing and right-vector backtransform | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.BalancesRealNonsymmetricMatrixAndBacktransformsRightVectors` |
| Dense generalized real nonsymmetric eigensystem | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.SolvesRealGeneralizedNonsymmetricDiagonalEigenvectors` |
| Dense generalized real nonsymmetric expert eigensystem | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.SolvesRealGeneralizedNonsymmetricExpertEigenvectorsWithDiagnostics` |
| Dense generalized real nonsymmetric balancing and right-vector backtransform | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.BalancesRealGeneralizedNonsymmetricPencilAndBacktransformsRightVectors` |
| Dense generalized real Hessenberg reduction | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.ComputesRealGeneralizedHessenbergReduction` |
| Dense generalized real Hessenberg Schur factorization | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.ComputesRealGeneralizedHessenbergSchurDecomposition` |
| Dense generalized real Schur factorization | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.ComputesRealGeneralizedSchurDecomposition` |
| Dense generalized real Schur reordering | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.ReordersRealGeneralizedSchurBlocks` |
| Dense generalized real Schur selected subspace condition estimates | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.SelectsRealGeneralizedSchurSubspaceWithConditionEstimates` |
| Dense generalized real Schur right eigenvectors | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.ComputesRealGeneralizedSchurRightEigenvectors` |
| Dense generalized real Schur eigenpair condition estimates | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.EstimatesRealGeneralizedSchurEigenpairConditioning` |
| Dense real Schur and reordering | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.ComputesRealSchur*`, `ReordersRealSchur*` |
| Dense real Hessenberg Schur factorization | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.ComputesRealHessenbergSchurDecomposition` |
| Dense real Schur selected subspace condition estimates | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.SelectsRealSchurSubspaceWithConditionEstimates` |
| Dense real Schur right eigenvectors | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.ComputesRealSchurRightEigenvectors` |
| Dense real Schur eigenpair condition estimates | yes | yes | n/a | n/a | `KrylovDenseSubspaceTypedTest.EstimatesRealSchurEigenpairConditioning` |
| Dense complex nonsymmetric eigensystem | n/a | n/a | yes | yes | `KrylovDenseSubspaceComplexTypedTest.SolvesComplexNonsymmetricEigenvalues` |
| Dense complex Schur and reordering | n/a | n/a | yes | yes | `KrylovDenseSubspaceComplexTypedTest.ComputesComplexSchur*`, `ReordersComplexSchur*` |
| Symmetric/Hermitian Lanczos, full projection | yes | yes | yes | yes | `KrylovSymmetricLanczosRealTypedTest.SolvesDiagonalLargestAlgebraicProblem`, `KrylovHermitianLanczosComplexTypedTest.SolvesImaginaryOffDiagonalHermitianProblem` |
| Symmetric/Hermitian Lanczos, restarted regular mode | yes | yes | yes | yes | `KrylovSymmetricLanczosRealTypedTest.RestartedSolveConvergesOnDiagonalLargestAlgebraicProblem`, `KrylovHermitianLanczosComplexTypedTest.RestartedSolveConvergesOnPhaseTwistedLaplacian` |
| Symmetric/Hermitian Lanczos, transformed shift-invert mapping | yes | yes | yes | yes | `KrylovSymmetricLanczosRealTypedTest.ShiftInvertSmallestAlgebraicSelectorActsInTransformedSpace`, `KrylovHermitianLanczosComplexTypedTest.ShiftInvertSelectorMapsComplexHermitianVectorPath` |
| Symmetric/Hermitian generalized Lanczos with `B` metric | yes | yes | yes | yes | `KrylovSymmetricLanczosRealTypedTest.GeneralizedRegularModeUsesBMetric`, `KrylovHermitianLanczosComplexTypedTest.GeneralizedRegularModeUsesComplexBMetricPath` |
| Real nonsymmetric Arnoldi, full projection | yes | yes | n/a | n/a | `KrylovNonsymmetricArnoldiRealTypedTest.SolvesNonrestartedRealNonsymmetricDiagonalProblem` |
| Real nonsymmetric Arnoldi, real Schur restart | yes | yes | n/a | n/a | `KrylovNonsymmetricArnoldiRealTypedTest.RestartedSolveConvergesOnRealNonsymmetricDiagonalProblem` |
| Complex nonsymmetric Arnoldi, full projection | n/a | n/a | yes | yes | `KrylovNonsymmetricArnoldiComplexTypedTest.SolvesNonrestartedComplexNonsymmetricDiagonalProblem` |
| Complex nonsymmetric Arnoldi, complex Schur restart | n/a | n/a | yes | yes | `KrylovNonsymmetricArnoldiComplexTypedTest.RestartedSolveConvergesOnComplexNonsymmetricDiagonalProblem`, `RestartedSolveConvergesOnTransferLikeClusteredSpectrum` |
| Hermitian Krylov exponential action, fixed subspace | yes | yes | yes | yes | `KrylovExponentialRealTypedTest.HermitianFullSubspaceMatchesDiagonalExponential`, `KrylovExponentialComplexTypedTest.HermitianZeroVectorReturnsZeroWithoutMatvec`, `KrylovExponentialComplexTypedTest.HermitianActionAcceptsComplexTime` |
| Nonsymmetric Krylov exponential action, fixed subspace | yes | yes | yes | yes | `KrylovExponentialRealTypedTest.NonsymmetricFullSubspaceMatchesJordanExponential`, `KrylovExponentialComplexTypedTest.NonsymmetricFullSubspaceMatchesComplexDiagonalExponential`, `KrylovExponentialComplexTypedTest.NonsymmetricActionAcceptsComplexTime` |
| Taylor exponential action, scaled Taylor validation path | yes | yes | yes | yes | `KrylovExponentialRealTypedTest.TaylorActionMatchesDiagonalExponential`, `KrylovExponentialRealTypedTest.TaylorActionMatchesJordanExponential`, `KrylovExponentialComplexTypedTest.TaylorActionAcceptsComplexTime`, `KrylovExponentialComplexTypedTest.TaylorZeroVectorReturnsZeroWithoutMatvec`, `KrylovExponentialTaylorAction.ThrowsWhenTaylorDegreeCannotMeetTolerance` |
| Lanczos-vs-Taylor exponential probe example | yes | yes | yes | yes | `KrylovExponentialProbeExample.Float`, `KrylovExponentialProbeExample.Double` |
| Lanczos exponential orthogonality-floor example | yes | yes | n/a | n/a | `KrylovExponentialOrthogonalityExample.Float` |
| TDVP Matrix Market exponential probe example | n/a | n/a | yes | yes | `KrylovExponentialMatrixMarketProbeExample.TdvpFloat`, `KrylovExponentialMatrixMarketProbeExample.TdvpDouble` |

`DenseHostVectorOps` also has compile-time `KrylovMatrixFreeOperator` assertions
for `s`, `d`, `c`, and `z`.

## Optional MPLAPACK Binary128 Coverage

This table records the `UNI20_ENABLE_MPLAPACK=ON` probes. The entries are not
part of the ordinary default test matrix. They check that the scalar-generic
paths genuinely operate at the configured binary128 precision and do not
silently collapse to double precision.

The native solver entry points are intended to mirror the ordinary precision
coverage where the scalar category makes sense: real solver paths gain `f128`,
complex solver paths gain `cf128`, and Hermitian paths gain both. The broad
dense projected helper inventory remains quarantined in
`dense_subspace_unused.hpp`; it is not part of the maintained binary128 test
matrix. See [Dense BLAS/LAPACK Wrapper Coverage](../linalg/dense_blas_lapack_coverage.md)
for that provider/helper inventory.

| algorithm or surface | `f128` | `cf128` | tests |
| --- | --- | --- | --- |
| Scalar aliases, numeric limits, scalar concepts, and scalar I/O | yes | yes | `MplapackBinary128Test.Uni20NumericLimitsSeesBackendScalar`, `Uni20ScalarConceptsSeeBackendScalar`, `Uni20ScalarIo*` |
| MPBLAS wrapper surface | yes | yes | `MplapackBinary128Test.LinksMpblasTransitively`, `Uni20BlasWrappersPreserveBinary128OnlyIncrements` |
| Dense one-norm and linear solve through CPU helpers | yes | n/a | `MplapackBinary128CpuOpsTest.MatrixOneNormPreservesBinary128Precision`, `SolveAcceptsPivotsBelowDoubleMinimum` |
| Tensor reductions through the CPU reference backend | yes | no | `MplapackBinary128CpuOpsTest.TensorReductionsPreserveBinary128Values` |
| Dense matrix exponential prescaling | yes | no | `MplapackBinary128CpuOpsTest.MatrixExponentialPrescalesWithinBinary128` |
| Exact Tensor SVD | yes | yes | `MplapackBinary128CpuOpsTest.ExactSvdPreservesRealAndComplexBinary128Values`, `AsyncSvdTest.SupportsConfiguredFloat128Backend` |
| Broad dense projected real helper inventory | not active | n/a | Quarantined source inventory; no maintained binary128 target. |
| Dense projected complex eigensystem and Schur helper inventory | n/a | not active | Quarantined source inventory; no maintained binary128 target. |
| Symmetric tridiagonal projected eigensystem | yes | n/a | `MplapackBinary128KrylovSolversTest.TridiagonalProjectionResolvesGapBelowDoublePrecision` |
| Symmetric/Hermitian Lanczos, full projection | yes | no | `MplapackBinary128KrylovSolversTest.SymmetricLanczosResolvesDiagonalGapBelowDoublePrecision` |
| Real nonsymmetric projected Schur kernels | yes | n/a | `MplapackBinary128KrylovSolversTest.RealSchurAndReorderUseBinary128ProjectedLAPACK`, `RealHessenbergSchurUsesBinary128ProjectedLAPACK` |
| Real nonsymmetric Arnoldi, full projection | yes | n/a | `MplapackBinary128KrylovSolversTest.RealArnoldiResolvesTriangularGapBelowDoublePrecision` |
| Complex nonsymmetric projected Schur kernels | n/a | yes | `MplapackBinary128KrylovSolversTest.ComplexSchurAndReorderUseBinary128ProjectedLAPACK` |
| Complex nonsymmetric Arnoldi, full projection | n/a | yes | `MplapackBinary128KrylovSolversTest.ComplexArnoldiResolvesComplexGapBelowDoublePrecision` |
| Hermitian Krylov exponential action, fixed subspace | yes | no | `MplapackBinary128KrylovSolversTest.HermitianExponentialActionPreservesBinary128OnlyIncrement` |

`no` in this optional table means no dedicated binary128 probe exists yet. It
does not necessarily mean the templated implementation cannot instantiate once
the corresponding dense projected kernel and matrix-free operation are covered.

## External ARPACK Oracle Coverage

During development, the native Krylov paths were compared against vendored
ARPACK-NG adapters for the upstream double-precision routines:

| ARPACK path | covered precision |
| --- | --- |
| symmetric `dsaupd`/`dseupd` | `d` |
| real nonsymmetric `dnaupd`/`dneupd` | `d` |
| complex nonsymmetric `znaupd`/`zneupd` | `z` |

Those oracle adapters are not part of the core Uni20 API or build. Future
ARPACK comparisons should live in the separate validation and benchmarking
repository.

## Known Gaps

Precision-specific hardening remains future work. The validation matrix proves
that the solver paths instantiate and solve simple representative problems for
their supported scalar tags; stress thresholds, breakdown floors, and Matrix
Market reliability tests still need type-specific tuning.

The Krylov exponential action tests currently validate exact full-subspace
projection cases, small analytic Taylor-action cases, diagonal
Lanczos-vs-Taylor-vs-exact probe examples, and the TDVP harmonic-oscillator
Matrix Market stress fixture. The probe examples now report several published
exponential-action error indicators; see
[Krylov Exponential Estimators](exponential_estimators.md). Adaptive
Lanczos time stepping, restart/error control, Al-Mohy/Higham-style norm
estimation for the Taylor reference path, and broader matrix-free exponential
stress tests remain future work.
