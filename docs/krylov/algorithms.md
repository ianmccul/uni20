# Krylov Algorithms

This document is the implementation inventory for Uni20's native Krylov
solvers. Keep it synchronized with solver entry points, supported scalar types,
public parameters, default policies, and internal convergence or restart tuning.

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

With `UNI20_ENABLE_MPLAPACK=ON`, Uni20 enables optional experimental binary128
probes for maintained matrix-free eigensolvers and exponential actions.
MPLAPACK is an external package dependency; Uni20 does not download or build
it. Ordinary typed tests focus on the stable `s`, `d`, `c`, and `z` paths,
while maintained `MplapackBinary128*` targets cover selected binary128 stress
cases. See [Krylov Precision Validation](precision_validation.md) for the
test-level `f128` and `cf128` validation matrix.

Dense provider and quarantined helper coverage is tracked separately in
[Dense BLAS/LAPACK Wrapper Coverage](../linalg/dense_blas_lapack_coverage.md).

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
Active projected eigensystem and Schur entry points use matrix-level linalg
operations backed by `LapackBackend`; projected exponentials use the same
dispatch layer with the current CPU-reference matrix-exponential kernel.
Symmetric/Hermitian eigensolvers need real LAPACK coverage for the underlying
real precision, complex nonsymmetric eigensolvers need dense complex LAPACK
coverage, and exponential actions need dense coverage for the projected scalar.
The exact active and experimental dense wrapper inventory is recorded in
[Dense BLAS/LAPACK Wrapper Coverage](../linalg/dense_blas_lapack_coverage.md).

## Algorithm Inventory

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
| `hermitian_krylov_exponential_action` | yes | yes | yes | yes | Lanczos approximation `||v|| V_m exp(t T_m) e_1`, with optional adaptive relative tolerance for unitary/nonexpansive actions. |
| `nonsymmetric_krylov_exponential_action` | yes | yes | yes | yes | Fixed-subspace Arnoldi approximation `||v|| V_m exp(t H_m) e_1`. |
| `taylor_exponential_action` | yes | yes | yes | yes | Validation-oriented scaled Taylor action using a caller-supplied operator norm bound. |

The projected exponential invokes `matrix_exponential_op` through the linalg
dispatcher. Its current selected implementation is the CPU dense
scaling-and-squaring Padé kernel. It accepts real multipliers and complex
multipliers; a real projected matrix with a complex multiplier promotes to a
complex projected exponential. The Taylor action is an independent
reference/emergency path, not an automatic fallback from Lanczos or Arnoldi. It
uses scaled Taylor series steps with a geometric tail estimate and an explicit
caller-supplied bound for `||A||`. Taylor scaling-step selection and
tail-error amplification are computed in the solver real scalar type, so
extended-precision real types are not silently rounded through `long double`.
`examples/krylov/krylov_exponential_probe_example.cpp` compares Lanczos, Taylor,
and exact diagonal exponential actions on deliberately awkward spectra. It
reports both an example-local full-reorthogonalized Lanczos recurrence,
mirroring the native Hermitian path while exposing projected data, and an
example-local legacy three-term recurrence so precision-floor and
loss-of-orthogonality effects can be inspected side by side. The probe
distinguishes the raw last projected exponential coefficient, matching the
historical Cytnx `abs(B_mat(i,0))` stopping indicator, from the endpoint defect
and direct defect-integral estimates used by the native Hermitian result. It
also reports Hermite-quadrature, Saad/Jia-Lv `phi_1`, and
Hochbruck-Lubich/Jawecki-style leading-bound estimates; see
[Krylov Exponential Estimators](exponential_estimators.md). Its
rebound diagnostics are intended to expose cases where asking for an
unrealistically small tolerance keeps increasing the Krylov dimension after the
true action error has already reached the scalar precision floor. The probe
also reports the final basis off-diagonal Gram defect and the largest
reorthogonalization correction ratio, which help distinguish estimator underflow
from loss of useful Lanczos orthogonality. Full Al-Mohy/Higham-style norm
estimation, block right-hand sides, adaptive Lanczos time stepping, restart
control, and broader benchmark examples are still future work.

`examples/krylov/krylov_exponential_orthogonality_example.cpp` is a smaller teaching
example focused on this failure mode. Its default float run requests
`tol=1e-8`, below the `sqrt(eps(float))` warning scale that motivated the old
TDVP/Cytnx policy for non-reorthogonalized or lightly reorthogonalized Lanczos.
It shows a case where the residual estimate falls below the requested tolerance
while the exact diagonal reference error has already saturated. The same table
contrasts the full-reorthogonalized Hermitian path with a legacy three-term
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
convergence diagnostic and benchmark aid; it is not an automatic fallback
policy.

Real vector spaces accept real time/coefficient values. Complex vector spaces
also accept complex time values, so unitary actions such as `exp(-i t H) v` can
use the Hermitian/Lanczos path without hiding the phase in the matrix-free
operator.

### Optional MPLAPACK Binary128 Algorithm Inventory

These entry points are available only when `UNI20_ENABLE_MPLAPACK=ON` and the
configured MPLAPACK package provides a real binary128 type. The table records
intended algorithm-level scalar coverage; dedicated stress-test coverage is
tracked separately in [Krylov Precision Validation](precision_validation.md).

| entry point | `f128` | `cf128` | notes |
| --- | --- | --- | --- |
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

## Relationship To ARPACK

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
| `krylov_dimension` | `0` | Hermitian and nonsymmetric exponential actions | Fixed projection dimension, or adaptive maximum when `relative_tolerance > 0`; zero selects the default policy. |
| `minimum_krylov_dimension` | `0` | Hermitian adaptive exponential action | Minimum projection dimension before relative-tolerance acceptance; zero selects `1`. |
| `relative_tolerance` | `0` | Hermitian adaptive exponential action | Relative error target for unitary/nonexpansive Hermitian actions; zero keeps fixed-dimension mode. |
| `estimate_safety_factor` | `1` | Hermitian adaptive exponential action | Multiplier applied to the direct defect-integral estimate before comparing with `relative_tolerance * ||v||`. |
| `breakdown_tolerance` | `0` | Hermitian and nonsymmetric exponential actions | Internal invariant-subspace threshold; zero selects `10 * epsilon`. |
| `assume_nonexpansive` | `false` | Hermitian adaptive exponential action | Allows adaptive acceptance when the caller has verified nonexpansiveness but the time coefficient is not automatically recognized as unitary. |
| `throw_on_nonconvergence` | `true` | Hermitian adaptive exponential action | Throw if the adaptive cap is reached before satisfying `relative_tolerance`. |
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
| Fixed-dimension exponential action | `min(problem_dimension, 30)` | n/a |
| Adaptive Hermitian exponential action | `min(problem_dimension, 64)` maximum cap; `1` minimum | n/a |
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
| Exponential action breakdown threshold | `10 * epsilon` | exponential actions | Treats a tiny expansion residual as invariant-subspace breakdown. |
| Hermitian exponential adaptive acceptance | direct defect integral `<= relative_tolerance * ||v|| / estimate_safety_factor` | unitary/nonexpansive Hermitian exponential action | Stops at the first projected dimension satisfying the Jawecki-Auzinger-Koch defect-integral bound. |
| Hermitian defect-integral quadrature | 1024-panel Simpson rule over the projected scalar defect | unitary/nonexpansive Hermitian exponential action | Deterministic projected-space integral; cheap relative to matrix-free tensor-network matvecs. |
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
matvec count, initial norm, final Krylov residual norm, a top-level
`error_estimate`, endpoint defect estimate, Hermitian defect-integral estimate,
target error, convergence state, happy-breakdown state, and optional summary
diagnostics. For Hermitian actions, `error_estimate` is the direct defect
integral. For nonsymmetric Arnoldi actions, it is currently the endpoint defect
estimate. A future Arnoldi defect-integral rule must include a nonexpansive or
semigroup-growth bound; the projected scalar defect integral alone is only a
diagnostic for general non-Hermitian operators. Hermitian exponential diagnostics
additionally include final-basis Gram defects and reorthogonalization correction
metrics when diagnostics are enabled.

Taylor exponential action results report the action vector, scaling step count,
maximum Taylor degree used, matvec count, estimated tail error, convergence
state, and optional diagnostics containing the supplied operator-norm bound and
final step tail estimate.

## Known Gaps

- Real nonsymmetric complex-pair output through real Schur two-planes is not a
  finished user-facing result path.
- Krylov exponential actions do not yet implement adaptive time stepping,
  restart control, nonsymmetric Arnoldi semigroup/error bounds,
  Al-Mohy/Higham-style norm estimation, or broad benchmark dashboards. The
  Hermitian path does support adaptive relative tolerance for
  unitary/nonexpansive single-step actions.
- Type-specific stress hardening for `s`, `d`, `c`, and `z` remains ongoing.
