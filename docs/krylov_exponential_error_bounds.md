# Krylov Exponential Error Bounds

This note records a practical lesson from testing Krylov time-evolution kernels. It is
intended as design guidance for the uni20 Krylov exponential implementation, not as a
complete review of exponential integrator error theory.

## Setting

For a Hermitian matrix-free operator `H`, the Lanczos exponential approximates

```text
exp(tau H) v ~= ||v|| V_m exp(tau T_m) e_1,
```

where `V_m` is the Lanczos basis and `T_m` is the projected tridiagonal matrix. In exact
arithmetic the Krylov space has dimension at most `n`, and reaching that dimension is
full convergence for a finite vector space. In floating point arithmetic, the useful
iteration count is usually smaller unless the basis is explicitly reorthogonalized.

## Projected Coefficient Is Not Enough

A common cheap convergence heuristic is to inspect the final Krylov coefficient in

```text
exp(tau T_m) e_1.
```

This is useful diagnostic information, but by itself it is not a reliable absolute error
estimate. It omits the next residual coupling and can be badly scaled. In particular,
for small `|tau|` the last projected coefficient can be misleading unless the product of
Lanczos off-diagonal coefficients is included.

For the Hermitian/Lanczos case, a better cheap estimate follows the structure of the
`[JaweckiAuzingerKoch2020]` bound:

```text
err_m <= ||v|| beta_{m+1,m} gamma_m |tau|^m / m!,
gamma_m = beta_{2,1} beta_{3,2} ... beta_{m,m-1}.
```

This is the natural quantity to use as the primary stopping estimate for a small dense
projected exponential. Compute it in log-space to avoid overflow/underflow:

```text
log(err_m) =
    log(||v||)
  + log(beta_{m+1,m})
  + sum_j log(beta_{j+1,j})
  + m log(|tau|)
  - log(m!).
```

The estimate should be interpreted as a practical bound/indicator for the projected
Krylov approximation, not as a substitute for monitoring basis orthogonality.
When diagnostics are enabled, the Hermitian exponential path records the final
Gram defect `||V_m^* V_m - I||` through maximum diagonal, maximum off-diagonal,
and Frobenius summaries, plus the largest residual reorthogonalization
correction ratio. A small final Gram defect means the stored basis was cleaned
successfully; a large correction ratio means the raw Lanczos three-term
recurrence was already losing orthogonality before cleanup.

## Tolerance Floor

Requesting a tolerance far below the attainable Krylov basis accuracy is actively
harmful. It forces the recurrence to keep adding vectors after the projected exponential
has already saturated, and the later vectors may mostly measure loss of orthogonality
rather than useful approximation error.

For double precision, `sqrt(eps) ~= 1.5e-8` is a natural default floor for a non- or
lightly-reorthogonalized Krylov exponential. A user-facing tolerance such as `1e-8` is
therefore reasonable. Asking for `1e-12` or `1e-14` should require a stronger algorithmic
contract, such as full or selective reorthogonalization and a residual/error estimate
designed for that level of accuracy.

For single precision, the corresponding floor is around `100 * eps(float) ~= 1e-5`.
That is also roughly the scale at which small projected residuals and numerical
breakdown become difficult to distinguish without additional orthogonality control.

## Breakdown Detection

Breakdown should be tested with a numerical threshold, not exact `beta == 0`. A useful
local threshold should scale with the recurrence quantities being subtracted:

```text
w = H q_j - alpha_j q_j - beta_j q_{j-1}.
```

Including `|alpha_j|`, `|beta_j|`, and `|beta_{j-1}|` in the scale is defensible because
large identity components in `H` can produce cancellation in `H q_j - alpha_j q_j`. A
matrix-free Krylov routine cannot cheaply detect and remove a large identity component;
callers that know `H = c I + A` should factor out `exp(tau c)` themselves.

## Implementation Policy

- Normalize the input vector internally, but scale the final error estimate by `||v||`.
- Track the product of accepted Lanczos off-diagonal coefficients.
- Compute the error estimate in log-space.
- Stop on numerical breakdown before attempting to normalize the next vector.
- Cap the Krylov dimension at the actual vector-space dimension when that dimension is
  known.
- Do not use a default maximum iteration count that is effectively infinite. If a solver
  has not converged in a modest number of Krylov steps, the caller probably needs a
  restarted method, a better shifted/scaled operator, or tighter orthogonality control.
- Treat `sqrt(eps)` as the default double-precision accuracy scale unless the algorithm
  explicitly maintains a basis accurate enough to justify a smaller tolerance.

## References

- `[HochbruckLubich1997]` M. Hochbruck and C. Lubich, "On Krylov subspace
  approximations to the matrix exponential operator", SIAM J. Numer. Anal.
  34(5):1911-1925, 1997. DOI: <https://doi.org/10.1137/S0036142995280572>.
- `[JaweckiAuzingerKoch2020]` T. Jawecki, W. Auzinger, and O. Koch,
  "Computable upper error bounds for Krylov approximations to matrix
  exponentials and associated phi-functions", BIT Numer. Math. 60:157-197,
  2020. DOI: <https://doi.org/10.1007/s10543-019-00782-z>. arXiv:
  <https://arxiv.org/abs/1809.03369>.
