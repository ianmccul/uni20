# Krylov Exponential Error Estimators

This note records the exponential-action convergence diagnostics measured by
`examples/krylov/krylov_exponential_probe_example.cpp`. For a smaller toy illustration
of an orthogonality-driven tolerance floor, run
`examples/krylov/krylov_exponential_orthogonality_example.cpp`.
For a real Matrix Market stress case copied from the Cytnx TDVP/Lanczos
harmonic-oscillator investigation, run
`examples/krylov/krylov_exponential_matrix_market_probe_example.cpp`; its default
`complex<double>` configuration uses
`tests/krylov/matrix_market/tdvp_lanczos/` and shows the old raw projected-tail
indicator below `1e-8` while the Taylor-reference action error is still above
`1e-8`.

The notation is

```text
A V_m = V_m H_m + h_{m+1,m} v_{m+1} e_m^*
exp(t A) v ~= ||v|| V_m exp(t H_m) e_1
```

For Hermitian Lanczos, `H_m` is tridiagonal. For the diagnostic diagonal
examples, the probe can compute exact errors, so the estimates are reported
against the true action error.

## Measured Quantities

| probe column | formula | purpose |
| --- | --- | --- |
| `tail coeff` | `abs(e_m^* exp(t H_m) e_1)` | Raw projected-exponential last coefficient. This mirrors the historical Cytnx-style `abs(B_mat(i,0))` stopping indicator. |
| `defect/||v||` | `h_{m+1,m} abs(e_m^* exp(t H_m) e_1)` | Final-time defect norm per input norm. The native exponential result reports this quantity multiplied by `||v||` as `endpoint_defect_estimate`; for Hermitian actions, `error_estimate` is instead the direct defect integral. See the residual/defect framing in `[BotchevGrimmHochbruck2013]` and the `xi_2`-style discussion in `[JiaLv2015]`. |
| `Hquad/||v||` | `h_{m+1,m} |t| abs(e_m^* exp(t H_m) e_1) / m` | Hermite-quadrature defect-integral estimate from the `[HochbruckLubich1997]` / `[JaweckiAuzingerKoch2020]` line of analysis. |
| `Saad phi1/||v||` | `h_{m+1,m} |t| abs(e_m^* phi_1(t H_m) e_1)` | First-term error expansion estimate from `[Saad1992]`, also used as an a posteriori estimate in `[JiaLv2015]`. Here `phi_1(z) = (exp(z)-1)/z`; the probe computes it by a small augmented matrix exponential. |
| `HL bound/||v||` | `h_{m+1,m} gamma_m |t|^m / m!` | Nonexpansive-case leading upper bound/asymptotic term from `[HochbruckLubich1997]`, with computable forms in `[JaweckiAuzingerKoch2020]`; `gamma_m = prod_{j=1}^{m-1} h_{j+1,j}`. The probe evaluates this in log space to avoid artificial overflow/underflow in `float`. |
| `orth offdiag` | `max_{i != j} abs(q_i^* q_j)` | Final-basis orthogonality defect. This is not an exponential error estimate; it indicates when the Krylov basis itself has reached a precision floor. |
| `reorth ratio` | `max abs(q_i^* r) / ||r||` during residual reorthogonalization | Measures how much the raw Lanczos residual had to be corrected. A large value means the three-term recurrence is no longer producing a nearly orthogonal residual before explicit reorthogonalization. |
| `passes` | maximum number of reorthogonalization passes used for any residual in that run | Distinguishes ordinary one-pass cleanup from the second-pass Daniel-Gragg-Kaufman-Stewart style refinement used by the implementation. |

The `Hquad`, `Saad phi1`, and `HL bound` estimates are most meaningful once the
Krylov approximation has entered its convergence regime. They can all become
misleading after the true error has reached the scalar precision floor; the
probe intentionally samples beyond that point to expose underflow and
over-solving behavior. The orthogonality columns help identify whether that
floor is dominated by loss of basis orthogonality rather than by the projected
exponential estimator itself. Simple diagonal examples are intentionally limited:
they are good for showing false stopping signals and basis loss, but real TDVP
operators are much more likely to exhibit severe post-threshold residual rebound.

## Defect-Integral Stopping Rule

The Hermitian exponential action reports both the final-time defect norm,

```text
h_{m+1,m} |delta_m(t)| ||v||,
delta_m(s) = e_m^* exp(sigma s H_m) e_1.
```

as `endpoint_defect_estimate`, and the direct defect integral as
`defect_integral_estimate`. The top-level Hermitian `error_estimate` uses the
direct integral, since endpoint sampling is often far too conservative and the
raw tail coefficient is not reliable.

For a nonexpansive exponential, JAK starts from

```text
L_m(t)v = int_0^t exp(sigma (t-s) A) D_m(s)v ds,
D_m(s)v = sigma h_{m+1,m} delta_m(s) v_{m+1}.
```

This gives the computable bound

```text
||L_m(t)v|| <= h_{m+1,m} int_0^t |delta_m(s)| ds.
```

The direct integral is therefore the natural quantity to target for adaptive
stopping. The cheap HL/JAK bound

```text
h_{m+1,m} gamma_m |t|^m / m!
```

is rigorous and asymptotically correct under the nonexpansive assumptions, but
it can be far too conservative in hot TDVP loops.

For the TDVP harmonic-oscillator fixture with `tau = -0.1968473663975394 i` and `||v|| = 1`, direct quadrature of the defect integral is much tighter:

| m | true error | final defect | direct defect integral | HL bound |
|---:|---:|---:|---:|---:|
| 33 | `1.398e-8` | `1.136e-6` | `3.225e-8` | `2.364e2` |
| 34 | `9.686e-9` | `7.735e-7` | `2.289e-8` | `1.753e2` |
| 40 | `3.254e-9` | `1.726e-7` | `8.369e-9` | `1.828e1` |
| 64 | `4.554e-12` | `1.105e-9` | `5.819e-12` | `8.160e-7` |

For a `1e-8` target, the direct integral accepts `m=40` on this fixture. The
endpoint-defect estimator accepts much later, around `m=64`, and the HL bound
later still, around `m=69`.

The native Hermitian action implements this direct-integral target for the
nonexpansive path:

1. Diagonalize the projected Hermitian matrix `H_m = Q Lambda Q^*`.
2. Evaluate `delta_m(s) = e_m^* Q exp(sigma s Lambda) Q^* e_1`.
3. Integrate `abs(delta_m(s))` on `[0, |t|]` with a deterministic 1024-panel
   Simpson rule.
4. Accept the first projected dimension whose direct integral is below
   `relative_tolerance * ||v||`, after applying the configured safety factor.

Pure-imaginary Hermitian time is recognized automatically as the unitary
`exp(-i t H)` case. Other Hermitian nonexpansive actions require the caller to
set `assume_nonexpansive`, since the Krylov layer cannot infer positivity or
contractivity from the matrix-free operator alone. The raw tail, endpoint
defect, and direct integral remain available as diagnostics; the probe also
continues to report Hermite/trapezoid-style, effective-order, and HL-family
quantities for comparison.

### Non-Hermitian Defect Integrals

The defect identity itself is not Hermitian-specific. For an Arnoldi projection,

```text
A V_m = V_m H_m + h_{m+1,m} v_{m+1} e_m^*,
```

the same residual/defect construction gives

```text
D_m(s)v = h_{m+1,m} e_m^* exp(s H_m) e_1 v_{m+1} ||v||.
```

The difference is the propagation factor in the error integral. The safe bound is

```text
||L_m(t)v|| <= h_{m+1,m} ||v||
  int_0^t ||exp((t-s) A)|| |e_m^* exp(s H_m) e_1| ds.
```

For Hermitian unitary evolution, `||exp((t-s) A)|| = 1`, so this reduces to the
projected scalar integral used above. For a general non-Hermitian or non-normal
operator, that simplification is not available: transient growth in
`||exp((t-s) A)||` can be large even when the spectrum looks stable. Therefore a
future Arnoldi adaptive rule can use the projected defect integral only as a
diagnostic unless the caller supplies, or the algorithm proves, a usable
nonexpansive or semigroup-growth bound. `[WangYe2017]` is the current reference
starting point for that non-Hermitian error-control work.

The trapezoid estimate,

```text
h_{m+1,m} |t| |delta_m(t)| / 2,
```

is a useful simple fallback. On the same fixture it would accept around `m=48`.
The Hermite estimate `h |t| |delta_m(t)| / m` is often too optimistic to use as
a sole stopping criterion, while the effective-order estimate can be very tight
but needs robust handling near roundoff and cancellation.

## References

The code comments use the same citation keys as this table.

| key | reference | relevant idea |
| --- | --- | --- |
| `[Saad1992]` | Y. Saad, "Analysis of some Krylov subspace approximations to the matrix exponential operator", SIAM J. Numer. Anal. 29(1):209-228, 1992. DOI: <https://doi.org/10.1137/0729014>. | Error expansion and the leading-term `phi_1` style estimate. |
| `[HochbruckLubich1997]` | M. Hochbruck and C. Lubich, "On Krylov subspace approximations to the matrix exponential operator", SIAM J. Numer. Anal. 34(5):1911-1925, 1997. DOI: <https://doi.org/10.1137/S0036142995280572>. | Asymptotic convergence analysis for Hermitian/skew-Hermitian exponential Krylov approximations. |
| `[BotchevGrimmHochbruck2013]` | M. Botchev, V. Grimm, and M. Hochbruck, "Residual, restarting and Richardson iteration for the matrix exponential", SIAM J. Sci. Comput. 35(3):A1376-A1397, 2013. DOI: <https://doi.org/10.1137/110820191>. arXiv: <https://arxiv.org/abs/1112.5670>. | Residual/defect viewpoint and restarted exponential Krylov context. |
| `[JiaLv2015]` | Z. Jia and H. Lv, "A posteriori error estimates of Krylov subspace approximations to matrix functions", Numer. Algorithms 69:1-28, 2015. DOI: <https://doi.org/10.1007/s11075-014-9885-x>. arXiv: <https://arxiv.org/abs/1307.7219>. | Justifies `xi_1` and `xi_2` a posteriori estimates, including the `phi_1` leading-term estimate. |
| `[JaweckiAuzingerKoch2020]` | T. Jawecki, W. Auzinger, and O. Koch, "Computable upper error bounds for Krylov approximations to matrix exponentials and associated phi-functions", BIT Numer. Math. 60:157-197, 2020. DOI: <https://doi.org/10.1007/s10543-019-00782-z>. arXiv: <https://arxiv.org/abs/1809.03369>. | Computable upper bound `h gamma_m t^m/m!`, Hermite quadrature estimate, effective-order quadrature, and step-size-control discussion. |
| `[WangYe2017]` | S. Wang and Q. Ye, "Error bounds for the Krylov subspace methods for computations of matrix exponentials", SIAM J. Matrix Anal. Appl. 38(1):155-187, 2017. arXiv: <https://arxiv.org/abs/1603.07358>. | Useful later for non-Hermitian Arnoldi exponential bounds; not yet measured by the current Hermitian probe. |

## Probe Scope

The probe reports two local recurrence variants:

| implementation | role |
| --- | --- |
| full reorthogonalized Lanczos | Mirrors the native Hermitian exponential path closely enough to inspect estimator behavior using projected data. |
| legacy three-term Lanczos | Reproduces the simpler recurrence style used by older Lanczos exponential implementations, useful for detecting precision floors and loss-of-orthogonality sensitivity. |

The example is diagnostic code, not a fallback policy. The native Hermitian
action now uses the direct defect integral for adaptive nonexpansive acceptance,
while the probe remains useful for comparing that choice against older tail,
endpoint-defect, Hermite/quadrature, and HL-style indicators.

## Future Estimators

Useful extensions not yet part of the native adaptive stopping rule:

| estimator | reason deferred |
| --- | --- |
| Trapezoid defect-integral estimate `h |t| abs(e_m^* exp(t H_m)e_1) / 2` | Cheap middle ground between the endpoint defect and direct quadrature. It is only justified under effective-order assumptions, but is less optimistic than the Hermite `1/m` factor. |
| Improved Hermite quadrature from the defect derivative | Requires extra projected derivative terms and, in matrix-free form, may need one additional matvec. |
| Effective-order quadrature | Useful for adaptive step-size control, but needs a robust definition of effective order near roundoff. |
| Non-Hermitian defect/residual bounds | Needed before Arnoldi exponential actions can use defect-integral stopping as a bound. The projected scalar integral is only diagnostic without a nonexpansive or semigroup-growth bound; `[WangYe2017]` is the current starting point. |
