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
| `defect/||v||` | `h_{m+1,m} abs(e_m^* exp(t H_m) e_1)` | Final-time defect norm per input norm. The native exponential result currently reports this quantity multiplied by `||v||` as `residual_estimate`. See the residual/defect framing in `[BotchevGrimmHochbruck2013]` and the `xi_2`-style discussion in `[JiaLv2015]`. |
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
| full reorthogonalized Lanczos | Mirrors the production exponential path closely enough to inspect estimator behavior using projected data that the public result does not expose. |
| legacy three-term Lanczos | Reproduces the simpler recurrence style used by older Lanczos exponential implementations, useful for detecting precision floors and loss-of-orthogonality sensitivity. |

The example is diagnostic code, not the adaptive stepping implementation. The
next adaptive exponential action should use these probes to choose conservative
stopping rules, but it should not automatically fall back to Taylor or hide
under-convergence.

## Future Estimators

Useful but not yet measured in the probe:

| estimator | reason deferred |
| --- | --- |
| Direct defect-integral quadrature `h int_0^t abs(e_m^* exp(s H_m)e_1) ds` | Requires many projected exponentials or a dedicated quadrature path. Good for validation, not cheap enough for the first probe. |
| Improved Hermite quadrature from the defect derivative | Requires extra projected derivative terms and, in matrix-free form, may need one additional matvec. |
| Effective-order quadrature | Useful for adaptive step-size control, but needs a robust definition of effective order near roundoff. |
| Non-Hermitian defect/residual bounds | Needed for Arnoldi exponential actions; `[WangYe2017]` is the current starting point. The present probe focuses on Hermitian/Lanczos behavior. |
