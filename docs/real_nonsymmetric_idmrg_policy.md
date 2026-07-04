# Real Nonsymmetric Arnoldi Policy for iDMRG

This note records the current design thinking for real nonsymmetric Arnoldi
solves in iDMRG transfer-matrix workflows. It is intentionally exploratory.

## Context

For real tensor-network problems, the transfer matrix is often real but
non-Hermitian. At convergence it may be close to normal if the fixed point does
not break time-reversal or reflection symmetry. During early iDMRG iterations,
however, the transfer operator can be substantially non-normal, and in some
physical cases the dominant fixed-point structure may genuinely require complex
arithmetic.

The Krylov layer should therefore not silently assume that a real nonsymmetric
operator has real wanted eigenvectors. It should make the real-versus-complex
decision visible to the caller.

## Policy Meaning

`RealNonsymmetricPolicy` is intended to answer one question:

> When a real nonsymmetric Arnoldi solve finds that a wanted Ritz value is not
> numerically real, what should the real-arithmetic solver do?

### RequireRealEigenpairs

Use this when the caller expects a real fixed point.

If a wanted Ritz value is classified as complex, or possibly ambiguous, the
solver should stop and report the condition rather than silently continue. This
is the conservative default for real iDMRG calculations where the physical
fixed point is expected to remain real.

### PromoteToComplexSuggested

Use this when the caller is willing to restart the larger calculation in complex
arithmetic, but the Krylov solver itself should not perform that promotion
implicitly.

For iDMRG this distinction matters because promotion affects tensors,
environments, caches, and possibly symmetry handling. It should be an outer
algorithm decision, not a hidden Krylov-side conversion.

### AllowRealSchurPairs

Use this when the caller can consume a real two-dimensional invariant subspace
for a complex conjugate pair.

For a real operator, complex eigenvectors `u +/- i v` correspond to a real
invariant plane spanned by `u` and `v`. This is mathematically valid and may be
useful for diagnostics or specialized algorithms. It is not automatically a
drop-in replacement for an iDMRG fixed-point vector unless the outer algorithm
knows how to use that plane.

This policy is currently a placeholder until real Schur two-plane output is
implemented.

## Status Semantics

The status set distinguishes the policy-specific outcomes for complex wanted
Ritz values:

```cpp
enum class NonsymmetricStatus {
  Converged,
  NotConverged,
  ComplexPairEncountered,
  ComplexPromotionRecommended,
  RealSchurPairRequired,
  AmbiguousReality,
  Breakdown,
  InvalidInput
};
```

Then the policy behavior would be:

| Policy | Complex wanted Ritz value |
| --- | --- |
| `RequireRealEigenpairs` | `ComplexPairEncountered` |
| `PromoteToComplexSuggested` | `ComplexPromotionRecommended` |
| `AllowRealSchurPairs` | return real Schur/two-plane output, or `RealSchurPairRequired` until implemented |

Ambiguous Ritz values should likely remain a separate `AmbiguousReality` status,
because they indicate that the imaginary part is near the classification
threshold and the caller may want to tighten tolerances, continue iterating, or
inspect diagnostics before making a physical symmetry-breaking decision.

## Non-Normality Is Separate

Complex wanted Ritz values are not the same thing as non-normality.

A real nonsymmetric transfer matrix can be highly non-normal while all wanted
eigenvalues are real. Conversely, a nearly normal real operator can have a
dominant complex conjugate pair. The policy above only handles the
real-versus-complex output decision. It does not fully diagnose non-normal
behavior.

Useful future diagnostics include:

- conditioning or angle diagnostics for selected Ritz vectors,
- residual history and stagnation information,
- detection of near-degenerate real Ritz values that may become a complex pair
  under perturbation,
- Schur residuals for nonnormal cases rather than relying only on individual
  eigenvector residuals.

For iDMRG, these diagnostics should help distinguish:

- harmless early-iteration non-normality,
- numerical ambiguity near a real fixed point,
- genuine complex fixed-point structure,
- cases where real arithmetic should stop and ask the outer algorithm to
  promote the calculation to complex arithmetic.

## Current Implementation State

The current Phase 4 real nonsymmetric path keeps the Arnoldi basis and
matrix-free operations real. It extracts complex Ritz values from the small
projected Hessenberg matrix, classifies wanted Ritz values as real, ambiguous,
or complex, reports the condition through `NonsymmetricStatus`, and records the
projected departure from normality
`||H^T H - H H^T||_F / max(1, ||H||_F^2)` in diagnostics.

The current implementation does not yet:

- return real Schur two-plane output,
- compute non-normality diagnostics for the original matrix-free operator
  beyond the projected Hessenberg diagnostic,
- restart the nonsymmetric Arnoldi factorization.

Those are future design and implementation steps.
