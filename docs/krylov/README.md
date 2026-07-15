# Krylov Documentation

This directory contains the implemented Krylov solver contracts, numerical
validation evidence, and supporting estimator rationale.

## Canonical Guides

- [Krylov Algorithms](algorithms.md) defines supported solvers, parameters,
  scalar behavior, and convergence semantics.
- [Solver Defaults](solver_defaults.md) records public defaults separately from
  internal tuning.
- [Precision Validation](precision_validation.md) records the compiler,
  precision, and provider validation matrix.
- [Test Matrices](test_matrices.md) describes deterministic Matrix Market
  fixtures and probe usage.

## Numerical Rationale

- [Exponential Error Bounds](exponential_error_bounds.md)
- [Exponential Estimators](exponential_estimators.md)

## Dense Dependencies

[Dense BLAS/LAPACK Wrapper Coverage](../linalg/dense_blas_lapack_coverage.md)
tracks provider and quarantined helper coverage used by projected problems. It
is maintained as a linalg inventory rather than part of the Krylov algorithm
contract.

## Exploratory Policy

- [Real Nonsymmetric Arnoldi for iDMRG](real_nonsymmetric_idmrg_policy.md)
  records exploratory policy for real nonsymmetric transfer problems. It is not
  a statement that all proposed result forms are implemented.

## Source Navigation

- [Krylov source map](../../src/uni20/krylov/README.md)
- [Dense linalg source map](../../src/uni20/linalg/README.md)
- [Krylov tests](../../tests/krylov/)
- [Krylov examples](../../examples/krylov/README.md)
