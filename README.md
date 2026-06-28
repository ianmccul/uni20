# uni20

This is the **uni20** tensor network library. It is currently in a very early stage of development, and is not yet useful for anything.

## Documentation

- [Scalar type policy](docs/scalar_policy.md): project-level scalar aliases,
  including the rule to spell complex scalar types as `uni20::complex<T>`.
- [Krylov algorithms](docs/krylov_algorithms.md): implemented Krylov solvers,
  supported scalar types, public parameters, defaults, and internal tuning.
- [Krylov solver defaults](docs/krylov_solver_defaults.md): focused note on
  default `ncv` and `nkeep` policies.
- [Krylov precision validation](docs/krylov_precision_validation.md): current
  precision-path validation matrix.
- [Tensor-network linear algebra API survey](docs/tensor_network_linalg_survey.md):
  wrapper priorities for Krylov and tensor-network dense kernels.

## Krylov / ARPACK Relationship

The matrix-free Krylov solvers are native Uni20 code. They follow the same
implicit-restart ideas as ARPACK where that behavior is useful, but core Uni20
does not vendor ARPACK or require a Fortran toolchain. ARPACK comparisons,
larger benchmark fixtures, and oracle dashboards belong in a separate
validation repository.
