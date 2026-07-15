# Krylov Examples

- `krylov_matrix_market_example.cpp` runs the symmetric/Hermitian Lanczos
  driver, including regular and transformed modes, on Matrix Market input.
- `krylov_nonsymmetric_matrix_market_example.cpp` runs real nonsymmetric
  Arnoldi on Matrix Market input.
- `krylov_exponential_probe_example.cpp` compares Krylov and Taylor
  exponential actions with exact synthetic references and estimator details.
- `krylov_exponential_orthogonality_example.cpp` isolates the tolerance floor,
  basis orthogonality, and reorthogonalization behavior.
- `krylov_exponential_matrix_market_probe_example.cpp` reproduces the TDVP
  harmonic-oscillator convergence case from tracked Matrix Market fixtures.

Several invocations are registered with CTest, including optional binary128
runs when MPLAPACK is configured. See the [examples index](../README.md),
[Krylov documentation](../../docs/krylov/README.md), and
[Krylov test matrices](../../docs/krylov/test_matrices.md).
