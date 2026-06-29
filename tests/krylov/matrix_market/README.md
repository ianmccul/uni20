# Krylov Matrix Market Fixtures

This directory contains small Matrix Market fixtures used by native Krylov
tests and examples.

| Local file | Type | Provenance | Purpose |
| --- | --- | --- | --- |
| `path_laplacian_30.mtx` | real symmetric coordinate, 30 x 30 | Generated fixture | Small symmetric Lanczos Matrix Market example. |
| `complex_phase_triangular_4.mtx` | complex general coordinate, 4 x 4 | Generated fixture | Complex nonsymmetric Arnoldi smoke test with known triangular spectrum. |
| `complex_grcar_8.mtx` | complex general coordinate, 8 x 8 | Generated fixture | Small nonnormal Arnoldi/ARPACK comparison example inspired by Grcar matrices. |
| `tdvp_lanczos/tdvp_ho_n8_hamiltonian.mtx` | real symmetric coordinate, 256 x 256 | Copied from local `cytnx-fixes` TDVP/Lanczos harmonic-oscillator investigation | Real Matrix Market stress case for Krylov exponential stopping diagnostics. |
| `tdvp_lanczos/tdvp_ho_n8_v0.mtx` | real general coordinate, 256 x 1 | Copied from local `cytnx-fixes` TDVP/Lanczos harmonic-oscillator investigation | Initial vector for the TDVP Krylov exponential stress case. |

Larger external fixtures are documented in the collection-specific README files:

- `tdvp_lanczos/README.md`
- `nep/README.md`
- `suitesparse/README.md`
