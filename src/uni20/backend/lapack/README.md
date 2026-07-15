# src/uni20/backend/lapack

This directory contains raw and checked LAPACK wrappers for configured CPU
providers. It is the external-library boundary below dense linalg policy.

`uni20_backend_lapack` owns LAPACK discovery and transitive provider linkage.
It remains available when Uni20's BLAS operation backend is disabled; a
conventional LAPACK provider may still link a BLAS implementation internally.

## Contents

- `lapack.hpp`: checked wrappers that translate LAPACK `INFO` values into
  invariant failures, structured exceptions, or operation-specific status
  returns.
- `lapack_error.hpp`: structured terminal LAPACK failure data.
- `lapack_error_presentation.hpp`: terminal and plain-text presentation for
  structured LAPACK failures.
- `common.hpp`: shared LAPACK backend helpers.
- [`reference/`](reference/README.md): ordinary LAPACK provider declarations grouped by routine
  family.
- [`mplapack/`](mplapack/README.md): MPLAPACK-backed declarations for supported extended-precision
  scalar types.

## Notes

- Keep pointer/leading-dimension interfaces and provider-specific overloads in
  this backend layer.
- Shape validation, view materialization, and tensor allocation policy belong in
  higher linalg helpers.

## Checked `INFO` Policy

Unchecked provider wrappers return LAPACK `INFO` verbatim. Checked wrappers use
the following policy after Uni20 has selected a routine and constructed its
arguments:

- `INFO < 0` means the provider rejected a wrapper-generated argument. This is
  a Uni20 lowering or provider-contract bug and unconditionally fails a
  `CHECK`, including when Python exception mode is active.
- `INFO > 0` means the provider ran but did not produce the requested result.
  The wrapper raises a structured `LapackError` containing the routine, `INFO`,
  and interpreted reason. `trace::raise` aborts with a diagnostic in native C++
  mode and preserves `LapackError` as an exception in Python mode.
- A positive value explicitly documented as a usable operation status remains
  a return value. For example, expert solve wrappers report reciprocal condition
  warnings as `bool`, and `trsyl` reports perturbation of close eigenvalues.

These terminal failures are not kernel declines. Once a backend calls LAPACK,
the operands may have been overwritten and dispatch must not continue to a
later backend.

## Related Documentation

- [Backend source layer](../README.md)
- [BLAS/LAPACK mdspan wrappers](../../../../docs/linalg/blas_lapack_wrappers.md)
- [Mdspan linear algebra dispatch](../../../../docs/linalg/mdspan_dispatch.md)
- [Trace macros and failure policy](../../../../docs/diagnostics/trace_macros.md)
