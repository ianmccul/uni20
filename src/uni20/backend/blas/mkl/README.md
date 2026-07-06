# `src/uni20/backend/blas/mkl`

This directory contains MKL-specific BLAS backend wiring. It selects the MKL
integer interface from the Uni20 LP64/ILP64 configuration and exposes separate
sequential or threaded target variants.

## Contents

- `backend_mkl.hpp`: MKL backend tag/include point.
- `reference_blas_mkl_direct.hpp`: direct MKL BLAS declarations/adapters.
- `CMakeLists.txt`: MKL package discovery and target setup.

## Notes

- Keep MKL threading and link quirks isolated in this directory.
- Do not make generic BLAS callers depend on MKL-only headers.
