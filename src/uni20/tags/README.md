# `src/uni20/tags`

This directory contains lightweight tag types for selecting or describing
backend families and libraries. Tags should stay cheap, structural, and free of
runtime ownership.

## Contents

- `cpu.hpp`, `cuda.hpp`: execution/backend family tags.
- `blas.hpp`, `lapack.hpp`: CPU library-family tags.
- `cublas.hpp`, `cusolver.hpp`: CUDA library-family tags.

## Notes

- Tags are selection metadata, not capability proofs. Runtime/library support
  still belongs in backend capability checks and dispatch.
- Keep tag headers independent of heavy backend-library includes.
