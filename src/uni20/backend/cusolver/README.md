# src/uni20/backend/cusolver

This directory is the cuSOLVER backend-library wiring point. It currently holds
target scaffolding for future cuSOLVER wrappers.

## Contents

- `CMakeLists.txt`: cuSOLVER backend target setup.

## Notes

- cuSOLVER support should be gated by both build-time availability and
  operation-specific runtime capability checks.
- Higher-level entry points belong in the
  [linalg backend layer](../../linalg/backends/README.md).

## Related Documentation

- [Backend source layer](../README.md)
- [cuSOLVER architecture](../../../../docs/backends/cuda/cusolver.md)
- [CUDA runtime resolution](../../../../docs/backends/cuda/runtime_resolution.md)
