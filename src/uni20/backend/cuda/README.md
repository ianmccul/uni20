# `src/uni20/backend/cuda`

This directory is the CUDA backend-library wiring point. It currently contains
target scaffolding rather than mature CUDA backend wrappers.

## Contents

- `CMakeLists.txt`: CUDA backend target setup.

## Notes

- Keep CUDA toolkit discovery and target properties local to backend targets.
- Runtime capability checks should remain operation-specific; do not assume one
  CUDA build option makes every CUDA library feature available.
