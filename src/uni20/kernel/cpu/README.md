# `src/uni20/kernel/cpu`

This directory contains always-available CPU kernel entry points. These kernels
provide the baseline implementation path when more specialized backend kernels
are unavailable.

## Contents

- `cpu.hpp`: CPU kernel include point and tag integration.
- `contract.hpp`: CPU contraction entry points.
- `CMakeLists.txt`: CPU kernel interface target.

## Notes

- CPU kernels should operate on resolved views and explicit operation tags.
- Keep high-level tensor, async, and symmetry decisions out of this layer.
