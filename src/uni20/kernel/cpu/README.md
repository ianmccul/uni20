# src/uni20/kernel/cpu

This directory contains always-available CPU kernel entry points. These kernels
provide the baseline implementation path when more specialized backend kernels
are unavailable.

## Contents

- `contract.hpp`: CPU contraction entry points.
- `CMakeLists.txt`: CPU kernel interface target.

## Notes

- CPU kernels should operate on resolved views.
- Keep high-level tensor, async, and symmetry decisions out of this layer.

## Related Documentation

- [Kernel source layer](../)
- [Raw primitives and symmetric lowering](../../../../docs/symmetry/raw_primitives_and_lowering.md)
