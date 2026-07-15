# src/uni20/kernel

This directory contains low-level tensor kernel entry points over resolved
views. Kernels are below tensor ownership, symmetry metadata, and async
scheduling policy; callers should lower to explicit views before reaching this
layer.

## Contents

- `operations.hpp`: Doxygen grouping for low-level tensor kernel interfaces.
- `contract.hpp`: generic contraction entry point backed by the CPU reference
  implementation.
- [`cpu/`](cpu/): always-available CPU kernels.

## Notes

- Backend dispatch for dense linear algebra lives in `linalg/`; this directory
  does not define a separate selector hierarchy.
- Keep kernel code free of block-sparse or quantum-number decisions. Symmetry
  code should resolve legal dense block work before calling kernels.

## Related Documentation

- [Source tree map](../)
- [Kernel dispatch](../../../docs/architecture/kernel_dispatch.md)
- [Raw primitives and symmetric lowering](../../../docs/symmetry/raw_primitives_and_lowering.md)
