# src/uni20/symmetry

This directory contains quantum-number and block-sparse symmetry infrastructure.
These types define which sectors and blocks exist; losing this metadata changes
the mathematical problem and is treated as a correctness bug.

## Contents

- `qnum.hpp`: quantum-number value types.
- `symmetry.hpp`, `symmetryimpl.hpp`, `symmetryfactor.hpp`: symmetry
  declarations, implementations, and factor helpers.
- `u1.hpp`: U(1) symmetry support.
- `block_space.hpp`: block-space metadata.

## Notes

- Symmetry-aware tensor paths must preserve quantum-number, block, and leg
  orientation metadata.
- Dense projections are allowed only as explicitly named diagnostics or
  reference conversions. They must not silently feed back into a symmetry-aware
  computation.

## Related Documentation

- [Source tree map](../README.md)
- [Symmetry and block-sparse documentation](../../../docs/symmetry/README.md)
- [Quantum numbers and symmetry](../../../docs/symmetry/qnum.md)
- [Raw primitives and symmetric lowering](../../../docs/symmetry/raw_primitives_and_lowering.md)
