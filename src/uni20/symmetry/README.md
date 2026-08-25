# src/uni20/symmetry

This directory contains quantum-number and block-sparse symmetry infrastructure.
These types define which sectors and blocks exist; losing this metadata changes
the mathematical problem and is treated as a correctness bug.

## Contents

- `qnum.hpp`: quantum-number value types.
- `block_sector.hpp`: shared `(QNum, dimension)` block records.
- `space.hpp`: common `Space` and `SymmetrySpace` concepts.
- `dual_space.hpp`: generic `Dual<S>` basis adaptor and `DualSpace` concept.
- `irregular_space.hpp`: immutable ordered segmented spaces with repeated
  sectors.
- `local_space.hpp`: immutable ordered local-state spaces.
- `qnum_space.hpp`: single-irrep tensor spaces.
- `dense_space.hpp`: symmetry-neutral dense tensor spaces.
- `morphism_boundary.hpp`: ordered `Domain<...>` and `Codomain<...>` space
  values.
- `block_key.hpp`: opaque logical block coordinates.
- `block_tensor_space_traits.hpp`: independent block-key-coordinate and
  dense-axis classification for concrete space kinds.
- `block_tensor_storage.hpp`: separate and packed sparse host storage policies.
- `block_tensor.hpp`: the first order-zero through order-four `BlockTensor`
  slice for `LocalSpace`, `QNumSpace`, `BlockSpace`, and explicit dual
  boundaries.
- `block_tensor_mapped_view.hpp`: shared logical-to-physical key and dense-axis
  mapping for zero-copy views.
- `block_tensor_permute.hpp`: zero-copy bosonic permutations within domain and
  codomain.
- `block_tensor_repartition.hpp`: zero-copy bosonic left/right wire-bending
  views with transformed logical keys and strided dense blocks.
- `block_tensor_linear.hpp`: structure-preserving copy, zero, scaling,
  addition, AXPY, inner-product, and norm operations.
- `block_tensor_contract.hpp`: adjacent pairwise sparse contraction over exact
  codomain/domain space values.
- `symmetry.hpp`, `symmetryimpl.hpp`, `symmetryfactor.hpp`: symmetry
  declarations, implementations, and factor helpers.
- `u1.hpp`: U(1) symmetry support.
- `block_space.hpp`: immutable coalesced block spaces and explicit
  local-to-block regularization.

## Notes

- Symmetry-aware tensor paths must preserve quantum-number, block, and leg
  orientation metadata.
- Dense projections are allowed only as explicitly named diagnostics or
  reference conversions. They must not silently feed back into a symmetry-aware
  computation.

## Related Documentation

- [Source tree map](../)
- [Symmetry and block-sparse documentation](../../../docs/symmetry/)
- [Quantum numbers and symmetry](../../../docs/symmetry/qnum.md)
- [Raw primitives and symmetric lowering](../../../docs/symmetry/raw_primitives_and_lowering.md)
