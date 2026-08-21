# Symmetry Examples

`block_tensor_example.cpp` constructs the first supported sparse
`BlockTensor` shapes:

- an order-two block matrix using `SeparateSparseBlockStorage`;
- an order-three MPS-like tensor mixing `BlockSpace` and `LocalSpace`, whose
  numerical blocks are matrices; and
- an order-four scalar-block MPO-like tensor using four `LocalSpace` factors
  and `PackedSparseBlockStorage`, whose numerical blocks are rank zero.

The example prints logical order, key-coordinate count, dense-block order,
stored and legal block counts, and the current packed scalar-block byte costs.
It demonstrates storage and selection-rule behavior only; tensor contractions
and async ownership are later slices.
