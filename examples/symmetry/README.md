# Symmetry Examples

`block_tensor_example.cpp` constructs the first supported sparse
`BlockTensor` shapes:

- an order-two block matrix using `SeparateSparseBlockStorage`;
- an order-three MPS-like tensor mixing `BlockSpace` and `LocalSpace`, whose
  numerical blocks are matrices;
- a zero-copy within-boundary permutation view of that MPS tensor;
- a zero-copy right-edge repartition view of that MPS tensor; and
- an order-four scalar-block MPO-like tensor using four `LocalSpace` factors
  and `PackedSparseBlockStorage`, whose numerical blocks are rank zero.

The example prints logical order, key-coordinate count, dense-block order,
stored and legal block counts, verifies that permutation and repartition retain
the source payload address, and reports the current packed scalar-block byte
costs. Tensor contractions and async ownership are later slices.
