# Symmetry Examples

`block_tensor_example.cpp` constructs the first supported sparse
`BlockTensor` shapes:

- an order-two block matrix using `SeparateSparseBlockStorage`;
- an order-three MPS-like tensor mixing `BlockSpace` and `LocalSpace`; and
- an order-four scalar-block MPO-like tensor using four `LocalSpace` factors
  and `PackedSparseBlockStorage`.

The example prints stored and legal block counts. It demonstrates storage and
selection-rule behavior only; tensor contractions and async ownership are later
slices.
