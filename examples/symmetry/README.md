# Symmetry Examples

## Construction And Storage

`block_tensor_example.cpp` constructs the supported sparse `BlockTensor`
shapes:

- an order-two block matrix using `SeparateSparseBlockStorage`;
- an order-three MPS-like tensor mixing `BlockSpace` and `LocalSpace`, whose
  numerical blocks are matrices;
- a zero-copy within-boundary permutation view of that MPS tensor;
- a zero-copy right-edge repartition view of that MPS tensor; and
- an order-four scalar-block MPO-like tensor using four `LocalSpace` factors
  and `PackedSparseBlockStorage`, whose numerical blocks are rank zero.

The example prints logical order, key-coordinate count, dense-block order,
stored and legal block counts, contracts the block matrix with itself, verifies
that permutation and repartition retain the source payload address, and reports
the current packed scalar-block byte costs. It also constructs an
`AsyncSeparateSparseBlockStorage` matrix, schedules its block contraction, and
waits for the resulting block through `DebugScheduler`.

## Product State

`block_tensor_product_state_example.cpp` constructs a two-site spin-half
product state with U(1) `Sz` symmetry. Each MPS site has one stored block and
one-dimensional cumulative-charge bond sectors. Contracting the two sites over
their exactly matched shared bond produces the `|up down>` two-site tensor with
unit amplitude, without materializing a dense symmetry-free state.

## AKLT State

`block_tensor_aklt_example.cpp` constructs the standard spin-1 AKLT MPS tensor
resolved by U(1) `Sz`. Its virtual `BlockSpace` contains charges `-1/2` and
`+1/2`; its physical `LocalSpace` contains `+1`, `0`, and `-1`. The four stored
blocks implement the usual bond-dimension-two AKLT matrices. The example checks
the one-site Frobenius norm, contracts two copies over their virtual bond, and
verifies the known `(+1,-1)` and `(0,0)` amplitudes.

## Sector-Global SVD Truncation

`block_tensor_svd_truncation_example.cpp` constructs a block matrix with three
singular states spread across two U(1) charge sectors. It factorizes each
sector once, applies a global maximum-rank policy, and independently
materializes the kept and complementary factors with distinct bond labels. The
example prints the globally ordered charge-labelled spectrum and verifies the
discarded weight.

## SVD Null Space

`block_tensor_svd_nullspace_example.cpp` constructs a rectangular block matrix
and requests a full right singular-vector basis. From the same decomposition
it materializes a rank-one retained factorization and the unpaired right null
space. It directly verifies that the null vector has unit norm and is
annihilated by the original block matrix.

All five executables are registered with CTest as executable documentation.
