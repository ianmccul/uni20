# Design Papers

This directory stores long-form source documents and rendered artifacts that do
not fit the ordinary Markdown reference set.

- `planar-tensor-network-design.tex` establishes the ordered planar-network
  model, explicit braiding, and interface invariants.
- `block-tensor-spaces-and-morphisms.tex` refines that model for `BlockTensor`.
  In particular, it replaces the earlier paper's `Space`/`CoSpace` spelling
  with independent `Space`/`DualSpace` object duality and
  `Domain`/`Codomain` morphism side.
- `tensor_contraction.tex` and `tensor_contraction.pdf` preserve a November 2025
  background paper on the earlier contraction design.

The paper is historical background. Its source paths, Tensor vocabulary, and
kernel layering predate the current Tensor and operation-tag dispatcher. Use
[Tensor Operations](../tensor/operations.md),
[Kernel Dispatch](../architecture/kernel_dispatch.md), and the
[documentation index](../) for current contracts.
