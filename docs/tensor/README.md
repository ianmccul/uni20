# Tensor Documentation

This directory owns dense Tensor value, view, scalar, creation, reshape, and
front-end operation semantics.

## Canonical Guides

- [Mdspec](mdspec.md) defines the structural
  `(data_descriptor, mapping, accessor)` representation, tensor-level view and
  lease concepts, and the initial host/CUDA acquisition APIs.
- [Tensor Operations](operations.md) defines ownership, mutation, return-value,
  materialization, and async-support contracts.
- [Scalar Tensors, Host Scalars, and Storage Transfer](scalar_tensors_and_transfer.md)
  separates rank-zero Tensor results, host control-flow values, and future
  device migration.
- [Generated Tensors and Reshape](creation_and_reshape.md) defines generated
  values, layout-aware materialization, and reshape behavior.
- [Scalar Policy](scalar_policy.md) defines Uni20 scalar spellings, concepts,
  promotion boundaries, numeric limits, and optional binary128 behavior.

## Background Design

- [Tensor Dispatch and View Semantics Draft](dispatch_and_view_semantics_draft.md)
  preserves the reasoning that led to the current Tensor and dispatch APIs.
  Prefer the canonical guides above for implemented behavior.

Async ownership and alias lifetime are documented in
[Async Storage](../async/storage.md). Backend execution starts at the
[Kernel Dispatch](../architecture/kernel_dispatch.md) layer.

## Source Navigation

- [Tensor values, views, and operations](../../src/uni20/tensor/)
- [Storage policies](../../src/uni20/storage/)
- [Mdspan concepts and accessors](../../src/uni20/mdspan/)
- [Scalar foundations](../../src/uni20/core/)
