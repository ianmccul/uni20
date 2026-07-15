# Tensor Documentation

This directory owns dense Tensor value, view, scalar, creation, reshape, and
front-end operation semantics.

## Canonical Guides

- [Tensor Operations](operations.md) defines ownership, mutation, return-value,
  materialization, and async-support contracts.
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
