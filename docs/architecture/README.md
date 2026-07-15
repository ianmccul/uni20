# Architecture Documentation

This directory explains how Uni20's tensor, async, dispatch, and future
heterogeneous-execution layers fit together.

## Current Contracts

- [Architecture Overview](overview.md) is the concise map of implemented and
  planned layers.
- [Kernel Dispatch](kernel_dispatch.md) is the canonical contract for operation
  tags, backend lists, type probing, runtime decline, diagnostics, and dispatch
  failure.

## Supporting Design

- [Backend Dispatch](backend_dispatch.md) develops the per-backend capability
  and clean-decline contract generalized by kernel dispatch.
- [Execution Architecture](execution.md) records the forward CPU, CUDA, and MPI
  mechanism/policy split.
- [Ordering and Backend Lowering](ordering_and_backend_lowering.md) assigns
  ordering, completion, and lifetime responsibilities across layers.
- [Storage Kind and Location](storage_kind_and_location.md) separates
  compile-time memory semantics from runtime placement.

When the two dispatch documents overlap, `kernel_dispatch.md` defines the
implemented dispatcher and `backend_dispatch.md` supplies rationale and future
backend guidance. The execution and lowering documents are active design notes,
not claims that CUDA or MPI execution is complete.

## Source Navigation

- [Source tree map](../../src/uni20/)
- [Dense linalg and dispatch](../../src/uni20/linalg/)
- [Async runtime](../../src/uni20/async/)
- [Provider backends](../../src/uni20/backend/)
- [Storage policies](../../src/uni20/storage/)
