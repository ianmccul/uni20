# CUDA Backend Documentation

Uni20 has CUDA task and build scaffolding, but no complete Tensor CUDA backend
on the current main branch. Documents in this directory therefore define design
constraints and preserve ecosystem research rather than promising runnable
Tensor operations.

## Active Design

- [Runtime Model](runtime.md)
- [Runtime Resolution](runtime_resolution.md)
- [GPU Epoch Design](epoch_design_draft.md)
- [cuSOLVER Architecture](cusolver.md)
- [Memory Allocation](memory_allocation.md)

## Background

- [Backend Library Compatibility](libraries.md) records provider/version
  constraints.
- [GPU Landscape](landscape.md) surveys tensor-network GPU libraries and
  implementation choices.
