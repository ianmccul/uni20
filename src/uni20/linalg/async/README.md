# `src/uni20/linalg/async`

This directory contains opt-in wrappers that schedule synchronous Tensor
linear-algebra operations over `Async<Tensor>` values.

## Contents

- `matrix_product.hpp`: all-async `assign_product` and `add_product` wrappers.
- `self_adjoint_eigh.hpp`: preserving and consuming `eigh` wrappers with
  independent async eigenvalue and eigenvector outputs.

## Rules

- Backends and leaf kernels remain synchronous and async-unaware.
- Every Tensor operand in one wrapper call is an `Async<T>`.
- Wrappers pass buffer handles and all other task state into coroutines by
  value; coroutine lambdas must be captureless and `static`.
- Default selectors are resolved statically from Tensor/storage types before
  scheduling. The runtime backend walk occurs after awaiting and resolving the
  Tensor mdspans.
- Immediate and async scalar operands are normalized with `async::read(...)`;
  the former uses an always-ready `ValueAwaiter` and the latter a real buffer.
- Every writer coroutine parameter is an automatic exception sink. Multi-output
  operations pass each output writer into the coroutine; a consuming input
  writer also receives failures after its value has been taken.
- Queue identity is used only for the cheap, exact output/input alias check.
  General dependency cycles are handled by runtime deadlock diagnostics.

See [`docs/async/kernel_authoring.md`](../../../../docs/async/kernel_authoring.md)
for the complete authoring contract.

`examples/async/async_tbb_matrix_product_batch_example.cpp` demonstrates these
wrappers as a parallel batch on `TbbScheduler`, with configurable matrix size,
product count, scheduler concurrency, precision, and backend selection. It uses
the normal kernel-dispatch path for `fp32`, `fp64`, and configured `fp128`
operands, and renders the validated result through the presentation layer.
