# `src/uni20/linalg/async`

This directory contains opt-in wrappers that schedule synchronous Tensor
linear-algebra operations over `Async<Tensor>` values.

## Contents

- `matrix_product.hpp`: all-async `assign_product` and `add_product` wrappers.

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
- A writer coroutine parameter is the normal exception sink for a one-output
  operation.
- Queue identity is used only for the cheap, exact output/input alias check.
  General dependency cycles are handled by runtime deadlock diagnostics.

See [`docs/async/kernel_authoring.md`](../../../../docs/async/kernel_authoring.md)
for the complete authoring contract.
