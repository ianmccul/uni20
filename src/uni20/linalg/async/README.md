# src/uni20/linalg/async

This directory contains opt-in wrappers that schedule synchronous Tensor
linear-algebra operations over `Async<Tensor>` values.

## Contents

- `matrix_product.hpp`: all-async fixed-output `gemm`, resizing
  `assign_product`, and fixed-output `add_product` wrappers.
- `dispatch.hpp`: coroutine-aware kernel dispatch; ordinary backends run their
  blocking `try_kernel` directly, while individual backend/operation pairs may
  provide a deferred task through `try_make_kernel_task`.
- `kernel_task.hpp`: clean-decline, completed-success, or deferred-task result
  returned by an optional coroutine backend implementation.
- `lq.hpp`: preserving and consuming reduced real LQ wrappers with independent
  async factor outputs.
- `reductions.hpp`: full and axis-selective async sums with storage-preserving
  or host-scalar results.
- `qr.hpp`: preserving and consuming reduced real QR wrappers with independent
  async factor outputs.
- `self_adjoint_eigh.hpp`: preserving and consuming `eigh` wrappers with
  independent async eigenvalue and eigenvector outputs.
- `svd.hpp`: preserving and consuming exact `singular_values`, `svd_left`,
  `svd_right`, and `svd` wrappers with independent async outputs.
- `truncated_svd.hpp`: preserving and consuming `truncated_svd` wrappers with
  independent `U`, `s`, `Vh`, and truncation-information outputs.
- `transform.hpp`: variadic all-async elementwise overwrite and update
  wrappers.
- [`detail/`](detail/): shared fixed-alias output capability used by wrappers
  that write through an owner-retaining descriptor.

## Rules

- Ordinary backend entry points and leaf kernels remain non-suspending and
  scheduler-unaware. `co_dispatch_kernel` invokes `try_kernel` directly unless
  a backend/operation pair provides `try_make_kernel_task`. The outer Tensor
  coroutine retains its epoch buffers until any deferred task has completed
  host-side submission and published storage completion state.
- Every Tensor operand in one wrapper call is an `Async<T>`.
- Wrappers pass buffer handles and all other task state into coroutines by
  value; coroutine lambdas must be captureless and `static`.
- Default selectors are resolved statically from Tensor/storage types before
  scheduling. The runtime backend walk occurs after awaiting the Tensor values
  and normalizing their fixed operands to mdspecs.
- Async wrapper signatures use `TensorView` and `MutableTensorView`; immediate
  accessibility is an internal backend or storage-reuse optimization, not an
  async API precondition.
- Immediate and async scalar operands are normalized with `async::read(...)`;
  the former uses an always-ready `ValueAwaiter` and the latter a real buffer.
- Elementwise callables are immediate operation state. They are moved into the
  coroutine and invoked as const by the synchronous backend after all Tensor
  operands are ready.
- Every writer coroutine parameter is an automatic exception sink. Multi-output
  operations pass each output writer into the coroutine; a consuming input
  writer also receives failures after its value has been taken.
- Queue identity is used only for the cheap, exact output/input alias check.
  General dependency cycles are handled by runtime deadlock diagnostics.
- Variadic transform inputs may share queues with one another, but none may
  share the output queue. An update reads its old output through the sole
  writer and does not enroll an output reader.
- Mutable async aliases are fixed-shape transform outputs. The coroutine copies
  the already-bound descriptor locally while holding its writer; it never
  replaces or retargets the stored alias.
- Sum axes are normalized before submission. Owning output shape preparation
  and backend dispatch occur after the input is readable; mutable alias outputs
  follow the same fixed-descriptor rule as transforms.

See [Async Tensor Kernel Authoring](../../../../docs/async/kernel_authoring.md)
for the complete authoring contract and [Tensor Operations](../../../../docs/tensor/operations.md)
for the current async-support matrix.

[The async TBB matrix-product batch example](../../../../examples/async/async_tbb_matrix_product_batch_example.cpp)
demonstrates these
wrappers as a parallel batch on `TbbScheduler`, with configurable matrix size,
product count, scheduler concurrency, precision, and backend selection. It uses
the normal kernel-dispatch path for `fp32`, `fp64`, and configured `fp128`
operands, and renders the validated result through the presentation layer.

Return to the [linalg source map](../).
