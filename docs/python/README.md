# Python Documentation

- [Python Bindings](bindings.md) describes the current smoke/build-information
  module, build requirements, error boundary, tests, and presentation guidance.
- [Dtype Promotion](dtype_promotion.md) is a design note for future Tensor
  arithmetic. It is not current Python API behavior.

Binding source lives under [`bindings/python`](../../bindings/python/).
The [Python example](../../examples/python/) demonstrates loading the compiled
module and rendering build information.

Python binding work must start from the native Uni20 contracts rather than
inventing a parallel Python tensor architecture:

- [Tensor Operations](../tensor/operations.md) defines values, views,
  ownership, storage selection, and async lowering.
- [Async Storage](../async/storage.md) defines native `Async<T>` value and alias
  semantics.
- [Kernel Dispatch](../architecture/kernel_dispatch.md) defines the dynamic
  binding boundary for unavailable kernels.
- [Presentation Formatting](../diagnostics/presentation.md) defines shared
  semantic presentation data and Python/notebook rendering constraints.

The representation of arbitrary-rank tensors at the Python boundary is not yet
decided. In particular, the Python docs do not define a second runtime-rank
Tensor, Storage, device, or async API alongside the native Uni20 types.
