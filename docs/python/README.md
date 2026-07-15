# Python Documentation

- [Python Bindings](bindings.md) describes the current smoke/build-information
  module and the intended presentation boundary.
- [Dtype Promotion](dtype_promotion.md) is a design note for future Tensor
  bindings; it is not current Python API behavior.

Future bindings should validate user input before entering C++ paths whose
contracts use `CHECK`, and should use dynamic kernel dispatch where a missing
compiled backend must become a Python exception.

Binding source lives under [`bindings/python`](../../bindings/python/); Tensor
and scalar behavior originates in the [Tensor source layer](../../src/uni20/tensor/README.md)
and [scalar foundations](../../src/uni20/core/README.md).

The [Python example](../../examples/python/README.md) demonstrates loading the
built module and rendering build information.
