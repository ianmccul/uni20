# Common Infrastructure Examples

- `buildinfo_example.cpp` renders configured compiler, provider, feature, and
  environment metadata through the presentation layer.
- `gtest_floating_eq_example.cpp` demonstrates real and complex ULP-aware
  GoogleTest assertions. It intentionally contains failing tests and should
  exit nonzero.
- `trace_example.cpp` exercises trace formatting, module channels, containers,
  and thread annotations, then intentionally terminates with `PANIC`.

See the [examples index](../README.md), [Build Information](../../docs/development/build_information.md),
and [Trace Macros](../../docs/diagnostics/trace_macros.md).
