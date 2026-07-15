# AD Examples

These programs exercise the current value-level async reverse-mode experiments.
Tensor AD is future work.

- `ad_example.cpp` is an early reusable-graph sketch over `Async<double>`. It
  uses low-level `unsafe_set`/`unsafe_value` operations and is retained as a
  development experiment, not the canonical async authoring pattern.
- `gradient_solver.cpp` builds a scalar `Var<double>` loss graph and performs
  repeated asynchronous gradient-descent updates with `DebugScheduler`.

See the [examples index](../), [Reverse-Mode AD](../../docs/async/reverse_mode_ad.md),
and the [Async Runtime Model](../../docs/async/runtime_model.md).
