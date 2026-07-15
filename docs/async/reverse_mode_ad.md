# Dataflow Reverse-Mode AD

This document explains the `Var<T>` model used by Uni20.
It focuses on behavior and usage patterns, not internal queue implementation.

## What Is Different from Tape AD

Uni20 AD is dataflow-based, not tape-replay-based.

- Each `Var` operation schedules forward and reverse coroutine work immediately.
- Reverse work waits on gradient availability through async buffers.
- There is no separate "run backward pass over a tape" phase.

This keeps AD on the same runtime model as normal async computation.

## Core Types

| Type | Role |
|---|---|
| `Var<T>` | user-facing AD variable (`value` + `grad`) |
| `Async<T>` | forward value channel |
| `ReverseValue<T>` | gradient accumulation channel |

`Var<T>` is:

- moveable
- copyable from non-const lvalues (this links gradient flow)
- not copyable from const lvalues

For custom ops, the common signature is `Var<T> f(Var<T> x)` (pass by value, not `const&`).

## Typical Lifecycle

1. Build an expression with `Var<T>`.
2. Seed the output gradient (usually `1` for scalar loss): `loss.grad = 1.0;`.
3. Get gradient channels with `backprop()` and wait only where needed.

```cpp
Var<double> x = 0.1;
Var<double> y = sin(x) * x;

y.grad = 1.0;                 // seed dL/dy*
Async<double> dx = x.backprop(); // dL/dx*
fmt::print("dx = {}\n", dx.get_wait());
```

`x.grad.wait()` waits for completion without materializing the value.
`x.grad.get_wait()` and `x.grad.final_wait()` are convenience forms for waiting and reading the value.
The explicit async forms are `x.backprop().get_wait()` and `x.grad.backprop().get_wait()`.

## Scheduler Behavior for AD

- Normal application code can rely on the default global scheduler.
- `ScopedScheduler` is mainly for tests or temporary scheduler overrides.
- Explicit `run_all()` is usually redundant; waiting on outputs/gradients drives execution.
- `run_all()` is still useful with `DebugScheduler` for deterministic debugging.

## Wirtinger Convention (Complex Gradients)

Uni20 propagates gradients in the `dL/dz*` direction (Wirtinger conjugate gradient),
assuming a real-valued scalar loss `L`.

For `f(z, z*)`, the implemented reverse update shape is:

`out_grad += in_grad * conj(df_dz) + conj(in_grad) * df_dz_conj`

where:

- `in_grad` is the upstream `dL/df*`
- `df_dz` is `df/dz`
- `df_dz_conj` is `df/dz*`

For holomorphic functions (`df_dz_conj = 0`), this reduces to:

`out_grad += in_grad * conj(df_dz)`

That is the pattern used by the `sin(Var<T>)` implementation.

## Defining a Custom Var Operation

Recommended structure:

1. Schedule forward computation (`value` path).
2. Schedule backward computation (`grad` path).
3. Return the result `Var`.

### Example Pattern (Holomorphic Unary Function)

```cpp
template <typename T>
Var<T> cube(Var<T> x)
{
  Var<T> result;

  schedule([](ReadBuffer<T> in, WriteBuffer<T> out) static -> AsyncTask {
    auto in_view = co_await in.transfer();
    T value = in_view.get();
    in_view.release();
    co_await out = value * value * value;
  }(x.value.read(), result.value.write()));

  schedule([](ReadBuffer<T> x_value,
              ReadBuffer<T> in_grad,
              WriteBuffer<T> out_grad) static -> AsyncTask {
    auto grad_view = co_await in_grad.transfer().or_cancel();
    T grad = grad_view.get();
    grad_view.release();

    auto x_view = co_await x_value.transfer();
    T value = x_view.get();
    x_view.release();

    T const df_dz = T(3) * value * value;
    co_await out_grad += grad * uni20::conj(df_dz);
  }(x.value.read(), result.grad.input(), x.grad.output()));

  return result;
}
```

Important points:

- coroutine lambdas must be `static` and captureless
- use `result.grad.input()` for upstream gradient
- use `x.grad.output()` for accumulation into inputs
- make the upstream gradient read with `or_cancel()` the first awaited reverse
  dependency
- do not pass a possibly absent gradient through generic async arithmetic
  before this read; generic arithmetic correctly treats a missing operand as an
  invalid computation rather than AD pruning

An output gradient that is never seeded is an absent value, not a zero tensor
and not a failure. Its `or_cancel()` reader prunes that reverse coroutine, whose
unwritten output gradient is then absent in the preceding accumulation stage.
This permits unused differentiable branches to leave scope harmlessly. A
genuine exception from a forward or reverse dependency is never converted into
this cancellation path; it propagates if the failed result is consumed and is
discarded only when the result and all of its dependents are unobserved.

The gradient channel of a temporary `Var` is finalized when that descriptor is
destroyed. A named intermediate retained while the graph is executed must
currently be finalized explicitly after graph construction:

```cpp
output.grad = 1.0;
intermediate.grad.finalize();
auto input_gradient = input.grad.backprop();
```

This tells `ReverseValue` that no further gradient contributions will be
scheduled for the intermediate. A future graph- or tape-level finalization API
may centralize this bookkeeping, but reverse operations must still use the
upstream-first pruning rule above.

`async_ad_stress_example` exercises this behavior with deterministic discarded
branches, deliberately failing unobserved branches, and an observed failure
whose original exception must survive:

```bash
./examples/async_ad_stress_example --threads=4 --iterations=10 \
    --depth=16 --dead-branches=32 --failing-branches=8 --seed=20
```

## Notes on Current ReverseValue API

The `ReverseValue` surface currently has overlap:

- `get_wait()` and `final_wait()` are equivalent
- `backprop()` exposes the underlying `Async<T>` channel
- `input()`/`read()` and `output()` expose buffer-level access

For most user code, the minimal set is:

- seed with `loss.grad = seed`
- read with `x.grad.get_wait()`

## Key implementation files

- `src/uni20/async/var.hpp`
- `src/uni20/async/var_toys.hpp`
- `src/uni20/async/reverse_value.hpp`
- `tests/async/test_var.cpp`
- `tests/async/test_reverse_value.cpp`
