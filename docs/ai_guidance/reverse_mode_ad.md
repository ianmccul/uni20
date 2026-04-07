# Uni20 Reverse-Mode AD: AI Guidance

This file is for questions about `Var<T>`, `ReverseValue<T>`, `backprop()`, gradient materialization, complex gradients, and custom reverse-mode kernels.

## File-level invariants

- `Var<T>` is the user-facing reverse-mode type.
- `Var<T>` uses the same async runtime as ordinary Uni20 async code.
- Uni20 reverse mode is dataflow-based.
- Uni20 reverse mode is not tape-replay-based.
- Current reverse accumulation is deterministic.

## `Var<T>`

### ROLE

- `Var<T>` is the user-facing reverse-mode variable type.
- `Var<T>` represents a differentiable value.

### INVARIANTS

- `Var<T>::value` is the forward `Async<T>` channel.
- `Var<T>::grad` is the reverse `ReverseValue<T>` channel.
- `Var<T>` is the current public name.
- `Dual<T>` is not the current public name.

### CAUSAL MODEL

- Forward kernels write `Var<T>::value` through ordinary async buffers.
- Reverse kernels read upstream gradients from `ReverseValue<T>` buffers.
- Reverse kernels write downstream contributions into other `ReverseValue<T>` channels.

### LIFETIME / OWNERSHIP

- `Var<T>` owns its `value` channel and its `grad` channel by value.
- `Var<T>` does not store a plain eager gradient scalar.

### FAILURE MODES

- Using `Var<T>` as if every node necessarily already had a materialized gradient.

### MISCONCEPTIONS

- `Var<T>` implies a separate global backward interpreter.
- `Var<T>` is just a renamed tape-based AD variable.

### RELATED

- `ReverseValue<T>`
- `backprop()`
- gradient materialization

## `ReverseValue<T>`

### ROLE

- `ReverseValue<T>` is the accumulation object for reverse mode.
- `ReverseValue<T>` stores gradient contributions for a `Var<T>`.

### INVARIANTS

- `ReverseValue<T>` exposes async reads and writes for reverse propagation.
- Current reverse accumulation is deterministic.

### CAUSAL MODEL

- `ReverseValue<T>` owns an internal `Async<T>` gradient channel.
- `ReverseValue<T>` creates read and write buffers over that channel.
- Current implementation creates those buffers through `ReverseEpochQueue`.
- Reverse kernels await those buffers exactly as ordinary async kernels do.

### LIFETIME / OWNERSHIP

- `ReverseValue<T>` owns the internal async gradient channel.
- `ReverseValue<T>` owns the reverse-ordering state that sequences accumulation.
- `final()` and `backprop()` expose that same owned gradient channel after finalization.

### FAILURE MODES

- Attaching gradient contributions while reasoning as if ordering were unordered and non-deterministic.
- Ignoring cancellation on reverse paths.

### MISCONCEPTIONS

- `ReverseValue<T>` is just a passive numeric field.
- `ReverseValue<T>` requires a separate global backward pass to become meaningful.

### RELATED

- `Var<T>`
- `backprop()`
- `ReverseEpochQueue`

## `backprop()`

### ROLE

- `backprop()` exposes the async gradient channel.

### INVARIANTS

- `backprop()` does not start a separate global backward phase.
- `backprop()` does not replay a tape.
- `x.backprop().get_wait()` is the canonical fully-async readout pattern.
- Use `wait()` when you only need completion and not the gradient value itself.

### CAUSAL MODEL

- A seeded downstream gradient makes the reverse graph runnable.
- Reverse kernels then propagate contributions through `ReverseValue<T>` buffers.
- `backprop()` finalizes and returns the underlying async gradient channel.

### LIFETIME / OWNERSHIP

- `backprop() &` returns a reference to the owned finalized `Async<T>` gradient channel.
- `backprop() const&` returns a const reference to the same owned channel.
- `backprop() &&` moves that owned `Async<T>` channel out.

### FAILURE MODES

- Treating `backprop()` as an imperative trigger for a separate backward execution stage.
- Waiting on `backprop()` without any seeded downstream gradient and assuming a value must exist.

### MISCONCEPTIONS

- `backprop()` means "run backward now".
- `backprop()` means "replay the tape".

### RELATED

- `Var<T>`
- `ReverseValue<T>`
- gradient materialization

## gradient materialization

### ROLE

- Gradient materialization is the point at which a gradient becomes concrete.

### INVARIANTS

- Gradients are not assumed to exist eagerly everywhere.
- A gradient becomes concrete when an upstream gradient is seeded.
- A gradient becomes concrete when reverse kernels run.
- A gradient becomes concrete when contributions accumulate into `ReverseValue<T>`.

### FAILURE MODES

- Assuming every node always has a materialized gradient.
- Ignoring the possibility of absent gradients when no seed exists.
- Ignoring the possibility of absent gradients when cancellation interrupts propagation.

### MISCONCEPTIONS

- Gradient materialization is global and eager.
- Gradient materialization happens at `Var<T>` construction time.

### RELATED

- `Var<T>`
- `ReverseValue<T>`
- `backprop()`

## Wirtinger `dL/dz*`

### ROLE

- `dL/dz*` is the complex-gradient convention used by Uni20 for real scalar losses.

### INVARIANTS

- Uni20 uses the Wirtinger `dL/dz*` convention for real scalar losses.
- For holomorphic functions, reverse accumulation has the form `out_grad += in_grad * conj(df_dz)`.
- If both `df/dz` and `df/dz*` matter, the reverse update must include both terms.

### MISCONCEPTIONS

- Complex reverse mode can ignore conjugation by default.
- The holomorphic formula is sufficient for every complex operation.

### RELATED

- `Var<std::complex<T>>`
- `ReverseValue<std::complex<T>>`

## custom reverse kernel

### ROLE

- A custom reverse kernel defines how a custom differentiable operation propagates gradients.

### INVARIANTS

- A custom differentiable operation usually creates a result `Var<T>`.
- A custom differentiable operation usually schedules a forward kernel.
- A custom differentiable operation usually schedules a reverse kernel.
- A custom differentiable operation usually returns the result `Var<T>`.

### CAUSAL MODEL

- Reverse kernels should read the upstream gradient.
- `or_cancel()` is often the correct read form for the upstream gradient.
- Reverse kernels should read any forward values needed for the derivative.
- Reverse kernels create dependency edges by awaiting those gradient and value buffers.
- Reverse kernels should accumulate into input gradients with `co_await out_grad += ...`.

### FAILURE MODES

- Ignoring cancellation on the upstream gradient path.
- Forgetting release ordering when a reverse kernel both reads and later writes related async state.
- Capturing coroutine state instead of passing explicit buffer parameters.

### MISCONCEPTIONS

- Reverse kernels may ignore the same lifetime and ordering rules as ordinary async kernels.
- Reverse kernels are safe to write as captured coroutine lambdas.

### RELATED

- `Var<T>`
- `ReverseValue<T>`
- `or_cancel()`

## Current API rough edges

### INVARIANTS

- `ReverseValue<T>` currently has overlapping surface area.
- `final_wait()` and `get_wait()` are effectively aliases.
- `backprop()`, `input()`, `read()`, and `output()` expose related machinery at different levels.
- API simplification is a legitimate design topic.
- API simplification must preserve async semantics and deterministic accumulation semantics.

## Push-back triggers

- Push back if a proposal describes Uni20 AD as tape-based.
- Push back if a proposal assumes eager global gradient materialization.
- Push back if a proposal treats unordered concurrent gradient writers as the current main mechanism.
- Push back if a proposal ignores cancellation on reverse paths.
- Push back if a proposal describes `backprop()` as a separate imperative backward phase.

## Related detailed docs

- `../async/reverse_mode_ad.md`
- `../async/buffers_and_awaiters.md`
- `../async/runtime_model.md`
- `../roadmap.md`
