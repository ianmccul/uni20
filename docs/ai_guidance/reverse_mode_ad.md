# Uni20 Reverse-Mode AD: AI Guidance

- **Audience:** remote assistants, coding agents, and reviewers
- **Authority:** non-normative retrieval summary
- **Reviewed against:** `Uni20-dev/uni20` `main`, 2026-07-18
- **Canonical sources:** `docs/async/reverse_mode_ad.md`,
  `src/uni20/async/var.hpp`, `src/uni20/async/var_toys.hpp`,
  `src/uni20/async/reverse_value.hpp`, and async AD tests

## Core invariants

- `Var<T>` is the user-facing reverse-mode type.
- `Var<T>` combines a forward `Async<T>` value and reverse `ReverseValue<T>` channel.
- Reverse mode uses the ordinary async runtime.
- Reverse mode is dataflow-based, not tape-replay-based.
- Operations schedule forward and reverse coroutine work as the expression is built.
- Reverse work waits for gradients through async buffers.
- Current accumulation is deterministic.

## `Var<T>`

- `value` is the forward async channel.
- `grad` is the reverse accumulation channel.
- `Var` owns both channels by value.
- Custom operations commonly take `Var<T>` by value.
- Copying from a non-const lvalue links gradient flow; do not assume ordinary
  value-copy semantics without inspecting the current API.

## `ReverseValue<T>`

- Owns the internal async gradient channel and reverse-ordering state.
- Exposes input/read buffers for upstream gradients and output/write buffers for
  downstream accumulation.
- A missing gradient is not automatically numeric zero.
- An unseeded/discarded branch may be pruned through cancellation-aware reads.

## Lifecycle

1. Build an expression.
2. Seed the output gradient, commonly `loss.grad = 1`.
3. Finalize any retained named intermediate gradients required by the current API.
4. Read needed gradients through `backprop()` or convenience waits.

Important current rule:

- Temporary `Var` gradient channels finalize when their descriptors are destroyed.
- A named intermediate retained while execution proceeds may need explicit
  `intermediate.grad.finalize()` after graph construction.
- Do not omit this rule from custom-operation or debugging advice.

`backprop()` exposes/finalizes the async gradient channel. It does not launch a
separate global backward interpreter and does not replay a tape.

## Reverse-kernel pattern

- Use captureless `static` coroutine lambdas.
- Pass buffers explicitly.
- Read the upstream gradient through `or_cancel()` before generic arithmetic.
- Read forward values needed for derivatives.
- Release reads when no longer needed.
- Accumulate contributions through the input variable's reverse output buffer.
- Preserve ordinary async lifetime, ordering, exception, and cancellation rules.

An unseeded upstream gradient is an absent value. Its `or_cancel()` path can prune
the reverse coroutine. A genuine exception is not converted into pruning.

## Complex gradients

- Uni20 uses the Wirtinger `dL/dz*` convention for a real scalar loss.
- General reverse update:
  `out_grad += in_grad * conj(df_dz) + conj(in_grad) * df_dz_conj`.
- For holomorphic functions:
  `out_grad += in_grad * conj(df_dz)`.
- Do not omit conjugation or apply the holomorphic formula to non-holomorphic operations.

## Current boundary

- Async value-level AD foundations are implemented.
- Tensor-linalg differentiation is not yet wired through the Tensor operation layer.
- `ReverseValue` has overlapping convenience and buffer-level APIs; simplification
  must preserve finalization, cancellation, and deterministic accumulation.

## Push-back triggers

- Describing Uni20 AD as tape based.
- Treating `backprop()` as “run backward now.”
- Assuming all gradients exist eagerly or that absent means zero.
- Omitting explicit finalization for retained intermediates.
- Ignoring `or_cancel()` on an optional reverse path.
- Capturing state in a reverse coroutine lambda.
