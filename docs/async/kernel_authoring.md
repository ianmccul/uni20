# Async Tensor Kernel Authoring

This guide defines the first supported pattern for lifting synchronous Tensor
operations onto Uni20's async runtime. The implemented reference is
`uni20::linalg::assign_product` and `uni20::linalg::add_product` from
`<uni20/linalg/async.hpp>`.

## Layer Boundary

An async Tensor operation is a scheduling wrapper around an existing
synchronous Tensor operation:

```text
Async<Tensor> handles
  -> static backend-selector resolution
  -> async::read(...) normalization of scalar operands
  -> ReadBuffer / WriteBuffer enrollment
  -> scheduled coroutine
  -> await stored Tensor values and async scalars
  -> synchronous Tensor operation with the resolved selector
  -> mdspan dispatch and backend kernel
```

Backends and leaf kernels remain unaware of `Async<T>`, epoch queues, and
schedulers. The async wrapper resolves the storage policy's static selector.
The synchronous operation still owns shape validation, mdspan resolution, and
the runtime backend walk.

Do not add async overloads directly to a backend or make `Async<Tensor>` model
`TensorView`.

## Public Operand Contract

The first checkpoint uses a strict all-async interface:

```cpp
void assign_product(Async<OutputTensor>& output,
                    Async<LhsTensor> const& lhs,
                    Async<RhsTensor> const& rhs,
                    Alpha&& alpha = Scalar{1});
```

The rules are:

- Every caller-supplied Tensor operand is an exact `Async<T>`.
- Synchronous and asynchronous Tensor operands are not mixed in one call.
- Inputs are passed as `Async<T> const&` and the output as `Async<T>&`.
- A scalar may be immediate or an async reader yielding a compatible value.
  `async::read(...)` turns the former into an always-ready `ValueAwaiter` and
  the latter into its real read buffer.
- Selectors, configuration values, and other non-Tensor state enter the
  coroutine by value. Async non-Tensor state enters through its buffer.
- A by-value object is not necessarily owning. Spans, string views, mdspans,
  and other borrowed descriptors still need an owner whose lifetime covers the
  task.

Async Tensor aliases such as `async::conj(tensor)` are valid input operands.
Their `Async<View>` handles retain the parent storage and share its epoch queue.

## Submission and Coroutine Phases

The non-coroutine wrapper performs only work that is valid on the async
handles:

1. Resolve the immutable default selector from the Tensor/storage types, or
   accept an explicit selector by value.
2. Check exact output/input queue aliasing.
3. Normalize each immediate-or-async scalar with `async::read(...)`.
4. Enroll one `WriteBuffer` for the output and one `ReadBuffer` for each Tensor
   input.
5. Move the selector, awaiters, and ordinary values into a coroutine.
6. Schedule the task.

The coroutine then:

1. Awaits the Tensor buffers and any async scalar buffers together.
2. Resolves references to the stored Tensor values.
3. Prepares an unconstructed overwrite output when supported.
4. Calls the existing synchronous Tensor operation with the resolved selector.

Use a named free coroutine, or a captureless C++23 `static` lambda. Never pass
references to the `Async<T>` handles into the coroutine. The moved buffers
already retain storage, queue context, and the selected epochs, so the original
handles may be destroyed immediately after submission.

## Access and Aliasing

An ordinary one-output operation has one writer and any number of input
readers. A mutating operation reads the old output through its `WriteBuffer`; it
must not acquire a separate output `ReadBuffer`.

The matrix-product wrapper rejects this exact condition before buffer
enrollment:

```text
address(output.queue()) == address(input.queue())
```

This catches the obvious same-value and parent/view alias cases that would
self-block with whole-parent queue sharing. Two inputs may share a queue because
read/read access is compatible.

This check is deliberately narrow. It does not prove that separately managed
storage ranges do not overlap, and it does not attempt to recognize arbitrary
dependency cycles. More involved invalid graphs belong to async deadlock
detection and debug task creation traces.

## Backend Selection

Default selector resolution is static: `select_backend_for<Tensors...>(op)`
uses the common storage-policy type and operation value without inspecting a
Tensor value. The immutable selector is therefore chosen before scheduling and
moved into the coroutine. An explicit selector follows the same ownership
rule and may carry immutable operation context.

This does not mean a concrete backend has run or accepted the operation before
scheduling. The coroutine awaits the Tensors, resolves their mdspans, and only
then enters the normal runtime backend walk. Layout- and accessor-dependent
declines therefore remain valid.

A future async block handle may expose a synchronous descriptor outside its
data epoch. Such a wrapper may perform more layout-dependent planning before
scheduling, but that is a different operand contract and must not be assumed
for `Async<Tensor>`.

## Output Semantics

`assign_product` is an overwrite operation:

- If the async output already contains a Tensor, the synchronous operation may
  retain or resize it according to `ensure_shape`.
- If the output is unconstructed and its Tensor type can be constructed from
  its extents type, the coroutine constructs it at the required product shape.
- Otherwise, unconstructed output is an error.

`add_product` is an update operation:

- The output must already be constructed.
- Its existing values participate in the result.
- Its shape must already match; it is never resized.

Apply the same distinction to future overwrite and compound-update operations.

## Exceptions

Passing the output `WriteBuffer` as a coroutine parameter automatically makes
the output epoch an exception sink. Failures from input epochs, shape checks,
backend dispatch, or the kernel are therefore observed when the output is read.

Do not convert a failed backend execution into a kernel decline. Backend
declines happen inside ordinary synchronous dispatch before side effects. Once
the selected synchronous operation fails to produce its result, the async task
fails and propagates that exception to its output.

For a task with multiple outputs, explicitly document and register every output
that must receive failures.

## Reference Shape

The core implementation pattern is:

```cpp
template <class Selector, class OutputTensor, class InputTensor,
          class ConfigAwaiter>
AsyncTask operation_task(Selector const selector,
                         WriteBuffer<OutputTensor> output,
                         ReadBuffer<InputTensor> input,
                         ConfigAwaiter config)
{
  auto output_storage = output.storage();
  auto awaited = co_await all(output_storage, input, config);
  auto& storage = std::get<0>(awaited);
  auto const& value = std::get<1>(awaited);
  auto const& config_value = std::get<2>(awaited);

  prepare_output(storage, value);
  synchronous_operation(selector, *storage, value, config_value);
  co_return;
}

template <class OutputTensor, class InputTensor, class Config>
void operation(Async<OutputTensor>& output,
               Async<InputTensor> const& input,
               Config&& config)
{
  auto selector = select_backend_for<OutputTensor, InputTensor>(operation_op{});
  validate_obvious_aliasing(output, input);
  auto task = operation_task(std::move(selector), output.write(), input.read(),
                             async::read(std::forward<Config>(config)));
  task.debug_name("operation");
  schedule(std::move(task));
}
```

The actual output preparation depends on whether the operation overwrites,
updates, or produces a differently typed result.

## Review Checklist

- All Tensor operands are async, with no accidental mixed overload.
- Coroutine parameters own buffers, immediate-value awaiters, and ordinary
  operation state.
- No coroutine lambda captures state; scheduled lambdas are `static`.
- The output uses one writer rather than a read/write pair.
- Simple output/input queue aliasing is rejected before enrollment when it
  would self-block.
- Static selector resolution happens before scheduling; the runtime backend
  walk happens after awaited mdspans are available.
- Empty-output construction and update-output requirements are explicit.
- Unhandled failures reach every output epoch.
- Tests cover pending-input lifetime, numerical behavior, aliases, output
  construction/update, and failure propagation.

Keep operation-specific wrappers explicit until several implemented operations
show a stable common shape. A generic async-dispatch abstraction is premature
while output construction, mutation, and multi-output exception routing still
differ materially between algorithms.
