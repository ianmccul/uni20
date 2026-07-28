# Async Tensor Kernel Authoring

This guide defines the first supported pattern for lifting synchronous Tensor
operations onto Uni20's async runtime. The implemented references are
`uni20::linalg::gemm`, `uni20::linalg::assign_product`,
`uni20::linalg::add_product`, and the
multi-output `uni20::linalg::eigh` and `uni20::linalg::svd` overloads, plus
full and axis-selective `uni20::sum`, from `<uni20/linalg/async.hpp>`.
See [Tensor Operations](../tensor/operations.md) for the canonical
operation semantics and current synchronous/Async support matrix.

## Layer Boundary

An async Tensor operation is a scheduling wrapper around the same operation-tag
kernel dispatch used by synchronous Tensor operations:

```text
Async<Tensor> handles
  -> static backend-selector resolution
  -> async::read(...) normalization of scalar operands
  -> ReadBuffer / WriteBuffer enrollment
  -> scheduled coroutine
  -> await stored Tensor values and async scalars
  -> operation-specific output preparation and mdspan resolution
  -> co_dispatch_kernel and backend kernel
```

Backends and leaf kernels remain unaware of `Async<T>`, epoch queues, and
schedulers. The async wrapper resolves the storage policy's static selector,
owns any output preparation, resolves stable mdspans, and then enters
`co_dispatch_kernel`.

Do not add async overloads directly to a backend or make `Async<Tensor>` model
`TensorView`.

## Public Operand Contract

The first checkpoint uses a strict all-async interface:

```cpp
void gemm(Async<OutputTensor>& output,
          Alpha&& alpha,
          Async<LhsTensor> const& lhs,
          Async<RhsTensor> const& rhs,
          Beta&& beta);

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

Async Tensor aliases such as `async::conj(tensor)` and
`async::reshape_view(tensor, ...)` are valid input operands. Their
`Async<View>` handles retain the complete parent storage chain and share its
epoch queue.

## Submission and Coroutine Phases

The non-coroutine wrapper performs only work that is valid on the async
handles:

1. Resolve the immutable default selector from the Tensor/storage types, or
   accept an explicit selector by value.
2. Classify the operation as overwrite or update and check exact output/input
   queue aliasing.
3. Normalize each immediate-or-async scalar with `async::read(...)`.
4. Enroll one `WriteBuffer` for the output and one `ReadBuffer` for each
   distinct Tensor input. For an update, the writer supplies the old output
   value; do not also enroll an output reader.
5. Move the selector, awaiters, and ordinary values into a coroutine.
6. Schedule the task.

The coroutine then:

1. Awaits the Tensor buffers and any async scalar buffers together.
2. Resolves references to the stored Tensor values.
3. Prepares an unconstructed overwrite output when supported.
4. Resolves stable mdspans and calls `co_dispatch_kernel` with the operation tag
   and resolved selector.

Use a named free coroutine, or a captureless C++23 `static` lambda. Never pass
references to the `Async<T>` handles into the coroutine. The moved buffers
already retain storage, queue context, and the selected epochs, so the original
handles may be destroyed immediately after submission.

## Coroutine and Task-Factory Names

Use names that distinguish C++ coroutine bodies from ordinary functions that
return task handles:

| Function kind | Naming form | Example |
|---|---|---|
| Coroutine body containing `co_await`, `co_yield`, or `co_return` | `co_<operation>` | `co_gemm_submission(...)` |
| Ordinary function returning a task | `make_<operation>_task` | `make_prefetch_task(...)` |
| Ordinary fallible factory returning an optional or attempt wrapper | `try_make_<operation>_task` | `try_make_kernel_task(...)` |
| Ordinary function that submits work to a scheduler | `schedule_<operation>` | `schedule_async_gemm(...)` |
| Synchronous implementation | `<operation>` | `gemm(...)` |

Reserve the `co_` prefix for actual coroutine bodies. Returning `AsyncTask`,
`CudaTask`, or a wrapper around one does not by itself make an ordinary factory
a coroutine. Conversely, `_async` describes an API or execution model and does
not identify whether the function body uses the C++ coroutine mechanism.

## Access and Aliasing

An ordinary one-output operation has one writer and any number of distinct
input readers. A mutating operation reads the old output through its
`WriteBuffer`; it must not acquire a separate output `ReadBuffer` or pass the
output again as an input. This preserves the synchronous operation's semantic
distinction between overwrite and update.

For example, async `A = f(A, B)` lowers as one writer for `A` and one reader for
`B`. It does not lower as a reader for `A`, a reader for `B`, and a later writer
for `A`. Async `A = f(A)` is a unary update with only the writer for `A`.

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

`gemm` is a fixed-output operation:

- The output must already be constructed with the matrix-product shape.
- `alpha` and `beta` may independently be immediate or async scalar operands.
- The output's existing values participate according to `beta`; `gemm` never
  constructs or resizes the output.

`assign_product` is an overwrite operation:

- If the async output already contains a Tensor, the synchronous operation may
  retain or resize it according to `ensure_shape`.
- If the output is unconstructed and its Tensor type can be constructed from
  its extents type, the coroutine constructs it at the required product shape.
- Otherwise, unconstructed output is an error.

`add_product` is an update operation:

- It forwards to async `gemm` with `beta = 1`.
- The output must already be constructed.
- Its existing values participate in the result.
- Its shape must already match; it is never resized.

Apply the same distinction to future overwrite and compound-update operations.

`assign_transform` and `transform_inplace` apply these rules to a variadic
reader pack. The callable is ordinary immutable task state, not an async
operand. It is moved into the coroutine and then into the synchronous operation
value after all Tensor readers and the output writer are ready. Read-only input
queues may coincide with one another; only output/input queue equality is
rejected. A mutable async alias remains fixed to its parent: while holding the
writer, the transform coroutine copies the bound descriptor locally and
dispatches through that copy rather than requesting value storage or replacing
the descriptor.

`sum` is a reduction overwrite. Runtime axes are ordinary configuration state:
the wrapper normalizes and validates them before scheduling, then moves the
fixed-rank descriptor into the coroutine. The coroutine awaits the input,
derives the required output extents, constructs or resizes an owning output,
and invokes the synchronous reduction. A mutable async alias is a valid
fixed-shape output; the coroutine copies its bound descriptor under the writer
and writes through that copy. Value-returning sums create a fresh independent
result epoch, while `sum_host` creates `Async<Element>` and schedules the host
result instead of blocking the submitting thread.

`eigh` is an allocating multi-output operation. It returns
`SelfAdjointEighResult<Async<Values>, Async<Vectors>>`, not an
`Async<SelfAdjointEighResult<...>>`. The two independent output epochs can be
structured-bound and passed directly to downstream async operations without an
extra extraction coroutine. The preserving overload reads an
`Async<Tensor> const&`; the consuming overload takes `Async<Tensor>&&` and may
transfer the stored owning Tensor's allocation to the eigenvector output.

The exact SVD family follows the same allocating pattern. `singular_values`
returns one `Async<S>`, `svd_left` returns
`SvdLeftResult<Async<U>, Async<S>>`, `svd_right` returns
`SvdRightResult<Async<S>, Async<Vh>>`, and `svd` returns
`SvdResult<Async<U>, Async<S>, Async<Vh>>`. Each returned value has an
independent epoch, while one coroutine computes and commits all outputs of a
multi-output operation.

`truncated_svd` extends the same pattern to four independent outputs:
`TruncatedSvdResult<Async<U>, Async<S>, Async<Vh>,
Async<SvdTruncationInfo<Real>>>`. Rank selection and factor right-sizing happen
inside the scheduled coroutine after the exact SVD is available. Every output
writer is therefore an exception sink; consuming lowering also retains the
input writer through result publication so exact-SVD or policy failures reach
the input and all four outputs.

Preserving overloads read `Async<Tensor> const&`. Consuming overloads take
`Async<OwningTensor>&&`, enroll the input writer, and obtain the matrix through
`take()`. A reduced singular-vector output may adopt the consumed allocation
through LAPACK's `O` job. Full factors and incompatible mappings allocate or
materialize as needed; the rvalue grants permission to reuse storage rather
than promising that every call will do so.

## Consuming Owning Inputs

An operation that may reuse an input allocation must accept only
`Async<Tensor>` values whose stored type models `OwningTensor`. A tensor view,
including a conjugating or slicing alias, never becomes consumable merely
because its descriptor was passed by value.

Enroll a consuming input with `WriteBuffer`, then obtain the value through
`take()` or `take_release()`. Do not read the Tensor and subsequently move from
the referenced object: that would leave the async storage marked constructed
while exposing a moved-from owner to later epochs and aliases.

`take()` moves out and destroys the stored value while retaining the writer.
Use it when the operation will reconstruct that same async value or when the
consumed epoch must remain gated until several outputs commit or receive an
exception. `take_release()` also releases the writer and is appropriate only
when ownership is permanently transferred elsewhere and early release is part
of the operation contract. Existing views and async aliases of a consumed owner
must not be used afterward. Epoch ordering prevents concurrent access, but it
does not extend the lifetime of an object removed from its storage.

## Exceptions

Passing a `WriteBuffer` as a coroutine parameter automatically makes its epoch
an exception sink. Failures from input epochs, shape checks, backend dispatch,
or the kernel are therefore observed when each output is read.

Do not convert a failed backend execution into a kernel decline. Backend
declines happen inside ordinary synchronous dispatch before side effects. Once
the selected synchronous operation fails to produce its result, the async task
fails and propagates that exception to its output.

For a task with multiple outputs, pass every output `WriteBuffer` as a coroutine
parameter and document that all output epochs receive failures. Automatic
argument processing registers every such writer; use
`propagate_exceptions_to(...)` only for an additional sink that is not already a
writer parameter.

A consuming input is also enrolled as a `WriteBuffer`. A multi-output operation
that uses `take()` keeps that writer gate active through output commit and
exception routing. If later work fails, reads from the invalidated input
timeline observe the same exception as the outputs. On success, the consumed
input has no readable value.

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
- The output uses one writer rather than a read/write pair, and an update output
  is not also passed as an input.
- Simple output/input queue aliasing is rejected before enrollment when it
  would self-block.
- Static selector resolution happens before scheduling; the runtime backend
  walk happens after awaited mdspans are available.
- Empty-output construction and update-output requirements are explicit.
- Unhandled failures reach every output epoch.
- Tests cover pending-input lifetime, numerical behavior, aliases, output
  construction/update, and failure propagation.

Keep operation-specific Tensor wrappers explicit because output construction,
mutation, and multi-output exception routing differ materially between
algorithms. Once the wrapper has awaited its values and resolved stable mdspan
operands, use generic `co_dispatch_kernel`. Blocking backends require no
coroutine wrapper; individual backend/operation pairs add `try_make_kernel_task`
only when resource admission or execution must suspend.
