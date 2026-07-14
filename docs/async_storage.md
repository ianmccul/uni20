# Async Storage, Identity, and Assignment

This note documents how `Async<T>` stores values and distinguishes two meanings
of assignment: replacing an async timeline and writing through an existing one.

For day-to-day usage details, see:

- `docs/async/buffers_and_awaiters.md`
- `docs/async/quick_reference.md`

## 1. Storage Model (shared_storage<T>)

`Async<T>` uses `shared_storage<T>` as the underlying object container.

`shared_storage<T>` has two independent states:

- control block validity (`valid()` / `operator bool`)
- value construction state (`constructed()`)

Important operations:

- `emplace(args...)`: destroy existing object (if any), then placement-construct a new `T`
- `get()`: pointer to constructed object, or `nullptr` if unconstructed
- `destroy()`: destroy current object but keep the control block
- `take()`: move value out, then destroy
- `reset()`: release ownership of the control block

`Async<T>()` starts with valid storage but unconstructed value.

### Alias storage

`make_shared_storage_alias<T>(owner, args...)` constructs a local `T` value
whose control block retains one strong reference to `owner`. This is a
hierarchical reference count:

- copies and buffer handles increment the alias control block's local count
- the alias control block contributes exactly one reference to its owner
- destroying the final alias handle destroys the local value, then releases
  the owner reference

An alias reports `constructed()` only when both its local value and its owner
chain are constructed. This allows a tensor-view descriptor to be created over
the stable address of an unconstructed `Async<Tensor>` while preventing access
until the parent writer constructs the tensor.

`make_async_alias<T>(parent, args...)` adds causal ordering to this lifetime
relationship. The resulting `Async<T>` shares the parent's exact `EpochQueue`;
using a fresh queue for aliased bytes is incorrect. Alias owners may themselves
be aliases, so composed views form a recursive owner chain while retaining one
common timeline.

Alias descriptor types declare `async_alias_tag`. Their `Async<T>` copies are
structural handle copies that retain the same storage and queue. Ordinary
`Async<T>` copies remain scheduled value copies.

## 2. Async Value Kinds

`async_value_kind_of<T>` classifies what an `Async<T>` represents:

| Kind | Construction and copy behavior |
|---|---|
| `async_value_kind::value` | root construction is allowed; copies create an independent value timeline and schedule a value copy |
| `async_value_kind::shared_alias` | root construction is disabled; copies share descriptor storage, lifetime ownership, and the exact epoch queue |

Ordinary types default to `value`. Durable alias descriptors declare
`async_alias_tag`, which selects `shared_alias`; external descriptor types may
specialize `async_value_kind_of<T>` instead.

This policy belongs at the `Async<T>` wrapper level. It cannot be inferred from
the stored type's assignment expression because it controls storage ownership
and causal identity, not assignment to a constructed `T`.

## 3. Assignment Kinds

The motivating distinction is between an owning async value and a mutable view
of another async value.

```cpp
Async<Tensor> x = make_first_tensor();
consume(x);

x = make_second_tensor();
consume(x);
```

The second assignment should detach `x` from its first storage and
`EpochQueue`, then bind it to fresh storage with a fresh queue. Operations that
already retained buffers from the first `x` continue on the old branch and may
run independently of operations using the second `x`.

The planned async reshape API has different requirements:

```cpp
Async<Tensor> x = make_tensor();
auto y = async::reshape_view(x, rows, columns); // planned: returns Async<View>
Async<Tensor> values = make_replacement_values();

y = values; // heterogeneous Async<Tensor>: write through y into x
```

Here `y` is an alias of `x`. Assignment must keep its descriptor storage,
lifetime owner, and exact shared `EpochQueue`; replacing any of them would stop
`y` from describing the same region of `x`.

`async_assignment_kind_of<T>` records this distinction:

| Kind | Direct immediate or heterogeneous assignment to `Async<T>` |
|---|---|
| `async_assignment_kind::rebind` | Detach the handle and construct a fresh value with fresh storage and a fresh `EpochQueue`. |
| `async_assignment_kind::write_through` | Keep the destination storage, owner chain, and queue; schedule assignment through the existing descriptor. |

Ordinary values default to `rebind`. A mutable reference-like descriptor opts
into `write_through` by declaring `async_write_through_tag`, or by specializing
`async_assignment_kind_of<T>`. Write-through payloads must also be
`shared_alias` values so their lifetime owner and queue are explicit. Read-only
aliases do not opt in.

These assignment kinds are independent of `async_value_kind_of<T>`:

| Example payload | Value kind | Assignment kind |
|---|---|---|
| owning `Tensor` | `value` | `rebind` |
| mutable reshape or slice descriptor | `shared_alias` | `write_through` |
| const or conjugating descriptor | `shared_alias` | `rebind`, with no heterogeneous value-assignment overload |

### Exact Async assignment

Exact `Async<T>` copy and move assignment uses the dedicated copy/move
overloads, not `async_assignment_kind_of<T>`:

```cpp
auto y = async::reshape_view(x, rows, columns);
y = async::reshape_view(z, rows, columns);
```

For a shared alias, this replaces the complete alias handle: descriptor
storage, lifetime owner, and queue move together. Buffers already obtained from
the old `y` retain its old descriptor and queue. To write through from another
view of the same type, use `async_assign(y, rhs)` or an explicit tensor
operation such as `copy(y, rhs)`.

### Explicit assignment into the current timeline

`async_assign(destination, source)` always enrolls a writer in the
destination's current queue. For an ordinary value it propagates a value into
that timeline. For a `write_through` descriptor it invokes the descriptor's
assignment operation without replacing its owner or queue.

A type declaring `async_write_through_tag` promises that every assignment
accepted through this path mutates the referenced value rather than retargeting
the descriptor. In particular, its same-type assignment must not silently copy
only a pointer or mdspan descriptor.

## 4. Write-Proxy Assignment

Write-proxy assignment follows the stored type's ordinary assignment semantics:

```cpp
co_await writer = rhs;
```

- when storage is empty, construct `T` from `rhs`
- when a `T` already exists, evaluate `stored_value = rhs`

The expression is available only when `rhs` can both construct `T` and make
`stored_value = rhs` a valid expression. This guarantees that it is valid in
either runtime construction state.

Consequences:

- a `BasicTensor` uses its ordinary assignment expression once constructed
- an mdspan-like descriptor uses its normal descriptor assignment
- a reference-like type follows whatever assignment its own API defines
- the async assignment-kind trait does not change this low-level proxy rule

## 5. Explicit Construction and Assignment

Use the operation that states the intended precondition:

```cpp
auto access = co_await writer;

access.emplace(args...); // construct or explicitly reconstruct T
access.get() = rhs;      // assign an object known to be constructed
```

Construct-only types use `emplace(...)`. Assign-only proxy types must already
exist and use `get() = rhs`. Proxy `operator=` is the convenience path only
when both initialization and later assignment are valid.

`emplace(...)` destroys and reconstructs `T` inside the same storage control
block and the same `EpochQueue`. It is not an async rebind and should not be
used to implement ordinary `Async<T>::operator=`.

## 6. Tensor Views and Aliases

Write-through assignment still delegates the actual element operation to the
stored mutable view type. Shape handling and backend dispatch belong in that
tensor-level operation; the async trait only decides whether the destination
timeline is retained.

A shared async alias has a descriptor tied to a specific lifetime owner and
epoch queue. Never reconstruct only that descriptor to point elsewhere: its
owner chain and queue would then describe different storage. Rebind by assigning
another complete `Async<Alias>`, so descriptor, owner, and queue move together.
