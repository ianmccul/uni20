# Async Storage, Value Kinds, and Assignment

This note documents how `Async<T>` stores values and how write-proxy assignment is interpreted.

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

## 3. Write-Proxy Assignment

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
- async storage does not override those semantics with a type trait

## 4. Explicit Construction and Assignment

Use the operation that states the intended precondition:

```cpp
auto access = co_await writer;

access.emplace(args...); // construct or explicitly reconstruct T
access.rebind(args...);  // named synonym for explicit reconstruction
access.get() = rhs;      // assign an object known to be constructed
```

Construct-only types use `emplace(...)`. Assign-only proxy types must already
exist and use `get() = rhs`. Proxy `operator=` is the convenience path only
when both initialization and later assignment are valid.

## 5. Tensor Views and Aliases

Assignment to a tensor or view remains defined by that type and by named tensor
operations. Element copying that requires shape handling or backend dispatch
should use `copy(...)`, not hidden async assignment policy.

A shared async alias has a descriptor tied to a specific lifetime owner and
epoch queue. Reconstructing that descriptor to point elsewhere would not update
its owner chain or queue and must not be used as a retargeting mechanism.
Construct a new alias when the referenced region changes.
