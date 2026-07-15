# Async Storage, Identity, and Assignment

This note documents how `Async<T>` stores values and distinguishes independent
value rebinding, mutable-alias write-through, and read-only aliases for which
assignment is forbidden.

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
relationship. `make_async_alias_from_parent<T>(parent, args...)` additionally
passes the stable address reserved for the parent value to a descriptor that
must defer dereferencing it. The resulting `Async<T>` shares the parent's exact
`EpochQueue`; using a fresh queue for aliased bytes is incorrect. Alias owners
may themselves be aliases, so composed views form a recursive owner chain while
retaining one common timeline.

Alias descriptor types declare `async_alias_tag`. Copy construction of their
`Async<T>` handles retains the same storage and queue. Copy construction of an
ordinary `Async<T>` remains a scheduled value copy.

## 2. Independent Values and Aliases

`is_async_alias<T>` classifies what an `Async<T>` represents. It is false for
ordinary independent values. Durable alias descriptors declare
`async_alias_tag`, or specialize `is_async_alias<T>`, to make
`is_async_alias_v<T>` true.

| Payload | Root construction | Copy construction |
|---|---|---|
| independent value | allowed | schedule a value copy into fresh storage and a fresh queue |
| async alias | only through an async alias factory | share descriptor storage, lifetime ownership, and the exact queue |

This classification belongs at the `Async<T>` wrapper level. It cannot be
inferred from `T::operator=` because it controls storage ownership and causal
identity, not assignment to a constructed `T`.

## 3. Assignment Semantics

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
run independently of operations using the second `x`. This is analogous to
register renaming: the C++ name is reused for a logically new value without
adding a false causal dependency on the old value.

The async reshape API has different requirements:

```cpp
Async<Tensor> x = make_tensor();
auto y = async::reshape_view(x, rows, columns); // returns Async<View>
Async<Tensor> values = make_replacement_values();

y = values; // heterogeneous Async<Tensor>: write through y into x
```

Here `y` is an alias of `x`. Assignment must keep its descriptor storage,
lifetime owner, and exact shared `EpochQueue`; replacing any of them would stop
`y` from describing the same region of `x`.

There is no separate assignment-kind trait. Assignment follows the payload
classification and the operations that the payload actually supports:

| Example payload | Direct `Async<T>` assignment |
|---|---|
| owning `Tensor` | detach and initialize a fresh value timeline |
| mutable reshape or slice descriptor | retain alias identity and invoke an ADL-visible `assign_through(target, source)` |
| const or conjugating descriptor | ill-formed because no matching `assign_through` exists |

Tensor aliases obtain write-through capability from `MutableTensorView`.
`uni20::assign_through` delegates to the ordinary backend-dispatched tensor
`copy` operation. A read-only tensor descriptor resolves a const-element mdspan,
does not satisfy `MutableTensorView`, and therefore has no tensor
`assign_through` overload.

### Exact Async sources

Exact `Async<T>` copy and move assignment follows the same capability rules as
heterogeneous assignment:

```cpp
auto y = async::reshape_view(x, rows, columns);
y = async::reshape_view(z, rows, columns);

auto read_only = async::conj(x);
read_only = async::conj(z); // ill-formed: the left-hand proxy is read-only
```

The first assignment writes values through `y` into `x`; it does not retarget
`y` toward `z`. The second assignment is rejected even though both operands
happen to have the same proxy type. Copy and move construction remain valid for
read-only aliases because constructing another handle does not use a proxy as
an assignment destination. Every `Async<T>` can create a `WriteBuffer<T>` to
represent exclusive epoch access, but its capability-aware proxy exposes no
write-through operation when the alias is read-only.

### Explicit assignment into the current timeline

`async_assign(destination, source)` enrolls assignment in the destination's
current queue. For an independent value, the destination may already be
constructed, so both construction and assignment must be valid. For an alias,
the operation invokes `assign_through` without replacing the descriptor, owner,
or queue.

## 4. Capability-Aware Write Proxies

`WriteBuffer<T>` always means exclusive access to one epoch. It is not split
into separate buffer types for construction, mutation, movement, or alias
write-through. The proxy returned by `co_await` exposes only operations valid
for the payload category:

| Proxy operation | Independent value | Async alias |
|---|---|---|
| `get()` and `operator->` | mutable `T` access | read-only descriptor access |
| proxy assignment | initialize-or-assign `T` | invoke matching `assign_through` |
| `emplace(...)` | available when constructible | unavailable |
| `+=`, `-=`, `*=`, `/=` | available when the expression is valid | unavailable |
| `take()`, `take_release()`, storage access | available when mechanically valid | unavailable |
| `release()` | available | available |

For an independent value, proxy assignment has initialize-or-assign semantics:

```cpp
co_await writer = rhs;
```

- when storage is empty, construct `T` from `rhs`
- when a `T` already exists, evaluate `stored_value = rhs`

The expression is available only when `rhs` can both construct `T` and make
`stored_value = rhs` a valid expression. This guarantees that it is valid in
either runtime construction state.

For an alias, the proxy never returns a mutable descriptor reference. This
prevents ordinary descriptor assignment, `emplace`, or movement from retargeting
the descriptor away from the lifetime owner and queue that make it valid.

## 5. Payload Requirements

`Async<T>` does not impose copyability, movability, assignability, or default
construction globally. Each operation participates only when its own mechanics
are valid:

| Operation | Requirement on an independent `T` |
|---|---|
| instantiate `Async<T>` | complete non-array object with a non-throwing destructor |
| default `Async<T>()` | none beyond the row above; storage starts unconstructed |
| direct value construction | constructible from the supplied arguments |
| `read()` / `get_wait()` | none; they return read-only access |
| move construction of `Async<T>` | none; the handle moves, not `T` |
| move assignment of `Async<T>` | none; the independent handle is rebound |
| copy construction or copy assignment of `Async<T>` | constructible from `T const&`; assignment initializes fresh storage and does not require `T::operator=` |
| direct `Async<T>::operator=(U)` | constructible from `U`; creates a fresh timeline |
| `async_assign(destination, source)` | constructible and assignable in the existing timeline |
| `emplace(args...)` | constructible from the supplied arguments |
| `take()` / `move_from_wait()` | constructible from `T&&` |

For an async alias, copy construction is structural and needs no payload copy.
Assignment participates only when ADL finds `assign_through(T&, Source)`.
Move construction transfers the alias handle, while move assignment remains
write-through so an already-bound alias is never retargeted.

This means a copy-constructible but non-assignable payload can still be copied
or copy-assigned as an `Async<T>` because those outer-handle operations create
a fresh timeline. Explicit assignment into an existing timeline remains
unavailable for that payload.

## 6. Explicit Value Construction

Use the operation that states the intended precondition:

```cpp
auto access = co_await writer;

access.emplace(args...); // construct or explicitly reconstruct T
access.get() = rhs;      // assign an independent object known to be constructed
```

Construct-only values use `emplace(...)`. Assign-only values must already exist
and use `get() = rhs`. Proxy `operator=` is the convenience path only when both
initialization and later assignment are valid. These descriptor-replacing
operations are deliberately unavailable for aliases.

`emplace(...)` destroys and reconstructs `T` inside the same storage control
block and the same `EpochQueue`. It is not an async rebind and should not be
used to implement ordinary `Async<T>::operator=`.

## 7. Tensor Views and Aliases

Write-through assignment still delegates the actual element operation to the
stored mutable view type. Shape handling and backend dispatch belong in that
tensor-level operation; `is_async_alias_v<T>` preserves identity while the
presence of `assign_through` determines write capability.

A shared async alias has a descriptor tied to a specific lifetime owner and
epoch queue. Never reconstruct only that descriptor to point elsewhere: its
owner chain and queue would then describe different storage. Construct a new
alias handle when a different region is required; assignment never retargets
an existing async alias.
