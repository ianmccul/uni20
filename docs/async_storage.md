# Async Storage and Assignment Semantics

This note documents how `Async<T>` stores values and how write-proxy assignment is interpreted.

For day-to-day usage details, see:

- `docs/async/buffers_and_awaiters.md`
- `docs/async/quick_reference.md`

## 1. Storage Model (`shared_storage<T>`)

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

## 2. Why Assignment Semantics Need a Trait

Not all `T` should interpret `co_await writer = rhs` the same way:

- value-like types usually want rebind/reconstruct semantics
- reference-proxy types may want write-through semantics

To make this explicit, async write proxies use:

- `uni20::async::assignment_semantics_of<T>`
- `uni20::async::assignment_semantics_v<T>`

Defined in:

- `src/uni20/async/assignment_semantics.hpp`

## 3. Available Semantics

`assignment_semantics::rebind` (default):

- `co_await writer = rhs` takes the rebind path (`emplace(...)`)
- first write into default `Async<T>` is naturally supported

`assignment_semantics::write_through`:

- `co_await writer = rhs` assigns through the existing object (`writer.data() = rhs`)
- requires already-constructed storage
- if storage is unconstructed, debug builds trip a precondition check

Write proxies always expose explicit structural replacement:

- `proxy.rebind(args...)`
- `proxy.emplace(args...)`

So even for write-through types, rebinding remains available as an intentional operation.

## 4. Specializing the Trait

Default behavior:

```cpp
template <typename T>
struct assignment_semantics_of
    : std::integral_constant<assignment_semantics, assignment_semantics::rebind> {};
```

Opt in to write-through for a proxy/reference-like type:

```cpp
namespace uni20::async {
template <>
struct assignment_semantics_of<MyProxyType>
    : std::integral_constant<assignment_semantics, assignment_semantics::write_through> {};
}
```

## 5. Practical Guidance

- Use `rebind` for ordinary value/handle types where replacement is expected.
- Use `write_through` only when `operator=` should mutate the referenced target.
- Keep rebinding explicit via `rebind(...)` to avoid ambiguous assignment behavior.
- Prefer type-driven semantics (trait specialization) over runtime branching.

## 6. Relation to `TensorView` and mdspan-like Types

`std::span` and `std::mdspan` generally have value/handle assignment semantics (rebind-style).
Reference-proxy types (such as a write-through tensor view abstraction) should specialize
`assignment_semantics_of<T>` to `write_through` when assignment is intended to mutate pointee data.

The trait intentionally describes async write assignment behavior only; it does not define ownership.
