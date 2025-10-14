# Uni20 Doxygen Documentation Guidelines

This document defines the conventions for writing and maintaining **Doxygen-style documentation** across the Uni20 C++ library.

These rules ensure that:
- Documentation is uniform and machine-parsable.  
- Preconditions and invariants are explicitly stated.  
- Automation tools (e.g., Codex, Doxygen XML, Sphinx+Breathe) can reliably process it.  
- Mathematical and concurrency semantics are preserved.

---

## 1. Comment Syntax

- Use `///` for all public API documentation.  
  Example:
  ```cpp
  /// \brief Adds two tensors asynchronously.
````

* Use regular `//` comments for **implementation details** not intended for Doxygen.
* Do **not** mix `/** ... */` and `///` styles in the same component.
* Place Doxygen comments **immediately above** the entity (function, struct, class) they describe.
* Use `/** ... */` **only** for `\defgroup` or file-level documentation.
* All ordinary macros, functions, and classes must use `///` Doxygen comments.

---

## 2. Documentation Structure

Each documented symbol (function, class, struct, typedef, concept, etc.) should follow this order:

1. `\brief` — One-sentence summary.
2. (blank line)
3. Optional `\details` — Extended description.
4. Optional clauses in the following order:

   * `\pre` — Preconditions
   * `\post` — Postconditions
   * `\throws` — Exception behavior
   * `\note` — Implementation, lifetime, or concurrency remarks
   * `\warning` — Hazard or race condition
5. Parameter and return tags:

   * `\tparam` — Template parameters
   * `\param` — Function parameters
   * `\return` — Return values
6. Grouping tags:

   * `\ingroup` — Link to a module
   * `\defgroup` / `\addtogroup` — For defining module groups

---

## 3. Parameter and Template Rules

* Use `\tparam` for every template parameter.

  ```cpp
  /// \tparam T Tensor element type (must satisfy TensorElement concept).
  ```

* Use `\param` for every parameter, even when the name is self-explanatory.
  This ensures automatic tooling works correctly.

  ```cpp
  /// \param a First operand tensor.
  /// \param b Second operand tensor.
  ```

* Use `\return` for **every non-void** function or method:

  ```cpp
  /// \return A tensor representing the element-wise sum.
  ```

---

## 4. Preconditions, Postconditions, and Exceptions

State explicit logical and lifetime guarantees:

```cpp
/// \pre Both tensors must have identical extents and layouts.
/// \post The result becomes available once both operands resolve.
/// \throws std::invalid_argument if tensor extents mismatch.
```

* Preconditions correspond to requirements checked via `PRECONDITION()` or `CHECK()`.
* Postconditions define observable guarantees after return.
* Use `\throws` for all thrown exceptions, even internal ones.

---

## 5. Notes, Warnings, and Concurrency

Concurrency and lifetime invariants must be explicit:

```cpp
/// \note Non-blocking; coroutine resumes once both dependencies complete.
/// \warning Unsafe if tensors alias overlapping memory regions.
```

Use:

* `\note` for descriptive context, ownership, and evaluation behavior.
* `\warning` for situations that can cause deadlocks, data races, or undefined behavior.

---

## 6. Grouping and Module Structure

Modules organize documentation hierarchically.

Example:

```cpp
/// \defgroup async Asynchronous Primitives
/// \ingroup core
///
/// Coroutines, awaiters, and schedulers for deferred tensor evaluation.
///
/// \{
...
/// \}
```

* Use `\defgroup` once per module (in a header or overview file).
* Use `\ingroup` on all API members that belong to that module.

---

## 7. Formatting Rules

* Always put a **blank line after `\brief`**.
* End all sentences and parameter descriptions with a period.
* Keep `\brief` to **one sentence** — the first period ends the summary.
* Prefer **imperative mood** (“Constructs…”, “Returns…”, “Throws…”).
* Avoid Unicode; prefer LaTeX for mathematical symbols:

  ```cpp
  /// Computes \( C = A B \) using Einstein summation.
  ```
* Avoid embedding Markdown inside Doxygen; use Doxygen’s native syntax.

---

## 8. Example Template

Here’s a canonical example illustrating the complete structure:

```cpp
/// \brief Constructs an asynchronous tensor addition task.
///
/// Performs element-wise addition of tensors `a` and `b`.
///
/// \details
/// The returned Async<Tensor> defers evaluation until awaited.
/// Useful for building lazy tensor computation graphs.
///
/// \tparam T Tensor element type.
/// \param a First operand tensor.
/// \param b Second operand tensor.
/// \return An Async<Tensor<T>> representing the sum.
/// \pre Both tensors must have identical extents.
/// \post The resulting tensor becomes available when both operands resolve.
/// \throws std::invalid_argument if tensor extents mismatch.
/// \note Non-blocking; evaluation deferred until awaited.
/// \warning Undefined behavior if tensors alias the same buffer.
/// \ingroup async
AsyncTensor operator+(AsyncTensor const& a, AsyncTensor const& b);
```

---

## 9. Special Considerations for Uni20

* **Coroutines:** Document lifetime explicitly. Use `\note` to indicate when parameters are passed by value to prevent dangling references.
* **Tensor math:** Prefer LaTeX for contractions, symmetries, and index notation.
* **Async APIs:** Always include `\note` on evaluation order and scheduler behavior.

---

## 10. Validation

Run:

```bash
doxygen Doxyfile
```

to verify formatting.
Warnings such as:

```
warning: argument 'b' of command @param is not found in the argument list
```

must be fixed immediately — no undocumented parameters are allowed.

---

## 11. Summary Checklist

| Category                | Requirement                    |
| ----------------------- | ------------------------------ |
| Comment style           | `///` only                     |
| Parameter tags          | Always present                 |
| Return tags             | Always present                 |
| Template tags           | Always present                 |
| Pre/Post/Throws         | Explicit when relevant         |
| Grouping                | Use `\ingroup` and `\defgroup` |
| Unicode                 | Avoid; prefer LaTeX            |
| Concurrency notes       | Required for async code        |
| Brief format            | One sentence + blank line      |
| Implementation comments | Use `//`, not Doxygen          |

