# GEMINI.md

Guidance for Gemini Code Assist reviews in this repository.

`AGENTS.md` is the canonical contributor guide. Read it first and treat this
file as a short list of review-specific reminders for recurring false positives.

## Hard Project Constraints

- Uni20 is a C++23 project. Do not request C++20 compatibility changes.
- Uni20 intentionally uses multidimensional `operator[]` for tensor, matrix, and
  mdspan-style indexing. Do not request `operator()` overloads or replacements
  for compatibility with older C++ standards.
- Do not suggest defining `MDSPAN_USE_PAREN_OPERATOR`.
- Uni20 spells complex scalar types as `uni20::complex<T>` in project code,
  tests, examples, and docs. It is intentionally an alias to `std::complex<T>`.
- Scalar-generic numerical code should use `uni20::numeric_limits<T>` rather
  than `std::numeric_limits<T>` directly.
- Prefer ADL-friendly math wrappers already present in the codebase, such as
  `detail::adl_abs`, `detail::adl_sqrt`, and `detail::abs_squared`, when
  reviewing scalar-generic code.

## Krylov Review Notes

- `tests/krylov/test_dense_linalg_unused.cpp` is intentionally compiled. It
  keeps quarantined dense-wrapper code tested without making those wrappers part
  of the normal Krylov API surface.
- The temporary dense `Matrix` and `DenseMatrix` helpers are prototype
  infrastructure for the Krylov branch. They will eventually be replaced by
  mdspan/rank-2 tensor entry points, so avoid API-expansion suggestions unless
  they are required for current correctness.
- Vendored ARPACK references, where present in historical notes or examples,
  are comparison context. The core Uni20 target is the native matrix-free Krylov
  implementation.

## Review Priorities

Focus on correctness, numerical stability, missing tests, thread/reentrancy
issues, scalar-generic behavior, and documentation drift. Avoid compatibility
suggestions that contradict the C++23 and indexing policies above.
