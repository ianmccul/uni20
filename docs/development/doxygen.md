# Uni20 Doxygen Documentation Guidelines

This document defines the conventions for Doxygen-style documentation across
the Uni20 C++ library. The goal is useful generated documentation and clear
developer-facing comments, not mechanical comment coverage.

## 1. Comment Syntax

- Prefer `///` for ordinary declaration documentation.
- Use `/** ... */` for file-level overviews, module/group definitions, and long
  multi-paragraph or LaTeX-heavy blocks.
- Use regular `//` comments for implementation details that should not appear
  in generated API docs.
- Place Doxygen comments immediately above the declaration they describe.
- Preserve existing comment form unless changing it clearly improves the
  surrounding code.

When a description line wraps, align continuation text under the previous text:

```cpp
/// \details Real numbers are unchanged by conjugation, so the value is returned verbatim.
///          The overload is constexpr, enabling compile-time evaluation for literals.
```

## 2. What To Document

Document new or substantially changed declarations when they are part of a
developer-facing API, encode a cross-module contract, or carry important
semantic, lifetime, ownership, concurrency, or mathematical invariants.

Internal helpers do not need Doxygen by default. Add Doxygen only when generated
docs would help, or when the helper is a conceptual interface in practice. Use
ordinary `//` comments for local implementation notes.

Do not add placeholder documentation such as `[TODO] Document this function`.
If no useful comment can be written from the local context, leave the code
undocumented and keep the task focused.

## 3. Tags

New or substantially edited public API Doxygen blocks should start with a
concise `\brief`. Use additional tags when they add useful information:

- `\details` for multi-sentence behavior or mathematical context.
- `\pre` and `\post` for observable requirements and guarantees.
- `\throws` for meaningful exception behavior.
- `\note` for ownership, lifetime, evaluation, or implementation context.
- `\warning` for hazards such as aliasing, races, invalid lifetimes, or
  undefined behavior.
- `\tparam`, `\param`, and `\return` when a template parameter, parameter, or
  return value has non-obvious semantics or constraints.

Avoid tautological tags:

```cpp
/// \param value The value.
/// \return The result.
```

When several tags appear, use this order:

```text
\brief, \details, \pre, \post, \throws, \note, \warning,
\tparam, \param, \return, \ingroup
```

Only use `\return` for callable entities that actually return a value. Do not
add it to classes, structs, aliases, concepts, variables, constructors, or
destructors.

## 4. Grouping

Define groups at module or file boundaries:

```cpp
/**
 * \defgroup async Asynchronous Primitives
 * \brief Coroutines, awaiters, and schedulers for deferred tensor evaluation.
 */
```

Use `\ingroup` sparingly. Add it where it improves generated navigation, not on
every declaration. Do not repeat `\ingroup` on members, nested types, or
declarations already covered by surrounding file or type documentation.

Namespaces such as `uni20::internal`, `detail`, and directories named
`internal` or `detail` already signal internal scope. Use `\internal ...
\endinternal` only when generated documentation needs an explicit hidden or
internal section.

## 5. Examples

A small declaration may need only a brief:

```cpp
/// \brief Returns the scalar conjugate, preserving real values unchanged.
template <typename Scalar>
constexpr auto conjugate(Scalar const& value);
```

A cross-module or lifetime-sensitive API should be more explicit:

```cpp
/// \brief Schedules an asynchronous tensor assignment.
/// \details The returned task completes after all source dependencies are
///          readable and the destination epoch has exclusive write access.
/// \pre The source and destination extents match.
/// \warning Source and destination buffers must not partially overlap.
/// \param scheduler Scheduler that owns the task lifetime.
/// \param destination Destination write buffer.
/// \param source Source read buffer.
/// \return Task representing the scheduled assignment.
AsyncTask assign(TbbScheduler& scheduler, WriteBuffer<Tensor>& destination,
                 ReadBuffer<Tensor> source);
```

## 6. Markdown Pages

Keep Markdown headings as plain text. Do not use inline-code backticks in
headings: Doxygen serializes those code spans as escaped `<tt>...</tt>` text in
generated HTML headings and page anchors. Use plain names such as
`DebugScheduler` in headings, and keep inline code styling for body text,
tables, and examples.

## 7. Validation

Validate the Markdown hierarchy, local links, subsystem indexes, and explicit
repository-root references before generating the API site:

```bash
scripts/check-docs.py
```

For documentation builds, use an out-of-source build directory:

```bash
cmake -S . -B ./build_codex/docs -DUNI20_DOCS_WEB=ON
cmake --build ./build_codex/docs --target doc
```

Fix real Doxygen warnings such as a `\param` whose name is not in the function
signature. Do not add broad documentation churn simply to satisfy a coverage
metric.

## 8. Checklist

| Category | Guidance |
| --- | --- |
| Comment style | Prefer `///`; use block comments for file/module overviews |
| Briefs | Use concise `\brief` on new or edited public API docs |
| Parameter tags | Add when semantics or constraints are useful |
| Return tags | Add for meaningful non-void return values |
| Grouping | Use `\ingroup` sparingly |
| Internal helpers | Document only when it adds useful context |
| Implementation comments | Use `//`, not Doxygen |
| Churn | Avoid placeholder comments and mechanical tag insertion |
