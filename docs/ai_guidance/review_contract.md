# Repository Review Contract

- **Audience:** remote assistants, coding agents, and reviewers
- **Authority:** non-normative operating contract
- **Canonical sources:** maintainer decisions, `AGENTS.md`, canonical subsystem
  documentation, source, tests, and build configuration in the named snapshot

This file contains durable Uni20 review rules. It deliberately does not record
which operations, backends, schedulers, bindings, or algorithms are currently
implemented.

## Establish the Evidence Snapshot

Before making an exact claim about Uni20:

1. Identify the branch, commit, tag, PR, or diff requested by the user.
2. If none is named, inspect current `main` and record the commit.
3. Read the relevant canonical docs, source, focused tests, and build files from
   that same snapshot.
4. Cite the paths that support the conclusion and distinguish observed behavior
   from intended or planned behavior.

Do not combine source from one branch with documentation or tests from another
without identifying the mismatch. Do not infer current support or absence from
uploaded knowledge, prior conversations, repository names, or old file paths.

If direct repository inspection is unavailable, state that current behavior
cannot be verified and request the exact source, diff, or commit needed.

## Authority and Conflicts

Maintainer decisions and canonical subsystem docs define intended behavior.
Tests encode selected parts of that contract. Source and build configuration
show the current implementation and may contain the defect under review.

There is no mechanical winner when these disagree. Report the conflict, show
the evidence, and ask the maintainer to resolve genuinely ambiguous semantics.
AI guidance never overrides the named repository snapshot.

## Stable Repository Conventions

- Uni20 uses C++23. The supported compiler floors are GCC 13 and upstream
  Clang 19.
- Uni20 is in active design. Prefer the clearest correct API over compatibility
  shims for unfinished in-tree interfaces unless compatibility is explicitly
  requested.
- Use the Kokkos reference mdspan implementation in namespace `stdex::`, with
  square-bracket multidimensional indexing. Do not enable
  `MDSPAN_USE_PAREN_OPERATOR`.
- Spell project complex scalars as `uni20::complex<T>` and use
  `uni20::numeric_limits<T>` in scalar-generic Uni20 algorithms.
- Async coroutine lambdas must be captureless and `static`; pass state as
  parameters so it is stored in the coroutine frame.
- Async paths use Uni20 schedulers and causal primitives rather than raw
  `std::thread` or ad hoc synchronization.
- API changes require focused tests and updates to the relevant canonical
  subsystem documentation.

Consult `AGENTS.md` in the named snapshot for the complete, current contributor
rules. The list above is an orientation aid, not a replacement.

## Durable Correctness Invariants

### Async causality

Legality comes from epoch and dependency causality, not observed scheduler
timing. A backend or scheduler implementation must not rely on a fortunate task
execution order.

### Mdspan accessors

The accessor defines value semantics. A pointer-shaped `data_handle_type` does
not by itself prove that raw memory may be read, written, or passed to a provider.
Custom accessors require explicit lowering, materialization, or a generic path.

### Backend fallback

Fallback is allowed only after a clean decline. A replaceable-output operation
may authorize provisional output construction, resizing, or replacement that a
later backend can reuse or replace. Inputs and fixed/update outputs must remain
unchanged, and once a backend writes result elements, submits work, commits a
completed result, or receives a terminal provider failure, the operation may
not continue with another backend.

### Symmetry preservation

Symmetry metadata is mathematical state. A symmetry-typed operation must not
silently flatten to a dense path or discard quantum-number, block-space,
local-space, or leg-orientation metadata.

### Explicit operational costs

Review hidden allocation, materialization, synchronization, transfer, and dense
projection as semantic and performance concerns. Do not assume they are harmless
implementation details.

### Reachable counterexamples

A correctness finding must show a concrete construction and execution path
using inputs permitted by the documented API and subsystem invariants. The
numeric limits of an underlying carrier type are not automatically valid tensor
dimensions, allocation or workspace sizes, strides, or model quantum numbers.

For an overflow finding, identify the exact arithmetic, show how valid operands
reach it, and explain why an earlier constructor, codec, storage, provider, or
allocation check does not reject them. Counterexamples requiring physically
unconstructible storage, model-invalid metadata, or violation of an established
precondition are contract questions rather than correctness defects. Do not
request hot-path guards solely for such states.

This rule does not excuse overflow in parsers, serialized or otherwise
untrusted metadata, provider integer lowering, allocation planning, or
arithmetic reached by ordinary valid inputs.

## Review Method

For a code or design review:

1. Lead with correctness defects, invariant violations, behavioral regressions,
   and missing evidence.
2. Verify exact type, layout, scalar, backend, async, and failure-path support
   from source and tests; do not infer uniform coverage.
3. Separate intended contract, current implementation, proposed direction, and
   open question when that distinction affects the conclusion.
4. Check neighboring paths that share the changed assumption or helper.
5. State which checks were run, which were unavailable, and the residual risk.

Useful starting points are `AGENTS.md`, `docs/`, `src/uni20/`, `tests/`, and the
relevant `CMakeLists.txt`. Follow the subsystem indexes in the snapshot rather
than relying on a fixed AI-maintained file list.
