# Uni20 Review Vocabulary

- **Audience:** repository-aware assistants and reviewers
- **Authority:** non-normative vocabulary
- **Canonical sources:** `AGENTS.md` and the named repository snapshot

This glossary defines only durable review language. It does not describe the
current class hierarchy, API surface, backend inventory, or roadmap.

## Evidence Terms

### Named snapshot

The branch, commit, tag, PR, or diff whose behavior is under discussion. Current
`main` is the default only when the user does not name another snapshot.

### Canonical subsystem documentation

Repository documentation that defines intended subsystem behavior. It must be
read from the named snapshot; an uploaded copy may be stale.

### Current implementation detail

Behavior established by source, tests, and build configuration in the named
snapshot. Its presence in an AI guidance file is not evidence that it still
exists.

### Design direction

A maintainer decision or canonical design statement not yet fully represented
by implementation. Do not infer it merely from an unfinished helper or draft.

## Correctness Terms

### Epoch causality

The dependency order that makes an async operation legal. Scheduler timing or a
particular task order is not a correctness argument.

### Semantic view

A view whose layout or accessor changes the values observed through it. A
pointer-shaped data handle alone does not permit bypassing those semantics.

### Clean decline

A backend's side-effect-free refusal to handle an operation. Once it mutates
state, submits work, or commits output, failure is an operation error and must
not trigger fallback.

### Symmetry metadata

Quantum-number, block-space, local-space, and leg-orientation information that
is part of the mathematical object. It must not be silently erased or replaced
by an implicit dense path.

### Explicit cost

Allocation, materialization, synchronization, host/device transfer, and dense
projection that is visible in the API or documented operation contract rather
than hidden in lowering.
