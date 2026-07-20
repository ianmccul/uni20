# Agent-Assisted Development

## Purpose

Uni20 uses coding agents as engineering collaborators. The objective is to
increase implementation and review capacity while keeping mathematical
semantics, architecture, and project direction under maintainer control.

This document describes the workflow that Uni20 uses now. It is deliberately
lighter than an autonomous issue-to-pull-request system.

## Authority and Evidence

Different repository artifacts have different roles:

| Artifact | Role |
|---|---|
| Maintainer-approved decision or canonical subsystem document | Defines intended semantics and architecture |
| Tests | Encode and enforce selected parts of the intended contract |
| Source code | Implements current behavior; it may contain the defect under investigation |
| `AGENTS.md` | Defines repository-wide contributor and safety rules |
| `.codex/skills/` | Provides reusable tool-specific workflows |
| `docs/ai_guidance/` | Provides non-normative repository-review conventions and durable invariants |

There is no automatic conflict rule between code, tests, and canonical
documentation. A conflict is evidence that the contract or implementation needs
review. The maintainer resolves the intended behavior.

## Operating Modes

### Interactive development

This is the default mode. The maintainer and agent inspect, discuss, implement,
test, and refine a change in the current working session.

- The user request and subsequent decisions form the working contract.
- The agent may propose alternatives and challenge weak assumptions.
- Once the maintainer chooses a direction, implementation can proceed without a
  separate GitHub issue or design-acceptance ceremony.
- The agent works with existing uncommitted changes and never resets, stashes,
  or overwrites unrelated work.
- Commits and pushes occur when requested by the maintainer.

When semantics remain unresolved, the agent should stop editing that part,
explain the alternatives, and obtain a decision. This is a local design pause,
not a requirement to abandon the entire task.

### Review or asynchronous handoff

A branch or pull request is useful when a change needs review outside the active
session, comes from an external contributor, or is large enough to benefit from
a durable review surface.

- State the intended contract and important design decisions.
- Record verification commands, unavailable checks, and residual risk.
- Use an independent reviewer for substantial numerical, async, AD, symmetry,
  backend, or architectural changes.
- A draft pull request is appropriate when implementation or validation remains
  incomplete, but it is not mandatory for ordinary agent-authored work.

### Unattended automation

Uni20 does not currently run an unattended agent that selects issues and opens
pull requests. If one is introduced, it is a separate execution mode and must:

- receive an explicit, bounded task with established behavior;
- use an isolated worktree or ephemeral checkout;
- have least-privilege credentials and no merge authority;
- create a draft pull request rather than modify the maintainer checkout;
- stop on unresolved numerical, ownership, symmetry, or architecture questions;
- preserve deterministic test evidence and explicit blocker information.

These restrictions do not apply retroactively to interactive maintainer-agent
development.

## Development Loop

The normal loop is:

```text
maintainer request or observed defect
                 |
                 v
inspect canonical docs, source, and tests
                 |
                 v
resolve important design questions with the maintainer
                 |
                 v
implement a coherent change and an independent oracle
                 |
                 v
run deterministic checks and inspect the final diff
                 |
                 v
independent review when risk warrants it
                 |
                 v
maintainer decides whether to commit, push, or continue
```

Agents should stay with the work through implementation and verification unless
the maintainer pauses the task or an unresolved semantic decision blocks further
progress.

## Proportional Review

Uni20 does not assign mandatory risk labels. Review depth follows behavior and
blast radius rather than file paths alone.

### Routine changes

Examples include spelling, links, focused Doxygen repair, and mechanical build
metadata. Use a focused diff and the directly relevant checks.

### Numerical or runtime changes

Examples include BLAS/LAPACK wrappers, scalar promotion, tensor ownership,
backend fallback, async ordering, cancellation, and AD rules. Require targeted
regression coverage, broader neighboring tests, documentation updates, and an
independent review where practical.

### Architectural or scientific changes

Examples include epoch semantics, symmetry selection rules, block-sparse
representation, distributed execution, truncation policy, and tensor-network
algorithm architecture. These require explicit maintainer-led design decisions
and evidence that the scientific and metadata invariants are preserved.

A documentation file can define high-risk semantics, and a source file in a
high-risk subsystem can receive a mechanical correction. Paths are review hints,
not automatic classifications.

## Independent Review

Independent review means a genuinely separate review context: another model,
another session, or a human reviewer who did not produce the implementation.
It supplements rather than replaces deterministic evidence.

Review should examine the accepted behavior, test oracle, implementation diff,
and nearby equivalent paths. Use [Reviewing Uni20 Changes](code_review.md) as the
shared checklist.

The useful property is independent scrutiny, not a particular AI vendor. A
policy-compliance pass is not a correctness review.

## Durable Records

Do not create mandatory session work logs. Store information according to its
lifetime:

- canonical semantics and architecture belong in subsystem documentation;
- regression behavior belongs in tests;
- implementation rationale belongs in focused comments when the code is not
  self-explanatory;
- verification and residual risk belong in the commit or pull-request summary;
- temporary investigations and agent notes remain uncommitted unless their
  conclusions are curated into canonical documentation.

Raw transcripts and generated reasoning dumps are not project documentation.

## Repository Audits

Agents are useful for read-only neighborhood and cross-cutting audits. Candidate
findings must be reproduced or otherwise verified before they become defects to
fix. Do not create batches of issues from speculative findings.

Useful audits include:

- scalar, layout, provider, and workspace parity in BLAS/LAPACK wrappers;
- rank, zero-extent, negative-stride, and non-contiguous mdspan behavior;
- async lifetime, exception, cancellation, and deadlock paths;
- accessor lowering and hidden tensor materialization;
- symmetry metadata preservation.

Keep audits scoped to one current code lineage. In particular, do not infer that
removed legacy elementwise or contraction prototypes define the contracts of
the tensor and linalg dispatch layers.

## Future Automation

Issue labels, machine-readable audit schemas, risk classifiers, guidance
manifests, and headless runners should be added only when a concrete workflow
needs them. They are not prerequisites for effective agent-assisted development
and should not become parallel sources of repository policy.
