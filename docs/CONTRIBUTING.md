# Contributing to Uni20

Uni20 is an early-stage C++23 project. Interfaces and implementation boundaries
are still being designed, so contributions should optimize for a clear and
correct design rather than compatibility with unfinished in-tree APIs.

## Start Here

Before editing code:

1. Read [`AGENTS.md`](../AGENTS.md). It contains the repository-wide coding,
   coroutine, mdspan, symmetry, testing, and documentation rules.
2. Read the relevant subsystem documentation and nearby tests.
3. Follow [Getting Started](getting_started.md) for configuration and build
   commands, and [Testing](development/testing.md) for the test organization.
4. Coding agents must also read `.codex/instructions.local.md` when it exists.
   This optional, git-ignored file records machine-local compiler and build-tree
   constraints without making them repository policy.

## Development Workflow

The normal workflow is direct collaboration between the maintainer and a human
or coding agent. A GitHub issue is useful when work needs an asynchronous handoff,
but it is not required for an interactive development session.

For a change:

1. Establish the intended behavior. A maintainer decision in the working session
   is sufficient; record durable semantics in the relevant design document.
2. Inspect the existing implementation and tests before choosing an API shape.
3. Implement the smallest coherent change that addresses the underlying design
   or defect. Uni20 does not require compatibility shims for unfinished APIs.
4. Add tests whose oracle is meaningfully independent of the implementation.
5. Update affected documentation and examples.
6. Run focused tests, then broader tests in proportion to the change.
7. Review the final diff for unrelated edits, hidden materialization, unsupported
   fallbacks, and missing scalar, layout, backend, or failure-path coverage.

The complete agent workflow is described in
[Agent-Assisted Development](development/agent_assisted_development.md).

## Semantic Authority

The intended contract comes from maintainer-approved decisions and canonical
subsystem documentation. Tests encode and enforce that contract. Source code
shows the current implementation, which may be the subject of a bug report.

When these disagree, report the conflict and resolve it explicitly. Do not infer
that current code or an existing test is automatically authoritative.

Files under `docs/ai_guidance/` are non-normative retrieval summaries. They help
agents find relevant concepts and known pitfalls but do not establish behavior.

## Evidence and Review

Evidence should match the risk of the change:

- Mechanical changes need a focused diff and the directly affected checks.
- Numerical and backend changes need regression tests with a reference result,
  differential comparison, exact small case, or another defensible oracle.
- Async, ownership, AD, and scheduler changes need success, failure,
  cancellation, and lifetime coverage where applicable.
- Symmetry and block-sparse changes must prove that quantum-number and block
  metadata are preserved. A dense fallback is not acceptable evidence for a
  symmetry-aware path.

Substantial changes benefit from an independent review context or model. Follow
[Reviewing Uni20 Changes](development/code_review.md); review findings should lead with
correctness defects and residual risks rather than style observations.

## Commits and Pull Requests

Keep commits focused and describe what changed and why. Pull requests are useful
for external contributions, asynchronous review, and large checkpoints, but
maintainer-directed work may be committed directly when explicitly requested.

A useful pull-request or checkpoint summary contains:

- the intended behavior or accepted design decision;
- the implementation summary;
- tests and other verification performed;
- skipped or unavailable checks;
- residual risks and deliberate follow-up work.

Do not commit raw agent transcripts, generated reasoning dumps, or temporary
working notes. Preserve durable design decisions in the canonical documentation
instead.

## Building and Testing

Use an out-of-source build and never reuse one build tree with a different
compiler or incompatible configuration. In the commands below, replace
`<build-dir>` with a locally appropriate ignored directory:

```bash
cmake -S . -B <build-dir> -DCMAKE_BUILD_TYPE=Debug
cmake --build <build-dir>
ctest --test-dir <build-dir> --output-on-failure
```

Use a supported compiler: GCC 13 or newer, or upstream Clang 19 or newer. Local
build roots, compiler paths, package-install policy, and optional dependency
configurations belong in `.codex/instructions.local.md` or another untracked
local configuration, not in portable repository documentation or skills.
Personal CMake presets may be stored in the git-ignored
`CMakeUserPresets.json`; project-wide presets, if added later, belong in the
tracked `CMakePresets.json`.
