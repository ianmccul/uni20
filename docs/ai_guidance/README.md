# AI Guidance for Uni20

- **Audience:** remote assistants, coding agents, reviewers, and maintainers
- **Authority:** non-normative operating guidance
- **Canonical sources:** maintainer decisions, `AGENTS.md`, canonical subsystem
  documentation, source, tests, and build configuration in the named repository
  snapshot

This directory configures repository-aware assistants without duplicating the
repository's implementation or design documentation. It intentionally contains
no subsystem status snapshots, API inventories, backend coverage tables, or
roadmap summaries.

## Core Rule

Use these files to learn **how to review Uni20**, not **what Uni20 currently
implements**.

Any claim about current APIs, type relationships, algorithms, backend support,
scalar coverage, async lowering, Python bindings, CUDA behavior, open work, or
roadmap status must come from direct inspection of the branch, commit, tag, PR,
or diff named by the user. If no snapshot is named, inspect current `main` and
identify the commit used.

If the repository snapshot cannot be inspected, say that the current state
cannot be verified. Ask for the relevant files, diff, or commit rather than
substituting remembered or uploaded implementation summaries.

## File Map

- [`review_contract.md`](review_contract.md): durable repository conventions,
  safety invariants, and the required evidence process
- [`glossary.md`](glossary.md): small vocabulary for stable review concepts
- [`custom_gpt_setup.md`](custom_gpt_setup.md): Custom GPT instruction block,
  Preview checks, and optional GitHub Action setup
- [`github_repo_action.openapi.yaml`](github_repo_action.openapi.yaml): narrow
  GitHub repository Action schema
- [`custom_gpt_action_privacy_policy.md`](custom_gpt_action_privacy_policy.md):
  privacy policy for the optional Action

## Maintenance Rule

Ordinary implementation refactors should not require changes here. Update this
directory only when one of these changes:

- repository-wide contributor conventions;
- a durable correctness or safety invariant;
- the evidence and review process;
- Custom GPT or GitHub Action configuration.

Put subsystem behavior and design decisions in canonical subsystem docs, not in
AI-only summaries. Do not reintroduce dated implementation snapshots or a second
semantic specification here.
