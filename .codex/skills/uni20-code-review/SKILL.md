---
name: uni20-code-review
description: Review Uni20 changes for correctness, regressions, missing evidence, and architectural consistency.
---

# Uni20 Code Review

## Overview

Use this skill for substantive review of a Uni20 diff, commit, branch, or pull
request. The canonical review process and subsystem checklists are in
`docs/development/code_review.md`.

## Workflow

1. Read `AGENTS.md` and `docs/development/code_review.md`.
2. Identify the maintainer-approved contract or canonical design source.
3. Read changed tests and establish whether their oracle is independent.
4. Inspect the implementation diff and nearby equivalent paths.
5. Check the relevant numerical, tensor/mdspan, async/AD, or symmetry checklist.
6. Report findings first, ordered by severity, with file and line references.
7. If there are no findings, state that clearly and report residual test or
   configuration risk.

## Constraints

- Do not treat policy compliance as proof of correctness.
- Do not infer intended behavior solely from current code.
- Do not manufacture style findings when the implementation is sound.
- Do not edit files unless the user also asks to address the findings.

## Reporting Shape

- findings ordered as blocker, important, suggestion, or question;
- open assumptions that affect correctness;
- concise change summary only after the findings;
- tests and configurations not exercised.
