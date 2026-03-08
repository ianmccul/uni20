# `.codex/` layout

This directory stores Codex-specific metadata and guidance.

## Files

- `.codex/instructions.md` (tracked): repository-wide Codex guidance.
- `.codex/instructions.local.md` (local only): machine-specific overrides (toolchains, local paths, etc.).
  - This file is git-ignored by `.gitignore`.
- `.codex/skills/` (tracked): reusable skill playbooks that may be shared with the repository.

## What belongs where

- Put **portable, repository-level** guidance in `.codex/instructions.md` or `.codex/skills/`.
- Put **machine-local** details in `.codex/instructions.local.md` only.
  - Examples: `/opt/...` compiler paths, local build cache locations, local package policy.

## Skills policy for this repo

Skills should remain machine-agnostic.  
If a skill needs local compiler or path details, it should defer to `.codex/instructions.local.md` instead of embedding them.

## Current skills

- `uni20-build-and-test`: default configure/build/test workflow.
- `uni20-cmake-dependency-triage`: dependency detection and cache-policy diagnosis.
- `uni20-build-matrix`: compiler/build-type feature matrix with auto-detected toolchains.
- `uni20-doxygen-audit`: AGENTS.md Doxygen policy auditing/normalization.
- `uni20-doc-sync`: documentation synchronization and stale-doc triage.

## How to invoke a skill

- Name the skill directly in your prompt, for example:
  - `Use uni20-cmake-dependency-triage to diagnose this configure failure.`
  - `Use uni20-build-matrix and run a compact matrix for this branch.`
- You can request more than one skill in a single prompt when needed.
- If your request clearly matches a skill, Codex may apply it automatically.
