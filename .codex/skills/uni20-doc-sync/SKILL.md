---
name: uni20-doc-sync
description: Keep Uni20 documentation aligned with current code behavior and API surface, with emphasis on async and developer-facing docs.
---

# Uni20 Doc Sync

## Overview

Use this skill to sync docs after API/behavior changes and keep docs readable for non-native speakers.

Prefer direct, concrete wording and examples over abstract descriptions.

## Workflow

1. Read `AGENTS.md` documentation expectations.
2. Determine code-change scope and affected docs.
3. Update docs for behavior/API changes, examples, and caveats.
4. Identify stale or redundant docs and propose keep/delete actions.
5. Summarize doc updates and unresolved gaps.

## Discovery Commands

Changed files:

```bash
git diff --name-only
```

Tracked docs:

```bash
git ls-files docs
```

Untracked docs candidates:

```bash
find docs -type f -name '*.md' | sort
```

## Sync Checklist

- Update docs when public behavior or usage patterns changed.
- Ensure examples match current API names and semantics.
- Keep terminology consistent across docs (`Var`, `Async`, `ReadBuffer`, `WriteBuffer`, etc.).
- Highlight ordering/lifetime/cancellation semantics where misuse is likely.
- Prefer short paragraphs, explicit bullets, and runnable command/code snippets.

## Keep/Delete Guidance

- Keep docs that explain current architecture, semantics, or workflows.
- Merge overlapping docs when one canonical page is enough.
- Delete clearly obsolete scratch notes once content is preserved elsewhere.
- Do not delete user-requested retained docs.

## Reporting Format

- `updated`: files updated and why
- `added`: new docs and target audience
- `obsolete`: candidates to remove/merge
- `gaps`: remaining undocumented behavior
