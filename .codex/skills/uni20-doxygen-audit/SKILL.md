---
name: uni20-doxygen-audit
description: Audit and normalize Uni20 Doxygen comments to AGENTS.md policy with minimal, scoped edits.
---

# Uni20 Doxygen Audit

## Overview

Use this skill to enforce the Doxygen policy in `AGENTS.md` section 7.

Keep edits scoped to requested files or changed files unless a full-tree audit is explicitly requested.

## Workflow

1. Read `AGENTS.md` Doxygen policy before editing.
2. Determine audit scope (`git diff` files by default).
3. Identify comment-shape and tag-order violations.
4. Patch only documentation issues (avoid unrelated code churn).
5. Re-scan scope for remaining violations and summarize.

## Audit Scope Helpers

Changed C++ headers/sources:

```bash
git diff --name-only -- '*.hpp' '*.h' '*.cpp' '*.cc' '*.cxx'
```

Find Doxygen blocks quickly:

```bash
rg -n "^\s*///|^\s*/\*\*|\\\brief|\\\param|\\\tparam|\\\return|\\\ingroup" src tests bindings
```

## Policy Checklist

- Prefer `///` for ordinary declarations; reserve `/** ... */` for file/module overviews.
- Every Doxygen block starts with `\brief`.
- Include `\param`, `\tparam`, `\return` where applicable.
- Do not emit `\return` on non-callable declarations.
- Keep tag order exactly as specified by `AGENTS.md`.
- Preserve useful implementation comments (`//`, `/* ... */`) as non-Doxygen comments.

## Output Expectations

- Summarize audited files and key fixes.
- Call out any ambiguous declarations requiring manual semantic clarification.
- If full compliance is not completed, list concrete remaining items by file.
