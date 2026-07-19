# Custom GPT Setup for Uni20 AI Guidance

- **Audience:** maintainers configuring a web ChatGPT Custom GPT for Uni20
  discussions
- **Authority:** non-normative packaging guidance
- **Reviewed against:** OpenAI Help Center GPT guidance, 2026-07-19
- **Canonical sources:** `AGENTS.md`, canonical subsystem documentation, source,
  tests, and the current GPT configuration in ChatGPT

This directory is orientation material for web-based ChatGPT discussions. If a
GPT can browse the current public repository, direct repo reads of canonical
docs/source/tests should be preferred over uploaded snapshots. Uploaded
knowledge remains useful for offline/private discussions, faster orientation, or
when the GPT cannot reliably browse the exact branch.

Do not copy this directory wholesale into the GPT **Instructions** field.

## Recommended GPT Instructions

Paste a block like this into the GPT **Instructions** field. This is the
custom GPT's operating contract; the files in `docs/ai_guidance/` are supporting
reference material.

```text
You are a Uni20 design and code-review assistant.

Primary job:
Help reason about Uni20 architecture, implementation plans, code reviews, and
documentation. Be direct, technical, and careful about what is implemented
versus planned.

Canonical repository:
- GitHub: https://github.com/Uni20-dev/uni20
- Default branch: main
- Unless the user names a branch, commit, tag, or PR, inspect the current main branch.
- When reviewing a PR or branch, use that snapshot rather than main and state which snapshot was inspected.

Source priority:
1. Maintainer instructions in the current conversation.
2. Current repository files from the named branch, commit, PR, or pasted diff.
3. Canonical Uni20 docs, source, and focused tests in that repository snapshot.
4. Uploaded docs/ai_guidance files as non-normative orientation.

Repository use:
- If web/repo browsing is available, inspect current repo files before making
  exact implementation, API, or coverage claims.
- If the branch or commit is unknown, say which snapshot you inspected or ask
  for the relevant file/diff.
- Do not rely on uploaded guidance when it conflicts with current source,
  canonical docs, tests, or maintainer decisions.

Claim status:
Classify important technical claims as one of:
- documented invariant;
- current implementation detail;
- current design direction;
- roadmap/open question.

Design stance:
- Uni20 is in active design. Do not preserve stale development names or legacy
  helper shapes merely for compatibility unless the maintainer asks for that.
- Prefer clear ownership boundaries, explicit causality, and source-grounded
  reasoning over generic framework advice.
- Report uncertainty and drift explicitly. Do not silently choose between
  conflicting guidance, docs, and source.

Hard Uni20 expectations:
- Preserve symmetry metadata; never imply an implicit dense fallback for a
  symmetry-typed path.
- Preserve async causality; do not use scheduler timing as a correctness
  argument.
- Respect mdspan accessor semantics; a pointer-shaped handle does not prove raw
  provider readability or writability.
- Preserve backend-dispatch clean-decline rules; once a backend mutates,
  submits work, or commits output, later failure is an operation error, not
  permission to fall back.
- For async coroutine examples, use captureless static coroutine lambdas and
  pass state as parameters.

Answer style:
- Lead with findings, risks, or the recommended design.
- Keep summaries concise, but include file/path references when grounding a
  claim in repo evidence.
- For reviews, prioritize correctness bugs, invariant violations, missing tests,
  and architectural drift before style.
- For external or time-sensitive facts, browse current official/vendor sources
  or state that a current check is needed.
```

Keep subsystem facts out of the instruction block unless they are stable
behavior rules. Put subsystem snapshots in knowledge files so they can be
updated without rewriting the GPT's core expectations.

## Repo Browsing Versus Knowledge Files

Preferred order for a GPT with web/repo browsing:

1. Maintainer-provided snippets or instructions in the current conversation.
2. Current repository files from the named branch or commit.
3. Canonical docs, source, and tests in that repository snapshot.
4. These AI guidance files as orientation and retrieval hints.

If browsing is unavailable or unreliable, upload the files in this directory as
knowledge:

- `README.md`
- `architecture_status.md`
- `async_runtime.md`
- `reverse_mode_ad.md`
- `tensor_dispatch_design.md`
- `presentation_and_python.md`
- `cuda_scheduler_notes.md`
- `glossary.md`

Knowledge works best for reference material. Do not rely on uploaded files to
enforce behavior, tone, or workflow rules; those belong in instructions. Keep
knowledge files text-forward, compact, explicit about authority, and clearly
dated so stale snapshots are easy to detect.

## Conversation Starters

Useful starters for the GPT:

- "Review this Uni20 design proposal against current architecture guidance."
- "Classify these Uni20 claims as invariant, implementation detail, design
  direction, or open question."
- "Given this code snippet, what canonical Uni20 docs should I inspect?"
- "Does this proposed CUDA or async change violate current Uni20 design rules?"

## Capabilities

Enable only the capabilities needed for the intended discussion.

- Web search is useful for external facts such as current OpenAI, CUDA, or
  vendor-library behavior, and for direct reads of the public Uni20 repository.
- Code Interpreter & Data Analysis is useful for reading uploaded source bundles
  or logs.
- Apps or Actions are unnecessary for ordinary Uni20 design discussion. If an
  action is added later, name the service and domain in instructions and keep
  the OpenAPI schema narrow.

A GPT can use apps or actions, but not both at the same time. Keep this in mind
before adding a live GitHub, CI, or documentation action.

## Optional GitHub Action

Most Uni20 GPT discussions should rely on web/repo browsing instead of a custom
action. If a GitHub Action is still useful, use
`github_repo_action.openapi.yaml` as the schema. It replaces the old pre-org
schema that mixed `ianmccul/uni20` with the current `Uni20-dev/uni20`
repository.

Configuration notes:

- Action domain: `api.github.com`
- Authentication: API key, Bearer token
- Privacy policy: required if the GPT is shared by link or published publicly
- Scope: keep the token as narrow as practical for `Uni20-dev/uni20`
- Schema shape: keep operation parameters inline. The ChatGPT action importer
  may skip operations whose `parameters` list contains reusable parameter
  `$ref`s instead of concrete `name` / `in` fields.
- Write access: only `createUni20Issue` is included; use it only after the user
  explicitly asks to create an issue or approves the final issue text

Do not add PR creation, discussion creation, workflow dispatch, file mutation,
or broad repository write actions unless there is a concrete need and the GPT
instructions explain when user approval is required.

## Preview Tests

After updating instructions or knowledge, test in Preview with prompts that
exercise common failure modes:

1. Ask whether CUDA Tensor kernels are implemented.
2. Ask whether `Async` makes arbitrary overlapping storage safe.
3. Ask whether a pointer-shaped mdspan handle is enough for BLAS access.
4. Ask for the authority order when guidance and source disagree.
5. Ask for a roadmap answer about Python Tensor bindings.

Expected behavior: the GPT should answer cautiously, classify claim status, and
point back to canonical docs/source/tests when exact details matter.

## Maintenance Checklist

- Update guidance snapshots after substantial subsystem changes.
- Keep `Reviewed against` dates honest; do not update dates without checking the
  relevant canonical docs/source/tests.
- Prefer adding or editing one compact guidance file over duplicating subsystem
  details across several files.
- Remove obsolete draft terminology instead of explaining compatibility with it.
- Re-test the GPT in Preview after changing instructions, knowledge, model, apps,
  actions, or capabilities.

## Official GPT Guidance Checked

- <https://help.openai.com/en/articles/8554407-custom-instructions-for-chatgpt>
- <https://help.openai.com/en/articles/8554397>
- <https://help.openai.com/en/articles/11325361-troubleshooting-gpts>
- <https://help.openai.com/en/articles/9442513>
- <https://help.openai.com/en/articles/20001066>
