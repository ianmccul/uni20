# Custom GPT Setup for Uni20

- **Audience:** maintainers configuring a web ChatGPT Custom GPT for Uni20
  discussions
- **Authority:** non-normative packaging guidance
- **Reviewed against:** OpenAI GPT Action and Help Center guidance, 2026-08-20
- **Canonical sources:** `AGENTS.md`, canonical subsystem documentation, source,
  tests, and the current GPT configuration in ChatGPT

The Custom GPT should be repository-first. Uploaded knowledge supplies durable
review conventions; it must not become a cached description of the current code.
Do not copy this directory wholesale into the GPT **Instructions** field.

## Recommended GPT Instructions

Paste a block like this into the GPT **Instructions** field:

```text
You are a Uni20 design and code-review assistant.

Primary job:
Help reason about Uni20 architecture, implementation plans, code reviews, and
documentation. Be direct, technical, and evidence-driven.

Canonical repository:
- GitHub: https://github.com/Uni20-dev/uni20
- Default branch: main
- When the user names a branch, commit, tag, PR, or diff, inspect that exact
  snapshot and state which snapshot was used.
- Otherwise inspect current main and identify the commit used.

Source priority:
1. Maintainer instructions and decisions in the current conversation.
2. Repository-wide AGENTS.md and canonical subsystem documentation in the
   named repository snapshot.
3. Source, focused tests, and build configuration in that same snapshot.
4. Uploaded docs/ai_guidance files as non-normative operating guidance only.

Repository-first rule:
- Inspect the named repository snapshot before making exact claims about current
  APIs, types, algorithms, backend or scalar coverage, async lowering, Python
  bindings, CUDA behavior, open work, or roadmap status.
- Never use uploaded AI guidance as evidence that a feature exists, is absent,
  has a particular API, or remains planned.
- Do not combine documentation, source, or tests from different snapshots
  without identifying the mismatch.
- If repository inspection is unavailable, say that current status cannot be
  verified and ask for the relevant files, diff, or commit. Do not answer from
  remembered implementation summaries.

Design and review stance:
- Uni20 is in active design. Do not preserve stale development names or helper
  shapes merely for compatibility unless the maintainer requests it.
- Distinguish intended contract, current implementation, proposed direction,
  and open questions when the distinction matters.
- Report conflicts among maintainer decisions, canonical docs, source, and tests
  instead of silently choosing one.
- Lead code reviews with correctness defects, invariant violations, behavioral
  regressions, and missing tests before style observations.

Durable Uni20 invariants:
- Preserve symmetry metadata; never introduce an implicit dense fallback for a
  symmetry-typed path.
- Async legality comes from epoch causality, not scheduler timing.
- Respect mdspan accessor semantics; a pointer-shaped handle does not prove raw
  provider readability or writability.
- Backend fallback is permitted only after a clean decline. An operation may
  authorize provisional preparation of a replaceable output; input mutation,
  result writes, work submission, or completed-result commitment are terminal.
- Async coroutine lambdas are captureless and static; pass state as parameters.

Answer style:
- Lead with findings, risks, or the recommended design.
- Cite repository paths for current-state claims and say what was inspected.
- Do not infer uniform scalar, layout, backend, or async support from one path.
- For external or time-sensitive facts, inspect current official or vendor
  sources, or state that a current check is required.
```

The instruction block contains process and durable invariants only. Do not add
subsystem inventories, exact class relationships, current operation coverage,
or roadmap summaries to it.

## Knowledge Files

Upload only the compact, refactor-resistant guidance:

- `README.md`
- `review_contract.md`
- `glossary.md`

These files explain how to inspect and review Uni20. They do not provide an
offline substitute for the repository. If direct repository access is
unavailable, the GPT should request the relevant files or commit rather than
answering a current-state question from uploaded knowledge.

Knowledge works best for reference material. Behavior, tone, and workflow rules
belong in the Instructions field. Current subsystem behavior belongs in
canonical repository docs, source, tests, and build files.

## Conversation Starters

Useful starters for the GPT:

- "Review this Uni20 design proposal against the named repository snapshot."
- "Inspect this PR for invariant violations and missing tests."
- "Which canonical Uni20 files establish the contract for this change?"
- "Does this CUDA or async proposal preserve the repository's causal rules?"

## Capabilities

Enable only the capabilities needed for the intended discussion.

- Web search is useful for direct reads of the public Uni20 repository and for
  current external facts such as CUDA or vendor-library behavior.
- Code Interpreter & Data Analysis is useful for reading uploaded source bundles
  or logs.
- Apps or Actions are unnecessary for ordinary Uni20 design discussion. If an
  action is added, keep its schema and credentials narrow.

A GPT can use apps or actions, but not both at the same time. Account for that
before adding a live GitHub, CI, or documentation action.

## Optional GitHub Action

Most Uni20 discussions should rely on direct repository browsing. If a GitHub
Action is useful, use `github_repo_action.openapi.yaml`.

Default read-only configuration:

- Action domain: `api.github.com`
- Authentication: `None` / no authentication
- Privacy policy URL:
  <https://github.com/Uni20-dev/uni20/blob/main/docs/ai_guidance/custom_gpt_action_privacy_policy.md>
- Schema shape: keep operation parameters inline. The ChatGPT action importer
  may skip operations whose `parameters` list contains reusable parameter
  `$ref`s instead of concrete `name` / `in` fields.

If public read operations such as `getUni20Repository` or `listUni20Branches`
raise a generic `ClientResponseError`, first check the GPT Action authentication
configuration. GitHub public REST reads succeed without a token, but an invalid
or empty `Authorization: Bearer ...` header can make otherwise-public requests
fail.

To enable issue and issue-comment writes, use the same schema but change the
Action authentication setting:

- Action domain: `api.github.com`
- Authentication: API key / Bearer token
- Token scope: keep the token as narrow as practical for `Uni20-dev/uni20`,
  normally repository metadata read plus issues read/write
- Write operations:
  - `createUni20Issue`
  - `updateUni20Issue`
  - `createUni20IssueComment`
  - `updateUni20IssueComment`
  - `deleteUni20IssueComment`

Every mutation is marked `x-openai-isConsequential: true`. This makes the
intended confirmation boundary explicit, but it does not change the default:
OpenAI already treats non-`GET` operations as consequential when the field is
absent. Do not treat adding the explicit field as a repair for a failed write.

Use mutation operations only after the user explicitly requests the write or
approves the final content. Before deleting a comment, verify the exact comment
ID and current body.

Do not configure the read-only schema with a bearer token solely because an
issue-write path may be useful later.

The GPT Action builder treats `api.github.com` as a single Action domain, so do
not install a second Uni20 GitHub schema for issue writes. Keep one schema and
switch only the Action authentication setting when write access is deliberately
needed.

Do not add PR creation, discussion creation, workflow dispatch, file mutation,
or broad repository write actions without a concrete need and explicit user
approval rules.

## Response Budget

GPT Action request and response payloads must remain below 100,000 characters,
and Action requests time out after 45 seconds. GitHub list, comparison, tree,
and full-commit responses can exceed that budget.

- Use `getUni20Branch` to obtain a named branch's head SHA.
- Use `getUni20Commit` only when full commit metadata, file lists, or patches are
  required.
- List operations default to at most five records. Five is a conservative
  current default, not a guarantee: response size still depends on issue,
  comment, pull-request, branch, or search-result content.
- Paginate focused list requests and avoid recursive trees or broad comparisons
  unless the task requires them.

Reducing an OpenAPI response schema does not make GitHub send fewer bytes. Use a
lighter endpoint, GraphQL field selection, or middleware when a strict response
shape is required.

## Write-Path Diagnostics

If a mutation hangs or returns no usable Action result, do not infer whether the
request was blocked before dispatch, rejected by GitHub, timed out, or failed
while the Action runtime processed the response. Obtain the raw Action status
and response before assigning a cause.

Test the same token value and repository-access configuration outside ChatGPT
with a request that cannot create an issue:

```bash
curl -i -X POST \
  -H "Authorization: Bearer $UNI20_GPT_GITHUB_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  -H "Content-Type: application/json" \
  -H "X-GitHub-Api-Version: 2026-03-10" \
  https://api.github.com/repos/Uni20-dev/uni20/issues \
  -d '{}'
```

The missing required `title` prevents creation. A `422` response that identifies
the missing title shows that the tested token reached the create endpoint and
was accepted far enough for request validation. This shell test does not verify
the GPT confirmation UI, Action dispatch, connector runtime, or response
processing.

After the token test:

1. Save or re-import the schema and test reads in the Action builder's Preview.
2. Test `getUni20Branch` with `branch=main`.
3. Test `listUni20Issues` with `per_page=5`.
4. Run one minimal, explicitly approved create request while recording whether
   confirmation appeared, the HTTP status, response body, and any request ID.
5. Verify the returned issue number with `getUni20Issue` before another write.

Use a new chat after publishing the revised GPT as a conservative way to avoid
relying on an already-open conversation's loaded configuration. This is an
operational precaution, not a documented guarantee about schema caching.

## Preview Tests

After the Action diagnostics pass, test the evidence process rather than
memorized answers:

1. Ask an exact current-support question. The GPT should inspect the repository,
   identify the snapshot, and cite relevant docs/source/tests.
2. Ask about a named PR or old commit. The GPT should use that snapshot rather
   than current `main`.
3. Ask whether scheduler timing, a pointer-shaped mdspan handle, a dirty backend
   decline, or symmetry erasure is acceptable. The durable invariants should be
   applied consistently.
4. Provide an uploaded-guidance statement that conflicts with current source or
   canonical docs. The GPT should report the conflict and reject the guidance as
   current-state evidence.
5. Disable repository access and ask a current implementation question. The GPT
   should say it cannot verify the answer and request exact repository evidence.

## Maintenance Checklist

- Do not update AI guidance after ordinary subsystem refactors.
- Update it only when repository-wide conventions, durable invariants, review
  process, or GPT/Action configuration changes.
- Keep current implementation and roadmap facts in canonical subsystem docs.
- Remove accidental status summaries instead of dating and maintaining them.
- Re-test in Preview after changing instructions, knowledge, model, apps,
  actions, or capabilities.
- Keep `Reviewed against` dates honest when external GPT guidance is checked.

## Official GPT Guidance Checked

- <https://developers.openai.com/api/docs/actions/production>
- <https://developers.openai.com/api/docs/actions/getting-started>
- <https://help.openai.com/en/articles/8554407-custom-instructions-for-chatgpt>
- <https://help.openai.com/en/articles/8554397>
- <https://help.openai.com/en/articles/11325361-troubleshooting-gpts>
- <https://help.openai.com/en/articles/9442513>
- <https://help.openai.com/en/articles/20001066>
