# AI Guidance for Uni20

This directory is for AI assistants. It is not optimized for normal human reading.

Human readers should usually prefer:

- `docs/async/`
- `docs/roadmap.md`
- `docs/architecture_diagram.md`

The files here are written for **retrieval quality** rather than for narrative readability.

That means:

- repeated exact term names such as `Async<T>`, `EpochQueue`, `ReadBuffer<T>`, `WriteBuffer<T>`, `Var<T>`, and `ReverseValue<T>`
- short declarative statements
- explicit invariants
- explicit negative knowledge
- minimal prose
- deliberate repetition

For a human reader, this style may feel dry, repetitive, or unnatural. That is intentional.

## Why this directory exists

Custom GPT instructions are small. Uni20 semantics are not small.

The intended split is:

- GPT instructions: policy, tone, skepticism, and hard safety rules
- uploaded files from this directory: project semantics and reference material

## Schema used in this directory

This directory uses a hybrid schema.

- Each file starts with a file-level summary.
- Key terms or subsystems then use structured blocks.

Common field names:

- `ROLE`
- `INVARIANTS`
- `CAUSAL MODEL`
- `LIFETIME / OWNERSHIP`
- `FAILURE MODES`
- `MISCONCEPTIONS`
- `STATUS`
- `SAFE CLAIMS`
- `DO NOT CLAIM`
- `RELATED`

Not every file uses every field. Use the fields that improve retrieval.

## Status and authority

- These files are summaries.
- These files are not a replacement for code or tests.
- `docs/async/` remains the canonical detailed async documentation.
- If AI guidance, code, and tests disagree, code and tests win.

## Design rules for these files

- Optimize for retrieval, not prose flow.
- Prefer explicit invariants over analogies.
- Prefer repeated exact terms over elegant paraphrase.
- Put negative knowledge near the relevant term when possible.
- Keep each file small enough to upload as custom-GPT knowledge.
- Avoid machine-specific instructions and private notes.
- Mark roadmap material clearly.

## File map

- `async_runtime.md`: `Async<T>`, `EpochQueue`, `ReadBuffer<T>`, `WriteBuffer<T>`, `assignment_semantics_of<T>`, release rules, aliasing limits
- `reverse_mode_ad.md`: `Var<T>`, `ReverseValue<T>`, `backprop()`, gradient materialization, Wirtinger convention, reverse-kernel rules
- `architecture_status.md`: mature areas, partial areas, design seams, and build-system cautions
- `glossary.md`: compact retrieval-first definitions for repeated Uni20 terms

## Recommended use

- Upload these files as GPT knowledge files.
- Keep behavioral rules in `chatgpt.md` in this directory, or in equivalent custom instructions.
- In answers, distinguish between:
  - documented invariant
  - current implementation detail
  - roadmap / open design question

## Related docs

- `../async/README.md`
- `../async/runtime_model.md`
- `../async/buffers_and_awaiters.md`
- `../async/reverse_mode_ad.md`
- `../roadmap.md`
