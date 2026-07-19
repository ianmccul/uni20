# AI Guidance for Uni20

- **Audience:** remote assistants, coding agents, reviewers, and maintainers
- **Authority:** non-normative index and retrieval guidance
- **Reviewed against:** `Uni20-dev/uni20` `main`, 2026-07-18
- **Custom GPT guidance checked:** OpenAI Help Center, 2026-07-19
- **Canonical sources:** `AGENTS.md`, canonical subsystem documentation, source, and tests

These files optimize for retrieval. They are not substitutes for canonical docs.

## Authority order

1. Maintainer-approved decisions and canonical subsystem documentation
2. Current source implementation
3. Focused tests
4. AI guidance summaries

When these disagree, report the drift. Do not silently choose the guidance file.

## Required answer distinctions

Label claims as one of:

- documented invariant;
- current implementation detail;
- current design direction;
- roadmap/open question.

## File map

- `architecture_status.md`: current implemented vertical slices and roadmap boundaries
- `async_runtime.md`: `Async`, epoch ordering, buffers, aliases, assignment, and wrappers
- `reverse_mode_ad.md`: dataflow AD, finalization, cancellation, and complex gradients
- `tensor_dispatch_design.md`: Tensor roles, accessor semantics, operations, and dispatch
- `presentation_and_python.md`: implemented presentation/preview and future Python display
- `cuda_scheduler_notes.md`: implemented CUDA foundation versus future scheduler work
- `custom_gpt_setup.md`: how to package this directory as Custom GPT knowledge
- `glossary.md`: compact terminology index

## Custom GPT use

Use this directory as orientation material, not as the GPT's main instruction
block. If the GPT can browse the current repository, prefer direct repo reads of
canonical docs/source/tests over uploaded snapshots. The short instruction block
and setup checklist are in `custom_gpt_setup.md`.

## Maintenance rule

Implementation-heavy edits should update the `Reviewed against` date and verify
the relevant canonical docs/source/tests. Keep the glossary compact; do not turn
it into a second semantic specification.
