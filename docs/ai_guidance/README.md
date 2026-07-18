# AI Guidance for Uni20

- **Audience:** remote assistants, coding agents, reviewers, and maintainers
- **Authority:** non-normative index and retrieval guidance
- **Reviewed against:** `Uni20-dev/uni20` `main`, 2026-07-18
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
- `glossary.md`: compact terminology index

## Maintenance rule

Implementation-heavy edits should update the `Reviewed against` date and verify
the relevant canonical docs/source/tests. Keep the glossary compact; do not turn
it into a second semantic specification.
