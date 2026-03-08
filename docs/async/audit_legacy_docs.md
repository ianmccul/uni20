# Audit of Existing Async Docs

This is a quick audit of async-related docs that existed before the current draft track under `docs/async/`.

## Summary

- Some async docs are tracked in git but describe pre-refactor semantics.
- Several local markdown files are not tracked and are also stale.
- The canonical reference is the focused docs set under `docs/async/`.

## File-by-file notes

## Tracked (historical)

The following files existed as tracked docs and were removed in favor of `docs/async/`:

- `docs/Async.md`
- `docs/DebugScheduler.md`
- `docs/Schedulers.md`
- `docs/TbbScheduler.md`

Reason for removal:

- Significant semantic drift from current runtime behavior.
- Overlap/conflict with the new canonical `docs/async/` set.

`docs/Epoch.md` remains an untracked local draft and is also stale.

## Local/untracked drafts found in workspace

- `docs/async.md`
  - Uses obsolete internal flags (`writer_claimed_`, `writer_task_set_`, `writer_done_`).

- `docs/async_api.md`
  - Describes outdated awaiter and queue internals.
  - Assumes old API semantics and misses explicit unconstructed-storage behavior.

- `docs/async_design.md`
  - Design exploration around `CreateBuffer`/`WritableBuffer`.
  - Not aligned with current canonical `ReadBuffer`/`WriteBuffer` model.

- `docs/async_new.md`
  - Conceptually close for dataflow AD framing.
  - Still diverges from current concrete API and exception-routing details.

## Main semantic mismatches to avoid in new docs

- Treating default `Async<T>` as default-constructed `T` (current behavior is unconstructed storage).
- Treating `WriteBuffer` as always returning writable reference (current behavior can throw `buffer_write_uninitialized`).
- Referring to removed write APIs (`mutate`, `MutateBuffer`, `EmplaceBuffer`).
- Using legacy task cancellation names.
- Describing epoch internals in terms of removed flags.

## Replacement plan

- Keep legacy docs for historical context only.
- Use the focused docs (`getting_started.md`, `runtime_model.md`, `buffers_and_awaiters.md`,
  `exceptions_and_cancellation.md`, `reverse_mode_ad.md`, and `quick_reference.md`) as canonical.
