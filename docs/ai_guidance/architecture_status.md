# Uni20 Architecture Status: AI Guidance

This file is for questions about project maturity, active design seams, and what should or should not be described as stable.

## File-level answer rule

- State what is implemented today.
- Separate implemented behavior from roadmap material.
- Treat `src/uni20/async/` as the current center of gravity.
- Treat other subsystems more cautiously unless specific files are inspected.

## `src/uni20/async/`

### STATUS

- `src/uni20/async/` is the most mature subsystem.

### SAFE CLAIMS

- `Async<T>`, `ReadBuffer<T>`, `WriteBuffer<T>`, `EpochQueue`, and `EpochContext` are established core types.
- `DebugScheduler`, `TbbScheduler`, and `TbbNumaScheduler` exist and are active scheduler types.
- Async ordering and coroutine safety are primary architectural constraints.

### DO NOT CLAIM

- Do not describe the async surface as fully simplified or permanently settled.

### RELATED

- `../async/README.md`
- `../async/runtime_model.md`

## reverse-mode AD

### STATUS

- Reverse-mode AD is usable and conceptually mature.

### SAFE CLAIMS

- Reverse-mode AD uses `Var<T>` and `ReverseValue<T>`.
- Reverse-mode AD is integrated with the async runtime.
- Current reverse accumulation is deterministic.

### DO NOT CLAIM

- Do not describe the reverse-mode API as fully cleaned up.
- Do not describe Uni20 AD as tape-replay-based.

### DESIGN SEAMS

- `ReverseValue<T>` still has overlapping API surface.
- API simplification is still a legitimate topic.

### RELATED

- `reverse_mode_ad.md`

## CPU / BLAS path

### STATUS

- CPU / BLAS support is usable.

### SAFE CLAIMS

- CPU linear algebra exists.
- BLAS integration exists.
- BLAS vendor selection uses `UNI20_BLAS_VENDOR` and then `BLA_VENDOR`.

### DO NOT CLAIM

- Do not use CPU / BLAS support as evidence that GPU support is equally mature.

## tensor / `TensorView`

### STATUS

- Tensor and view semantics are still evolving.

### SAFE CLAIMS

- `TensorView` is conceptually important.
- Ownership and lifetime sharing are not fully settled.
- Async-safe aliasing rules are not fully settled.

### DO NOT CLAIM

- Do not claim that tensor/view lifetime semantics are finalized.
- Do not claim that aliasing is solved by the async runtime.

### DESIGN SEAMS

- `assignment_semantics_of<T>` is important for future tensor/view write semantics.
- View-like types may need `write_through` rather than `rebind`.

### RELATED

- `async_runtime.md`
- `../roadmap.md`

## expression layer

### STATUS

- The expression layer is partial.

### SAFE CLAIMS

- Explicit operations exist.
- A fuller expression or fusion layer is still roadmap material.

### DO NOT CLAIM

- Do not claim that fusion or lazy expression lowering is already a mature subsystem.

## GPU / heterogeneous execution

### STATUS

- GPU / heterogeneous execution is partial.

### SAFE CLAIMS

- CUDA-related directories and types exist.
- Full CUDA execution is not complete.
- Full cuSOLVER execution is not complete.

### DO NOT CLAIM

- Do not describe GPU support as complete unless inspected code proves it.

## build / CMake

### STATUS

- Uni20 uses modern CMake.

### SAFE CLAIMS

- Uni20 uses both system-package discovery and `FetchContent`.
- Some dependency handling is centralized in `cmake/Uni20Dependencies.cmake`.
- Not every dependency path necessarily uses the same helper.

### DO NOT CLAIM

- Do not infer dependency wiring without inspecting CMake.
- Do not infer imported targets without inspecting CMake.
- Do not infer transitive linkage without inspecting CMake.

## testing status

### STATUS

- Async and AD behavior have relatively strong test coverage compared with the rest of the project.

### SAFE CLAIMS

- `tests/async/` is a relatively strong source of behavioral ground truth.

### DO NOT CLAIM

- Do not assume equal test maturity across non-async subsystems.

## final caution

- Avoid overselling tensor maturity.
- Avoid overselling GPU maturity.
- Treat async ordering, coroutine safety, and lifetime rules as primary constraints.

## Related detailed docs

- `../roadmap.md`
- `../architecture_diagram.md`
- `../async/README.md`
- `../async/runtime_model.md`
