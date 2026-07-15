---
name: uni20-build-matrix
description: Run a machine-agnostic Uni20 configure/build/test matrix by detecting available compilers and applying a minimal coverage strategy.
---

# Uni20 Build Matrix

## Overview

Run a compact but meaningful build matrix without assuming specific compiler install paths.

This skill is for compatibility checks, regressions, and release confidence. Keep matrix size proportional to the request.

## Workflow

1. Read `AGENTS.md`, then `.codex/instructions.local.md` (if present).
2. Detect available compilers on the current machine.
3. Pick a minimal matrix that covers requested dimensions (compiler, build type, stacktrace/LTO toggles).
4. Select an out-of-source build root from the local override, the user's
   request, or a git-ignored default, then configure each cell under
   `<build-root>/<cell-name>`.
5. Summarize pass/fail per cell with failing targets/tests.

## Compiler Detection

Prefer local toolchain overrides if documented. Otherwise auto-detect:

```bash
for cxx in g++ g++-16 g++-14 g++-13 clang++ clang++-21 clang++-20 clang++-19; do
  command -v "$cxx" >/dev/null && echo "$cxx"
done | awk '!seen[$0]++'
```

Reject Clang versions below the supported Clang 19 floor. Prefer GCC 13 as the
compatibility baseline when it is available.

## Matrix Construction Rules

- Always include one **baseline** cell (default compiler, `Debug`).
- Add `Release` when performance/packaging concerns are in scope.
- Add `UNI20_ENABLE_STACKTRACE=0` cell when async-debug/stacktrace behavior is relevant.
- Add `CMAKE_INTERPROCEDURAL_OPTIMIZATION=ON` cell for LTO validation.
- Add second-compiler smoke cell only if another compiler is installed and requested.

## Configure Templates

Baseline:

```bash
cmake -S . -B <build-root>/<cell-name> -DCMAKE_BUILD_TYPE=Debug
```

Compiler override:

```bash
cmake -S . -B <build-root>/<cell-name> \
  -DCMAKE_C_COMPILER=<cc> \
  -DCMAKE_CXX_COMPILER=<cxx> \
  -DCMAKE_BUILD_TYPE=<Debug|Release>
```

Optional toggles:

- `-DUNI20_ENABLE_STACKTRACE=0`
- `-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON`

## Build and Test

```bash
cmake --build <build-root>/<cell-name>
ctest --test-dir <build-root>/<cell-name> --output-on-failure
```

For quick smoke checks, run targeted tests first, then full `ctest` if requested.

## Reporting Format

- `matrix`: list of executed cells and options
- `configure/build`: pass/fail per cell
- `tests`: pass/fail counts and failed test names per cell
- `notes`: actionable follow-ups (cache cleanup, option conflict, compiler-specific failures)
