---
name: uni20-build-and-test
description: Configure, build, and run tests for the Uni20 repository with project-specific conventions. Keep this skill machine-agnostic and defer local toolchain/path details to .codex/instructions.local.md when available.
---

# Uni20 Build and Test

## Overview

Run reproducible configure, build, and test workflows for Uni20.
Use exact commands, keep output concise, and report failures with actionable root-cause hints.

## Workflow

1. Read `AGENTS.md` before running commands.
2. If present, read `.codex/instructions.local.md` and apply local overrides from there.
3. Configure in `./build_codex` (or subdirectories of `./build_codex`) only.
4. Build requested targets first; run full builds only when needed.
5. Run targeted tests for changed areas, then broader `ctest` if requested.
6. Summarize command results with failing target/test names and next steps.

## Configure

Use these command templates unless the user asks for different options.
Do not hardcode machine-local compiler paths in this skill.

```bash
cmake -S . -B ./build_codex
```

```bash
cmake -S . -B ./build_codex/no_stacktrace -DUNI20_ENABLE_STACKTRACE=0
```

## Build

```bash
cmake --build ./build_codex
```

```bash
cmake --build ./build_codex --target uni20_async_tests
```

```bash
cmake --build ./build_codex/no_stacktrace --target uni20_async_tests
```

## Test

```bash
ctest --test-dir ./build_codex --output-on-failure
```

```bash
ctest --test-dir ./build_codex --output-on-failure -R "TaskRegistryDebugTest.DumpShowsTaskStateAndTransitions"
```

```bash
ctest --test-dir ./build_codex --output-on-failure -R "AsyncTaskAwaitTest.AsyncTaskAwait_NestedAssignment|TbbScheduler.BasicSchedule|TbbScheduler.AsyncAccumulationGetWait"
```

```bash
ctest --test-dir ./build_codex/no_stacktrace --output-on-failure -R "TaskRegistryDebugTest.DumpShowsTaskStateAndTransitions"
```

## Failure Triage

- Handle system dependency version mismatches by setting the corresponding `UNI20_USE_SYSTEM_<DEP>=OFF` override when appropriate.
- Handle `generator does not match the generator used previously` by clearing stale cache files in that build directory, then re-running configure.
- Handle stacktrace support issues by switching to `-DUNI20_ENABLE_STACKTRACE=0` and continuing with degraded debug output.
- Report unresolved failures with exact failing target or test names and the first relevant error lines.

## Reporting Format

Use this summary shape in the final response:

- `configure`: command and status
- `build`: targets built and status
- `tests`: tests run, pass or fail counts, failed names
- `notes`: root cause and next action
