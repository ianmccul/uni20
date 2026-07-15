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
3. Select an out-of-source build directory from the local override, the user's
   request, or a git-ignored default. Do not reuse it with another compiler or
   incompatible configuration.
4. Build requested targets first; run full builds only when needed.
5. Run targeted tests for changed areas, then broader `ctest` if requested.
6. Summarize command results with failing target/test names and next steps.

## Configure

Use these command templates unless the user asks for different options.
Replace `<build-dir>` with the selected local path. Do not hardcode
machine-local build roots or compiler paths in this skill.

```bash
cmake -S . -B <build-dir>
```

```bash
cmake -S . -B <no-stacktrace-build-dir> -DUNI20_ENABLE_STACKTRACE=0
```

## Build

```bash
cmake --build <build-dir>
```

```bash
cmake --build <build-dir> --target uni20_async_tests
```

```bash
cmake --build <no-stacktrace-build-dir> --target uni20_async_tests
```

## Test

```bash
ctest --test-dir <build-dir> --output-on-failure
```

```bash
ctest --test-dir <build-dir> --output-on-failure -R "TaskRegistryDebugTest.DumpShowsTaskStateAndTransitions"
```

```bash
ctest --test-dir <build-dir> --output-on-failure -R "AsyncTaskAwaitTest.AsyncTaskAwait_NestedAssignment|TbbScheduler.BasicSchedule|TbbScheduler.AsyncAccumulationGetWait"
```

```bash
ctest --test-dir <no-stacktrace-build-dir> --output-on-failure -R "TaskRegistryDebugTest.DumpShowsTaskStateAndTransitions"
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
