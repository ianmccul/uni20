---
name: uni20-cmake-dependency-triage
description: Diagnose and fix Uni20 dependency detection/configuration failures with minimal, reproducible CMake changes.
---

# Uni20 CMake Dependency Triage

## Overview

Use this skill when configure fails, a dependency is unexpectedly fetched, a system dependency is rejected, or dependency diagnostics look wrong.

Keep recommendations minimal and explicit. Prefer fixing root-cause configuration over adding one-off workarounds.

## Workflow

1. Read `AGENTS.md`, then `.codex/instructions.local.md` (if present).
2. Identify the exact failing build tree and failing configure output.
3. Capture relevant cache/configuration variables.
4. Classify the failure mode (not found, found but incompatible version, policy mismatch, stale cache path, vendor mismatch).
5. Propose the smallest viable fix and the exact command to validate it.
6. Re-run configure; if needed, run targeted tests for affected modules.

## Data Collection

Use build-tree-local inspection first.

```bash
cmake -N -LAH ./build_codex | rg "UNI20_(USE_SYSTEM|DETECTED|FETCHCONTENT|BLAS)|BLA_VENDOR|_DIR|_FOUND|_VERSION"
```

```bash
rg -n "UNI20_DETECTED_|System .* found|requires >=|Fetching from|BLA_VENDOR|BLAS" ./build_codex/CMakeCache.txt ./build_codex/CMakeFiles/CMakeOutput.log ./build_codex/CMakeFiles/CMakeError.log
```

If package lookup behavior is unclear, re-run configure with find diagnostics:

```bash
cmake -S . -B ./build_codex --debug-find
```

## Common Failure Classes

- **Not found:** package truly unavailable in lookup paths.
- **Found but incompatible:** version exists but is below required minimum.
- **Policy mismatch:** `UNI20_USE_SYSTEM_<DEP>` set to `ON`/`OFF`/`AUTO` opposite to intent.
- **Stale cache path:** cached `<dep>_DIR` points to old build/fetch location.
- **BLAS vendor mismatch:** configured vendor and detected library/provider are inconsistent.

## Resolution Patterns

- Set explicit dependency policy:
  - `-DUNI20_USE_SYSTEM_<DEP>=AUTO`
  - `-DUNI20_USE_SYSTEM_<DEP>=ON`
  - `-DUNI20_USE_SYSTEM_<DEP>=OFF`
- Clear only stale cache entries in the affected build tree (avoid deleting unrelated state).
- For BLAS issues, align `UNI20_BLAS_VENDOR` and expected provider, then reconfigure.
- Prefer deterministic configuration over implicit fallback when debugging.

## Reporting Format

- `problem`: concise symptom and failing dependency
- `evidence`: relevant cache/log lines
- `classification`: failure class
- `fix`: exact CMake flag or cache cleanup action
- `validation`: configure result and any tests run
