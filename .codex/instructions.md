# Codex Instructions (Repository-Wide)

These rules are intended for Codex agents working in this repository.
Detailed coding, style, coroutine safety, and documentation policy is defined in `AGENTS.md`.

## 1. Instruction Precedence

Before building or editing:

1. Read `AGENTS.md`.
2. Read `.codex/instructions.local.md` for machine/local overrides.

If there is a conflict, follow:

- user/developer/system prompt
- `.codex/instructions.local.md` for machine-specific constraints
- `AGENTS.md` for repository coding/build/testing policy
- this file

## 2. Build and Test Discipline

- Use out-of-source builds only.
- Prefer `./build_codex` (or subdirectories) for all configure/build/test work.
- Do not build in `./build`, `cmake-build-*`, or the source tree.
- Run tests after changes unless explicitly told not to.

## 3. Change Scope

- Keep changes focused and minimal.
- Do not touch unrelated code/files.
- Update documentation when behavior or APIs change.

## 4. Notes

Machine-specific compiler/toolchain guidance belongs in:

- `.codex/instructions.local.md`
