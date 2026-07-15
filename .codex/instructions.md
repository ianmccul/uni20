# Codex Instructions (Repository-Wide)

These rules are intended for Codex agents working in this repository.
Detailed coding, style, coroutine safety, and documentation policy is defined in `AGENTS.md`.

## 1. Instruction Precedence

Before building or editing:

1. Read `AGENTS.md`.
2. If it exists, read `.codex/instructions.local.md` for machine-local
   overrides.

If there is a conflict, follow:

- user/developer/system prompt
- `.codex/instructions.local.md` for machine-specific constraints
- `AGENTS.md` for repository coding/build/testing policy
- this file

## 2. Build and Test Discipline

- Use out-of-source builds only.
- Select the build root from `.codex/instructions.local.md` when present;
  otherwise use `CMakeUserPresets.json` when provided or choose a git-ignored
  out-of-source directory.
- Keep separate build trees for different compilers and incompatible
  configurations.
- Do not build in the source tree.
- Run tests after changes unless explicitly told not to.

## 3. Change Scope

- Keep changes focused and minimal.
- Do not touch unrelated code/files.
- Update documentation when behavior or APIs change.
- Use `uni20::complex<T>` for complex scalar types in Uni20 code, tests,
  examples, and docs. Direct `std::complex<T>` spellings are reserved for the
  alias definition, explicit alias tests/docs, or narrow external interop
  boundaries; see `docs/scalar_policy.md`.
- Use `uni20::numeric_limits<T>` in scalar-generic Uni20 algorithms so
  extension and library scalar types can supply limits without specializing
  standard-library templates.

## 4. Notes

Machine-specific compiler/toolchain guidance belongs in:

- `.codex/instructions.local.md`, which is git-ignored.
