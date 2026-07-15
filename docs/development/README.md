# Developer Documentation

This directory contains repository process, build metadata, test, review, and
documentation tooling guides.

- [Testing](testing.md)
- [Reviewing Changes](code_review.md)
- [Agent-Assisted Development](agent_assisted_development.md)
- [Doxygen](doxygen.md)
- [Build Information](build_information.md)
- [Using Git and VS Code (Traditional Chinese)](git_and_vscode_zh_tw.md)

Run `scripts/check-docs.py` after adding, moving, or renaming documentation. It
checks the topic hierarchy, subsystem indexes, local links, code fences,
headings, and repository-root `docs/...` references.

Repository-wide coding and correctness rules remain in
[`AGENTS.md`](../../AGENTS.md). Machine-specific build trees and compiler paths
belong in untracked local instructions, not these portable guides.
