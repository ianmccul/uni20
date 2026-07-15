#!/usr/bin/env python3
"""Validate Uni20's Markdown hierarchy and local documentation references."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
DOCS = REPOSITORY / "docs"
SOURCE = REPOSITORY / "src" / "uni20"
EXAMPLES = REPOSITORY / "examples"

TOP_LEVEL_MARKDOWN = {
    "README.md",
    "about.md",
    "CONTRIBUTING.md",
    "getting_started.md",
    "mainpage.md",
    "roadmap.md",
}

LEGACY_ASYNC_DOCUMENTS = {
    "docs/Async.md",
    "docs/DebugScheduler.md",
    "docs/Schedulers.md",
    "docs/TbbScheduler.md",
    "docs/Epoch.md",
    "docs/async.md",
    "docs/async_api.md",
    "docs/async_design.md",
    "docs/async_new.md",
}
LEGACY_ASYNC_INDEXES = {
    Path("docs/async/README.md"),
    Path("docs/async/audit_legacy_docs.md"),
}

MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
ROOT_DOC_REFERENCE = re.compile(r"(?<![A-Za-z0-9_./-])(docs/[A-Za-z0-9_./-]+\.md)")
FENCE = re.compile(r"(```+|~~~+)")


def repository_relative(path: Path) -> Path:
    return path.relative_to(REPOSITORY)


def markdown_files() -> list[Path]:
    return sorted(DOCS.rglob("*.md"))


def source_readmes() -> list[Path]:
    return sorted(SOURCE.rglob("README.md"))


def example_readmes() -> list[Path]:
    return sorted(EXAMPLES.rglob("README.md"))


def tracked_existing_files() -> list[Path]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=REPOSITORY)
    paths = [REPOSITORY / value for value in output.decode().split("\0") if value]
    return [path for path in paths if path.is_file()]


def tracked_directories(root: Path) -> set[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z", "--", repository_relative(root).as_posix()],
        cwd=REPOSITORY,
    )
    directories: set[Path] = set()
    for value in output.decode().split("\0"):
        if not value:
            continue
        directory = (REPOSITORY / value).parent
        while directory == root or root in directory.parents:
            directories.add(directory)
            if directory == root:
                break
            directory = directory.parent
    return directories


def split_link_target(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("<") and ">" in raw:
        return raw[1 : raw.index(">")]
    return raw.split(maxsplit=1)[0]


def validate_markdown(path: Path, problems: list[str]) -> None:
    in_fence = False
    fence_character: str | None = None

    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        stripped = line.lstrip()
        marker = FENCE.match(stripped)
        if marker:
            character = marker.group(1)[0]
            if not in_fence:
                in_fence = True
                fence_character = character
            elif character == fence_character:
                in_fence = False
                fence_character = None
            continue

        if in_fence:
            continue

        if re.match(r"^\s+#{1,6}(?!#)(?:\s|\S)", line):
            problems.append(f"{repository_relative(path)}:{line_number}: indented heading")
        elif re.match(r"^#{1,6}(?!#)\S", line):
            problems.append(f"{repository_relative(path)}:{line_number}: heading is missing a space")
        elif re.match(r"^#{1,6}\s.*\s\{#[^}]+\}\s*$", line):
            problems.append(
                f"{repository_relative(path)}:{line_number}: "
                "Doxygen-style heading attributes are not portable Markdown"
            )

        for match in MARKDOWN_LINK.finditer(line):
            target = split_link_target(match.group(1))
            path_part = target.partition("#")[0].partition("?")[0]
            if not path_part or path_part.startswith(("http://", "https://", "mailto:", "data:")):
                continue
            if path_part.startswith("/"):
                resolved = REPOSITORY / path_part.removeprefix("/")
            else:
                resolved = Path(os.path.normpath(path.parent / path_part))
            if not resolved.exists():
                problems.append(
                    f"{repository_relative(path)}:{line_number}: missing link target {target}"
                )

    if in_fence:
        problems.append(f"{repository_relative(path)}: unclosed code fence")


def validate_doc_indexes(problems: list[str]) -> None:
    for directory in sorted(path for path in DOCS.rglob("*") if path.is_dir()):
        documents = sorted(directory.glob("*.md"))
        child_indexes = sorted(directory.glob("*/README.md"))
        if not documents and not child_indexes:
            continue

        index = directory / "README.md"
        if not index.exists():
            problems.append(f"{repository_relative(directory)}: missing README.md")
            continue

        content = index.read_text()
        for document in documents:
            if document.name != "README.md" and document.name not in content:
                problems.append(
                    f"{repository_relative(document)}: not named in {repository_relative(index)}"
                )
        for child_index in child_indexes:
            relative = child_index.relative_to(directory).as_posix()
            if relative not in content:
                problems.append(
                    f"{repository_relative(child_index)}: not named in {repository_relative(index)}"
                )


def validate_directory_indexes(
    root: Path, readmes: list[Path], kind: str, problems: list[str]
) -> None:
    directories = tracked_directories(root)
    readme_set = set(readmes)

    for directory in sorted(directories):
        index = directory / "README.md"
        if index not in readme_set:
            problems.append(f"{repository_relative(directory)}: missing README.md")
            continue

        content = index.read_text()
        for child in sorted(candidate for candidate in directories if candidate.parent == directory):
            child_index = child / "README.md"
            relative = child_index.relative_to(directory).as_posix()
            if relative not in content:
                problems.append(
                    f"{repository_relative(child_index)}: not named in {repository_relative(index)}"
                )

        has_docs_link = False
        for match in MARKDOWN_LINK.finditer(content):
            target = split_link_target(match.group(1)).partition("#")[0].partition("?")[0]
            if not target or target.startswith(("http://", "https://", "mailto:", "data:")):
                continue
            resolved = Path(os.path.normpath(index.parent / target))
            if resolved == DOCS or DOCS in resolved.parents:
                has_docs_link = True
                break
        if not has_docs_link:
            problems.append(
                f"{repository_relative(index)}: no link to relevant docs from {kind} index"
            )


def validate_root_references(paths: list[Path], problems: list[str]) -> None:
    for path in paths:
        if path == Path(__file__).resolve():
            continue
        try:
            content = path.read_text()
        except UnicodeDecodeError:
            continue
        source = repository_relative(path)
        for reference in ROOT_DOC_REFERENCE.findall(content):
            if source in LEGACY_ASYNC_INDEXES and reference in LEGACY_ASYNC_DOCUMENTS:
                continue
            if not (REPOSITORY / reference).exists():
                problems.append(f"{source}: missing repository-root reference {reference}")


def main() -> int:
    problems: list[str] = []
    documents = markdown_files()
    source_indexes = source_readmes()
    example_indexes = example_readmes()

    top_level = {path.name for path in DOCS.glob("*.md")}
    unexpected = sorted(top_level - TOP_LEVEL_MARKDOWN)
    for name in unexpected:
        problems.append(f"docs/{name}: unclassified top-level Markdown document")

    for path in documents + source_indexes + example_indexes:
        validate_markdown(path, problems)
    validate_doc_indexes(problems)
    validate_directory_indexes(SOURCE, source_indexes, "source", problems)
    validate_directory_indexes(EXAMPLES, example_indexes, "example", problems)

    root_reference_paths = set(tracked_existing_files()) | set(documents)
    validate_root_references(sorted(root_reference_paths), problems)

    if problems:
        print("Documentation validation failed:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print(
        f"Documentation validation passed: {len(documents)} Markdown files, "
        f"{len(source_indexes)} source indexes, {len(example_indexes)} example "
        "indexes, local links, hierarchy, and repository-root references checked."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
