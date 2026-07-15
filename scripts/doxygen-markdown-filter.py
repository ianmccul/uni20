#!/usr/bin/env python3
"""Resolve repository directory links to README pages for Doxygen."""

from __future__ import annotations

from pathlib import Path
import re
import sys


FENCE = re.compile(r"(```+|~~~+)")
MARKDOWN_LINK = re.compile(r"(\[[^\]]*\]\()([^)]+)(\))")


def rewrite_target(raw: str, source: Path) -> str:
    stripped = raw.strip()
    leading = raw[: len(raw) - len(raw.lstrip())]
    trailing = raw[len(raw.rstrip()) :]

    if stripped.startswith("<") and ">" in stripped:
        close = stripped.index(">")
        target = stripped[1:close]
        remainder = stripped[close + 1 :]
        angle_brackets = True
    else:
        parts = stripped.split(maxsplit=1)
        target = parts[0]
        remainder = f" {parts[1]}" if len(parts) == 2 else ""
        angle_brackets = False

    if not target or target.startswith(
        ("/", "http://", "https://", "mailto:", "data:")
    ):
        return raw

    match = re.fullmatch(r"([^?#]*)(.*)", target)
    if match is None:
        return raw
    path_part, suffix = match.groups()
    if not path_part.endswith("/"):
        return raw

    directory = Path(source.parent / path_part)
    if not (directory / "README.md").is_file():
        return raw

    rewritten = f"{path_part}README.md{suffix}"
    if angle_brackets:
        rewritten = f"<{rewritten}>{remainder}"
    else:
        rewritten = f"{rewritten}{remainder}"
    return f"{leading}{rewritten}{trailing}"


def filter_markdown(source: Path) -> str:
    output: list[str] = []
    in_fence = False
    fence_character: str | None = None

    for line in source.read_text().splitlines(keepends=True):
        marker = FENCE.match(line.lstrip())
        if marker:
            character = marker.group(1)[0]
            if not in_fence:
                in_fence = True
                fence_character = character
            elif character == fence_character:
                in_fence = False
                fence_character = None
            output.append(line)
            continue

        if in_fence:
            output.append(line)
            continue

        output.append(
            MARKDOWN_LINK.sub(
                lambda match: (
                    f"{match.group(1)}"
                    f"{rewrite_target(match.group(2), source)}"
                    f"{match.group(3)}"
                ),
                line,
            )
        )

    return "".join(output)


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} MARKDOWN_FILE", file=sys.stderr)
        return 2
    sys.stdout.write(filter_markdown(Path(sys.argv[1]).resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
