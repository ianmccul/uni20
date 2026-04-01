#!/usr/bin/env python3
"""Find coroutine handles that never reached final suspend in a trace log.

Usage:
    scripts/find-handles.py coroutines.txt trace.txt
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


HANDLE_PATTERN = re.compile(r"coroutine_handle<[^>]+> @ (0x[0-9a-fA-F]+)")


def load_handles(file_path: Path) -> list[str]:
    """Load the expected coroutine handles, one hex address per line."""
    return [line.strip() for line in file_path.read_text().splitlines() if line.strip()]


def extract_final_suspend_handles(file_path: Path) -> set[str]:
    """Extract handles seen at 'Final suspend of coroutine' trace points."""
    handles: set[str] = set()
    for line in file_path.read_text().splitlines():
        if "Final suspend of coroutine" not in line:
            continue
        match = HANDLE_PATTERN.search(line)
        if match:
            handles.add(match.group(1))
    return handles


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"Usage: {argv[0]} coroutines.txt trace.txt")
        return 1

    coroutines_file = Path(argv[1])
    trace_file = Path(argv[2])

    all_handles = load_handles(coroutines_file)
    final_suspend_handles = extract_final_suspend_handles(trace_file)
    missing = [handle for handle in all_handles if handle not in final_suspend_handles]

    if missing:
        print("Handles missing 'Final suspend':")
        for handle in missing:
            print(handle)
    else:
        print("All coroutine handles have a 'Final suspend' entry.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
