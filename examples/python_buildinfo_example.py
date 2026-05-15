#!/usr/bin/env python3
"""Minimal example that prints Uni20 Python build information."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    """Load the bindings and print buildinfo.
To load the bindings,
export PYTHONPATH=~/path/to/uni20/build/bindings/python:$PYTHONPATH
or supply the path as a parameter."""
    if len(sys.argv) > 1:
        bindings_dir = Path(sys.argv[1]).resolve()
        sys.path.insert(0, str(bindings_dir))

    import uni20

    print(uni20.buildinfo_pretty())


if __name__ == "__main__":
    main()
