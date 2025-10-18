#!/usr/bin/env python3
"""Minimal example that prints the Uni20 Python buildinfo dictionary."""

from __future__ import annotations

import pprint
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

    pprint.pp(uni20.buildinfo())


if __name__ == "__main__":
    main()
