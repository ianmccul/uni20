#!/usr/bin/env python3
"""Smoke tests for the Uni20 Python bindings."""

from __future__ import annotations

import importlib
import pathlib
import sys
import unittest


def _load_module() -> object:
    if len(sys.argv) < 2:
        raise unittest.SkipTest("Bindings output directory not provided")

    bindings_dir = pathlib.Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(bindings_dir))
    # Remove the injected argument so unittest does not attempt to parse it.
    sys.argv = [sys.argv[0]]
    return importlib.import_module("uni20_python")


_UNI20_MODULE = _load_module()


class GreetTests(unittest.TestCase):
    """Verify the sample greeting binding remains functional."""

    def test_greet_returns_expected_message(self) -> None:
        self.assertEqual(_UNI20_MODULE.greet(), "Hello from uni20!")


if __name__ == "__main__":
    unittest.main() 
