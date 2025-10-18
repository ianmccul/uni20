#!/usr/bin/env python3
"""Validate the Uni20 Python buildinfo binding."""

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
    return importlib.import_module("uni20")


_UNI20_MODULE = _load_module()


class BuildInfoTests(unittest.TestCase):
    """Exercise the buildinfo metadata surface."""

    def test_buildinfo_reports_metadata(self) -> None:
        info = _UNI20_MODULE.buildinfo()

        self.assertIsInstance(info, dict)

        expected_keys = {
            "generator",
            "build_type",
            "system_name",
            "system_version",
            "system_processor",
            "cxx_compiler_id",
            "cxx_compiler_version",
            "cxx_compiler_path",
        }

        self.assertTrue(expected_keys.issubset(info.keys()))
        for key in expected_keys:
            self.assertIsInstance(info[key], str)
            self.assertTrue(info[key])

        for collection in ("build_options", "detected_environment"):
            self.assertIn(collection, info)
            self.assertIsInstance(info[collection], dict)
            self.assertTrue(info[collection])

            for option_name, metadata in info[collection].items():
                self.assertIsInstance(option_name, str)
                self.assertIsInstance(metadata, dict)
                self.assertIn("value", metadata)
                self.assertIsInstance(metadata["value"], str)
                self.assertTrue(metadata["value"] or metadata.get("help"))
                if "help" in metadata:
                    self.assertIsInstance(metadata["help"], str)


if __name__ == "__main__":
    unittest.main()
