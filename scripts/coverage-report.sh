#!/usr/bin/env bash
# Generate an HTML coverage report with gcovr.
#
# Usage:
#   scripts/coverage-report.sh [build-dir]
#
# The default build directory is `build`.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
BUILD_DIR="${1:-build}"

cd "$REPO_ROOT"

gcovr \
  --root "$REPO_ROOT" \
  --object-directory "$BUILD_DIR" \
  --html \
  --html-details \
  --output "$BUILD_DIR/coverage.html" \
  --exclude '.*third_party.*' \
  --exclude '.*tests/.*' \
  --exclude '.*examples/.*' \
  --exclude '.*benchmarks/.*' \
  --gcov-ignore-parse-errors=negative_hits.warn_once_per_file \
  --verbose
