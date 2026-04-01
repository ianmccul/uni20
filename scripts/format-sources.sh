#!/usr/bin/env bash
# Format all tracked C and C++ sources in the repository with clang-format.
#
# Keeping this script repo-root aware makes it safe to run from any subdirectory.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

echo "Running clang-format on tracked C++ files..."

git ls-files -z '*.cpp' '*.hpp' '*.cc' '*.cxx' '*.h' \
  | xargs -0 clang-format -i

echo "Done."
