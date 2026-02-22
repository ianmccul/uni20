#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

echo "Running clang-format on tracked C++ files..."

git ls-files -z '*.cpp' '*.hpp' '*.cc' '*.cxx' '*.h' \
  | xargs -0 clang-format -i

echo "Done."

