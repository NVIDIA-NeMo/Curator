#!/bin/bash
# Diff v26.02 docs against a release branch to see what changed.
#
# Usage:
#   ./diff_from_release.sh [release_branch]
#
# Examples:
#   ./diff_from_release.sh main          # Diff against main
#   ./diff_from_release.sh origin/main   # Diff against remote main
#
# With no argument, diffs v25.09 vs v26.02 (shows changes between versions).

set -e

RELEASE_BRANCH="${1:-}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

if [ -z "$RELEASE_BRANCH" ]; then
    echo "=== Diff: v25.09 vs v26.02 ==="
    git diff --stat fern/v25.09 fern/v26.02
    echo ""
    echo "Full diff:"
    git diff fern/v25.09 fern/v26.02
else
    echo "=== Diff: $RELEASE_BRANCH:fern/ vs current fern/v26.02 ==="
    echo "(Comparing docs from $RELEASE_BRANCH to local v26.02)"
    echo ""
    # Show diff of fern/ between release branch and current, focusing on v26.02
    git diff "$RELEASE_BRANCH" -- fern/
fi
