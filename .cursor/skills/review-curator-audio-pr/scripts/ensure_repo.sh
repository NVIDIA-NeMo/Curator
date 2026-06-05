#!/usr/bin/env bash
# Make a local NVIDIA-NeMo/Curator checkout available WITHOUT cloning the whole
# repo unnecessarily.
#
# Strategy (cheapest first):
#   1. If we are already inside a Curator checkout, use it.
#   2. Else search the current directory tree (bounded depth) for an existing
#      Curator checkout and reuse it.
#   3. Else shallow-clone (--depth 1, no full history) into ./Curator.
#
# Prints diagnostics to stderr and, on the LAST stdout line:
#   CURATOR_REPO=<absolute path to the checkout>
#
# Usage: ensure_repo.sh [CLONE_DIR]   (default CLONE_DIR=Curator)
#   MAXDEPTH=<n>   bound the downward search (default 4)
#
# Clone source: https://github.com/NVIDIA-NeMo/Curator
set -euo pipefail

REPO_URL="https://github.com/NVIDIA-NeMo/Curator.git"
CLONE_DIR="${1:-Curator}"
MAXDEPTH="${MAXDEPTH:-4}"
MARKER="nemo_curator/stages/audio/README.md"

is_curator() { [[ -f "$1/${MARKER}" && -d "$1/.cursor/rules" ]]; }
abspath()    { (cd "$1" && pwd); }

# 1) Already inside a Curator checkout?
if top="$(git rev-parse --show-toplevel 2>/dev/null)" && is_curator "${top}"; then
    echo "ensure_repo: already inside Curator checkout (${top}); not cloning." >&2
    echo "CURATOR_REPO=${top}"; exit 0
fi
if is_curator "."; then
    echo "ensure_repo: current directory is a Curator checkout; not cloning." >&2
    echo "CURATOR_REPO=$(abspath .)"; exit 0
fi

# 2) Existing checkout somewhere under the current tree?
while IFS= read -r marker; do
    cand="${marker%/${MARKER}}"
    if is_curator "${cand}"; then
        echo "ensure_repo: reusing existing checkout at ${cand}; not cloning." >&2
        echo "CURATOR_REPO=$(abspath "${cand}")"; exit 0
    fi
done < <(find . -maxdepth "${MAXDEPTH}" -path "*/${MARKER}" 2>/dev/null)

# 3) Reuse a prior clone dir if present, else shallow-clone.
if is_curator "${CLONE_DIR}"; then
    echo "ensure_repo: reusing ${CLONE_DIR}; not cloning." >&2
    echo "CURATOR_REPO=$(abspath "${CLONE_DIR}")"; exit 0
fi

command -v git >/dev/null || { echo "ensure_repo: git not found" >&2; exit 2; }
echo "ensure_repo: no local checkout found; shallow-cloning ${REPO_URL} -> ${CLONE_DIR}" >&2
git clone --depth 1 "${REPO_URL}" "${CLONE_DIR}"
is_curator "${CLONE_DIR}" || { echo "ensure_repo: clone did not produce a valid checkout" >&2; exit 1; }
echo "CURATOR_REPO=$(abspath "${CLONE_DIR}")"
