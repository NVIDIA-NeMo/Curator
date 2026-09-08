#!/usr/bin/env bash
# Generate the Python API reference locally, without Fern auth (requires Docker).
#
# Why the temporary docs.yml edit: the Fern CLI has two conflicting rules —
#   * `fern docs md generate --local` only accepts libraries with a `path` input
#     (git inputs are generated remotely and need auth), but
#   * `fern docs dev` / `fern generate --docs` refuse to parse a docs.yml that
#     contains any `path` input library ("'path' input which is not yet
#     supported. Please use 'git' input.") and render a blank page.
#
# So the path-based entry cannot live in the committed docs.yml. This script
# injects it, runs local generation, and restores docs.yml on exit.
set -euo pipefail

cd "$(dirname "$0")/.."

LIBRARY_NAME="nemo-curator-local"
SOURCE_PATH="../nemo_curator"
OUTPUT_PATH="./product-docs/nemo-curator/Full-Library-Reference"

if ! grep -q '^libraries:' docs.yml; then
  echo "error: no 'libraries:' block found in docs.yml" >&2
  exit 1
fi
if grep -q "^  ${LIBRARY_NAME}:" docs.yml; then
  echo "error: docs.yml already contains '${LIBRARY_NAME}' — remove it; a committed path-input entry breaks 'fern docs dev'" >&2
  exit 1
fi

cp docs.yml docs.yml.bak
trap 'mv docs.yml.bak docs.yml' EXIT

awk -v lib="$LIBRARY_NAME" -v src="$SOURCE_PATH" -v out="$OUTPUT_PATH" '
  1
  /^libraries:/ {
    print "  " lib ":"
    print "    input:"
    print "      path: " src
    print "    output:"
    print "      path: " out
    print "    lang: python"
  }
' docs.yml.bak > docs.yml

npx -y "fern-api@$(node -p 'require("./fern.config.json").version')" \
  docs md generate --local --library "$LIBRARY_NAME"
