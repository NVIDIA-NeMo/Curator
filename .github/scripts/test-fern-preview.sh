#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
ROOT=$(git rev-parse --show-toplevel)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
export RUNNER_TEMP="$WORK/runner"
mkdir -p "$RUNNER_TEMP" "$WORK/pr-source/fern/versions/main/pages" "$WORK/pr-source/fern/components"
# Exercise the actual inline staging block, not a second implementation.
sed -n '/^      - name: Prepare preview docs$/,/^      - name: Setup Node.js$/p' \
  "$ROOT/.github/workflows/fern-docs-preview.yml" | sed '1,2d;$d;s/^          //' > "$WORK/stage.sh"
bash -n "$WORK/stage.sh"
cd "$WORK/pr-source"
git init -q
git config user.name 'Preview fixture'
git config user.email 'preview@example.invalid'
commit() { git -c commit.gpgsign=false commit -qm "$1"; }
printf 'old\n' > fern/versions/main/pages/page.mdx
printf 'deleted\n' > 'fern/versions/main/pages/deleted page.mdx'
printf 'navigation\n' > fern/versions/main.yml
ln -s main.yml fern/versions/latest.yml
printf 'trusted component\n' > fern/components/example.mdx
printf 'trusted config\n' > fern/docs.yml
printf 'trusted script\n' > fern/tool.sh
git add .
commit base
BASE=$(git rev-parse HEAD)
git update-ref refs/remotes/origin/main "$BASE"
printf 'updated\n' > fern/versions/main/pages/page.mdx
rm 'fern/versions/main/pages/deleted page.mdx'
printf 'new\n' > fern/versions/main/pages/new.mdx
printf 'PR config\n' > fern/docs.yml
printf 'PR component\n' > fern/components/example.mdx
printf 'PR script\n' > fern/tool.sh
printf '{"version":"file:untrusted"}\n' > fern/fern.config.json
printf '{"scripts":{"generate:library":"exit 1"}}\n' > fern/package.json
git add .
commit content
cd "$WORK"
mkdir trusted-source
git -C pr-source archive "$BASE" | tar -x -C trusted-source
printf 'newer main page\n' > trusted-source/fern/versions/main/pages/main-only.mdx
bash stage.sh
STAGED="$RUNNER_TEMP/fern-preview/fern"
test -L "$STAGED/versions/latest.yml"
test "$(readlink "$STAGED/versions/latest.yml")" = main.yml
grep -qx updated "$STAGED/versions/main/pages/page.mdx"
grep -qx new "$STAGED/versions/main/pages/new.mdx"
test ! -e "$STAGED/versions/main/pages/deleted page.mdx"
grep -qx 'newer main page' "$STAGED/versions/main/pages/main-only.mdx"
cmp trusted-source/fern/docs.yml "$STAGED/docs.yml"
cmp trusted-source/fern/components/example.mdx "$STAGED/components/example.mdx"
cmp trusted-source/fern/tool.sh "$STAGED/tool.sh"
test ! -e "$STAGED/package.json"
grep -qx '{"organization":"nvidia","version":"5.123.0"}' "$STAGED/fern.config.json"
reject_link() {
  rm -rf "$RUNNER_TEMP/fern-preview"
  if bash stage.sh; then
    echo 'Unexpectedly accepted a PR symlink' >&2
    exit 1
  fi
}
ln -s /tmp pr-source/fern/versions/main/pages/new-link
reject_link
rm pr-source/fern/versions/main/pages/new-link
rm pr-source/fern/versions/latest.yml
ln -s /tmp pr-source/fern/versions/latest.yml
reject_link
rm pr-source/fern/versions/latest.yml
ln -s main.yml pr-source/fern/versions/latest.yml
mv pr-source/fern pr-source/content
ln -s content pr-source/fern
reject_link
rm pr-source/fern
mv pr-source/content pr-source/fern
printf 'Preview staging regressions passed\n'
