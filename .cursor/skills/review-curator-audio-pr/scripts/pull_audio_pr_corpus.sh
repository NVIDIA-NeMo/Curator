#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Discover audio-modality PRs opened after the AudioTask framework redesign
# (#1608 by default), open or closed/merged, and pull each one's reviews +
# comments into a corpus directory. Consolidate with build_corpus.py.
#
# Incremental by default: PRs already pulled into OUTDIR are skipped.
#
# Usage: pull_audio_pr_corpus.sh [--since N] [--outdir DIR] [--repo OWNER/REPO] [--limit N] [--refresh]
#
# Requires the GitHub CLI (`gh`) authenticated against github.com.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../audio_paths.sh
source "${SCRIPT_DIR}/../audio_paths.sh"
# shellcheck source=../../review-curator-pr/scripts/lib/gh_paginate.sh
source "${SCRIPT_DIR}/../../review-curator-pr/scripts/lib/gh_paginate.sh"

REPO="NVIDIA-NeMo/Curator"
SINCE=1608
OUTDIR=".curator-pr-review/audio-corpus"
LIMIT=600
REFRESH=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --since)   SINCE="$2";  shift 2 ;;
        --outdir)  OUTDIR="$2"; shift 2 ;;
        --repo)    REPO="$2";   shift 2 ;;
        --limit)   LIMIT="$2";  shift 2 ;;
        --refresh) REFRESH=1;   shift ;;
        -h|--help)
            echo "Usage: pull_audio_pr_corpus.sh [--since N] [--outdir DIR] [--repo OWNER/REPO] [--limit N] [--refresh]"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

command -v gh >/dev/null || { echo "error: gh (GitHub CLI) not found" >&2; exit 2; }
mkdir -p "${OUTDIR}"

NONAUDIO_CACHE="${OUTDIR}/_non_audio_prs.txt"
touch "${NONAUDIO_CACHE}"
declare -A IS_NONAUDIO=()
while IFS= read -r x; do [[ -n "${x}" ]] && IS_NONAUDIO["${x}"]=1; done < "${NONAUDIO_CACHE}"

echo "=== corpus discovery: ${REPO} PRs > #${SINCE} (state=all, limit=${LIMIT}) ===" >&2
gh pr list --repo "${REPO}" --state all --limit "${LIMIT}" \
    --json number,title,state,author,createdAt,updatedAt,url \
    > "${OUTDIR}/_all_prs.json"

mapfile -t CANDIDATES < <(python3 - "${OUTDIR}/_all_prs.json" "${SINCE}" <<'PY'
import json, sys
prs = json.load(open(sys.argv[1])); since = int(sys.argv[2])
nums = sorted((p["number"] for p in prs if p["number"] > since), reverse=True)
print("\n".join(str(n) for n in nums))
PY
)
echo "candidates after #${SINCE}: ${#CANDIDATES[@]}" >&2

AUDIO_NUMS=()
for n in "${CANDIDATES[@]}"; do
    if [[ ${REFRESH} -eq 0 ]]; then
        if [[ -f "${OUTDIR}/pr${n}_gh.json" ]]; then
            AUDIO_NUMS+=("${n}"); echo "  pr${n}: AUDIO (on disk, skip)" >&2; continue
        fi
        if [[ -n "${IS_NONAUDIO[$n]:-}" ]]; then
            echo "  pr${n}: non-audio (cached, skip)" >&2; continue
        fi
    fi
    files_json="${OUTDIR}/pr${n}_files.json"
    if ! pull_paginated_json "pulls/${n}/files" "${files_json}" \
        "repos/${REPO}/pulls/${n}/files" 2>/dev/null; then
        echo "  pr${n}: files fetch failed, skip" >&2; continue
    fi
    if AUDIO_PATH_REGEX="${AUDIO_PATH_REGEX}" python3 - "${files_json}" <<'PY'
import json, os, re, sys
files = [f.get("filename", "") for f in json.load(open(sys.argv[1]))]
rx = re.compile(os.environ["AUDIO_PATH_REGEX"])
sys.exit(0 if any(rx.search(f) for f in files) else 1)
PY
    then
        AUDIO_NUMS+=("${n}")
        echo "  pr${n}: AUDIO" >&2
    else
        rm -f "${files_json}"
        if [[ -z "${IS_NONAUDIO[$n]:-}" ]]; then
            echo "${n}" >> "${NONAUDIO_CACHE}"; IS_NONAUDIO["${n}"]=1
        fi
    fi
done

echo "audio PRs in scope: ${#AUDIO_NUMS[@]}" >&2
printf '%s\n' "${AUDIO_NUMS[@]}" > "${OUTDIR}/_audio_pr_numbers.txt"

pulled=0
skipped=0
for n in "${AUDIO_NUMS[@]}"; do
    if [[ ${REFRESH} -eq 0 && -f "${OUTDIR}/pr${n}_gh.json" \
          && -f "${OUTDIR}/pr${n}_reviews.json" \
          && -f "${OUTDIR}/pr${n}_review_comments.json" \
          && -f "${OUTDIR}/pr${n}_issue_comments.json" ]]; then
        echo "--- pr${n}: already on disk, skip (use --refresh to re-pull) ---" >&2
        skipped=$((skipped + 1)); continue
    fi
    echo "--- pulling pr${n} reviews/comments ---" >&2
    gh pr view "${n}" --repo "${REPO}" \
        --json number,title,state,author,createdAt,updatedAt,mergedAt,closedAt,url,body \
        > "${OUTDIR}/pr${n}_gh.json"
    pull_paginated_json "pulls/${n}/reviews" "${OUTDIR}/pr${n}_reviews.json" \
        "repos/${REPO}/pulls/${n}/reviews"
    pull_paginated_json "pulls/${n}/comments" "${OUTDIR}/pr${n}_review_comments.json" \
        "repos/${REPO}/pulls/${n}/comments"
    pull_paginated_json "issues/${n}/comments" "${OUTDIR}/pr${n}_issue_comments.json" \
        "repos/${REPO}/issues/${n}/comments"
    pulled=$((pulled + 1))
done

echo "AUDIO_PR_CORPUS_PULL_DONE  outdir=${OUTDIR}  audio_prs=${#AUDIO_NUMS[@]}  pulled=${pulled}  skipped=${skipped}  refresh=${REFRESH}" >&2
