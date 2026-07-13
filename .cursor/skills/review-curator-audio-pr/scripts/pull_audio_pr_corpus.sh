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

# Discover every post-#1608 audio PR and pull complete review conversations.
# Cached data is reused only while the PR's GitHub updatedAt value is unchanged.
#
# Usage: pull_audio_pr_corpus.sh [--since N] [--outdir DIR]
#        [--repo OWNER/REPO] [--refresh]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_DIR="${SCRIPT_DIR}/../../../scripts/curator-pr-review"
# shellcheck source=../audio_paths.sh
source "${SCRIPT_DIR}/../audio_paths.sh"
# shellcheck source=../../../scripts/curator-pr-review/lib/gh_paginate.sh
source "${SHARED_DIR}/lib/gh_paginate.sh"

REPO="NVIDIA-NeMo/Curator"
SINCE=1608
OUTDIR=".curator-pr-review/audio-corpus"
REFRESH=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --since)   SINCE="$2";  shift 2 ;;
        --outdir)  OUTDIR="$2"; shift 2 ;;
        --repo)    REPO="$2";   shift 2 ;;
        --refresh) REFRESH=1;    shift ;;
        -h|--help)
            echo "Usage: pull_audio_pr_corpus.sh [--since N] [--outdir DIR] [--repo OWNER/REPO] [--refresh]"
            exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

command -v gh >/dev/null || { echo "error: gh (GitHub CLI) not found" >&2; exit 2; }
mkdir -p "${OUTDIR}"

NONAUDIO_CACHE="${OUTDIR}/_non_audio_prs.tsv"
PATH_REGEX_CACHE="${OUTDIR}/_audio_path_regex.txt"
PATH_FILTER_CHANGED=1
if [[ -f "${PATH_REGEX_CACHE}" && "$(<"${PATH_REGEX_CACHE}")" == "${AUDIO_PATH_REGEX}" ]]; then
    PATH_FILTER_CHANGED=0
fi
touch "${NONAUDIO_CACHE}"
declare -A NONAUDIO_UPDATED=()
while IFS=$'\t' read -r number updated_at; do
    [[ -n "${number}" ]] && NONAUDIO_UPDATED["${number}"]="${updated_at:-}"
done < "${NONAUDIO_CACHE}"

echo "=== corpus discovery: all ${REPO} PRs > #${SINCE} (state=all) ===" >&2
LOG="" pull_paginated_json "all pull requests" "${OUTDIR}/_all_prs.json" \
    "repos/${REPO}/pulls?state=all&sort=created&direction=desc&per_page=100"

mapfile -t CANDIDATES < <(python3 - "${OUTDIR}/_all_prs.json" "${SINCE}" <<'PYDATA'
import json, sys
prs = json.load(open(sys.argv[1])); since = int(sys.argv[2])
rows = ((p["number"], p["updated_at"]) for p in prs if p["number"] > since)
for number, updated_at in sorted(rows, reverse=True):
    print(f"{number}\t{updated_at}")
PYDATA
)
echo "candidates after #${SINCE}: ${#CANDIDATES[@]}" >&2

declare -A UPDATED_AT=()
AUDIO_NUMS=()
for row in "${CANDIDATES[@]}"; do
    IFS=$'\t' read -r n updated_at <<< "${row}"
    UPDATED_AT["${n}"]="${updated_at}"
    cached_gh="${OUTDIR}/pr${n}_gh.json"
    cached_updated=""
    if [[ -f "${cached_gh}" ]]; then
        cached_updated="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("updatedAt", ""))' "${cached_gh}")"
    fi

    if [[ ${REFRESH} -eq 0 && ${PATH_FILTER_CHANGED} -eq 0 \
          && "${cached_updated}" == "${updated_at}" \
          && -f "${OUTDIR}/pr${n}_reviews.json" \
          && -f "${OUTDIR}/pr${n}_review_comments.json" \
          && -f "${OUTDIR}/pr${n}_issue_comments.json" ]]; then
        AUDIO_NUMS+=("${n}")
        echo "  pr${n}: AUDIO (cache current)" >&2
        continue
    fi
    if [[ ${REFRESH} -eq 0 && ${PATH_FILTER_CHANGED} -eq 0 \
          && "${NONAUDIO_UPDATED[$n]:-}" == "${updated_at}" ]]; then
        echo "  pr${n}: non-audio (cache current)" >&2
        continue
    fi

    files_json="${OUTDIR}/pr${n}_files.json"
    pull_paginated_json "pulls/${n}/files" "${files_json}" \
        "repos/${REPO}/pulls/${n}/files"
    if "${SHARED_DIR}/path_matches.py" "${files_json}" --regex "${AUDIO_PATH_REGEX}"; then
        AUDIO_NUMS+=("${n}")
        unset 'NONAUDIO_UPDATED[$n]'
        echo "  pr${n}: AUDIO" >&2
    else
        status=$?
        if [[ ${status} -ne 1 ]]; then
            echo "error: path matching failed for PR ${n}" >&2
            exit "${status}"
        fi
        rm -f "${files_json}"
        NONAUDIO_UPDATED["${n}"]="${updated_at}"
        echo "  pr${n}: non-audio" >&2
    fi
done

printf '%s' "${AUDIO_PATH_REGEX}" > "${PATH_REGEX_CACHE}"
: > "${NONAUDIO_CACHE}"
for n in "${!NONAUDIO_UPDATED[@]}"; do
    printf '%s\t%s\n' "${n}" "${NONAUDIO_UPDATED[$n]}"
done | sort -n > "${NONAUDIO_CACHE}"

printf '%s\n' "${AUDIO_NUMS[@]}" > "${OUTDIR}/_audio_pr_numbers.txt"
echo "audio PRs in scope: ${#AUDIO_NUMS[@]}" >&2

pulled=0
skipped=0
for n in "${AUDIO_NUMS[@]}"; do
    cached_updated=""
    if [[ -f "${OUTDIR}/pr${n}_gh.json" ]]; then
        cached_updated="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("updatedAt", ""))' "${OUTDIR}/pr${n}_gh.json")"
    fi
    if [[ ${REFRESH} -eq 0 && "${cached_updated}" == "${UPDATED_AT[$n]}" \
          && -f "${OUTDIR}/pr${n}_reviews.json" \
          && -f "${OUTDIR}/pr${n}_review_comments.json" \
          && -f "${OUTDIR}/pr${n}_issue_comments.json" ]]; then
        echo "--- pr${n}: cache current, skip ---" >&2
        skipped=$((skipped + 1))
        continue
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
