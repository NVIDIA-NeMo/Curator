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
# Usage: pull_audio_pr_corpus.sh [--since N] [--cache-dir DIR]
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
CACHE_DIR=""
REFRESH=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --since)     SINCE="$2";     shift 2 ;;
        --cache-dir) CACHE_DIR="$2"; shift 2 ;;
        --outdir)
            echo "error: corpus --outdir was replaced by --cache-dir so raw comments are not written into a per-review output directory" >&2
            exit 2 ;;
        --repo)      REPO="$2";      shift 2 ;;
        --refresh)   REFRESH=1;       shift ;;
        -h|--help)
            echo "Usage: pull_audio_pr_corpus.sh [--since N] [--cache-dir DIR] [--repo OWNER/REPO] [--refresh]"
            exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

command -v gh >/dev/null || { echo "error: gh (GitHub CLI) not found" >&2; exit 2; }
if [[ -z "${CACHE_DIR}" ]]; then
    CACHE_DIR="$(audio_corpus_cache_dir "${REPO}")"
fi
SCOPE_KEY="$(audio_corpus_scope_key "${SINCE}")"
NUMBERS_FILE="_audio_pr_numbers_${SCOPE_KEY}.txt"
NUMBERS_PATH="${CACHE_DIR}/${NUMBERS_FILE}"
NONAUDIO_CACHE="${CACHE_DIR}/_non_audio_prs_${SCOPE_KEY}.tsv"
mkdir -p "${CACHE_DIR}"

# The cache is shared across review directories and checkouts. Hold an
# exclusive advisory lock for the complete discovery/refresh transaction.
# build_corpus.py takes a shared lock on this same file while reading, so a
# review can never render a partially refreshed cache. Because the open file
# description belongs to this shell, the lock is released even after a signal.
LOCK_FILE="${CACHE_DIR}/.corpus.lock"
exec 9>>"${LOCK_FILE}"
python3 - <<'PYLOCK'
import fcntl

fcntl.flock(9, fcntl.LOCK_EX)
PYLOCK

TEMP_FILES=()
cleanup_temporary_files() {
    local temp_file
    for temp_file in "${TEMP_FILES[@]}"; do
        rm -f -- "${temp_file}"
    done
}
trap cleanup_temporary_files EXIT

atomic_pull_paginated_json() {
    local label="$1" target="$2" temp
    shift 2
    temp="${target}.tmp.$$"
    TEMP_FILES+=("${temp}")
    pull_paginated_json "${label}" "${temp}" "$@"
    mv -- "${temp}" "${target}"
}

cached_updated_at() {
    python3 - "$1" <<'PY'
import json
import sys

try:
    print(json.load(open(sys.argv[1], encoding="utf-8")).get("updatedAt", ""))
except (OSError, AttributeError, json.JSONDecodeError):
    print("")
PY
}

declare -A CACHED_AUDIO=()
if [[ -f "${NUMBERS_PATH}" ]]; then
    while read -r number; do
        [[ -n "${number}" ]] && CACHED_AUDIO["${number}"]=1
    done < "${NUMBERS_PATH}"
fi

declare -A NONAUDIO_UPDATED=()
if [[ -f "${NONAUDIO_CACHE}" ]]; then
    while IFS=$'\t' read -r number updated_at; do
        [[ -n "${number}" ]] && NONAUDIO_UPDATED["${number}"]="${updated_at:-}"
    done < "${NONAUDIO_CACHE}"
fi

echo "=== corpus discovery: all ${REPO} PRs > #${SINCE} (state=all) ===" >&2
LOG="" atomic_pull_paginated_json "all pull requests" "${CACHE_DIR}/_all_prs.json" \
    "repos/${REPO}/pulls?state=all&sort=created&direction=desc&per_page=100"

mapfile -t CANDIDATES < <(python3 - "${CACHE_DIR}/_all_prs.json" "${SINCE}" <<'PYDATA'
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
    cached_gh="${CACHE_DIR}/pr${n}_gh.json"
    cached_updated=""
    if [[ -f "${cached_gh}" ]]; then
        cached_updated="$(cached_updated_at "${cached_gh}")"
    fi

    if [[ ${REFRESH} -eq 0 && "${CACHED_AUDIO[$n]:-}" == 1 \
          && "${cached_updated}" == "${updated_at}" \
          && -f "${CACHE_DIR}/pr${n}_reviews.json" \
          && -f "${CACHE_DIR}/pr${n}_review_comments.json" \
          && -f "${CACHE_DIR}/pr${n}_issue_comments.json" ]]; then
        AUDIO_NUMS+=("${n}")
        echo "  pr${n}: AUDIO (cache current)" >&2
        continue
    fi
    if [[ ${REFRESH} -eq 0 \
          && "${NONAUDIO_UPDATED[$n]:-}" == "${updated_at}" ]]; then
        echo "  pr${n}: non-audio (cache current)" >&2
        continue
    fi

    files_json="${CACHE_DIR}/pr${n}_files.json"
    files_temp="${files_json}.tmp.$$"
    TEMP_FILES+=("${files_temp}")
    pull_paginated_json "pulls/${n}/files" "${files_temp}" \
        "repos/${REPO}/pulls/${n}/files"
    if "${SHARED_DIR}/path_matches.py" "${files_temp}" --regex "${AUDIO_PATH_REGEX}"; then
        mv -- "${files_temp}" "${files_json}"
        AUDIO_NUMS+=("${n}")
        unset 'NONAUDIO_UPDATED[$n]'
        echo "  pr${n}: AUDIO" >&2
    else
        status=$?
        if [[ ${status} -ne 1 ]]; then
            echo "error: path matching failed for PR ${n}" >&2
            exit "${status}"
        fi
        rm -f -- "${files_temp}"
        rm -f "${files_json}"
        NONAUDIO_UPDATED["${n}"]="${updated_at}"
        echo "  pr${n}: non-audio" >&2
    fi
done

echo "audio PRs in scope: ${#AUDIO_NUMS[@]}" >&2

pulled=0
skipped=0
for n in "${AUDIO_NUMS[@]}"; do
    cached_updated=""
    if [[ -f "${CACHE_DIR}/pr${n}_gh.json" ]]; then
        cached_updated="$(cached_updated_at "${CACHE_DIR}/pr${n}_gh.json")"
    fi
    if [[ ${REFRESH} -eq 0 && "${cached_updated}" == "${UPDATED_AT[$n]}" \
          && -f "${CACHE_DIR}/pr${n}_reviews.json" \
          && -f "${CACHE_DIR}/pr${n}_review_comments.json" \
          && -f "${CACHE_DIR}/pr${n}_issue_comments.json" ]]; then
        echo "--- pr${n}: cache current, skip ---" >&2
        skipped=$((skipped + 1))
        continue
    fi

    echo "--- pulling pr${n} reviews/comments ---" >&2
    gh_temp="${CACHE_DIR}/pr${n}_gh.json.tmp.$$"
    reviews_temp="${CACHE_DIR}/pr${n}_reviews.json.tmp.$$"
    review_comments_temp="${CACHE_DIR}/pr${n}_review_comments.json.tmp.$$"
    issue_comments_temp="${CACHE_DIR}/pr${n}_issue_comments.json.tmp.$$"
    TEMP_FILES+=("${gh_temp}" "${reviews_temp}" "${review_comments_temp}" "${issue_comments_temp}")

    gh pr view "${n}" --repo "${REPO}" \
        --json number,title,state,author,createdAt,updatedAt,mergedAt,closedAt,url,body \
        > "${gh_temp}"
    python3 - "${gh_temp}" <<'PYJSON'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as infile:
    payload = json.load(infile)
if not isinstance(payload, dict):
    raise TypeError("gh pr view did not return a JSON object")
PYJSON
    pull_paginated_json "pulls/${n}/reviews" "${reviews_temp}" \
        "repos/${REPO}/pulls/${n}/reviews"
    pull_paginated_json "pulls/${n}/comments" "${review_comments_temp}" \
        "repos/${REPO}/pulls/${n}/comments"
    pull_paginated_json "issues/${n}/comments" "${issue_comments_temp}" \
        "repos/${REPO}/issues/${n}/comments"

    # Publish the metadata marker last. If an interruption happens between
    # renames, its old updatedAt forces the next invocation to refresh again.
    mv -- "${reviews_temp}" "${CACHE_DIR}/pr${n}_reviews.json"
    mv -- "${review_comments_temp}" "${CACHE_DIR}/pr${n}_review_comments.json"
    mv -- "${issue_comments_temp}" "${CACHE_DIR}/pr${n}_issue_comments.json"
    mv -- "${gh_temp}" "${CACHE_DIR}/pr${n}_gh.json"
    pulled=$((pulled + 1))
done

# Publish scope state only after every selected PR has complete JSON. The
# numbers manifest is the commit marker consumed by build-corpus.
nonaudio_temp="${NONAUDIO_CACHE}.tmp.$$"
numbers_temp="${NUMBERS_PATH}.tmp.$$"
TEMP_FILES+=("${nonaudio_temp}" "${numbers_temp}")
for n in "${!NONAUDIO_UPDATED[@]}"; do
    printf '%s\t%s\n' "${n}" "${NONAUDIO_UPDATED[$n]}"
done | sort -n > "${nonaudio_temp}"
printf '%s\n' "${AUDIO_NUMS[@]}" > "${numbers_temp}"
mv -- "${nonaudio_temp}" "${NONAUDIO_CACHE}"
mv -- "${numbers_temp}" "${NUMBERS_PATH}"

echo "AUDIO_PR_CORPUS_PULL_DONE  cache_dir=${CACHE_DIR}  scope=${SCOPE_KEY}  audio_prs=${#AUDIO_NUMS[@]}  pulled=${pulled}  skipped=${skipped}  refresh=${REFRESH}" >&2
