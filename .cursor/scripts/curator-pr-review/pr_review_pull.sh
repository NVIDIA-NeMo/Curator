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

# Fetch GitHub data for an NVIDIA-NeMo/Curator PR review.
#
# Usage: pr_review_pull.sh <PR_NUMBER> [--outdir DIR] [--repo OWNER/REPO]
#        [--path-regex REGEX] [--modality-label LABEL]
#
# When --path-regex is set, the script aborts (exit 3) if the PR touches no
# matching path. Modality skills pass their path filter; omit both flags for a
# modality-agnostic pull.
#
# Writes pr<N>_*_latest.json (consumed by build_digest.py) plus timestamped
# snapshots so a prior pull is preserved for delta analysis.
#
# Requires the GitHub CLI (`gh`) authenticated against github.com.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/gh_paginate.sh
source "${SCRIPT_DIR}/lib/gh_paginate.sh"

PR=""
REPO="NVIDIA-NeMo/Curator"
OUTDIR=".curator-pr-review"
PATH_REGEX=""
MODALITY_LABEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --outdir) OUTDIR="$2"; shift 2 ;;
        --repo)   REPO="$2";   shift 2 ;;
        --path-regex) PATH_REGEX="$2"; shift 2 ;;
        --modality-label) MODALITY_LABEL="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: pr_review_pull.sh <PR_NUMBER> [--outdir DIR] [--repo OWNER/REPO] [--path-regex REGEX] [--modality-label LABEL]"
            exit 0 ;;
        *) PR="$1"; shift ;;
    esac
done
[[ -n "${PR}" ]] || { echo "error: PR number required" >&2; exit 2; }

command -v gh >/dev/null || { echo "error: gh (GitHub CLI) not found" >&2; exit 2; }

mkdir -p "${OUTDIR}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${OUTDIR}/pr${PR}_review_pull_${TS}.log"

{
    echo "=== Pull start ${TS}  PR=${PR}  repo=${REPO} ==="
    gh --version | head -1
} | tee -a "${LOG}"

pull_endpoint() {
    local label="$1"; shift
    local outfile="$1"; shift
    echo "--- ${label} -> ${outfile} ---" | tee -a "${LOG}"
    "$@" > "${outfile}"
    printf 'bytes=%s\n\n' "$(stat -c%s "${outfile}" 2>/dev/null || wc -c < "${outfile}")" | tee -a "${LOG}"
}

GH_FIELDS="number,title,state,isDraft,mergeable,mergeStateStatus,headRefName,headRefOid,baseRefName,baseRefOid,additions,deletions,changedFiles,commits,reviewDecision,reviewRequests,labels,milestone,createdAt,updatedAt,closedAt,mergedAt,author,body,url,statusCheckRollup"

pull_endpoint "pr ${PR} metadata" "${OUTDIR}/pr${PR}_gh_${TS}.json" \
    gh pr view "${PR}" --repo "${REPO}" --json "${GH_FIELDS}"

FILES_JSON="${OUTDIR}/pr${PR}_files_${TS}.json"
LOG="${LOG}" pull_paginated_json "pulls/${PR}/files" "${FILES_JSON}" \
    "repos/${REPO}/pulls/${PR}/files"

if [[ -n "${PATH_REGEX}" ]]; then
    label="${MODALITY_LABEL:-modality}"
    if "${SCRIPT_DIR}/path_matches.py" "${FILES_JSON}" --regex "${PATH_REGEX}"; then
        :
    else
        status=$?
        if [[ ${status} -ne 1 ]]; then
            echo "error: path matching failed for PR ${PR}" >&2
            exit "${status}"
        fi
        echo "error: PR ${PR} touches no ${label} path; aborting." >&2
        exit 3
    fi
fi

LOG="${LOG}" pull_paginated_json "pulls/${PR}/reviews" "${OUTDIR}/pr${PR}_reviews_${TS}.json" \
    "repos/${REPO}/pulls/${PR}/reviews"
LOG="${LOG}" pull_paginated_json "pulls/${PR}/comments (inline)" "${OUTDIR}/pr${PR}_review_comments_${TS}.json" \
    "repos/${REPO}/pulls/${PR}/comments"
LOG="${LOG}" pull_paginated_json "issues/${PR}/comments (top-level)" "${OUTDIR}/pr${PR}_issue_comments_${TS}.json" \
    "repos/${REPO}/issues/${PR}/comments"
LOG="${LOG}" pull_paginated_json "pulls/${PR}/commits" "${OUTDIR}/pr${PR}_commits_${TS}.json" \
    "repos/${REPO}/pulls/${PR}/commits"

pull_endpoint "graphql reviewThreads" "${OUTDIR}/pr${PR}_review_threads_${TS}.json" \
    "${SCRIPT_DIR}/pull_review_threads.py" --repo "${REPO}" --pr "${PR}"

for kind in gh reviews review_comments issue_comments files commits review_threads; do
    cp -f "${OUTDIR}/pr${PR}_${kind}_${TS}.json" "${OUTDIR}/pr${PR}_${kind}_latest.json"
done

{
    echo "--- counts ---"
    for kind in reviews review_comments issue_comments files commits; do
        f="${OUTDIR}/pr${PR}_${kind}_${TS}.json"
        n=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(len(d) if isinstance(d,list) else 1)" "${f}")
        printf 'pr%s_%s: %s entries\n' "${PR}" "${kind}" "${n}"
    done
    echo "--- head SHA / activity ---"
    python3 - "${OUTDIR}/pr${PR}_gh_latest.json" <<'PY'
import json, sys
gh = json.loads(open(sys.argv[1]).read())
print(f"head_oid:   {gh.get('headRefOid')}")
print(f"base_oid:   {gh.get('baseRefOid')}")
print(f"state:      {gh.get('state')}  draft={gh.get('isDraft')}  mergeStatus={gh.get('mergeStateStatus')}  reviewDecision={gh.get('reviewDecision')}")
print(f"counts:     files={gh.get('changedFiles')} +{gh.get('additions')}/-{gh.get('deletions')}")
print(f"updated_at: {gh.get('updatedAt')}")
PY
} | tee -a "${LOG}"

echo "PR${PR}_REVIEW_PULL_DONE  outdir=${OUTDIR}  log=${LOG}" | tee -a "${LOG}"
