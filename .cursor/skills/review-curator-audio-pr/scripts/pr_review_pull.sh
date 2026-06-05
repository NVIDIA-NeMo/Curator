#!/usr/bin/env bash
# Fetch GitHub data for an NVIDIA-NeMo/Curator PR review.
#
# Usage: pr_review_pull.sh <PR_NUMBER> [--outdir DIR] [--repo OWNER/REPO]
#
# Writes pr<N>_*_latest.json (consumed by build_digest.py) plus timestamped
# snapshots so a prior pull is preserved for delta analysis. Pulls six REST
# endpoints and the GraphQL review threads (which carry isResolved/isOutdated).
#
# Requires the GitHub CLI (`gh`) authenticated against github.com.
set -euo pipefail

PR=""
REPO="NVIDIA-NeMo/Curator"
OUTDIR=".curator-pr-review"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --outdir) OUTDIR="$2"; shift 2 ;;
        --repo)   REPO="$2";   shift 2 ;;
        -h|--help)
            echo "Usage: pr_review_pull.sh <PR_NUMBER> [--outdir DIR] [--repo OWNER/REPO]"; exit 0 ;;
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
pull_endpoint "pulls/${PR}/reviews" "${OUTDIR}/pr${PR}_reviews_${TS}.json" \
    gh api --paginate "repos/${REPO}/pulls/${PR}/reviews"
pull_endpoint "pulls/${PR}/comments (inline)" "${OUTDIR}/pr${PR}_review_comments_${TS}.json" \
    gh api --paginate "repos/${REPO}/pulls/${PR}/comments"
pull_endpoint "issues/${PR}/comments (top-level)" "${OUTDIR}/pr${PR}_issue_comments_${TS}.json" \
    gh api --paginate "repos/${REPO}/issues/${PR}/comments"
pull_endpoint "pulls/${PR}/files" "${OUTDIR}/pr${PR}_files_${TS}.json" \
    gh api --paginate "repos/${REPO}/pulls/${PR}/files"
pull_endpoint "pulls/${PR}/commits" "${OUTDIR}/pr${PR}_commits_${TS}.json" \
    gh api --paginate "repos/${REPO}/pulls/${PR}/commits"

OWNER="${REPO%%/*}"
NAME="${REPO##*/}"
pull_endpoint "graphql reviewThreads" "${OUTDIR}/pr${PR}_review_threads_${TS}.json" \
    gh api graphql \
      -f query='query($owner:String!,$repo:String!,$pr:Int!){ repository(owner:$owner,name:$repo){ pullRequest(number:$pr){ reviewThreads(first:100){ nodes{ id isResolved isOutdated isCollapsed line originalLine path comments(first:50){ nodes{ databaseId } } } } } } }' \
      -f owner="${OWNER}" -f repo="${NAME}" -F pr="${PR}"

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
