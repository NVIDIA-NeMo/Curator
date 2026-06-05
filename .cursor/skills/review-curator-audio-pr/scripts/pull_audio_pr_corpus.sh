#!/usr/bin/env bash
# Discover audio-modality PRs opened AFTER a baseline PR (#1608 by default),
# open or closed/merged, and pull each one's reviews + comments into a corpus
# directory. Consolidate with build_corpus.py.
#
# PR numbers are monotonic in time, so "number > --since" == "opened after that
# PR". We then keep only PRs whose changed files touch audio paths.
#
# Usage: pull_audio_pr_corpus.sh [--since N] [--outdir DIR] [--repo OWNER/REPO] [--limit N]
#
# Requires the GitHub CLI (`gh`) authenticated against github.com.
set -euo pipefail

REPO="NVIDIA-NeMo/Curator"
SINCE=1608
OUTDIR=".curator-pr-review/audio-corpus"
LIMIT=600

while [[ $# -gt 0 ]]; do
    case "$1" in
        --since)  SINCE="$2";  shift 2 ;;
        --outdir) OUTDIR="$2"; shift 2 ;;
        --repo)   REPO="$2";   shift 2 ;;
        --limit)  LIMIT="$2";  shift 2 ;;
        -h|--help)
            echo "Usage: pull_audio_pr_corpus.sh [--since N] [--outdir DIR] [--repo OWNER/REPO] [--limit N]"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

command -v gh >/dev/null || { echo "error: gh (GitHub CLI) not found" >&2; exit 2; }
mkdir -p "${OUTDIR}"

# Audio-path test (extended regex over a PR's changed-file list).
AUDIO_RE='^(nemo_curator/stages/audio/|nemo_curator/tasks/audio_task\.py|tutorials/audio/|tests/stages/audio/|tests/tasks/test_audio|benchmarking/.*([Aa]udio|ALM|alm))'

echo "=== corpus discovery: ${REPO} PRs > #${SINCE} (state=all, limit=${LIMIT}) ===" >&2
gh pr list --repo "${REPO}" --state all --limit "${LIMIT}" \
    --json number,title,state,author,createdAt,updatedAt,url \
    > "${OUTDIR}/_all_prs.json"

# Candidate PR numbers (> SINCE), newest first.
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
    files_json="${OUTDIR}/pr${n}_files.json"
    gh api --paginate "repos/${REPO}/pulls/${n}/files" --jq '[.[].filename]' \
        > "${files_json}" 2>/dev/null || { echo "  pr${n}: files fetch failed, skip" >&2; continue; }
    if AUDIO_RE="${AUDIO_RE}" python3 - "${files_json}" <<'PY'
import json, os, re, sys
files = json.load(open(sys.argv[1]))
rx = re.compile(os.environ["AUDIO_RE"])
sys.exit(0 if any(rx.search(f) for f in files) else 1)
PY
    then
        AUDIO_NUMS+=("${n}")
        echo "  pr${n}: AUDIO" >&2
    else
        rm -f "${files_json}"
    fi
done

echo "audio PRs after #${SINCE}: ${#AUDIO_NUMS[@]}" >&2
printf '%s\n' "${AUDIO_NUMS[@]}" > "${OUTDIR}/_audio_pr_numbers.txt"

for n in "${AUDIO_NUMS[@]}"; do
    echo "--- pulling pr${n} reviews/comments ---" >&2
    gh pr view "${n}" --repo "${REPO}" \
        --json number,title,state,author,createdAt,updatedAt,mergedAt,closedAt,url,body \
        > "${OUTDIR}/pr${n}_gh.json"
    gh api --paginate "repos/${REPO}/pulls/${n}/reviews"  > "${OUTDIR}/pr${n}_reviews.json"
    gh api --paginate "repos/${REPO}/pulls/${n}/comments" > "${OUTDIR}/pr${n}_review_comments.json"
    gh api --paginate "repos/${REPO}/issues/${n}/comments" > "${OUTDIR}/pr${n}_issue_comments.json"
done

echo "AUDIO_PR_CORPUS_PULL_DONE  outdir=${OUTDIR}  audio_prs=${#AUDIO_NUMS[@]}" >&2
