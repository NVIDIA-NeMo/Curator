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

# Audio-specific entry point for the shared Curator PR review helpers.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_DIR="${SCRIPT_DIR}/../../../scripts/curator-pr-review"
# shellcheck source=../audio_paths.sh
source "${SCRIPT_DIR}/../audio_paths.sh"

usage() {
    cat <<'USAGE'
Usage: review_audio_pr.sh <command> [args]

Commands:
  ensure-repo [CLONE_DIR]
  pull <PR_NUMBER> [--outdir DIR] [--repo OWNER/REPO]
  digest <PR_NUMBER> [--outdir DIR] [--repo OWNER/REPO] [--today YYYY-MM-DD]
         [--prev-head SHA] [--baseline-ts TS]
  build-corpus [--cache-dir DIR] [--outdir DIR] [--repo OWNER/REPO] [--since N]
               [--today YYYY-MM-DD]
USAGE
}

[[ $# -gt 0 ]] || { usage >&2; exit 2; }
command="$1"
shift

case "${command}" in
    ensure-repo)
        exec "${SHARED_DIR}/ensure_repo.sh" "$@"
        ;;
    pull)
        exec "${SHARED_DIR}/pr_review_pull.sh" "$@" \
            --path-regex "${AUDIO_PATH_REGEX}" \
            --modality-label "${AUDIO_MODALITY_LABEL}"
        ;;
    digest)
        exec "${SHARED_DIR}/build_digest.py" "$@" \
            --path-regex "${AUDIO_PATH_REGEX}" \
            --modality-label "${AUDIO_MODALITY_LABEL}" \
            --area-rules "${SCRIPT_DIR}/../area_rules.json"
        ;;
    build-corpus)
        repo="NVIDIA-NeMo/Curator"
        since=1608
        cache_dir=""
        outdir=".curator-pr-review/audio-corpus"
        forwarded=()
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --cache-dir) cache_dir="$2"; shift 2 ;;
                --outdir)    outdir="$2";    shift 2 ;;
                --repo)      repo="$2";      shift 2 ;;
                --since)     since="$2";     shift 2 ;;
                *)           forwarded+=("$1"); shift ;;
            esac
        done
        if [[ -z "${cache_dir}" ]]; then
            cache_dir="$(audio_corpus_cache_dir "${repo}")"
        fi
        numbers_file="$(audio_corpus_numbers_file "${since}")"
        exec "${SHARED_DIR}/build_corpus.py" \
            --cache-dir "${cache_dir}" \
            --outdir "${outdir}" \
            --numbers-file "${numbers_file}" \
            --repo "${repo}" \
            --title "Audio PR review corpus (post-#${since})" \
            --intro "Consolidated reviewer feedback on audio PRs opened after #${since} (open + closed/merged). Read-only pre-review context: recognise patterns reviewers repeatedly raise, and check the PR in front of you against them." \
            --output-prefix "audio_pr_corpus" \
            "${forwarded[@]}"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "error: unknown command: ${command}" >&2
        usage >&2
        exit 2
        ;;
esac
