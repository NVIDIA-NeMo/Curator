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

# Audio-modality path filter shared by review-curator-audio-pr scripts.
# Fern paths qualify only when their path is explicitly audio-scoped. Generic
# navigation files (for example versions/main.yml) remain visible in a qualifying
# PR but do not make an unrelated Fern PR audio-specific by themselves.
AUDIO_PATH_REGEX='^(nemo_curator/stages/audio/|nemo_curator/tasks/audio_task\.py|tutorials/audio/|tests/stages/audio/|tests/tasks/test_audio|benchmarking/.*([Aa]udio|ALM|alm)|fern/versions/[^/]+/pages/(get-started/audio\.mdx|curate-audio/|about/concepts/audio/|api-reference/tasks/audio-task\.mdx))'
AUDIO_MODALITY_LABEL='audio'

# Return the one shared raw-corpus cache for this repository. Per-review output
# directories must never contain these files: a corpus cache contains raw
# comments and metadata for every in-scope audio PR, not only the PR currently
# being reviewed.
audio_corpus_cache_dir() {
    local repo="${1:-NVIDIA-NeMo/Curator}"
    local cache_root="${CURATOR_PR_REVIEW_CACHE_ROOT:-}"

    if [[ ! "${repo}" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]]; then
        echo "error: invalid GitHub repository name for corpus cache: ${repo}" >&2
        return 2
    fi

    if [[ -z "${cache_root}" ]]; then
        if [[ -n "${XDG_CACHE_HOME:-}" ]]; then
            cache_root="${XDG_CACHE_HOME}/nemo-curator-pr-review"
        elif [[ -n "${HOME:-}" ]]; then
            cache_root="${HOME}/.cache/nemo-curator-pr-review"
        else
            echo "error: set CURATOR_PR_REVIEW_CACHE_ROOT (or XDG_CACHE_HOME/HOME)" >&2
            return 2
        fi
    fi

    local repo_key="${repo//\//_}"
    printf '%s/%s/audio-corpus\n' "${cache_root%/}" "${repo_key}"
}

# Identify one corpus-selection scope. The PR data cache is shared, while each
# --since/path-filter combination keeps its own selection manifest.
audio_corpus_scope_key() {
    local since="${1:-1608}"

    if [[ ! "${since}" =~ ^[0-9]+$ ]]; then
        echo "error: --since must be a non-negative integer: ${since}" >&2
        return 2
    fi

    python3 - "${since}" "${AUDIO_PATH_REGEX}" <<'PY'
import hashlib
import sys

scope = f"since={sys.argv[1]}\0path_regex={sys.argv[2]}".encode()
print(f"since{sys.argv[1]}_{hashlib.sha256(scope).hexdigest()[:16]}")
PY
}

audio_corpus_numbers_file() {
    local scope_key
    scope_key="$(audio_corpus_scope_key "${1:-1608}")"
    printf '_audio_pr_numbers_%s.txt\n' "${scope_key}"
}
