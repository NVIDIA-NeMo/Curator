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

# Accumulate paginated `gh api` pages into one JSON array.
# Sourced by other review scripts; do not execute directly.
#
# Usage (after sourcing):
#   pull_paginated_json "label" outfile.json gh api args...

pull_paginated_json() {
    local label="$1" outfile="$2"; shift 2
    if [[ -n "${LOG:-}" ]]; then
        echo "--- ${label} -> ${outfile} ---" | tee -a "${LOG}"
    else
        echo "--- ${label} -> ${outfile} ---" >&2
    fi
    gh api --paginate --jq '.[]' "$@" \
        | python3 -c "import json,sys; print(json.dumps(list(map(json.loads, sys.stdin))))" \
        > "${outfile}"
    if [[ -n "${LOG:-}" ]]; then
        printf 'bytes=%s\n\n' "$(stat -c%s "${outfile}" 2>/dev/null || wc -c < "${outfile}")" | tee -a "${LOG}"
    fi
}
