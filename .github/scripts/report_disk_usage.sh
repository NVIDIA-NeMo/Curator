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

set -euo pipefail

printf 'TMPDIR=%s\n' "${TMPDIR:-<unset; defaults to /tmp>}"
for path in / /tmp "${TMPDIR:-/tmp}" "${RUNNER_TEMP:-/tmp}" "${GITHUB_WORKSPACE:-$PWD}"; do
    echo "::group::Storage for $path"
    if [[ -d "$path" ]]; then
        df -h -- "$path" || echo "::warning::Could not read capacity for $path"
        df -i -- "$path" || echo "::warning::Could not read inode usage for $path"
        findmnt -T "$path" -o TARGET,SOURCE,FSTYPE,OPTIONS || echo "::warning::Could not read mounts for $path"
    else
        echo "::warning::Storage path does not exist: $path"
    fi
    echo "::endgroup::"
done

# Capacity and mount reports alone cannot establish applicable quota limits.
if command -v quota >/dev/null 2>&1; then
    quota -s || echo "::warning::Could not read user quotas"
else
    echo "quota is not installed; quota limits are not available in this report."
fi
