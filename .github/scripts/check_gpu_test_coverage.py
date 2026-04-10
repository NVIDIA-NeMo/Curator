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

"""Verify that all test files with @pytest.mark.gpu are covered by gpu_test_groups.json."""

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TESTS_DIR = REPO_ROOT / "tests"
GPU_GROUPS_FILE = TESTS_DIR / "gpu_test_groups.json"
GPU_MARKER = re.compile(r"@pytest\.mark\.gpu|\.gpu\b.*pytest\.mark")


def get_gpu_test_files():
    """Find all test files containing @pytest.mark.gpu."""
    gpu_files = []
    for py_file in TESTS_DIR.rglob("*.py"):
        if py_file.name.startswith("test_"):
            text = py_file.read_text()
            if GPU_MARKER.search(text):
                gpu_files.append(py_file.relative_to(REPO_ROOT))
    return sorted(gpu_files)


def get_covered_paths():
    """Load paths from gpu_test_groups.json and resolve them."""
    with open(GPU_GROUPS_FILE) as f:
        groups = json.load(f)

    covered = []
    for group in groups.values():
        for p in group["paths"]:
            covered.append(Path(p))
    return covered


def is_covered(test_file, covered_paths):
    """Check if a test file falls under any covered path."""
    return any(test_file == cp or cp in test_file.parents for cp in covered_paths)


def main():
    gpu_files = get_gpu_test_files()
    covered_paths = get_covered_paths()
    uncovered = [f for f in gpu_files if not is_covered(f, covered_paths)]

    if uncovered:
        print("ERROR: The following GPU test files are not covered by gpu_test_groups.json:")
        for f in uncovered:
            print(f"  - {f}")
        print()
        print("Add the missing test paths to tests/gpu_test_groups.json under an existing or new group.")
        return 1

    print(f"OK: All {len(gpu_files)} GPU test files are covered by gpu_test_groups.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
