# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Replace documentation variables in MDX files (CI and local).

Run before ``fern generate`` to substitute ``{{ variable }}`` in MDX.

Usage (from repo root)::

    python fern/substitute_variables.py versions/v25.09 --version 25.09
    python fern/substitute_variables.py versions/v26.02 --version 26.02
"""

import argparse
import re
from pathlib import Path

# Documentation variables - single source of truth (version overridden via --version)
DEFAULT_VARIABLES = {
    "product_name": "NeMo Curator",
    "product_name_short": "Curator",
    "company": "NVIDIA",
    "version": "25.09",
    "container_version": "25.09",
    "current_year": "2025",
    "github_repo": "https://github.com/NVIDIA-NeMo/Curator",
    "docs_url": "https://docs.nvidia.com/nemo-curator",
    "support_email": "nemo-curator-support@nvidia.com",
    "min_python_version": "3.10",
    "recommended_cuda": "12.0+",
    "current_release": "25.09",
}


def substitute_variables(content: str, variables: dict) -> str:
    """Replace {{ variable }} patterns with their values."""
    for var, value in variables.items():
        # Handle both {{ var }} and {{var}} patterns
        content = re.sub(rf"{{\{{\s*{var}\s*}}}}", value, content)
    return content


def process_file(filepath: Path, variables: dict, dry_run: bool = False) -> bool:
    """Process a single MDX file. Returns True if file was modified."""
    content = filepath.read_text()
    updated = substitute_variables(content, variables)

    if content != updated:
        if dry_run:
            print(f"Would update: {filepath}")
        else:
            filepath.write_text(updated)
            print(f"Updated: {filepath}")
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Substitute documentation variables in MDX files")
    parser.add_argument(
        "directory",
        nargs="?",
        default="versions/v26.02",
        help="Path under fern/ containing MDX (e.g. versions/v25.09, versions/v26.02)",
    )
    parser.add_argument("--version", help="Version string for version/container_version/current_release (e.g. 25.09, 26.02)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without modifying files")
    args = parser.parse_args()

    # fern/substitute_variables.py -> fern/<version> or fern/versions/<version>
    fern_root = Path(__file__).resolve().parent
    base_dir = fern_root / args.directory
    if not base_dir.exists():
        candidate = fern_root / "versions" / args.directory
        if candidate.exists():
            base_dir = candidate
    if not base_dir.exists():
        print(f"Error: Directory not found: {base_dir}")
        return 1

    variables = dict(DEFAULT_VARIABLES)
    if args.version:
        variables["version"] = args.version
        variables["container_version"] = args.version
        variables["current_release"] = args.version

    modified_count = 0
    for mdx_file in base_dir.rglob("*.mdx"):
        if process_file(mdx_file, variables, args.dry_run):
            modified_count += 1

    print(f"\n{'Would modify' if args.dry_run else 'Modified'}: {modified_count} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
