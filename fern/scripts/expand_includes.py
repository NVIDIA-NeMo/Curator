#!/usr/bin/env python3
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

"""Expand {include} directives in MDX files. Resolves from docs/ (or git)."""

import argparse
import re
from pathlib import Path


def expand_include_in_content(
    content: str, file_path: Path, pages_dir: Path, docs_dir: Path
) -> str:
    """Replace {include} directives with file content. Paths relative to source doc."""
    pattern = r"```\{include\}\s+([^\s\n]+)(?:\s*\n(?::[^\n]+\n)*)?```"

    def replace_include(match: re.Match[str]) -> str:
        include_path_str = match.group(1).strip()
        rel = file_path.relative_to(pages_dir)
        source_dir = docs_dir / rel.parent
        if rel.name == "index.mdx":
            source_dir = docs_dir
        resolved = (source_dir / include_path_str).resolve()

        if not resolved.exists():
            return f"<!-- Include file not found: {resolved} -->"
        return resolved.read_text()

    return re.sub(pattern, replace_include, content)


def expand_file(filepath: Path, pages_dir: Path, docs_dir: Path) -> bool:
    """Expand includes in a single file. Returns True if changes were made."""
    content = filepath.read_text()
    if "{include}" not in content:
        return False

    new_content = expand_include_in_content(content, filepath, pages_dir, docs_dir)
    if new_content != content:
        filepath.write_text(new_content)
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expand {include} directives in MDX files"
    )
    parser.add_argument(
        "pages_dir",
        type=Path,
        help="Path to pages directory (e.g. fern/v26.02/pages)",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=None,
        help="Path to docs directory (default: repo_root/docs)",
    )
    args = parser.parse_args()

    pages_dir = args.pages_dir.resolve()
    if not pages_dir.exists():
        raise SystemExit(f"Error: pages directory not found at {pages_dir}")

    repo_root = pages_dir.parent.parent.parent
    docs_dir = args.docs_dir.resolve() if args.docs_dir else repo_root / "docs"

    expanded = []
    for mdx_file in sorted(pages_dir.rglob("*.mdx")):
        if expand_file(mdx_file, pages_dir, docs_dir):
            expanded.append(mdx_file.relative_to(pages_dir))
            print(f"  Expanded: {mdx_file.relative_to(pages_dir)}")

    print(f"\nExpanded {len(expanded)} files")


if __name__ == "__main__":
    main()
