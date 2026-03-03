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

"""Add frontmatter (title, description) to MDX files derived from first H1."""

import argparse
import re
from pathlib import Path


def derive_title(content: str) -> str:
    """Extract title from first # Heading."""
    match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if match:
        title = match.group(1).strip()
        title = re.sub(r"\{[^}]+\}`[^`]*`", "", title).strip()
        return title or "Untitled"
    return "Untitled"


def add_frontmatter(filepath: Path) -> bool:
    """Add frontmatter if missing. Returns True if changes were made."""
    content = filepath.read_text()

    if content.strip().startswith("---"):
        return False

    title = derive_title(content)
    title_escaped = title.replace('"', '\\"')
    frontmatter = f'---\ntitle: "{title_escaped}"\ndescription: ""\n---\n\n'
    body = content.lstrip()

    # Remove duplicate H1 that matches title (Fern uses frontmatter title)
    body = re.sub(r"^#\s+" + re.escape(title) + r"\s*\n+", "", body, count=1)

    new_content = frontmatter + body
    filepath.write_text(new_content)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Add frontmatter to MDX files")
    parser.add_argument(
        "pages_dir",
        type=Path,
        help="Path to pages directory (e.g. fern/v26.02/pages)",
    )
    args = parser.parse_args()

    pages_dir = args.pages_dir.resolve()
    if not pages_dir.exists():
        raise SystemExit(f"Error: pages directory not found at {pages_dir}")

    changed = []
    for mdx_file in sorted(pages_dir.rglob("*.mdx")):
        if add_frontmatter(mdx_file):
            changed.append(mdx_file.relative_to(pages_dir))
            print(f"  Added frontmatter: {mdx_file.relative_to(pages_dir)}")

    print(f"\nAdded frontmatter to {len(changed)} files")


if __name__ == "__main__":
    main()
