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

"""Update internal links: .md -> Fern paths, relative paths -> absolute."""

import argparse
import re
from pathlib import Path


def normalize_url(url: str, file_dir: Path | None = None, pages_root: Path | None = None) -> str:
    """Normalize a URL to Fern path format. Resolve relative image paths when file_dir/pages_root given."""
    clean = url.replace(".md", "").replace(".mdx", "")
    if url.startswith(("http://", "https://", "#", "mailto:")):
        return url
    # Fern expects video diagrams in /assets/images/ (copied there by sync)
    if "stages-pipelines-diagram.png" in clean or "video-pipeline-diagram.png" in clean:
        return "/assets/images/" + ("stages-pipelines-diagram.png" if "stages-pipelines" in clean else "video-pipeline-diagram.png")
    # NeMo Curator: resolve relative image paths relative to the file's directory
    if (file_dir is not None and pages_root is not None and
            (clean.startswith("./") or clean.startswith("../") or ("_images/" in clean and not clean.startswith("/")))):
        resolved = (file_dir / clean).resolve()
        try:
            rel = resolved.relative_to(pages_root.resolve())
            clean = "/" + str(rel).replace("\\", "/")
            return clean
        except ValueError:
            pass
    if "_images/" in clean or clean.startswith("./") or clean.startswith("../"):
        if not clean.startswith("/"):
            clean = "/" + clean.lstrip("./")
        return clean
    if not clean.startswith("/"):
        clean = "/" + clean
    return clean


def update_links_in_content(content: str, file_dir: Path, pages_root: Path) -> str:
    """Update markdown links and image paths: .md/.mdx -> Fern paths."""

    def replace_link(match: re.Match[str]) -> str:
        text, url = match.group(1), match.group(2)
        clean = normalize_url(url)
        return f"[{text}]({clean})"

    def replace_image(match: re.Match[str]) -> str:
        alt, url = match.group(1), match.group(2)
        clean = normalize_url(url, file_dir, pages_root)
        return f"![{alt}]({clean})"

    # Process images first, then links (negative lookbehind avoids matching images)
    content = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", replace_image, content)
    content = re.sub(r"(?<!!)\[([^\]]*)\]\(([^)]+)\)", replace_link, content)
    return content


def update_file(filepath: Path, pages_root: Path) -> bool:
    """Update links in a single file. Returns True if changes were made."""
    content = filepath.read_text()
    file_dir = filepath.parent
    new_content = update_links_in_content(content, file_dir, pages_root)

    if new_content != content:
        filepath.write_text(new_content)
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Update internal links in MDX files")
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
        if update_file(mdx_file, pages_dir):
            changed.append(mdx_file.relative_to(pages_dir))
            print(f"  Updated: {mdx_file.relative_to(pages_dir)}")

    print(f"\nUpdated {len(changed)} files")


if __name__ == "__main__":
    main()
