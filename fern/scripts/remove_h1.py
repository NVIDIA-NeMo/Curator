#!/usr/bin/env python3
"""Remove duplicate H1 headers that match the frontmatter title."""

import re
from pathlib import Path


def remove_h1_after_frontmatter(filepath: Path) -> bool:
    """Remove the H1 that appears right after frontmatter if it duplicates the title.
    
    Returns True if changes were made.
    """
    content = filepath.read_text()
    
    # Pattern: frontmatter ending with ---, then blank line, then # Title, then blank line
    # We want to remove the # Title line and one surrounding blank line
    pattern = r"(---\n)\n(# [^\n]+)\n"
    
    match = re.search(pattern, content)
    if match:
        # Replace with just the frontmatter closing and single newline
        new_content = re.sub(pattern, r"\1", content, count=1)
        if new_content != content:
            filepath.write_text(new_content)
            return True
    
    return False


def main():
    script_dir = Path(__file__).parent
    pages_dir = script_dir.parent / "pages"
    
    if not pages_dir.exists():
        print(f"Error: pages directory not found at {pages_dir}")
        return
    
    changed_files = []
    
    for mdx_file in sorted(pages_dir.rglob("*.mdx")):
        if remove_h1_after_frontmatter(mdx_file):
            rel_path = mdx_file.relative_to(pages_dir)
            changed_files.append(rel_path)
            print(f"✓ Removed H1: {rel_path}")
    
    print(f"\n{'='*50}")
    print(f"Files modified: {len(changed_files)}")


if __name__ == "__main__":
    main()
