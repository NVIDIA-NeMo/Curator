#!/usr/bin/env python3
"""
Replace documentation variables in MDX files.

This script is run during CI/CD before `fern generate` to substitute
template variables with their actual values.

Usage:
    python substitute_variables.py [--dry-run] [directory]
"""

import argparse
import re
from pathlib import Path

# Documentation variables - single source of truth
VARIABLES = {
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


def substitute_variables(content: str) -> str:
    """Replace {{ variable }} patterns with their values."""
    for var, value in VARIABLES.items():
        # Handle both {{ var }} and {{var}} patterns
        content = re.sub(rf"{{\{{\s*{var}\s*}}}}", value, content)
    return content


def process_file(filepath: Path, dry_run: bool = False) -> bool:
    """Process a single MDX file. Returns True if file was modified."""
    content = filepath.read_text()
    updated = substitute_variables(content)

    if content != updated:
        if dry_run:
            print(f"Would update: {filepath}")
        else:
            filepath.write_text(updated)
            print(f"Updated: {filepath}")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Substitute documentation variables in MDX files")
    parser.add_argument("directory", nargs="?", default="pages", help="Directory to process (default: pages)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without modifying files")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent / args.directory
    if not base_dir.exists():
        print(f"Error: Directory not found: {base_dir}")
        return 1

    modified_count = 0
    for mdx_file in base_dir.rglob("*.mdx"):
        if process_file(mdx_file, args.dry_run):
            modified_count += 1

    print(f"\n{'Would modify' if args.dry_run else 'Modified'}: {modified_count} files")
    return 0


if __name__ == "__main__":
    exit(main())
