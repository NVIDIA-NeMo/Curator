#!/usr/bin/env python3
"""
Sync edited docs from r1.1.0 branch to fern v26.02.
Extracts MyST content, converts to Fern MDX, writes to fern/v26.02/pages/.
"""

import subprocess
import sys
from pathlib import Path

# Import conversion functions from convert_myst_to_fern
sys.path.insert(0, str(Path(__file__).parent))
from convert_myst_to_fern import (
    convert_admonitions,
    convert_dropdowns,
    convert_tab_sets,
    convert_grid_cards,
    remove_toctree,
    convert_html_comments,
    remove_directive_options,
    fix_malformed_tags,
    clean_multiple_newlines,
)

REPO_ROOT = Path(__file__).parent.parent.parent
TARGET_DIR = REPO_ROOT / "fern" / "v26.02" / "pages"
BRANCH = "r1.1.0"

# Edited pages in f1e0a2c5 (2026-02-12) - docs path -> no conversion needed for path
EDITED_PAGES = [
    "docs/about/release-notes/index.md",
    "docs/about/release-notes/migration-guide.md",
    "docs/about/concepts/video/abstractions.md",
    "docs/admin/installation.md",
    "docs/curate-text/index.md",
    "docs/curate-text/process-data/deduplication/fuzzy.md",
    "docs/curate-text/process-data/deduplication/index.md",
    "docs/curate-text/process-data/deduplication/semdedup.md",
    "docs/curate-text/process-data/quality-assessment/distributed-classifier.md",
    "docs/curate-text/process-data/quality-assessment/heuristic.md",
    "docs/curate-audio/tutorials/beginner.md",
    "docs/curate-video/index.md",
    "docs/curate-video/load-data/index.md",
    "docs/curate-video/process-data/captions-preview.md",
    "docs/curate-video/process-data/clipping.md",
    "docs/curate-video/process-data/dedup.md",
    "docs/curate-video/process-data/embeddings.md",
    "docs/curate-video/process-data/filtering.md",
    "docs/curate-video/process-data/frame-extraction.md",
    "docs/curate-video/process-data/index.md",
    "docs/curate-video/save-export.md",
    "docs/curate-video/tutorials/beginner.md",
    "docs/curate-video/tutorials/pipeline-customization/add-cust-env.md",
    "docs/curate-video/tutorials/pipeline-customization/add-cust-model.md",
    "docs/curate-video/tutorials/pipeline-customization/add-cust-stage.md",
    "docs/curate-video/tutorials/split-dedup.md",
    "docs/get-started/audio.md",
    "docs/get-started/image.md",
    "docs/get-started/text.md",
    "docs/get-started/video.md",
    "docs/index.md",
    "docs/reference/infrastructure/container-environments.md",
    "docs/reference/infrastructure/execution-backends.md",
]


def preprocess_myst(content: str) -> str:
    """Preprocess MyST-specific syntax before main conversion."""
    import re
    # ```{mermaid} -> ```mermaid
    content = re.sub(r"```\s*\{mermaid\}\s*", "```mermaid\n", content)
    # {ref}`target` or {ref}`title <target>` -> [title](/path) - simplify to link
    content = re.sub(r"\{ref\}\`([^`<]+)\s*<([^`>]+)>\`", r"[\1](/\2)", content)
    content = re.sub(r"\{ref\}\`([^`]+)\`", r"[\1](/about/release-notes/migration-guide)", content)
    # {literalinclude} - remove or replace
    content = re.sub(
        r"```\{literalinclude\}.*?```",
        "See configuration files in the repository.",
        content,
        flags=re.DOTALL,
    )
    # {seealso} - convert to Note
    content = re.sub(
        r"```\{seealso\}\s*\n(.*?)```",
        r"<Note>\n\1\n</Note>",
        content,
        flags=re.DOTALL,
    )
    return content


def convert_content(content: str) -> str:
    """Apply MyST to Fern conversions."""
    content = preprocess_myst(content)
    content = convert_admonitions(content)
    content = convert_dropdowns(content)
    content = convert_tab_sets(content)
    content = convert_grid_cards(content)
    content = remove_toctree(content)
    content = convert_html_comments(content)
    content = remove_directive_options(content)
    content = fix_malformed_tags(content)
    content = clean_multiple_newlines(content)
    return content


def extract_from_git(docs_path: str) -> str:
    """Extract file content from r1.1.0 branch."""
    result = subprocess.run(
        ["git", "show", f"{BRANCH}:{docs_path}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to extract {docs_path}: {result.stderr}")
    return result.stdout


def main():
    success = 0
    failed = []

    for docs_path in EDITED_PAGES:
        try:
            content = extract_from_git(docs_path)
            content = convert_content(content)

            # Map docs/X.md -> fern/v26.02/pages/X.mdx
            rel_path = docs_path.replace("docs/", "").replace(".md", ".mdx")
            out_path = TARGET_DIR / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(content)
            print(f"✓ {rel_path}")
            success += 1
        except Exception as e:
            print(f"✗ {docs_path}: {e}")
            failed.append(docs_path)

    print(f"\n{'='*50}")
    print(f"Synced: {success}/{len(EDITED_PAGES)}")
    if failed:
        print(f"Failed: {len(failed)}")
        for f in failed:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
