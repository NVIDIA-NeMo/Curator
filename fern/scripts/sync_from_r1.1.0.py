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
    convert_tabs_and_cards_with_context,
    convert_image_directive,
    remove_toctree,
    escape_angle_bracket_refs,
    escape_comparison_operators,
    convert_angle_bracket_urls,
    convert_list_table,
    remove_octicon,
    remove_bdg,
    remove_plus_plus_plus,
    remove_header_labels,
    convert_html_comments,
    remove_directive_options,
    fix_malformed_tags,
    strip_remaining_colons,
    clean_multiple_newlines,
)

REPO_ROOT = Path(__file__).parent.parent.parent
TARGET_DIR = REPO_ROOT / "fern" / "v26.02" / "pages"
BRANCH = "r1.1.0"

SKIP_FILES = {"README.md", "RFC-FERN-MIGRATION.md"}
SKIP_DIRS = {"_extensions", "_templates", "_build", "apidocs", ".venv", ".git"}


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
    content = remove_octicon(content)
    content = remove_bdg(content)
    content = remove_plus_plus_plus(content)
    content = convert_list_table(content)
    content = remove_header_labels(content)
    content = remove_toctree(content)
    content = convert_html_comments(content)  # Before escape_angle_bracket_refs so comments aren't misparsed as refs
    content = escape_comparison_operators(content)  # Before refs so <= and >= aren't misparsed
    content = escape_angle_bracket_refs(content)
    content = convert_admonitions(content)  # Before tabs/cards so :::: isn't consumed first
    content = convert_dropdowns(content)
    content = convert_tabs_and_cards_with_context(content)
    content = convert_image_directive(content)
    content = remove_directive_options(content)
    content = fix_malformed_tags(content)
    content = convert_angle_bracket_urls(content)
    content = strip_remaining_colons(content)
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


def get_image_paths() -> list[str]:
    """Discover all image files in docs/ on r1.1.0."""
    result = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", BRANCH, "--", "docs/"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    exts = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}
    paths = []
    for line in result.stdout.strip().splitlines():
        if any(part in SKIP_DIRS for part in line.split("/")):
            continue
        if Path(line).suffix.lower() in exts:
            paths.append(line)
    return sorted(paths)


def sync_images() -> int:
    """Copy docs images to fern/v26.02/pages preserving structure."""
    paths = get_image_paths()
    copied = 0
    for docs_path in paths:
        try:
            result = subprocess.run(
                ["git", "show", f"{BRANCH}:{docs_path}"],
                cwd=REPO_ROOT,
                capture_output=True,
            )
            if result.returncode != 0:
                continue
            rel = docs_path.replace("docs/", "")
            out_path = TARGET_DIR / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(result.stdout)
            print(f"✓ {rel}")
            copied += 1
        except Exception as e:
            print(f"✗ {docs_path}: {e}")
    # Copy video diagrams to assets/images for Fern (expects /assets/images/ paths)
    assets_dir = REPO_ROOT / "fern" / "assets" / "images"
    for name in ["stages-pipelines-diagram.png", "video-pipeline-diagram.png"]:
        src = TARGET_DIR / "about" / "concepts" / "video" / "_images" / name
        if src.exists():
            assets_dir.mkdir(parents=True, exist_ok=True)
            (assets_dir / name).write_bytes(src.read_bytes())
            print(f"✓ assets/images/{name}")
    return copied


def run_post_processing(pages_dir: Path) -> None:
    """Run Automodel post-processing scripts: curator-specific, frontmatter, links, duplicate H1."""
    from add_frontmatter import add_frontmatter
    from update_links import update_file
    from remove_duplicate_h1 import remove_duplicate_h1
    from convert_curator_specific import convert_file as convert_curator_file

    for mdx_file in sorted(pages_dir.rglob("*.mdx")):
        convert_curator_file(mdx_file)
    for mdx_file in sorted(pages_dir.rglob("*.mdx")):
        add_frontmatter(mdx_file)
    for mdx_file in sorted(pages_dir.rglob("*.mdx")):
        update_file(mdx_file, pages_dir)
    for mdx_file in sorted(pages_dir.rglob("*.mdx")):
        remove_duplicate_h1(mdx_file)


def get_docs_paths() -> list[str]:
    """Discover all docs/*.md files on r1.1.0 branch, excluding skip lists."""
    result = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", BRANCH, "--", "docs/"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to list docs: {result.stderr}")
    paths = []
    for line in result.stdout.strip().splitlines():
        if not line.endswith(".md"):
            continue
        rel = line.replace("docs/", "")
        if rel in SKIP_FILES:
            continue
        if any(part in SKIP_DIRS or part.startswith(".") for part in rel.split("/")):
            continue
        paths.append(line)
    return sorted(paths)


def main():
    docs_paths = get_docs_paths()
    success = 0
    failed = []

    for docs_path in docs_paths:
        try:
            content = extract_from_git(docs_path)
            content = convert_content(content)

            rel_path = docs_path.replace("docs/", "").replace(".md", ".mdx")
            out_path = TARGET_DIR / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(content)
            print(f"✓ {rel_path}")
            success += 1
        except Exception as e:
            print(f"✗ {docs_path}: {e}")
            failed.append(docs_path)

    print(f"\nSyncing images...")
    img_count = sync_images()

    print(f"\nRunning post-processing (Automodel scripts)...")
    run_post_processing(TARGET_DIR)

    print(f"\n{'='*50}")
    print(f"Synced: {success}/{len(docs_paths)} docs, {img_count} images")
    if failed:
        print(f"Failed: {len(failed)}")
        for f in failed:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
