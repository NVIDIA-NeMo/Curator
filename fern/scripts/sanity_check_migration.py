#!/usr/bin/env python3
"""Sanity check: compare r1.1.0 docs/ content with fern/v26.02/pages/ output.

Verifies migration preserves content (format may differ: MyST -> MDX).
"""

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
BRANCH = "r1.1.0"
DOCS_PREFIX = "docs/"
PAGES_DIR = REPO_ROOT / "fern" / "v26.02" / "pages"


def get_doc_paths() -> list[str]:
    result = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", BRANCH, "--", DOCS_PREFIX],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("Failed to list docs")
    return [
        p for p in result.stdout.strip().splitlines()
        if p.endswith(".md") and "README" not in p
    ]


def fetch_source(docs_path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{BRANCH}:{docs_path}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to fetch {docs_path}")
    return result.stdout


def normalize_for_compare(text: str) -> str:
    """Strip formatting to compare core content."""
    # Remove frontmatter
    text = re.sub(r"^---\s*\n.*?\n---\s*\n", "", text, flags=re.DOTALL)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def extract_headings(text: str) -> list[str]:
    # Exclude headings inside code blocks
    lines = text.split("\n")
    in_code = False
    headings = []
    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if not in_code:
            m = re.match(r"^#{1,6}\s+(.+)$", line)
            if m:
                headings.append(m.group(1))
    return headings


def extract_code_blocks(text: str) -> list[str]:
    return re.findall(r"```[\s\S]*?```", text)


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def docs_path_to_fern(docs_path: str) -> Path:
    rel = docs_path.replace(DOCS_PREFIX, "").replace(".md", ".mdx")
    return PAGES_DIR / rel


def main() -> None:
    doc_paths = get_doc_paths()
    missing = []
    diff_heading = []
    diff_words = []
    errors = []

    for docs_path in sorted(doc_paths):
        fern_path = docs_path_to_fern(docs_path)
        if not fern_path.exists():
            missing.append(docs_path)
            continue

        try:
            source = fetch_source(docs_path)
            target = fern_path.read_text()
        except Exception as e:
            errors.append((docs_path, str(e)))
            continue

        src_headings = extract_headings(source)
        tgt_headings = extract_headings(target)

        # Normalize headings for comparison (strip formatting)
        src_norm = [re.sub(r"\s+", " ", h.strip()) for h in src_headings]
        tgt_norm = [re.sub(r"\s+", " ", h.strip()) for h in tgt_headings]

        # Check heading count and key headings
        if len(src_norm) != len(tgt_norm):
            diff_heading.append((docs_path, len(src_norm), len(tgt_norm)))

        src_words = count_words(normalize_for_compare(source))
        tgt_words = count_words(normalize_for_compare(target))
        diff_pct = abs(tgt_words - src_words) / max(src_words, 1) * 100
        if diff_pct > 15:  # Flag if >15% word difference
            diff_words.append((docs_path, src_words, tgt_words, diff_pct))

    print("=" * 60)
    print("Fern Migration Sanity Check")

    if missing:
        print(f"\n❌ MISSING ({len(missing)}): docs not in fern output")
        for p in missing[:10]:
            print(f"   {p}")
        if len(missing) > 10:
            print(f"   ... and {len(missing) - 10} more")

    if errors:
        print(f"\n❌ ERRORS ({len(errors)})")
        for p, e in errors[:5]:
            print(f"   {p}: {e}")

    if diff_heading:
        print(f"\n⚠️  HEADING COUNT DIFF ({len(diff_heading)}):")
        for p, src, tgt in diff_heading[:15]:
            print(f"   {p}: source {src} vs fern {tgt}")

    if diff_words:
        print(f"\n⚠️  WORD COUNT DIFF >15% ({len(diff_words)}):")
        for p, src, tgt, pct in diff_words[:15]:
            print(f"   {p}: {src} -> {tgt} ({pct:.0f}% diff)")

    ok = len(doc_paths) - len(missing) - len(errors)
    print(f"\n✅ OK: {ok}/{len(doc_paths)} docs present")
    if not missing and not errors:
        print("   All docs from r1.1.0 have corresponding fern output")
    print()


if __name__ == "__main__":
    main()
