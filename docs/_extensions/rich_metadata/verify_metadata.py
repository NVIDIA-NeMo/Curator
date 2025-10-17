#!/usr/bin/env python3
"""
Verification script for rich metadata extension.

This script checks if metadata has been properly injected into built HTML files.

Usage:
    python verify_metadata.py <path_to_built_html>

Example:
    python verify_metadata.py ../../_build/html/index.html
    python verify_metadata.py ../../_build/html/get-started/text.html
"""

import argparse
import json
import re
import sys
from pathlib import Path


def extract_meta_tags(html_content: str) -> dict[str, list[str]]:
    """Extract all meta tags from HTML content."""
    meta_tags = {
        "standard": [],
        "open_graph": [],
        "twitter": [],
        "custom": [],
    }

    # Extract standard meta tags
    for match in re.finditer(r'<meta name="([^"]+)" content="([^"]*)"', html_content):
        name, content = match.groups()
        meta_tags["standard"].append(f"{name}: {content}")

    # Extract Open Graph tags
    for match in re.finditer(r'<meta property="og:([^"]+)" content="([^"]*)"', html_content):
        name, content = match.groups()
        meta_tags["open_graph"].append(f"og:{name}: {content}")

    # Extract Twitter Card tags
    for match in re.finditer(r'<meta name="twitter:([^"]+)" content="([^"]*)"', html_content):
        name, content = match.groups()
        meta_tags["twitter"].append(f"twitter:{name}: {content}")

    return meta_tags


def extract_json_ld(html_content: str) -> dict | None:
    """Extract JSON-LD structured data from HTML content."""
    match = re.search(
        r'<script type="application/ld\+json">\s*(\{.*?\})\s*</script>',
        html_content,
        re.DOTALL,
    )

    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing JSON-LD: {e}")
            return None

    return None


def _display_meta_tags(tags: list[str], tag_type: str) -> bool:
    """Display meta tags of a specific type."""
    if tags:
        print(f"✅ {tag_type}:")
        for tag in tags:
            print(f"   • {tag}")
        print()
        return True

    print(f"⚠️  No {tag_type.lower()} found\n")
    return False


def _display_json_ld(json_ld: dict | None) -> bool:
    """Display JSON-LD structured data."""
    if not json_ld:
        print("⚠️  No JSON-LD structured data found\n")
        return False

    print("✅ JSON-LD Structured Data:")
    print(f"   • @type: {json_ld.get('@type', 'N/A')}")
    print(f"   • headline: {json_ld.get('headline', 'N/A')}")

    description = json_ld.get("description", "N/A")
    if description != "N/A":
        print(f"   • description: {description[:80]}...")

    if "keywords" in json_ld and isinstance(json_ld["keywords"], list):
        print(f"   • keywords: {', '.join(json_ld['keywords'][:5])}")

    if "audience" in json_ld:
        audience_type = json_ld["audience"].get("audienceType", [])
        if isinstance(audience_type, list):
            print(f"   • audience: {', '.join(audience_type)}")

    if "proficiencyLevel" in json_ld:
        print(f"   • proficiency: {json_ld['proficiencyLevel']}")

    print()
    return True


def _display_no_metadata_help() -> None:
    """Display help message when no metadata is found."""
    print("❌ No rich metadata found in this file.")
    print("   This could mean:")
    print("   • The page has no frontmatter")
    print("   • The extension is not enabled in conf.py")
    print("   • The template is not rendering {{ metatags }} or {{ rich_metadata }}")


def verify_html_file(html_path: Path) -> bool:
    """
    Verify that a built HTML file contains rich metadata.

    Returns:
        True if metadata is present, False otherwise
    """
    if not html_path.exists():
        print(f"❌ File not found: {html_path}")
        return False

    print(f"\n{'='*80}")
    print(f"Verifying: {html_path.name}")
    print(f"{'='*80}\n")

    html_content = html_path.read_text(encoding="utf-8")

    # Extract metadata
    meta_tags = extract_meta_tags(html_content)
    json_ld = extract_json_ld(html_content)

    # Display results and track if any metadata was found
    has_metadata = False
    has_metadata |= _display_meta_tags(meta_tags["standard"], "Standard Meta Tags")
    has_metadata |= _display_meta_tags(meta_tags["open_graph"], "Open Graph Tags")
    has_metadata |= _display_meta_tags(meta_tags["twitter"], "Twitter Card Tags")
    has_metadata |= _display_json_ld(json_ld)

    # Overall result
    if has_metadata:
        print("✅ Rich metadata extension is working!")
        return True

    _display_no_metadata_help()
    return False


def main() -> None:
    """Main entry point for the verification script."""
    parser = argparse.ArgumentParser(
        description="Verify rich metadata injection in built HTML files"
    )
    parser.add_argument(
        "html_files",
        nargs="+",
        type=Path,
        help="Path(s) to HTML file(s) to verify",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output",
    )

    args = parser.parse_args()

    all_passed = True
    for html_file in args.html_files:
        if not verify_html_file(html_file):
            all_passed = False

    print(f"\n{'='*80}")
    if all_passed:
        print("✅ All files verified successfully!")
    else:
        print("⚠️  Some files are missing metadata")
    print(f"{'='*80}\n")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

