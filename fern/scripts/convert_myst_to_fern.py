#!/usr/bin/env python3
"""Convert MyST Markdown syntax to Fern MDX components.

This script handles:
- Admonitions: {note}, {warning}, {tip}, {important}, {seealso} → <Note>, <Warning>, <Tip>, <Info>
- Dropdowns: {dropdown} → <Accordion>
- Tab sets: {tab-set}, {tab-item} → <Tabs>, <Tab>
- Grid cards: {grid}, {grid-item-card} → <Cards>, <Card>
- Toctree: {toctree} → removed entirely
- HTML comments: <!-- --> → {/* */}
- Cross-references with titles
"""

import re
from pathlib import Path


def convert_admonitions(content: str) -> str:
    """Convert MyST admonitions to Fern components."""
    # Map MyST admonition types to Fern components
    admonition_map = {
        "note": "Note",
        "warning": "Warning",
        "tip": "Tip",
        "important": "Info",
        "seealso": "Note",
        "caution": "Warning",
        "danger": "Warning",
        "attention": "Warning",
        "hint": "Tip",
    }

    for myst_type, fern_component in admonition_map.items():
        # Pattern: ```{type}\ncontent\n```
        pattern = rf"```\{{{myst_type}\}}\s*\n(.*?)```"
        replacement = rf"<{fern_component}>\n\1</{fern_component}>"
        content = re.sub(pattern, replacement, content, flags=re.DOTALL | re.IGNORECASE)

        # Also handle ::: syntax
        pattern = rf":::\{{{myst_type}\}}\s*\n(.*?):::"
        content = re.sub(pattern, replacement, content, flags=re.DOTALL | re.IGNORECASE)

    return content


def convert_dropdowns(content: str) -> str:
    """Convert MyST dropdowns to Fern Accordion components."""
    # Pattern: ```{dropdown} Title\ncontent\n```
    pattern = r"```\{dropdown\}\s+([^\n]+)\s*\n(.*?)```"

    def replace_dropdown(match):
        title = match.group(1).strip()
        body = match.group(2).strip()
        # Escape quotes in title if needed
        if '"' in title:
            title = title.replace('"', "'")
        return f'<Accordion title="{title}">\n{body}\n</Accordion>'

    content = re.sub(pattern, replace_dropdown, content, flags=re.DOTALL)
    return content


def convert_tab_sets(content: str) -> str:
    """Convert MyST tab sets to Fern Tabs components."""
    # Convert tab-set opening
    content = re.sub(r"::::+\s*\{tab-set\}\s*", "<Tabs>\n", content)
    content = re.sub(r"```\{tab-set\}\s*", "<Tabs>\n", content)

    # Convert tab-item with title
    def replace_tab_item(match):
        title = match.group(1).strip()
        return f'<Tab title="{title}">'

    content = re.sub(r"::::*\s*\{tab-item\}\s+([^\n]+)", replace_tab_item, content)
    content = re.sub(r":::*\s*\{tab-item\}\s+([^\n]+)", replace_tab_item, content)

    # Close tab items (look for :::: or ::: followed by another tab-item or end of tabs)
    # This is tricky - we need to add </Tab> before the next <Tab> or </Tabs>
    lines = content.split("\n")
    result = []
    in_tab = False

    for i, line in enumerate(lines):
        # Check if this line starts a new tab
        if '<Tab title="' in line:
            if in_tab:
                result.append("</Tab>\n")
            in_tab = True
            result.append(line)
        # Check if this is a closing marker for tabs
        elif line.strip() in [":::::", "::::", ":::", "</Tabs>"]:
            if in_tab and line.strip() != "</Tabs>":
                result.append("</Tab>")
                in_tab = False
            if line.strip() in [":::::", "::::"]:
                result.append("</Tabs>")
            else:
                result.append(line)
        else:
            result.append(line)

    content = "\n".join(result)

    # Clean up any remaining :::: or ::: markers
    content = re.sub(r"\n::::+\n", "\n", content)
    content = re.sub(r"\n:::+\n", "\n", content)

    return content


def convert_grid_cards(content: str) -> str:
    """Convert MyST grid cards to Fern Cards components."""
    # Convert grid opening
    content = re.sub(r"::::+\s*\{grid\}[^\n]*\n", "<Cards>\n", content)
    content = re.sub(r"```\{grid\}[^\n]*\n", "<Cards>\n", content)

    # Convert grid-item-card with link
    def replace_card(match):
        full_match = match.group(0)
        # Extract title from the line
        title_match = re.search(r"\{grid-item-card\}\s+(.+?)(?:\n|$)", full_match)
        title = title_match.group(1).strip() if title_match else "Card"

        # Look for :link: directive
        link_match = re.search(r":link:\s*(\S+)", full_match)
        href = link_match.group(1) if link_match else ""

        if href:
            return f'<Card title="{title}" href="{href}">'
        return f'<Card title="{title}">'

    # Handle grid-item-card with various formats
    content = re.sub(
        r"::::*\s*\{grid-item-card\}[^\n]*(?:\n:link:[^\n]*)?",
        replace_card,
        content,
    )
    content = re.sub(
        r":::*\s*\{grid-item-card\}[^\n]*(?:\n:link:[^\n]*)?",
        replace_card,
        content,
    )

    # Close cards - similar logic to tabs
    lines = content.split("\n")
    result = []
    in_card = False

    for line in lines:
        if '<Card title="' in line:
            if in_card:
                result.append("</Card>\n")
            in_card = True
            result.append(line)
        elif line.strip() in [":::::", "::::", ":::", "</Cards>"]:
            if in_card and line.strip() != "</Cards>":
                result.append("</Card>")
                in_card = False
            if line.strip() in [":::::", "::::"]:
                result.append("</Cards>")
            else:
                result.append(line)
        else:
            result.append(line)

    content = "\n".join(result)
    return content


def remove_toctree(content: str) -> str:
    """Remove toctree blocks entirely."""
    content = re.sub(r"```\{toctree\}.*?```", "", content, flags=re.DOTALL)
    content = re.sub(r":::\{toctree\}.*?:::", "", content, flags=re.DOTALL)
    return content


def convert_html_comments(content: str) -> str:
    """Convert HTML comments to JSX comments."""
    content = re.sub(r"<!--\s*(.*?)\s*-->", r"{/* \1 */}", content, flags=re.DOTALL)
    return content


def remove_directive_options(content: str) -> str:
    """Remove MyST directive options like :icon:, :class:, etc."""
    # Remove lines that are just directive options
    content = re.sub(r"\n:icon:[^\n]*", "", content)
    content = re.sub(r"\n:class:[^\n]*", "", content)
    content = re.sub(r"\n:columns:[^\n]*", "", content)
    content = re.sub(r"\n:gutter:[^\n]*", "", content)
    content = re.sub(r"\n:margin:[^\n]*", "", content)
    content = re.sub(r"\n:padding:[^\n]*", "", content)
    content = re.sub(r"\n:link-type:[^\n]*", "", content)
    content = re.sub(r"\n:maxdepth:[^\n]*", "", content)
    content = re.sub(r"\n:titlesonly:[^\n]*", "", content)
    content = re.sub(r"\n:hidden:[^\n]*", "", content)
    return content


def fix_malformed_tags(content: str) -> str:
    """Fix common malformed tag issues."""
    # Fix </Cards> that should be </Tabs>
    # This requires context - look for <Tabs> without matching </Tabs>
    # For now, we'll handle specific known patterns

    # Fix empty title attributes
    content = re.sub(r'title=""', 'title="Details"', content)

    # Fix unclosed self-closing style tags
    content = re.sub(r"<(Note|Warning|Tip|Info)([^>]*)/>\s*\n([^<]+)", r"<\1\2>\n\3</\1>", content)

    return content


def clean_multiple_newlines(content: str) -> str:
    """Clean up excessive newlines."""
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip() + "\n"


def convert_file(filepath: Path) -> bool:
    """Convert a single file. Returns True if changes were made."""
    content = filepath.read_text()
    original = content

    # Apply all conversions in order
    content = convert_admonitions(content)
    content = convert_dropdowns(content)
    content = convert_tab_sets(content)
    content = convert_grid_cards(content)
    content = remove_toctree(content)
    content = convert_html_comments(content)
    content = remove_directive_options(content)
    content = fix_malformed_tags(content)
    content = clean_multiple_newlines(content)

    if content != original:
        filepath.write_text(content)
        return True
    return False


def main():
    """Run conversion on all MDX files in the pages directory."""
    script_dir = Path(__file__).parent
    pages_dir = script_dir.parent / "pages"

    if not pages_dir.exists():
        print(f"Error: pages directory not found at {pages_dir}")
        return

    changed_files = []
    total_files = 0

    for mdx_file in pages_dir.rglob("*.mdx"):
        total_files += 1
        if convert_file(mdx_file):
            changed_files.append(mdx_file)
            print(f"✓ Converted: {mdx_file.relative_to(pages_dir)}")

    print(f"\n{'='*50}")
    print(f"Total files scanned: {total_files}")
    print(f"Files modified: {len(changed_files)}")

    if changed_files:
        print("\nModified files:")
        for f in changed_files:
            print(f"  - {f.relative_to(pages_dir)}")


if __name__ == "__main__":
    main()
