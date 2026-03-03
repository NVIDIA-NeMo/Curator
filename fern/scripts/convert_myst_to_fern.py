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
    """Convert MyST admonitions to Fern components. Handles :::, ::::, ::::: (from Automodel)."""
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
        replacement = rf"<{fern_component}>\n\1\n</{fern_component}>"

        # Pattern: ```{type}\ncontent\n```
        pattern = rf"```\{{{myst_type}\}}\s*\n(.*?)```"
        content = re.sub(pattern, replacement, content, flags=re.DOTALL | re.IGNORECASE)

        # :::, ::::, ::::: with {type} (Automodel pattern)
        for colons in [r":::", r"::::", r":::::"]:
            pattern = rf"{colons}\s*\{{{myst_type}\}}\s*\n(.*?){colons}"
            content = re.sub(pattern, replacement, content, flags=re.DOTALL | re.IGNORECASE)
            pattern = rf"{colons}\s+\{{{myst_type}\}}\s*\n(.*?){colons}"
            content = re.sub(pattern, replacement, content, flags=re.DOTALL | re.IGNORECASE)

        # Shorthand :::note (no braces)
        pattern = rf":::\s*{myst_type}\s*\n(.*?):::"
        content = re.sub(pattern, replacement, content, flags=re.DOTALL | re.IGNORECASE)
        for colons in [r"::::", r":::::"]:
            pattern = rf"{colons}\s*{myst_type}\s*\n(.*?){colons}"
            content = re.sub(pattern, replacement, content, flags=re.DOTALL | re.IGNORECASE)

        # Inline or multiline: :::{type} content (same line or following lines) until closing
        pattern = rf":::\s*\{{{myst_type}\}}\s+(.*?)\n\s*:::"
        content = re.sub(pattern, replacement, content, flags=re.DOTALL | re.IGNORECASE)
        for colons in [r"::::", r":::::"]:
            pattern = rf"{colons}\s*\{{{myst_type}\}}\s+(.*?)\n\s*{colons}"
            content = re.sub(pattern, replacement, content, flags=re.DOTALL | re.IGNORECASE)

    return content


def convert_dropdowns(content: str) -> str:
    """Convert MyST dropdowns to Fern Accordion components."""
    def replace_dropdown(match):
        title = match.group(1).strip()
        body = match.group(2).strip()
        # Escape quotes in title if needed
        if '"' in title:
            title = title.replace('"', "'")
        return f'<Accordion title="{title}">\n{body}\n</Accordion>'

    # Pattern 1: ```{dropdown} Title\ncontent\n```
    pattern1 = r"```\{dropdown\}\s+([^\n]+)\s*\n(.*?)```"
    content = re.sub(pattern1, replace_dropdown, content, flags=re.DOTALL)

    # Pattern 2: ::: or :::: {dropdown} Title ... content until </Tab>, </Accordion>, :::, or next directive
    pattern2 = r":::+?\s*\{dropdown\}\s+([^\n]+)\s*\n(.*?)(?=\n:::+|\n</Tab>|\n</Accordion>|\n:::\s*\{|\Z)"
    content = re.sub(pattern2, replace_dropdown, content, flags=re.DOTALL)

    # Pattern 3: <details><summary>Title</summary> ... </details>
    pattern3 = r"<details>\s*<summary>([^<]+)</summary>\s*\n(.*?)</details>"
    content = re.sub(pattern3, replace_dropdown, content, flags=re.DOTALL)

    return content


def convert_tab_sets(content: str) -> str:
    """Convert MyST tab sets to Fern Tabs components."""
    content = re.sub(r"::::+\s*\{tab-set\}\s*", "<Tabs>\n", content)
    content = re.sub(r"```\{tab-set\}\s*", "<Tabs>\n", content)

    def replace_tab_item(match):
        title = match.group(1).strip()
        return f'<Tab title="{title}">'

    content = re.sub(r"::::*\s*\{tab-item\}\s+([^\n]+)", replace_tab_item, content)
    content = re.sub(r":::*\s*\{tab-item\}\s+([^\n]+)", replace_tab_item, content)

    lines = content.split("\n")
    result = []
    in_tab = False

    for line in lines:
        if '<Tab title="' in line:
            if in_tab:
                result.append("</Tab>\n")
            in_tab = True
            result.append(line)
        elif line.strip() in [":::::", "::::", ":::", "</Tabs>"]:
            if in_tab and line.strip() != "</Tabs>":
                result.append("</Tab>")
                in_tab = False
            if line.strip() in [":::::", "::::"]:
                result.append("</Tabs>")
            elif line.strip() != ":::":
                result.append(line)
        else:
            result.append(line)

    content = "\n".join(result)
    content = re.sub(r"\n::::+\n", "\n", content)
    content = re.sub(r"\n:::+\n", "\n", content)
    return content


def convert_tabs_and_cards_with_context(content: str) -> str:
    """Run tab-set and grid conversions in one pass so :::: is correctly attributed to Tabs vs Cards."""
    content = re.sub(r"::::+\s*\{tab-set\}\s*", "<Tabs>\n", content)
    content = re.sub(r"```\{tab-set\}\s*", "<Tabs>\n", content)

    def replace_tab_item(match):
        title = match.group(1).strip()
        return f'<Tab title="{title}">'

    content = re.sub(r"::::*\s*\{tab-item\}\s+([^\n]+)", replace_tab_item, content)
    content = re.sub(r":::*\s*\{tab-item\}\s+([^\n]+)", replace_tab_item, content)

    content = re.sub(r"::::+\s*\{grid\}[^\n]*\n", "<Cards>\n", content)
    content = re.sub(r"```\{grid\}[^\n]*\n", "<Cards>\n", content)

    def replace_card(match):
        full_match = match.group(0)
        title_match = re.search(r"\{grid-item-card\}\s+(.+?)(?:\n|$)", full_match)
        title = title_match.group(1).strip() if title_match else "Card"
        link_match = re.search(r":link:\s*(\S+)", full_match)
        link_type_match = re.search(r":link-type:\s*(\S+)", full_match)
        href = ""
        if link_match:
            link_val = link_match.group(1).strip()
            if link_type_match and link_type_match.group(1) == "ref":
                href = _ref_to_path(link_val)
            elif link_val.startswith("http"):
                href = link_val
            else:
                href = "/" + link_val.replace(".md", "").replace(".mdx", "")
        if href:
            return f'<Card title="{title}" href="{href}">'
        return f'<Card title="{title}">'

    content = re.sub(
        r"::::*\s*\{grid-item-card\}[^\n]*(?:\n(?::link:[^\n]*|:link-type:[^\n]*))*",
        replace_card,
        content,
    )
    content = re.sub(
        r":::*\s*\{grid-item-card\}[^\n]*(?:\n(?::link:[^\n]*|:link-type:[^\n]*))*",
        replace_card,
        content,
    )

    lines = content.split("\n")
    result = []
    in_tab = False
    in_card = False
    last_block = None

    for line in lines:
        if '<Tab title="' in line:
            if in_tab:
                result.append("</Tab>\n")
            in_tab = True
            in_card = False
            result.append(line)
        elif '<Card title="' in line:
            if in_card:
                result.append("</Card>\n")
            in_card = True
            in_tab = False
            result.append(line)
        elif line.strip() in [":::::", "::::", ":::", "</Tabs>", "</Cards>"]:
            if in_tab and line.strip() not in ["</Tabs>"]:
                result.append("</Tab>")
                in_tab = False
            if in_card and line.strip() not in ["</Cards>"]:
                result.append("</Card>")
                in_card = False
            if line.strip() in [":::::", "::::"]:
                if last_block == "tabs":
                    result.append("</Tabs>")
                elif last_block == "cards":
                    result.append("</Cards>")
                last_block = None
            elif line.strip() not in [":::", "</Tabs>", "</Cards>"]:
                result.append(line)
        else:
            if "<Tabs>" in line:
                last_block = "tabs"
            elif "<Cards>" in line:
                last_block = "cards"
            result.append(line)

    content = "\n".join(result)
    content = re.sub(r"\n::::+\n", "\n", content)
    content = re.sub(r"\n:::+\n", "\n", content)
    return content


REF_LABEL_TO_PATH = {
    "about-overview": "/about",
    "about-key-features": "/about/key-features",
    "about-concepts": "/about/concepts",
    "gs-text": "/get-started/text",
    "gs-image": "/get-started/image",
    "gs-video": "/get-started/video",
    "gs-audio": "/get-started/audio",
}


def _ref_to_path(ref: str) -> str:
    """Convert Sphinx ref label to Fern path."""
    if ref in REF_LABEL_TO_PATH:
        return REF_LABEL_TO_PATH[ref]
    return "/" + ref.replace("-", "/")


def convert_list_table(content: str) -> str:
    """Convert MyST {list-table} to Markdown table.
    Format: * - cell1
             - cell2
           * - next row...
    """

    def replace_list_table(match: re.Match[str]) -> str:
        block = match.group(0)
        header_rows = 1 if ":header-rows: 1" in block or ":header-rows:1" in block else 0
        raw = re.sub(r"```\{list-table\}.*?\n", "", block)
        raw = re.sub(r"^:.*$", "", raw, flags=re.MULTILINE)
        raw = raw.replace("```", "").strip()
        row_blobs = re.split(r"\n\s*\*\s+-\s+", raw)
        rows = []
        for blob in row_blobs:
            blob = blob.strip()
            if not blob:
                continue
            if blob.startswith("* - "):
                blob = blob[4:]
            cells = re.split(r"\n\s+-\s+", blob)
            row_cells = []
            for c in cells:
                cell = c.strip().replace("\n", " ").replace("  ", " ")
                if cell.startswith("* - "):
                    cell = cell[4:]
                if cell.strip():
                    row_cells.append(cell.strip())
            if row_cells:
                rows.append(row_cells)
        if not rows:
            return ""
        num_cols = max(len(r) for r in rows)
        table_lines = []
        for row in rows:
            padded = row + [""] * (num_cols - len(row))
            escaped = [c.replace("|", "\\|") for c in padded]
            table_lines.append("| " + " | ".join(escaped) + " |")
        sep = "| " + " | ".join(["---"] * num_cols) + " |"
        if header_rows and len(table_lines) > 1:
            return table_lines[0] + "\n" + sep + "\n" + "\n".join(table_lines[1:])
        return table_lines[0] + "\n" + sep + "\n" + "\n".join(table_lines[1:]) if table_lines[1:] else table_lines[0]

    content = re.sub(
        r"```\{list-table\}.*?```",
        replace_list_table,
        content,
        flags=re.DOTALL,
    )
    return content


def remove_octicon(content: str) -> str:
    """Remove {octicon}`icon;size;class` from content (e.g. in grid-item-card titles)."""
    content = re.sub(r"\{octicon\}`[^`]+`\s*", "", content)
    return content


def remove_bdg(content: str) -> str:
    """Remove {bdg-primary}`, {bdg-secondary}` etc. - replace with nothing or badge text."""
    content = re.sub(r"\{bdg-(?:primary|secondary|success|warning|danger)\}`([^`]*)`", r"\1", content)
    return content


def remove_plus_plus_plus(content: str) -> str:
    """Remove MyST +++ separator lines (used in grid-item-card footer)."""
    content = re.sub(r"\n\+\+\+\s*\n", "\n", content)
    return content


def remove_header_labels(content: str) -> str:
    """Remove (label)= from content - standalone lines or inline with headers."""
    content = re.sub(r"^\([^)]+\)=\s*\n", "", content, flags=re.MULTILINE)
    content = re.sub(r"\n\([^)]+\)=\s*\n", "\n", content, flags=re.MULTILINE)
    content = re.sub(r"([#]+\s+[^\n]+?)\s*\([^)]+\)=\s*$", r"\1", content, flags=re.MULTILINE)
    return content


def convert_image_directive(content: str) -> str:
    """Convert {image} directive to markdown image."""
    pattern = r"```\{image\}\s+([^\s\n]+)(?:\s*\n(?::[^\n]+\n)*)?```"
    def replace(match: re.Match[str]) -> str:
        img_path = match.group(1).strip()
        alt_match = re.search(r":alt:\s*(.+)", match.group(0))
        alt = alt_match.group(1).strip() if alt_match else Path(img_path).name
        return f"![{alt}]({img_path})"
    return re.sub(pattern, replace, content)


def remove_toctree(content: str) -> str:
    """Remove toctree blocks entirely."""
    content = re.sub(r"```\{toctree\}.*?```", "", content, flags=re.DOTALL)
    content = re.sub(r":::\{toctree\}.*?:::", "", content, flags=re.DOTALL)
    content = re.sub(r"::::\{toctree\}.*?::::", "", content, flags=re.DOTALL)
    return content


def escape_angle_bracket_refs(content: str) -> str:
    """Convert MyST refs like 'Text <path/file.md>' to markdown links so MDX doesn't parse < > as JSX."""
    FERN_TAGS = {"note", "tip", "warning", "info", "cards", "tabs", "card", "tab"}

    def path_to_href(path: str) -> str:
        return "/" + path.replace(".md", "").replace(".mdx", "").replace(".rst", "")

    def replace_with_title(match: re.Match[str]) -> str:
        text, path = match.group(1).strip(), match.group(2).strip()
        path_lower = path.lower().strip("/").split("/")[0].split(".")[0]
        if path_lower in FERN_TAGS or path.rstrip("/") in ("br",):
            return match.group(0)
        return f"[{text}]({path_to_href(path)})"

    # Title <path> - only match when path looks like a doc ref (has .md/.mdx/.rst or /)
    # Exclude closing tags (</Note>, </Tip>) and <br/> - use negative lookahead (?!\/)
    content = re.sub(
        r"([^<\s][^<]*?)\s+<(?!\/)([^>]+\.(?:md|mdx|rst)|[^>]+/[^>]+)>",
        replace_with_title,
        content,
    )
    return content


def escape_comparison_operators(content: str) -> str:
    """Escape < and > so MDX doesn't parse them as JSX. Handles numeric (<100K, >0, < 1), operators (<=, >=), and prose (for <, for >)."""
    content = re.sub(r"<(\.?\d[\dKMG]*)", r"&lt;\1", content)
    content = re.sub(r"<\s*(\d[\dKMG%]*)", r"&lt; \1", content)  # < 1, < 80%
    content = re.sub(r">\s*(\.?\d[\dKMG]*)", r"&gt;\1", content)  # > 0, >1M
    content = re.sub(r"<=", "&lt;=", content)
    content = re.sub(r">=", "&gt;=", content)
    # Prose patterns like "`lt` for <" or "for <," - avoid corrupting code blocks
    content = re.sub(r" for <(?=[,\s)])", " for &lt;", content)
    content = re.sub(r" for >(?=[,\s)])", " for &gt;", content)
    return content


def convert_html_comments(content: str) -> str:
    """Convert HTML comments to JSX comments. Remove blocks that contain MyST directives (grid, tab, etc.) to avoid parse errors."""
    def replace_comment(match: re.Match[str]) -> str:
        inner = match.group(1)
        if "{grid" in inner or "{tab-" in inner or "::::" in inner:
            return ""
        return f"{{/* {inner} */}}"
    content = re.sub(r"<!--\s*(.*?)\s*-->", replace_comment, content, flags=re.DOTALL)
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
    # Fix empty title attributes
    content = re.sub(r'title=""', 'title="Details"', content)

    # Fix unclosed self-closing style tags
    content = re.sub(r"<(Note|Warning|Tip|Info)([^>]*)/>\s*\n([^<]+)", r"<\1\2>\n\3</\1>", content)

    return content


def strip_remaining_colons(content: str) -> str:
    """Remove any remaining :::, ::::, ::::: lines (Fern MDX must have none)."""
    content = re.sub(r"\n:+\s*\n", "\n", content)
    content = re.sub(r"^:+\s*\n", "", content, flags=re.MULTILINE)
    return content


def convert_angle_bracket_urls(content: str) -> str:
    """Convert <url> and <email> to Markdown links so MDX doesn't parse them as JSX (from Automodel)."""
    content = re.sub(r"<(https?://[^>]+)>", r"[\1](\1)", content)
    content = re.sub(
        r"<([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})>",
        r"[\1](mailto:\1)",
        content,
    )
    return content


def clean_multiple_newlines(content: str) -> str:
    """Clean up excessive newlines and stray directive markers."""
    content = re.sub(r"\n{3,}", "\n\n", content)
    content = re.sub(r"\n^:+\s*$", "", content, flags=re.MULTILINE)
    return content.strip() + "\n"


def convert_file(filepath: Path) -> bool:
    """Convert a single file. Returns True if changes were made."""
    content = filepath.read_text()
    original = content

    # Apply all conversions in order
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
