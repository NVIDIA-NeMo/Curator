# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import re
from functools import partial
from bs4 import BeautifulSoup, Tag
from .markdown import escape_markdown


def html2markdown(root: str | Tag, *, is_root: bool = True, debug: bool = False, debug_depth: int = 0):
    """Convert Wikipedia HTML to Mathpix Markdown."""
    if isinstance(root, str):
        root = BeautifulSoup(root, "html.parser")
    if not root.contents:
        return escape_markdown((root.string or "").strip())

    markdown = ""
    def append(kind: str, text: str):
        nonlocal markdown
        if debug:
            print(f"{'  ' * debug_depth}{kind}: {repr(text)}")
        markdown += text
    html2markdown_ = partial(html2markdown, is_root=False, debug=debug, debug_depth=debug_depth + 1)
    for child in root.children:
        if debug and isinstance(child, Tag):
            print(f"{'  ' * debug_depth}html: {repr(child.decode_contents())}")
        style = child.get("style", "") if isinstance(child, Tag) else ""
        is_bold = (child.name in ["strong", "b"]) or ("font-weight:bold" in style.replace(" ", ""))
        is_italic = (child.name in ["em", "i", "q", "dfn", "var"]) or ("font-style:italic" in style.replace(" ", ""))
        if child.name is None:
            append("string", escape_markdown(child.string or "", is_col0=(markdown == "" or markdown.endswith("\n"))))
        elif "visibility:hidden" in child.get("style", "").replace(" ", ""):
            continue
        elif "display:none" in child.get("style", "").replace(" ", ""):
            continue
        elif child.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            append("heading", convert_heading(child))
        elif child.name == "p":
            append("paragraph", html2markdown_(child) + "\n\n")
        elif child.name == "a":
            # visually just different color, no markdown needed
            append("link", html2markdown_(child))
        elif child.name == "img":
            # ignore images
            continue
        elif is_bold and is_italic:
            child_md = html2markdown_(child)
            append("bold_italic", wrap(wrap(child_md, "_"), "**"))
        elif is_bold:
            child_md = html2markdown_(child)
            append("bold", wrap(child_md, "**"))
        elif is_italic:
            child_md = html2markdown_(child)
            append("italic", wrap(child_md, "_"))
        elif child.name in ["code", "samp", "kbd"]:
            child_md = html2markdown_(child)
            append("code", wrap(child_md, "`"))
        elif child.name in ["s", "del"]:
            child_md = html2markdown_(child)
            append("strikethrough", wrap(child_md, "~~"))
        elif child.name in ["sub", "sup", "u"]:
            append("underline", f"<{child.name}>{html2markdown_(child)}</{child.name}>")
        elif child.name in ["ul", "ol"]:
            append("list", convert_list(child))
        elif child.name == "pre":
            append("code", convert_pre(child))
        elif child.name == "blockquote":
            append("blockquote", convert_blockquote(child))
        elif child.name == "hr":
            append("hr", "---\n\n")
        elif child.name == "br":
            append("br", "\n\n")
        elif child.name == "span" and child.get("typeof") == "mw:Entity":
            content = child.get_text()
            if content == "\xa0" or content == " ":
                append("entity_space", " ")
            else:
                append("entity_content", content)
        elif child.name in [
            "li",
            "span",
            "center",
            "cite",
            "div",
            "dd",
            "dt",
            "small",
            "big",
            "caption",
            "figcaption",
            "mark",
            "tt",
            "abbr",
            "bdi",
            "bdo",
            "time",
        ]:
            append("span", html2markdown_(child))
        elif child.name == "dl":
            append("definition_list", convert_definition_list(child, html2markdown_))
        elif child.name in ["table", "tbody"]:
            append("table", convert_table(child, debug=debug, debug_depth=debug_depth))
        elif child.name == "ruby":
            append("ruby", convert_ruby(child))
        elif child.name in ["link", "style", "script", "meta"]:
            pass
        else:
            if child.get_text().strip():
                raise ValueError(
                    f"Unknown HTML element: {child.name}: {child.get_text()}"
                )

    # Merge adjacent subscripts/superscripts
    markdown = markdown.replace("</sup><sup>", "").replace("</sub><sub>", "")
    # Clean up extra newlines
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    markdown = normalize_whitespace(markdown)
    # Strip leading newlines but preserve leading spaces
    res = re.sub(r"^\n+", "", markdown)
    if is_root:
        res = res.rstrip()
    return res


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace based on visual appearance."""
    text = text.replace("\xa0", " ")  # "&nbsp;" looks like regular space
    text = text.replace("\u200a", "")  # "&hairsp;" is too small for OCR
    return text


def collapse_whitespace(text: str) -> str:
    """Collapse whitespace according to CSS "white-space: normal" rules."""
    text = normalize_whitespace(text)
    if not text:
        return text
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text


def convert_heading(node: Tag):
    level = int(node.name[1])
    text = escape_markdown(node.get_text().strip(), is_col0=True)
    return "#" * level + " " + text + "\n\n"


def convert_list(node: Tag):
    list_type = "*" if node.name == "ul" else "1."
    result = "\n"
    for item in node.find_all("li", recursive=False):
        text = escape_markdown(item.get_text().strip(), is_col0=True)
        result += f"{list_type} {text}\n"
    return result + "\n"


def convert_definition_list(node: Tag, html2markdown_):
    result = "\n"
    for child in node.children:
        if child.name == "dt":
            result += html2markdown_(child) + "\n"
        elif child.name == "dd":
            result += ": " + html2markdown_(child) + "\n"
        else:
            result += html2markdown_(child) + "\n"
    return result + "\n"


def convert_pre(node: Tag):
    code = node.find("code")
    if code:
        language = ""
        class_attr = code.get("class")
        if class_attr:
            for cls in class_attr:
                if cls.startswith("language-"):
                    language = cls[9:]
                    break
        text = escape_markdown(code.get_text().strip(), is_col0=False)
        return f"```{language}\n{text}\n```\n\n"
    else:
        text = escape_markdown(node.get_text().strip(), is_col0=False)
        return f"```\n{text}\n```\n\n"


def convert_blockquote(node: Tag):
    lines = node.get_text().strip().split("\n")
    return "> " + "\n> ".join(map(partial(escape_markdown, is_col0=False), lines)) + "\n\n"


def convert_ruby(node: Tag):
    # https://developer.mozilla.org/en-US/docs/Web/HTML/Element/ruby
    text = escape_markdown(node.get_text().strip(), is_col0=False)
    return f"[{text}]"


def convert_table(table: str | BeautifulSoup, *, debug: bool = False, debug_depth: int = 0):
    """Convert HTML table to Mathpix Markdown table (LaTeX & Markdown hybrid)."""
    if isinstance(table, str):
        soup = BeautifulSoup(table, "html.parser")
        # Find the first table element
        table_elem = soup.find("table")
        if table_elem:
            return convert_table(table_elem, debug=debug, debug_depth=debug_depth)
        return ""

    rows = table.find_all("tr")
    caption = table.find("caption")

    columns_count = 0
    for row in rows:
        col_count = 0
        for cell in row.find_all(["td", "th"]):
            colspan = maybe_int(cell.get("colspan", ""), 1)
            col_count += colspan
        columns_count = max(columns_count, col_count)

    mmd = []
    mmd.append(f"\\begin{{tabular}}{{{'c' * columns_count}}}")

    # Create a grid to track cell occupancy for multirow/multicolumn spans
    grid = [[False for _ in range(columns_count)] for _ in range(len(rows))]

    for row_idx, row in enumerate(rows):
        cells = []
        col_idx = 0

        # Skip cells that are already occupied by multirow spans from previous rows
        while (
            col_idx < columns_count
            and col_idx < len(grid[row_idx])
            and grid[row_idx][col_idx]
        ):
            cells.append("")
            col_idx += 1

        for cell in row.find_all(["td", "th"]):
            cell_content = convert_table_cell(cell, debug=debug, debug_depth=debug_depth + 1)
            if cell.name == "th":
                cell_content = wrap(cell_content, "**")
            if debug:
                print(f"{'  ' * debug_depth}cell {row_idx},{col_idx}: {cell_content}")

            rowspan = maybe_int(cell.get("rowspan", ""), 1)
            colspan = maybe_int(cell.get("colspan", ""), 1)

            # Mark grid cells as occupied for this cell span
            for r in range(row_idx, min(row_idx + rowspan, len(grid))):
                for c in range(col_idx, min(col_idx + colspan, columns_count)):
                    grid[r][c] = True

            if colspan > 1:
                if rowspan > 1:
                    content = f"\\multicolumn{{{colspan}}}{{c}}{{\\multirow{{{rowspan}}}{{*}}{{{cell_content}}}}}"
                else:
                    content = f"\\multicolumn{{{colspan}}}{{c}}{{{cell_content}}}"
            elif rowspan > 1:
                content = f"\\multirow{{{rowspan}}}{{*}}{{{cell_content}}}"
            else:
                content = cell_content

            cells.append(content)
            col_idx += colspan

            # Skip to next unoccupied column if any
            while (
                col_idx < columns_count
                and col_idx < len(grid[row_idx])
                and grid[row_idx][col_idx]
            ):
                cells.append("")
                col_idx += 1

        while col_idx < columns_count:
            cells.append("")
            col_idx += 1

        if cells:
            mmd.append(" & ".join(cells) + " \\\\")

    # Remove empty rows
    mmd = [line for line in mmd if line and not line.strip() == "\\\\"]

    mmd.append("\\end{tabular}")

    if caption:
        caption_text = convert_table_cell(caption, debug=debug, debug_depth=debug_depth + 1)
        mmd.append(f"\\caption{{{caption_text}}}")

    res = "\n".join(mmd) + "\n"

    # Normalize empty cells to single space
    res = re.sub(r"(?<!\\)(?<=&) +(&|\\\\)", " \\1", res)
    return res


def convert_table_cell(element: Tag, *, debug: bool = False, debug_depth: int = 0) -> str:
    """Convert HTML table cell to markdown."""
    if debug and isinstance(element, Tag):
        print(f"{'  ' * debug_depth}html: {repr(element.decode_contents())}")
    if element.name is None:
        text = escape_markdown(element.string or "", is_col0=False)
        text = collapse_whitespace(text)
        escapes = {
            "$": "\\$",
            "&": "\\&",
            "<": "\\<",
            ">": "\\>",
        }
        for char, replacement in escapes.items():
            text = re.sub(r"(?<!\\)" + re.escape(char), replacement, text)
        if debug:
            print(f"{'  ' * debug_depth}text: {repr(text)}")
        return text

    if debug:
        def log(kind, text):
            print(f"{'  ' * debug_depth}{kind}: {repr(text)}")
            return text
        convert_table_cell_ = partial(convert_table_cell, debug=debug, debug_depth=debug_depth + 1)
    else:
        log = lambda kind, text: text
        convert_table_cell_ = convert_table_cell

    if element.name == "br":
        return log("br", "<br>")

    if element.name == "img":
        return log("image", "")

    if element.name == "a":
        child_content = "".join(convert_table_cell_(child) for child in element.children)
        return log("link", child_content)

    style = element.get("style", "")
    is_bold = element.name in ["strong", "b"] or "font-weight:bold" in style.replace(" ", "")
    is_italic = element.name in ["em", "i", "q", "dfn", "var"] or "font-style:italic" in style.replace(" ", "")
    
    if is_bold and is_italic:
        child_content = "".join(convert_table_cell_(child) for child in element.children)
        return log("bold_italic", wrap(wrap(child_content, "_"), "**"))
    elif is_bold:
        child_content = "".join(convert_table_cell_(child) for child in element.children)
        return log("bold", wrap(child_content, "**"))
    elif is_italic:
        child_content = "".join(convert_table_cell_(child) for child in element.children)
        return log("italic", wrap(child_content, "_"))

    if element.name in ["s", "del"]:
        child_content = "".join(convert_table_cell_(child) for child in element.children)
        return log("strikethrough", wrap(child_content, "~~"))

    if element.name in ["code", "samp", "kbd"]:
        child_content = "".join(convert_table_cell_(child) for child in element.children)
        return log("code", wrap(child_content, "`"))

    if element.name == "ul":
        items = []
        for li in element.find_all("li", recursive=False):
            text = convert_table_cell_(li).strip()
            if text:
                items.append(text)
                # extract_doclaynet.js stores `::after` as data-after
                items.append(li.attrs.get("data-after", " <br> "))
        if items and items[-1] == " <br> ":
            items.pop()
        return log("ul", "".join(items))

    if element.name in ["sub", "sup", "u"]:
        child_content = "".join(convert_table_cell_(child) for child in element.children)
        return log("sup_sub", f"<{element.name}>{child_content}</{element.name}>")

    # everything else (abbr, span, center, cite, small, big, mark, tt, bdi, bdo, ...)
    result = "".join(convert_table_cell_(child) for child in element.children)
    result = collapse_whitespace(result)

    # Trim only for table cells; preserve boundary spaces inside inline spans
    if hasattr(element, 'name') and element.name in ['td', 'th']:
        if result.strip() == "":
            return ""
        result = result.strip()
    return log("span", result)


def wrap(text: str, op: str) -> str:
    text = text.strip()
    if not text:
        return ""
    if text.startswith(op) or text.endswith(op):
        return text
    return f"{op}{text}{op}"


def maybe_int(text: str, default: int) -> int:
    try:
        return int(text)
    except ValueError:
        return default


# Example usage
if __name__ == "__main__":
    html = """
    <html>
    <body>
        <h1>Sample Heading</h1>
        <p>This is a <strong>bold</strong> and <em>italic</em> text with a <a href="https://example.com">link</a>.</p>
        <ul>
            <li>Item 1</li>
            <li>Item 2</li>
        </ul>
        <pre><code class="language-python">print("Hello World")</code></pre>
        <blockquote>This is a quote</blockquote>
    </body>
    </html>
    """

    print(html2markdown(html))
