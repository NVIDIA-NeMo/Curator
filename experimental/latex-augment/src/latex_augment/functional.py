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

from .color import to_xcolor
from .fonts import PDFLATEX_FONTS
from .latex_parser import LatexDocument, get_node_end_byte_ws


def remove_bibliography(doc: LatexDocument) -> LatexDocument:
    """Remove bibliography from a LaTeX document."""
    doc = remove_bibtex(doc)
    doc = remove_bib(doc)
    return doc


def remove_bibtex(doc: LatexDocument) -> LatexDocument:
    """Remove \\bibliography{...}."""
    try:
        node = doc.find(lambda node: node.type == "bibtex_include")
    except ValueError:
        return doc
    else:
        return LatexDocument(
            doc.source[: node.start_byte] + doc.source[get_node_end_byte_ws(node) :]
        )


def remove_bib(doc: LatexDocument) -> LatexDocument:
    """Remove \\begin{thebibliography}...\\end{thebibliography}."""
    try:
        begin = doc.find(
            lambda node: node.type == "begin"
            and node.text.startswith(b"\\begin{thebibliography}")
        )
        end = doc.find(
            lambda node: node.type == "end"
            and node.text.startswith(b"\\end{thebibliography}")
        )
    except ValueError:
        return doc
    return LatexDocument(
        doc.source[: begin.start_byte] + doc.source[get_node_end_byte_ws(end) :]
    )


def remove_captions(doc: LatexDocument) -> LatexDocument:
    """Remove captions from the document."""
    # use unique name to avoid name clashes with existing packages
    doc = doc.append_preamble("\\newcommand{\\latex_augment_comment}[1]{}")
    editor = doc.editor()
    for caption_span in doc.find_command("caption"):
        caption_text = doc.source[caption_span]
        comment_text = caption_text.replace(b"\\caption", b"\\latex_augment_comment", 1)
        editor.replace_at(caption_span, comment_text)
    return editor.execute()


def remove_headings(doc: LatexDocument) -> LatexDocument:
    """Convert headings to plain text."""
    commands = ("part", "chapter", "section", "subsection", "subsubsection")
    editor = doc.editor()
    for cmd in commands:
        for span in doc.find_command(cmd):
            heading_text = doc.source[span]
            plain_text = heading_text.replace(b"\\" + cmd.encode("utf8"), b"", 1)
            editor.replace_at(span, plain_text)
    return editor.execute()


def remove_page_numbers(doc: LatexDocument) -> LatexDocument:
    """Remove page numbers from the document."""
    doc = doc.wrap_body("\\pagestyle{empty}", "")
    doc = doc.wrap_body("\\pagenumbering{gobble}", "")
    return doc


def set_page_size_and_margins(
    doc: LatexDocument,
    *,
    width: float = None,
    height: float = None,
    top: float = None,
    bottom: float = None,
    right: float = None,
    left: float = None,
    show_crop: bool = False,
    show_frame: bool = False,
) -> LatexDocument:
    """Set the page size and margins in a LaTeX document."""
    geometry = []
    if width:
        doc = remove_documentclass_options(doc, "letterpaper", "legalpaper", "a4paper")
        assert width >= 50 and width <= 800, "width out of range"
        assert height >= 50 and height <= 800, "height out of range"
        geometry.append(f"paperwidth={round(width, 1)}mm")
        geometry.append(f"paperheight={round(height, 1)}mm")
        if not top:
            geometry.append(f"textwidth={round(width * 7 / 9, 1)}mm")
            geometry.append(f"textheight={round(height * 6 / 9, 1)}mm")
    if top:
        assert top >= 1 and top <= 800, "top out of range"
        assert bottom >= 1 and bottom <= 800, "bottom out of range"
        assert right >= 1 and right <= 800, "right out of range"
        assert left >= 1 and left <= 800, "left out of range"
        geometry.append(f"top={round(top, 1)}mm")
        geometry.append(f"bottom={round(bottom, 1)}mm")
        geometry.append(f"right={round(right, 1)}mm")
        geometry.append(f"left={round(left, 1)}mm")
    if show_crop:
        geometry.append("showcrop")
    if show_frame:
        geometry.append("showframe")
    geometry.append("ignoremp")
    doc = doc.with_package("geometry")
    doc = doc.append_preamble("\\geometry{" + ",".join(geometry) + "}")
    return doc


def set_column_layout_preamble(doc: LatexDocument) -> LatexDocument:
    """Set the column layout in a LaTeX document."""
    doc = remove_documentclass_options(doc, "onecolumn", "twocolumn")
    doc = doc.with_package("multicol")
    return doc


def column_layout_commands(
    ncols=1, columnsep=1, rulewidth=0, balanced=True
) -> tuple[str, str]:
    """Set the column layout in a LaTeX document."""
    if ncols == 1:
        return "", ""
    env = "multicols" if balanced else "multicols*"
    begin = (
        f"\\let\\oldcolumnsep\\columnsep\n"
        f"\\setlength\\columnsep{{{round(columnsep, 2)}\\columnsep}}\n"
        f"\\setlength\\columnseprule{{{round(rulewidth, 1)}pt}}\n"
        f"\\begin{{{env}}}{{{ncols}}}\n"
    )
    # add newline before \end{multicols} to avoid LaTeX error:
    # ! Improper \prevdepth.
    end = f"\n\n\\end{{{env}}}\n\\let\\columnsep\\oldcolumnsep\n"
    return begin, end


def set_column_layout(
    doc: LatexDocument, *, ncols=1, columnsep=1, rulewidth=0, balanced=True
) -> LatexDocument:
    """Set the column layout in a LaTeX document."""
    begin, end = column_layout_commands(ncols, columnsep, rulewidth, balanced)
    return set_column_layout_preamble(doc).wrap_body(begin, end)


def set_font(doc: LatexDocument, font: str) -> LatexDocument:
    """Set the font in a LaTeX document."""
    return doc.append_preamble(PDFLATEX_FONTS[font])


def set_font_size(doc: LatexDocument, size=10) -> LatexDocument:
    """Set the font size in a LaTeX document."""
    return doc.with_package("fontsize", [f"fontsize={round(size, 1)}pt"])


def set_line_spacing(doc: LatexDocument, spacing=1.0) -> LatexDocument:
    """Set the line spacing in a LaTeX document."""
    return doc.wrap_body(
        f"\\renewcommand{{\\baselinestretch}}{{{round(spacing, 2)}}}", ""
    )


def set_letter_spacing(doc: LatexDocument, tracking=0) -> LatexDocument:
    """Set the letter spacing (tracking) in a LaTeX document."""
    assert -1 <= tracking <= 1, "tracking must be between -1 and 1"
    doc = doc.with_package("microtype", [f"letterspace={round(1000 * tracking)}"])
    doc = doc.wrap_body("\\lsstyle", "")
    return doc


def set_word_spacing(doc: LatexDocument, spacing=1) -> LatexDocument:
    """Set the word spacing in a LaTeX document."""
    doc = doc.wrap_body(f"\\fontdimen2\\font={round(spacing, 2)}\\fontdimen2\\font", "")
    return doc


def set_text_alignment(doc: LatexDocument, alignment="left") -> LatexDocument:
    """Set the text alignment in a LaTeX document."""
    # raggedright, raggedleft, center, flushleft, flushright
    # https://www.overleaf.com/learn/latex/Text_alignment
    # return doc.wrap_body("\\raggedright", "")


def set_text_color(doc: LatexDocument, color) -> LatexDocument:
    """Set the text color in a LaTeX document."""
    color = to_xcolor(color)
    doc = doc.with_package("xcolor")
    doc = doc.wrap_body(f"\\color{{{color}}}", "")
    return doc


def set_page_color(doc: LatexDocument, color) -> LatexDocument:
    """Set the page color in a LaTeX document."""
    color = to_xcolor(color)
    doc = doc.with_package("xcolor")
    doc = doc.wrap_body(f"\\pagecolor{{{color}}}", "")
    return doc


def remove_documentclass_options(doc: LatexDocument, *options) -> LatexDocument:
    """Remove documentclass options in a LaTeX document."""
    updated = [opt for opt in doc.documentclass.options if opt not in options]
    return doc.with_documentclass(doc.documentclass.style, updated)
