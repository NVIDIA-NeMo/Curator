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

import logging
import math
import random
import re

from . import functional as F
from .color import contrast_ratio, hue_shift, from_xcolor, to_xcolor
from .fonts import PDFLATEX_FONTS
from .latex_compiler import LatexCompileError, compile_latex
from .latex_parser import LatexDocument, LatexError, get_field_text


logger = logging.getLogger(__name__)


class Transform:
    """Transform the document."""

    def __init__(self, *, p: float = 1.0):
        assert 0.0 <= p <= 1.0, f"probability out of range: {p}"
        self.p = p

    def __call__(
        self, doc: bytes | LatexDocument, *, rng: random.Random = None
    ) -> LatexDocument:
        if isinstance(doc, bytes):
            return self(LatexDocument(doc), rng=rng)
        if rng is None:
            rng = random.Random()
        if rng.random() > self.p:
            return doc
        return self.apply(doc, rng=rng)

    def apply(self, doc: LatexDocument, *, rng: random.Random) -> LatexDocument:
        raise NotImplementedError


class Compose:
    """Compose multiple transforms together."""

    def __init__(self, transforms: list[Transform], *, p_any: float = None):
        """
        :param p_any: Probability of applying any of the transforms.
        """
        self.transforms = transforms
        if p_any is not None:
            import numpy as np
            from scipy.optimize import brentq

            original_probs = np.array([t.p for t in transforms])
            objective = (
                lambda scale: 1
                - np.prod(1 - np.minimum(1.0, scale * original_probs))
                - p_any
            )
            scale = brentq(objective, 0, 10)
            for t in self.transforms:
                t.p = min(1.0, scale * t.p)
                assert 0.0 <= t.p <= 1.0, f"probability out of range: {t.p}"

    def __call__(
        self, doc: LatexDocument, *, rng: random.Random = None
    ) -> LatexDocument:
        """Apply all transforms to the LaTeX document."""
        if isinstance(doc, bytes):
            return self(LatexDocument(doc), rng=rng)
        for t in self.transforms:
            doc = t(doc, rng=rng)
        return doc

    def check(
        self,
        doc: LatexDocument,
        *,
        dirname: str = None,
        latex: str = "pdflatex",
        bibtex: str = "bibtex",
        retries: int = 3,
        verbose: bool = False,
    ) -> tuple[LatexDocument, bytes]:
        """Run transformation and check if the LaTeX compiles.
        :param latex: LaTeX document to transform.
        :param dirname: Directory for LaTeX document files.
        :param pdflatex: Path to pdflatex executable.
        :param bibtex: Path to bibtex executable.
        :param retries: Number of retries if the LaTeX does not compile.
        """
        for retry in range(retries):
            transformed = self(doc)
            try:
                pdf = compile_latex(
                    transformed.source,
                    dirname=dirname,
                    latex=latex,
                    bibtex=bibtex,
                    quick=True,
                    verbose=verbose,
                )
                return transformed, pdf
            except LatexCompileError:
                if retry == retries - 1:
                    raise


class RemoveBibliography(Transform):
    """Remove bibliography from the document."""

    def apply(self, doc: LatexDocument, *, rng: random.Random) -> LatexDocument:
        return F.remove_bibliography(doc)


class RemoveCaptions(Transform):
    """Remove captions from the document."""

    def apply(self, doc: LatexDocument, *, rng: random.Random) -> LatexDocument:
        try:
            return F.remove_captions(doc)
        except LatexError as err:
            logger.warning("%s: %s", self.__class__.__name__, err)
            return doc


class RemoveHeadings(Transform):
    """Remove headings from the document."""

    def apply(self, doc: LatexDocument, *, rng: random.Random) -> LatexDocument:
        try:
            return F.remove_headings(doc)
        except LatexError as err:
            logger.warning("%s: %s", self.__class__.__name__, err)
            return doc


class RemovePageNumbers(Transform):
    """Remove page numbers from the document."""

    def apply(self, doc: LatexDocument, *, rng: random.Random) -> LatexDocument:
        return F.remove_page_numbers(doc)


class RandomPageSize(Transform):
    """Randomize the document page size.
    :param max_size: Maximum page size (long side) in mm.
    :param max_short_size: Maximum short side of page size in mm.
    :param scale: Scale factor for page size sampling.
    """

    def __init__(
        self, *, p: float = 1.0, max_size=800, max_short_size=None, scale: float = 1.0
    ):
        self.max_size = max_size
        self.max_short_size = max_short_size
        self.scale = scale
        super().__init__(p=p)

    @staticmethod
    def sample(
        rng: random.Random, *, max_size=800, max_short_size=None, scale: float = 1.0
    ):
        # mean at 1.414
        aspect_ratio = math.exp(rng.uniform(math.log(0.667), math.log(3)))
        portrait = rng.random() < 0.8
        if portrait and aspect_ratio > 1:
            aspect_ratio = 1 / aspect_ratio
        # mean at 210mm (letter & A4)
        scale *= math.exp(rng.normalvariate(0, 0.2))
        width = 210 * scale * math.sqrt(1.414 * aspect_ratio)
        height = width / aspect_ratio
        # limit max size keeping aspect ratio
        max_short_size = max_short_size or max_size
        max_width = max_short_size if portrait else max_size
        max_height = max_size if portrait else max_short_size
        scale_factor = min(max_width / width, max_height / height, 1.0)
        width *= scale_factor
        height *= scale_factor
        return max(width, 50), max(height, 50)

    def apply(self, doc: LatexDocument, *, rng: random.Random) -> LatexDocument:
        width, height = self.sample(
            rng,
            max_size=self.max_size,
            max_short_size=self.max_short_size,
            scale=self.scale,
        )
        show_crop = rng.random() < 0.1
        show_frame = rng.random() < 0.1
        doc = F.set_page_size_and_margins(
            doc,
            width=width,
            height=height,
            show_crop=show_crop,
            show_frame=show_frame,
        )
        if rng.random() < 0.3:
            doc = wrap_floats_adjustbox(doc)
        return doc


class RandomPageMargins(Transform):
    """Randomize the document page margins (assuming A4 page size)."""

    @staticmethod
    def sample(width, height, rng: random.Random):
        def samp(min, mean):
            return rng.uniform(min, min + 2 * (mean - min))

        left = samp(3, width / 9)
        right = samp(3, width / 9)
        top = samp(3, height / 9)
        bottom = samp(3, height / 6)
        return top, bottom, right, left

    def apply(self, doc: LatexDocument, *, rng: random.Random) -> LatexDocument:
        top, bottom, right, left = self.sample(210, 297, rng)
        show_crop = rng.random() < 0.1
        show_frame = rng.random() < 0.1
        doc = F.set_page_size_and_margins(
            doc,
            top=top,
            bottom=bottom,
            right=right,
            left=left,
            show_crop=show_crop,
            show_frame=show_frame,
        )
        if rng.random() < 0.3:
            doc = wrap_floats_adjustbox(doc)
        return doc


class RandomPageSizeAndMargins(Transform):
    """Randomize the document page size and margins."""

    def __init__(
        self, *, p: float = 1.0, max_size=800, max_short_size=None, scale: float = 1.0
    ):
        self.max_size = max_size
        self.max_short_size = max_short_size
        self.scale = scale
        super().__init__(p=p)

    def apply(self, doc: LatexDocument, *, rng: random.Random) -> LatexDocument:
        width, height = RandomPageSize.sample(
            rng,
            max_size=self.max_size,
            max_short_size=self.max_short_size,
            scale=self.scale,
        )
        top, bottom, right, left = RandomPageMargins.sample(width, height, rng)
        show_crop = rng.random() < 0.1
        show_frame = rng.random() < 0.1
        doc = F.set_page_size_and_margins(
            doc,
            width=width,
            height=height,
            top=top,
            bottom=bottom,
            right=right,
            left=left,
            show_crop=show_crop,
            show_frame=show_frame,
        )
        if rng.random() < 0.3:
            doc = wrap_floats_adjustbox(doc)
        return doc


class RandomColumnLayout(Transform):
    """Randomize the document column count and layout."""

    def apply(self, doc: LatexDocument, *, rng: random.Random) -> LatexDocument:
        # The multicol package cannot be used with revtex* or IEEEtran
        if (
            doc.documentclass.style.startswith("revtex")
            or doc.documentclass.style == "IEEEtran"
        ):
            return doc
        ncols = rng.randint(1, 3)
        if ncols < 3:
            doc = F.remove_documentclass_options(doc, "onecolumn", "twocolumn")
            doc = doc.with_documentclass(
                doc.documentclass.style,
                doc.documentclass.options
                + ["onecolumn" if ncols == 1 else "twocolumn"],
            )
            return doc
        # don't nest multicols
        if b"\\begin{multicols" in doc.source:
            return doc
        columnsep = rng.lognormvariate(0, 0.2)
        # mean at 1pt
        rulewidth = math.exp(rng.uniform(math.log(0.2), math.log(5)))
        rulewidth = rulewidth * (rng.random() < 0.1)
        balanced = rng.random() < 0.8
        doc = F.set_column_layout(
            doc,
            ncols=ncols,
            columnsep=columnsep,
            rulewidth=rulewidth,
            balanced=balanced,
        )
        # Floats and marginpars not allowed inside multicols environment
        doc = wrap_floats_wrapfig(doc, rng)
        return doc


class RandomSubsectionColumnLayout(Transform):
    """Randomize the column count and layout in each subsection."""

    def apply(self, doc: LatexDocument, *, rng: random.Random) -> LatexDocument:
        # The multicol package cannot be used with revtex4* or IEEEtran
        if (
            doc.documentclass.style.startswith("revtex4")
            or doc.documentclass.style == "IEEEtran"
        ):
            return doc
        # don't nest multicols
        if b"\\begin{multicols" in doc.source:
            return doc
        doc = F.set_column_layout_preamble(doc)
        editor = doc.editor()
        for span in doc.subsections:
            ncols = rng.randint(1, 3)
            columnsep = rng.lognormvariate(0, 0.2)
            # mean at 1pt
            rulewidth = math.exp(rng.uniform(math.log(0.2), math.log(5)))
            rulewidth = rulewidth * (rng.random() < 0.1)
            balanced = rng.random() < 0.8
            begin, end = F.column_layout_commands(ncols, columnsep, rulewidth, balanced)
            editor.wrap(span, begin, end)
        doc = editor.execute()
        # Floats and marginpars not allowed inside multicols environment
        doc = wrap_floats_wrapfig(doc, rng)
        return doc


class RandomSubsectionTextColor(Transform):
    """Randomize text color in each subsection."""

    def apply(self, doc: LatexDocument, *, rng: random.Random) -> LatexDocument:
        doc = doc.with_package("xcolor")
        try:
            pagecolor, _ = parse_page_color(doc)
        except ValueError as err:
            # unknown/unsupported color => skip
            logger.warning("%s: %s", self.__class__.__name__, err)
            return doc
        editor = doc.editor()
        for span in doc.subsections:
            color = sample_color(rng, pagecolor)
            editor.wrap(span, f"\\color{{{to_xcolor(color)}}}\n", "")
        return editor.execute()


class RandomFont(Transform):
    """Randomize the document font."""

    def apply(self, doc: LatexDocument, *, rng: random.Random) -> LatexDocument:
        if "CJKutf8" in doc.packages:
            return self._apply_cjk(doc, rng=rng)
        else:
            return self._apply_latin(doc, rng=rng)

    def _apply_latin(self, doc: LatexDocument, *, rng: random.Random) -> LatexDocument:
        font = rng.choice(list(PDFLATEX_FONTS.keys()))
        doc = F.set_font(doc, font)
        if rng.random() < 0.3:
            doc = wrap_floats_adjustbox(doc)
        return doc

    def _apply_cjk(self, doc: LatexDocument, *, rng: random.Random) -> LatexDocument:
        font_map = {
            # chinese
            "gbsn": "gbsn gkai".split(),
            # japanese
            "min": "min goth".split(),
            # korean
            "mj": "mj".split(),
        }
        font_re = re.compile(rb"\\begin{CJK\*}{UTF8}{(.*?)}")
        font_match = font_re.search(doc.source)
        if not font_match:
            logger.warning("%s: No CJK font found", self.__class__.__name__)
            return doc
        current_font = font_match.group(1).decode("utf8")
        if current_font not in font_map:
            logger.warning(
                "%s: Unknown CJK font: %s", self.__class__.__name__, current_font
            )
            return doc
        font = rng.choice(font_map[current_font])
        if font == current_font:
            return doc
        doc = LatexDocument(
            font_re.sub(
                # NB. backslash needs to be escaped in the replacement string!
                rb"\\begin{CJK*}{UTF8}{" + font.encode() + b"}", doc.source
            )
        )
        return doc


class RandomFontSize(Transform):
    """Randomize the document font size."""

    def apply(self, doc: LatexDocument, *, rng: random.Random) -> LatexDocument:
        # remove 10pt/11pt/12pt class
        doc = F.remove_documentclass_options(doc, "10pt", "11pt", "12pt")
        # sample font size proportional to text (line) width
        try:
            textwidth, _ = parse_text_area(doc)
        except ValueError:
            logger.warning("%s: assuming A4 page size", self.__class__.__name__)
            textwidth = 210 * 7 / 9

        # mean 11pt
        min_size = 5  # pt
        max_size = 24  # pt

        # limit the maximum number of characters per line
        min_size = max(min_size, textwidth / 15)
        if min_size > max_size:
            max_size = min_size
        size = math.exp(rng.uniform(math.log(min_size), math.log(max_size)))
        doc = F.set_font_size(doc, size)
        if rng.random() < 0.3:
            doc = wrap_floats_adjustbox(doc)
        return doc


def parse_text_area(doc: LatexDocument) -> tuple[float, float]:
    if isinstance(doc, bytes):
        doc = LatexDocument(doc)
    geom = parse_geometry(doc)
    paperwidth = geom["paperwidth"]
    paperheight = geom["paperheight"]
    left = geom.get("left", paperwidth / 9)
    right = geom.get("right", paperwidth / 9)
    top = geom.get("top", paperheight / 9)
    bottom = geom.get("bottom", paperheight / 6)
    textwidth = paperwidth - left - right
    textheight = paperheight - top - bottom
    return textwidth, textheight


def parse_geometry(doc: LatexDocument) -> dict[str, float]:
    # \geometry{paperwidth=218.6mm,paperheight=288.9mm,top=16.8mm,bottom=5.8mm,right=35.6mm,left=6.6mm,ignoremp}
    geom = doc.find(lambda node: node.text == b"\\geometry")
    kvs = geom.next_sibling.text[1:-1].decode("utf8").split(",")
    kvs = [kv.split("=", 1) for kv in kvs if "=" in kv]
    kvs = {k: float(v[:-2]) for k, v in kvs if v.endswith("mm")}
    if "paperwidth" not in kvs or "paperheight" not in kvs:
        raise ValueError("No \\geometry{...} found")
    return kvs


class RandomTextColor(Transform):
    """Randomize the document text color."""

    def apply(self, doc: LatexDocument, *, rng: random.Random) -> LatexDocument:
        try:
            pagecolor, _ = parse_page_color(doc)
        except ValueError as err:
            # unknown/unsupported color => skip
            logger.warning("%s: %s", self.__class__.__name__, err)
            return doc
        color = sample_color(rng, pagecolor)
        return F.set_text_color(doc, color)


class RandomPageColor(Transform):
    """Randomize the document page color."""

    def apply(self, doc: LatexDocument, *, rng: random.Random) -> LatexDocument:
        try:
            textcolor, _ = parse_text_color(doc)
        except ValueError as err:
            # unknown/unsupported color => skip
            logger.warning("%s: %s", self.__class__.__name__, err)
            return doc
        color = sample_color(rng, textcolor)
        return F.set_page_color(doc, color)


def sample_color(rng: random.Random, other: list[float]) -> list[float]:
    """Sample color with minimum contrast ratio to other color."""
    for _ in range(20):
        color = [rng.random() for _ in range(3)]
        # W3C requirement is 4.5:1 but we want to support also low contrast
        # (but not unity contrast because that encourages hallucination)
        if contrast_ratio(color, other) > 2:
            return color
    raise ValueError("Failed to sample color")


class RandomSepiaPageColor(Transform):
    """Randomize the document page color to sepia tint."""

    def apply(self, doc: LatexDocument, *, rng: random.Random) -> LatexDocument:
        color = [135 / 255, 87 / 255, 43 / 255]  # 87572b
        # shift hue randomly
        shift = rng.uniform(-10, 10)
        color = hue_shift(color, shift)
        # vary lightness randomly (interpolate with white)
        scale = rng.random()
        color = [scale * c + (1 - scale) for c in color]
        return F.set_page_color(doc, color)


class RandomInvertedColors(Transform):
    """Randomly invert page and text colors."""

    def apply(self, doc: LatexDocument, *, rng: random.Random) -> LatexDocument:
        doc = doc.with_package("xcolor")
        try:
            pagecolor, pagecolor_span = parse_page_color(doc)
            textcolor, textcolor_span = parse_text_color(doc)
        except ValueError as err:
            # unknown/unsupported color => skip
            logger.warning("%s: %s", self.__class__.__name__, err)
            return doc
        editor = doc.editor()
        editor.replace_at(
            pagecolor_span, f"\\pagecolor{{{to_xcolor(textcolor)}}}\n".encode()
        )
        editor.replace_at(
            textcolor_span, f"\\color{{{to_xcolor(pagecolor)}}}\n".encode()
        )
        return editor.execute()


def parse_page_color(doc: LatexDocument) -> tuple[str, slice]:
    try:
        node = doc.find(lambda node: node.text == b"\\pagecolor")
        pos = node.end_byte
        while pos < len(doc.source) and doc.source[pos] != ord(b"}"):
            pos += 1
        pagecolor = doc.source[node.end_byte + 1 : pos].decode("utf8")
        pagecolor_span = slice(node.start_byte, pos + 1)
    except ValueError:
        pagecolor = "white"
        pagecolor_span = slice(doc.body_start, doc.body_start)
    return from_xcolor(pagecolor), pagecolor_span


def parse_text_color(doc: LatexDocument) -> tuple[str, slice]:
    try:
        node = doc.find(lambda node: node.text == b"\\color")
        pos = node.end_byte
        while pos < len(doc.source) and doc.source[pos] != ord(b"}"):
            pos += 1
        textcolor = doc.source[node.end_byte + 1 : pos].decode("utf8")
        textcolor_span = slice(node.start_byte, pos + 1)
    except ValueError:
        textcolor = "black"
        textcolor_span = slice(doc.body_start, doc.body_start)
    return from_xcolor(textcolor), textcolor_span


class RandomPageBackground(Transform):
    """Add random lines, rectangles, etc. to the background."""

    # https://tex.stackexchange.com/questions/82206/background-color-gradient-for-entire-document/82237#82237

    def apply(self, doc: LatexDocument, *, rng: random.Random) -> LatexDocument:
        color1 = to_xcolor([rng.random() for _ in range(3)])
        color2 = to_xcolor([rng.random() for _ in range(3)])
        color3 = to_xcolor([rng.random() for _ in range(3)])
        angle = rng.uniform(0, 360)
        opacity = rng.uniform(0, 1)
        num_lines = rng.randint(0, 30)
        num_rectangles = rng.randint(0, 20)
        drawrect = "\\filldraw" if rng.random() < 0.5 else "\\draw"
        doc = doc.with_package("background")
        doc = doc.with_package("tikz")
        doc = doc.wrap_body(
            f"""
\\backgroundsetup{{
scale=1,
angle=0,
opacity={opacity},
position={{0.5\\paperwidth,0.5\\paperheight}},
contents={{%
\\begin{{tikzpicture}}[remember picture,overlay]
  \\path[transform canvas={{rotate={angle}}},
         left color = {{{color1}}},
         middle color = {{{color2}}}, 
         right color = {{{color3}}}] 
         (current page.south west) rectangle 
         (current page.north east);
  \\foreach \\i in {{1,...,{num_lines}}} {{
    \\draw[black, line width=rand*1pt]
        (2*rand*\\paperwidth, 2*rand*\\paperheight) -- 
        (2*rand*\\paperwidth, 2*rand*\\paperheight);
  }}
  \\foreach \\i in {{1,...,{num_rectangles}}} {{
    {drawrect}[draw={{{color1}}}, fill={{{color1}}}, line width=0.5pt]
        (rand*\\paperwidth, rand*\\paperheight) 
        rectangle ++
        (rand*0.5*\\paperwidth, rand*0.5*\\paperheight);
  }}
\\end{{tikzpicture}}}}
}}
""",
            "",
        )
        return doc


class RandomLineSpacing(Transform):
    """Randomize the document line spacing."""

    def apply(self, doc: LatexDocument, *, rng: random.Random) -> LatexDocument:
        # mean should be 1.0
        spacing = math.exp(rng.uniform(math.log(0.6), math.log(1.67)))
        return F.set_line_spacing(doc, spacing)


class RandomParagraphMargins(Transform):
    """Randomize the document paragraph margins."""

    # \begingroup
    # \leftskip4em
    # \rightskip\leftskip
    # \blindtext
    # \par
    # \endgroup


class RandomLetterSpacing(Transform):
    """Randomize the document letter spacing (tracking)."""

    def apply(self, doc: LatexDocument, *, rng: random.Random) -> LatexDocument:
        # mean should be 0.1
        spacing = rng.normalvariate(0.1, 0.1)
        return F.set_letter_spacing(doc, spacing)


class RandomWordSpacing(Transform):
    """Randomize the document word spacing."""

    def apply(self, doc: LatexDocument, *, rng: random.Random) -> LatexDocument:
        # mean should be 1.0
        spacing = rng.lognormvariate(0, 0.2)
        return F.set_word_spacing(doc, spacing)


class RandomTextAlignment(Transform):
    """Randomize the document text alignment."""

    def apply(self, doc: LatexDocument, *, rng: random.Random) -> LatexDocument:
        align = rng.choice(["none", "left", "right", "center"])
        if align == "none":
            return doc
        if align == "right":
            doc = doc.with_package("ragged2e", ["document"])
        else:
            doc = doc.with_package("ragged2e")
            env = "FlushLeft" if align == "left" else "Center"
            doc = doc.wrap_body(f"\\begin{{{env}}}", f"\\end{{{env}}}")
        return doc


class RandomFloatRotation(Transform):
    """Rotate figures and tables randomly."""

    def apply(self, doc: LatexDocument, *, rng: random.Random) -> LatexDocument:
        doc = wrap_floats_adjustbox(doc)
        doc = doc.with_package("rotating")
        editor = doc.editor()
        for node in doc.floats:
            if rng.random() < 0.5:
                continue
            angle = rng.choice([0, 90, 180, 270])
            if angle == 0:
                continue
            editor.wrap(
                node,
                f"\\begin{{turn}}{{{angle}}}\n",
                "\\end{turn}\n",
                inner=True,
            )
        return editor.execute()


class RandomTableColumnSeparators(Transform):
    """Randomize the document table column separators."""

    def apply(self, doc: LatexDocument, *, rng: random.Random) -> LatexDocument:
        nodes = list(
            doc.findall(
                lambda node: node.type == "generic_environment"
                and node.text.startswith(b"\\begin{tabular}")
            )
        )
        editor = doc.editor()
        for node in nodes:
            if (
                node.children[0].type != "begin"
                or node.children[0].children[2].type != "curly_group"
            ):
                continue
            spec_node = node.children[0].children[2]
            spec = split_latex_column_spec(spec_node.text[1:-1].decode("utf8"))
            spec = [col for col in spec if col != "|"]
            separator_prob = rng.random()
            double_prob = rng.random()

            def sample_sep():
                if rng.random() < separator_prob:
                    if rng.random() < double_prob:
                        return "||"
                    return "|"
                return ""

            spec = "".join(sample_sep() + col for col in spec)
            spec += sample_sep()
            arg = ("{" + spec + "}").encode("utf-8")
            editor.replace_at(slice(*spec_node.byte_range), arg)
        return editor.execute()


def split_latex_column_spec(spec):
    pattern = r"""
        (
            \*\{[^{}]*\}\{[^{}]*\} |  # *{num}{cols}
            [lrc]                  |  # Basic types
            [pmbw]\{[^}]*\}        |  # Width columns
            [@!]\{[^}]*\}          |  # @{...} and !{...} modifiers
            [<>]\{[^}]*\}          |  # >{...} and <{...} array package modifiers
            \|                     |  # Vertical lines
            \|{2,}                    # Double/multiple lines
        )
        \s*                           # Trailing whitespace
    """
    return re.findall(pattern, spec, flags=re.VERBOSE)


def wrap_floats_adjustbox(doc: LatexDocument) -> LatexDocument:
    """Wrap float in adjustbox to try and prevent tables from overflowing."""
    doc = doc.with_package("adjustbox")
    editor = doc.editor()
    for node in doc.floats:
        name = get_field_text(node, "begin.name.text", "")
        width = "\\textwidth" if name.endswith("*") else "\\columnwidth"
        if b"\\begin{adjustbox}" in node.text:
            continue
        editor.wrap(
            node,
            f"""
\\begin{{adjustbox}}{{max width={width},max totalheight=\\textheight}}
\\begin{{minipage}}{{{width}}}
""",
            "\\end{minipage}\\end{adjustbox}\n",
            inner=True,
        )
    return editor.execute()


def wrap_floats_wrapfig(doc: LatexDocument, rng: random.Random) -> LatexDocument:
    """Convert floats to use wrapfig."""
    doc = doc.with_package("wrapfig")
    editor = doc.editor()
    for node in doc.floats:
        begin = node.child_by_field_name("begin")
        end = node.child_by_field_name("end")
        if not begin and end:
            continue
        float_type = get_field_text(begin, "name.text", "")
        width = "\\textwidth" if float_type.endswith("*") else "\\columnwidth"
        width = str(round(rng.uniform(0.5, 1.0), 2)) + width
        wrap_type = "wrap" + float_type.replace("*", "")
        position = rng.choice(["L", "R"])
        wrap_begin = f"\\begin{{{wrap_type}}}{{{position}}}{{{width}}}"
        wrap_end = f"\\end{{{wrap_type}}}"
        editor.replace_at(slice(*begin.byte_range), wrap_begin.encode())
        editor.replace_at(slice(*end.byte_range), wrap_end.encode())
    return editor.execute()


class RandomUnicodeCharacters(Transform):
    """Randomize the document Unicode characters."""

    def apply(self, doc: LatexDocument, *, rng: random.Random) -> LatexDocument:
        nodes = list(doc.findall(lambda node: node.type == "word"))
        # if rng.random() < 0.1:
        breakpoint()
        return doc
