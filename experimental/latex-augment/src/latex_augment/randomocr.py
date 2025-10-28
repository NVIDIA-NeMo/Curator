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

r"""
1000 docs in 1 min on my laptop
"""

import gzip
import hashlib
import io
import logging
import math
import re
import string
from itertools import accumulate
from random import Random

import pymupdf
from PIL import Image
from sequence_align.pairwise import hirschberg

from latex_augment import LatexDocument
from latex_augment import fonts as F
from latex_augment import transforms as T
from latex_augment.convert.markdown import escape_markdown
from latex_augment.latex_characters import (
    get_unicode_charset,
    intersect_charset,
    subtract_charset,
)
from latex_augment.latex_compiler import LatexCompileError, compile_latex
from latex_augment.latex_symbols import (
    ASCII_NUMBERS,
    ASCII_PUNCTUATION,
    CHINESE_PUNCTUATION,
    MULTISCRIPT_CHARSET,
    SYMBOL_CHARSET,
)
from latex_augment.transforms import parse_geometry


SUBSTITUTE_CHARS = "\x00\ufffd\ufffe\uffff"
UNKNOWN_SYMBOL = "\\<|unk|\\>"
MAX_WIDTH = 1024
MAX_HEIGHT = 1280


def upweight_cursive_fonts(fonts):
    return 9 * list(fonts) + list(F.filter_cursive_fonts(fonts))


# minimum legible resolution for 10 point font
MIN_DPI = {
    "ascii": 150,
    "english": 150,
    "latin": 150,
    "greek": 150,
    "chinese": 300,
    "japanese": 300,
    "korean": 300,
}

# add up to 2x high res images without changing proportion of font and page size
MAX_DPI_BOOST = 2.0

FONTS = {
    "ascii": upweight_cursive_fonts(F.list_latin_fonts()),
    "english": upweight_cursive_fonts(F.list_latin_fonts()),
    "latin": upweight_cursive_fonts(F.list_latin_fonts()),
    "greek": upweight_cursive_fonts(F.list_greek_fonts()),
    "chinese": F.list_simplified_chinese_fonts(),
    "japanese": F.list_japanese_fonts(),
    "korean": F.list_korean_fonts(),
}

ENGLISH_WORDS = gzip.open("webster1913_english_words.txt.gz", "rt").read().splitlines()


class GenerationFailed(Exception):
    pass


def generate_sample(script, sample_id):
    # hash script & sample id for seeding
    seed = md5_int32(f"{script}:{sample_id}:randomocr")
    rng = Random(seed)
    while True:
        try:
            latex_template, font, alt_font = sample_latex_template(script, rng=rng)
            font_charset = subtract_charset(font.charset, SUBSTITUTE_CHARS)
            alt_font_charset = subtract_charset(alt_font.charset, SUBSTITUTE_CHARS)
            english_prob = rng.choice([0.0, 0.01, 0.03, 0.1])
            multiscript_prob = rng.choice([0.0, 0.01, 0.03, 0.1])
            number_prob = rng.choice([0.0, 0.01, 0.03, 0.1])
            punctuation_prob = rng.choice([0.01, 0.03, 0.1, 0.3])
            symbol_prob = rng.choice([0.0, 0.01, 0.03, 0.3])
            text = generate_text(
                2000,
                rng=rng,
                script=script,
                font_charset=font_charset,
                alt_font_charset=alt_font_charset,
                english_prob=english_prob,
                multiscript_prob=multiscript_prob,
                number_prob=number_prob,
                punctuation_prob=punctuation_prob,
                symbol_prob=symbol_prob,
            )
            latex, text, pdf, line_count = truncate_latex(latex_template, text)
            make_pixelated = rng.random() < 0.03
            if make_pixelated:
                markdown = text = re.sub(r"\S", UNKNOWN_SYMBOL, text)
            else:
                markdown = escape_markdown(text)
            dpi = MIN_DPI[script] * (MAX_DPI_BOOST ** rng.random())
            label = extract_label(
                latex, markdown, dpi=dpi, font=font.name, alt_font=alt_font.name
            )
            image = extract_image_from_pdf(pdf, page=0, dpi=dpi)
            assert (
                abs(label["metadata"]["page_width_px"] - image.size[0]) < 3
            ), "image width mismatch"
            assert (
                abs(label["metadata"]["page_height_px"] - image.size[1]) < 3
            ), "image height mismatch"
            bbox_height = label["ann"][0]["bbox"][3] - label["ann"][0]["bbox"][1]
            line_height = bbox_height / line_count
            if line_height < 12 * dpi / 150:
                raise GenerationFailed("line height is too small (risk of pixelation)")
            if make_pixelated:
                # downsample to ~6 pixel line height
                scale = rng.uniform(6, 7) / line_height
                image = pixelate_image(image, scale=scale)
            png = to_png(image)
            return sample_id, png, label, latex, text, pdf
        except (LatexCompileError, GenerationFailed) as e:
            logging.error(f"Error in {sample_id}: {e} (ignored)")
        except Exception as e:
            logging.exception(f"Unhandled exception in {sample_id}: {e}")
            raise


def extract_label(latex: bytes, text: str, *, dpi: float, font: str, alt_font: str):
    geom = parse_geometry(LatexDocument(latex))
    paperwidth = geom["paperwidth"]
    paperheight = geom["paperheight"]
    left = geom["left"]
    right = geom["right"]
    top = geom["top"]
    bottom = geom["bottom"]
    mm_to_px = dpi / 25.4
    bbox = [
        round(mm_to_px * left),
        round(mm_to_px * top),
        round(mm_to_px * (paperwidth - right)),
        round(mm_to_px * (paperheight - bottom)),
    ]

    label = {
        "metadata": {
            "dpi": dpi,
            "font": font,
            "alt_font": alt_font,
            "page_width_px": round(mm_to_px * paperwidth),
            "page_height_px": round(mm_to_px * paperheight),
        },
        "ann": [
            {
                "category_id": "Text",
                "bbox": bbox,
                "content": text,
            },
        ],
    }
    return label


def extract_image_from_pdf(pdf_buf, *, page=0, dpi=72):
    """Extract a PIL image from a PDF page.
    :param dpi: DPI for the image.
    """
    pdf_document = pymupdf.open("pdf", pdf_buf)
    page = pdf_document.load_page(page)
    zoom_factor = 8  # render at 576 DPI
    mat = pymupdf.Matrix(zoom_factor, zoom_factor)
    pix = page.get_pixmap(matrix=mat)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    # downsample to requested DPI
    width = round(image.size[0] * dpi / 72 / zoom_factor)
    height = round(image.size[1] * dpi / 72 / zoom_factor)
    # FIXME PIL generates artifacts at right and bottom edges when downsampling
    # FIXME should expand with background color before downsampling
    # use Hamming filter to avoid ringing, especially around condensed fonts
    image = image.resize((width, height), Image.Resampling.HAMMING)
    return image


def pixelate_image(image, *, scale):
    lowres_size = (int(image.size[0] * scale), int(image.size[1] * scale))
    lowres_image = image.resize(lowres_size, Image.Resampling.HAMMING)
    # FIXME PIL generates artifacts at right and bottom edges when downsampling
    # FIXME should expand with background color before downsampling
    image = lowres_image.resize(image.size, Image.Resampling.NEAREST)
    return image


def to_png(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def md5_int32(text):
    """Hash text into signed 32-bit integer"""
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def generate_text(
    num_words: int,
    *,
    rng: Random,
    script: str,
    font_charset: str,
    alt_font_charset: str,
    english_prob: float = 0.0,
    number_prob: float = 0.0,
    multiscript_prob: float = 0.0,
    punctuation_prob: float = 0.0,
    symbol_prob: float = 0.0,
) -> str:
    """Generate gibberish test in given script and charset.
    The text is plain text with some html-ish tags for alt fonts and hyphenation.
    """
    main_charset = filter_charset(script, font_charset) if script != "english" else None
    punctuation_charset = intersect_charset(
        CHINESE_PUNCTUATION if script == "chinese" else ASCII_PUNCTUATION, font_charset
    )
    if not punctuation_charset:
        punctuation_prob = 0.0
    symbols = intersect_charset(SYMBOL_CHARSET, font_charset)
    alt_symbols = intersect_charset(SYMBOL_CHARSET, alt_font_charset)
    if not (symbols or alt_symbols):
        symbol_prob = 0.0
    multiscripts = [
        intersect_charset(multiscript_charset, font_charset)
        for multiscript_charset in MULTISCRIPT_CHARSET.values()
    ]
    multiscripts = [x for x in multiscripts if x]
    alt_multiscripts = [
        intersect_charset(multiscript_charset, alt_font_charset)
        for multiscript_charset in MULTISCRIPT_CHARSET.values()
    ]
    alt_multiscripts = [x for x in alt_multiscripts if x]
    if not (multiscripts or alt_multiscripts):
        multiscript_prob = 0.0
    text = ""
    for _ in range(num_words):
        word_type = rng.choices(
            ["english", "number", "multiscript", "word"],
            weights=[
                english_prob,
                number_prob,
                multiscript_prob,
                1 - english_prob - number_prob - multiscript_prob,
            ],
        )[0]
        use_alt_font = False
        if word_type == "english":
            word = random_english_word(rng=rng)
            use_alt_font = rng.random() < 0.5
        elif word_type == "number":
            word = random_word(ASCII_NUMBERS, rng=rng, repetition=True)
            use_alt_font = rng.random() < 0.5
        elif word_type == "multiscript":
            if multiscripts and (not alt_multiscripts or rng.random() < 0.5):
                word = random_word(rng.choice(multiscripts), rng=rng)
            else:
                word = random_word(rng.choice(alt_multiscripts), rng=rng)
                use_alt_font = True
        elif word_type == "word" and script == "english":
            word = random_english_word(rng=rng)
        else:
            word = random_word(main_charset, rng=rng)
        punct_type = rng.choices(
            ["punctuation", "symbol", "none"],
            weights=[punctuation_prob, symbol_prob, 1 - punctuation_prob - symbol_prob],
        )[0]
        if punct_type == "punctuation":
            punct = rng.choice(punctuation_charset)
        elif punct_type == "symbol":
            if symbols and (not alt_symbols or rng.random() < 0.5):
                punct = rng.choice(symbols)
            else:
                punct = "<alt>" + rng.choice(alt_symbols) + "</alt>"
        else:
            punct = ""
        punct_pos = rng.choices(
            ["word", "before", "after", "nospace", "within"],
            weights=[0.3, 0.3, 0.3, 0.09, 0.01],
        )[0]

        def alt(text):
            return "<alt>" + text + "</alt>" if use_alt_font else text

        if punct_pos == "word":
            text += alt(word) + " " + punct + " "
        elif punct_pos == "before":
            text += punct + alt(word) + " "
        elif punct_pos == "after":
            text += alt(word) + punct + " "
        elif punct_pos == "within":
            pos = rng.randint(0, len(word))
            text += alt(word[:pos]) + punct + alt(word[pos:]) + " "
        else:  # nospace
            text += alt(word) + punct + "<shy/>"  # add hyphenation hint
    # remove double spaces
    text = (
        text.replace(" <alt> ", "<alt> ")
        .replace(" </alt> ", " </alt>")
        .replace("</alt> <alt>", " ")
        .replace("</alt><alt>", "")
    )
    text = " ".join(text.split())
    if script in ("chinese", "japanese", "korean"):
        text = fix_cjk_spaces(text)
    return text


def sample_latex_template(script: str, rng: Random) -> tuple[bytes, F.Font, F.Font]:
    latex = """
\\documentclass[10pt]{article}
\\tracinglostchars=3  % fail if characters are missing
\\XeTeXtracingfonts=1 % log fonts
\\usepackage{hyphenat}
\\usepackage{fontspec}
"""

    if script in ("ascii", "english", "latin"):
        latex += "\\usepackage[english]{babel}\n"
    elif script in ("greek", "chinese", "japanese", "korean"):
        latex += "\\usepackage[" + script + ",provide=*]{babel}\n"
    else:
        raise ValueError(f"Unsupported script: {script}")

    font = rng.choice(FONTS[script])
    alt_font = rng.choice(FONTS["latin"])
    latex += (
        f"%%%font={font.name}@{font.filename}, {alt_font.name}@{alt_font.filename}%%%\n"
    )
    latex += font.fontspec("\\setmainfont") + "\n"
    latex += alt_font.fontspec("\\setsansfont") + "\n"

    latex += """
\\begin{document}
\\pagenumbering{gobble}
\\pagestyle{empty}
%%%BEGIN%%%
%%%END%%%
\\end{document}
"""

    # current fontspec font selection uses same font for bold/italic
    # if rng.random() < 0.2:
    #     latex = latex.replace("%%%BEGIN%%%", "\\textbf{%\n%%%BEGIN%%%")
    #     latex = latex.replace("%%%END%%%\n", "%%%END%%%\n}\n")
    # if rng.random() < 0.2:
    #     latex = latex.replace("%%%BEGIN%%%", "\\textit{%\n%%%BEGIN%%%")
    #     latex = latex.replace("%%%END%%%\n", "%%%END%%%\n}\n")

    # reduce the page size logarithmically (for curriculum learning)
    if rng.random() < 0.5:
        scale = 1 / (10 ** rng.random())
    else:
        scale = 1.0
    # compute page size using minimum DPI to ensure minimum font size in pixels
    dpi = MIN_DPI[script]
    max_width_mm = MAX_WIDTH * 25.4 / dpi
    max_height_mm = MAX_HEIGHT * 25.4 / dpi
    transforms = [
        T.RandomPageSizeAndMargins(
            scale=scale,
            max_size=max(max_width_mm, max_height_mm),
            max_short_size=min(max_width_mm, max_height_mm),
        ),
        T.RandomTextAlignment(),
        T.RandomLineSpacing(),
        T.RandomWordSpacing(),
        T.RandomFontSize(),
        T.RandomSepiaPageColor(p=0.5),
        T.RandomPageColor(p=0.1),
        T.RandomTextColor(p=0.5),
    ]
    # don't use T.RandomLetterSpacing() as microtype letterspacing inserts
    # spaces in pdf content which breaks the target data

    augment = T.Compose(transforms)
    doc = LatexDocument(latex.encode("utf-8"))
    try:
        doc = augment(doc, rng=rng)
    except Exception as e:
        # e.g. "Failed to sample color"
        raise GenerationFailed(f"Augmentation failed: {e}")
    return doc.source, font, alt_font


def filter_charset(script: str, font_charset: str) -> str:
    script_charset = get_script_charset(script)
    charset = intersect_charset(font_charset, script_charset)
    if len(set(script_charset)) >= 100 and len(charset) < 100:
        raise GenerationFailed(
            f"font has too few characters for script: {len(charset)}"
        )
    return charset


def filter_vocabulary(vocabulary: str, charset: str) -> tuple[list[str], list[float]]:
    """Filter vocabulary based on charset supported by font."""
    pattern = rf"^[{re.escape(charset)}]+$"
    filtered = re.findall(pattern, vocabulary, re.MULTILINE)

    # Create CDF for filtered vocabulary using zipf distribution
    weights = [(1 + idx) ** -0.9 for idx in range(len(filtered))]
    cdf = list(accumulate(weights))
    return filtered, cdf


def get_script_charset(script: str) -> str:
    if script == "ascii":
        return 5 * string.ascii_letters + ASCII_PUNCTUATION
    elif script == "greek":
        # https://en.wikipedia.org/wiki/Greek_script_in_Unicode
        return get_unicode_charset(["Greek and Coptic", "Greek Extended"])
    elif script == "latin":
        # https://en.wikipedia.org/wiki/Latin_script_in_Unicode
        return get_unicode_charset(
            [
                "Basic Latin",
                "Latin-1 Supplement",
                "Latin Extended-A",
                "Latin Extended-B",
                "IPA Extensions",
                "Spacing Modifier Letters",
                "Phonetic Extensions",
                "Phonetic Extensions Supplement",
                "Latin Extended Additional",
                "Superscripts and Subscripts",
                "Letterlike Symbols",
                "Number Forms",
                "Latin Extended-C",
                "Latin Extended-D",
                "Latin Extended-E",
                "Alphabetic Presentation Forms",
                "Halfwidth and Fullwidth Forms",
                "Latin Extended-F",
                "Latin Extended-G",
            ]
        )
    elif script == "chinese":
        # https://en.wikipedia.org/wiki/CJK_Unified_Ideographs
        return get_unicode_charset(
            [
                "CJK Unified Ideographs",
                "CJK Unified Ideographs Extension A",
                "CJK Compatibility Forms",
                "CJK Strokes",
            ]
        )
    elif script == "japanese":
        return get_unicode_charset(["Hiragana", "Katakana"])
    elif script == "korean":
        # skip Extended-A and Extended-B as they are archaic
        return get_unicode_charset(["Hangul Syllables", "Hangul Jamo"])
    else:
        raise ValueError(f"Unsupported script: {script}")


def random_english_word(*, rng: Random) -> str:
    word = rng.choice(ENGLISH_WORDS)
    if rng.random() < 0.1:
        word = word[0].upper() + word[1:]
    elif rng.random() < 0.01:
        word = word.upper()
    return word


def random_word(charset, *, rng: Random, repetition: bool = False) -> str:
    # Sigurd et al 2004. "Word length, sentence length and frequency – Zipf revisited"
    # freq(L) = 11.74 * L^3 * 0.4^L
    shape = 4
    scale = -1 / math.log(0.4)
    wordlen = rng.gammavariate(shape, scale)
    if not repetition:
        wordlen *= min(1.0, len(charset) / 128)  # avoid repetition in small charsets
    return random_chars(charset, round(wordlen), rng=rng)


def random_chars(charset, size: int, *, rng: Random) -> str:
    text = ""
    for _ in range(size):
        idx = rng.randint(0, len(charset) - 1)
        text += charset[idx]
    return text


CJK_CHAR_PATTERN = (
    r"["
    r"\u1100-\u11FF"  # Hangul Jamo
    r"\u3040-\u309F"  # Hiragana
    r"\u30A0-\u30FF"  # Katakana
    r"\u3130-\u318F"  # Hangul Compatibility Jamo
    r"\u3400-\u4DBF"  # CJK Unified Ideographs Extension A
    r"\u4E00-\u9FFF"  # CJK Unified Ideographs
    r"\uA960-\uA97F"  # Hangul Jamo Extended-A
    r"\uAC00-\uD7AF"  # Hangul Syllables
    r"\uD7B0-\uD7FF"  # Hangul Jamo Extended-B
    r"\uFF65-\uFF9F"  # Hangul Jamo Extended-C
    r"]"
)
CJK_SPACE_RE = re.compile(f"\\s*({CJK_CHAR_PATTERN})\\s*")


def fix_cjk_spaces(text: str) -> str:
    """Remove spaces around CJK characters"""
    return CJK_SPACE_RE.sub(lambda match: match.group(1), text)


def escape_latex(text: str) -> str:
    return (
        text.replace("\\-", "[HYPHENHINT]")
        .replace("\\", "\\textbackslash[END]")
        .replace("~", "\\textasciitilde[END]")
        # .replace("'", "\\textquotesingle[END]")
        # LaTeX Error: Command \textquotedbl unavailable in encoding OT1.
        # .replace('"', "\\textquotedbl[END]")
        .replace("`", "\\textasciigrave[END]")
        .replace("^", "\\textasciicircum[END]")
        .replace("-", "\\hyp[END]")  # normal hyphen prevents automatic hyphenation
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("[HYPHENHINT]", "\\-")
        .replace("[END]", "{}")
    )


def hyphenate_words(text: str) -> str:
    """Add LaTeX hyphenation hints mechanistically."""
    words = text.split()
    for i, word in enumerate(words):
        if len(word) >= 5:
            parts = [word[i : i + 3] for i in range(0, len(word), 3)]
            words[i] = "\\-".join(parts)
    return " ".join(words)


def truncate_latex(latex_template: bytes, text: str) -> tuple[bytes, str, bytes, bytes]:
    """Truncate latex document until it fits in one page."""
    body_pos = latex_template.index(b"%%%END%%%")

    latex_text = hyphenate_words(text)
    latex_text = (
        escape_latex(text)
        .replace("<alt>", "\\textsf{")
        .replace("</alt>", "}")
        .replace("<shy/>", "\\-")
        .encode("utf-8")
        + b"\n"
    )
    plain_text = text.replace("<alt>", "").replace("</alt>", "").replace("<shy/>", "")

    latex = latex_template[:body_pos] + latex_text + latex_template[body_pos:]
    pdf_buf = compile_latex(latex, latex="xelatex", quick=True)  # , verbose=True)
    pdf_doc = pymupdf.open("pdf", pdf_buf)
    if len(pdf_doc) == 1:
        raise GenerationFailed("initial text is less than one full page")
    font = re.search(b"%%%font=(.*?)%%%", latex_template).group(1).decode("utf-8")
    text, line_count = get_first_page_text(plain_text, pdf_doc, font)
    return latex, text, pdf_buf, line_count


def get_first_page_text(
    text: str, pdf_doc: pymupdf.Document, font: str
) -> tuple[str, int]:
    """Extract prefix of given text that matches the first page of the pdf."""
    pdf_page0_text = pdf_doc.get_page_text(0)
    if any(c in pdf_page0_text for c in SUBSTITUTE_CHARS):
        raise GenerationFailed(f'unprintable character(s) in font "{font}"')
    line_count = pdf_page0_text.count("\n")
    # read ground truth from original text because some fonts (e.g. handwriting)
    # have ligatures, alternative forms, etc. that end up as garbage glyphs in pdf data
    paginated = match_pdf_pagination(text, pdf_doc)
    text = paginated[0]
    if pdf_page0_text.endswith("-\n") and not text.rstrip().endswith("-"):
        text = text.rstrip() + "-"
    return text, line_count


def match_pdf_pagination(text: str, pdf_doc: pymupdf.Document) -> list[str]:
    """Split text into pages that match given PDF document."""
    pdf_texts = [
        pdf_doc.get_page_text(page_num).strip() for page_num in range(len(pdf_doc))
    ]
    pdf_seq = [x for txt in pdf_texts for x in list(txt) + ["<pagebreak>"]]
    pdf_aligned, text_aligned = hirschberg(pdf_seq, list(text), gap="")
    ids = [0] + [i for i, x in enumerate(pdf_aligned) if x == "<pagebreak>"]
    paginated = ["".join(text_aligned[i:j]) for i, j in zip(ids[:-1], ids[1:])]
    return paginated
