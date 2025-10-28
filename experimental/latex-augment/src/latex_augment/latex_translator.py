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
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from tree_sitter import Node

from .latex_characters import remove_pdflatex_unsupported_chars
from .latex_parser import (
    LatexDocument,
    LatexError,
    format_latex_error_msg,
    get_field_text,
    get_node_end_byte_ws,
    get_parser_node_kinds,
    convert_line_endings,
    convert_to_utf8,
)
from .sentences import split_sentences

CJK_LANGS = ("zh", "ja", "ko")
BABEL_LANGS = {
    "de": "german",
    "el": "greek",
    "es": "spanish",
    "it": "italian",
    "fr": "french",
    "pt": "portuguese",
}
logger = logging.getLogger(__name__)


# Known environments and their argument specs; environment contents are translated.
# Other environments are copied as is and not translated.
# FIXME maybe the argument specs should be defined in grammar.js
KNOWN_ENVIRONMENTS = {
    "document": "",
    "article": "",
    "algorithm": "",
    "opening": "",
    "adjustbox": "{}",
    "abstract": "",
    "acknowledgements": "",
    "acks": "",
    "appendix": "",
    "bibliography": "",
    "center": "",
    "footnotesize": "",
    "description": "",
    "enumerate": "",
    "equation": "",
    "figure": "[]",
    "framed": "",
    "frontmatter": "",
    "sidewaysfigure": "[]",
    "itemize": "",
    "list": "",
    "quotation": "",
    "quote": "",
    "table": "[]",
    "sidewaystable": "[]",
    "tabular": "[]{}",
    "tabularx": "{}[]{}",
    "proof": "[t]",
    "keywords": "",
    "titlepage": "",
    "verse": "",
    "flushleft": "",
    "small": "",
    "CJK": "[]{}{}",
    "minipage": "[][][]{}",
    "wrapfigure": "[]{}{}",
    "wraptable": "[]{}{}",
}

# Known commands with their argument specs. Commands not listed here are parsed as is
# and not translated, including any options (in brackets) and args (in curly braces).
# FIXME maybe the argument specs should be defined in grammar.js
KNOWN_COMMANDS = {
    "\\'": "",
    "\\ ": "",
    "\\bf": "",
    "\\em": "",
    "\\emph": "{t}",
    "\\textbf": "{t}",
    "\\textit": "{t}",
    "\\textsc": "{t}",
    "\\textsf": "{t}",
    "\\textsl": "{t}",
    "\\texttt": "{t}",
    "\\underline": "{t}",
    "\\latex": "",
    "\\hline": "",
    "\\newline": "",
    "\\toprule": "",
    "\\midrule": "",
    "\\bottomrule": "",
    "\\shortstack": "{t}",
    "\\makecell": "[]{t}",
    "\\multicolumn": "{}{}{t}",
    "\\multirow": "{}{}{t}",
    "\\resizebox": "{}{}{t}",
    "\\affiliation": "{t}",
    "\\footnote": "[]{t}",
    "\\footnotetext": "[]{t}",
    "\\markboth": "{t}{t}",
    "\\author": "{t}",
    "\\surname": "{t}",
    "\\date": "{t}",
    "\\thanks": "{t}",
    "\\runningauthor": "{t}",
    "\\runningtitle": "{t}",
    "\\institute": "{t}",
    "\\keywords": "{t}",
    "\\title": "[t]{t}",
    "\\subfloat": "[t][t]{t}",
    "\\slugcomment": "{t}",
    "\\shorttitle": "{t}",
    "\\shortauthors": "{t}",
    "\\vparagraph": "{t}",
    "\\framebox": "[][]{t}",
    "\\makebox": "[][]{t}",
    "\\fbox": "{t}",
    "\\mbox": "{t}",
    "\\parbox": "[][][]{}{t}",
    "\\texorpdfstring": "{t}{t}",
    "\\textsuperscript": "{t}",
    "\\textsubscript": "{t}",
    "\\afterpage": "{t}",
    "\\adjustbox": "{}{t}",
    "\\begin": "{}",  # arg spec is overridden in collect_environment
    "\\end": "{}",  # only used when there is a parse error (w.r.t. generic_environment)
    "\\url": "{}",
}

TEXT_NODES = {"text", "word", "operator", ",", "(", ")"}

NODES_TO_EMBED = {
    "inline_formula",
    "displayed_equation",
    "citation",
    "label_reference",
    "label_reference_range",
    "verb",
}

NODES_TO_TRANSLATE = {
    "source_file",
    "part",
    "chapter",
    "section",
    "subsection",
    "subsubsection",
    "paragraph",
    "subparagraph",
    "text",
    "caption",
    "footnote",
    "label",
    "title_declaration",
    "author_declaration",
    "author",
    "curly_group",
    "curly_group_author_list",
    "curly_group_text",
    "curly_group_text_list",
    "brack_group_text",
    "enum_item",
    "cmidrule",
    *TEXT_NODES,
    *NODES_TO_EMBED,
}

CURLY_GROUP_NODES = {x for x in get_parser_node_kinds() if x.startswith("curly_group")}
BRACK_GROUP_NODES = {x for x in get_parser_node_kinds() if x.startswith("brack_group")}
COMMENT_NODES = {x for x in get_parser_node_kinds() if "comment" in x}

# COPY: copy latex segment as is; marks translation chunk boundary
# EMBED: embed inline latex segment; part of translation chunk
# TRANSLATE: translate text; part of translation chunk
COPY, EMBED, TRANSLATE = range(3)

# define string markers such that tree sitter parses these as singular word nodes without splitting
_MAGIC = "EP5gBacfX7"
BOS = _MAGIC + "BOS"
EOS = _MAGIC + "EOS"
BOLATEX = _MAGIC + "BOLATEX"
EOLATEX = _MAGIC + "EOLATEX"

# LaTeX whitespace including "~" and newline, but not when preceded by exactly one backslash
_WS_PAT = r"(?:(?<!\\)[\s~]|(?<=\\\\)[\s~])"
_WS_RE = re.compile("^" + _WS_PAT + "+", re.DOTALL)

# extract preceding and following whitespace
_PRE_POST_WS_RE = re.compile(
    r"^(" + _WS_PAT + r"*)(.*?)(" + _WS_PAT + r"*)$", re.DOTALL
)

# <latex>...</latex> tags used in place of LaTeX equations etc.
_LATEX_TAG_RE = re.compile(r"<latex>(.*?)</latex>", re.DOTALL)

_LATEX_MARK_RE = re.compile(BOLATEX + r"(.*?)" + EOLATEX, re.DOTALL)

# <s>...</s> tags used to mark sentences for translation
_SENT_TAG_RE = re.compile(r"<s>(.*?)</s>", re.DOTALL)

# <a0>, <a1>, ... anchors used in place of LaTeX equations etc.
_ANCHOR_RE = re.compile(r"(<a\d+>)")

# translated options/args must be after non-translated options/args
_ARG_SPEC_RE = re.compile(r"^(\[\]|{})*(\[t\]|{t})*$")


@dataclass
class Config:
    """LaTeX parsing configuration."""

    ignore_errors: bool
    known_environments: dict[str, str]
    known_commands: dict[str, str]


def collect_nodes(
    node: Node, *, doc=None, config: Config, force_recurse=False
) -> Iterable[tuple[int, Node]]:
    """Collect nodes recursively for translation.
    :param ignore_errors: Ignore parse errors and continue with best effort
    :param force_recurse: Recurse into children regardless of node type
    """
    if node.is_error:
        raise_or_warn(
            "LaTeX parse error", node=node, doc=doc, ignore_errors=config.ignore_errors
        )

    elif node.type == "latex_include":
        raise_or_warn(
            "please load included files before translation",
            node=node,
            doc=doc,
            ignore_errors=config.ignore_errors,
        )

    elif node.type == "title_declaration":
        yield from collect_command(node, doc=doc, config=config)
        return

    elif node.type == "generic_environment":
        yield from collect_environment(node, doc=doc, config=config)
        return

    elif node.type == "generic_command":
        yield from collect_command(node, doc=doc, config=config)
        return

    elif node.type == "theorem_definition":
        yield from collect_newtheorem(node, doc=doc, config=config)
        return

    elif node.type == "tikzstyle_definition":
        yield COPY, node
        return

    elif node.type == "operator":
        yield TRANSLATE, node
        return

    elif node.type in NODES_TO_EMBED:
        yield EMBED, node
        return

    # main heuristic: copy all nodes as is, except those that are listed for translation
    elif node.type not in NODES_TO_TRANSLATE and not force_recurse:
        # NB. chunk boundary
        yield COPY, node
        return

    # traverse in document order (depth-first search)
    for child in node.children:
        # force_recurse is not applied recursively
        yield from collect_nodes(child, doc=doc, config=config)

    if node.children:
        return

    if node.is_missing:
        # skip ghost nodes inserted by tree-sitter
        return
    elif node.type in TEXT_NODES:
        yield TRANSLATE, node
    else:
        # NB. chunk boundary
        yield COPY, node


def collect_environment(
    node: Node, *, doc=None, config: Config
) -> Iterable[tuple[int, Node]]:
    """Collect environment nodes \\begin{...}...\\end{...} recursively."""
    # translate contents of specific environments
    begin = node.children[0]
    name = begin.children[1].children[1].text.decode("utf8")
    name = name.rstrip("*")
    if name in config.known_environments:
        # add arg for environment name
        arg_spec = "{}" + config.known_environments[name]
        yield from collect_command(begin, doc=doc, config=config, arg_spec=arg_spec)
        end = node.children[-1]
        for child in node.children[1:-1]:
            yield from collect_nodes(child, doc=doc, config=config)
        yield COPY, end
    else:
        yield COPY, node


def collect_command(
    node: Node, *, doc=None, config: Config, arg_spec: str | list[str] = None
) -> Iterable[tuple[int, Node]]:
    """Collect command nodes \\foo[...]{...} recursively."""
    name = node.children[0].text.decode("utf8")
    if name == "\\\\":
        # special case: line break is always chunk boundary
        yield COPY, node
        return
    elif name == "\\%":
        # special case: percentages are handles as translated text
        yield TRANSLATE, node
        return
    elif name not in config.known_commands:
        yield EMBED, node
        return
    # 1st child is command_name, then come args
    cmd = node.children[0]
    if arg_spec is None:
        arg_spec = config.known_commands[name]
    args, textargs, rest = split_args(
        node.children[1:], arg_spec, doc=doc, ignore_errors=config.ignore_errors
    )
    if args or textargs:
        # complex commands (with arguments) create a chunk boundary
        yield COPY, cmd
        for arg in args:
            yield COPY, arg
    else:
        yield EMBED, cmd
    for child in textargs:
        yield from collect_nodes(child, doc=doc, config=config, force_recurse=True)
    for child in rest:
        yield from collect_nodes(child, doc=doc, config=config)


def split_args(
    children: list[Node], spec: str | list[str], *, doc=None, ignore_errors=False
) -> tuple[list[Node], list[Node], list[Node]]:
    """Split LaTeX command/environment node children into arguments and text.
    Argument spec strings contain permissible option and argument ordering:
    - [] means optional arguments to copy as is
    - {} means required arguments to copy as is
    - [t] means optional arguments to translate
    - {t} means required arguments to translate
    Skipped options/args must precede translated options/args.
    Argument spec can also be a list (union) of spec strings.
    :param spec: Argument specification
    :return: Lists of (1) non-translated arguments (incl. options), (2) translated arguments, and (3) translated text/rest
    """
    args, textargs, text = [], [], []

    assert _ARG_SPEC_RE.match(spec), f"Invalid argument spec: {spec}"
    remaining_spec = re.findall(r"\[t?\]|{t?}", spec)
    for child in children:
        if not remaining_spec:
            text.append(child)
        elif child.type in COMMENT_NODES:
            # remove comment nodes during argument parsing (best effort)
            continue
        elif child.type in BRACK_GROUP_NODES:
            if remaining_spec[0] == "[]":
                assert not (textargs or text), "Arguments must precede translated text"
                args.append(child)
                remaining_spec.pop(0)
            elif remaining_spec[0] == "[t]":
                textargs.append(child)
                remaining_spec.pop(0)
            else:
                raise_or_warn(
                    f"Option not allowed in spec: {remaining_spec}",
                    node=child,
                    doc=doc,
                    ignore_errors=ignore_errors,
                )
                args.append(child)
        elif child.type in CURLY_GROUP_NODES:
            while remaining_spec and remaining_spec[0].startswith("["):
                remaining_spec.pop(0)
            if not remaining_spec:
                text.append(child)
            elif remaining_spec[0] == "{}":
                assert not (textargs or text), "Arguments must precede translated text"
                args.append(child)
                remaining_spec.pop(0)
            elif remaining_spec[0] == "{t}":
                textargs.append(child)
                remaining_spec.pop(0)
            else:
                raise SyntaxError(f"Invalid argument spec: {remaining_spec}")
        else:
            while remaining_spec and remaining_spec[0].startswith("["):
                remaining_spec.pop(0)
            if remaining_spec:
                raise_or_warn(
                    f"Argument required in spec: {remaining_spec}",
                    node=child,
                    doc=doc,
                    ignore_errors=ignore_errors,
                )
            text.append(child)
    while remaining_spec and remaining_spec[0].startswith("["):
        remaining_spec.pop(0)
    if remaining_spec:
        raise_or_warn(
            f"Argument required in spec: {remaining_spec}", ignore_errors=ignore_errors
        )
    return args, textargs, text


def collect_newtheorem(
    node: Node, *, doc=None, config: Config
) -> Iterable[tuple[int, Node]]:
    """Collect \\newtheorem{...}{...} recursively."""
    name = (
        node.child_by_field_name("name").child_by_field_name("text").text.decode("utf8")
    )
    config.known_environments[name] = "[t]"  # \begin{theorem}[Pythagorean theorem]
    for idx, child in enumerate(node.children):
        if node.field_name_for_child(idx) == "title":
            # recurse into curly_group
            yield from collect_nodes(child, doc=doc, config=config)
        else:
            yield COPY, child


def translate_latex(
    latex: bytes | str,
    *,
    document: bool = False,
    keep_tags: bool = False,
    lang: str | None = None,
    translator: callable = None,
    verbose: bool = False,
) -> str:
    """Translate LaTeX document/fragment."""
    if isinstance(latex, str):
        latex = latex.encode("utf8")
    latex = convert_line_endings(latex)
    doc = LatexDocument(latex)
    parsed = prepare_document(doc, document=document, verbose=verbose)
    return "".join(
        translate_split_document(
            parsed, keep_tags=keep_tags, lang=lang, translator=translator
        )
    )


def prepare_document(doc: LatexDocument, *, document=False, verbose=False) -> str:
    """Prepare LaTeX document for translation."""
    if document:
        # check for \begin{document} (raises LatexError if not found)
        doc.body_start
        # https://tex.stackexchange.com/questions/307673/tell-latex-to-ignore-all-unicode-errors
        doc = doc.append_preamble(r"""
\makeatletter
\def\UTFviii@defined#1{%
  \ifx#1\relax
      ?%
  \else\expandafter
    #1%
  \fi
}
\makeatother
""")
    doc = convert_to_utf8(doc)
    try:
        doc = move_maketitle(doc)
    except ValueError:
        pass
    doc = LatexDocument(
        doc.source
        .replace(b"<s>", b"\\textless{}s>")
        .replace(b"</s>", b"\\textless{}/s>")
        .replace(b"<latex>", b"\\textless{}latex>")
        .replace(b"</latex>", b"\\textless{}/latex>")
    )
    text = "".join(split_document(doc, verbose=verbose))
    # convert tags to XML style
    text = (
        text
        .replace(BOS, "<s>")
        .replace(EOS, "</s>")
        .replace(BOLATEX, "<latex>")
        .replace(EOLATEX, "</latex>")
    )
    return text


def split_document(doc: LatexDocument, *, strict=False, verbose=False) -> Iterable[str]:
    """Split LaTeX document into sentences for translation."""
    if verbose:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.ERROR)
    config = Config(
        ignore_errors=not strict,
        known_environments=dict(KNOWN_ENVIRONMENTS),  # shallow copy
        known_commands=dict(KNOWN_COMMANDS),  # shallow copy
    )
    # aggregate consecutive TRANSLATE and EMBED nodes into a paragraph
    para_src = ""
    para_text = ""
    pos = 0
    for mode, child in collect_nodes(doc.root_node, config=config, doc=doc):
        if child.start_byte > pos:
            yield from mark_sentences(para_src, para_text)
            para_src = ""
            para_text = ""
            # cannot escape "</foo>" as string because "/" parses as separate operator node
            # so use BOS etc. here and replace with </s> and </latex> after concatenation (in prepare_document)
            src = doc.source[pos : child.start_byte].decode("utf8")
            assert _MAGIC not in src, "logic error: BOS/EOS markers are not unique"
            yield src
            pos = child.start_byte
        pos = get_node_end_byte_ws(child)
        src = doc.extract_text(child).decode("utf8")
        assert _MAGIC not in src, "logic error: BOS/EOS markers are not unique"
        if mode == COPY:
            yield from mark_sentences(para_src, para_text)
            para_src = ""
            para_text = ""
            yield src
        elif mode == EMBED:
            # exclude preceding and following whitespace
            pre_ws, embed, post_ws = _PRE_POST_WS_RE.match(src).groups()
            embed = BOLATEX + embed + EOLATEX
            para_src += pre_ws + embed + post_ws
            para_text += pre_ws + (" " * len(embed)) + post_ws
        else:
            para_src += src
            para_text += src
    yield from mark_sentences(para_src, para_text)


def mark_sentences(src: str, text: str) -> Iterable[str]:
    assert len(src) == len(text), "logic error: src and text have different lengths"
    idx = 0
    for span in split_sentences(text):
        assert span.start >= idx, "logic error: span.start < idx"
        if span.start > idx:
            yield src[idx : span.start].replace(BOLATEX, "").replace(EOLATEX, "")
        sent = src[span]
        # only mark for translation if sentence contains (unicode) word characters
        # excluding latex embeds
        sent_nolatex = _LATEX_MARK_RE.sub("", sent).replace(BOLATEX, "").replace(EOLATEX, "")
        if re.search(r"[^\W\d_]", sent_nolatex):
            yield BOS + sent + EOS
        else:
            yield sent.replace(BOLATEX, "").replace(EOLATEX, "")
        idx = span.stop
    if idx < len(src):
        yield src[idx:].replace(BOLATEX, "").replace(EOLATEX, "")


def translate_split_document(
    latex: str,
    *,
    lang: str,
    keep_tags: bool = False,
    translator: callable = None,
) -> Iterable[str]:
    """Translate LaTeX document/fragment after splitting."""
    doc = LatexDocument(latex.encode("utf8"))
    if lang == "en" or lang == "none" or lang is None:
        pass
    elif lang in CJK_LANGS:
        doc = add_cjk(doc, lang)
    else:
        doc = doc.with_package("babel", [BABEL_LANGS[lang]])
    latex = doc.source.decode("utf8")
    pos = 0
    for match in _SENT_TAG_RE.finditer(latex):
        if match.start() > pos:
            yield latex[pos : match.start()]
        src_text, embeds = preprocess_sentence(match.group(1), keep_tags=keep_tags)
        dst_text = translator(src_text)
        translated = postprocess_sentence(dst_text, embeds)
        translated = remove_pdflatex_unsupported_chars(translated)
        if keep_tags:
            translated = f"<s>{translated}</s>"
        yield translated
        pos = match.end()
    if pos < len(latex):
        yield latex[pos:]


def preprocess_sentence(
    sent: str, *, keep_tags: bool = False
) -> tuple[str, dict[str, str]]:
    """Preprocess LaTeX sentence for translation."""
    sent, embeds = remove_latex_embeds(sent, keep_tags=keep_tags)
    sent = sent.replace("\n", " ")
    # replace "~" but not "\~"
    sent = re.sub(r"(?<!\\)~", "\N{NO-BREAK SPACE}", sent)
    sent = unescape_latex(sent)
    return sent, embeds


def postprocess_sentence(translated: str, embeds: dict[str, str]) -> str:
    """Postprocess translated sentence back to LaTeX."""
    dst = escape_latex(translated)
    translated = dst.replace("\N{NO-BREAK SPACE}", "~")
    translated = restore_latex_embeds(translated, embeds)
    return translated


def remove_latex_embeds(
    src: str, *, keep_tags: bool = False
) -> tuple[str, dict[str, str]]:
    """Replace LaTeX segments including commands, math, etc. with numbered anchors.
    For example, "this is \\latex where $1+1=2$" becomes "this is <a0> where <a1>".
    """
    embeds = {}

    def make_anchor(match):
        idx = len(embeds)
        anchor = f"<a{idx}>"
        if keep_tags:
            embeds[anchor] = match.group(0)
        else:
            embeds[anchor] = match.group(1)
        return anchor

    # handle existing anchors
    src = _ANCHOR_RE.sub(make_anchor, src)
    # convert latex to anchors
    src = _LATEX_TAG_RE.sub(make_anchor, src)
    return src, embeds


def restore_latex_embeds(text: str, embeds: dict[str, str]) -> str:
    """Restore embedded LaTeX segments after translation."""

    def anchor_replacer(match):
        anchor = match.group(0)
        next_char = text[match.end()] if match.end() < len(text) else None
        embed = embeds.get(anchor, "")
        # braces prevent concatenation of commands and translated text (CJK)
        if not (
            embed.endswith("}")
            or next_char in (" ", "~", "\n", "\t", "{", ".", ",", "!", "?", ";", ":", "\\")
        ):
            embed += "{}"
        return embed

    text = _ANCHOR_RE.sub(anchor_replacer, text)
    return text


def startswith_ws(text: str) -> bool:
    """Check if text starts with LaTeX whitespace."""
    return _WS_RE.match(text) is not None


def endswith_ws(text: str) -> bool:
    """Check if text ends with LaTeX whitespace."""
    return startswith_ws(text[::-1])


def escape_latex(text: str) -> str:
    """Escape LaTeX special characters in translation output."""
    special_chars = r"\%{}$#&_^"
    for char in special_chars:
        text = text.replace(char, "\\" + char)
    return text


def unescape_latex(text: str) -> str:
    """Unescape LaTeX special characters in translation input."""
    special_chars = reversed(r"\%{}$#&_^")
    for char in special_chars:
        text = text.replace("\\" + char, char)
    return text


def add_cjk(doc: LatexDocument, lang: str) -> LatexDocument:
    """Add CJK environment to LaTeX document."""
    if (
        doc.documentclass.style.startswith("revtex")
        or doc.documentclass.style == "IEEEtran"
    ):
        raise LatexError("CJK is not supported for revtex or IEEEtran")
    # default fonts
    if lang == "zh":
        font = "gbsn"
    elif lang == "ja":
        font = "min"
    elif lang == "ko":
        font = "mj"
    else:
        raise ValueError(f"Unsupported CJK language: {lang}")
    doc = doc.with_package("CJKutf8")
    doc = doc.with_package("CJKspace")
    doc = doc.wrap_body("\\begin{CJK*}{UTF8}{" + font + "}", "\\end{CJK*}")
    return doc


def move_maketitle(doc: LatexDocument) -> LatexDocument:
    """Move maketitle from preamble to body (to be translated)."""
    # extract title commands from preamble
    try:
        preamble_end = doc.preamble_end
    except LatexError:
        return doc
    try:
        maketitle = doc.find(lambda node: node.text == b"\\maketitle")
    except ValueError:
        logger.warning("No \\maketitle found")
        return doc
    spans = []
    sources = []
    for command in ("title", "author", "date", "thanks", "affil", "affiliation"):
        for ids in doc.find_command(command, end=preamble_end):
            spans.append(ids)
            sources.append(doc.source[ids])
    # remove nested commands
    ids = [
        i
        for i, x in enumerate(spans)
        if not any(
            x.start >= y.start and x.stop <= y.stop
            for j, y in enumerate(spans)
            if j != i
        )
    ]
    editor = doc.editor()
    for i in ids:
        editor.replace_at(spans[i], b"")
        editor.insert_at(maketitle.start_byte, sources[i])
    return editor.execute()


def update_stats(stats: dict, node: Node):
    if node.type == "generic_environment":
        stats[get_field_text(node, "begin.name.text")] += 1
        stats[get_field_text(node, "end.name.text")] += 1
    else:
        command = get_field_text(
            node, "command.command_name", get_field_text(node, "command", None)
        )
        if command:
            stats[command] += 1
        else:
            stats[node.type] += 1


def raise_or_warn(msg: str, *, node=None, doc=None, ignore_errors: bool = False):
    if ignore_errors:
        msg = msg + " (ignored)"
        if doc:
            msg = format_latex_error_msg(msg, node=node, doc=doc)
        logger.warning(msg)
        # continue with best effort
    else:
        raise LatexError(msg, node=node, doc=doc)


if __name__ == "__main__":
    path = "paper.tex"
    with open(path, "rb") as f:
        latex = f.read()
    stats = defaultdict(int)
    translated = translate_latex(latex, translator=lambda text: text, stats=stats)
    for k, v in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        print(f"{k}: {v}")
