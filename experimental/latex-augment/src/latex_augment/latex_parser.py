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

import os
import re
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import NamedTuple

import tree_sitter_latex
from tree_sitter import Language, Node, Parser


@dataclass(frozen=True)
class LatexDocument:
    source: bytes

    def __post_init__(self):
        assert isinstance(self.source, bytes), "Source must be bytes"

    @classmethod
    def from_file(cls, path: str) -> "LatexDocument":
        with open(path, "rb") as f:
            source = f.read()
        source = convert_line_endings(source)
        source = load_includes(source, os.path.dirname(path), strict=False)
        doc = cls(source)
        doc = convert_to_utf8(doc, strict=False)
        return doc

    @property
    def root_node(self) -> Node:
        """Parse LaTeX document with tree-sitter."""
        LATEX_LANGUAGE = Language(tree_sitter_latex.language())
        parser = Parser(LATEX_LANGUAGE)
        tree = parser.parse(self.source)
        return tree.root_node

    @property
    def has_preamble(self) -> bool:
        try:
            self.preamble_end
            return True
        except LatexError:
            return False

    @property
    def preamble(self) -> bytes:
        """Document preamble before \\begin{document}."""
        return self.source[: self.preamble_end]

    @property
    def body(self) -> bytes:
        """Main content between \\begin{document} and \\end{document}."""
        return self.source[self.body_start : self.body_end]

    @property
    def preamble_end(self) -> int:
        try:
            return self.index_of(rb"\\begin *\{document\}")
        except ValueError:
            raise LatexError("No \\begin{document} found.") from None

    @property
    def body_start(self) -> int:
        # don't use tree-sitter for basic augmentations (lotsa parse errors)
        try:
            begin = self.index_of(rb"\\begin *\{document\}")
            return self.source.index(b"\n", begin) + 1
        except ValueError:
            raise LatexError("No \\begin{document} found.") from None

    @property
    def body_end(self) -> int:
        # don't use tree-sitter for basic augmentations (lotsa parse errors)
        try:
            end = self.index_of(rb"\\end *\{document\}", start=self.body_start)
        except ValueError:
            end = len(self.source)
        return end

    @property
    def documentclass(self) -> tuple[str, list[str]]:
        class Documentclass(NamedTuple):
            style: str
            options: list[str]

        try:
            documentclass = self.find(lambda node: node.type == "class_include")
        except ValueError:
            raise ValueError("No \\documentclass{...} found") from None
        style = get_field_text(documentclass, "path.path")
        options = parse_options(documentclass)
        return Documentclass(style, options)

    @property
    def packages(self) -> dict[str, tuple[str, list[str]]]:
        class Package(NamedTuple):
            name: str
            options: list[str]

        nodes = self.findall(lambda node: node.type == "package_include")
        packages = [
            Package(name, parse_options(node))
            for node in nodes
            for name in parse_list(node.children[-1])
        ]
        return {p.name: p for p in packages}

    @property
    def floats(self) -> Iterator[Node]:
        return self.findall(
            lambda node: node.type == "generic_environment"
            and get_field_text(node, "begin.name.text", "")
            in ("figure", "figure*", "table", "table*")
        )

    def editor(self) -> "LatexEditor":
        return LatexEditor(self)

    def append_preamble(self, source: bytes | str) -> "LatexDocument":
        """Add LaTeX source to end of preamble."""
        if isinstance(source, str):
            source = source.encode("utf8")
        if not source.endswith(b"\n"):
            source += b"\n"
        return LatexDocument(self.preamble + source + self.source[len(self.preamble) :])

    def replace_preamble(self, source: bytes | str) -> "LatexDocument":
        """Replace whole preamble with new LaTeX source."""
        if isinstance(source, str):
            source = source.encode("utf8")
        if not source.endswith(b"\n"):
            source += b"\n"
        return LatexDocument(source + self.source[len(self.preamble) :])

    def with_documentclass(self, style: str, options: list[str]) -> "LatexDocument":
        """Change documentclass definition."""
        try:
            documentclass = self.find(lambda node: node.type == "class_include")
        except ValueError:
            raise ValueError("No \\documentclass{...} found") from None
        new_documentclass = format_unary_command("\\documentclass", style, options)
        editor = self.editor()
        editor.replace_at(documentclass, new_documentclass)
        return editor.execute()

    def with_package(self, name: str, options: list[str] = None) -> "LatexDocument":
        """Add package to LaTeX document."""
        cmd: bytes = format_unary_command("\\usepackage", name, options)
        editor = self.editor()
        for node in self.findall(
            lambda node: node.type == "package_include"
            and get_field_text(node, "paths.path", None) == name
        ):
            # replace first occurrence of package
            # FIXME doesn't handle multiple packages \\usepackage{a,b,c}
            editor.replace_at(node, cmd)
            cmd = b""
        doc = editor.execute()
        if cmd:
            doc = doc.append_preamble(cmd)
        return doc

    def wrap_body(self, begin: str, end: str) -> "LatexDocument":
        """Wrap document body in begin and end environment."""
        begin = begin.strip()
        end = end.strip()
        if begin and not begin.startswith("\\"):
            begin = "\\begin{" + begin + "}"
        if end and not end.startswith("\\"):
            end = "\\end{" + end + "}"
        editor = self.editor()
        editor.insert_at(self.body_start, begin.encode("utf8") + b"\n")
        editor.insert_at(self.body_end, end.encode("utf8") + b"\n")
        return editor.execute()

    def find(self, filter: callable) -> Node:
        """Find first node in tree for which filter(.) returns True."""
        return find(self.root_node, filter)

    def findall(self, filter: callable) -> Iterator[Node]:
        """Find all nodes in tree for which filter(.) returns True."""
        return findall(self.root_node, filter)

    def find_command(
        self, command: str, *, start: int = 0, end: int = None
    ) -> Iterator[slice]:
        """Find all occurrences of command in document."""
        # FIXME doesn't work if there is whitespace or comments between command and brace
        pattern = b"\\" + command.encode("utf-8") + b"{"
        idx = start
        if end is None:
            end = len(self.source)
        while idx < end:
            try:
                start_idx = self.index_of(re.escape(pattern), start=idx, end=end)
            except ValueError:
                return
            brace_count = 1
            for end_idx in range(start_idx + len(pattern), end):
                if self.source[end_idx] == ord(b"{") and self.source[end_idx - 1] != ord(b"\\"):
                    brace_count += 1
                elif self.source[end_idx] == ord(b"}") and self.source[end_idx - 1] != ord(b"\\"):
                    brace_count -= 1
                if brace_count == 0:
                    break
            if brace_count != 0:
                raise LatexError("Mismatched braces")
            # capture trailing whitespace
            end_idx += 1
            while end_idx < end and self.source[end_idx] in (ord(b" "), ord(b"\n")):
                end_idx += 1
            yield slice(start_idx, end_idx)
            assert end_idx > idx, "logic error: infinite loop in find_command"
            idx = end_idx

    def index_of(self, pattern: bytes, *, start: int = 0, end: int = None) -> int:
        """Find the start byte of given text in the file (ignoring comments).
        :param pattern: regular expression to search for
        """
        if end is None:
            end = len(self.source)
        assert b"\n" not in pattern, "pattern must not contain newlines"
        pat = b"^[^%\n]*?(" + pattern + b")"
        # re.MULTILINE assumes Linux line endings
        assert b"\r" not in self.source, "line endings are not Linux"
        match = re.search(pat, self.source[start:end], flags=re.MULTILINE)
        if match:
            return start + match.start(1)
        raise ValueError(f"Not found: {pattern}")

    def span_of(self, node: Node, *, include_ws: bool = True) -> slice:
        """Find start and end bytes of single node."""
        if include_ws:
            try:
                next_node = peek_next_node(node)
                return slice(node.start_byte, next_node.start_byte)
            except EOFError:
                return slice(node.start_byte, len(self.source))
        return slice(node.start_byte, node.end_byte)

    def extract_text(self, node: Node) -> bytes:
        """Extract text from single node."""
        return self.source[self.span_of(node)]

    def extract_source_line(self, pos: int) -> bytes:
        """Extract source line from node."""
        start_idx = pos
        end_idx = pos + 1
        try:
            start_idx = self.source.rindex(b"\n", 0, start_idx) + 1
        except ValueError:
            start_idx = 0
        try:
            end_idx = self.source.index(b"\n", end_idx) + 1
        except ValueError:
            end_idx = len(self.source)
        return self.source[start_idx:end_idx], start_idx

    def split_content_by(self, filter: callable) -> list[slice]:
        """Split main content into regions."""
        content_start = self.body_start
        content_end = self.body_end
        splits = [
            node.start_byte
            for node in self.findall(filter)
            if node.start_byte >= content_start and node.end_byte <= content_end
        ]
        splits += [content_start, content_end]
        # take unique positions as filter may match on multiple tree depths
        splits = sorted(set(splits))
        return [slice(start, end) for start, end in zip(splits, splits[1:])]

    @property
    def subsections(self) -> list[slice]:
        types = ("part", "chapter", "section", "subsection")
        return self.split_content_by(
            # skip section headings that are immediately after a higher-level heading
            # (third child after command and title)
            lambda node: (
                node.type in types
                and not (
                    node.parent
                    and len(node.parent.children) >= 3
                    and node.parent.children[2].start_byte == node.start_byte
                )
            )
        )


class LatexEditor:
    def __init__(self, doc: LatexDocument):
        self.doc = doc
        self.spans = []
        self.sources = []

    def insert_at(self, position: int | Node, source: bytes) -> "LatexEditor":
        """Insert new sources at given positions in document."""
        if isinstance(position, Node):
            position = position.start_byte
        self.spans.append(slice(position, position))
        self.sources.append(source)
        return self

    def replace_at(self, span: slice | Node, source: bytes) -> "LatexEditor":
        """Replace content at given spans with new sources."""
        if isinstance(span, Node):
            span = self.doc.span_of(span)
        self.spans.append(span)
        self.sources.append(source)
        return self

    def wrap(
        self, node: Node | slice, begin: str, end: str, *, inner: bool = False
    ) -> "LatexEditor":
        """Add content before and after node."""
        if isinstance(node, slice):
            assert not inner, "Cannot wrap inner with slice."
            span = node
        elif inner:
            span = slice(node.children[1].start_byte, node.children[-1].start_byte)
        else:
            span = self.doc.span_of(node)
        self.insert_at(span.start, begin.encode("utf8") + b"\n")
        self.insert_at(span.stop, end.encode("utf8") + b"\n")
        return self

    def execute(self) -> "LatexDocument":
        """Execute edits and return new document."""
        assert all(s.step is None for s in self.spans), "Spans must not have a step."
        sorting = sorted(
            range(len(self.spans)),
            key=lambda i: (self.spans[i].start, self.spans[i].stop),
        )
        self.spans = [self.spans[i] for i in sorting]
        self.sources = [self.sources[i] for i in sorting]
        positions = [p for s in self.spans for p in [s.start, s.stop]]
        assert positions == sorted(positions), "Spans must not overlap."
        # split old source into chunks
        positions = [0] + positions + [len(self.doc.source)]
        chunks = [self.doc.source[a:b] for a, b in zip(positions, positions[1:])]
        # replace chunks
        chunks[1::2] = self.sources
        return LatexDocument(b"".join(chunks))


class LatexError(ValueError):
    """LaTeX document syntax error."""

    def __init__(self, msg, *, node=None, pos=None, row=None, doc=None):
        if doc:
            msg = format_latex_error_msg(msg, node=node, pos=pos, row=row, doc=doc)
        super().__init__(msg)


def parse_list(node: Node) -> list[str]:
    """Parse list of strings from curly/brack group."""
    assert node.children, "Not a list"
    assert node.children[0].type in ("{", "["), "Not a list"
    return [
        child.text.decode("utf8")
        for child in node.children[1:-1]
        if child.type not in (",", "line_comment") and child.text
    ]


def parse_options(node: Node) -> list[str]:
    """Parse options from command."""
    options = node.child_by_field_name("options")
    if options:
        return parse_list(options)
    return []


def format_unary_command(command: str, arg: str, options: list[str] = None) -> bytes:
    if options:
        assert isinstance(options, list), "Options must be a list"
        options = "[" + ",".join(options) + "]"
    else:
        options = ""
    return f"{command}{options}{{{arg}}}\n".encode("utf8")


def format_latex_error_msg(msg: str, *, node=None, pos=None, row=None, doc=None) -> str:
    assert doc is not None
    if node is not None:
        pos = node.start_byte
        row = node.start_point.row
    context, start_pos = doc.extract_source_line(pos)
    caret_offset = len(context[: pos - start_pos].decode("utf8"))
    context = context.decode("utf8").rstrip()
    if row is not None:
        prefix = f"{row + 1}: "
    else:
        prefix = ""
    caret = " " * (caret_offset + len(prefix)) + "^"
    return f"{msg}\n{prefix}{context}\n{caret}"


def get_parser_node_kinds() -> list[str]:
    """Get all node kinds defined in the LaTeX parser."""
    LATEX_LANGUAGE = Language(tree_sitter_latex.language())
    return [
        LATEX_LANGUAGE.node_kind_for_id(i)
        for i in range(LATEX_LANGUAGE.node_kind_count)
    ]


def find(node: Node, filter: callable) -> Node:
    """Find first node in tree for which filter(.) returns True."""
    try:
        return next(findall(node, filter))
    except StopIteration:
        raise ValueError("Node not found") from None


def findall(node: Node, filter: callable) -> Iterator[Node]:
    """Find all nodes in tree for which filter(.) returns True."""
    for child in node.children:
        yield from findall(child, filter)
    if filter(node):
        yield node


def get_node_end_byte_ws(node: Node) -> int:
    """Return the end byte of the node, including trailing whitespace."""
    try:
        next_node = peek_next_node(node)
        return next_node.start_byte
    except EOFError:
        while node.parent:
            node = node.parent
        return node.end_byte


class _no_default:
    pass


def get_field(node: Node, field_names: str, default=_no_default) -> Node:
    """Get the child node with the given field name(s)."""
    res = node
    for name in field_names.split("."):
        res = res.child_by_field_name(name)
        if res is None:
            if default is _no_default:
                raise KeyError(f"{name} (in {field_names})")
            return default
    return res


def get_field_text(node: Node, field_names: str, default=_no_default) -> str:
    """Get the text of the child node with the given field name(s)."""
    try:
        return get_field(node, field_names).text.decode("utf8")
    except KeyError:
        if default is _no_default:
            raise
        return default


def peek_next_node(node):
    """Peek at the next (grand)sibling node in the document."""
    if node.next_sibling:
        return node.next_sibling
    if not node.parent:
        raise EOFError("Reached end of document.")
    return peek_next_node(node.parent)


def load_includes(
    latex: bytes, include_dir: str = None, files: dict = None, *, strict: bool = True
) -> bytes:
    """Load included files from LaTeX document.

    Args:
        latex: LaTeX source code
        include_dir: Base directory for includes
        dir_files: Optional dict mapping filenames to their contents for files in include_dir
    """
    assert not (include_dir and files), "Cannot provide both include_dir and files"

    def exists(path):
        if files is not None:
            return path in files
        else:
            return os.path.isfile(os.path.join(include_dir, path))

    def read(path):
        if files is not None:
            return files[path]
        else:
            with open(os.path.join(include_dir, path), "rb") as f:
                return f.read()

    root = LatexDocument(latex).root_node
    new_latex = b""
    prev_pos = 0
    for incl in findall(root, lambda node: node.type == "latex_include"):
        new_latex += latex[prev_pos : incl.start_byte]
        # latex_include -> path: curly_group_path -> path: path
        try:
            path = get_field_text(incl, "path.path")
        except KeyError:
            raise LatexError(f"Cannot parse include path: {incl.text}") from None
        # Normalize path to handle ./ and ../ components
        path = os.path.normpath(path)
        if not path.endswith(".tex") and exists(path + ".tex"):
            path = path + ".tex"
        
        if exists(path):
            included_latex = read(path)
            included_latex = convert_line_endings(included_latex)
            included_latex = load_includes(
                included_latex, include_dir, files, strict=strict
            )
            
            # Find end of line after \input{} command
            line_end = incl.end_byte
            try:
                line_end = latex.index(b"\n", incl.end_byte)
            except ValueError:
                line_end = len(latex)
            
            # Get content after \input{} on same line
            after_input = latex[incl.end_byte:line_end]
            
            # Add included content
            new_latex += included_latex
            
            # Handle what comes after the \input{} command
            if after_input.strip():
                # Non-whitespace content after \input{} - put on new line
                if not new_latex.endswith(b"\n"):
                    new_latex += b"\n"
                new_latex += after_input.lstrip()
                new_latex += b"\n"
                prev_pos = line_end + 1 if line_end < len(latex) else line_end
            else:
                # Only whitespace after \input{} - preserve structure
                if not included_latex.endswith(b"\n"):
                    new_latex += b"\n"
                prev_pos = line_end + 1 if line_end < len(latex) else line_end
        else:
            if strict:
                raise LatexError(f"File not found: {path}")
            new_latex += latex[incl.start_byte:incl.end_byte]
            prev_pos = incl.end_byte

    new_latex += latex[prev_pos:]
    return new_latex


def convert_to_utf8(doc: LatexDocument, *, strict: bool = True) -> LatexDocument:
    """Convert LaTeX document to UTF-8 encoding."""
    if "inputenc" in doc.packages and doc.packages["inputenc"].options:
        encoding = doc.packages["inputenc"].options[0]
        # map nonstandard encoding names; unhandled ones like
        # "utf8", "latin1", "cp1252" are supported by python decode
        if encoding == "utf8x":
            encoding = "utf8"
        elif encoding == "ansinew":
            # https://mirror.5i.fi/tex-archive/macros/latex/base/inputenc.pdf
            encoding = "cp1252"
        elif encoding in ("apple mac", "applemac", "mac"):
            # https://en.wikipedia.org/wiki/Mac_OS_Roman
            encoding = "macintosh"
        elif encoding == "isolatin":
            encoding = "latin-1"
    else:
        try:
            doc.source.decode("utf-8")
            encoding = "utf-8"
        except UnicodeDecodeError:
            logging.info("assuming latin-1 input encoding")
            encoding = "latin-1"
    try:
        converted = doc.source.decode(encoding).encode("utf-8")
    except UnicodeDecodeError as e:
        if strict:
            raise LatexError(f"invalid document encoding ({encoding}): {e}") from None
        logging.warning(f"invalid document encoding ({encoding}): {e} (ignored)")
        # some documents still have invalid utf-8 (e.g. latin-1) in comments,
        # so remove invalid bytes
        converted = doc.source.decode(encoding, errors="ignore").encode("utf-8")
    doc = LatexDocument(converted)
    if doc.has_preamble:
        doc = doc.with_package("inputenc", ["utf8"])
    return doc


def convert_line_endings(source: bytes) -> bytes:
    """Convert Windows (CRLF) and Mac (CR) line endings to Unix (LF)."""
    return source.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
