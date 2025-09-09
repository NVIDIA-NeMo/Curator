# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import re
from dataclasses import dataclass
from typing import Any

import magic
from resiliparse.parse.encoding import bytes_to_str, detect_encoding

from nemo_curator.stages.text.download.base.extract import DocumentExtractor

from .html_extractors.lynx import LynxExtractor


def _remove_xml_encoding_declaration(text: str) -> str:
    return re.sub(r"^\s*<\?xml.*\?>", "", text)


def _decode_bytes(binary_content: bytes | None) -> str | None:
    if binary_content is None:
        return None
    try:
        content = bytes_to_str(binary_content, "utf-8")
    except (UnicodeDecodeError, UnicodeError, LookupError):
        encoding = detect_encoding(binary_content)
        if encoding is None or encoding == "utf-8":
            return None
        try:
            content = bytes_to_str(binary_content, encoding)
        except (UnicodeDecodeError, UnicodeError, LookupError):
            return None
    return _remove_xml_encoding_declaration(content)


def _is_notebook(content: str) -> bool:
    try:
        data = json.loads(content)
        return (
            isinstance(data, dict)
            and "nbformat" in data
            and "nbformat_minor" in data
            and "cells" in data
            and isinstance(data["cells"], list)
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return False


def _notebook_to_text(content: str) -> str:
    data = json.loads(content)
    out = ""
    for cell in data.get("cells", []):
        t = cell.get("cell_type")
        if t in ["code", "markdown", "raw"]:
            out += "".join(cell.get("source", []))
        if t == "code" and "outputs" in cell:
            for o in cell["outputs"]:
                if o.get("output_type") == "stream":
                    out += "".join(o.get("text", []))
                elif o.get("output_type") in ["execute_result", "display_data"]:
                    d = o.get("data", {})
                    if "text/plain" in d:
                        out += "".join(d["text/plain"])
                elif o.get("output_type") == "text":
                    out += "".join(o.get("text", []))
    return out


@dataclass
class MathContentExtractor(DocumentExtractor):
    """Extractor that decodes bytes, detects type, and extracts text using Lynx for HTML."""

    binary_column: str = "binary_content"
    url_column: str = "url"
    mime_type_column: str = "mime_type"
    lynx_timeout_sec: int = 20

    # Lazily-initialized, avoid unpickleable objects during deepcopy in with_()
    _lynx: Any | None = None
    _magic: Any | None = None

    def __post_init__(self):
        self._lynx = None
        self._magic = None

    def input_columns(self) -> list[str]:
        return [self.binary_column, self.url_column, self.mime_type_column]

    def output_columns(self) -> list[str]:
        return ["text", self.url_column, "type", "magic_mime_type"]

    def extract(self, record: dict[str, Any]) -> dict[str, Any] | None:
        binary = record.get(self.binary_column)
        url = record.get(self.url_column)
        mime_type = record.get(self.mime_type_column)

        # Compute magic mime from bytes if available (lazy init)
        magic_mime_type = None
        if isinstance(binary, (bytes, bytearray)):
            try:
                if self._magic is None:
                    self._magic = magic.Magic(mime=True)
                magic_mime_type = self._magic.from_buffer(binary)
            except Exception:  # noqa: BLE001
                magic_mime_type = None

        content = _decode_bytes(binary if isinstance(binary, (bytes, bytearray)) else None)
        if not content:
            return None

        doc_type = self._determine_type(content, magic_mime_type, mime_type, url)

        if doc_type == "notebook":
            return {
                "text": _notebook_to_text(content),
                self.url_column: url,
                "type": doc_type,
                "magic_mime_type": magic_mime_type,
            }
        if doc_type == "html":
            # lazy init lynx extractor
            if self._lynx is None:
                self._lynx = LynxExtractor(timeout_sec=self.lynx_timeout_sec)
            return {
                "text": self._lynx.extract_text(content),
                self.url_column: url,
                "type": doc_type,
                "magic_mime_type": magic_mime_type,
            }
        return {
            "text": content,
            self.url_column: url,
            "type": doc_type,
            "magic_mime_type": magic_mime_type,
        }

    def _is_html_document(self, text: str) -> bool:
        has_html_open = re.search(r"<html[^>]*>", text, re.IGNORECASE)
        has_html_close = re.search(r"</html\s*>", text, re.IGNORECASE)
        has_head_open = re.search(r"<head[^>]*>", text, re.IGNORECASE)
        has_head_close = re.search(r"</head\s*>", text, re.IGNORECASE)
        has_body_open = re.search(r"<body[^>]*>", text, re.IGNORECASE)
        has_body_close = re.search(r"</body\s*>", text, re.IGNORECASE)
        return all([has_html_open, has_head_open, has_body_open, has_head_close, has_html_close, has_body_close])

    def _determine_type(
        self, content: str | None, magic_mime_type: str | None, mime_type: str | None, url: str | None
    ) -> str:
        if not content:
            return "text"

        # Notebook takes precedence
        if self._is_notebook_type(content, magic_mime_type, url):
            return "notebook"

        result: str | None = None

        if magic_mime_type is None:
            if mime_type in self._get_text_mime_types():
                result = "text"
            elif mime_type in self._get_html_mime_types() or self._is_html_document(content):
                result = "html"
            else:
                result = "html"
        elif magic_mime_type in self._get_html_magic_types() or (
            mime_type and mime_type in self._get_html_mime_types()
        ):
            result = "html"
        elif mime_type in self._get_text_mime_types() or magic_mime_type in self._get_text_magic_types():
            result = "text"
        else:
            result = "html"

        return result or "html"

    def _is_notebook_type(self, content: str, magic_mime_type: str | None, url: str | None) -> bool:
        """Check if content is a Jupyter notebook."""
        try:
            return ((magic_mime_type == "application/json") or (url and url.endswith(".ipynb"))) and _is_notebook(
                content
            )
        except (TypeError, AttributeError, ValueError):
            return False

    def _get_text_mime_types(self) -> set[str]:
        """Get set of MIME types that indicate text content."""
        return {
            "text/x-web-markdown",
            "text/x-verilog",
            "text/x-rst",
            "text/x-ruby",
            "text/x-rsrc",
            "text/x-python",
            "text/x-perl",
            "text/x-pascal",
            "text/x-objcsrc",
            "text/x-ml",
            "text/x-matlab",
            "text/x-log",
            "text/x-haskell",
            "text/x-fortran",
            "text/x-expect",
            "text/x-diff",
            "text/x-csrc",
            "text/x-common-lisp",
            "text/x-chdr",
            "text/x-cgi",
            "text/x-c++src",
            "text/x-basic",
            "text/vtt",
            "text/x-assembly",
            "text/troff",
            "text/plain",
            "message/rfc822",
            "message/news",
            "application/mathematica",
            "application/mbox",
            "application/postscript",
            "application/x-elc",
            "application/x-matlab-data",
            "application/x-sas",
            "application/x-sh",
            "application/x-subrip",
            "application/x-tex",
            "application/x-tika-msoffice",
        }

    def _get_html_mime_types(self) -> set[str]:
        """Get set of MIME types that indicate HTML content."""
        return {
            "text/x-php",
            "text/x-jsp",
            "text/x-coldfusion",
            "text/html",
            "message/x-emlx",
            "text/asp",
            "image/svg+xml",
            "application/xml",
            "application/atom+xml",
            "application/rdf+xml",
            "application/rss+xml",
            "application/x-bibtex-text-file",
            "application/xhtml+xml",
        }

    def _get_text_magic_types(self) -> set[str]:
        """Get set of magic MIME types that indicate text content."""
        return {
            "text/x-shellscript",
            "text/x-perl",
            "text/x-lisp",
            "text/x-java",
            "text/x-fortran",
            "text/x-diff",
            "application/postscript",
            "application/x-matlab-data",
            "message/news",
            "message/rfc822",
            "text/plain",
            "text/texmacs",
            "text/x-Algol68",
        }

    def _get_html_magic_types(self) -> set[str]:
        """Get set of magic MIME types that indicate HTML content."""
        return {
            "text/xml",
            "text/x-tex",
            "text/x-php",
            "text/x-ruby",
            "text/x-script.python",
            "text/x-objective-c",
            "text/x-forth",
            "text/x-c",
            "text/x-c++",
            "text/csv",
            "text/html",
            "application/octet-stream",
            "application/x-appleworks3",
            "application/x-bytecode.python",
            "application/x-setupscript",
            "application/x-wine-extension-ini",
            "image/svg+xml",
        }
