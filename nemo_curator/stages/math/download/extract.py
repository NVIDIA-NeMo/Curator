# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
import threading
from dataclasses import dataclass, field
from typing import Any

import magic
import pandas as pd
from loguru import logger
from resiliparse.parse.encoding import bytes_to_str, detect_encoding

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.download.base.extract import DocumentExtractor
from nemo_curator.tasks import DocumentBatch
from nemo_curator.utils.column_utils import resolve_filename_column

from .html_extractors.lynx import LynxExtractor
from .mime_types import HTML_MAGIC_TYPES, HTML_MIME_TYPES, TEXT_MAGIC_TYPES, TEXT_MIME_TYPES


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
    _lynx: Any | None = field(default=None, init=False, repr=False)
    _magic: Any | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self):
        self._lynx = None
        self._magic = None
        self._lock = threading.Lock()

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
                    with self._lock:
                        if self._magic is None:
                            self._magic = magic.Magic(mime=True)
                magic_mime_type = self._magic.from_buffer(binary)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Magic MIME detection failed: {e}")
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
                with self._lock:
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
            if mime_type in TEXT_MIME_TYPES:
                result = "text"
            elif mime_type in HTML_MIME_TYPES or self._is_html_document(content):
                result = "html"
            else:
                result = "html"
        elif magic_mime_type in HTML_MAGIC_TYPES or (mime_type and mime_type in HTML_MIME_TYPES):
            result = "html"
        elif mime_type in TEXT_MIME_TYPES or magic_mime_type in TEXT_MAGIC_TYPES:
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


@dataclass
class MathExtractStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Processing stage that applies a DocumentExtractor row-by-row to a DocumentBatch.

    Designed for use after CommonCrawlWARCReader, where binary content has already
    been fetched into a DocumentBatch. Each row is passed to the extractor and rows
    where extraction returns None are filtered out.
    """

    extractor: DocumentExtractor
    add_filename_column: bool | str = False

    def __post_init__(self) -> None:
        self.filename_col = resolve_filename_column(self.add_filename_column)
        self.name = f"extract_{self.extractor.__class__.__name__.lower()}"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], self.extractor.input_columns()

    def outputs(self) -> tuple[list[str], list[str]]:
        cols = self.extractor.output_columns()
        if self.filename_col:
            cols = [*cols, self.filename_col]
        return ["data"], cols

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()
        records = []
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            extracted = self.extractor.extract(row_dict)
            if extracted is None:
                continue
            if self.filename_col and self.filename_col in row_dict:
                extracted[self.filename_col] = row_dict[self.filename_col]
            records.append(extracted)

        output_cols = self.extractor.output_columns()
        if self.filename_col:
            output_cols = [*output_cols, self.filename_col]

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=pd.DataFrame(records) if records else pd.DataFrame(columns=output_cols),
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
