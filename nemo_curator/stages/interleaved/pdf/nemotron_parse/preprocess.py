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

"""CPU preprocess stage: extract PDFs and render pages to images."""

from __future__ import annotations

import contextlib
import json
import multiprocessing
import os
import queue
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar

import pandas as pd
import pyarrow as pa
from loguru import logger
from PIL import Image

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse._image_transport import PNG_METADATA_KEY, png_batch_provenance
from nemo_curator.stages.interleaved.pdf.nemotron_parse.page_budget import (
    ExtractionConfigMismatchError,
    output_version,
)
from nemo_curator.stages.interleaved.pdf.nemotron_parse.utils import (
    extract_pdf_from_jsonl,
    extract_pdf_from_zip,
    extract_pdfs_from_jsonl_batch,
    image_to_bytes,
    render_pdf_pages,
)
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import FileGroupTask, InterleavedBatch


@dataclass
class PDFPreprocessStage(ProcessingStage[FileGroupTask, InterleavedBatch]):
    """CPU stage: extract PDFs and render pages to images.

    Each entry in the input ``FileGroupTask.data`` is a JSON string with at
    minimum a ``file_name`` key and optionally a ``url`` key.

    PDF bytes are obtained in one of three ways:

    - **Zip archive mode** (``zip_base_dir`` is set): PDFs are extracted from
      CC-MAIN-style zip archives using :func:`extract_pdf_from_zip`.
    - **Directory mode** (``pdf_dir`` is set): PDFs are read directly from
      ``<pdf_dir>/<file_name>``.
    - **JSONL mode** (``jsonl_base_dir`` is set): PDFs are decoded from
      base64 ``content`` fields in JSONL files (e.g. GitHub PDF datasets).
      Entries must include ``jsonl_file`` and either ``byte_offset`` (preferred,
      O(1) seek) or ``line_idx`` (legacy, O(N) scan).

    Produces an :class:`InterleavedBatch` with one row per page, where
    ``binary_content`` holds the PNG-encoded page image and ``text_content``
    is empty (to be filled by the GPU inference stage).

    Parameters
    ----------
    zip_base_dir
        Root of CC-MAIN zip archive hierarchy.
    pdf_dir
        Directory containing loose PDF files.
    jsonl_base_dir
        Root directory for JSONL-based PDF datasets (e.g. GitHub PDFs).
    dpi
        Resolution for PDF page rendering.
    max_pages
        Maximum number of pages to render per PDF.
    retain_failures
        Preserve physical page indexes and failed-page rows for the served parser.
    reuse_encoded_images
        Attach private ordered-batch provenance for optional verified PNG reuse.
    """

    zip_base_dir: str | None = None
    pdf_dir: str | None = None
    jsonl_base_dir: str | None = None
    dpi: int = 300
    max_pages: int = 50
    retain_failures: bool = False
    reuse_encoded_images: bool = field(default=False, kw_only=True)
    name: str = "pdf_preprocess"
    resources: Resources = field(default_factory=lambda: Resources(cpus=2.0))
    _extraction_config: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def _check_extraction_settings(self) -> None:
        if self._extraction_config is None:
            return
        effective = {
            "dpi": self.dpi,
            "proc_size": [2048, 1664],
            "timeout_s": self._RENDER_TIMEOUT_S,
            "recovery_budget_s": self._RENDER_RECOVERY_BUDGET_S,
            "retry_timeout_s": self._RENDER_RETRY_TIMEOUT_S,
            "max_attempts": self._RENDER_MAX_ATTEMPTS,
        }
        if self._extraction_config["render"] != effective:
            msg = "Effective rendering/recovery settings differ from admitted extraction configuration"
            raise ExtractionConfigMismatchError(msg)

    def __post_init__(self) -> None:
        if type(self.reuse_encoded_images) is not bool:
            msg = "reuse_encoded_images must be a boolean"
            raise TypeError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def _get_pdf_bytes(self, file_name: str, entry: dict | None = None) -> bytes | None:
        if self.zip_base_dir is not None:
            return extract_pdf_from_zip(file_name, self.zip_base_dir)
        if self.jsonl_base_dir is not None and entry is not None:
            jsonl_file = os.path.join(self.jsonl_base_dir, entry["jsonl_file"])
            return extract_pdf_from_jsonl(
                jsonl_file,
                line_idx=entry.get("line_idx"),
                byte_offset=entry.get("byte_offset"),
            )
        if self.pdf_dir is not None:
            path = os.path.join(self.pdf_dir, file_name)
            try:
                with open(path, "rb") as f:
                    return f.read()
            except OSError:
                return None
        msg = "One of zip_base_dir, pdf_dir, or jsonl_base_dir must be set"
        raise ValueError(msg)

    def _batch_fetch_jsonl(self, entries: list[dict]) -> dict[str, bytes | None]:
        """Fetch PDF bytes for all JSONL-mode entries using one file open per JSONL.

        Groups entries by jsonl_file, then calls extract_pdfs_from_jsonl_batch so
        each source file is opened exactly once.  Entries without byte_offset fall
        back to the single-entry path.

        Returns a dict mapping entry index (position in `entries`) -> pdf_bytes.
        """
        # Separate entries that have byte_offset (fast path) from those that don't
        by_file: dict[str, list[tuple[int, int]]] = {}  # jsonl_path -> [(idx, offset)]
        fallback: list[int] = []  # indices needing line_idx scan

        for idx, entry in enumerate(entries):
            if "byte_offset" in entry:
                jsonl_path = os.path.join(self.jsonl_base_dir, entry["jsonl_file"])
                by_file.setdefault(jsonl_path, []).append((idx, entry["byte_offset"]))
            else:
                fallback.append(idx)

        results: dict[int, bytes | None] = {}

        # Batch path: one file open per JSONL file
        for jsonl_path, idx_offset_pairs in by_file.items():
            offsets = [offset for _, offset in idx_offset_pairs]
            fetched = extract_pdfs_from_jsonl_batch(jsonl_path, offsets)
            for idx, offset in idx_offset_pairs:
                results[idx] = fetched.get(offset)

        # Fallback: legacy line_idx scan (one open per entry)
        for idx in fallback:
            entry = entries[idx]
            jsonl_path = os.path.join(self.jsonl_base_dir, entry["jsonl_file"])
            results[idx] = extract_pdf_from_jsonl(jsonl_path, line_idx=entry.get("line_idx"))

        return results

    _RENDER_TIMEOUT_S: int = 60
    _RENDER_RECOVERY_BUDGET_S: ClassVar[float] = 120
    _RENDER_RETRY_TIMEOUT_S: ClassVar[float] = 60
    _RENDER_MAX_ATTEMPTS: ClassVar[int] = 7

    def _render_once(  # noqa: C901, PLR0915
        self,
        pdf_bytes: bytes,
        page_indices: list[int] | None,
        *,
        timeout_s: float | None = None,
        expected_page_count: int | None = None,
    ) -> tuple[list[Image.Image | None], bool, int | None]:
        """Render one window, always closing queues and reaping its child."""
        ctx = multiprocessing.get_context("fork")
        result_q: multiprocessing.Queue = ctx.Queue()
        page_count_q: multiprocessing.Queue = ctx.Queue()

        def _worker() -> None:
            try:
                if page_indices is None or expected_page_count is not None:
                    import pypdfium2 as pdfium

                    doc = pdfium.PdfDocument(pdf_bytes)
                    try:
                        page_count_q.put(len(doc))
                    finally:
                        doc.close()
                result_q.put(
                    render_pdf_pages(
                        pdf_bytes,
                        dpi=self.dpi,
                        max_pages=self.max_pages,
                        keep_failed=self.retain_failures or page_indices is not None,
                        page_indices=page_indices,
                    )
                )
            except Exception:  # noqa: BLE001
                result_q.put([])

        proc = ctx.Process(target=_worker)
        pages = []
        timed_out = False
        page_count = None
        try:
            proc.start()
            deadline = time.monotonic() + (self._RENDER_TIMEOUT_S if timeout_s is None else timeout_s)
            # Read before joining: a large result can fill the pipe and block the child feeder.
            while (remaining := deadline - time.monotonic()) > 0:
                try:
                    pages = result_q.get(timeout=min(0.1, remaining))
                    break
                except queue.Empty:
                    if not proc.is_alive():
                        break
            else:
                timed_out = proc.is_alive()
        finally:
            try:
                if proc.pid is not None:
                    proc.join(timeout=0.1)
                    if proc.is_alive():
                        proc.kill()
                        proc.join(timeout=2)
                    if proc.is_alive():
                        msg = f"Render subprocess {proc.pid} did not exit after being killed"
                        raise RuntimeError(msg)
            finally:
                try:
                    if page_indices is None or expected_page_count is not None:
                        with contextlib.suppress(queue.Empty):
                            page_count = page_count_q.get(timeout=0.1)
                finally:
                    try:
                        result_q.close()
                    finally:
                        page_count_q.close()
        return pages, timed_out, page_count

    @staticmethod
    def _valid_render_result(pages: object, count: int) -> bool:
        return (
            isinstance(pages, list)
            and len(pages) == count
            and all(page is None or isinstance(page, Image.Image) for page in pages)
        )

    def _render_with_timeout(  # noqa: C901
        self,
        pdf_bytes: bytes,
        file_name: str,
        page_indices: list[int] | None = None,
        *,
        expected_page_count: int | None = None,
    ) -> list[Image.Image | None]:
        """Recover served-parser windows within one shared time and attempt budget.

        Every known physical page keeps its slot, including when a child exits
        without a result. The default parser retains its existing single-attempt
        timeout behavior. The initial timeout can be overridden on a run's stage
        instance without increasing either the retry timeout or recovery budget.
        """
        if expected_page_count is None:
            pages, timed_out, page_count = self._render_once(pdf_bytes, page_indices)
        else:
            pages, timed_out, page_count = self._render_once(
                pdf_bytes, page_indices, expected_page_count=expected_page_count
            )
            if page_count is not None and page_count != expected_page_count:
                msg = f"Stale num_pages for {file_name}: manifest={expected_page_count}, actual={page_count}"
                raise ValueError(msg)
        if not self.retain_failures:
            return [] if timed_out else pages
        requested = page_indices
        if requested is None:
            if page_count is not None:
                requested = list(range(min(page_count, self.max_pages)))
            elif not timed_out and isinstance(pages, list) and pages:
                requested = list(range(len(pages)))
            else:
                logger.warning(f"No render result or page count for {file_name}")
                return []
        if not timed_out and self._valid_render_result(pages, len(requested)):
            return pages

        deadline = time.monotonic() + self._RENDER_RECOVERY_BUDGET_S
        attempts = 1

        def recover(indexes: list[int]) -> list[Image.Image | None]:
            nonlocal attempts
            remaining = deadline - time.monotonic()
            if remaining <= 0 or attempts >= self._RENDER_MAX_ATTEMPTS:
                return [None] * len(indexes)
            attempts += 1
            result, timeout, _ = self._render_once(
                pdf_bytes, indexes, timeout_s=min(self._RENDER_RETRY_TIMEOUT_S, remaining)
            )
            if not timeout and self._valid_render_result(result, len(indexes)):
                return result
            if len(indexes) <= 1:
                return [None] * len(indexes)
            midpoint = len(indexes) // 2
            return recover(indexes[:midpoint]) + recover(indexes[midpoint:])

        logger.warning(f"Recovering failed render window for {file_name}; preserving physical page slots")
        if len(requested) <= 1:
            return [None] * len(requested)
        midpoint = len(requested) // 2
        return recover(requested[:midpoint]) + recover(requested[midpoint:])

    @staticmethod
    def _failed_row(sample_id: str, file_name: str, url: str, stage: str, message: str) -> dict[str, Any]:
        return {
            "sample_id": sample_id,
            "position": -1,
            "page_number": None,
            "modality": "page_image",
            "content_type": "image/png",
            "text_content": "",
            "binary_content": None,
            "source_ref": None,
            "url": url,
            "pdf_name": file_name,
            "materialize_error": json.dumps({"stage": stage, "type": "PDFError", "message": message}),
        }

    def _page_row(
        self, sample_id: str, file_name: str, url: str, page_num: int, page_img: Image.Image | None
    ) -> dict[str, Any]:
        row = {
            "sample_id": sample_id,
            "position": page_num,
            "modality": "page_image",
            "content_type": "image/png",
            "text_content": "",
            "binary_content": None if page_img is None else image_to_bytes(page_img),
            "source_ref": None,
            "url": url,
            "pdf_name": file_name,
        }
        if not self.retain_failures:
            return row
        error = None
        if page_img is None:
            error = json.dumps({"stage": "render_pdf", "type": "RenderError", "message": "Page rendering failed"})
        return {**row, "page_number": page_num, "materialize_error": error}

    def process(self, task: FileGroupTask) -> InterleavedBatch | None:  # noqa: C901, PLR0912, PLR0915
        self._check_extraction_settings()
        if self._extraction_config is not None and output_version(task._metadata) != output_version(
            {"pdf_extraction_config": self._extraction_config}
        ):
            msg = "Page task does not match this renderer's admitted extraction identity"
            raise ExtractionConfigMismatchError(msg)
        rows: list[dict[str, Any]] = []

        parsed: list[dict] = [json.loads(e) for e in task.data]
        if task._metadata.get("pdf_page_chunks") and not self.retain_failures:
            msg = "Page budgeting requires retain_failures=True"
            raise ValueError(msg)
        metrics = (
            {"pdf_read_s": 0.0, "pdf_render_s": 0.0, "pdf_source_bytes": 0.0, "pdf_source_reads": float(len(parsed))}
            if task._metadata.get("pdf_page_chunks")
            else None
        )

        def failed_rows(entry: dict, sample_id: str, url: str, stage: str, message: str) -> list[dict]:
            failed = self._failed_row(sample_id, entry["file_name"], url, stage, message)
            if chunk := entry.get("_page_chunk"):
                return [
                    {**failed, "position": page, "page_number": page, "document_id": chunk["document_id"]}
                    for page in chunk["page_indices"]
                ]
            return [failed]

        # Pre-fetch all JSONL PDFs in batch (one file open per source JSONL)
        jsonl_bytes: dict[int, bytes | None] | None = None
        if self.jsonl_base_dir is not None:
            started = time.perf_counter() if metrics is not None else None
            jsonl_bytes = self._batch_fetch_jsonl(parsed)
            if metrics is not None:
                metrics["pdf_read_s"] += time.perf_counter() - started

        for idx, entry in enumerate(parsed):
            file_name = entry["file_name"]
            url = entry.get("url", "")
            sample_id = file_name.rsplit(".", 1)[0]

            started = time.perf_counter() if metrics is not None else None
            if jsonl_bytes is not None:
                pdf_bytes = jsonl_bytes.get(idx)
            else:
                pdf_bytes = self._get_pdf_bytes(file_name, entry=entry)
            if metrics is not None:
                metrics["pdf_read_s"] += time.perf_counter() - started
                metrics["pdf_source_bytes"] += len(pdf_bytes) if pdf_bytes is not None else 0
            if pdf_bytes is None:
                if self.retain_failures:
                    rows.extend(failed_rows(entry, sample_id, url, "read_pdf", "Could not read PDF"))
                logger.warning(f"Could not read PDF: {file_name}")
                continue

            page_indices = entry.get("selected_page_indices")
            if page_indices is not None and (
                not isinstance(page_indices, list)
                or not page_indices
                or any(not isinstance(page, int) or isinstance(page, bool) or page < 0 for page in page_indices)
                or len(page_indices) != len(set(page_indices))
            ):
                msg = "selected_page_indices must be a nonempty list of unique nonnegative integers"
                if self.retain_failures:
                    rows.extend(failed_rows(entry, sample_id, url, "manifest", msg))
                logger.warning(f"Invalid selected_page_indices for {file_name}: {page_indices!r}")
                continue

            started = time.perf_counter() if metrics is not None else None
            if entry.get("_page_chunk") and entry.get("num_pages") is not None:
                page_images = self._render_with_timeout(
                    pdf_bytes, file_name, page_indices, expected_page_count=entry["num_pages"]
                )
            else:
                page_images = self._render_with_timeout(pdf_bytes, file_name, page_indices)
            if metrics is not None:
                metrics["pdf_render_s"] += time.perf_counter() - started
            if not page_images:
                if self.retain_failures:
                    rows.extend(failed_rows(entry, sample_id, url, "render_pdf", "No render result"))
                logger.warning(f"No pages rendered from {file_name}; skipping")
                continue

            logger.debug(f"Rendered {file_name}: {len(page_images)} pages")
            physical_indexes = range(len(page_images)) if page_indices is None else page_indices
            page_rows = (
                self._page_row(sample_id, file_name, url, page_num, page_img)
                for page_num, page_img in zip(physical_indexes, page_images, strict=True)
                if self.retain_failures or page_img is not None
            )
            if chunk := entry.get("_page_chunk"):
                rows.extend({**row, "document_id": chunk["document_id"]} for row in page_rows)
            else:
                rows.extend(page_rows)

        if not rows:
            return None
        if metrics is not None:
            self._log_metrics(metrics)

        pages_df = pd.DataFrame(rows)
        if self.retain_failures:
            # An unopened document has no page number; without this pandas stores the column as float.
            pages_df["page_number"] = pages_df["page_number"].astype("Int64")
        metadata = {key: value for key, value in task._metadata.items() if key != PNG_METADATA_KEY}
        metadata = (
            {**metadata, PNG_METADATA_KEY: png_batch_provenance(row["binary_content"] for row in rows)}
            if self.reuse_encoded_images
            else metadata
        )
        return InterleavedBatch(
            dataset_name=task.dataset_name,
            data=pa.Table.from_pandas(pages_df, preserve_index=False),
            _metadata=metadata,
            _stage_perf=task._stage_perf,
        )
