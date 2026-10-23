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

import json
import multiprocessing
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd
import pyarrow as pa
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.utils import (
    extract_pdf_from_jsonl,
    extract_pdf_from_zip,
    extract_pdfs_from_jsonl_batch,
    image_to_bytes,
    render_pdf_pages,
)
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import FileGroupTask, InterleavedBatch

if TYPE_CHECKING:
    from PIL import Image


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
    """

    zip_base_dir: str | None = None
    pdf_dir: str | None = None
    jsonl_base_dir: str | None = None
    dpi: int = 300
    max_pages: int = 50
    retain_failures: bool = False
    name: str = "pdf_preprocess"
    resources: Resources = field(default_factory=lambda: Resources(cpus=2.0))

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

    _RENDER_TIMEOUT_S: int = 60  # hard limit per PDF render; kills hung C-extension

    def _render_with_timeout(self, pdf_bytes: bytes, file_name: str, page_indices: list[int] | None = None) -> list:
        """Render PDF with a process-based timeout.

        SIGALRM cannot be used here because Xenna runs stage workers in non-main
        threads and signal.signal() is restricted to the main thread.  Instead we
        fork a child process (inheriting pdf_bytes via copy-on-write) and kill it
        if it exceeds the timeout, which reliably escapes any C-extension hang.
        """
        dpi = self.dpi
        max_pages = self.max_pages

        ctx = multiprocessing.get_context("fork")
        result_q: multiprocessing.Queue = ctx.Queue()

        keep_failed = self.retain_failures

        def _worker() -> None:
            try:
                pages = render_pdf_pages(
                    pdf_bytes,
                    dpi=dpi,
                    max_pages=max_pages,
                    keep_failed=keep_failed,
                    page_indices=page_indices,
                )
                result_q.put(pages)
            except Exception:  # noqa: BLE001
                result_q.put([])

        proc = ctx.Process(target=_worker)
        proc.start()
        # Drain the queue BEFORE joining. If we join first, the child can
        # deadlock: after result_q.put(pages) the internal feeder thread must
        # flush all serialized data through the OS pipe; with 50 pages at
        # 300 DPI the pipe fills up and the feeder stalls until the parent
        # reads. Since the parent is blocked in join(), neither side makes
        # progress and the 60 s timeout fires on a perfectly valid render.
        try:
            pages = result_q.get(timeout=self._RENDER_TIMEOUT_S)
        except Exception:  # noqa: BLE001
            pages = []
        proc.join(timeout=2)
        if proc.is_alive():
            proc.kill()
            proc.join()
            if not pages:
                logger.warning(f"Render timed out ({self._RENDER_TIMEOUT_S}s) for {file_name}, skipping")
                return []
            logger.debug(f"Forcibly cleaned up render process for {file_name} (result was obtained)")
        return pages

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

    def process(self, task: FileGroupTask) -> InterleavedBatch | None:  # noqa: C901
        rows: list[dict[str, Any]] = []

        parsed: list[dict] = [json.loads(e) for e in task.data]

        # Pre-fetch all JSONL PDFs in batch (one file open per source JSONL)
        jsonl_bytes: dict[int, bytes | None] | None = None
        if self.jsonl_base_dir is not None:
            jsonl_bytes = self._batch_fetch_jsonl(parsed)

        for idx, entry in enumerate(parsed):
            file_name = entry["file_name"]
            url = entry.get("url", "")
            sample_id = file_name.rsplit(".", 1)[0]

            if jsonl_bytes is not None:
                pdf_bytes = jsonl_bytes.get(idx)
            else:
                pdf_bytes = self._get_pdf_bytes(file_name, entry=entry)
            if pdf_bytes is None:
                if self.retain_failures:
                    rows.append(self._failed_row(sample_id, file_name, url, "read_pdf", "Could not read PDF"))
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
                    rows.append(self._failed_row(sample_id, file_name, url, "manifest", msg))
                logger.warning(f"Invalid selected_page_indices for {file_name}: {page_indices!r}")
                continue

            page_images = self._render_with_timeout(pdf_bytes, file_name, page_indices)
            if not page_images:
                if self.retain_failures:
                    rows.append(self._failed_row(sample_id, file_name, url, "render_pdf", "No render result"))
                logger.warning(f"No pages rendered from {file_name}; skipping")
                continue

            logger.debug(f"Rendered {file_name}: {len(page_images)} pages")
            physical_indexes = range(len(page_images)) if page_indices is None else page_indices
            rows.extend(
                self._page_row(sample_id, file_name, url, page_num, page_img)
                for page_num, page_img in zip(physical_indexes, page_images, strict=True)
            )

        if not rows:
            return None

        pages_df = pd.DataFrame(rows)
        if self.retain_failures:
            # An unopened document has no page number; without this pandas stores the column as float.
            pages_df["page_number"] = pages_df["page_number"].astype("Int64")
        return InterleavedBatch(
            dataset_name=task.dataset_name,
            data=pa.Table.from_pandas(pages_df, preserve_index=False),
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )
