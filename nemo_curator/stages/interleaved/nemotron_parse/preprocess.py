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
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pyarrow as pa
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.interleaved.nemotron_parse.utils import (
    extract_pdf_from_zip,
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

    PDF bytes are obtained in one of two ways:

    - **Zip archive mode** (``zip_base_dir`` is set): PDFs are extracted from
      CC-MAIN-style zip archives using :func:`extract_pdf_from_zip`.
    - **Directory mode** (``pdf_dir`` is set): PDFs are read directly from
      ``<pdf_dir>/<file_name>``.

    Produces an :class:`InterleavedBatch` with one row per page, where
    ``binary_content`` holds the PNG-encoded page image and ``text_content``
    is empty (to be filled by the GPU inference stage).

    Parameters
    ----------
    zip_base_dir
        Root of CC-MAIN zip archive hierarchy.  Mutually exclusive with ``pdf_dir``.
    pdf_dir
        Directory containing loose PDF files.  Mutually exclusive with ``zip_base_dir``.
    dpi
        Resolution for PDF page rendering.
    max_pages
        Maximum number of pages to render per PDF.
    """

    zip_base_dir: str | None = None
    pdf_dir: str | None = None
    dpi: int = 300
    max_pages: int = 50
    name: str = "pdf_preprocess"
    resources: Resources = field(default_factory=lambda: Resources(cpus=2.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def _get_pdf_bytes(self, file_name: str) -> bytes | None:
        if self.zip_base_dir is not None:
            return extract_pdf_from_zip(file_name, self.zip_base_dir)
        if self.pdf_dir is not None:
            import os

            path = os.path.join(self.pdf_dir, file_name)
            try:
                with open(path, "rb") as f:
                    return f.read()
            except OSError:
                return None
        msg = "Either zip_base_dir or pdf_dir must be set"
        raise ValueError(msg)

    def process(self, task: FileGroupTask) -> InterleavedBatch | None:
        rows: list[dict[str, Any]] = []

        for entry_json in task.data:
            entry = json.loads(entry_json)
            file_name = entry["file_name"]
            url = entry.get("url", "")
            sample_id = file_name.rsplit(".", 1)[0]

            pdf_bytes = self._get_pdf_bytes(file_name)
            if pdf_bytes is None:
                logger.warning(f"Could not read PDF: {file_name}")
                continue

            page_images = render_pdf_pages(pdf_bytes, dpi=self.dpi, max_pages=self.max_pages)
            if not page_images:
                logger.warning(f"0 pages rendered for {file_name}")
                continue

            logger.debug(f"Rendered {file_name}: {len(page_images)} pages")
            for page_num, page_img in enumerate(page_images):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "position": page_num,
                        "modality": "page_image",
                        "content_type": "image/png",
                        "text_content": "",
                        "binary_content": image_to_bytes(page_img),
                        "source_ref": None,
                        "url": url,
                        "pdf_name": file_name,
                    }
                )

        if not rows:
            return None

        pages_df = pd.DataFrame(rows)
        return InterleavedBatch(
            task_id=f"{task.task_id}_preprocessed",
            dataset_name=task.dataset_name,
            data=pa.Table.from_pandas(pages_df, preserve_index=False),
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )
