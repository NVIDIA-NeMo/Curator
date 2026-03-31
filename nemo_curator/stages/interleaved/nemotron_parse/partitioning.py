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

"""Partitioning stage for PDF processing pipelines."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import FileGroupTask, _EmptyTask


@dataclass
class PDFPartitioningStage(ProcessingStage[_EmptyTask, FileGroupTask]):
    """Read a JSONL manifest and produce FileGroupTasks for downstream processing.

    Each line in the JSONL file must contain at least a ``file_name`` field.
    An optional ``url`` field is preserved for provenance tracking.

    For CC-MAIN-2021-31-PDF-UNTRUNCATED datasets, the manifest can also use the
    ``cc_pdf_file_names`` field (a list of filenames per URL entry) along with
    ``url``.  Each filename is expanded into an individual entry.

    Example JSONL formats::

        # Simple: one PDF per line
        {"file_name": "0001234.pdf", "url": "http://example.com/doc.pdf"}

        # CC-MAIN: multiple PDFs per URL
        {"cc_pdf_file_names": ["0001234.pdf", "0001235.pdf"], "url": "http://..."}

    Parameters
    ----------
    manifest_path
        Path to a JSONL file listing PDFs to process.
    pdfs_per_task
        Number of PDFs to pack into each FileGroupTask.
    max_pdfs
        If set, limit the total number of PDFs to process.
    dataset_name
        Name assigned to output tasks.
    file_name_field
        JSONL field containing a single PDF filename.
    file_names_field
        JSONL field containing a list of PDF filenames (CC-MAIN style).
    url_field
        JSONL field containing the source URL.
    """

    manifest_path: str
    pdfs_per_task: int = 10
    max_pdfs: int | None = None
    dataset_name: str = "pdf_dataset"
    file_name_field: str = "file_name"
    file_names_field: str = "cc_pdf_file_names"
    url_field: str = "url"
    name: str = "pdf_partitioning"
    resources: Resources = field(default_factory=lambda: Resources(cpus=0.5))

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def xenna_stage_spec(self) -> dict[str, Any]:
        return {"num_workers_per_node": 1}

    def _parse_manifest(self) -> list[str]:
        """Read manifest and return list of JSON-serialized entries."""
        entries: list[str] = []

        with open(self.manifest_path) as f:
            for line in f:
                record = json.loads(line)
                url = record.get(self.url_field, "")

                if self.file_names_field in record:
                    # CC-MAIN style: multiple filenames per line, no per-file extra fields
                    file_names = record[self.file_names_field]
                    extra: dict = {}
                elif self.file_name_field in record:
                    # Single file per line — preserve extra fields (e.g. jsonl_file, byte_offset)
                    file_names = [record[self.file_name_field]]
                    extra = {k: v for k, v in record.items()
                             if k not in (self.file_name_field, self.url_field, self.file_names_field)}
                else:
                    logger.warning(f"Skipping manifest line: no '{self.file_name_field}' or '{self.file_names_field}'")
                    continue

                for fname in dict.fromkeys(file_names):
                    if not fname:
                        continue
                    entries.append(json.dumps({"file_name": fname, "url": url, **extra}))

                if self.max_pdfs and len(entries) >= self.max_pdfs:
                    entries = entries[: self.max_pdfs]
                    break

        return entries

    def process(self, _: _EmptyTask) -> list[FileGroupTask]:
        entries = self._parse_manifest()

        tasks: list[FileGroupTask] = []
        for i in range(0, len(entries), self.pdfs_per_task):
            batch = entries[i : i + self.pdfs_per_task]
            task_idx = i // self.pdfs_per_task
            task_id = f"pdf_batch_{task_idx:06d}"
            tasks.append(
                FileGroupTask(
                    task_id=task_id,
                    dataset_name=self.dataset_name,
                    data=batch,
                    _metadata={"source_files": batch, "partition_index": task_idx},
                )
            )

        logger.info(f"Created {len(tasks)} tasks from {len(entries)} PDFs")
        return tasks
