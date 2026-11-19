# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small deterministic parser-output fixtures shared by page-budget tests."""

import json
from pathlib import Path

import pyarrow as pa
from PIL import Image

from nemo_curator.stages.interleaved.pdf.nemotron_parse.page_budget import output_version
from nemo_curator.stages.interleaved.pdf.nemotron_parse.partitioning import PDFPartitioningStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.postprocess import NemotronParsePostprocessStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.utils import image_to_bytes
from nemo_curator.tasks import EmptyTask, FileGroupTask, InterleavedBatch

TEST_EXTRACTION_CONFIG = {
    "schema": "pdf-page-extraction.v1",
    "parser": {"version": "test-parser", "source_sha256": "a" * 64},
    "model": {"revision": "test-immutable-model", "artifacts_sha256": "b" * 64},
    "request": {
        "model": "model@immutable",
        "stage": "fullpage",
        "prompt": "test prompt",
        "generation": {"max_tokens": 8744, "temperature": 0.2, "seed": 0},
    },
    "render": {
        "dpi": 300,
        "proc_size": [2048, 1664],
        "timeout_s": 60,
        "recovery_budget_s": 120,
        "retry_timeout_s": 60,
        "max_attempts": 7,
    },
    "transport": {"request_timeout_s": 300.0, "max_retries": 3, "retry_base_delay_s": 1.0},
    "projection": {"schema": "pdf-page-budget.v1", "min_crop_px": 10},
}
TEST_OUTPUT_VERSION = output_version({"pdf_extraction_config": TEST_EXTRACTION_CONFIG})


def write_manifest(tmp_path: Path, entries: list[dict]) -> Path:
    manifest = tmp_path / "manifest.jsonl"
    records = [
        {
            "source_locator": f"file:///corpus/{entry['file_name']}",
            "source_version": "test-immutable-version",
            "url": "https://example.test/document",
            **entry,
        }
        for entry in entries
    ]
    manifest.write_text("".join(json.dumps(entry) + "\n" for entry in records))
    return manifest


def partition(tmp_path: Path, entries: list[dict], size: int = 64, pdfs: int = 8) -> tuple[Path, list]:
    manifest = write_manifest(tmp_path, entries)
    tasks = PDFPartitioningStage(manifest_path=str(manifest), pages_per_task=size, pdfs_per_task=pdfs).process(
        EmptyTask()
    )
    return manifest, tasks


def project(
    task: FileGroupTask,
    *,
    failed: set[int] | None = None,
    no_elements: set[int] | None = None,
    text_only: set[int] | None = None,
) -> list[InterleavedBatch]:
    rows = []
    failed, no_elements = failed or set(), no_elements or set()
    for serialized in task.data:
        entry = json.loads(serialized)
        chunk = entry["_page_chunk"]
        for page in chunk["page_indices"]:
            blocks = [
                {"class": "Text", "bbox": [0.1, 0.1, 0.9, 0.9], "content": f"page {page}"},
                {"class": "Table", "bbox": [0.1, 0.1, 0.9, 0.9], "content": "<table/>"},
                {"class": "Picture", "bbox": [0.1, 0.1, 0.9, 0.9], "content": "picture"},
            ]
            if text_only is not None and page in text_only:
                blocks = blocks[:1]
            if page in no_elements:
                blocks = [{"class": "Picture", "bbox": [0, 0, 0, 0], "content": ""}]
            error = json.dumps({"stage": "fullpage", "type": "GenerationError", "message": "retained"})
            rows.append(
                {
                    "sample_id": chunk["sample_id"],
                    "document_id": chunk["document_id"],
                    "pdf_name": entry["file_name"],
                    "url": entry["url"],
                    "page_number": page,
                    "binary_content": image_to_bytes(Image.new("RGB", (20, 20), (page % 256, 0, 0))),
                    "parse_document": None if page in failed else json.dumps({"blocks": blocks}),
                    "materialize_error": error if page in failed else None,
                }
            )
    batch = InterleavedBatch(
        dataset_name="test",
        data=pa.Table.from_pylist(rows),
        _metadata={
            **task._metadata,
            "parse_mode": "single",
            "model_path": "model@immutable",
            "pdf_extraction_config": TEST_EXTRACTION_CONFIG,
        },
    )
    return NemotronParsePostprocessStage().process(batch)
