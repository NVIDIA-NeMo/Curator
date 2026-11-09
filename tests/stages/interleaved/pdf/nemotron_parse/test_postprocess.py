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

"""Model-owned content and landscape crop geometry."""

import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from nemo_curator.backends.ray_data.executor import RayDataExecutor
from nemo_curator.stages.interleaved.io import InterleavedParquetWriterStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.postprocess import NemotronParsePostprocessStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.reassembly import PDFDocumentMergeStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.schema import PAGE_OUTPUT_SCHEMA
from nemo_curator.stages.interleaved.pdf.nemotron_parse.utils import build_interleaved_rows, image_to_bytes
from nemo_curator.stages.interleaved.pdf.nemotron_parse.validation import PDFPageBudgetValidationStage
from nemo_curator.stages.interleaved.stages import BaseInterleavedFilterStage
from nemo_curator.tasks import FileGroupTask, InterleavedBatch
from nemo_curator.tasks.utils import TaskPerfUtils
from nemo_curator.utils.performance_utils import StagePerfStats
from tests.stages.interleaved.pdf.nemotron_parse._page_budget_helpers import (
    TEST_EXTRACTION_CONFIG,
    TEST_OUTPUT_VERSION,
    partition,
)


def test_landscape_picture_crops_original_content_not_center_padding():
    package = pytest.importorskip("nemotron_parse.inference")
    image = Image.new("RGB", (2200, 1700), "red")
    image.paste("blue", (220, 170, 1980, 1530))
    raw = "<x_0.1><y_0.1><x_0.9><y_0.9><class_Picture>"
    parser = package.DocumentParser(backend=lambda *_a, **_k: {"raw_out": raw, "finish_reason": "stop"})
    document = parser.parse(image)
    assert document["blocks"][0]["bbox_pixels"] == [220, 170, 1980, 1530]
    batch = InterleavedBatch(
        dataset_name="test",
        data=pa.Table.from_pylist(
            [
                {
                    "sample_id": "wide",
                    "url": "source",
                    "pdf_name": "wide.pdf",
                    "page_number": 4,
                    "binary_content": image_to_bytes(image),
                    "parse_document": json.dumps(document),
                    "materialize_error": None,
                }
            ]
        ),
        _metadata={"parse_mode": "single"},
    )
    metadata, row = NemotronParsePostprocessStage().process(batch).to_pyarrow().to_pylist()
    assert (metadata["position"], json.loads(metadata["text_content"])["num_pages"]) == (-1, 1)
    with Image.open(io.BytesIO(row["binary_content"])) as crop:
        assert crop.size == (1760, 1360)
        assert crop.getextrema() == ((0, 0), (0, 0), (255, 255))
    assert row["page_number"] == 4


def test_legacy_projection_remains_unchanged():
    image = Image.new("RGB", (220, 170), "red")
    raw = "<x_0.1><y_0.1>Hello<x_0.9><y_0.9><class_Text>"
    batch = InterleavedBatch(
        dataset_name="test",
        data=pa.Table.from_pylist(
            [
                {
                    "sample_id": "legacy",
                    "url": "source",
                    "pdf_name": "legacy.pdf",
                    "position": 0,
                    "binary_content": image_to_bytes(image),
                    "text_content": raw,
                }
            ]
        ),
        _metadata={"model_path": "nvidia/NVIDIA-Nemotron-Parse-v1.2"},
    )
    actual = NemotronParsePostprocessStage().process(batch).to_pyarrow().to_pylist()
    expected = build_interleaved_rows("legacy", "source", "legacy.pdf", [image], [raw], reorder_floaters=False)
    expected = [{**row, "materialize_error": None} for row in expected]
    assert actual == expected


def test_legacy_selected_pages_keep_pdf_page_numbers():
    page = image_to_bytes(Image.new("RGB", (220, 170), "red"))
    raw = "<x_0.1><y_0.1>Hello<x_0.9><y_0.9><class_Text>"
    batch = InterleavedBatch(
        dataset_name="test",
        data=pa.Table.from_pylist(
            [
                {
                    "sample_id": "s",
                    "url": "u",
                    "pdf_name": "s.pdf",
                    "position": position,
                    "binary_content": page,
                    "text_content": raw,
                }
                for position in (2, 0)
            ]
        ),
        _metadata={"model_path": "nvidia/NVIDIA-Nemotron-Parse-v1.2"},
    )
    rows = [
        row
        for row in NemotronParsePostprocessStage().process(batch).to_pyarrow().to_pylist()
        if row["modality"] == "text"
    ]
    assert [row["page_number"] for row in rows] == [0, 2]
    assert [json.loads(row["source_ref"])["page"] for row in rows] == [0, 2]


def test_error_row_roundtrip_uses_writer_warn_policy(tmp_path: Path):
    from nemo_curator.stages.interleaved.io import InterleavedParquetWriterStage

    error = json.dumps({"stage": "fullpage", "type": "GenerationError", "message": "cut off"})
    batch = InterleavedBatch(
        dataset_name="test",
        data=pa.Table.from_pylist(
            [
                {
                    "sample_id": "failed",
                    "pdf_name": "failed.pdf",
                    "url": "source",
                    "position": 5,
                    "page_number": 5,
                    "modality": "page_image",
                    "binary_content": None,
                    "parse_document": None,
                    "materialize_error": error,
                }
            ]
        ),
        _metadata={"parse_mode": "single"},
    )
    projected = NemotronParsePostprocessStage().process(batch)
    assert BaseInterleavedFilterStage._basic_row_validity_mask(projected.to_pandas()).all()
    strict = InterleavedParquetWriterStage(path=str(tmp_path / "strict"), materialize_on_write=False)
    with pytest.raises(RuntimeError, match="Materialization failed"):
        strict.process(projected)
    writer = InterleavedParquetWriterStage(
        path=str(tmp_path / "warn"), materialize_on_write=False, on_materialize_error="warn"
    )
    writer.process(projected)
    files = list((tmp_path / "warn").glob("*.parquet"))
    assert len(files) == 1
    metadata, failed = pq.read_table(files[0]).to_pylist()
    assert (metadata["position"], metadata["materialize_error"]) == (-1, None)
    assert (failed["position"], failed["modality"], failed["page_number"]) == (0, "text", 5)
    assert failed["materialize_error"] == error


def test_outputs_with_and_without_errors_read_back_together(tmp_path: Path):
    from nemo_curator.stages.interleaved.io import InterleavedParquetWriterStage
    from nemo_curator.stages.interleaved.io.readers.parquet import InterleavedParquetReaderStage

    error = json.dumps({"stage": "fullpage", "type": "GenerationError", "message": "cut off"})
    document = json.dumps({"blocks": [{"class": "Text", "bbox": [0.1, 0.1, 0.9, 0.9], "content": "Hello"}]})
    pages = {
        "parsed": {"page_number": 0, "binary_content": image_to_bytes(Image.new("RGB", (220, 170))), "error": None},
        "failed": {"page_number": 0, "binary_content": None, "error": error},
        "unopened": {"page_number": None, "binary_content": None, "error": error},
    }
    writer = InterleavedParquetWriterStage(path=str(tmp_path), materialize_on_write=False, on_materialize_error="warn")
    files = []
    for sample_id, page in pages.items():
        row = {
            "sample_id": sample_id,
            "pdf_name": f"{sample_id}.pdf",
            "url": "source",
            "page_number": page["page_number"],
            "binary_content": page["binary_content"],
            "parse_document": None if page["error"] else document,
            "materialize_error": page["error"],
        }
        batch = InterleavedBatch(
            dataset_name="test", data=pa.Table.from_pylist([row]), _metadata={"parse_mode": "single"}
        )
        files.extend(writer.process(NemotronParsePostprocessStage().process(batch)).data)
    read = InterleavedParquetReaderStage().process(FileGroupTask(dataset_name="test", data=files))
    rows = pa.concat_tables([batch.to_pyarrow() for batch in (read if isinstance(read, list) else [read])]).to_pylist()
    errors = {row["sample_id"]: row["materialize_error"] for row in rows if row["materialize_error"]}
    assert errors == {"failed": error, "unopened": error}
    assert any(row["sample_id"] == "parsed" and row["text_content"] == "Hello" for row in rows)


@pytest.mark.usefixtures("shared_ray_client")
def test_page_fanout_metrics_survive_real_ray_regrouping(tmp_path: Path):
    entries = [{"file_name": f"short-{index}.pdf", "num_pages": 2} for index in range(8)]
    manifest, tasks = partition(tmp_path, [*entries, {"file_name": "long.pdf", "num_pages": 40}], size=16)
    source_stats = StagePerfStats("source", process_time=0.25)
    initial = []
    image = image_to_bytes(Image.new("RGB", (20, 20)))
    document = json.dumps({"blocks": [{"class": "Text", "bbox": [0.1, 0.1, 0.9, 0.9], "content": "text"}]})
    for task in tasks:
        rows = []
        for serialized in task.data:
            entry = json.loads(serialized)
            chunk = entry["_page_chunk"]
            for page in chunk["page_indices"]:
                failed = page == 0
                rows.append(
                    {
                        "sample_id": chunk["sample_id"],
                        "document_id": chunk["document_id"],
                        "pdf_name": entry["file_name"],
                        "url": entry["url"],
                        "page_number": page,
                        "binary_content": image,
                        "parse_document": None if failed else document,
                        "materialize_error": json.dumps({"message": "retained failure"}) if failed else None,
                    }
                )
        valid = sum(row["materialize_error"] is None for row in rows)
        initial.append(
            InterleavedBatch(
                dataset_name="test",
                data=pa.Table.from_pylist(rows),
                _metadata={**task._metadata, "parse_mode": "single", "pdf_extraction_config": TEST_EXTRACTION_CONFIG},
                _stage_perf=[
                    source_stats,
                    StagePerfStats(
                        "nemotron_parse_inference",
                        process_time=1.0,
                        num_items_processed=len(rows),
                        custom_metrics={
                            "num_input_pages": float(len(rows)),
                            "num_valid_pages": float(valid),
                            "num_error_pages": float(len(rows) - valid),
                            "total_prompt_tokens": float(len(rows) * 10),
                            "total_output_tokens": float(valid * 4),
                        },
                    ),
                ],
            )
        )
    results = RayDataExecutor().execute(
        [
            NemotronParsePostprocessStage(),
            PDFDocumentMergeStage(),
            InterleavedParquetWriterStage(
                path=str(tmp_path / "out"),
                materialize_on_write=False,
                on_materialize_error="warn",
                schema=PAGE_OUTPUT_SCHEMA,
                reject_unknown_columns=True,
                atomic_local_write=True,
                use_output_file_id=True,
                mode="error",
            ),
            PDFPageBudgetValidationStage(manifest_path=str(manifest), expected_output_version=TEST_OUTPUT_VERSION),
        ],
        initial_tasks=initial,
    )
    assert len(results) == 9
    validation = results[0]._metadata["pdf_page_budget_validation"]
    assert validation["pages"] == 56
    assert validation["successful_pages"] == 47
    assert validation["failed_pages"] == 9
    metrics = TaskPerfUtils.aggregate_task_metrics(results)
    assert metrics["source_process_time_sum"] == 0.25
    assert metrics["nemotron_parse_inference_process_time_sum"] == 4.0
    for name, expected in {
        "num_input_pages": 56,
        "num_valid_pages": 47,
        "num_error_pages": 9,
        "total_prompt_tokens": 560,
        "total_output_tokens": 188,
    }.items():
        assert metrics[f"nemotron_parse_inference_custom.{name}_sum"] == expected
    stages = TaskPerfUtils.collect_stage_metrics(results)
    assert len(stages["nemotron_parse_postprocess"]["process_time"]) == len(initial)
    # InterleavedBatch.num_items counts unique documents, not page rows.
    assert stages["nemotron_parse_postprocess"]["num_items_processed"].sum() == sum(task.num_items for task in initial)
    assert (stages["nemotron_parse_postprocess"]["process_time"] > 0).all()
