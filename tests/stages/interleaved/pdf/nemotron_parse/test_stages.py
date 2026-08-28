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

"""Tests for Nemotron-Parse pipeline stages (CPU-only, no GPU required)."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import json
import zipfile
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.interleaved.pdf.nemotron_parse.partitioning import PDFPartitioningStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.postprocess import NemotronParsePostprocessStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.preprocess import PDFPreprocessStage
from nemo_curator.tasks import EmptyTask

if TYPE_CHECKING:
    from pathlib import Path


def _empty_task() -> EmptyTask:
    return EmptyTask(dataset_name="test", data=None)


class TestPDFPartitioningStage:
    def test_worker_defaults(self, tmp_path: Path):
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text(json.dumps({"file_name": "a.pdf"}) + "\n")

        stage = PDFPartitioningStage(manifest_path=str(manifest))

        assert stage.ray_stage_spec()[RayStageSpecKeys.IS_FANOUT_STAGE] is True
        assert stage.num_workers() == 1
        assert stage.xenna_stage_spec() == {}

    def test_simple_manifest(self, tmp_path: Path):
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text(
            json.dumps({"file_name": "a.pdf", "url": "http://a"})
            + "\n"
            + json.dumps({"file_name": "b.pdf", "url": "http://b"})
            + "\n"
        )

        stage = PDFPartitioningStage(
            manifest_path=str(manifest),
            pdfs_per_task=2,
        )
        tasks = stage.process(_empty_task())
        assert len(tasks) == 1
        assert len(tasks[0].data) == 2

    def test_cc_main_manifest_format(self, tmp_path: Path):
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text(json.dumps({"cc_pdf_file_names": ["001.pdf", "002.pdf"], "url": "http://x"}) + "\n")

        stage = PDFPartitioningStage(
            manifest_path=str(manifest),
            pdfs_per_task=5,
        )
        tasks = stage.process(_empty_task())
        assert len(tasks) == 1
        entries = [json.loads(e) for e in tasks[0].data]
        assert len(entries) == 2
        assert entries[0]["file_name"] == "001.pdf"

    def test_max_pdfs_limit(self, tmp_path: Path):
        manifest = tmp_path / "manifest.jsonl"
        lines = [json.dumps({"file_name": f"{i}.pdf"}) for i in range(20)]
        manifest.write_text("\n".join(lines) + "\n")

        stage = PDFPartitioningStage(
            manifest_path=str(manifest),
            pdfs_per_task=5,
            max_pdfs=7,
        )
        tasks = stage.process(_empty_task())
        total_pdfs = sum(len(t.data) for t in tasks)
        assert total_pdfs == 7

    def test_multiple_tasks(self, tmp_path: Path):
        manifest = tmp_path / "manifest.jsonl"
        lines = [json.dumps({"file_name": f"{i}.pdf"}) for i in range(5)]
        manifest.write_text("\n".join(lines) + "\n")

        stage = PDFPartitioningStage(
            manifest_path=str(manifest),
            pdfs_per_task=2,
        )
        tasks = stage.process(_empty_task())
        assert len(tasks) == 3
        assert len(tasks[0].data) == 2
        assert len(tasks[2].data) == 1

    def test_extra_fields_preserved_in_single_file_entry(self, tmp_path: Path):
        """Extra fields like jsonl_file and byte_offset must be forwarded downstream."""
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text(
            json.dumps({"file_name": "a.pdf", "url": "http://a", "jsonl_file": "x.jsonl", "byte_offset": 42}) + "\n"
        )
        stage = PDFPartitioningStage(manifest_path=str(manifest), pdfs_per_task=5)
        tasks = stage.process(_empty_task())
        assert len(tasks) == 1
        entry = json.loads(tasks[0].data[0])
        assert entry["jsonl_file"] == "x.jsonl"
        assert entry["byte_offset"] == 42

    def test_unrecognized_line_is_skipped(self, tmp_path: Path):
        """Lines without file_name or cc_pdf_file_names should be skipped with a warning."""
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text(json.dumps({"unknown_key": "value"}) + "\n" + json.dumps({"file_name": "b.pdf"}) + "\n")
        stage = PDFPartitioningStage(manifest_path=str(manifest), pdfs_per_task=5)
        tasks = stage.process(_empty_task())
        total = sum(len(t.data) for t in tasks)
        assert total == 1
        assert json.loads(tasks[0].data[0])["file_name"] == "b.pdf"

    def test_duplicate_filenames_deduplicated(self, tmp_path: Path):
        """dict.fromkeys deduplication should collapse repeated filenames."""
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text(json.dumps({"cc_pdf_file_names": ["a.pdf", "a.pdf", "b.pdf"], "url": "http://x"}) + "\n")
        stage = PDFPartitioningStage(manifest_path=str(manifest), pdfs_per_task=5)
        tasks = stage.process(_empty_task())
        entries = [json.loads(e) for e in tasks[0].data]
        assert len(entries) == 2
        assert entries[0]["file_name"] == "a.pdf"
        assert entries[1]["file_name"] == "b.pdf"

    def test_blank_lines_ignored(self, tmp_path: Path):
        """Blank lines in the manifest should be silently skipped."""
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text("\n" + json.dumps({"file_name": "a.pdf"}) + "\n\n")
        stage = PDFPartitioningStage(manifest_path=str(manifest), pdfs_per_task=5)
        tasks = stage.process(_empty_task())
        assert sum(len(t.data) for t in tasks) == 1


def _has_pypdfium2() -> bool:
    try:
        import pypdfium2  # noqa: F401
    except ImportError:
        return False
    else:
        return True


@pytest.mark.skipif(not _has_pypdfium2(), reason="pypdfium2 not installed")
class TestPDFPreprocessStage:
    @staticmethod
    def _make_minimal_pdf() -> bytes:
        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument.new()
        doc.new_page(width=100, height=100)
        pdf_bytes = doc.save()
        doc.close()
        return bytes(pdf_bytes)

    def test_pdf_dir_mode(self, tmp_path: Path):
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        (pdf_dir / "test.pdf").write_bytes(self._make_minimal_pdf())

        entry = json.dumps({"file_name": "test.pdf", "url": "http://test"})
        from nemo_curator.tasks import FileGroupTask

        task = FileGroupTask(
            dataset_name="test",
            data=[entry],
        )

        stage = PDFPreprocessStage(pdf_dir=str(pdf_dir))
        result = stage.process(task)
        assert result is not None
        result_df = result.to_pandas()
        assert len(result_df) == 1
        assert result_df["sample_id"].iloc[0] == "test"
        assert result_df["modality"].iloc[0] == "page_image"
        assert len(result_df["binary_content"].iloc[0]) > 0

    def test_missing_pdf_returns_none(self, tmp_path: Path):
        pdf_dir = tmp_path / "empty"
        pdf_dir.mkdir()

        entry = json.dumps({"file_name": "missing.pdf"})
        from nemo_curator.tasks import FileGroupTask

        task = FileGroupTask(
            dataset_name="test",
            data=[entry],
        )

        stage = PDFPreprocessStage(pdf_dir=str(pdf_dir))
        result = stage.process(task)
        assert result is None

    def test_zip_mode(self, tmp_path: Path):
        """PDFs extracted from CC-MAIN zip archives."""
        zip_dir = tmp_path / "0000-0999"
        zip_dir.mkdir(parents=True)
        pdf_bytes = self._make_minimal_pdf()
        with zipfile.ZipFile(zip_dir / "0001.zip", "w") as zf:
            zf.writestr("0001234.pdf", pdf_bytes)

        entry = json.dumps({"file_name": "0001234.pdf", "url": "http://test"})
        from nemo_curator.tasks import FileGroupTask

        task = FileGroupTask(dataset_name="test", data=[entry])
        stage = PDFPreprocessStage(zip_base_dir=str(tmp_path))
        result = stage.process(task)
        assert result is not None
        assert len(result.to_pandas()) >= 1

    def test_jsonl_mode_with_byte_offset(self, tmp_path: Path):
        """PDFs decoded from base64 JSONL (GitHub-style) using byte_offset fast path."""
        pdf_bytes = self._make_minimal_pdf()
        content = base64.b64encode(pdf_bytes).decode()
        line = json.dumps({"content": content}) + "\n"

        jsonl_dir = tmp_path / "jsonl"
        jsonl_dir.mkdir()
        (jsonl_dir / "data.jsonl").write_bytes(line.encode())

        entry = json.dumps(
            {"file_name": "test.pdf", "url": "http://test", "jsonl_file": "data.jsonl", "byte_offset": 0}
        )
        from nemo_curator.tasks import FileGroupTask

        task = FileGroupTask(dataset_name="test", data=[entry])
        stage = PDFPreprocessStage(jsonl_base_dir=str(jsonl_dir))
        result = stage.process(task)
        assert result is not None
        assert len(result.to_pandas()) >= 1

    def test_jsonl_mode_with_line_idx(self, tmp_path: Path):
        """PDFs decoded from base64 JSONL using legacy line_idx fallback path."""
        pdf_bytes = self._make_minimal_pdf()
        content = base64.b64encode(pdf_bytes).decode()
        # Two lines; target is line 1
        line0 = json.dumps({"content": base64.b64encode(b"other").decode()}) + "\n"
        line1 = json.dumps({"content": content}) + "\n"

        jsonl_dir = tmp_path / "jsonl"
        jsonl_dir.mkdir()
        (jsonl_dir / "data.jsonl").write_bytes((line0 + line1).encode())

        # No byte_offset → falls back to line_idx scan
        entry = json.dumps({"file_name": "test.pdf", "url": "http://test", "jsonl_file": "data.jsonl", "line_idx": 1})
        from nemo_curator.tasks import FileGroupTask

        task = FileGroupTask(dataset_name="test", data=[entry])
        stage = PDFPreprocessStage(jsonl_base_dir=str(jsonl_dir))
        result = stage.process(task)
        assert result is not None

    def test_no_mode_raises_value_error(self):
        """When no source mode is configured, process() should raise ValueError."""
        entry = json.dumps({"file_name": "test.pdf"})
        from nemo_curator.tasks import FileGroupTask

        task = FileGroupTask(dataset_name="test", data=[entry])
        stage = PDFPreprocessStage()
        with pytest.raises(ValueError, match="One of"):
            stage.process(task)


class TestNemotronParsePostprocessStage:
    def test_postprocess_basic(self):
        import pandas as pd
        import pyarrow as pa

        from nemo_curator.tasks import InterleavedBatch

        img = Image.new("RGB", (100, 100), color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        result_df = pd.DataFrame(
            [
                {
                    "sample_id": "s1",
                    "position": 0,
                    "modality": "page_image",
                    "content_type": "image/png",
                    "text_content": "<x_0.0><y_0.0>Hello<x_1.0><y_1.0><class_Text>",
                    "binary_content": buf.getvalue(),
                    "source_ref": None,
                    "url": "http://test",
                    "pdf_name": "test.pdf",
                }
            ]
        )

        task = InterleavedBatch(
            dataset_name="test",
            data=pa.Table.from_pandas(result_df),
            _metadata={"proc_size": [100, 100], "model_path": "v1.2"},
        )

        stage = NemotronParsePostprocessStage(proc_size=(100, 100))
        result = stage.process(task)
        assert result is not None
        out_df = result.to_pandas()
        assert len(out_df) >= 2
        assert out_df.iloc[0]["modality"] == "metadata"
        text_rows = out_df[out_df["modality"] == "text"]
        assert len(text_rows) == 1
        assert text_rows.iloc[0]["text_content"] == "Hello"

    def test_no_valid_output_returns_none(self):
        """A task where all pages have empty model output produces no rows."""
        import pandas as pd
        import pyarrow as pa

        from nemo_curator.tasks import InterleavedBatch

        img = Image.new("RGB", (10, 10), color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        result_df = pd.DataFrame(
            [
                {
                    "sample_id": "s1",
                    "position": 0,
                    "modality": "page_image",
                    "content_type": "image/png",
                    "text_content": "",
                    "binary_content": buf.getvalue(),
                    "source_ref": None,
                    "url": "http://test",
                    "pdf_name": "test.pdf",
                }
            ]
        )

        task = InterleavedBatch(
            dataset_name="test",
            data=pa.Table.from_pandas(result_df),
            _metadata={"proc_size": [100, 100], "model_path": "v1.2"},
        )

        stage = NemotronParsePostprocessStage(proc_size=(100, 100))
        result = stage.process(task)
        # Empty model output still produces a metadata row, so result is not None
        assert result is not None
        out_df = result.to_pandas()
        assert out_df.iloc[0]["modality"] == "metadata"


class TestNemotronParseInferenceStageMetrics:
    def test_vllm_metrics_from_outputs(self) -> None:
        from nemo_curator.stages.interleaved.pdf.nemotron_parse.inference import NemotronParseInferenceStage

        outputs = [
            SimpleNamespace(
                prompt_token_ids=[1, 2, 3],
                outputs=[SimpleNamespace(text="hello", token_ids=[4, 5, 6], finish_reason="stop")],
            ),
            SimpleNamespace(
                prompt_token_ids=[7, 8],
                outputs=[SimpleNamespace(text="", token_ids=[], finish_reason="length")],
            ),
        ]

        metrics = NemotronParseInferenceStage._vllm_metrics_from_outputs(
            outputs,
            inference_time_s=1.5,
            num_input_pages=3,
            num_valid_pages=2,
            num_skipped_pages=1,
            vllm_retries=1,
        )

        assert metrics["vllm_inference_time"] == 1.5
        assert metrics["num_input_pages"] == 3.0
        assert metrics["num_valid_pages"] == 2.0
        assert metrics["num_skipped_pages"] == 1.0
        assert metrics["total_prompt_tokens"] == 5.0
        assert metrics["total_output_tokens"] == 3.0
        assert metrics["total_output_chars"] == 5.0
        assert metrics["num_output_length_truncated"] == 1.0
        assert metrics["num_empty_outputs"] == 1.0
        assert metrics["vllm_retries"] == 1.0
        assert "avg_output_tokens_per_page" not in metrics
        assert "avg_output_chars_per_page" not in metrics

    def test_process_logs_vllm_metrics(self) -> None:
        import pandas as pd
        import pyarrow as pa

        from nemo_curator.stages.interleaved.pdf.nemotron_parse.inference import NemotronParseInferenceStage
        from nemo_curator.tasks import InterleavedBatch

        img = Image.new("RGB", (10, 10), color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        task_df = pd.DataFrame(
            [
                {
                    "sample_id": "s1",
                    "position": 0,
                    "modality": "page_image",
                    "content_type": "image/png",
                    "text_content": None,
                    "binary_content": buf.getvalue(),
                    "source_ref": None,
                }
            ]
        )
        task = InterleavedBatch(dataset_name="test", data=pa.Table.from_pandas(task_df))

        stage = NemotronParseInferenceStage(backend="vllm")
        stage._proc_size = (100, 100)

        raw_outputs = [
            SimpleNamespace(
                prompt_token_ids=[1, 2],
                outputs=[SimpleNamespace(text="parsed", token_ids=[3, 4, 5], finish_reason="stop")],
            )
        ]

        with patch.object(stage, "_infer_vllm", return_value=(["parsed"], raw_outputs, 0)):
            stage.process(task)

        assert hasattr(stage, "_custom_metrics")
        assert stage._custom_metrics["num_valid_pages"] == 1.0
        assert stage._custom_metrics["total_output_tokens"] == 3.0
        assert stage._custom_metrics["total_output_chars"] == 6.0
        assert "vllm_inference_time" in stage._custom_metrics

    def test_setup_vllm_engine_kwargs_override_stage_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys
        import types

        from nemo_curator.stages.interleaved.pdf.nemotron_parse.inference import NemotronParseInferenceStage
        from nemo_curator.utils import vllm_utils

        fake_vllm = types.ModuleType("vllm")

        class FakeSamplingParams:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake_vllm.SamplingParams = FakeSamplingParams
        monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

        captured_kwargs: dict = {}

        def fake_create_vllm_llm(model_path: str, **kwargs) -> object:
            captured_kwargs["model_path"] = model_path
            captured_kwargs.update(kwargs)
            return object()

        fake_processor = SimpleNamespace(image_processor=SimpleNamespace(final_size=(100, 100)))
        monkeypatch.setattr(vllm_utils, "resolve_local_model_path", lambda _path: "/models/nemotron")
        monkeypatch.setattr(vllm_utils, "create_vllm_llm", fake_create_vllm_llm)
        monkeypatch.setattr("transformers.AutoProcessor.from_pretrained", lambda *_args, **_kwargs: fake_processor)

        stage = NemotronParseInferenceStage(
            backend="vllm",
            max_num_seqs=64,
            enforce_eager=False,
            engine_kwargs={
                "max_num_seqs": 8,
                "enforce_eager": True,
                "gpu_memory_utilization": 0.9,
            },
        )

        stage._setup_vllm()

        assert captured_kwargs["model_path"] == "/models/nemotron"
        assert captured_kwargs["max_num_seqs"] == 8
        assert captured_kwargs["enforce_eager"] is True
        assert captured_kwargs["gpu_memory_utilization"] == 0.9
        assert stage._proc_size == (100, 100)

    def test_in_process_and_server_sampling_parameters_match(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys
        import types

        from nemo_curator.stages.interleaved.pdf.nemotron_parse.inference import (
            NemotronParseInferenceServerStage,
            NemotronParseInferenceStage,
        )
        from nemo_curator.utils import vllm_utils

        fake_vllm = types.ModuleType("vllm")

        class FakeSamplingParams:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake_vllm.SamplingParams = FakeSamplingParams
        monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
        monkeypatch.setattr(vllm_utils, "resolve_local_model_path", lambda _path: "/models/nemotron")
        monkeypatch.setattr(vllm_utils, "create_vllm_llm", lambda *_args, **_kwargs: object())
        fake_processor = SimpleNamespace(image_processor=SimpleNamespace(final_size=(100, 100)))
        monkeypatch.setattr("transformers.AutoProcessor.from_pretrained", lambda *_args, **_kwargs: fake_processor)

        in_process_stage = NemotronParseInferenceStage(backend="vllm", max_tokens=1234)
        in_process_stage._setup_vllm()
        server_stage = NemotronParseInferenceServerStage(
            endpoint="http://localhost:8000/v1",
            model_name="nemotron-parse",
            max_tokens=1234,
        )

        expected = {
            "temperature": 0,
            "top_p": 1.0,
            "top_k": 1,
            "repetition_penalty": 1.1,
            "max_tokens": 1234,
            "skip_special_tokens": False,
            "seed": None,
        }
        server_config = server_stage._generation_config
        server_params = {
            "temperature": server_config.temperature,
            "top_p": server_config.top_p,
            "max_tokens": server_config.max_tokens,
            "seed": server_config.seed,
            **server_config.extra_kwargs["extra_body"],
        }

        assert in_process_stage._sampling_params.kwargs == expected
        assert server_params == expected

    def test_in_process_and_server_default_to_8192_max_tokens(self) -> None:
        from nemo_curator.stages.interleaved.pdf.nemotron_parse.inference import (
            NemotronParseInferenceServerStage,
            NemotronParseInferenceStage,
        )

        in_process_stage = NemotronParseInferenceStage()
        server_stage = NemotronParseInferenceServerStage(
            endpoint="http://localhost:8000/v1",
            model_name="nemotron-parse",
        )

        assert in_process_stage.max_tokens == 8192
        assert server_stage.max_tokens == 8192
        assert server_stage._generation_config.max_tokens == 8192

    def test_infer_vllm_empty_outputs_produces_empty_string(self) -> None:
        """RequestOutput with no completions should yield '' rather than IndexError."""
        from nemo_curator.stages.interleaved.pdf.nemotron_parse.inference import NemotronParseInferenceStage

        stage = NemotronParseInferenceStage(backend="vllm")
        stage._sampling_params = SimpleNamespace()
        # Simulate a RequestOutput where the model returned no completions.
        empty_req_output = SimpleNamespace(prompt_token_ids=[1, 2], outputs=[])
        stage._llm = SimpleNamespace(generate=lambda _p, _s: [empty_req_output])

        texts, raw, retries = stage._infer_vllm([Image.new("RGB", (10, 10))])

        assert texts == [""]
        assert raw == [empty_req_output]
        assert retries == 0

    def test_infer_vllm_unreachable_loop_path_raises(self) -> None:
        from nemo_curator.stages.interleaved.pdf.nemotron_parse.inference import NemotronParseInferenceStage

        stage = NemotronParseInferenceStage(backend="vllm")
        stage._sampling_params = SimpleNamespace()
        stage._llm = SimpleNamespace(generate=lambda _p, _s: [])
        image = Image.new("RGB", (10, 10))

        with patch("builtins.range", return_value=()), pytest.raises(RuntimeError, match="unreachable"):
            stage._infer_vllm([image])


class TestNemotronParseInferenceServerStage:
    def test_rejects_non_positive_inference_batch_size(self) -> None:
        from nemo_curator.stages.interleaved.pdf.nemotron_parse.inference import (
            NemotronParseInferenceServerStage,
        )

        with pytest.raises(ValueError, match="inference_batch_size must be at least 1"):
            NemotronParseInferenceServerStage(
                endpoint="http://localhost:8000/v1",
                model_name="nemotron-parse",
                inference_batch_size=0,
            )

    def test_query_pages_runs_up_to_inference_batch_size_concurrently(self) -> None:
        from nemo_curator.stages.interleaved.pdf.nemotron_parse.inference import (
            NemotronParseInferenceServerStage,
        )

        active_requests = 0
        max_active_requests = 0
        two_requests_started = asyncio.Event()

        async def create_response(**create_kwargs: object) -> SimpleNamespace:
            nonlocal active_requests, max_active_requests
            active_requests += 1
            max_active_requests = max(max_active_requests, active_requests)
            if active_requests == 2:
                two_requests_started.set()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(two_requests_started.wait(), timeout=0.05)

            messages = create_kwargs["messages"]
            image_url = messages[0]["content"][1]["image_url"]["url"]  # type: ignore[index]
            page_text = base64.b64decode(image_url.split(",", 1)[1]).decode()
            active_requests -= 1
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=page_text), finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

        sdk_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_response)),
            close=AsyncMock(),
        )
        stage = NemotronParseInferenceServerStage(
            endpoint="http://localhost:8000/v1",
            model_name="nemotron-parse",
        )
        stage.inference_batch_size = 2

        with patch("nemo_curator.models.client.openai_client.AsyncOpenAI", return_value=sdk_client):
            results = asyncio.run(stage._query_pages([(f"page-{index}".encode(), "image/png") for index in range(3)]))

        assert max_active_requests == 2
        assert [result.text for result in results] == ["page-0", "page-1", "page-2"]

    def test_process_uses_openai_client_response_and_records_usage(self) -> None:
        import pandas as pd
        import pyarrow as pa

        from nemo_curator.stages.interleaved.pdf.nemotron_parse.inference import (
            NemotronParseInferenceServerStage,
        )
        from nemo_curator.tasks import InterleavedBatch

        task = InterleavedBatch(
            dataset_name="test",
            data=pa.Table.from_pandas(
                pd.DataFrame(
                    [
                        {
                            "sample_id": "s1",
                            "position": 0,
                            "modality": "page_image",
                            "content_type": "image/png",
                            "text_content": None,
                            "binary_content": b"png-bytes",
                            "source_ref": None,
                        }
                    ]
                )
            ),
        )
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="parsed page"), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=7),
        )
        stage = NemotronParseInferenceServerStage(
            endpoint="http://localhost:8000/v1",
            model_name="nemotron-parse",
            model_path="/models/NVIDIA-Nemotron-Parse-v1.1",
        )

        with patch(
            "nemo_curator.stages.interleaved.pdf.nemotron_parse.inference.AsyncOpenAIClient.query_model_response",
            new_callable=AsyncMock,
            return_value=response,
        ):
            result = stage.process(task)

        assert result is not None
        assert result.to_pandas().iloc[0]["text_content"] == "parsed page"
        assert result._metadata["inference_server_endpoint"] == "http://localhost:8000/v1"
        assert result._metadata["model_path"] == "/models/NVIDIA-Nemotron-Parse-v1.1"
        assert result._metadata["proc_size"] == [2048, 1664]
        assert stage._custom_metrics["total_prompt_tokens"] == 5.0
        assert stage._custom_metrics["total_output_tokens"] == 7.0
        assert stage._custom_metrics["num_request_errors"] == 0.0
        assert "vllm_inference_time" not in stage._custom_metrics


class TestNemotronParsePipelineFactory:
    def test_creates_one_http_inference_stage_with_configured_workers(self) -> None:
        from nemo_curator.stages.interleaved.pdf.nemotron_parse.inference import (
            NemotronParseInferenceServerStage,
            NemotronParseInferenceStage,
        )
        from tutorials.interleaved.nemotron_parse_pdf.main import (
            create_nemotron_parse_pdf_argparser,
            create_nemotron_parse_pdf_pipeline,
        )

        args = create_nemotron_parse_pdf_argparser().parse_args(
            [
                "--manifest",
                "manifest.jsonl",
                "--pdf-dir",
                "pdfs",
                "--output-dir",
                "output",
                "--max-tokens",
                "1234",
                "--inference-batch-size",
                "32",
            ]
        )

        pipeline = create_nemotron_parse_pdf_pipeline(
            args,
            inference_server_endpoint="http://localhost:8000/v1",
            inference_server_model_name="nemotron-parse",
            inference_server_num_workers=8,
        )
        pipeline.build()

        inference_stages = [
            stage
            for stage in pipeline.stages
            if isinstance(stage, (NemotronParseInferenceStage, NemotronParseInferenceServerStage))
        ]
        assert len(inference_stages) == 1
        assert isinstance(inference_stages[0], NemotronParseInferenceServerStage)
        assert inference_stages[0].endpoint == "http://localhost:8000/v1"
        assert inference_stages[0].model_name == "nemotron-parse"
        assert inference_stages[0].inference_batch_size == 32
        assert inference_stages[0].max_tokens == 1234
        assert inference_stages[0].num_workers() == 8
