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

"""Served parser integration with the actual optional model package."""

import asyncio
import io
import json
from dataclasses import asdict
from types import SimpleNamespace

import pyarrow as pa
import pytest
from PIL import Image

from nemo_curator.stages.interleaved.pdf.nemotron_parse import inference
from nemo_curator.stages.interleaved.pdf.nemotron_parse.postprocess import NemotronParsePostprocessStage
from nemo_curator.tasks import InterleavedBatch

TABLE = r"<x_0.1><y_0.1>\begin{tabular}{cc}A & B\\1 & 2\end{tabular}<x_0.9><y_0.9><class_Table>"


def task(colors: tuple[str, ...] = ("red",)):
    rows = []
    for index, color in enumerate(colors):
        image = Image.new("RGB", (220, 170), color)
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        rows.append(
            {
                "sample_id": "pdf",
                "pdf_name": "pdf.pdf",
                "url": "source",
                "position": index,
                "page_number": index,
                "modality": "page_image",
                "binary_content": stream.getvalue(),
                "content_type": "image/png",
                "text_content": "",
                "materialize_error": None,
            }
        )
    return InterleavedBatch(dataset_name="test", data=pa.Table.from_pylist(rows))


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("single", ["fullpage"]), ("double", ["fullpage", "crop"]), ("triple", ["fullpage", "detection", "crop"])],
)
def test_real_parser_modes(mode: str, expected: list[str]):
    package = pytest.importorskip("nemotron_parse.inference")
    calls = []

    def backend(image: Image.Image, *, prompt: str, stage: str, settings: object) -> dict:
        calls.append((stage, prompt, asdict(settings), image.size))
        return {"raw_out": TABLE, "finish_reason": "stop", "usage": {"completion_tokens": 7}}

    stage = inference.NemotronParseHTTPClientStage(endpoint="http://unused/v1", model_name="served", parse_mode=mode)
    stage.setup()
    stage._parser.close()
    stage._parser = package.DocumentParser(backend=backend)
    try:
        result = stage.process(task())
        document = json.loads(result.to_pyarrow().to_pylist()[0]["parse_document"])
        assert [entry[0] for entry in calls] == expected
        assert calls[0][1] == package.PROMPT
        assert calls[0][2] == asdict(package.GenerationSettings())
        if mode == "triple":
            assert calls[1][1] == package.DETECT_PROMPT
            assert calls[1][2] == asdict(package.GenerationSettings(max_tokens=4096))
        if mode != "single":
            assert calls[-1][2] == asdict(package.GenerationSettings(temperature=0.1, repetition_penalty=1.0))
        assert document["coordinate_frame"] == "content_normalized"
        projected = NemotronParsePostprocessStage().process(result).to_pyarrow().to_pylist()
        tables = [row for row in projected if row["modality"] == "table"]
        assert tables
        assert all(row["content_type"] == "text/html" for row in tables)
        assert "<table" in tables[0]["text_content"]
    finally:
        stage.teardown()
    assert stage._parser is None


def test_generation_failure_preserves_one_error_row_and_successful_sibling():
    package = pytest.importorskip("nemotron_parse.inference")

    def backend(image: Image.Image, **_kwargs) -> dict:
        if image.getpixel((0, 0))[0] > 0:
            return {"raw_out": "unfinished", "finish_reason": "length"}
        return {"raw_out": TABLE, "finish_reason": "stop"}

    stage = inference.NemotronParseHTTPClientStage(
        endpoint="http://unused/v1", model_name="served", parse_mode="single"
    )
    stage.setup()
    stage._parser = package.DocumentParser(backend=backend)
    try:
        result = NemotronParsePostprocessStage().process(stage.process(task(("red", "blue"))))
        rows = result.to_pyarrow().to_pylist()
        failed = [row for row in rows if row["materialize_error"]]
        assert len(failed) == 1
        assert failed[0]["page_number"] == 0
        assert json.loads(failed[0]["materialize_error"])["type"] == "GenerationError"
        assert any(row["page_number"] == 1 and row["modality"] == "table" for row in rows)
    finally:
        stage.teardown()


def test_real_parser_uses_bounded_existing_transport(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("nemotron_parse.inference")
    active = peak = 0
    configs = []
    closed = []

    class Client:
        def __init__(self, **kwargs):
            configs.append(kwargs)
            self.client = self

        async def query_model_response(self, **kwargs) -> SimpleNamespace:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            assert kwargs["messages"][0]["content"][0]["text"].endswith("<predict_text_in_pic>")
            assert kwargs["generation_config"].temperature == 0.2
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=TABLE), finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3),
            )

        async def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(inference, "AsyncOpenAIClient", Client)
    stage = inference.NemotronParseHTTPClientStage(
        endpoint="http://unused/v1",
        model_name="served",
        parse_mode="single",
        inference_batch_size=2,
        max_retries=7,
        request_timeout_s=91,
    )
    stage.setup()
    try:
        result = stage.process(task(("red", "blue", "green", "yellow")))
        assert result.to_pyarrow().num_rows == 4
        assert peak == 2
        assert configs[0]["max_retries"] == 7
        assert configs[0]["timeout"] == 91
        assert closed == [True]
    finally:
        stage.teardown()


def test_prior_render_failure_is_not_sent_to_model():
    package = pytest.importorskip("nemotron_parse.inference")
    rows = task().to_pyarrow().to_pylist()
    rows[0]["binary_content"] = None
    rows[0]["materialize_error"] = json.dumps({"stage": "render_pdf", "message": "bad page"})
    stage = inference.NemotronParseHTTPClientStage(
        endpoint="http://unused/v1", model_name="served", parse_mode="single"
    )
    stage.setup()
    stage._parser = package.DocumentParser(backend=lambda *_a, **_k: pytest.fail("must not infer a render failure"))
    try:
        batch = stage.process(InterleavedBatch(dataset_name="test", data=pa.Table.from_pylist(rows)))
        output = NemotronParsePostprocessStage().process(batch).to_pyarrow().to_pylist()
        assert len(output) == 1
        assert output[0]["page_number"] == 0
        assert output[0]["materialize_error"] == rows[0]["materialize_error"]
    finally:
        stage.teardown()


@pytest.mark.parametrize("failure_stage", ["fullpage", "detection", "crop"])
def test_each_generation_stage_failure_has_one_error_row(failure_stage: str):
    package = pytest.importorskip("nemotron_parse.inference")
    calls = []

    def backend(_image: Image.Image, *, stage: str, **_kwargs) -> dict:
        calls.append(stage)
        return {"raw_out": TABLE, "finish_reason": "length" if stage == failure_stage else "stop"}

    stage = inference.NemotronParseHTTPClientStage(
        endpoint="http://unused/v1", model_name="served", parse_mode="triple"
    )
    stage.setup()
    stage._parser = package.DocumentParser(backend=backend)
    try:
        batch = stage.process(task())
        rows = NemotronParsePostprocessStage().process(batch).to_pyarrow().to_pylist()
        assert len(rows) == 1
        assert json.loads(rows[0]["materialize_error"])["stage"] == failure_stage
        assert calls.count(failure_stage) == 1
    finally:
        stage.teardown()


def test_http_status_error_retains_failed_page_and_sibling():
    package = pytest.importorskip("nemotron_parse.inference")
    import httpx
    from openai import APIStatusError

    response = httpx.Response(400, request=httpx.Request("POST", "http://unused/v1/chat/completions"))

    def backend(image: Image.Image, **_kwargs) -> dict:
        if image.getpixel((0, 0))[0] > 0:
            message = "bad request"
            raise APIStatusError(message, response=response, body={"error": message})
        return {"raw_out": TABLE, "finish_reason": "stop"}

    stage = inference.NemotronParseHTTPClientStage(
        endpoint="http://unused/v1", model_name="served", parse_mode="single"
    )
    stage.setup()
    stage._parser = package.DocumentParser(backend=backend)
    try:
        rows = NemotronParsePostprocessStage().process(stage.process(task(("red", "blue")))).to_pyarrow().to_pylist()
        failed = [row for row in rows if row["materialize_error"]]
        assert len(failed) == 1
        assert failed[0]["page_number"] == 0
        assert json.loads(failed[0]["materialize_error"])["type"] == "APIStatusError"
        assert any(row["page_number"] == 1 and row["modality"] == "table" for row in rows)
    finally:
        stage.teardown()
