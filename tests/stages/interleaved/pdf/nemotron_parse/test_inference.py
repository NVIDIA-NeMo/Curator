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
import base64
import contextlib
import io
import json
import threading
import urllib.error
from collections.abc import Callable
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.stages.interleaved.io import InterleavedParquetWriterStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse import inference
from nemo_curator.stages.interleaved.pdf.nemotron_parse._image_transport import PNG_METADATA_KEY, png_batch_provenance
from nemo_curator.stages.interleaved.pdf.nemotron_parse.postprocess import NemotronParsePostprocessStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.preprocess import PDFPreprocessStage
from nemo_curator.tasks import FileGroupTask, InterleavedBatch

TABLE = r"<x_0.1><y_0.1>\begin{tabular}{cc}A & B\\1 & 2\end{tabular}<x_0.9><y_0.9><class_Table>"
PARSED_TABLE = {"raw_out": TABLE, "blocks": [{"class": "Table", "bbox": [0.1, 0.1, 0.9, 0.9], "content": "<table/>"}]}


def test_parse_mode_does_not_rebind_existing_positional_client_options():
    stage = inference.NemotronParseHTTPClientStage("http://unused/v1", "served", None, True, "custom prompt", 120.0)
    assert stage.text_in_pic
    assert stage.task_prompt == "custom prompt"
    assert stage.request_timeout_s == 120.0
    assert stage.parse_mode is None


@pytest.fixture(autouse=True)
def served_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        inference.urllib.request, "urlopen", lambda *_a, **_k: io.BytesIO(b'{"data": [{"id": "served"}]}')
    )


@dataclass(frozen=True)
class Settings:
    temperature: float = 0.0
    top_k: int = 1


class StubParser:
    """Replaces the optional package where a test checks only Curator's transport and rows."""

    def __init__(self, parse: Callable[..., dict]) -> None:
        self.parse = parse

    def close(self) -> None:
        pass


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


def test_full_parser_pool_does_not_starve_client_threads(monkeypatch: pytest.MonkeyPatch):
    class Client:
        def __init__(self, **_kwargs):
            self.client = self

        async def query_model_response(self, **_kwargs) -> SimpleNamespace:
            # The OpenAI SDK runs part of each request on the loop's default executor.
            await asyncio.to_thread(lambda: None)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=TABLE), finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3),
            )

        async def close(self) -> None:
            pass

    monkeypatch.setattr(inference, "AsyncOpenAIClient", Client)
    stage = inference.NemotronParseHTTPClientStage(
        endpoint="http://unused/v1", model_name="served", parse_mode="single", inference_batch_size=32
    )
    stage._parser = StubParser(
        lambda image, **_: {
            **PARSED_TABLE,
            **stage._parse_backend(image, prompt="p", stage="fullpage", settings=Settings()),
        }
    )
    try:
        pages = task(("red",) * 32).to_pyarrow().to_pylist()
        rows = asyncio.run(asyncio.wait_for(stage._parse_pages(pages), timeout=30))
    finally:
        stage.teardown()
    assert len(rows) == 32
    assert not any(row["materialize_error"] for row in rows)


def test_prior_render_failure_is_not_sent_to_model():
    rows = task().to_pyarrow().to_pylist()
    rows[0]["binary_content"] = None
    rows[0]["materialize_error"] = json.dumps({"stage": "render_pdf", "message": "bad page"})
    stage = inference.NemotronParseHTTPClientStage(
        endpoint="http://unused/v1", model_name="served", parse_mode="single"
    )
    stage._parser = StubParser(lambda *_a, **_k: pytest.fail("must not infer a render failure"))
    try:
        batch = stage.process(InterleavedBatch(dataset_name="test", data=pa.Table.from_pylist(rows)))
        metadata, failed = NemotronParsePostprocessStage().process(batch).to_pyarrow().to_pylist()
        assert (metadata["position"], metadata["modality"], metadata["materialize_error"]) == (-1, "metadata", None)
        assert (failed["position"], failed["modality"], failed["page_number"]) == (0, "text", 0)
        assert failed["materialize_error"] == rows[0]["materialize_error"]
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
        _, failed = NemotronParsePostprocessStage().process(batch).to_pyarrow().to_pylist()
        assert json.loads(failed["materialize_error"])["stage"] == failure_stage
        assert calls.count(failure_stage) == 1
    finally:
        stage.teardown()


def test_http_status_error_retains_failed_page_and_sibling():
    import httpx
    from openai import APIStatusError

    response = httpx.Response(400, request=httpx.Request("POST", "http://unused/v1/chat/completions"))

    def parse(image: Image.Image, **_kwargs) -> dict:
        if image.getpixel((0, 0))[0] > 0:
            message = "bad request"
            raise APIStatusError(message, response=response, body={"error": message})
        return PARSED_TABLE

    stage = inference.NemotronParseHTTPClientStage(
        endpoint="http://unused/v1", model_name="served", parse_mode="single"
    )
    stage._parser = StubParser(parse)
    try:
        rows = NemotronParsePostprocessStage().process(stage.process(task(("red", "blue")))).to_pyarrow().to_pylist()
        failed = [row for row in rows if row["materialize_error"]]
        assert len(failed) == 1
        assert failed[0]["page_number"] == 0
        assert json.loads(failed[0]["materialize_error"])["type"] == "APIStatusError"
        assert any(row["page_number"] == 1 and row["modality"] == "table" for row in rows)
    finally:
        stage.teardown()


def test_endpoint_must_list_served_model(monkeypatch: pytest.MonkeyPatch):
    calls = []

    def refuse(url: str, **_kwargs) -> None:
        calls.append(url)
        reason = "refused"
        raise urllib.error.URLError(reason)

    stage = inference.NemotronParseHTTPClientStage(
        endpoint="http://unused/v1/", model_name="served", parse_mode="single", max_retries=1, retry_base_delay_s=0
    )
    stage._require_served_model()
    monkeypatch.setattr(
        inference.urllib.request, "urlopen", lambda *_a, **_k: io.BytesIO(b'{"data": [{"id": "other"}]}')
    )
    with pytest.raises(RuntimeError, match=r"does not serve 'served'; it serves \['other'\]"):
        stage._require_served_model()
    monkeypatch.setattr(inference.urllib.request, "urlopen", refuse)
    with pytest.raises(RuntimeError, match="Cannot list models at http://unused/v1/models"):
        stage._require_served_model()
    assert calls == ["http://unused/v1/models"] * 2


def test_malformed_output_keeps_raw_response_and_physical_error_row():
    stage = inference.NemotronParseHTTPClientStage(
        endpoint="http://unused/v1", model_name="served", parse_mode="single"
    )
    stage._parser = StubParser(lambda *_a, **_k: {"raw_out": "<x_0.1>unfinished map labels", "blocks": []})
    try:
        batch = stage.process(task())
        rows = NemotronParsePostprocessStage().process(batch).to_pyarrow().to_pylist()
        failed = [row for row in rows if row["materialize_error"]]
        assert len(failed) == 1
        assert failed[0]["page_number"] == 0
        error = json.loads(failed[0]["materialize_error"])
        assert error["type"] == "NoDecodedBlocks"
        assert error["raw_response"]["raw_out"] == "<x_0.1>unfinished map labels"
        assert stage._custom_metrics["num_valid_pages"] == 0
        assert stage._custom_metrics["num_error_pages"] == 1
    finally:
        stage.teardown()


@pytest.mark.parametrize(("mode", "call_count"), [("single", 1), ("double", 2), ("triple", 3)])
def test_served_metrics_count_every_stage_response(monkeypatch: pytest.MonkeyPatch, mode: str, call_count: int):
    class Client:
        def __init__(self, **_kwargs):
            pass

        async def query_model_response(self, **_kwargs) -> SimpleNamespace:
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=TABLE), finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3),
            )

    monkeypatch.setattr(inference, "AsyncOpenAIClient", Client)
    stage = inference.NemotronParseHTTPClientStage(endpoint="http://unused/v1", model_name="served", parse_mode=mode)

    def parse(image: Image.Image, **_kwargs) -> dict:
        for name in ("fullpage", "detection", "crop")[:call_count]:
            stage._parse_backend(image, prompt="p", stage=name, settings=Settings())
        return PARSED_TABLE

    stage._parser = StubParser(parse)
    try:
        result = stage.process(task())
        assert "_parse_metrics" not in result.to_pyarrow().column_names
        assert stage._custom_metrics["num_valid_pages"] == 1
        assert stage._custom_metrics["num_requests"] == call_count
        assert stage._custom_metrics["total_prompt_tokens"] == 2 * call_count
        assert stage._custom_metrics["total_output_tokens"] == 3 * call_count
    finally:
        stage.teardown()


def test_failed_page_keeps_prior_request_usage_and_raw_response(monkeypatch: pytest.MonkeyPatch):
    calls = []

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def query_model_response(self, **_kwargs) -> SimpleNamespace:
            calls.append(True)
            if len(calls) == 2:
                msg = "crop connection failed"
                raise RuntimeError(msg)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=TABLE), finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3),
            )

    monkeypatch.setattr(inference, "AsyncOpenAIClient", Client)
    stage = inference.NemotronParseHTTPClientStage(
        endpoint="http://unused/v1", model_name="served", parse_mode="double"
    )

    def parse(image: Image.Image, **_kwargs) -> dict:
        for name in ("fullpage", "crop"):
            stage._parse_backend(image, prompt="p", stage=name, settings=Settings())
        return PARSED_TABLE

    stage._parser = StubParser(parse)
    try:
        result = stage.process(task())
        error = json.loads(result.to_pyarrow().to_pylist()[0]["materialize_error"])
        assert error["requests"][0]["raw_out"] == TABLE
        assert stage._custom_metrics["num_requests"] == 2
        assert stage._custom_metrics["num_failed_requests"] == 1
        assert stage._custom_metrics["total_output_tokens"] == 3
        assert stage._custom_metrics["num_valid_pages"] == 0
    finally:
        stage.teardown()


def test_actual_parser_malformed_layout_retains_failure_and_successful_sibling():
    package = pytest.importorskip("nemotron_parse.inference")
    malformed = "<x_0.1><y_0.2>Unclosed map labels"

    def backend(image: Image.Image, **_kwargs) -> dict:
        raw = malformed if image.getpixel((0, 0))[0] > 0 else TABLE
        return {"raw_out": raw, "finish_reason": "stop"}

    stage = inference.NemotronParseHTTPClientStage(
        endpoint="http://unused/v1", model_name="served", parse_mode="single"
    )
    stage._parser = package.DocumentParser(backend=backend)
    try:
        result = NemotronParsePostprocessStage().process(stage.process(task(("red", "blue"))))
        rows = result.to_pyarrow().to_pylist()
        failed = [row for row in rows if row["materialize_error"]]
        assert len(failed) == 1
        assert failed[0]["page_number"] == 0
        error = json.loads(failed[0]["materialize_error"])
        assert error["type"] == "NoDecodedBlocks"
        assert error["raw_response"]["stages"]["fullpage"]["raw_out"] == malformed
        assert any(row["page_number"] == 1 and row["modality"] == "table" for row in rows)
    finally:
        stage.teardown()


@pytest.fixture
def http_endpoint():
    """Exercise real SDK sockets, response decoding, retries and keep-alive."""
    state = SimpleNamespace(requests=[], peers=[], statuses=[], outputs={}, entered=threading.Event(), release=None)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            payload = json.dumps({"data": [{"id": "served", "object": "model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.requests.append(body)
            state.peers.append(self.client_address)
            state.entered.set()
            if state.release is not None:
                state.release.wait(timeout=5)
            status = state.statuses.pop(0) if state.statuses else 200
            image_url = body["messages"][0]["content"][1]["image_url"]["url"]
            output, finish_reason = state.outputs.get(image_url, (TABLE, "stop"))
            response = (
                {
                    "id": "test-completion",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "served",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": output},
                            "finish_reason": finish_reason,
                        }
                    ],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
                }
                if status == 200
                else {
                    "error": {
                        "message": "rate limited" if status == 429 else "bad request",
                        "type": "rate_limit" if status == 429 else "invalid_request",
                        "code": "rate_limit" if status == 429 else "invalid_request",
                    }
                }
            )
            payload = json.dumps(response).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Retry-After", "0.01")
            self.end_headers()
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(payload)

        def log_message(self, *_args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    state.endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        yield state
    finally:
        if state.release is not None:
            state.release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _http_stage(endpoint: str, **kwargs) -> inference.NemotronParseHTTPClientStage:
    stage = inference.NemotronParseHTTPClientStage(
        endpoint=endpoint,
        model_name="served",
        parse_mode="single",
        inference_batch_size=2,
        request_timeout_s=2,
        **kwargs,
    )
    stage._parser = StubParser(
        lambda image, **_: {
            **PARSED_TABLE,
            **stage._parse_backend(image, prompt="p", stage="fullpage", settings=Settings()),
        }
    )
    return stage


def test_reused_producer_pngs_preserve_actual_sdk_parser_and_parquet(
    http_endpoint: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    pytest.importorskip("nemotron_parse.inference")
    images = {
        "alpha.pdf": [Image.new("RGB", (37, 29), color) for color in ("red", "blue")],
        "beta.pdf": [Image.frombytes("RGB", (31, 23), bytes((i * 17) % 256 for i in range(31 * 23 * 3)))],
    }
    original_save = Image.Image.save
    outputs, requests, encode_counts = [], [], []
    for enabled in (False, True):
        producer = PDFPreprocessStage(retain_failures=True, reuse_encoded_images=enabled)
        monkeypatch.setattr(producer, "_get_pdf_bytes", lambda name, *_args, **_kwargs: name.encode())
        monkeypatch.setattr(producer, "_render_with_timeout", lambda _data, name, *_args: images[name])
        batch = producer.process(
            FileGroupTask(
                dataset_name="test",
                data=[json.dumps({"file_name": name}) for name in images],
                _metadata={"source_files": list(images)},
            )
        )
        rows = batch.to_pyarrow().to_pylist()
        for row, response in zip(
            rows, (("unfinished", "length"), (TABLE, "stop"), ("malformed", "stop")), strict=True
        ):
            image_url = "data:image/png;base64," + base64.b64encode(row["binary_content"]).decode("ascii")
            http_endpoint.outputs[image_url] = response
        client = inference.NemotronParseHTTPClientStage(
            endpoint=http_endpoint.endpoint,
            model_name="served",
            parse_mode="single",
            inference_batch_size=3,
            reuse_encoded_images=enabled,
        )
        client.setup()
        encoded = []

        def save(image: Image.Image, *args, _encoded: list[tuple[int, int]] = encoded, **kwargs) -> None:
            _encoded.append(image.size)
            return original_save(image, *args, **kwargs)

        start = len(http_endpoint.requests)
        try:
            with monkeypatch.context() as patched:
                patched.setattr(Image.Image, "save", save)
                inferred = client.process(batch)
            encode_counts.append(len(encoded))
            projected = NemotronParsePostprocessStage().process(inferred)
            table = projected.to_pyarrow()
            writer = InterleavedParquetWriterStage(
                path=str(tmp_path / str(enabled)),
                materialize_on_write=False,
                on_materialize_error="warn",
                schema=table.schema,
            )
            written = writer.process(projected)
            outputs.append(pq.read_table(written.data[0]))
            requests.append(
                sorted(http_endpoint.requests[start:], key=lambda request: json.dumps(request, sort_keys=True))
            )
        finally:
            client.teardown()
        assert inference._PARSE_ENCODED_IMAGE.get() is None
    assert encode_counts == [3, 0]
    assert requests[0] == requests[1]
    assert outputs[0].equals(outputs[1])
    failed = [(row["pdf_name"], row["page_number"]) for row in outputs[1].to_pylist() if row["materialize_error"]]
    assert sorted(failed) == [("alpha.pdf", 0), ("beta.pdf", 0)]
    assert any(row["pdf_name"] == "alpha.pdf" and row["page_number"] == 1 for row in outputs[1].to_pylist())


@pytest.mark.parametrize(
    "case", ["eligible", "missing", "unknown", "stale", "rgba", "metadata", "mutated", "in_place", "crop"]
)
def test_encoded_image_reuse_falls_back_without_changing_request_pixels(  # noqa: C901, PLR0915
    http_endpoint: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, case: str
):
    batch = task()
    rows = batch.to_pyarrow().to_pylist()
    if case == "rgba":
        image = Image.new("RGBA", (220, 170), (255, 0, 0, 127))
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        rows[0]["binary_content"] = stream.getvalue()
    elif case == "metadata":
        from PIL.PngImagePlugin import PngInfo

        info = PngInfo()
        info.add_text("source", "external")
        stream = io.BytesIO()
        Image.new("RGB", (220, 170), "red").save(stream, format="PNG", pnginfo=info)
        rows[0]["binary_content"] = stream.getvalue()
    metadata = {PNG_METADATA_KEY: png_batch_provenance(row["binary_content"] for row in rows)}
    if case == "missing":
        metadata.clear()
    elif case == "unknown":
        metadata[PNG_METADATA_KEY]["pillow"] = "unknown"
    elif case == "stale":
        stream = io.BytesIO()
        Image.new("RGB", (220, 170), "red").save(stream, format="PNG", compress_level=1)
        assert stream.getvalue() != rows[0]["binary_content"]
        rows[0]["binary_content"] = stream.getvalue()
    batch = InterleavedBatch(dataset_name="test", data=pa.Table.from_pylist(rows), _metadata=metadata)
    client = _http_stage(http_endpoint.endpoint, reuse_encoded_images=True)
    expected = {}

    def parse(image: Image.Image, **_kwargs) -> dict:
        if case == "in_place":
            image.putpixel((0, 0), (0, 255, 0))
        converted = image.convert("RGB")
        if case == "mutated":
            converted.putpixel((0, 0), (0, 255, 0))
        expected["pixels"] = converted.tobytes()
        client._parse_backend(
            converted, prompt="p", stage="crop" if case == "crop" else "fullpage", settings=Settings()
        )
        return PARSED_TABLE

    client._parser = StubParser(parse)
    original_save = Image.Image.save
    encodes = []

    def save(image: Image.Image, *args, **kwargs) -> None:
        encodes.append(True)
        return original_save(image, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", save)
    try:
        result = client.process(batch)
        assert not result.to_pyarrow().to_pylist()[0]["materialize_error"]
    finally:
        client.teardown()
    assert len(encodes) == (0 if case == "eligible" else 1)
    assert client._custom_metrics["image_reuse_count"] == (1 if case == "eligible" else 0)
    assert client._custom_metrics["image_encode_count"] == len(encodes)
    assert client._custom_metrics["image_reuse_fallback_count"] == len(encodes)
    data_url = http_endpoint.requests[-1]["messages"][0]["content"][1]["image_url"]["url"]
    with Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))) as image:
        assert image.tobytes() == expected["pixels"]
    if case == "eligible":
        assert base64.b64decode(data_url.split(",", 1)[1]) == rows[0]["binary_content"]


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
def test_encoded_image_context_resets_and_input_stream_closes_after_parser_failure(failure: type[BaseException]):
    client = inference.NemotronParseHTTPClientStage(
        endpoint="http://unused/v1", model_name="served", parse_mode="single", reuse_encoded_images=True
    )
    seen = []

    def parse(image: Image.Image, **_kwargs) -> dict:
        seen.append(image)
        assert inference._PARSE_ENCODED_IMAGE.get() is not None
        msg = "parser failed"
        raise failure(msg)

    client._parser = StubParser(parse)
    assert inference._PARSE_ENCODED_IMAGE.get() is None
    with pytest.raises(failure, match="parser failed"):
        client._parse_image(task().to_pyarrow().to_pylist()[0]["binary_content"], True)
    assert inference._PARSE_ENCODED_IMAGE.get() is None
    assert seen[0].fp is None
    assert seen[0]._fp.closed


def test_encoded_image_reuse_does_not_leak_to_unmarked_envelopes(
    http_endpoint: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
):
    marked = task(("red", "blue"))
    marked._metadata[PNG_METADATA_KEY] = png_batch_provenance(marked.to_pyarrow().column("binary_content").to_pylist())
    client = _http_stage(http_endpoint.endpoint, reuse_encoded_images=True)
    original_save = Image.Image.save
    encoded = []

    def save(image: Image.Image, *args, **kwargs) -> None:
        encoded.append(True)
        return original_save(image, *args, **kwargs)

    unmarked = task(("red", "blue"))
    monkeypatch.setattr(Image.Image, "save", save)
    try:
        assert client.process(marked).to_pyarrow().num_rows == 2
        assert encoded == []
        assert client.process(unmarked).to_pyarrow().num_rows == 2
        assert len(encoded) == 2
        assert client.process(marked).to_pyarrow().num_rows == 2
        assert len(encoded) == 2
    finally:
        client.teardown()


@pytest.mark.parametrize("status", [429, 400])
def test_encoded_image_reuse_preserves_actual_sdk_retry_and_failure(http_endpoint: SimpleNamespace, status: int):
    batch = task()
    payload = batch.to_pyarrow().column("binary_content").to_pylist()
    batch._metadata[PNG_METADATA_KEY] = png_batch_provenance(payload)
    http_endpoint.statuses = [status, 200]
    client = _http_stage(
        http_endpoint.endpoint,
        reuse_encoded_images=True,
        max_retries=1,
        retry_base_delay_s=0,
    )
    try:
        rows = client.process(batch).to_pyarrow().to_pylist()
        assert bool(rows[0]["materialize_error"]) is (status == 400)
        if status == 429:
            assert len(http_endpoint.requests) == 2
            assert http_endpoint.requests[0] == http_endpoint.requests[1]
        else:
            assert json.loads(rows[0]["materialize_error"])["type"] == "BadRequestError"
        assert client._custom_metrics["image_reuse_count"] == 1
        assert client._custom_metrics["image_encode_count"] == 0
    finally:
        client.teardown()


@pytest.mark.usefixtures("shared_ray_client")
def test_real_ray_data_preserves_png_reuse_and_page_identity(http_endpoint: SimpleNamespace, tmp_path: Path):
    pytest.importorskip("nemotron_parse.inference")
    document = io.BytesIO()
    images = [Image.new("RGB", (37, 29), color) for color in ("red", "blue")]
    images[0].save(document, format="PDF", save_all=True, append_images=images[1:])
    (tmp_path / "two.pdf").write_bytes(document.getvalue())
    outputs, requests = [], []
    for enabled in (False, True):
        start = len(http_endpoint.requests)
        results = RayDataExecutor().execute(
            [
                PDFPreprocessStage(pdf_dir=str(tmp_path), retain_failures=True, reuse_encoded_images=enabled),
                inference.NemotronParseHTTPClientStage(
                    endpoint=http_endpoint.endpoint,
                    model_name="served",
                    parse_mode="single",
                    inference_batch_size=2,
                    reuse_encoded_images=enabled,
                ).with_(num_workers=1),
                NemotronParsePostprocessStage(),
                InterleavedParquetWriterStage(
                    path=str(tmp_path / str(enabled)),
                    materialize_on_write=False,
                    on_materialize_error="warn",
                ),
            ],
            initial_tasks=[
                FileGroupTask(
                    dataset_name="test",
                    data=[json.dumps({"file_name": "two.pdf", "selected_page_indices": [1, 0]})],
                    _metadata={"source_files": ["two.pdf"]},
                )
            ],
        )
        assert len(results) == 1
        table = pq.read_table(results[0].data[0])
        outputs.append(table)
        requests.append(
            sorted(http_endpoint.requests[start:], key=lambda request: json.dumps(request, sort_keys=True))
        )
        rows = table.to_pylist()
        assert {row["page_number"] for row in rows if row["modality"] != "document"} >= {0, 1}
        assert not any(row["materialize_error"] for row in rows)
        metrics = results[0]._stage_perf
        # Real transport must preserve the producer marker, rather than quietly encoding again.
        assert any(entry.custom_metrics.get("image_reuse_count", 0) == 2 for entry in metrics) is enabled
    assert len(requests[0]) == len(requests[1]) == 2
    assert requests[0] == requests[1]
    assert outputs[0].equals(outputs[1])


def test_extraction_config_does_not_rebind_legacy_positional_client_arguments():
    from nemo_curator.stages.resources import Resources

    stage = inference.NemotronParseHTTPClientStage(
        "http://unused/v1",
        "served",
        None,
        None,
        "union",
        False,
        None,
        300.0,
        3,
        1.0,
        4,
        8192,
        (2048, 1664),
        "legacy",
        Resources(cpus=2),
    )
    assert stage.name == "legacy"
    assert stage.resources.cpus == 2
    assert stage.extraction_config is None
