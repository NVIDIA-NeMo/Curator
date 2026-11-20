# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise early render admission through the same served Parse HTTP path."""

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from nemo_curator.stages.interleaved.pdf.nemotron_parse import _early_admission, inference
from nemo_curator.stages.interleaved.pdf.nemotron_parse.extraction import parser_package_identity
from nemo_curator.stages.interleaved.pdf.nemotron_parse.preprocess import PDFPreprocessStage
from tests.stages.interleaved.pdf.nemotron_parse._page_budget_helpers import TEST_EXTRACTION_CONFIG, partition
from tests.stages.interleaved.pdf.nemotron_parse.test_inference import http_endpoint  # noqa: F401 - pytest fixture


def _stages(
    endpoint: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[PDFPreprocessStage, inference.NemotronParseHTTPClientStage, list]:
    package = pytest.importorskip("nemotron_parse.inference")
    _, parents = partition(
        tmp_path,
        [{"file_name": f"{name}.pdf", "num_pages": 1} for name in ("red", "blue", "missing")],
    )
    renderer = PDFPreprocessStage(retain_failures=True, reuse_encoded_images=True)
    monkeypatch.setattr(renderer, "_get_pdf_bytes", lambda name, **_: None if name == "missing.pdf" else name.encode())
    parser = inference.NemotronParseHTTPClientStage(
        endpoint=endpoint,
        model_name="served",
        parse_mode="single",
        inference_batch_size=2,
        reuse_encoded_images=True,
    )
    defaults = package.DocumentParser(model="served", backend=parser._parse_backend)
    try:
        request = asdict(
            inference._server_generation_config(
                {
                    **asdict(defaults.settings["fullpage"]),
                    "skip_special_tokens": False,
                }
            )
        )
    finally:
        defaults.close()
    config = {
        **TEST_EXTRACTION_CONFIG,
        "parser": parser_package_identity(),
        "request": {"model": "served", "stage": "fullpage", "prompt": package.PROMPT, "generation": request},
    }
    renderer._extraction_config = config
    parser.extraction_config = config
    for task in parents:
        task._metadata["pdf_extraction_config"] = config
    return renderer, parser, parents


def test_completed_range_starts_http_before_sibling_and_retains_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    http_endpoint: SimpleNamespace,  # noqa: F811
) -> None:
    pytest.importorskip("pypdfium2")
    renderer, parser, parents = _stages(http_endpoint.endpoint, tmp_path, monkeypatch)
    (parent,) = parents
    for name in ("red", "blue"):
        with Image.new("RGB", (40, 25), name) as image:
            image.save(tmp_path / f"{name}.pdf")

    monkeypatch.setattr(
        renderer,
        "_get_pdf_bytes",
        lambda name, **_: None if name == "missing.pdf" else (tmp_path / name).read_bytes(),
    )
    standard_render = renderer._render_with_timeout
    overlap = False

    def render(data: bytes, name: str, *args: object, **kwargs: object) -> list[Image.Image | None]:
        if overlap and name == "blue.pdf":
            assert http_endpoint.entered.wait(3)
        return standard_render(data, name, *args, **kwargs)

    monkeypatch.setattr(renderer, "_render_with_timeout", render)
    parser.setup()
    try:
        baseline = parser.process(renderer.process(parent))
    finally:
        parser.teardown()
    requests = sorted(json.dumps(row, sort_keys=True) for row in http_endpoint.requests)
    http_endpoint.requests.clear()
    http_endpoint.entered.clear()

    overlap = True
    renderer.persistent_rendering = os.name == "posix"
    fused = _early_admission.EarlyAdmissionStage(renderer, parser)
    fused.setup()
    try:
        actual = fused.process(parent)
        if renderer.persistent_rendering:
            assert renderer._persistent_renderer.worker_pid is not None
    finally:
        fused.teardown()

    assert actual.to_pyarrow().equals(baseline.to_pyarrow())
    assert actual._metadata == baseline._metadata
    assert len(requests) == 2
    assert requests == sorted(json.dumps(row, sort_keys=True) for row in http_endpoint.requests)
    assert fused._custom_metrics["num_valid_pages"] == 2
    assert fused._custom_metrics["num_error_pages"] == 1
    assert fused._custom_metrics["image_reuse_count"] == 2


def test_cancelling_active_http_closes_worker_even_if_source_read_is_stuck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    http_endpoint: SimpleNamespace,  # noqa: F811
) -> None:
    renderer, parser, parents = _stages(http_endpoint.endpoint, tmp_path, monkeypatch)
    (parent,) = parents
    allow_read = threading.Event()
    waiting = threading.Event()
    http_endpoint.release = threading.Event()

    def render(_data: bytes, name: str, *_args: object, **_kwargs: object) -> list[Image.Image]:
        if name == "blue.pdf":
            waiting.set()
            assert allow_read.wait(10)
        return [Image.new("RGB", (40, 25), Path(name).stem)]

    monkeypatch.setattr(renderer, "_render_with_timeout", render)
    monkeypatch.setattr(_early_admission, "_RENDER_JOIN_SECONDS", 0.01)
    fused = _early_admission.EarlyAdmissionStage(renderer, parser)
    fused.setup()
    with ThreadPoolExecutor(max_workers=2) as pool:
        processing = pool.submit(fused.process, parent)
        try:
            assert http_endpoint.entered.wait(3)
            assert waiting.wait(3)
            stopping = pool.submit(fused.teardown)
            with pytest.raises(RuntimeError, match="worker cancelled"):
                processing.result(timeout=3)
            stopping.result(timeout=3)
            assert len(http_endpoint.requests) == 1
            with pytest.raises(RuntimeError, match="worker is closed"):
                fused.process(parent)
        finally:
            allow_read.set()
            http_endpoint.release.set()
