# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify effective contracts without a model, endpoint or Ray cluster."""

import asyncio
import copy
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from nemo_curator.stages.interleaved.pdf.nemotron_parse import extraction, inference
from nemo_curator.stages.interleaved.pdf.nemotron_parse.composite import NemotronParsePDFReader
from nemo_curator.stages.interleaved.pdf.nemotron_parse.page_budget import ExtractionConfigMismatchError
from nemo_curator.stages.interleaved.pdf.nemotron_parse.validation import PDFPageBudgetValidationStage
from nemo_curator.tasks import FileGroupTask
from tests.stages.interleaved.pdf.nemotron_parse._page_budget_helpers import TEST_EXTRACTION_CONFIG


@dataclass
class Settings:
    temperature: float = 0.2
    max_tokens: int = 8744


@pytest.fixture
def contracted_client(monkeypatch: pytest.MonkeyPatch) -> inference.NemotronParseHTTPClientStage:
    package = ModuleType("nemotron_parse.inference")
    package.PROMPT = "actual parser prompt"
    package.DocumentParser = lambda **_kwargs: SimpleNamespace(settings={"fullpage": Settings()})
    monkeypatch.setitem(sys.modules, "nemotron_parse", ModuleType("nemotron_parse"))
    monkeypatch.setitem(sys.modules, "nemotron_parse.inference", package)
    monkeypatch.setattr(inference.NemotronParseHTTPClientStage, "_require_served_model", lambda _self: None)
    monkeypatch.setattr(inference, "parser_package_identity", lambda: TEST_EXTRACTION_CONFIG["parser"])
    config = copy.deepcopy(TEST_EXTRACTION_CONFIG)
    generation = inference._server_generation_config({**asdict(Settings()), "skip_special_tokens": False})
    config["request"] = {
        "model": "served-alias",
        "prompt": package.PROMPT,
        "stage": "fullpage",
        "generation": asdict(generation),
    }
    return inference.NemotronParseHTTPClientStage(
        endpoint="http://unused/v1",
        model_name="served-alias",
        model_path="/different/immutable/snapshot",
        parse_mode="single",
        extraction_config=config,
    )


def test_contract_uses_served_alias_and_actual_parser_settings(
    contracted_client: inference.NemotronParseHTTPClientStage,
):
    contracted_client.setup()


@pytest.mark.parametrize("change", ["parser", "request", "transport"])
def test_changed_effective_settings_fail_before_requests(
    contracted_client: inference.NemotronParseHTTPClientStage, change: str
):
    if change == "parser":
        contracted_client.extraction_config["parser"]["version"] = "wrong"
    elif change == "request":
        contracted_client.extraction_config["request"]["generation"]["temperature"] = 0.9
    else:
        contracted_client.request_timeout_s = 1
    with pytest.raises(ExtractionConfigMismatchError):
        contracted_client.setup()


def test_contract_failure_is_fatal_not_an_ordinary_failed_page(
    monkeypatch: pytest.MonkeyPatch, contracted_client: inference.NemotronParseHTTPClientStage
):
    contracted_client.setup()

    def fail(_image: bytes) -> dict:
        msg = "runtime request differs"
        raise ExtractionConfigMismatchError(msg)

    monkeypatch.setattr(contracted_client, "_parse_image", fail)
    monkeypatch.setattr(inference, "AsyncOpenAIClient", lambda **_kwargs: SimpleNamespace())
    with pytest.raises(ExtractionConfigMismatchError, match="runtime request"):
        asyncio.run(contracted_client._parse_pages([{"binary_content": b"unused"}]))


def test_render_contract_checks_final_harness_settings_before_work():
    from nemo_curator.backends.ray_data.utils import is_actor_stage

    config = copy.deepcopy(TEST_EXTRACTION_CONFIG)
    config["render"]["timeout_s"] = 600
    reader = NemotronParsePDFReader(
        manifest_path="unused.jsonl",
        pages_per_task=64,
        parse_mode="single",
        inference_server_endpoint="http://unused/v1",
        page_budget_extraction_config=config,
    )
    renderer = reader._preprocessor
    assert not is_actor_stage(renderer)
    with pytest.raises(ExtractionConfigMismatchError, match="rendering"):
        renderer._check_extraction_settings()
    renderer._RENDER_TIMEOUT_S = 600
    renderer._check_extraction_settings()
    renderer._RENDER_TIMEOUT_S = 601
    with pytest.raises(ExtractionConfigMismatchError, match="rendering"):
        renderer.process(FileGroupTask(dataset_name="test", data=[]))


def test_parser_identity_changes_with_code_and_version(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    init = tmp_path / "__init__.py"
    init.write_text("# same parser\n")
    monkeypatch.setattr(extraction.importlib.util, "find_spec", lambda _name: SimpleNamespace(origin=str(init)))
    monkeypatch.setattr(extraction.importlib.metadata, "version", lambda _name: "1.0")
    original = extraction.parser_package_identity()
    init.write_text("# changed parser\n")
    assert extraction.parser_package_identity()["source_sha256"] != original["source_sha256"]
    init.write_text("# same parser\n")
    monkeypatch.setattr(extraction.importlib.metadata, "version", lambda _name: "1.1")
    assert extraction.parser_package_identity() == {**original, "version": "1.1"}


@pytest.mark.parametrize("value", ["", "not-a-version", "z" * 64])
def test_invalid_finalizer_contract_is_rejected_at_construction(value: str):
    with pytest.raises(ValueError, match="expected_output_version"):
        PDFPageBudgetValidationStage(manifest_path="unused.jsonl", expected_output_version=value)
