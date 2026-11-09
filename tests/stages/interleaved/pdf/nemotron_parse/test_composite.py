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

"""Validate opt-in served reader configuration."""

import pytest

from nemo_curator.stages.interleaved.pdf.nemotron_parse.composite import NemotronParsePDFReader
from nemo_curator.stages.interleaved.pdf.nemotron_parse.inference import DEFAULT_MODEL_PATH
from tests.stages.interleaved.pdf.nemotron_parse._page_budget_helpers import TEST_EXTRACTION_CONFIG


def test_parse_mode_requires_endpoint():
    with pytest.raises(ValueError, match="requires inference_server_endpoint"):
        NemotronParsePDFReader(manifest_path="input.jsonl", parse_mode="single")


def test_served_modes_keep_default_model_and_client_workers():
    reader = NemotronParsePDFReader(
        manifest_path="input.jsonl",
        parse_mode="triple",
        triple_regions="detector",
        inference_server_endpoint="http://unused/v1",
        inference_server_client_num_workers=8,
    )
    assert reader.model_path == DEFAULT_MODEL_PATH
    assert reader._inference.model_name == DEFAULT_MODEL_PATH
    assert reader._inference.parse_mode == "triple"
    assert reader._inference.triple_regions == "detector"
    assert reader._preprocessor.retain_failures
    assert reader._inference.ray_stage_spec() == {"is_actor_stage": True}
    legacy = NemotronParsePDFReader(manifest_path="input.jsonl", inference_server_endpoint="http://unused/v1")
    assert not legacy._preprocessor.retain_failures
    assert legacy._inference.ray_stage_spec() == {"is_actor_stage": False}


def test_encoded_image_reuse_wires_both_stages_without_changing_execution_controls():
    settings = {
        "manifest_path": "input.jsonl",
        "parse_mode": "single",
        "inference_server_endpoint": "http://unused/v1",
    }
    control = NemotronParsePDFReader(**settings)
    candidate = NemotronParsePDFReader(**settings, inference_server_reuse_encoded_images=True)
    assert not control._preprocessor.reuse_encoded_images
    assert not control._inference.reuse_encoded_images
    assert candidate._preprocessor.reuse_encoded_images
    assert candidate._inference.reuse_encoded_images
    assert candidate._inference.inference_batch_size == control._inference.inference_batch_size
    assert candidate._inference.num_workers() == control._inference.num_workers()


@pytest.mark.parametrize("mode", [None, "double", "triple"])
def test_encoded_image_reuse_requires_served_single_mode(mode: str | None):
    with pytest.raises(ValueError, match=r"requires served Parse 3\.5 single mode"):
        NemotronParsePDFReader(
            manifest_path="input.jsonl",
            parse_mode=mode,
            inference_server_endpoint="http://unused/v1",
            inference_server_reuse_encoded_images=True,
        )


def test_encoded_image_reuse_requires_endpoint_and_boolean():
    with pytest.raises(ValueError, match=r"requires served Parse 3\.5 single mode"):
        NemotronParsePDFReader(
            manifest_path="input.jsonl", parse_mode="single", inference_server_reuse_encoded_images=True
        )
    with pytest.raises(TypeError, match="must be a boolean"):
        NemotronParsePDFReader(manifest_path="input.jsonl", inference_server_reuse_encoded_images=1)


@pytest.mark.parametrize("mode", [None, "double", "triple"])
def test_page_budget_rejects_untested_parser_modes(mode: str | None):
    with pytest.raises(ValueError, match="single mode"):
        NemotronParsePDFReader(
            manifest_path="input.jsonl",
            parse_mode=mode,
            pages_per_task=64,
            inference_server_endpoint="http://unused/v1",
        )


def test_page_budget_keeps_http_capacity_and_adds_nonresumable_merge():
    reader = NemotronParsePDFReader(
        manifest_path="input.jsonl",
        parse_mode="single",
        pages_per_task=64,
        page_budget_extraction_config=TEST_EXTRACTION_CONFIG,
        inference_server_endpoint="http://unused/v1",
        inference_batch_size=64,
        inference_server_client_num_workers=36,
    )
    assert reader._inference.inference_batch_size == 64
    assert reader._inference.num_workers() == 36
    assert reader._partitioner.pages_per_task == 64
    assert reader._postprocessor.ray_stage_spec() == {"is_fanout_stage": True}
    assert not reader.decompose()[-1].is_resumable


def test_legacy_postprocess_graph_does_not_gain_a_fanout_boundary():
    reader = NemotronParsePDFReader(manifest_path="input.jsonl")
    assert reader._postprocessor.ray_stage_spec() == {"is_fanout_stage": False}


def test_ignored_legacy_controls_do_not_change_page_mode_output_version():
    settings = {
        "manifest_path": "input.jsonl",
        "parse_mode": "single",
        "pages_per_task": 64,
        "page_budget_extraction_config": TEST_EXTRACTION_CONFIG,
        "inference_server_endpoint": "http://unused/v1",
    }
    first = NemotronParsePDFReader(**settings, pdfs_per_task=1, max_pages=1)
    second = NemotronParsePDFReader(**settings, pdfs_per_task=100, max_pages=500)
    assert first._partitioner._parser_config == second._partitioner._parser_config


def test_page_budget_rejects_other_executors_before_processing():
    from nemo_curator.pipeline import Pipeline

    pipeline = Pipeline(name="wrong_backend")
    pipeline.add_stage(
        NemotronParsePDFReader(
            manifest_path="not-opened.jsonl",
            parse_mode="single",
            pages_per_task=64,
            page_budget_extraction_config=TEST_EXTRACTION_CONFIG,
            inference_server_endpoint="http://unused/v1",
        )
    )
    with pytest.raises(ValueError, match="RayDataExecutor"):
        pipeline.run(executor=object())


def test_page_budget_does_not_rebind_legacy_positional_reader_arguments():
    reader = NemotronParsePDFReader("input.jsonl", None, "pdfs", None, "legacy-model", "vllm", 8, 17)
    assert reader.max_pdfs == reader._partitioner.max_pdfs == 17
    assert reader.pdfs_per_task == 8
    assert reader.pages_per_task is None
