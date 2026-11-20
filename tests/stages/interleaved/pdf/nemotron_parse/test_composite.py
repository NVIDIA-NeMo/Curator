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


def test_served_options_do_not_rebind_existing_positional_endpoint():
    reader = NemotronParsePDFReader(
        "input.jsonl",
        None,
        "pdfs",
        None,
        "legacy-model",
        "vllm",
        10,
        None,
        300,
        50,
        4,
        64,
        8192,
        False,
        False,
        10,
        "pdf_dataset",
        "file_name",
        "cc_pdf_file_names",
        "url",
        "http://unused/v1",
        "served-model",
        8,
    )
    assert reader.inference_server_endpoint == "http://unused/v1"
    assert reader._inference.model_name == "served-model"
    assert reader._inference.num_workers() == 8
    assert reader.parse_mode is None


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
