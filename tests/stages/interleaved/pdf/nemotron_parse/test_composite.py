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
