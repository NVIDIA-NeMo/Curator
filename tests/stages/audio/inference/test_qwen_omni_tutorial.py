# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import pytest
from omegaconf import OmegaConf

from nemo_curator.stages.audio.inference.qwen_omni import InferenceQwenOmniStage
from nemo_curator.stages.audio.io.nemo_tarred_reader import NemoTarredAudioReader, NemoTarShardReaderStage
from nemo_curator.stages.audio.io.sharded_manifest_writer import ShardedManifestWriterStage
from tutorials.audio.qwen_omni_inprocess.main import (
    _prefetch_models,
    _safe_config_yaml,
    build_granary_v2_pipeline,
)


def test_safe_config_yaml_redacts_hf_token_but_keeps_token_counts() -> None:
    cfg = OmegaConf.create({
        "hf_token": "hf_secret_value",
        "max_output_tokens": 256,
        "nested": {"api_key": "secret-key"},
    })

    rendered = _safe_config_yaml(cfg)

    assert "hf_secret_value" not in rendered
    assert "secret-key" not in rendered
    assert "hf_token: <redacted>" in rendered
    assert "api_key: <redacted>" in rendered
    assert "max_output_tokens: 256" in rendered


def test_prefetch_models_raises_when_fail_fast_enabled(monkeypatch) -> None:
    class Stage:
        model_id = "missing/model"

    def fail_snapshot(_model_id: str) -> str:
        raise RuntimeError("missing auth")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fail_snapshot)

    with pytest.raises(RuntimeError, match="Model pre-fetch failed"):
        _prefetch_models([Stage()], fail_on_error=True)


def test_prefetch_models_can_continue_for_local_development(monkeypatch) -> None:
    class Stage:
        model_id = "missing/model"

    def fail_snapshot(_model_id: str) -> str:
        raise RuntimeError("offline")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fail_snapshot)

    _prefetch_models([Stage()], fail_on_error=False)


def test_tutorial_config_wires_launcher_overrides_to_stages(tmp_path) -> None:
    cfg = OmegaConf.load("tutorials/audio/qwen_omni_inprocess/qwen_omni_inprocess.yaml")
    cfg.input_manifest = "data_config.yaml"
    cfg.workspace_dir = str(tmp_path)
    cfg.final_manifest = str(tmp_path / "final.jsonl")
    cfg.language_short = "es"
    cfg.max_segment_length = 12.5

    pipeline = build_granary_v2_pipeline(cfg)
    reader, qwen_stage, writer = pipeline.stages

    assert isinstance(reader, NemoTarredAudioReader)
    shard_reader = reader.decompose()[1]
    assert isinstance(shard_reader, NemoTarShardReaderStage)
    assert shard_reader.max_duration_s == 12.5
    assert isinstance(qwen_stage, InferenceQwenOmniStage)
    assert qwen_stage.default_language == "es"
    assert isinstance(writer, ShardedManifestWriterStage)
    assert writer.final_manifest_path == str(tmp_path / "final.jsonl")
