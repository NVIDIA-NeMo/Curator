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

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import hydra
import pytest
from omegaconf import OmegaConf

from nemo_curator.stages.audio.text_filtering import AudioTextLLMStage
from tutorials.audio.text_processing import run

CONFIG_DIR = Path(__file__).resolve().parents[2] / "tutorials/audio/text_processing/config"
# Resolved prompts from nithinraok/Curator at 1a260a35472229a8a75a122b9a2bb9d92d5855f9.
PROMPT_HASHES = {
    "recover_entities": "e6728bc4c8358c9e4f812ede9d553de09fab30eabe98c6e971aac8b7913340e8",
    "pnc": "47e429e1d6ae5fb44aacae07164e6bdfc4d25cd1f3c9f7e965da93103b6ff39d",
    "pnc_indic": "e21bc48756162d9effccaa577c56624ba0ea01e5c392714fc2ba716e21b1741a",
    "tn": "e8f7ea8c70178f4e9bc0049cf21516e154713bc37e7d1a3c3fe5f1b33097ced4",
    "itn": "0f2d8785bab5592a6d417fd4f250e3aeec5e89efa4e62a3e21c9cb46e7388511",
    "disfluency": "db1d502ad02bf7cbd0116d39c2c3cecb3992419c73e10c6d3678b4df0e7cc67a",
    "captioning": "054fc8cf5c35e6ee90486e41687a9b523480dc5e04d71576ada900f01b6fc48e",
    "code_switching": "b9c035b7ec7cc539e73352a61191e806d14c64ee7418fa11e8ac563f94b45291",
    "speech_qa": "9f04d3148446170e1541f21638eed25eb37e7aec048592e543de69a01efa81d0",
    "language_id": "567741809a9bb3a524d55acd96b73071d21ecaa0ad105f70baf5f034247219f0",
}


@pytest.fixture(scope="session")
def shared_ray_cluster() -> None:
    """Configuration and lifecycle tests do not start a Ray cluster."""


@pytest.mark.parametrize("operation", PROMPT_HASHES)
def test_operation_config_creates_audio_stage(operation: str):
    with hydra.initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = hydra.compose(
            config_name="single_operation",
            overrides=[
                f"operation={operation}",
                "manifest_path=/data/input.jsonl",
                "output_path=/data/output.jsonl",
                "endpoint=http://inference-host:8000/v1",
            ],
        )

    stage = hydra.utils.instantiate(cfg.stages[1])
    assert isinstance(stage, AudioTextLLMStage)
    assert stage.resources.gpus == 0
    assert stage.output_text_key == cfg.operation.output_text_key
    assert stage.generation_config.extra_kwargs == {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    assert stage.prompt == stage.prompt.strip()
    assert hashlib.sha256(stage.prompt.encode()).hexdigest() == PROMPT_HASHES[operation]
    messages = stage._build_messages("नमस्ते दुनिया", {"source_lang": "hi", "granary_v1_prediction": "मूल पाठ"})
    assert any("नमस्ते दुनिया" in message["content"] for message in messages)
    assert all("{language_rules}" not in message["content"] for message in messages)
    if operation == "pnc_indic":
        assert len(stage.language_rules) == 22
        assert stage.language_rules["hi"] in messages[0]["content"]
    elif operation == "recover_entities":
        assert stage.mode == "recover_entities"
        assert messages[1] == {"role": "user", "content": 'INITIAL:\n"मूल पाठ"\n\nNORMALIZED:\n"नमस्ते दुनिया"'}
        assert stage.generation_config.max_tokens == 1024
    elif operation == "language_id":
        assert stage.mode == "language_id"
        assert "_skipme" in stage.outputs()[1]


def test_full_config_preserves_reference_dependencies_and_serving_geometry():
    with hydra.initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = hydra.compose(
            config_name="text",
            overrides=[
                "manifest_path=/data/input.jsonl",
                "output_path=/data/output.jsonl",
                "endpoint=http://inference-host:8000/v1",
                "text_key=normalized",
                "ground_truth_key=reference",
                "language_key=language",
            ],
        )

    stages = [hydra.utils.instantiate(stage_cfg) for stage_cfg in cfg.stages[1:-1]]
    assert all(isinstance(stage, AudioTextLLMStage) for stage in stages)
    assert [stage.name for stage in stages] == [
        "RecoverEntities",
        "PnCRestoration",
        "LanguageID",
        "TextNormalization",
        "FusedTextLLM",
        "ITNRestoration",
        "DisfluencyRemoval",
    ]
    available = {"normalized", "reference", "language"}
    for stage in stages:
        assert set(stage.inputs()[1]) <= available
        available.update(stage.outputs()[1])
        assert stage.resources.gpus == 0
        assert stage.num_workers() == 12
        assert stage.batch_size == 128
        assert stage.max_concurrent_requests == 256
        assert stage.request_timeout_s == 600

    operations = [operation for stage in stages for operation in stage.operation_stages()]
    assert [(operation.text_key, operation.output_text_key) for operation in operations] == [
        ("normalized", "entity_recovered_text"),
        ("entity_recovered_text", "pnc_text"),
        ("pnc_text", "llm_language_prediction"),
        ("pnc_text", "tn_raw"),
        ("tn_raw", "captioning_text"),
        ("tn_raw", "code_switched_text"),
        ("tn_raw", "speech_qa_text"),
        ("tn_raw", "itn_raw"),
        ("itn_raw", "itn_clean"),
    ]
    assert stages[0].ground_truth_key == "reference"
    assert stages[2].mode == "language_id"
    assert [operation.name for operation in stages[4].operation_stages()] == [
        "Captioning",
        "CodeSwitching",
        "SpeechQA",
    ]
    assert all(operation.language_key == "language" for operation in operations)
    assert all(operation.prompt == operation.prompt.strip() for operation in operations)
    assert [operation.generation_config.max_tokens for operation in operations] == [1024, *([512] * 8)]

    model = hydra.utils.instantiate(cfg.inference_server.models[0])
    assert model.model_identifier == "RedHatAI/gemma-4-31B-it-FP8-dynamic"
    assert model.num_replicas * model.engine_kwargs["tensor_parallel_size"] == cfg.ray_client.num_gpus == 8
    assert model.engine_kwargs == {
        "tensor_parallel_size": 2,
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.95,
        "kv_cache_dtype": "auto",
        "max_num_seqs": 256,
        "max_num_batched_tokens": 8192,
        "language_model_only": True,
        "trust_remote_code": True,
    }
    assert model.runtime_env == {"uv": {"packages": ["transformers>=5"]}}
    assert cfg.inference_server.health_check_timeout_s == 1200


@pytest.mark.parametrize("endpoint", [None, "http://external:8000/v1"])
def test_runner_cleans_up_owned_services_after_pipeline_failure(monkeypatch: pytest.MonkeyPatch, endpoint: str | None):
    events = []

    @contextmanager
    def service(name: str, value: object) -> Iterator[object]:
        events.append(f"start:{name}")
        try:
            yield value
        finally:
            events.append(f"stop:{name}")

    cfg = OmegaConf.create(
        {
            "manifest_path": "/data/input.jsonl",
            "output_path": "/data/output.jsonl",
            "endpoint": endpoint,
            "inference_server": {},
        }
    )
    server = SimpleNamespace(endpoint="http://local:8000/v1")
    instantiate = Mock(return_value=service("server", server))
    pipeline = Mock()
    pipeline.run.side_effect = RuntimeError("pipeline failed")
    executor = object()
    monkeypatch.setattr(run, "create_ray_client_from_yaml", lambda _: service("ray", object()))
    monkeypatch.setattr(run.hydra.utils, "instantiate", instantiate)
    monkeypatch.setattr(run, "create_pipeline_from_yaml", lambda *_args, **_kwargs: pipeline)
    monkeypatch.setattr(run, "create_executor_from_yaml", lambda _: executor)

    with pytest.raises(RuntimeError, match="pipeline failed"):
        run.main.__wrapped__(cfg)

    pipeline.run.assert_called_once_with(executor=executor)
    if endpoint:
        instantiate.assert_not_called()
        assert cfg.endpoint == endpoint
        assert events == ["start:ray", "stop:ray"]
    else:
        instantiate.assert_called_once_with(cfg.inference_server)
        assert cfg.endpoint == server.endpoint
        assert events == ["start:ray", "start:server", "stop:server", "stop:ray"]


def test_runner_rejects_output_alias_of_input_before_starting_services(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manifest = tmp_path / "input.jsonl"
    manifest.write_text('{"text":"hello"}\n')
    output = tmp_path / "output.jsonl"
    output.symlink_to(manifest)
    cfg = OmegaConf.create({"manifest_path": str(manifest), "output_path": str(output)})
    start = Mock()
    monkeypatch.setattr(run, "create_ray_client_from_yaml", start)

    with pytest.raises(ValueError, match="output_path must differ"):
        run.main.__wrapped__(cfg)

    start.assert_not_called()
    assert manifest.read_text() == '{"text":"hello"}\n'
