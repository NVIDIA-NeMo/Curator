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
import json
import subprocess
import sys
from pathlib import Path

import hydra
import pytest
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import InterpolationToMissingValueError

from nemo_curator.config.run import create_pipeline_from_yaml
from nemo_curator.core.serve.dynamo.infra import engine_kwargs_to_cli_flags
from nemo_curator.core.serve.dynamo.vllm import dynamo_runtime_env
from nemo_curator.stages.audio.text_filtering import AudioTextLLMStage
from nemo_curator.tasks import AudioTask

TUTORIAL_DIR = Path(__file__).resolve().parents[2] / "tutorials/audio/granary_v2/text_pipeline"
# Resolved prompts from nithinraok/Curator at 1a260a35472229a8a75a122b9a2bb9d92d5855f9.
PROMPT_HASHES = {
    "RecoverEntities": "e6728bc4c8358c9e4f812ede9d553de09fab30eabe98c6e971aac8b7913340e8",  # pragma: allowlist secret
    "PnCRestoration": "47e429e1d6ae5fb44aacae07164e6bdfc4d25cd1f3c9f7e965da93103b6ff39d",  # pragma: allowlist secret
    "LanguageID": "567741809a9bb3a524d55acd96b73071d21ecaa0ad105f70baf5f034247219f0",  # pragma: allowlist secret
    "TextNormalization": "e8f7ea8c70178f4e9bc0049cf21516e154713bc37e7d1a3c3fe5f1b33097ced4",  # pragma: allowlist secret
    "Captioning": "054fc8cf5c35e6ee90486e41687a9b523480dc5e04d71576ada900f01b6fc48e",  # pragma: allowlist secret
    "CodeSwitching": "b9c035b7ec7cc539e73352a61191e806d14c64ee7418fa11e8ac563f94b45291",  # pragma: allowlist secret
    "SpeechQA": "9f04d3148446170e1541f21638eed25eb37e7aec048592e543de69a01efa81d0",  # pragma: allowlist secret
    "ITNRestoration": "0f2d8785bab5592a6d417fd4f250e3aeec5e89efa4e62a3e21c9cb46e7388511",  # pragma: allowlist secret
    "DisfluencyRemoval": "db1d502ad02bf7cbd0116d39c2c3cecb3992419c73e10c6d3678b4df0e7cc67a",  # pragma: allowlist secret
}


@pytest.fixture(scope="session")
def shared_ray_cluster() -> None:
    """Configuration and lifecycle tests do not start a Ray cluster."""


@pytest.fixture
def pipeline_config() -> DictConfig:
    with hydra.initialize_config_dir(version_base=None, config_dir=str(TUTORIAL_DIR)):
        return hydra.compose(
            config_name="text_pipeline",
            overrides=[
                "manifest_path=/data/input.jsonl",
                "output_path=/data/output.jsonl",
                "endpoint=http://inference-host:8000/v1",
                "cache_root=/cache",
            ],
        )


def test_tutorial_has_one_self_contained_yaml_and_no_custom_runner():
    assert sorted(path.relative_to(TUTORIAL_DIR).as_posix() for path in TUTORIAL_DIR.rglob("*.yaml")) == [
        "text_pipeline.yaml"
    ]
    assert not list(TUTORIAL_DIR.rglob("*.yml"))
    assert not list(TUTORIAL_DIR.rglob("run.py"))


@pytest.mark.parametrize("operation_name", PROMPT_HASHES)
def test_pipeline_preserves_reference_prompts_and_audio_contract(pipeline_config: DictConfig, operation_name: str):
    pipeline = create_pipeline_from_yaml(pipeline_config, log_config=False)
    operations = {
        operation.name: operation for stage in pipeline.stages[1:-1] for operation in stage.operation_stages()
    }
    stage = operations[operation_name]
    assert isinstance(stage, AudioTextLLMStage)
    assert stage.resources.gpus == 0
    assert stage.generation_config.extra_kwargs == {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    assert stage.prompt == stage.prompt.strip()
    assert hashlib.sha256(stage.prompt.encode()).hexdigest() == PROMPT_HASHES[operation_name]
    messages = stage._build_messages("hello world", {"source_lang": "en", "text": "Hello World"})
    assert any("hello world" in message["content"] for message in messages)
    assert all("{language}" not in message["content"] for message in messages)
    if operation_name == "RecoverEntities":
        assert stage.mode == "recover_entities"
        assert messages[1] == {"role": "user", "content": 'INITIAL:\n"Hello World"\n\nNORMALIZED:\n"hello world"'}
        assert stage.generation_config.max_tokens == 1024
    elif operation_name == "LanguageID":
        assert stage.mode == "language_id"
        assert "_skipme" in stage.outputs()[1]

    task = AudioTask(data={stage.text_key: "", "audio_filepath": "/data/original.wav", "metadata": {"keep": True}})
    assert stage.process(task) is task
    assert task.data["audio_filepath"] == "/data/original.wav"
    assert task.data["metadata"] == {"keep": True}
    assert stage.output_text_key in task.data


def test_pipeline_preserves_english_dependencies_and_fused_order(pipeline_config: DictConfig):
    assert pipeline_config.ground_truth_key == "text"
    assert pipeline_config.text_key == "abbreviated_text"
    pipeline_config.text_key = "normalized"
    pipeline_config.ground_truth_key = "reference"
    pipeline_config.language_key = "language"
    stages = create_pipeline_from_yaml(pipeline_config, log_config=False).stages[1:-1]
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
        assert (stage.num_workers(), stage.batch_size, stage.max_concurrent_requests, stage.request_timeout_s) == (
            12,
            128,
            256,
            600,
        )

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
    assert stages[3].validation_mode == "none"
    assert [operation.name for operation in stages[4].operation_stages()] == [
        "Captioning",
        "CodeSwitching",
        "SpeechQA",
    ]
    assert all(operation.language_key == "language" for operation in operations)
    assert [operation.generation_config.max_tokens for operation in operations] == [1024, *([512] * 8)]


def test_serving_config_uses_existing_api_and_reference_geometry(pipeline_config: DictConfig):
    model = hydra.utils.instantiate(pipeline_config.inference_server.models[0])
    backend = hydra.utils.instantiate(pipeline_config.inference_server.backend)
    assert model.model_identifier == "RedHatAI/gemma-4-31B-it-FP8-dynamic"
    assert model.num_replicas == 4
    assert model.engine_kwargs["tensor_parallel_size"] == 2
    assert model.mode == "aggregated"
    assert backend is not None
    assert pipeline_config.ray_client.num_gpus == 0
    assert pipeline_config.inference_server.health_check_timeout_s == 1200
    expected = {
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.95,
        "kv_cache_dtype": "auto",
        "max_num_seqs": 256,
        "max_num_batched_tokens": 8192,
        "language_model_only": True,
        "trust_remote_code": True,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "async_scheduling": False,
        "enforce_eager": False,
        "revision": "d4ab4f579dd3516f97d8a6a4c98d0653480bad15",  # pragma: allowlist secret
    }
    assert all(model.engine_kwargs[key] == value for key, value in expected.items())
    runtime_env = dynamo_runtime_env(model)
    assert "transformers==5.8.0" in runtime_env["uv"]["packages"]
    assert any(package.startswith("ai-dynamo[vllm]") for package in runtime_env["uv"]["packages"])
    assert runtime_env["env_vars"] == {
        "VLLM_USE_DEEP_GEMM": "0",
        "VLLM_CACHE_ROOT": "/cache/vllm",
        "TORCHINDUCTOR_CACHE_DIR": "/cache/torchinductor",
        "TRITON_CACHE_DIR": "/cache/triton",
    }
    flags = engine_kwargs_to_cli_flags(model.engine_kwargs)
    assert "--no-async-scheduling" in flags
    assert "--no-enforce-eager" in flags
    assert json.loads(flags[flags.index("--compilation-config") + 1]) == {"cache_dir": "/cache/vllm/compile"}


def test_managed_serving_requires_an_explicit_cache_directory():
    with hydra.initialize_config_dir(version_base=None, config_dir=str(TUTORIAL_DIR)):
        cfg = hydra.compose(config_name="text_pipeline")
    with pytest.raises(InterpolationToMissingValueError):
        OmegaConf.to_container(cfg.inference_server.models[0], resolve=True)


def test_shared_runner_requires_endpoint_before_building_audio_pipeline():
    with hydra.initialize_config_dir(version_base=None, config_dir=str(TUTORIAL_DIR)):
        cfg = hydra.compose(
            config_name="text_pipeline",
            overrides=["manifest_path=/data/input.jsonl", "output_path=/data/output.jsonl"],
        )
    with pytest.raises(InterpolationToMissingValueError):
        create_pipeline_from_yaml(cfg, log_config=False)
    cfg.endpoint = "http://inference-host:8000/v1"
    pipeline = create_pipeline_from_yaml(cfg, log_config=False)
    assert len(pipeline.stages) == 9
    assert pipeline.stages[1].endpoint == cfg.endpoint


def test_tutorial_composes_with_shared_runner_cli():
    repo_root = Path(__file__).parents[2]
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(repo_root / "nemo_curator/config/run.py"),
            "--config-path",
            "../../tutorials/audio/granary_v2/text_pipeline",
            "--config-name",
            "text_pipeline",
            "--cfg",
            "job",
            "manifest_path=/data/input.jsonl",
            "output_path=/data/output.jsonl",
            "endpoint=http://inference-host:8000/v1",
            "cache_root=/cache",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    cfg = OmegaConf.create(result.stdout)
    assert cfg.manifest_path == "/data/input.jsonl"
    assert cfg.output_path == "/data/output.jsonl"
    assert cfg.inference_server._target_ == "nemo_curator.core.serve.InferenceServer"
    assert len(create_pipeline_from_yaml(cfg, log_config=False).stages) == 9
