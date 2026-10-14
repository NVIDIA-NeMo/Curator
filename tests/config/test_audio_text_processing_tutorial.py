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
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationToMissingValueError

from nemo_curator.core.serve.dynamo.infra import engine_kwargs_to_cli_flags
from nemo_curator.core.serve.dynamo.vllm import (
    dynamo_runtime_env,
    merge_model_runtime_envs,
    resolve_disagg_role_config,
)
from nemo_curator.stages.audio.text_filtering import AudioTextLLMStage

CONFIG_DIR = Path(__file__).resolve().parents[2] / "tutorials/audio/text_processing/config"
# Resolved prompts from nithinraok/Curator at 1a260a35472229a8a75a122b9a2bb9d92d5855f9.
PROMPT_HASHES = {
    "recover_entities": "e6728bc4c8358c9e4f812ede9d553de09fab30eabe98c6e971aac8b7913340e8",  # pragma: allowlist secret
    "pnc": "47e429e1d6ae5fb44aacae07164e6bdfc4d25cd1f3c9f7e965da93103b6ff39d",  # pragma: allowlist secret
    "pnc_indic": "e21bc48756162d9effccaa577c56624ba0ea01e5c392714fc2ba716e21b1741a",  # pragma: allowlist secret
    "tn": "e8f7ea8c70178f4e9bc0049cf21516e154713bc37e7d1a3c3fe5f1b33097ced4",  # pragma: allowlist secret
    "itn": "0f2d8785bab5592a6d417fd4f250e3aeec5e89efa4e62a3e21c9cb46e7388511",  # pragma: allowlist secret
    "disfluency": "db1d502ad02bf7cbd0116d39c2c3cecb3992419c73e10c6d3678b4df0e7cc67a",  # pragma: allowlist secret
    "captioning": "054fc8cf5c35e6ee90486e41687a9b523480dc5e04d71576ada900f01b6fc48e",  # pragma: allowlist secret
    "code_switching": "b9c035b7ec7cc539e73352a61191e806d14c64ee7418fa11e8ac563f94b45291",  # pragma: allowlist secret
    "speech_qa": "9f04d3148446170e1541f21638eed25eb37e7aec048592e543de69a01efa81d0",  # pragma: allowlist secret
    "language_id": "567741809a9bb3a524d55acd96b73071d21ecaa0ad105f70baf5f034247219f0",  # pragma: allowlist secret
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


@pytest.mark.parametrize("serving", ["reference", "explicit", "tp1_uni", "kv", "disaggregated", "speculative"])
def test_english_serving_profiles_preserve_graph_and_eight_gpu_budget(serving: str):
    overrides = ["manifest_path=/data/input.jsonl", "output_path=/data/output.jsonl"]
    with hydra.initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        baseline = hydra.compose(
            config_name="text", overrides=[*overrides, "ground_truth_key=text", "tn.validation_mode=none"]
        )
        cfg = hydra.compose(config_name="english", overrides=[*overrides, "cache_root=/cache", f"serving={serving}"])

    assert OmegaConf.to_container(cfg.stages, resolve=True) == OmegaConf.to_container(baseline.stages, resolve=True)
    assert cfg.tn.validation_mode == "none"
    assert (cfg.batch_size, cfg.num_workers, cfg.max_concurrent_requests) == (128, 12, 256)
    model = hydra.utils.instantiate(cfg.inference_server.models[0])
    backend = hydra.utils.instantiate(cfg.inference_server.backend)
    if serving == "disaggregated":
        roles = [resolve_disagg_role_config(model, role) for role in ("prefill", "decode")]
        assert sum(replicas * kwargs["tensor_parallel_size"] for replicas, kwargs in roles) == 8
        assert backend.router.mode == "kv"
    else:
        assert model.mode == "aggregated"
        assert model.num_replicas * model.engine_kwargs["tensor_parallel_size"] == 8
    assert backend.admission is None
    if serving == "tp1_uni":
        assert model.engine_kwargs["distributed_executor_backend"] == "uni"
    if serving == "kv":
        assert backend.router.mode == "kv"
        assert not backend.router.kv_events

    for key in ("max_model_len", "gpu_memory_utilization", "kv_cache_dtype", "max_num_seqs", "max_num_batched_tokens"):
        assert model.engine_kwargs[key] == baseline.serving.model.engine_kwargs[key]
    assert model.engine_kwargs["language_model_only"] is True
    assert model.install_runtime_dependencies is False
    assert cfg.runtime.versions == {
        "ai-dynamo": "1.3.1",
        "vllm": "0.23.0+cu129",
        "transformers": "5.8.0",
        "torch": "2.11.0+cu129",
        "ray": "2.57.0",
    }
    for runtime_env in (dynamo_runtime_env(model), merge_model_runtime_envs([model])):
        assert set(runtime_env) == {"env_vars"}
        assert all(value.startswith("/cache/") for value in runtime_env["env_vars"].values())

    if serving != "reference":
        kwargs = model.engine_kwargs
        assert kwargs["enable_prefix_caching"] is True
        assert kwargs["enable_chunked_prefill"] is True
        assert kwargs["async_scheduling"] is False
        assert kwargs["enforce_eager"] is False
        assert kwargs["revision"] == "d4ab4f579dd3516f97d8a6a4c98d0653480bad15"  # pragma: allowlist secret
        flags = engine_kwargs_to_cli_flags(kwargs)
        assert "--no-async-scheduling" in flags
        assert "--no-enforce-eager" in flags
        assert json.loads(flags[flags.index("--compilation-config") + 1]) == {
            "cache_dir": "/cache/vllm-0.23.0-cu129/compile"
        }
        if serving == "speculative":
            assert json.loads(flags[flags.index("--speculative-config") + 1]) == {
                "method": "mtp",
                "model": "google/gemma-4-31B-it-assistant",
                "revision": "627c5ec1458b9086b841a91e0512fd31fd2fbbf1",  # pragma: allowlist secret
                "num_speculative_tokens": 4,
                "draft_tensor_parallel_size": 2,
                "quantization": None,
            }


def test_english_requires_cache_path_and_supports_explicit_tuning():
    with hydra.initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = hydra.compose(config_name="english")
        with pytest.raises(InterpolationToMissingValueError):
            OmegaConf.to_container(cfg.inference_server.models[0], resolve=True)
        cfg = hydra.compose(
            config_name="english",
            overrides=[
                "cache_root=/cache",
                "serving=speculative",
                "serving.model.num_replicas=8",
                "serving.model.engine_kwargs.tensor_parallel_size=1",
                "+serving.model.engine_kwargs.distributed_executor_backend=uni",
                "serving.model.engine_kwargs.enforce_eager=true",
                "+serving.backend.admission={_target_:nemo_curator.core.serve.DynamoAdmissionConfig,"
                "max_waiting_requests:64,max_concurrent_requests:1024}",
            ],
        )
    model = hydra.utils.instantiate(cfg.inference_server.models[0])
    backend = hydra.utils.instantiate(cfg.inference_server.backend)
    assert model.engine_kwargs["speculative_config"]["draft_tensor_parallel_size"] == 1
    assert model.engine_kwargs["enforce_eager"] is True
    assert backend.admission.max_waiting_requests == 64
    assert backend.admission.max_concurrent_requests == 1024


@pytest.mark.parametrize("config_name", ["text", "english", "single_operation"])
def test_tutorial_composes_with_shared_runner_cli(config_name: str):
    repo_root = Path(__file__).parents[2]
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(repo_root / "nemo_curator/config/run.py"),
            "--config-path",
            "../../tutorials/audio/text_processing/config",
            "--config-name",
            config_name,
            "--cfg",
            "job",
            "manifest_path=/data/input.jsonl",
            "output_path=/data/output.jsonl",
            *(["cache_root=/cache"] if config_name == "english" else []),
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
    assert len(cfg.stages) == (3 if config_name == "single_operation" else 9)
