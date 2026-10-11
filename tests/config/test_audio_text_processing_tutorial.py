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


@pytest.mark.parametrize(
    "operation",
    ["pnc", "pnc_indic", "tn", "itn", "disfluency", "captioning", "code_switching", "speech_qa", "language_id"],
)
def test_operation_config_creates_audio_stage(operation: str):
    with hydra.initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = hydra.compose(
            config_name="text",
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
    messages = stage._build_messages("नमस्ते दुनिया", {"source_lang": "hi"})
    assert any("नमस्ते दुनिया" in message["content"] for message in messages)
    assert all("{language_rules}" not in message["content"] for message in messages)
    if operation == "pnc_indic":
        assert len(stage.language_rules) == 22
        assert stage.language_rules["hi"] in messages[0]["content"]


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
