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

import re
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarking" / "scripts"))

from audio_yaml_pipeline_benchmark import apply_overrides, count_output_rows

from nemo_curator.config.run import create_pipeline_from_yaml
from nemo_curator.tasks import AudioTask

_REPO = Path(__file__).resolve().parents[3]


def test_count_output_rows_skips_blank_lines_and_missing_files(tmp_path: Path) -> None:
    manifest = tmp_path / "out.jsonl"
    manifest.write_text('{"a": 1}\n\n{"a": 2}\n')

    assert count_output_rows(manifest) == 2
    assert count_output_rows(tmp_path / "missing.jsonl") == 0


def _yaml_entries() -> list[dict]:
    config = yaml.safe_load((_REPO / "benchmarking" / "benchmarks.yaml").read_text())
    return [e for e in config["entries"] if e.get("script") == "audio_yaml_pipeline_benchmark.py"]


def _entry(name: str) -> dict:
    return next(entry for entry in _yaml_entries() if entry["name"] == name)


def _build_pipeline(entry: dict, placeholder: str = "/tmp/benchmark") -> list:  # noqa: S108
    args = entry["args"].replace("{curator_repo_dir}", str(_REPO))
    args = re.sub(r"\{[^}]+\}", placeholder, args)
    cfg = OmegaConf.load(re.search(r"--pipeline-yaml=(\S+)", args).group(1))
    apply_overrides(cfg, re.findall(r"--set=(\S+)", args))
    return create_pipeline_from_yaml(cfg).stages


@pytest.mark.parametrize("entry", _yaml_entries(), ids=lambda e: e["name"])
def test_registered_pipeline_yamls_build_with_their_overrides(entry: dict) -> None:
    assert _build_pipeline(entry)


def test_overrides_reject_top_level_keys_the_yaml_does_not_define() -> None:
    cfg = OmegaConf.create({"manifest_path": "???", "stages": []})

    with pytest.raises(ValueError, match="input_manifest"):
        apply_overrides(cfg, ["input_manifest=/data/in.jsonl"])


@pytest.mark.parametrize("name", ["audio_model_stages_pytorch_xenna", "audio_model_stages_tensorrt_xenna"])
def test_model_stage_adapters_accept_the_entry_kwargs(name: str) -> None:
    adapter_stages = [stage for stage in _build_pipeline(_entry(name)) if hasattr(stage, "_create_adapter")]

    assert len(adapter_stages) == 3
    for stage in adapter_stages:
        stage._create_adapter()
        assert stage.resources.requires_gpu, stage.name


def test_whisper_hallucination_entry_runs_on_a_manifest_without_duration(tmp_path: Path) -> None:
    audio_path = tmp_path / "clip.flac"
    sf.write(audio_path, np.zeros(16000, dtype=np.float32), 16000)
    stages = _build_pipeline(_entry("audio_whisper_hallucination_raydata"), placeholder=str(tmp_path / "unused"))
    tasks = [AudioTask(data={"audio_filepath": str(audio_path), "text": "a clean reference transcript"})]

    for stage in stages[1:-1]:
        stage.setup()
        tasks = stage.process_batch(tasks)

    assert tasks[0].data["duration"] == pytest.approx(1.0)
    assert "_skipme" in tasks[0].data
