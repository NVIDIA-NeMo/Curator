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

import pytest
import yaml
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarking" / "scripts"))

from audio_yaml_pipeline_benchmark import count_output_rows

from nemo_curator.config.run import create_pipeline_from_yaml

_REPO = Path(__file__).resolve().parents[3]


def test_count_output_rows_skips_blank_lines_and_missing_files(tmp_path: Path) -> None:
    manifest = tmp_path / "out.jsonl"
    manifest.write_text('{"a": 1}\n\n{"a": 2}\n')

    assert count_output_rows(manifest) == 2
    assert count_output_rows(tmp_path / "missing.jsonl") == 0


def _yaml_entries() -> list[dict]:
    config = yaml.safe_load((_REPO / "benchmarking" / "benchmarks.yaml").read_text())
    return [e for e in config["entries"] if e.get("script") == "audio_yaml_pipeline_benchmark.py"]


@pytest.mark.parametrize("entry", _yaml_entries(), ids=lambda e: e["name"])
def test_registered_pipeline_yamls_build_with_their_overrides(entry: dict) -> None:
    args = entry["args"].replace("{curator_repo_dir}", str(_REPO))
    args = re.sub(r"\{[^}]+\}", "/tmp/benchmark", args)  # noqa: S108
    cfg = OmegaConf.load(re.search(r"--pipeline-yaml=(\S+)", args).group(1))
    cfg.merge_with_dotlist(re.findall(r"--set=(\S+)", args))

    assert create_pipeline_from_yaml(cfg).stages
