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

"""Check explicit selection and benchmark-only 26.07 argument adaptation."""

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch


def load_module(name: str):
    path = Path(__file__).resolve().parents[2] / "benchmarking/release_compatibility" / name
    spec = importlib.util.spec_from_file_location("compat", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_profile_selection(monkeypatch: "MonkeyPatch"):
    compat = load_module("__init__.py")
    monkeypatch.delenv("CURATOR_BENCHMARK_COMPAT_PROFILE", raising=False)
    assert compat.selected_profile() is None
    monkeypatch.setenv("CURATOR_BENCHMARK_COMPAT_PROFILE", "26.07")
    assert compat.selected_profile() == "26.07"
    monkeypatch.setenv("CURATOR_BENCHMARK_COMPAT_PROFILE", "unknown")
    with pytest.raises(ValueError, match="Unknown"):
        compat.selected_profile()


def test_minhash_preserves_arguments_and_rejects_normalization():
    compat = load_module("curator_26_07.py")
    kwargs = {"output_path": "output", "seed": 42, "num_hashes": 260, "pool": True}
    assert compat.create_minhash_stage(dict, normalize_text=False, **kwargs) == kwargs
    with pytest.raises(ValueError, match="normalize_text=False"):
        compat.create_minhash_stage(dict, normalize_text=True, **kwargs)


def test_profile_yaml_disables_only_unavailable_checks(monkeypatch: "MonkeyPatch", tmp_path: Path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "benchmarking"))
    from nemo_curator_benchmarking.config import load_benchmark_config, merge_config_files, remove_disabled_blocks

    name = "minhash_file_group_task_ray_actors"
    requirements = [
        {"metric": "num_documents_processed", "exact_value": 1046344809},
        {"metric": "throughput_docs_per_sec", "min_value": 2000000},
        {"metric": "minhash_compute_worker_time_s_mean", "min_value": 0.12, "max_value": 0.24},
        {"metric": "minhash_input_prep_worker_time_s_mean", "min_value": 0.22, "max_value": 0.35},
        {"metric": "minhash_write_worker_time_s_mean", "min_value": 0.12, "max_value": 0.35},
    ]
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump({"entries": [{"name": name, "requirements": requirements}]}))
    override_path = tmp_path / "sku.yaml"
    override_path.write_text(
        yaml.safe_dump(
            {
                "entries": [
                    {
                        "name": name,
                        "requirements": [
                            {"metric": "throughput_docs_per_sec", "min_value": 1000000},
                            {"metric": "minhash_compute_worker_time_s_mean", "enabled": True},
                        ],
                    }
                ]
            }
        )
    )
    paths = [base_path, override_path]
    monkeypatch.delenv("CURATOR_BENCHMARK_COMPAT_PROFILE", raising=False)
    normal = merge_config_files(paths)
    assert len(remove_disabled_blocks(normal)["entries"][0]["requirements"]) == 5
    monkeypatch.setenv("CURATOR_BENCHMARK_COMPAT_PROFILE", "26.07")
    generated = load_benchmark_config(paths)
    runtime = merge_config_files(paths)
    assert generated == runtime
    assert [entry["name"] for entry in runtime["entries"]] == [name]
    enabled = remove_disabled_blocks(runtime)["entries"][0]["requirements"]
    assert enabled == [requirements[0], {"metric": "throughput_docs_per_sec", "min_value": 1000000}]
    assert all(req["enabled"] is False for req in runtime["entries"][0]["requirements"][2:])


def test_profile_yaml_does_not_change_unrelated_entries(monkeypatch: "MonkeyPatch", tmp_path: Path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "benchmarking"))
    from nemo_curator_benchmarking.config import merge_config_files

    monkeypatch.setenv("CURATOR_BENCHMARK_COMPAT_PROFILE", "26.07")
    base = {"entries": [{"name": "unrelated", "requirements": [{"metric": "count", "exact_value": 10}]}]}
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump(base))
    assert merge_config_files([path]) == base
