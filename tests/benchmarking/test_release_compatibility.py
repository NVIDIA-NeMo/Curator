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

import argparse
import ast
import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    assert compat.validate_profile(None) is None
    monkeypatch.setenv("CURATOR_BENCHMARK_COMPAT_PROFILE", "26.07")
    assert compat.compatibility_config() == {}
    assert compat.validate_profile("26.07") == "26.07"
    with pytest.raises(ValueError, match="Unknown"):
        compat.validate_profile("unknown")


def test_profile_is_passed_only_to_affected_scripts():
    compat = load_module("__init__.py")
    command = "python minhash_benchmark.py --seed 42"
    name = "minhash_file_group_task_ray_actors"
    assert compat.apply_script_profile(command, name, "26.07") == command + " --benchmark-compat-profile 26.07"
    assert compat.apply_script_profile(command, name, None) == command
    assert compat.apply_script_profile(command, "unrelated", "26.07") == command


@pytest.mark.parametrize(
    "script",
    [
        "run.py",
        "scripts/minhash_benchmark.py",
        "scripts/exact_dedup_identification_benchmark.py",
        "scripts/fuzzy_dedup_identification_benchmark.py",
        "scripts/audio_tagging_benchmark.py",
    ],
)
def test_profile_cli_argument(script: str, monkeypatch: "MonkeyPatch"):
    # Exercise the real parsers without importing GPU-only product dependencies.
    root = Path(__file__).resolve().parents[2] / "benchmarking"
    tree = ast.parse((root / script).read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    statements = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in ("_dataset_ratio", "_parse_json_object")
    ]
    for statement in main.body:
        statements.append(statement)
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "args" for target in statement.targets
        ):
            break
    code = compile(ast.Module(body=statements, type_ignores=[]), str(root / script), "exec")
    compat = load_module("__init__.py")
    namespace = {
        "argparse": argparse,
        "Path": Path,
        "validate_profile": compat.validate_profile,
        "json": json,
        "Any": Any,
        "parse_memory_size": str,
        "DEFAULT_AUDIO_TAGGING_CACHE_DIR": "cache",
    }
    required = ["--benchmark-results-path", "results", "--input-path", "input", "--output-path", "output"]
    if script == "run.py":
        required = ["--config", "test.yaml"]
    elif script == "scripts/fuzzy_dedup_identification_benchmark.py":
        required.extend(["--cache-path", "cache"])
    elif script == "scripts/audio_tagging_benchmark.py":
        required = [
            "--benchmark-results-path",
            "results",
            "--scratch-output-path",
            "scratch",
            "--diarization-model-path",
            "model",
        ]
    monkeypatch.setenv("CURATOR_BENCHMARK_COMPAT_PROFILE", "26.07")
    monkeypatch.setattr("sys.argv", [script, *required])
    exec(code, namespace)  # noqa: S102
    assert namespace["args"].benchmark_compat_profile is None
    monkeypatch.setattr("sys.argv", [script, *required, "--benchmark-compat-profile", "26.07"])
    exec(code, namespace)  # noqa: S102
    assert namespace["args"].benchmark_compat_profile == "26.07"
    monkeypatch.setattr("sys.argv", [script, *required, "--benchmark-compat-profile", "unknown"])
    with pytest.raises(SystemExit) as error:
        exec(code, namespace)  # noqa: S102
    assert error.value.code == 2


def test_minhash_preserves_arguments_and_rejects_normalization():
    compat = load_module("curator_26_07.py")
    kwargs = {"output_path": "output", "seed": 42, "num_hashes": 260, "pool": True}
    assert compat.create_minhash_stage(dict, normalize_text=False, **kwargs) == kwargs
    with pytest.raises(ValueError, match="normalize_text=False"):
        compat.create_minhash_stage(dict, normalize_text=True, **kwargs)


def test_dedup_preserves_workload_arguments_and_rejects_normalization():
    compat = load_module("curator_26_07.py")
    kwargs = {"input_path": "input", "output_path": "output", "text_field": "text", "rmm_pool_size": "auto"}
    for use_async_memory in (False, True):
        assert (
            compat.create_dedup_workflow(dict, normalize_text=False, use_async_memory=use_async_memory, **kwargs)
            == kwargs
        )
    with pytest.raises(ValueError, match="normalize_text=False"):
        compat.create_dedup_workflow(dict, normalize_text=True, use_async_memory=True, **kwargs)


def test_diarization_preserves_model_settings_and_supplies_auth(monkeypatch: "MonkeyPatch"):
    compat = load_module("curator_26_07.py")
    kwargs = {"model_name": "staged-model", "segmentation_batch_size": 128, "embedding_batch_size": 128}
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert compat.create_diarization_stage(dict, **kwargs) == {"hf_token": None, **kwargs}
    monkeypatch.setenv("HF_TOKEN", "test-token")
    assert compat.create_diarization_stage(dict, **kwargs) == {"hf_token": "test-token", **kwargs}


def test_asr_aligner_uses_release_defaults_without_changing_workload(caplog: pytest.LogCaptureFixture):
    compat = load_module("curator_26_07.py")
    kwargs = {"model_name": "staged-model", "batch_size": 1, "transcribe_batch_size": 32, "decoder_type": "rnnt"}
    assert compat.create_asr_aligner_stage(dict, use_cuda_graphs=True, **kwargs) == kwargs
    assert "Explicit graph enablement is not guaranteed" in caplog.text
    with pytest.raises(ValueError, match="cannot honor"):
        compat.create_asr_aligner_stage(dict, use_cuda_graphs=False, **kwargs)


def test_squim_adapter_preserves_tasks_and_configuration():
    compat = load_module("curator_26_07.py")

    class ReleaseSquim:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def process_batch(self, tasks: list[object]) -> list[object]:
            assert isinstance(tasks, list)
            return tasks

    stage = compat.create_squim_stage(ReleaseSquim, name="SquimMetrics", compute_batch_size=32)
    assert stage.kwargs == {"name": "SquimMetrics", "compute_batch_size": 32}
    tasks = (object(), object())
    result = stage.process_batch(tasks)
    assert result == list(tasks)
    assert all(actual is expected for actual, expected in zip(result, tasks, strict=True))
    assert stage.process_batch(()) == []


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
    monkeypatch.setenv("CURATOR_BENCHMARK_COMPAT_PROFILE", "26.07")
    normal = merge_config_files(paths)
    assert len(remove_disabled_blocks(normal)["entries"][0]["requirements"]) == 5
    generated = load_benchmark_config(paths, benchmark_compat_profile="26.07")
    runtime = merge_config_files(paths, benchmark_compat_profile="26.07")
    assert generated == runtime
    assert [entry["name"] for entry in runtime["entries"]] == [name]
    enabled = remove_disabled_blocks(runtime)["entries"][0]["requirements"]
    assert enabled == [requirements[0], {"metric": "throughput_docs_per_sec", "min_value": 1000000}]
    assert all(req["enabled"] is False for req in runtime["entries"][0]["requirements"][2:])


def test_profile_yaml_does_not_change_unrelated_entries(monkeypatch: "MonkeyPatch", tmp_path: Path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "benchmarking"))
    from nemo_curator_benchmarking.config import merge_config_files

    base = {"entries": [{"name": "unrelated", "requirements": [{"metric": "count", "exact_value": 10}]}]}
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump(base))
    assert merge_config_files([path], benchmark_compat_profile="26.07") == base
