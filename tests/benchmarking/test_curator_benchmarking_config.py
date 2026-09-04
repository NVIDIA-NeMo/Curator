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

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarking"))

from curator_benchmarking.config import (
    build_benchmark_config_plan,
    entry_names,
    exact_entry_config,
    legacy_path_config,
    load_benchmark_config,
    plan_entry_slurm_timeout,
    plan_slurm_timeout,
)
from curator_benchmarking.dependencies import (
    dependency_groups_from_config,
    pyproject_dependency_requirements,
    pyproject_optional_dependency_names,
    pyproject_optional_dependency_requirements,
    python_extras_for_dependency_groups,
    system_dependency_groups_for_dependency_groups,
    validate_dependency_groups,
)
from curator_benchmarking.paths import (
    benchmark_package_dir,
    resolve_benchmark_suite_dir,
    volume_mount_pairs_from_configs,
)
from curator_benchmarking.system_tools import (
    system_dependency_install_command,
    system_dependency_names,
)


def test_load_benchmark_config_uses_runner_merge_semantics(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    override = tmp_path / "override.yaml"
    base.write_text(
        """
paths:
  - name: results_path
    host_path: /results
entries:
  - name: entry_a
    timeout_s: 100
    ray:
      num_cpus: 16
  - name: entry_b
    timeout_s: 200
""".lstrip()
    )
    override.write_text(
        """
entries:
  - name: entry_a
    ray:
      num_gpus: 4
  - name: entry_c
    timeout_s: 300
""".lstrip()
    )

    config = load_benchmark_config([base, override])

    entries = {entry["name"]: entry for entry in config["entries"]}
    assert entries["entry_a"]["timeout_s"] == 100
    assert entries["entry_a"]["ray"] == {"num_cpus": 16, "num_gpus": 4}
    assert [entry["name"] for entry in config["entries"]] == ["entry_a", "entry_b", "entry_c"]


def test_build_benchmark_config_plan_filters_disabled_and_applies_defaults() -> None:
    plan = build_benchmark_config_plan(
        {
            "default_timeout_s": 1000,
            "startup_timeout_s": 100,
            "cleanup_timeout_s": 10,
            "ray": {"num_cpus": 128, "num_gpus": 8},
            "entries": [
                {"name": "entry_a", "ray": {"num_gpus": 4}},
                {"name": "entry_b", "timeout_s": 200, "enabled": False},
            ],
            "data_setups": [{"name": "dataset_a", "timeout_s": 300}],
        }
    )

    assert [entry.name for entry in plan.entries] == ["entry_a"]
    assert plan.entries[0].timeout_s == 1000
    assert plan.entries[0].ray == {"num_cpus": 128, "num_gpus": 4}
    assert [setup.name for setup in plan.data_setups] == ["dataset_a"]
    assert plan.data_setups[0].timeout_s == 300


def test_entry_names_and_exact_entry_config_support_legacy_single_entry_overrides() -> None:
    config = {
        "entries": [
            {"name": "math_preprocess", "timeout_s": 100},
            {"name": "math_preprocess_classifier", "timeout_s": 200},
            {"name": "disabled_entry", "enabled": False},
        ]
    }

    assert entry_names(config) == ["math_preprocess", "math_preprocess_classifier"]
    assert exact_entry_config(config, "math_preprocess") == {
        "entries": [{"name": "math_preprocess", "timeout_s": 100}]
    }

    with pytest.raises(ValueError, match="disabled_entry"):
        exact_entry_config(config, "disabled_entry", enabled_only=True)


def test_plan_slurm_timeout_caps_entry_runtime_before_adding_buffers() -> None:
    timeout = plan_slurm_timeout(
        14400,
        startup_timeout_s=0,
        cleanup_timeout_s=60,
        max_timeout_s=14340,
        slurm_max_time_s=14400,
    )

    assert timeout.entry_timeout_s == 14340
    assert timeout.wall_time_s == 14400
    assert timeout.effective_max_timeout_s == 14340
    assert timeout.capped is True


def test_plan_entry_slurm_timeout_uses_benchmark_config_plan_defaults() -> None:
    plan = build_benchmark_config_plan(
        {
            "default_timeout_s": 2000,
            "startup_timeout_s": 300,
            "cleanup_timeout_s": 60,
            "min_timeout_s": 600,
            "entries": [{"name": "entry_a"}],
        }
    )

    timeout = plan_entry_slurm_timeout(plan.entries[0], plan)

    assert timeout.entry_timeout_s == 2000
    assert timeout.wall_time_s == 2360


def test_legacy_path_config_converts_paths_and_merges_dataset_formats() -> None:
    base_config = {
        "datasets": [
            {
                "name": "commoncrawl",
                "formats": [
                    {"type": "jsonl", "path": "/old/jsonl"},
                    {"type": "parquet", "path": "/old/parquet"},
                ],
            }
        ]
    }
    override_config = {
        "paths": [
            {"name": "results_path", "host_path": "/results"},
            {"name": "datasets_path", "host_path": "/datasets"},
            {"name": "model_weights_path", "host_path": "/weights"},
        ],
        "datasets": [
            {
                "name": "commoncrawl",
                "formats": [{"type": "jsonl", "path": "{datasets_path}/commoncrawl"}],
            }
        ],
    }

    legacy = legacy_path_config(base_config, override_config)

    assert legacy["results_path"] == "/results"
    assert legacy["datasets_path"] == "/datasets"
    assert legacy["model_weights_path"] == "/weights"
    assert legacy["datasets"][0]["formats"] == [
        {"type": "jsonl", "path": "{datasets_path}/commoncrawl"},
        {"type": "parquet", "path": "/old/parquet"},
    ]


def test_volume_mount_pairs_from_configs_reads_merged_paths(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    override = tmp_path / "override.yaml"
    base.write_text(
        """
paths:
  - name: results_path
    host_path: /base/results
entries: []
""".lstrip()
    )
    override.write_text(
        """
paths:
  - name: results_path
    host_path: /override/results
    container_path: /container/results
""".lstrip()
    )

    assert volume_mount_pairs_from_configs([base, override]) == [
        (Path("/override/results"), Path("/container/results"))
    ]


def test_resolve_benchmark_suite_dir_accepts_checkout_root_and_package_dir(tmp_path: Path) -> None:
    checkout_dir = tmp_path / "Curator"
    package_dir = checkout_dir / "benchmarking"
    (package_dir / "curator_benchmarking").mkdir(parents=True)
    (package_dir / "pyproject.toml").write_text("")

    assert resolve_benchmark_suite_dir(checkout_dir) == package_dir
    assert resolve_benchmark_suite_dir(package_dir) == package_dir
    assert benchmark_package_dir(checkout_dir) == package_dir


def test_dependency_helpers_use_explicit_yaml_dependency_groups() -> None:
    config = {
        "dependencies": ["visual"],
        "sinks": [
            {"name": "slack", "enabled": True, "dependencies": ["sinks"]},
            {"name": "gdrive", "enabled": False, "dependencies": ["sinks"]},
        ],
        "data_setups": [
            {"name": "audio_setup", "enabled": True, "dependencies": ["audio"]},
        ],
        "entries": [
            {"name": "video_entry", "enabled": True, "dependencies": ["video"]},
            {"name": "math_entry", "enabled": True, "dependencies": ["math"]},
            {"name": "disabled_entry", "enabled": False, "dependencies": ["nemotron_parse"]},
        ],
    }

    assert dependency_groups_from_config(config) == ("audio", "math", "sinks", "video", "visual")
    assert dependency_groups_from_config(config, entry_names=["video_entry"]) == (
        "audio",
        "sinks",
        "video",
        "visual",
    )
    assert python_extras_for_dependency_groups(["audio", "math", "sinks", "video", "visual"]) == (
        "audio",
        "sinks",
        "video",
        "visual",
    )
    assert system_dependency_groups_for_dependency_groups(["audio", "math", "sinks", "video"]) == (
        "audio",
        "math",
        "video",
    )


def test_dependency_groups_validate_against_pyproject_extras_and_system_deps() -> None:
    assert "audio" in pyproject_optional_dependency_names()
    assert validate_dependency_groups(["audio", "math", "video"]) is None

    with pytest.raises(ValueError, match="unknown benchmark dependency group"):
        validate_dependency_groups(["does_not_exist"])


def test_pyproject_requirement_helpers_read_core_and_optional_dependencies(tmp_path: Path) -> None:
    package_dir = tmp_path / "benchmarking"
    (package_dir / "curator_benchmarking").mkdir(parents=True)
    (package_dir / "pyproject.toml").write_text(
        """
[project]
dependencies = [
    "core-package>=1",
]

[project.optional-dependencies]
audio = [
    "audio-package==2",
]
video = [
    "video-package",
]
""".lstrip()
    )

    assert pyproject_dependency_requirements(package_dir) == ("core-package>=1",)
    assert pyproject_optional_dependency_requirements(["video", "audio"], suite_dir=package_dir) == (
        ("audio", ("audio-package==2",)),
        ("video", ("video-package",)),
    )


def test_dependency_groups_reject_unknown_selected_entry_name() -> None:
    config = {"entries": [{"name": "known_entry", "dependencies": ["audio"]}]}

    with pytest.raises(ValueError, match="does not contain enabled entry"):
        dependency_groups_from_config(config, entry_names=["missing_entry"])


def test_system_dependency_install_command_uses_command_suite_dir() -> None:
    command = system_dependency_install_command(
        ["audio"],
        suite_dir=Path("/container/suite"),
        source_suite_dir=Path.cwd(),
    )

    assert system_dependency_names() == ("audio", "math", "video")
    assert "/container/suite/system_deps/audio/install.sh" in command
    assert str(Path.cwd()) not in command
