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

"""Focused regressions for resource-planning and calibration safety."""

from __future__ import annotations

import json
import os
import stat
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from nemo_curator.audio_agent import _ray, calibration, calibration_store, planner, verbs
from nemo_curator.audio_agent.contracts import EnvProfile
from nemo_curator.audio_agent.report import _dedup_stage_perf
from nemo_curator.stages.audio.common import ManifestReader
from nemo_curator.stages.resources import Resources
from nemo_curator.utils.performance_utils import (
    StagePerfStats,
    resource_probe_metrics,
)


class _Index:
    def __init__(self, cards: dict[str, dict[str, Any]] | None = None):
        self.cards = cards or {}

    def card(self, name: str) -> dict[str, Any] | None:
        return self.cards.get(name)


class _Stage:
    def __init__(
        self,
        *,
        resources: Resources,
        num_workers: int | None = None,
        name: str = "runtime-stage",
    ):
        self.resources = resources
        self._num_workers = num_workers
        self.name = name

    def num_workers(self) -> int | None:
        return self._num_workers


def _contract(*, requires_gpu: bool = False) -> SimpleNamespace:
    return SimpleNamespace(gates=SimpleNamespace(requires_gpu=requires_gpu))


def _plan(  # noqa: ANN202
    stage: Any,  # noqa: ANN401
    env: EnvProfile,
    *,
    card: dict[str, Any] | None = None,
    calibration_facts: dict[str, Any] | None = None,
):
    return planner.plan(
        [stage],
        [_contract()],
        env,
        index=_Index({type(stage).__name__: card or {}}),
        calibration=calibration_facts,
    )


def test_cpu_demand_and_exact_ray_reservation_are_distinct() -> None:
    stage = _Stage(resources=Resources(cpus=3.5))
    result = _plan(
        stage,
        EnvProfile(total_cpus=4, total_ram_gb=64),
        card={"resource": {"cpus": 1.0, "host_mem_gb": 1.0}},
    )

    assert result.per_stage[0]["cpus"] == 1.0
    assert result.per_stage[0]["cpu_reservation"] == 3.5
    assert result.estimate["sum_cpu_demand"] == 1.0
    assert result.estimate["sum_cpu_reservation"] == 3.5
    assert result.estimate["allocatable_cpus"] == 3
    assert result.mode == "batch"
    assert result.feasible is False
    assert any("reserves 3.5 Ray CPU" in item for item in result.escalations)


def test_ray_cluster_capacity_replaces_driver_scheduling_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = EnvProfile(
        total_cpus=64,
        total_ram_gb=256,
        has_gpu=True,
        gpu_count=4,
        gpu_mem_gb=80,
        gpu_names=["driver-gpu"],
    )
    monkeypatch.setattr(
        _ray,
        "cluster_resources",
        lambda _address: {
            "CPU": 8.0,
            "GPU": 1.0,
            "memory": float(32 * 1024**3),
        },
    )

    result = verbs._apply_ray_cluster_capacity(
        env,
        "10.20.30.40:6379",
    )

    assert result.total_cpus == 8.0
    assert result.total_ram_gb == 32.0
    assert result.gpu_count == 1
    assert result.gpu_mem_gb == 0.0
    assert result.gpu_names == []
    assert any("bound to Ray cluster capacity" in note for note in result.notes)


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        (None, True),
        ("", True),
        ("auto", True),
        ("local", True),
        ("localhost:6379", True),
        ("127.0.0.1:6379", True),
        ("127.99.1.2:6379", True),
        ("[::1]:6379", True),
        ("ray://localhost:10001", True),
        ("ray://[::1]:10001", True),
        ("10.20.30.40:6379", False),
        ("ray://remote.example:10001", False),
        ("0.0.0.0:6379", False),
    ],
)
def test_ray_address_locality_is_deterministic(
    address: str | None,
    expected: bool,
) -> None:
    assert verbs._ray_address_is_local(address) is expected


def test_loopback_ray_capacity_keeps_local_driver_gpu_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = EnvProfile(
        total_cpus=64,
        total_ram_gb=256,
        has_gpu=True,
        gpu_count=1,
        gpu_mem_gb=80,
        gpu_names=["local-driver-gpu"],
    )
    monkeypatch.setattr(
        _ray,
        "cluster_resources",
        lambda _address: {
            "CPU": 8.0,
            "GPU": 1.0,
            "memory": float(32 * 1024**3),
        },
    )

    result = verbs._apply_ray_cluster_capacity(
        env,
        "ray://[::1]:10001",
    )

    assert result.gpu_mem_gb == 80
    assert result.gpu_names == ["local-driver-gpu"]


def test_ray_cluster_probe_failure_does_not_fall_back_to_driver_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = EnvProfile(total_cpus=64, total_ram_gb=256, gpu_count=4)
    monkeypatch.setattr(
        _ray,
        "cluster_resources",
        lambda _address: (_ for _ in ()).throw(RuntimeError("remote cluster unavailable")),
    )

    with pytest.raises(RuntimeError, match="refusing to substitute driver resources"):
        verbs._apply_ray_cluster_capacity(env, "10.20.30.40:6379")


def test_ray_cluster_with_no_cpu_capacity_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _ray,
        "cluster_resources",
        lambda _address: {"CPU": 0.0, "GPU": 1.0},
    )

    with pytest.raises(RuntimeError, match="no positive finite CPU capacity"):
        verbs._apply_ray_cluster_capacity(
            EnvProfile(total_cpus=64, total_ram_gb=256),
            "10.20.30.40:6379",
        )


def test_custom_executor_never_inherits_driver_capacity() -> None:
    driver = EnvProfile(
        has_gpu=True,
        gpu_count=4,
        gpu_mem_gb=80,
        gpu_names=["driver-only"],
        total_cpus=64,
        total_ram_gb=256,
    )

    target = verbs._resource_environment(driver, None, "custom_executor")

    assert target.has_gpu is False
    assert target.gpu_count == 0
    assert target.gpu_mem_gb == 0
    assert target.total_cpus == 0
    assert driver.gpu_count == 4


def test_custom_executor_owns_capacity_and_bounded_remote_smoke_can_probe_vram() -> None:
    custom = SimpleNamespace(
        feasible=False,
        escalations=["a stage reserves 1 GPU but no GPU is available"],
        notes=[],
    )
    remote_smoke = SimpleNamespace(
        feasible=False,
        escalations=["GPU VRAM capacity is unknown; cannot prove a stage needing 8 GB fits"],
        notes=[],
    )
    remote_run = SimpleNamespace(
        feasible=False,
        escalations=["GPU VRAM capacity is unknown; cannot prove a stage needing 8 GB fits"],
        notes=[],
    )

    verbs._adapt_resource_plan_for_target(
        custom,
        execution_target="custom_executor",
        operation="run",
    )
    verbs._adapt_resource_plan_for_target(
        remote_smoke,
        execution_target="external_ray",
        operation="smoke",
    )
    verbs._adapt_resource_plan_for_target(
        remote_run,
        execution_target="external_ray",
        operation="run",
    )

    assert custom.feasible is True
    assert custom.escalations == []
    assert remote_smoke.feasible is True
    assert remote_smoke.escalations == []
    assert remote_run.feasible is False
    assert remote_run.escalations


def test_smoke_stops_only_the_ray_head_it_bootstrapped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,  # noqa: ANN001
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text('{"audio_filepath":"clip.wav"}\n', encoding="utf-8")
    stopped: list[str] = []
    bootstrapped = False

    def bootstrap():  # noqa: ANN202
        nonlocal bootstrapped
        bootstrapped = True
        return "127.0.0.1:62000"

    def owns_after_bootstrap(address=None):  # noqa: ANN001, ANN202
        # Keyed off the bootstrap itself rather than a count of how many times ownership
        # happens to be consulted. The count encoded the call sequence of one version of
        # the verb, so any additional ownership check -- such as the guarantee that a head
        # is stopped when the verb raises -- silently inverted the stub's answer.
        return bootstrapped and address in {None, "127.0.0.1:62000"}

    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.setattr(verbs, "_profile_binding", lambda _binding: None)
    monkeypatch.setattr(verbs, "_bootstrap_ray", bootstrap)
    monkeypatch.setattr(_ray, "owns_cluster", owns_after_bootstrap)
    monkeypatch.setattr(
        _ray,
        "shutdown_cluster",
        lambda address=None: stopped.append(str(address)) or True,
    )
    monkeypatch.setattr(verbs, "_apply_ray_cluster_capacity", lambda env, _address: env)
    monkeypatch.setattr(verbs, "probe_env", lambda: EnvProfile(total_cpus=8, total_ram_gb=32))
    monkeypatch.setattr(
        verbs,
        "_plan_resources",
        lambda *_args, **_kwargs: SimpleNamespace(
            mode="batch",
            feasible=True,
            escalations=[],
            machine_fingerprint="machine",
        ),
    )
    monkeypatch.setattr(
        verbs,
        "_run_pipeline_autofallback",
        lambda *_args, **_kwargs: ([], "batch"),
    )

    result = verbs.smoke(
        {
            "stages": [
                {
                    "ref": "ManifestReader",
                    "params": {"manifest_path": str(source)},
                }
            ]
        },
        sample=1,
        bootstrap_ray=True,
    )

    assert result["status"] == "completed"
    assert stopped == ["127.0.0.1:62000"]
    assert "ray_bootstrap_cleanup=completed" in result["notes"]


def test_an_unexpected_error_after_execution_still_stops_the_head_run_started(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,  # noqa: ANN001
) -> None:
    """``run`` stops its head on every path it anticipates -- and the tail is not one.

    ``smoke`` puts its stop in a ``finally``; ``run`` does not. Between execution and its
    final stop sit dataset re-binding, artifact publishing and report assembly, none of it
    inside a handler. An error there escaped past the stop and left a live local head behind
    in a process that had already given up. The next call then found RAY_ADDRESS pointing at
    it and refused as ambiguous, so one unrelated failure made every later run refuse until
    the process was restarted.
    """
    source = tmp_path / "source.jsonl"
    source.write_text('{"audio_filepath": "/tmp/a.wav"}\n', encoding="utf-8")
    stopped: list[str] = []
    bootstrapped = False

    def bootstrap():  # noqa: ANN202
        nonlocal bootstrapped
        bootstrapped = True
        return "127.0.0.1:62000"

    def exploding_tail(*_args, **_kwargs):  # noqa: ANN202
        msg = "artifact publishing blew up"
        raise RuntimeError(msg)

    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.delenv("AUDIO_AGENT_REQUIRE_SMOKE", raising=False)
    monkeypatch.setattr(verbs, "_bootstrap_ray", bootstrap)
    monkeypatch.setattr(
        _ray, "owns_cluster", lambda address=None: bootstrapped and address in {None, "127.0.0.1:62000"}
    )
    monkeypatch.setattr(_ray, "shutdown_cluster", lambda address=None: stopped.append("stopped") or True)  # noqa: ARG005
    monkeypatch.setattr(verbs, "_apply_ray_cluster_capacity", lambda env, _address: env)
    monkeypatch.setattr(verbs, "probe_env", lambda: EnvProfile(total_cpus=8, total_ram_gb=32))
    monkeypatch.setattr(verbs, "build_stages", lambda _rec: ([object()], []))
    monkeypatch.setattr(
        verbs,
        "_plan_resources",
        lambda *_args, **_kwargs: SimpleNamespace(
            mode="batch",
            feasible=True,
            escalations=[],
            machine_fingerprint="machine",
            to_dict=lambda: {"mode": "batch"},
        ),
    )
    monkeypatch.setattr(verbs, "_run_pipeline_autofallback", lambda *_a, **_k: ([object()], "batch"))
    monkeypatch.setattr(verbs, "_publish_artifacts", exploding_tail)

    with pytest.raises(RuntimeError, match="artifact publishing blew up"):
        verbs.run(
            {"stages": [{"ref": "ManifestReader", "params": {"manifest_path": str(source)}}]},
            confirm=True,
            bootstrap_ray=True,
        )

    assert stopped == ["stopped"]


def test_fixed_workers_multiply_cpu_and_gpu_scheduling_footprints() -> None:
    stage = _Stage(resources=Resources(cpus=0.5, gpus=0.5), num_workers=3)
    result = _plan(
        stage,
        EnvProfile(
            has_gpu=True,
            gpu_count=1,
            gpu_mem_gb=24,
            total_cpus=8,
            total_ram_gb=64,
        ),
        card={
            "resource": {
                "cpus": 0.5,
                "gpu_mem_gb": 1.0,
                "host_mem_gb": 1.0,
                "gpu_optional": True,
            }
        },
    )

    assert result.per_stage[0]["num_workers"] == 3
    assert result.estimate["sum_cpu_reservation"] == 1.5
    assert result.estimate["sum_gpu_reservation"] == 1.5
    assert result.mode == "batch"
    assert result.feasible is False
    assert any("reserves 1.5 Ray GPU" in item for item in result.escalations)


def test_positive_gpu_reservation_needs_a_gpu_even_when_card_says_optional() -> None:
    stage = _Stage(resources=Resources(cpus=1, gpus=0.1))
    result = _plan(
        stage,
        EnvProfile(gpu_count=0, total_cpus=8, total_ram_gb=64),
        card={
            "resource": {
                "cpus": 1.0,
                "gpu_mem_gb": 0.1,
                "host_mem_gb": 1.0,
                "gpu_optional": True,
            }
        },
    )

    assert result.feasible is False
    assert any("but no GPU is available" in item for item in result.escalations)


def test_composite_uses_configured_child_resources_and_workers() -> None:
    stage = ManifestReader(manifest_path="/tmp/input.jsonl")  # noqa: S108
    stage.with_(
        {
            "file_partitioning": {
                "resources": Resources(cpus=2.0),
                "num_workers": 2,
            }
        }
    )
    operations_before = deepcopy(stage._with_operations)

    result = planner.plan(
        [stage],
        [_contract()],
        EnvProfile(total_cpus=4, total_ram_gb=64),
        index=_Index(),
    )

    assert result.mode == "batch"
    assert result.feasible is False
    assert result.estimate["sum_cpu_reservation"] == 5.0
    assert any("expanded" in note for note in result.notes)
    assert stage._with_operations == operations_before


def test_planning_does_not_mutate_stage_resources_or_calibration() -> None:
    resources = Resources(cpus=1.25, gpus=0.25)
    stage = _Stage(resources=resources, num_workers=2)
    env = EnvProfile(
        has_gpu=True,
        gpu_count=2,
        gpu_mem_gb=24,
        total_cpus=16,
        total_ram_gb=64,
    )
    facts = {
        "calibration": {
            "_Stage": {
                "cpus": 1.5,
                "host_mem_gb": 2.0,
                "source": "measured",
                "machine_fingerprint": env.fingerprint(),
            }
        }
    }
    resources_before = deepcopy(resources)
    facts_before = deepcopy(facts)

    _plan(stage, env, calibration_facts=facts)

    assert stage.resources is resources
    assert stage.resources == resources_before
    assert facts == facts_before


def test_calibrate_wrapper_is_accepted_on_matching_machine() -> None:
    stage = _Stage(resources=Resources(cpus=1))
    env = EnvProfile(total_cpus=8, total_ram_gb=32)
    result = _plan(
        stage,
        env,
        card={"resource": {"cpus": 1.0, "host_mem_gb": 1.0}},
        calibration_facts={
            "machine_fingerprint": env.fingerprint(),
            "calibration": {
                "_Stage": {
                    "cpus": 2.5,
                    "host_mem_gb": 3.0,
                    "source": "measured",
                }
            },
        },
    )

    assert result.per_stage[0]["cpus"] == 2.5
    assert result.per_stage[0]["host_mem_gb"] == 3.0
    assert result.per_stage[0]["source"] == "measured"


def test_bounded_calibration_cannot_lower_card_resource_estimates() -> None:
    stage = _Stage(resources=Resources(cpus=1))
    env = EnvProfile(total_cpus=16, total_ram_gb=64, gpu_count=1, gpu_mem_gb=24)
    result = _plan(
        stage,
        env,
        card={
            "resource": {
                "cpus": 4.0,
                "host_mem_gb": 8.0,
                "gpu_mem_gb": 12.0,
            }
        },
        calibration_facts={
            "_Stage": {
                "cpus": 1.0,
                "host_mem_gb": 2.0,
                "gpu_mem_gb": 3.0,
                "source": "measured",
                "machine_fingerprint": env.fingerprint(),
            }
        },
    )

    assert result.per_stage[0]["cpus"] == 4.0
    assert result.per_stage[0]["host_mem_gb"] == 8.0
    assert result.per_stage[0]["gpu_mem_gb"] == 12.0
    assert result.per_stage[0]["source"] == "card"
    assert result.per_stage[0]["resource_sources"] == {
        "cpus": "card",
        "gpu_mem_gb": "card",
        "host_mem_gb": "card",
    }
    assert not any("using measured calibration" in note for note in result.notes)


def test_calibration_provenance_only_marks_resources_it_actually_raised() -> None:
    stage = _Stage(resources=Resources(cpus=1, gpus=1))
    env = EnvProfile(
        total_cpus=16,
        total_ram_gb=64,
        gpu_count=1,
        gpu_mem_gb=24,
    )
    result = _plan(
        stage,
        env,
        card={
            "resource": {
                "cpus": 4.0,
                "host_mem_gb": 8.0,
                "gpu_mem_gb": 12.0,
                "gpu_optional": False,
            }
        },
        calibration_facts={
            "_Stage": {
                "cpus": 6.0,
                "host_mem_gb": 2.0,
                "gpu_mem_gb": 12.0,
                "source": "measured",
                "machine_fingerprint": env.fingerprint(),
            }
        },
    )

    assert result.per_stage[0]["source"] == "measured"
    assert result.per_stage[0]["resource_sources"] == {
        "cpus": "measured",
        "gpu_mem_gb": "card",
        "host_mem_gb": "card",
    }
    assert any("using measured calibration for 1 stage" in note for note in result.notes)


def test_unknown_vram_is_advisory_until_smoke_measures_the_real_fit() -> None:
    stage = _Stage(resources=Resources(cpus=1, gpus=1))
    result = _plan(
        stage,
        EnvProfile(
            total_cpus=8,
            total_ram_gb=64,
            gpu_count=1,
            gpu_mem_gb=0,
        ),
        card={
            "resource": {
                "cpus": 1.0,
                "host_mem_gb": 1.0,
                "gpu_mem_gb": 8.0,
                "gpu_optional": False,
            }
        },
    )

    assert result.mode == "streaming"
    assert result.feasible is True
    assert result.estimate["gpu_mem_known"] is False
    assert any("GPU VRAM for this machine is unknown" in item for item in result.notes)
    assert not any("VRAM" in item for item in result.escalations)


def test_unknown_vram_does_not_block_gpu_optional_cpu_execution() -> None:
    stage = _Stage(resources=Resources(cpus=1, gpus=0))
    result = _plan(
        stage,
        EnvProfile(
            total_cpus=8,
            total_ram_gb=64,
            gpu_count=1,
            gpu_mem_gb=0,
        ),
        card={
            "resource": {
                "cpus": 1.0,
                "host_mem_gb": 1.0,
                "gpu_mem_gb": 8.0,
                "gpu_optional": True,
            }
        },
    )

    assert result.mode == "streaming"
    assert result.feasible is True
    assert result.estimate["sum_gpu_mem_gb"] == 0


def test_mismatched_machine_calibration_is_ignored() -> None:
    stage = _Stage(resources=Resources(cpus=1))
    result = _plan(
        stage,
        EnvProfile(total_cpus=8, total_ram_gb=32),
        card={"resource": {"cpus": 1.0, "host_mem_gb": 1.0}},
        calibration_facts={
            "_Stage": {
                "cpus": 7.0,
                "source": "measured",
                "machine_fingerprint": "another-machine",
            }
        },
    )

    assert result.per_stage[0]["cpus"] == 1.0
    assert result.per_stage[0]["source"] == "card"
    assert any("does not match this machine" in note for note in result.notes)


def test_nonfinite_and_negative_calibration_values_are_ignored() -> None:
    stage = _Stage(resources=Resources(cpus=1))
    result = _plan(
        stage,
        EnvProfile(total_cpus=8, total_ram_gb=32),
        card={"resource": {"cpus": 1.0, "host_mem_gb": 1.0}},
        calibration_facts={
            "_Stage": {
                "cpus": float("inf"),
                "host_mem_gb": -1,
                "source": "measured",
            }
        },
    )

    assert result.per_stage[0]["cpus"] == 1.0
    assert result.per_stage[0]["host_mem_gb"] == 1.0
    assert result.per_stage[0]["source"] == "card"
    assert sum("finite non-negative" in note for note in result.notes) == 2


def test_perf_aggregates_include_extrema_and_calibration_uses_peak() -> None:
    first = StagePerfStats(
        stage_name="scorer",
        custom_metrics={"peak_vram_gb": 1.0, "throughput": 2.0},
    )
    second = StagePerfStats(
        stage_name="scorer",
        custom_metrics={"peak_vram_gb": 3.0, "throughput": 4.0},
    )
    metrics = _dedup_stage_perf(
        [
            SimpleNamespace(_stage_perf=[first]),
            SimpleNamespace(_stage_perf=[second]),
        ]
    )

    assert metrics["scorer"]["custom.peak_vram_gb"] == {
        "sum": 4.0,
        "mean": 2.0,
        "min": 1.0,
        "max": 3.0,
        "count": 2,
    }
    measured = calibration.from_smoke({"per_stage_metrics": metrics})
    assert measured["scorer"]["gpu_mem_gb"] == 3.0
    assert measured["scorer"]["throughput"] == 3.0


def test_runtime_resource_probe_produces_host_memory_and_throughput() -> None:
    metrics = resource_probe_metrics(
        gpu_probe_started=False,
        process_time=2.0,
        num_items=5,
    )

    assert metrics["throughput"] == 2.5
    assert metrics["peak_host_mem_gb"] > 0
    assert "peak_vram_gb" not in metrics


def test_calibration_extraction_ignores_invalid_measurements() -> None:
    measured = calibration.from_smoke(
        {
            "per_stage_metrics": {
                "scorer": {
                    "peak_vram_gb": {"max": float("nan")},
                    "peak_host_mem_gb": {"max": -1.0},
                    "throughput": {"mean": float("inf")},
                }
            }
        }
    )

    assert measured == {}


def test_recalibrating_a_saved_smoke_preserves_its_machine_fingerprint() -> None:
    measured = calibration.from_smoke(
        {
            "per_stage_metrics": {
                "scorer": {
                    "peak_vram_gb": {"max": 3.0},
                }
            },
            "calibration": {
                "prior-stage": {
                    "gpu_mem_gb": 1.0,
                    "source": "measured",
                    "machine_fingerprint": "machine-A",
                }
            },
        }
    )

    assert measured["scorer"]["machine_fingerprint"] == "machine-A"


def test_env_fingerprint_changes_with_total_ram() -> None:
    base = EnvProfile(total_cpus=8, total_ram_gb=32)
    more_ram = EnvProfile(total_cpus=8, total_ram_gb=64)

    assert base.fingerprint() != more_ram.fingerprint()


def _pair_plan(env: EnvProfile, calibration_facts: dict[str, Any] | None = None):  # noqa: ANN202
    """Two identical stages whose card declares no host-RAM fact (so the 1 GB floor applies)."""
    stages = [
        _Stage(resources=Resources(cpus=1.0), name="stage-a"),
        _Stage(resources=Resources(cpus=1.0), name="stage-b"),
    ]
    return planner.plan(
        stages,
        [_contract(), _contract()],
        env,
        index=_Index({"_Stage": {"resource": {"cpus": 1.0}}}),
        calibration=calibration_facts,
    )


class TestMeasurementsReachTheRunThatNeedsThem:
    """A smoke measures what each stage really used; the next run has to plan with it.

    That hand-off used to depend on the caller passing ``--calibration``, and nothing could
    warn when they didn't: a run with no measurements is legitimate, so a forgotten flag was
    indistinguishable from having nothing to apply. The planner then fell back to a 1 GB
    per-stage floor -- the number that decides streaming vs batch.
    """

    @pytest.fixture(autouse=True)
    def _isolated_store(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
        monkeypatch.setenv("AUDIO_AGENT_RUNS_DIR", str(tmp_path / "runs"))

    def test_a_run_with_no_flag_plans_from_the_stored_measurements(self) -> None:
        calibration_store.save("cfg-1", {"stage-a": {"host_mem_gb": 9.0, "source": "measured"}})

        resolved, note = verbs._calibration_for_run(None, "cfg-1")

        assert resolved["calibration"]["stage-a"]["host_mem_gb"] == 9.0
        assert "none passed" in note

    def test_what_the_caller_passed_is_never_overridden(self) -> None:
        calibration_store.save("cfg-1", {"stage-a": {"host_mem_gb": 9.0}})
        explicit = {"calibration": {"stage-a": {"host_mem_gb": 2.0}}}

        resolved, note = verbs._calibration_for_run(explicit, "cfg-1")

        assert resolved is explicit
        assert note is None

    def test_editing_the_recipe_does_not_inherit_the_old_measurements(self) -> None:
        calibration_store.save("cfg-1", {"stage-a": {"host_mem_gb": 9.0}})

        assert verbs._calibration_for_run(None, "cfg-2") == (None, None)

    def test_stored_measurements_can_flip_streaming_to_batch(self) -> None:
        # The whole point of the hand-off: 2 x 1 GB of card floor fits a 16 GB box and plans
        # every stage concurrently; 2 x 9 GB of measured truth does not and has to serialize.
        env = EnvProfile(total_cpus=8, total_ram_gb=16)
        calibration_store.save(
            "cfg-1",
            {
                "stage-a": {"host_mem_gb": 9.0, "source": "measured"},
                "stage-b": {"host_mem_gb": 9.0, "source": "measured"},
            },
            machine_fingerprint=env.fingerprint(),
        )
        resolved, _note = verbs._calibration_for_run(None, "cfg-1")

        assert _pair_plan(env).mode == "streaming"
        informed = _pair_plan(env, resolved)
        assert informed.mode == "batch"
        assert [s["host_mem_gb"] for s in informed.per_stage] == [9.0, 9.0]
        assert {s["source"] for s in informed.per_stage} == {"measured"}

    def test_measurements_carried_from_another_machine_are_dropped(self) -> None:
        env = EnvProfile(total_cpus=8, total_ram_gb=16)
        calibration_store.save(
            "cfg-1",
            {"stage-a": {"host_mem_gb": 9.0}, "stage-b": {"host_mem_gb": 9.0}},
            machine_fingerprint="a-much-larger-box",
        )
        resolved, _note = verbs._calibration_for_run(None, "cfg-1")

        result = _pair_plan(env, resolved)

        assert result.mode == "streaming"  # planned from card facts, not a foreign machine's
        assert any("machine_fingerprint does not match" in n for n in result.notes)


class TestTheMeasurementStoreIsHardened:
    @pytest.fixture(autouse=True)
    def _isolated_store(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
        monkeypatch.setenv("AUDIO_AGENT_RUNS_DIR", str(tmp_path / "runs"))

    def test_it_follows_the_configured_run_directory(self, tmp_path) -> None:  # noqa: ANN001
        assert calibration_store.store_dir() == str(tmp_path / "runs" / "calibration")

    def test_a_half_written_record_plans_from_card_facts_rather_than_half_a_measurement(self) -> None:
        path = calibration_store.save("cfg-1", {"stage-a": {"host_mem_gb": 9.0}})
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"calibration": {"stage-a"')

        assert calibration_store.load("cfg-1") is None

    def test_a_record_with_no_measurements_is_not_treated_as_one(self) -> None:
        assert calibration_store.save("cfg-1", {}) is None
        assert calibration_store.load("cfg-1") is None

    def test_a_key_that_is_not_filename_shaped_is_refused(self) -> None:
        # The key comes from a caller-supplied recipe, so it never reaches a path unchecked.
        assert calibration_store.path_for("../../etc/passwd") is None
        assert calibration_store.save("../../etc/passwd", {"stage-a": {"host_mem_gb": 1.0}}) is None
        assert calibration_store.load("../../etc/passwd") is None

    def test_records_are_no_more_readable_than_the_run_records_beside_them(self) -> None:
        # Stage names and machine shape are the same class of local history as a run record.
        path = calibration_store.save("cfg-1", {"stage-a": {"host_mem_gb": 9.0}})

        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600

    def test_what_is_written_is_what_calibrate_returns(self) -> None:
        path = calibration_store.save(
            "cfg-1",
            {"stage-a": {"host_mem_gb": 9.0}},
            machine_fingerprint="machine-A",
        )
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)

        # planner._calibration_mapping unwraps exactly this envelope, so a stored calibration
        # and one passed by hand are the same thing to the planner.
        assert payload["calibration"] == {"stage-a": {"host_mem_gb": 9.0}}
        assert payload["machine_fingerprint"] == "machine-A"
        assert payload["config_hash"] == "cfg-1"
        assert payload["created_at"].endswith("Z")

    @pytest.mark.skipif(getattr(os, "geteuid", lambda: 1)() == 0, reason="root ignores mode bits")
    def test_a_store_it_cannot_write_never_fails_the_smoke_that_produced_it(self) -> None:
        os.makedirs(calibration_store.store_dir(), mode=0o500)

        assert calibration_store.save("cfg-1", {"stage-a": {"host_mem_gb": 9.0}}) is None

    def test_a_read_only_filesystem_is_reported_rather_than_raised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The permission test above cannot run as root, and this path must hold everywhere:
        # storing telemetry is never worth failing the smoke that produced it.
        def _refuse(*_args: Any, **_kwargs: Any) -> None:  # noqa: ANN401
            msg = "Read-only file system"
            raise OSError(msg)

        monkeypatch.setattr(calibration_store, "_write_private_json", _refuse)

        assert calibration_store.save("cfg-1", {"stage-a": {"host_mem_gb": 9.0}}) is None
