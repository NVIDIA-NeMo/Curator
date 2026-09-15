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

"""Env Doctor: the checks, the aggregate verdict, and the failures they explain.

See ``nemo_curator/audio_agent/ENVIRONMENT.md`` for what each check means.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

import pytest

from nemo_curator.audio_agent import env_health as eh
from nemo_curator.audio_agent.contracts import EnvProfile
from nemo_curator.audio_agent.failures import classify

_Ancestors = Callable[[list[list[str] | None]], None]


@pytest.fixture
def fake_ancestors(monkeypatch: pytest.MonkeyPatch) -> _Ancestors:
    """Stand in for the process tree: each entry is an ancestor's cmdline, nearest first.

    ``None`` makes that ancestor raise, the way a process that exits mid-walk does.
    ``psutil`` is imported inside the check, so ``sys.modules`` is the seam.
    """

    def install(cmdlines: list[list[str] | None]) -> None:
        class _Proc:
            def __init__(self, cmdline: list[str] | None) -> None:
                self._cmdline = cmdline

            def cmdline(self) -> list[str]:
                if self._cmdline is None:
                    msg = "process is gone"
                    raise RuntimeError(msg)
                return self._cmdline

            def parents(self) -> list[_Proc]:
                return [_Proc(c) for c in cmdlines]

        monkeypatch.setenv("UV_RUN_RECURSION_DEPTH", "1")
        monkeypatch.setitem(sys.modules, "psutil", type("psutil", (), {"Process": lambda: _Proc(None)}))

    return install


class TestWorkerEnv:
    """Will a Ray WORKER import what the driver just proved it can?

    Every other check probes the driver, so this is the one that catches an environment
    that is perfect here and empty where the work actually happens.
    """

    def test_a_direct_launch_is_fine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(eh, "_uv_run_cmdline", lambda: None)
        check = eh._worker_env(EnvProfile())
        assert check.status == "ok"
        assert "inherit" in check.finding

    def test_uv_run_without_an_extra_is_blocking(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(eh, "_uv_run_cmdline", lambda: ["uv", "run", "python", "-m", "nemo_curator.audio_agent"])
        check = eh._worker_env(EnvProfile())
        assert check.status == "fail"
        # The point of the check is that reinstalling fixes nothing, so say what does.
        assert any(".venv/bin/python" in f for f in check.fix)
        assert any("--extra" in f for f in check.fix)
        assert "NOT a missing install" in check.impact

    @pytest.mark.parametrize(
        "dependency_args",
        [
            ["--extra", "audio_cuda12"],
            ["--extra=audio_cpu"],
            ["--all-extras"],
            ["--with", "nemo_curator[audio_cuda12]"],
        ],
    )
    def test_uv_run_carrying_audio_dependencies_is_fine(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dependency_args: list[str],
    ) -> None:
        monkeypatch.setattr(
            eh,
            "_uv_run_cmdline",
            lambda: ["uv", "run", *dependency_args, "python", "-m", "x"],
        )
        assert eh._worker_env(EnvProfile()).status == "ok"

    @pytest.mark.parametrize(
        "dependency_args",
        [
            ["--with", "requests"],
            ["--group", "linting"],
            ["--all-groups"],
            ["--with-requirements", "requirements.txt"],
        ],
    )
    def test_arbitrary_uv_dependency_flags_are_unverified(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dependency_args: list[str],
    ) -> None:
        monkeypatch.setattr(
            eh,
            "_uv_run_cmdline",
            lambda: ["uv", "run", *dependency_args, "python", "-m", "x"],
        )
        check = eh._worker_env(EnvProfile())
        assert check.status == "warn"
        assert check.confidence == "unknown"

    def test_non_audio_extra_does_not_claim_audio_workers_are_ready(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            eh,
            "_uv_run_cmdline",
            lambda: ["uv", "run", "--extra", "image_cpu", "python", "-m", "x"],
        )
        assert eh._worker_env(EnvProfile()).status == "fail"

    def test_not_launched_by_uv_needs_no_process_inspection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("UV_RUN_RECURSION_DEPTH", raising=False)
        assert eh._uv_run_cmdline() is None

    def test_uv_run_above_a_shell_is_still_found(self, fake_ancestors: _Ancestors) -> None:
        # `uv run bash -c "python -m ..."`: uv is the GRANDparent. Ray walks every ancestor,
        # so a check that only looked at the parent would call a broken launch healthy.
        fake_ancestors([["bash", "-c", "python -m x"], ["uv", "run", "bash"]])
        assert eh._uv_run_cmdline() == ["uv", "run", "bash"]

    def test_no_uv_ancestor_is_healthy_even_with_the_env_var(self, fake_ancestors: _Ancestors) -> None:
        # The env var is inherited by descendants, so it can outlive the `uv run` that set it.
        # Ray only rewrites the env when it finds the ancestor, so neither should we complain.
        fake_ancestors([["bash"], ["sshd"]])
        assert eh._uv_run_cmdline() is None

    def test_an_unreadable_ancestor_does_not_stop_the_walk(self, fake_ancestors: _Ancestors) -> None:
        fake_ancestors([None, ["uv", "run", "python"]])  # None => raises, as a dead process does
        assert eh._uv_run_cmdline() == ["uv", "run", "python"]


class TestVerdict:
    def test_the_worst_check_decides_the_overall_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            eh,
            "_CHECKS",
            [
                lambda _e: eh.HealthCheck("a", "ok", "fine"),
                lambda _e: eh.HealthCheck("b", "warn", "meh"),
                lambda _e: eh.HealthCheck("c", "fail", "broken"),
            ],
        )
        report = eh.env_report(EnvProfile())
        assert report.status == "fail"
        assert "1 blocking issue" in report.summary

    def test_a_healthy_machine_says_so(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(eh, "_CHECKS", [lambda _e: eh.HealthCheck("a", "ok", "fine")])
        assert eh.env_report(EnvProfile()).status == "ok"

    def test_rendering_shows_fixes_only_where_something_is_wrong(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            eh,
            "_CHECKS",
            [
                lambda _e: eh.HealthCheck("healthy", "ok", "fine", fix=["not shown"]),
                lambda _e: eh.HealthCheck("broken", "fail", "bad", impact="things die", fix=["do this"]),
            ],
        )
        text = eh.render_doctor(eh.env_report(EnvProfile()).to_dict())
        assert "do this" in text
        assert "not shown" not in text


class TestFailureAttribution:
    """A worker-side import error must not be blamed on a missing install."""

    def test_a_worker_side_import_error_points_at_the_launcher(self) -> None:
        raw = (
            "Node setup failed for stage Stage 02 - GetAudioDurationStage on node abc123: "
            "ModuleNotFoundError: No module named 'soundfile'"
        )
        assert classify(raw)["code"] == "worker_env_mismatch"

    def test_a_genuine_missing_install_still_says_install_it(self) -> None:
        assert classify("ModuleNotFoundError: No module named 'silero_vad'")["code"] == "missing_dependency"


class TestGpuDoctorIsMaskAware:
    """doctor is the skill's 'single source of truth for env', so it must not tell a
    sandboxed-but-GPU-capable user to repair a healthy driver. A masked GPU leads with
    're-verify with full device access', matching the recipe preflight (diagnostics.py).
    """

    def _masked(self) -> EnvProfile:
        return EnvProfile(
            has_gpu=False,
            gpu_visibility="torch_cuda_unavailable",  # devices visible, torch can't init
            gpu_possibly_masked=True,
            torch_cuda_built=True,
            nvidia_device_nodes=1,
        )

    def test_masked_gpu_leads_with_reverify_not_driver_repair(self) -> None:
        check = eh._gpu(self._masked())
        assert check.status == "warn"  # never a hard fail on a possibly-present GPU
        assert check.options[0].id == "reverify_full_device_access"
        assert check.options[0].recommended is True
        assert "full device access" in check.options[0].label.lower()
        assert "sandbox" in check.options[0].summary.lower()
        # no other option is recommended -- re-verify is THE action
        assert not any(o.recommended for o in check.options if o.id != "reverify_full_device_access")
        # the finding no longer reads as a hardware/driver fault
        assert "not reachable" in check.finding.lower()

    def test_cpu_only_torch_is_not_treated_as_masked(self) -> None:
        env = EnvProfile(has_gpu=False, gpu_visibility="cpu_only_torch", gpu_possibly_masked=False)
        check = eh._gpu(env)
        assert check.status == "warn"
        assert all(o.id != "reverify_full_device_access" for o in check.options)

    def test_available_gpu_is_ok(self) -> None:
        env = EnvProfile(has_gpu=True, gpu_count=1, gpu_names=["RTX 4090"], gpu_mem_gb=24.0)
        assert eh._gpu(env).status == "ok"
