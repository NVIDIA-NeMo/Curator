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

"""Environment health -- ONE place that checks the machine and tells the user how to fix it.

This is the single source of truth for *environment* concerns (interpreter, GPU driver vs the
CUDA toolkit torch was built with, ffmpeg, audio extras, disk). Capability cards and the failure
taxonomy describe *stage*/error specifics and defer here for setup remediation, instead of each
restating env fixes.

The design is a small **check registry**: each check reads the already-probed ``EnvProfile`` and
returns a self-describing ``HealthCheck`` (status + finding + impact + concrete fix steps). Adding
a new environment concern is one function with an ``@_check`` decorator -- generic, so this is
reusable for any env-related use case, not just audio.

Severity:
  * ``fail`` -- broken/misconfigured; a common GPU workload WILL fail here (e.g. driver/toolkit
    mismatch with a GPU present, torch unusable).
  * ``warn`` -- a limitation or missing-optional (no GPU -> CPU-only, unsupported interpreter,
    missing extra, low disk); light pipelines still run.
  * ``ok``   -- healthy.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import asdict, dataclass, field, fields, replace
from typing import TYPE_CHECKING, Any, Literal

from nemo_curator.audio_agent.profiler import probe_env

if TYPE_CHECKING:
    from collections.abc import Callable

    from nemo_curator.audio_agent.contracts import EnvProfile

Status = Literal["ok", "warn", "fail"]
OptionKind = Literal[
    "host_change",
    "environment_change",
    "launch_change",
    "recipe_variant",
    "credential",
    "diagnostic",
]
Availability = Literal["available", "conditional", "unavailable", "unknown"]
_RANK: dict[str, int] = {"ok": 0, "warn": 1, "fail": 2}
_DOCS = "nemo_curator/audio_agent/ENVIRONMENT.md"
# The project's own audio install guidance is versioned and is re-published with each
# release. Point at it instead of restating a command here, which would silently go
# stale; we only decide the SHAPE of the command from the detected install mode below.
_AUDIO_SETUP_DOC = "https://docs.nvidia.com/nemo/curator/get-started/audio"


def _from_source_checkout() -> bool:
    """Whether this package is running from a source checkout rather than an install.

    ``uv sync`` resolves a project's ``pyproject``/lock, so prescribing it to someone who
    installed the published package points them at a command that cannot run in their
    environment. Resolved from the package location (never the CWD), so it is correct
    regardless of where the caller happens to be.
    """
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.isfile(os.path.join(root, "pyproject.toml"))


def _dependency_profile_steps(extra: str) -> list[str]:
    """Steps to install/restore an audio dependency profile, matched to the install mode.

    Deliberately does not spell out the published install command (it carries release
    specifics such as an override file); it names the documented source of truth instead.
    """
    if _from_source_checkout():
        return [f"run `uv sync --extra {extra}`", "then launch that environment's interpreter directly"]
    return [
        f"reinstall the published package with the `{extra}` extra",
        f"use the exact current command from {_AUDIO_SETUP_DOC}",
    ]


_LOW_DISK_GB = 5.0  # below this, model downloads and intermediates start failing mid-run


@dataclass(frozen=True)
class RemediationOption:
    """One grounded approach the host LLM may explain and offer to the user.

    This is deliberately a proposal, never an executable action. Environment,
    host, credential, launch, and recipe changes all require an explicit user
    decision and fresh verification afterward.
    """

    id: str
    kind: OptionKind
    label: str
    summary: str
    steps: list[str] = field(default_factory=list)
    availability: Availability = "available"
    reason: str = ""
    preserves_recipe: bool = True
    tradeoffs: list[str] = field(default_factory=list)
    requires_confirmation: bool = True
    verify: list[str] = field(default_factory=lambda: ["rerun `doctor --json` / recipe preflight"])
    recommended: bool = False
    applies_to: list[str] = field(default_factory=list)
    resolves: list[str] = field(default_factory=list)
    remaining_blockers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _option(  # noqa: PLR0913 - options deliberately expose each user-decision attribute
    option_id: str,
    kind: OptionKind,
    label: str,
    summary: str,
    *,
    steps: list[str],
    availability: Availability = "available",
    reason: str = "",
    preserves_recipe: bool = True,
    tradeoffs: list[str] | None = None,
    recommended: bool = False,
    applies_to: list[str] | None = None,
) -> RemediationOption:
    return RemediationOption(
        id=option_id,
        kind=kind,
        label=label,
        summary=summary,
        steps=steps,
        availability=availability,
        reason=reason,
        preserves_recipe=preserves_recipe,
        tradeoffs=list(tradeoffs or []),
        recommended=recommended,
        applies_to=list(applies_to or []),
    )


@dataclass
class HealthCheck:
    """One environment concern: what was found, why it matters, and how to fix it."""

    id: str
    status: Status
    finding: str
    impact: str = ""
    fix: list[str] = field(default_factory=list)
    docs: str = _DOCS
    confidence: Literal["high", "medium", "low", "unknown"] = "high"
    capabilities: list[str] = field(default_factory=list)
    options: list[RemediationOption] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EnvHealthReport:
    """Aggregated machine health: overall status + per-check detail + the raw probe facts."""

    status: Status
    summary: str
    checks: list[HealthCheck]
    env: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        actionable = [c for c in self.checks if c.status != "ok"]
        recommended = [
            option.id
            for check in actionable
            for option in check.options
            if option.recommended and option.availability != "unavailable"
        ]
        return {
            "status": self.status,
            "summary": self.summary,
            "checks": [c.to_dict() for c in self.checks],
            "env": self.env,
            "decision": {
                # Machine-wide doctor cannot know whether a warning applies to a
                # future recipe. Recipe preflight narrows it before execution.
                "state": (
                    "action_required" if self.status == "fail" else "review" if self.status == "warn" else "ready"
                ),
                "prompt_user": self.status == "fail",
                "recommended": recommended[0] if recommended else None,
                "host_directive": (
                    "Explain detected facts and their impact on the selected recipe; "
                    "recommend only grounded, applicable options; ask before any host, "
                    "environment, launch, credential, device, or recipe change; never "
                    "switch to CPU or apply a fix silently; rerun preflight after the choice."
                ),
            },
        }


_CHECKS: list[Callable[[EnvProfile], HealthCheck]] = []


def _check(fn: Callable[[EnvProfile], HealthCheck]) -> Callable[[EnvProfile], HealthCheck]:
    """Register an env health check (registration order = display order)."""
    _CHECKS.append(fn)
    return fn


# --------------------------------------------------------------------------- checks


@_check
def _python(env: EnvProfile) -> HealthCheck:
    if env.python_supported:
        return HealthCheck("python", "ok", f"Python {env.python_version} satisfies the project's requires-python")
    return HealthCheck(
        "python",
        "warn",
        f"Python {env.python_version} is outside the project's supported range",
        impact="imports or heavy model stages may misbehave on an untested interpreter",
        fix=["create the venv with a supported interpreter (see the project's requires-python)"],
        capabilities=["all"],
        options=[
            _option(
                "recreate_supported_python_env",
                "environment_change",
                "Use a supported Python environment",
                "Recreate the project environment with a Python version allowed by requires-python.",
                steps=[
                    "inspect the project's requires-python range",
                    "recreate/sync the project virtual environment with a supported interpreter",
                ],
                tradeoffs=["changes the project environment; cached wheels/models may need to be restored"],
                recommended=True,
            )
        ],
    )


@_check
def _gpu(env: EnvProfile) -> HealthCheck:
    if env.has_gpu:
        names = ", ".join(env.gpu_names) or "GPU"
        return HealthCheck("gpu", "ok", f"{env.gpu_count}x {names} ({env.gpu_mem_gb} GB/GPU)")
    visibility = str(getattr(env, "gpu_visibility", "unknown") or "unknown")
    masked = bool(getattr(env, "gpu_possibly_masked", False))
    findings = {
        "cpu_only_torch": "torch is not built with CUDA support",
        "masked_by_cuda_visible_devices": "GPU access is masked by CUDA_VISIBLE_DEVICES",
        "torch_cuda_unavailable": "NVIDIA devices are visible, but torch CUDA initialization is unavailable",
        "driver_or_device_exposure_error": "the NVIDIA driver/device is not usable from this process",
        "torch_unavailable": "torch could not be imported",
        "not_detected": "no usable GPU was detected from this process",
    }
    finding = findings.get(visibility, "GPU availability could not be established")
    options: list[RemediationOption] = []
    if visibility == "masked_by_cuda_visible_devices":
        options.append(
            _option(
                "request_or_expose_gpu",
                "host_change",
                "Expose an allocated GPU",
                "Run in a container/scheduler allocation that exposes the NVIDIA device.",
                steps=[
                    "verify the process has a GPU allocation/device mapping",
                    "correct CUDA_VISIBLE_DEVICES or the container/scheduler GPU request",
                ],
                recommended=True,
            )
        )
    elif visibility in {"driver_or_device_exposure_error", "torch_cuda_unavailable"}:
        options.extend(
            [
                _option(
                    "repair_or_upgrade_nvidia_driver",
                    "host_change",
                    "Repair the host GPU path",
                    "Make the NVIDIA driver and device nodes usable by this process.",
                    steps=[
                        "check `nvidia-smi -L` on the execution host",
                        "repair/upgrade the NVIDIA driver or expose `/dev/nvidia*` to the workload",
                    ],
                    recommended=True,
                ),
                _option(
                    "request_or_expose_gpu",
                    "host_change",
                    "Use a GPU-enabled allocation",
                    "Move the workload to a scheduler/container allocation with GPU device access.",
                    steps=["request a GPU resource and verify it is visible inside the workload"],
                ),
            ]
        )
    elif visibility in {"cpu_only_torch", "torch_unavailable"}:
        hardware_visible = env.nvidia_smi_status == "ok" or env.nvidia_smi_gpu_count > 0 or env.nvidia_device_nodes > 0
        if not hardware_visible:
            options.append(
                _option(
                    "use_gpu_host",
                    "host_change",
                    "Use a GPU-enabled host or allocation",
                    "The current process has no hardware/device evidence, so a CUDA environment alone is insufficient.",
                    steps=[
                        "request or select a supported GPU host/allocation",
                        "verify device exposure there before choosing a GPU dependency profile",
                    ],
                    recommended=True,
                )
            )
        options.append(
            _option(
                "sync_gpu_project_environment",
                "environment_change",
                "Provide the GPU dependency profile",
                "Install/sync the GPU audio dependency profile (`audio_cuda12`) for this environment.",
                steps=_dependency_profile_steps("audio_cuda12"),
                availability="available" if hardware_visible else "conditional",
                reason=(
                    ""
                    if hardware_visible
                    else "first establish that a supported NVIDIA device is allocated and exposed"
                ),
                tradeoffs=["changes the project environment"],
                recommended=hardware_visible,
            )
        )
    else:
        options.append(
            _option(
                "use_gpu_host",
                "host_change",
                "Use a GPU host",
                "Run the same project environment on a machine/allocation with a supported NVIDIA GPU.",
                steps=["request or select a GPU host", "rerun the environment preflight there"],
                recommended=True,
            )
        )
    options.append(
        _option(
            "use_cpu_recipe_variant",
            "recipe_variant",
            "Use a CPU-compatible recipe",
            "Create a new recipe variant only if every selected GPU stage explicitly supports CPU.",
            steps=["inspect recipe-specific CPU feasibility", "build and validate a new CPU recipe"],
            availability="conditional",
            reason="doctor is machine-wide; recipe preflight must prove every selected stage supports CPU",
            preserves_recipe=False,
            tradeoffs=["often much slower; some model stages have no supported CPU path"],
        )
    )
    if masked:
        # A GPU is likely PRESENT but unreachable from THIS process (a sandbox/container
        # blocking /dev/nvidia*, or CUDA_VISIBLE_DEVICES masking). The correct FIRST action
        # is to re-probe with full device access -- NOT to repair a healthy driver or move
        # hosts. Lead with re-verify and demote the host/env options so doctor agrees with
        # the recipe preflight (diagnostics.py) and the skill, instead of reintroducing the
        # false "no GPU / fix your machine" story on a sandboxed but GPU-capable box.
        options = [replace(option, recommended=False) for option in options]
        options.insert(
            0,
            _option(
                "reverify_full_device_access",
                "diagnostic",
                "Re-verify the GPU with full device access",
                "Re-run doctor (and any smoke/run) OUTSIDE the sandbox/container so the process "
                "can open /dev/nvidia*. Only conclude 'no GPU' if a full-access probe still finds none.",
                steps=[
                    "re-run `.venv/bin/python -m nemo_curator.audio_agent doctor --json` with full device access",
                    "run smoke/run with full device access (they already require it for GPU stages)",
                ],
                recommended=True,
            ),
        )
        finding = (
            "a GPU is likely present but not reachable from this process "
            f"({finding}) -- probably a sandbox/container masking device access, not a "
            "hardware/driver fault; re-verify with full device access before concluding no GPU"
        )
    return HealthCheck(
        "gpu",
        "warn",
        f"{finding} (visibility={visibility})",
        impact="GPU model stages (ASR, diarization, quality metrics) are very slow or unusable",
        fix=[
            "inspect the layered GPU evidence below (torch build, driver, device exposure, CUDA_VISIBLE_DEVICES)",
            "run on/fix a GPU host, or use a recipe whose selected stages explicitly support CPU",
        ],
        capabilities=["gpu"],
        options=options,
    )


@_check
def _cuda_driver_toolkit(env: EnvProfile) -> HealthCheck:
    if not env.has_gpu:
        return HealthCheck("cuda_driver_toolkit", "ok", "no GPU; CUDA driver/toolkit check not applicable")
    if not env.cuda_runtime_version or not env.cuda_driver_max_version:
        return HealthCheck(
            "cuda_driver_toolkit",
            "warn",
            f"could not determine CUDA versions (torch={env.cuda_runtime_version or '?'}, "
            f"driver_max={env.cuda_driver_max_version or '?'})",
            impact="cannot confirm the driver can run the CUDA toolkit torch was built with",
            fix=["check `nvidia-smi` (driver CUDA) and `python -c 'import torch;print(torch.version.cuda)'`"],
            confidence="unknown",
            capabilities=["gpu"],
            options=[
                _option(
                    "inspect_cuda_versions",
                    "diagnostic",
                    "Verify driver and PyTorch CUDA versions",
                    "Collect the two version facts needed to decide compatibility.",
                    steps=[
                        "run `nvidia-smi` on the execution host",
                        "run `.venv/bin/python -c 'import torch; print(torch.version.cuda)'`",
                    ],
                    recommended=True,
                )
            ],
        )
    if env.cuda_compatible:
        return HealthCheck(
            "cuda_driver_toolkit",
            "ok",
            f"GPU driver supports CUDA {env.cuda_driver_max_version} >= torch's CUDA {env.cuda_runtime_version}",
        )
    return HealthCheck(
        "cuda_driver_toolkit",
        "fail",
        (
            f"driver supports only CUDA {env.cuda_driver_max_version} "
            f"but torch is built for CUDA {env.cuda_runtime_version}"
        ),
        impact=(
            "basic GPU ops work, but runtime-JIT / CUDA-graph kernels (e.g. NeMo RNNT/TDT ASR decode) fail with "
            "CUDA_ERROR_UNSUPPORTED_PTX_VERSION (error 222) -- this is a driver<->toolkit mismatch, NOT a Python issue"
        ),
        fix=[
            f"upgrade the NVIDIA driver to one supporting CUDA >= {env.cuda_runtime_version}",
            f"or install a torch built for CUDA <= {env.cuda_driver_max_version} (e.g. a matching +cuXXX wheel)",
            "or, for ASR alignment only, set decoder_type='ctc' (hybrid tdt_ctc checkpoint, no CUDA graphs)",
        ],
        capabilities=["gpu"],
        options=[
            _option(
                "upgrade_nvidia_driver",
                "host_change",
                "Upgrade the host NVIDIA driver",
                (
                    f"Use a driver supporting CUDA >= {env.cuda_runtime_version} while "
                    "keeping the repository environment and recipe unchanged."
                ),
                steps=[
                    f"upgrade the execution host driver to support CUDA >= {env.cuda_runtime_version}",
                    "reboot/reload the driver if the host procedure requires it",
                ],
                tradeoffs=["requires host administration and may affect other GPU workloads"],
                recommended=True,
            ),
            _option(
                "use_driver_compatible_torch",
                "environment_change",
                "Use a driver-compatible PyTorch build",
                (f"Use a supported project environment whose PyTorch CUDA is <= {env.cuda_driver_max_version}."),
                steps=[
                    "confirm the repository supports that CUDA/PyTorch variant",
                    "resolve and sync a separate compatible environment",
                ],
                availability="conditional",
                reason="the repository dependency lock may intentionally pin a newer CUDA build",
                tradeoffs=["changes the project environment and may diverge from the lock/tutorial baseline"],
            ),
            _option(
                "use_ctc_alignment_variant",
                "recipe_variant",
                "Use a CTC alignment variant",
                "Avoid the CUDA-graph decoder only for a selected hybrid NeMo alignment stage that supports CTC.",
                steps=["recipe preflight must prove the selected alignment stage/checkpoint supports CTC"],
                availability="conditional",
                reason="not valid for arbitrary GPU stages or pure-TDT transcription",
                preserves_recipe=False,
                tradeoffs=["changes recipe behavior and requires a new validation/smoke/confirmation cycle"],
                applies_to=["NeMoASRAlignerStage", "SplitASRAlignJoinStage"],
            ),
            _option(
                "use_cpu_recipe_variant",
                "recipe_variant",
                "Use a CPU-compatible recipe",
                "Create a new CPU variant only when every affected stage explicitly supports CPU.",
                steps=["recipe preflight must prove all affected execution leaves are CPU-capable"],
                availability="conditional",
                reason="many ASR/diarization stages are GPU-only",
                preserves_recipe=False,
                tradeoffs=["can be substantially slower and may require different stage choices"],
            ),
        ],
    )


@_check
def _ffmpeg(env: EnvProfile) -> HealthCheck:
    if env.has_ffmpeg:
        return HealthCheck("ffmpeg", "ok", "ffmpeg is on PATH")
    return HealthCheck(
        "ffmpeg",
        "warn",
        "ffmpeg not found on PATH",
        impact="resample/convert and compressed-format (mp3/opus/...) decode stages will fail",
        fix=[
            "install ffmpeg (e.g. `apt-get install ffmpeg`, "
            "`conda install -c conda-forge ffmpeg`, or `brew install ffmpeg`)"
        ],
        capabilities=["ffmpeg"],
        options=[
            _option(
                "install_ffmpeg",
                "host_change",
                "Install ffmpeg",
                "Install ffmpeg on the execution host and make it available on PATH.",
                steps=[
                    "use the host's approved package manager to install ffmpeg",
                    "verify with `ffmpeg -version` in the same launch environment",
                ],
                tradeoffs=["changes host software"],
                recommended=True,
            ),
            _option(
                "remove_ffmpeg_dependent_stages",
                "recipe_variant",
                "Use an ffmpeg-free recipe/input",
                "Replan only if the requested outcome and actual audio formats do not require ffmpeg.",
                steps=["inspect affected stages and source codecs", "create and validate a new recipe"],
                availability="conditional",
                reason="not possible when conversion/resampling or compressed decode is required",
                preserves_recipe=False,
            ),
        ],
    )


@_check
def _audio_extras(env: EnvProfile) -> HealthCheck:
    if not env.missing_packages:
        return HealthCheck("audio_extras", "ok", "audio dependency modules are discoverable")
    missing = ", ".join(env.missing_packages)
    return HealthCheck(
        "audio_extras",
        "warn",
        f"audio packages not discoverable: {missing}",
        impact="selected stages that depend on these may fail to import or initialize",
        fix=[
            "install an audio dependency profile: `audio_cuda12` (GPU) or `audio_cpu` (CPU)",
            f"current install commands: {_AUDIO_SETUP_DOC}",
        ],
        confidence="medium",
        capabilities=["audio_dependencies"],
        options=[
            _option(
                "sync_audio_cuda_extra",
                "environment_change",
                "Install the GPU audio dependencies",
                "Provide the GPU audio dependency profile (`audio_cuda12`).",
                steps=_dependency_profile_steps("audio_cuda12"),
                tradeoffs=["changes the environment this agent runs in"],
                recommended=bool(env.has_gpu),
            ),
            _option(
                "sync_audio_cpu_extra",
                "environment_change",
                "Install the CPU audio dependencies",
                "Provide the CPU audio dependency profile (`audio_cpu`).",
                steps=_dependency_profile_steps("audio_cpu"),
                tradeoffs=["changes the environment this agent runs in; GPU-only stages remain unavailable"],
                recommended=not bool(env.has_gpu),
            ),
        ],
    )


@_check
def _worker_env(_env: EnvProfile) -> HealthCheck:
    """Will a Ray WORKER have the same imports the driver just proved it has?

    Every other check probes this process. Pipelines execute in Ray workers, and Ray's uv
    integration rebuilds the worker environment by re-running the driver's ``uv run`` command
    line -- which resolves only the BASE dependency set unless the audio extra is named. The
    driver then imports soundfile/nemo happily while the worker dies on
    ``ModuleNotFoundError``, which reads like a broken install rather than a launch flag.
    """
    launcher = _uv_run_cmdline()
    if launcher is None:
        return HealthCheck("worker_env", "ok", "launched directly; Ray workers inherit this interpreter")
    dependency_state = _uv_audio_dependency_state(launcher)
    if dependency_state == "present":
        return HealthCheck(
            "worker_env",
            "ok",
            "launched via `uv run` carrying an audio extra; workers resolve it too",
        )
    if dependency_state == "unknown":
        return HealthCheck(
            "worker_env",
            "warn",
            (
                "launched via `uv run` with custom dependency flags, but they do "
                "not prove the audio extra is carried to Ray workers"
            ),
            impact=(
                "worker imports remain unverified; arbitrary --with/--group/"
                "--with-requirements values are not equivalent to an audio extra"
            ),
            confidence="unknown",
            capabilities=["ray"],
            options=[
                _option(
                    "inspect_uv_worker_dependencies",
                    "diagnostic",
                    "Verify the uv worker dependency set",
                    "Check whether the exact uv launch resolves the selected audio profile.",
                    steps=[
                        "inspect the uv flags/requirements used by the current launcher",
                        "run a bounded Ray worker import check",
                    ],
                    recommended=True,
                ),
                _option(
                    "launch_venv_directly",
                    "launch_change",
                    "Launch the project interpreter directly",
                    "Let Ray workers inherit the already-synced project environment.",
                    steps=["rerun with `.venv/bin/python -m nemo_curator.audio_agent ...`"],
                ),
            ],
        )
    return HealthCheck(
        "worker_env",
        "fail",
        "launched via `uv run` without an audio extra, so Ray workers will rebuild the env WITHOUT it",
        impact=(
            "the driver imports the audio stack fine but every worker fails with "
            "ModuleNotFoundError (soundfile/librosa/nemo), so runs die at 'Node setup failed for stage ...' "
            "-- a launch-flag problem, NOT a missing install"
        ),
        fix=[
            "launch the venv interpreter directly: `.venv/bin/python -m nemo_curator.audio_agent ...`",
            "or carry the extra through: `uv run --extra audio_cuda12 python -m nemo_curator.audio_agent ...`",
        ],
        capabilities=["ray"],
        options=[
            _option(
                "launch_venv_directly",
                "launch_change",
                "Launch the project interpreter directly",
                "Let Ray workers inherit the already-synced project environment.",
                steps=["rerun with `.venv/bin/python -m nemo_curator.audio_agent ...`"],
                recommended=True,
            ),
            _option(
                "carry_uv_audio_extra",
                "launch_change",
                "Carry the audio extra through uv run",
                "Include the selected audio dependency profile in the command Ray reconstructs.",
                steps=["rerun with `uv run --extra audio_cuda12 python -m nemo_curator.audio_agent ...`"],
                tradeoffs=["uv resolves the launch environment again"],
            ),
        ],
    )


_AUDIO_EXTRA_NAMES = frozenset({"audio_cpu", "audio_cuda12"})
_UV_AMBIGUOUS_DEPENDENCY_FLAGS = frozenset(
    {
        "--group",
        "--only-group",
        "--all-groups",
        "--with",
        "--with-editable",
        "--with-requirements",
    }
)


def _uv_audio_dependency_state(
    launcher: list[str],
) -> Literal["present", "absent", "unknown"]:
    """Whether a ``uv run`` command proves Ray will receive an audio extra."""
    ambiguous = False
    for index, argument in enumerate(launcher):
        flag, separator, inline_value = argument.partition("=")
        value = inline_value if separator else (launcher[index + 1] if index + 1 < len(launcher) else "")
        if flag == "--all-extras":
            return "present"
        if flag == "--extra":
            if value in _AUDIO_EXTRA_NAMES:
                return "present"
            continue
        if flag in {"--with", "--with-editable"}:
            if any(name in value for name in _AUDIO_EXTRA_NAMES):
                return "present"
            ambiguous = True
        elif flag in _UV_AMBIGUOUS_DEPENDENCY_FLAGS:
            ambiguous = True
    return "unknown" if ambiguous else "absent"


def _uv_run_cmdline() -> list[str] | None:
    """The nearest ancestor's ``uv run ...`` command line, or None if not launched that way.

    Deliberately mirrors ``ray._private.runtime_env.uv_runtime_env_hook``, because this check
    exists to predict exactly what that hook will do: it walks EVERY ancestor, not just the
    parent, since ``uv run`` can sit above a shell (``uv run bash -c "python -m ..."``).
    Checking only the parent there would report a clean launch as broken.

    Finding no ``uv run`` ancestor means Ray leaves the runtime env alone and workers inherit
    this interpreter -- so an inconclusive answer is ``None`` (healthy), never a false alarm.
    """
    if not os.environ.get("UV_RUN_RECURSION_DEPTH"):  # uv sets this for every `uv run`
        return None
    uv_run = 2  # a "uv run" cmdline needs at least the binary and the subcommand
    try:
        import psutil

        for parent in psutil.Process().parents():
            with contextlib.suppress(Exception):  # a parent can exit or be unreadable mid-walk
                cmdline = parent.cmdline()
                if len(cmdline) >= uv_run and os.path.basename(cmdline[0]) == "uv" and cmdline[1] == "run":
                    return cmdline
    except Exception:  # noqa: BLE001 - without psutil we cannot tell; stay silent, the
        return None  # worker_env_mismatch failure code still explains it if a run does break
    return None


@_check
def _disk(env: EnvProfile) -> HealthCheck:
    free = env.free_disk_gb
    if free is None:
        return HealthCheck(
            "disk",
            "warn",
            "free disk capacity could not be determined",
            impact="cannot confirm model caches, temporary files, and outputs will fit",
            confidence="unknown",
            capabilities=["disk_write", "model_download"],
            options=[
                _option(
                    "inspect_disk_capacity",
                    "diagnostic",
                    "Check writable-volume capacity",
                    "Measure free space on the actual output, cache, and temporary filesystems.",
                    steps=["inspect capacity for output paths, model cache, and Ray/tmp directories"],
                    recommended=True,
                )
            ],
        )
    if free >= _LOW_DISK_GB:
        return HealthCheck("disk", "ok", f"{free} GB free")
    return HealthCheck(
        "disk",
        "warn",
        f"low free disk ({free} GB)",
        impact="model downloads (hundreds of MB to several GB) and intermediate WAVs may fail",
        fix=["free up disk, or set a cache/output dir on a larger volume"],
        capabilities=["disk_write", "model_download"],
        options=[
            _option(
                "free_disk_space",
                "host_change",
                "Free disk space",
                "Make enough space available on the execution/output filesystems.",
                steps=["remove only user-approved expendable data or expand the filesystem"],
                tradeoffs=["deletion must be separately reviewed and explicitly approved"],
                recommended=True,
            ),
            _option(
                "move_cache_or_output",
                "recipe_variant",
                "Use a larger cache/output volume",
                "Point model caches, temporary files, or recipe outputs at a filesystem with adequate capacity.",
                steps=["select a writable larger volume", "create and validate a new recipe/launch configuration"],
                preserves_recipe=False,
                tradeoffs=["changes output/cache locations"],
            ),
        ],
    )


# --------------------------------------------------------------------------- aggregate + render


def _normalize_env(env: Any) -> EnvProfile:  # noqa: ANN401
    """Coerce partial test/adapter profiles to the full additive contract."""
    from nemo_curator.audio_agent.contracts import EnvProfile

    if isinstance(env, EnvProfile):
        return env
    normalized = EnvProfile()
    for item in fields(EnvProfile):
        if hasattr(env, item.name):
            setattr(normalized, item.name, getattr(env, item.name))
    return normalized


def env_report(env: EnvProfile | None = None) -> EnvHealthReport:
    """Probe the machine and return a structured health report (checks + overall status).

    Not named ``env_health``: that would shadow this module on the package, so
    ``from nemo_curator.audio_agent import env_health`` would hand back a function.
    """
    env = _normalize_env(env if env is not None else probe_env())
    checks: list[HealthCheck] = []
    for fn in _CHECKS:
        try:
            checks.append(fn(env))
        except Exception as exc:  # noqa: BLE001 - one diagnostic must not hide all others
            checks.append(
                HealthCheck(
                    id=f"diagnostic_{fn.__name__.lstrip('_')}",
                    status="warn",
                    finding=f"environment check could not complete: {type(exc).__name__}",
                    impact="this concern remains unverified; no compatibility claim was made",
                    confidence="unknown",
                    options=[
                        _option(
                            f"inspect_{fn.__name__.lstrip('_')}",
                            "diagnostic",
                            "Inspect the failed environment check",
                            "Collect the missing fact without assuming a cause or applying a fix.",
                            steps=[f"run `doctor --json` with verbose logs and inspect check {fn.__name__}"],
                            recommended=True,
                        )
                    ],
                )
            )
    overall: Status = "ok"
    for c in checks:
        if _RANK[c.status] > _RANK[overall]:
            overall = c.status
    n_fail = sum(c.status == "fail" for c in checks)
    n_warn = sum(c.status == "warn" for c in checks)
    if overall == "ok":
        summary = "environment healthy"
    elif overall == "warn":
        summary = f"usable with {n_warn} warning(s) -- some stages may be limited"
    else:
        summary = f"{n_fail} blocking issue(s) and {n_warn} warning(s) -- fix the FAILs before GPU model runs"
    return EnvHealthReport(status=overall, summary=summary, checks=checks, env=env.to_dict())


def doctor() -> dict[str, Any]:
    """JSON-able env health report (the ``doctor`` verb)."""
    return env_report().to_dict()


_ICON = {"ok": "OK  ", "warn": "WARN", "fail": "FAIL"}


def render_doctor(report: dict[str, Any]) -> str:
    """Human-readable rendering of a ``doctor`` report dict."""
    lines = [f"Environment health: {str(report.get('status', 'ok')).upper()} -- {report.get('summary', '')}", ""]
    for c in report.get("checks", []):
        lines.append(f"[{_ICON.get(c['status'], c['status'])}] {c['id']:20} {c['finding']}")
        if c["status"] != "ok":
            if c.get("impact"):
                lines.append(f"        impact: {c['impact']}")
            for step in c.get("fix", []):
                lines.append(f"        fix:    {step}")
            for option in c.get("options", []):
                marker = " (recommended)" if option.get("recommended") else ""
                availability = option.get("availability", "available")
                lines.append(
                    f"        option: {option.get('id')} [{availability}]{marker} - {option.get('label', '')}"
                )
    lines.append("")
    lines.append(f"(details: {_DOCS})")
    return "\n".join(lines)
