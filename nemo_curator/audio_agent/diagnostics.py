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

"""Recipe-aware environment diagnosis for the host-driven audio agent.

The deterministic core owns detection, applicability, and safe option metadata;
the host LLM owns explanation, prioritization against the user's constraints, and
the final question. Nothing in this module installs software, changes a driver,
sets a credential, mutates a stage/recipe, or silently switches devices.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from nemo_curator.audio_agent._safety import redact_secret_text
from nemo_curator.audio_agent.contracts import EnvProfile, Issue
from nemo_curator.audio_agent.env_health import (
    RemediationOption,
    env_report,
)

DecisionStatus = Literal["ready", "degraded", "action_required", "unknown"]
ExecutionTarget = Literal["local", "external_ray", "custom_executor"]

_GPU_BLOCKERS = frozenset({"gpu_unavailable", "cuda_driver_toolkit"})
_SAFE_ERROR_LIMIT = 500


@dataclass
class StageEnvironmentRequirement:
    """Environment facts for one concrete execution leaf."""

    recipe_index: int
    stage: str
    uses_gpu: bool = False
    hard_gpu: bool = False
    gpu_optional: bool | None = None
    cuda_runtime_jit: bool | None = False
    requires_ffmpeg: bool = False
    requires_internet_first_run: bool = False
    writes_to_disk: bool = False
    runtime_secrets: list[str] = field(default_factory=list)
    satisfied_runtime_secrets: list[str] = field(default_factory=list)
    metadata_known: bool = True
    note: str = ""

    @property
    def needs_gpu(self) -> bool:
        return self.uses_gpu or self.hard_gpu

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EnvironmentDecision:
    """The LLM-ready, recipe-specific preflight packet."""

    status: DecisionStatus
    operation: str
    execution_target: ExecutionTarget
    can_execute: bool
    decision_required: bool
    prompt_user: bool
    summary: str
    issues: list[dict[str, Any]] = field(default_factory=list)
    choices: list[dict[str, Any]] = field(default_factory=list)
    recommended: str | None = None
    question: str = ""
    requirements: list[dict[str, Any]] = field(default_factory=list)
    doctor_status: str = "ok"
    ignored_machine_checks: list[str] = field(default_factory=list)
    host_directive: str = (
        "Separate detected facts from inference; explain how each relevant issue affects "
        "this recipe; recommend the best available grounded option using the user's stated "
        "constraints; ask one outcome-level question before any host, environment, launch, "
        "credential, device, model/decoder, or recipe change. Never apply a fix, expose a "
        "secret, switch to CPU, or retry silently. After a choice, rerun preflight from fresh "
        "evidence. A recipe variant must validate, smoke, receive a new config hash, and be "
        "confirmed again."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _execution_requirements(stages: list[Any]) -> list[StageEnvironmentRequirement]:  # noqa: C901, PLR0915
    """Flatten configured composites and read their actual execution gates/cards."""
    try:
        from nemo_curator.stages.base import CompositeStage
    except Exception:  # noqa: BLE001 - missing base type makes composite support unknown
        CompositeStage = ()  # type: ignore[assignment]  # noqa: N806

    from nemo_curator.audio_agent.index import get_index
    from nemo_curator.stages.audio import agent as foundation

    idx = get_index()
    out: list[StageEnvironmentRequirement] = []

    def leaf(stage: Any, recipe_index: int, *, known: bool = True, note: str = "") -> None:  # noqa: ANN401
        name = type(stage).__name__
        card = idx.card(name) or {}
        resource = card.get("resource") or {}
        gpu_optional_raw = resource.get("gpu_optional")
        gpu_optional = gpu_optional_raw if isinstance(gpu_optional_raw, bool) else None
        hard_gpu = resource.get("bound") == "gpu" and gpu_optional is False
        resources = getattr(stage, "resources", None)
        reservation = float(getattr(resources, "gpus", 0.0) or 0.0)
        try:
            gates = foundation.build_contract(stage).gates
        except Exception as exc:  # noqa: BLE001 - unknown becomes explicit, never a guessed pass
            out.append(
                StageEnvironmentRequirement(
                    recipe_index=recipe_index,
                    stage=name,
                    uses_gpu=reservation > 0,
                    hard_gpu=hard_gpu,
                    gpu_optional=gpu_optional,
                    metadata_known=False,
                    note=f"contract unavailable: {type(exc).__name__}",
                )
            )
            return
        runtime_secrets = list(getattr(gates, "runtime_secrets", []) or [])
        # A gate names the credential it needs, but existing stages may receive
        # that credential through an explicit constructor parameter rather than
        # the process environment. Honour that established execution path while
        # reporting presence only; never copy the configured value into evidence.
        satisfied_runtime_secrets = [
            secret for secret in runtime_secrets if bool(getattr(stage, secret.lower(), None))
        ]
        cuda_runtime_jit: bool | None = False
        if name in {"NeMoASRAlignerStage", "SplitASRAlignJoinStage"}:
            cuda_runtime_jit = str(getattr(stage, "decoder_type", "rnnt") or "rnnt").lower() != "ctc"
        elif name == "ASRStage":
            adapter_target = str(getattr(stage, "adapter_target", "") or "")
            if adapter_target == "nemo_curator.models.asr.nemo_asr.NeMoASRAdapter":
                model_id = str(getattr(stage, "model_id", "") or "").lower()
                if "tdt" in model_id or "rnnt" in model_id:
                    cuda_runtime_jit = True
                elif "ctc" in model_id:
                    cuda_runtime_jit = False
                else:
                    cuda_runtime_jit = None
            else:
                cuda_runtime_jit = None
        out.append(
            StageEnvironmentRequirement(
                recipe_index=recipe_index,
                stage=name,
                uses_gpu=bool(getattr(gates, "requires_gpu", False)) or reservation > 0,
                hard_gpu=hard_gpu,
                gpu_optional=gpu_optional,
                cuda_runtime_jit=cuda_runtime_jit,
                requires_ffmpeg=bool(getattr(gates, "requires_ffmpeg", False)),
                requires_internet_first_run=bool(getattr(gates, "requires_internet_first_run", False)),
                writes_to_disk=bool(getattr(gates, "writes_to_disk", False)),
                runtime_secrets=runtime_secrets,
                satisfied_runtime_secrets=satisfied_runtime_secrets,
                metadata_known=known,
                note=note,
            )
        )

    def visit(stage: Any, recipe_index: int, depth: int) -> None:  # noqa: ANN401
        if depth >= 8:  # noqa: PLR2004
            leaf(stage, recipe_index, known=False, note="composite expansion depth exceeded")
            return
        if CompositeStage and isinstance(stage, CompositeStage):
            try:
                inner = list(stage.decompose_and_apply_with() or [])
            except Exception as exc:  # noqa: BLE001 - do not claim hidden leaves are CPU-safe
                leaf(
                    stage,
                    recipe_index,
                    known=False,
                    note=f"composite could not be expanded: {type(exc).__name__}",
                )
                return
            if not inner:
                leaf(stage, recipe_index, known=False, note="composite expansion returned no leaves")
                return
            for child in inner:
                visit(child, recipe_index, depth + 1)
            return
        leaf(stage, recipe_index)

    for index, stage in enumerate(stages):
        visit(stage, index, 0)
    return out


def _health_checks(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(check.get("id")): check
        for check in report.get("checks", [])
        if isinstance(check, dict) and check.get("id")
    }


def _issue(  # noqa: PLR0913 - issues keep evidence and choices explicit at construction
    code: str,
    *,
    blocking: bool,
    finding: str,
    impact: str,
    affected: list[str],
    confidence: str = "high",
    source_check: str = "",
    options: list[dict[str, Any]] | None = None,
    note: str = "",
    resolution_key: str = "",
) -> dict[str, Any]:
    return {
        "code": code,
        "resolution_key": resolution_key or code,
        "severity": "error" if blocking else "warning",
        "blocking": blocking,
        "finding": finding,
        "impact": impact,
        "affected_stages": sorted(set(affected)),
        "confidence": confidence,
        "source_check": source_check or code,
        "options": list(options or []),
        **({"note": note} if note else {}),
    }


def _option_dicts(check: dict[str, Any] | None) -> list[dict[str, Any]]:
    return [dict(option) for option in (check or {}).get("options", []) if isinstance(option, dict)]


def _reverify_gpu_options() -> list[dict[str, Any]]:
    """The remedy for a possibly-masked GPU: re-probe with full device access.

    A masked GPU is an ACCESS fact, not a hardware fact -- the only correct action
    is to re-run the probe (and any GPU verb) outside the sandbox/container, never to
    install a CPU build or switch to CPU.
    """
    return [
        RemediationOption(
            id="reverify_full_device_access",
            kind="diagnostic",
            label="Re-verify the GPU with full device access",
            summary=(
                "Re-run `doctor --json` (and any smoke/run) with FULL device access -- "
                "outside the sandbox/container that is blocking /dev/nvidia* -- so torch "
                "can see the GPU. Only conclude 'no GPU' if a full-access probe still finds none."
            ),
            steps=[
                "re-run `.venv/bin/python -m nemo_curator.audio_agent doctor --json` with full device access",
                "run smoke/run with full device access (they already require it for GPU stages)",
            ],
            recommended=True,
        ).to_dict()
    ]


def _cpu_choice(
    requirements: list[StageEnvironmentRequirement],
    blocking_codes: list[str],
) -> dict[str, Any] | None:
    affected = [req for req in requirements if req.needs_gpu]
    if not affected:
        return None
    unknown = [req.stage for req in affected if not req.metadata_known or req.gpu_optional is None]
    unsupported = [req.stage for req in affected if req.gpu_optional is False]
    if unsupported:
        availability: Literal["conditional", "unavailable", "unknown"] = "unavailable"
        reason = "GPU-only execution leaves: " + ", ".join(sorted(set(unsupported)))
    elif unknown:
        availability = "unknown"
        reason = "CPU support is not proven for: " + ", ".join(sorted(set(unknown)))
    else:
        availability = "conditional"
        reason = (
            "every affected execution leaf declares GPU optional; a new explicit "
            "CPU recipe still must build, resource-plan, validate, and smoke"
        )

    remaining = sorted(set(blocking_codes) - _GPU_BLOCKERS)
    option = RemediationOption(
        id="use_cpu_recipe_variant",
        kind="recipe_variant",
        label="Create a CPU-compatible recipe variant",
        summary=(
            "Replan the affected stages for CPU without changing their module defaults or mutating the current recipe."
        ),
        steps=[
            "create a new explicit recipe with CPU resources/stage alternatives",
            "validate the new recipe and run a bounded CPU smoke",
            "present its new config hash, performance trade-off, and request confirmation",
        ],
        availability=availability,
        reason=reason,
        preserves_recipe=False,
        tradeoffs=["usually slower; model-stage alternatives can change output behavior"],
        recommended=False,
        applies_to=sorted({req.stage for req in affected}),
        resolves=sorted(set(blocking_codes) & _GPU_BLOCKERS),
        remaining_blockers=remaining,
        verify=[
            "new recipe preflight is ready",
            "new recipe validates",
            "bounded CPU smoke meets the user's acceptance criteria",
        ],
    )
    return option.to_dict()


def _ctc_choice(requirements: list[StageEnvironmentRequirement]) -> dict[str, Any] | None:
    eligible = sorted(
        {req.stage for req in requirements if req.stage in {"NeMoASRAlignerStage", "SplitASRAlignJoinStage"}}
    )
    if not eligible:
        return None
    return RemediationOption(
        id="use_ctc_alignment_variant",
        kind="recipe_variant",
        label="Use a verified CTC alignment variant",
        summary=(
            "Avoid the CUDA-graph decoder only when the configured hybrid alignment "
            "checkpoint is proven to expose a CTC head."
        ),
        steps=[
            "verify the selected alignment checkpoint supports CTC",
            "create a new recipe with decoder_type='ctc'",
            "validate, smoke, and request confirmation for its new config hash",
        ],
        availability="conditional",
        reason="checkpoint capability must be verified; this is not valid for pure-TDT transcription",
        preserves_recipe=False,
        tradeoffs=["changes decoder behavior"],
        applies_to=eligible,
        resolves=["cuda_driver_toolkit"],
        verify=["new alignment recipe validates and its bounded smoke succeeds"],
    ).to_dict()


def _choices(
    issues: list[dict[str, Any]],
    requirements: list[StageEnvironmentRequirement],
) -> list[dict[str, Any]]:
    blocking_codes = [
        str(issue.get("resolution_key") or issue.get("code")) for issue in issues if issue.get("blocking")
    ]
    out: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for issue in issues:
        for option in issue.get("options", []):
            option_id = str(option.get("id") or "")
            # These two need recipe-specific applicability below.
            if not option_id or option_id in {
                "use_cpu_recipe_variant",
                "use_ctc_alignment_variant",
            }:
                continue
            resolves = set(option.get("resolves") or [])
            resolves.add(str(issue.get("resolution_key") or issue.get("code") or ""))
            if option_id in by_id:
                merged = set(by_id[option_id].get("resolves") or [])
                by_id[option_id]["resolves"] = sorted(merged | resolves)
            else:
                item = dict(option)
                item["resolves"] = sorted(code for code in resolves if code)
                out.append(item)
                by_id[option_id] = item
    cpu = _cpu_choice(requirements, blocking_codes)
    if cpu is not None:
        out.append(cpu)
        by_id[str(cpu["id"])] = cpu
    if "cuda_driver_toolkit" in blocking_codes:
        ctc = _ctc_choice(requirements)
        if ctc is not None and str(ctc["id"]) not in by_id:
            out.append(ctc)
    blockers = set(blocking_codes)
    for option in out:
        resolves = set(option.get("resolves") or [])
        option["remaining_blockers"] = sorted(blockers - resolves)
    return out


def _recommended(choices: list[dict[str, Any]]) -> str | None:
    viable = [
        option
        for option in choices
        if option.get("availability") == "available" and not option.get("remaining_blockers")
    ]
    for option in viable:
        if option.get("recommended"):
            return str(option.get("id"))
    for option in viable:
        if option.get("preserves_recipe"):
            return str(option.get("id"))
    return str(viable[0].get("id")) if viable else None


def environment_preflight(  # noqa: C901, PLR0912, PLR0915 - one auditable decision tree
    stages: list[Any],
    env: EnvProfile | Any | None = None,  # noqa: ANN401 - accepts additive adapter profiles
    *,
    operation: str = "validate",
    execution_target: ExecutionTarget = "local",
) -> dict[str, Any]:
    """Correlate machine health with the selected concrete recipe.

    Global warnings that do not apply to this recipe are omitted. A known,
    relevant execution blocker produces ``action_required``; uncertain/low-risk
    facts produce ``degraded`` and never fabricate a compatibility claim.
    """
    report = env_report(env).to_dict()
    env_data = dict(report.get("env") or {})
    checks = _health_checks(report)
    requirements = _execution_requirements(stages)
    stage_names = sorted({req.stage for req in requirements})
    gpu_needs = [req for req in requirements if req.needs_gpu]
    ffmpeg_users = [req for req in requirements if req.requires_ffmpeg]
    disk_users = [req for req in requirements if req.writes_to_disk or req.requires_internet_first_run]
    issues: list[dict[str, Any]] = []

    worker = checks.get("worker_env")
    if stage_names and execution_target == "local" and worker and worker.get("status") == "fail":
        issues.append(
            _issue(
                "worker_env_mismatch",
                blocking=True,
                finding=str(worker.get("finding") or ""),
                impact=str(worker.get("impact") or ""),
                affected=stage_names,
                source_check="worker_env",
                options=_option_dicts(worker),
            )
        )
    elif stage_names and execution_target == "local" and worker and worker.get("status") == "warn":
        issues.append(
            _issue(
                "worker_env_unverified",
                blocking=False,
                finding=str(worker.get("finding") or ""),
                impact=str(worker.get("impact") or ""),
                affected=stage_names,
                confidence="unknown",
                source_check="worker_env",
                options=_option_dicts(worker),
            )
        )

    if execution_target == "local":
        if gpu_needs and not bool(env_data.get("has_gpu")):
            gpu = checks.get("gpu") or {}
            # A GPU-required stage hard-blocks only when the GPU is definitively ABSENT (a
            # CPU-only torch build). Merely unreachable from this possibly-sandboxed process
            # (``possibly_masked``) or ``unknown`` yields a non-blocking re-verify, never
            # runnable:false -- which is what stops a masked GPU reading as the recurring
            # false "no GPU".
            gpu_status = str(env_data.get("gpu_status") or "unknown")
            reverify = gpu_status in {"possibly_masked", "unknown"}
            if gpu_status == "possibly_masked":
                code = "gpu_possibly_masked"
                finding = str(gpu.get("finding") or "a GPU is likely present but not reachable from this process")
                impact = (
                    "GPU-required stages cannot be probed here, but this is NOT a hardware "
                    "absence (a sandbox/container is blocking /dev/nvidia*, or CUDA_VISIBLE_DEVICES "
                    "masks the device) -- re-verify with full device access before concluding no GPU"
                )
                options = _reverify_gpu_options()
            elif gpu_status == "unknown":
                code = "gpu_availability_unknown"
                finding = str(gpu.get("finding") or "GPU availability is unknown")
                impact = (
                    "the selected recipe contains GPU execution leaves, but the "
                    "environment supplied no GPU visibility facts"
                )
                options = _reverify_gpu_options()
            else:  # absent -- a CPU-only torch build cannot use a GPU regardless of hardware
                code = "gpu_unavailable"
                finding = str(gpu.get("finding") or "no usable GPU on this host (CPU-only torch build)")
                impact = "the selected recipe contains GPU-required execution leaves"
                options = _option_dicts(gpu)
            issues.append(
                _issue(
                    code,
                    blocking=not reverify,
                    finding=finding,
                    impact=impact,
                    affected=[req.stage for req in gpu_needs],
                    confidence="unknown" if reverify else str(gpu.get("confidence") or "high"),
                    source_check="gpu",
                    options=options,
                )
            )
        elif gpu_needs and env_data.get("cuda_compatible") is False:
            cuda = checks.get("cuda_driver_toolkit") or {}
            jit_users = [req for req in gpu_needs if req.cuda_runtime_jit is True]
            other_gpu_users = [req for req in gpu_needs if req.cuda_runtime_jit is not True]
            if jit_users:
                issues.append(
                    _issue(
                        "cuda_driver_toolkit",
                        blocking=True,
                        finding=str(cuda.get("finding") or "CUDA driver/toolkit mismatch"),
                        impact=(
                            "the selected runtime-JIT/CUDA-graph decoder is known to "
                            "target PTX that this driver cannot load"
                        ),
                        affected=[req.stage for req in jit_users],
                        confidence=str(cuda.get("confidence") or "high"),
                        source_check="cuda_driver_toolkit",
                        options=_option_dicts(cuda),
                    )
                )
            if other_gpu_users:
                issues.append(
                    _issue(
                        "cuda_driver_toolkit_mismatch",
                        blocking=False,
                        finding=str(cuda.get("finding") or "CUDA driver/toolkit mismatch"),
                        impact=(
                            "the mismatch is real, but this stage is not known to require "
                            "runtime PTX/JIT; verify it with a bounded smoke"
                        ),
                        affected=[req.stage for req in other_gpu_users],
                        confidence="medium",
                        source_check="cuda_driver_toolkit",
                        options=[
                            RemediationOption(
                                id="verify_gpu_stage_with_bounded_smoke",
                                kind="diagnostic",
                                label="Verify this GPU path with a bounded smoke",
                                summary=(
                                    "Collect stage-specific execution evidence before "
                                    "changing a working host or project environment."
                                ),
                                steps=[
                                    "run a bounded smoke on the selected execution target",
                                    "if it fails, diagnose the exact CUDA/runtime signature",
                                ],
                                recommended=True,
                                applies_to=[req.stage for req in other_gpu_users],
                            ).to_dict()
                        ],
                    )
                )
        elif gpu_needs and (not env_data.get("cuda_runtime_version") or not env_data.get("cuda_driver_max_version")):
            cuda = checks.get("cuda_driver_toolkit") or {}
            issues.append(
                _issue(
                    "cuda_compatibility_unknown",
                    blocking=False,
                    finding=str(cuda.get("finding") or "CUDA compatibility is unknown"),
                    impact=str(cuda.get("impact") or "GPU compatibility is unverified"),
                    affected=[req.stage for req in gpu_needs],
                    confidence="unknown",
                    source_check="cuda_driver_toolkit",
                    options=_option_dicts(cuda),
                )
            )
    elif requirements:
        target_label = "external Ray workers" if execution_target == "external_ray" else "the caller-supplied executor"
        issues.append(
            _issue(
                (
                    "remote_worker_environment_unverified"
                    if execution_target == "external_ray"
                    else "custom_executor_environment_unverified"
                ),
                blocking=False,
                finding=f"machine health was probed on the driver, not {target_label}",
                impact=(
                    "target Python, dependencies, credentials, executables, disk, and GPU/CUDA "
                    "compatibility remain unverified until a bounded target-side smoke"
                ),
                affected=stage_names,
                confidence="unknown",
                source_check="execution_target",
                options=[
                    RemediationOption(
                        id="verify_execution_target",
                        kind="diagnostic",
                        label="Verify the execution target",
                        summary=f"Run a bounded smoke against {target_label} and inspect target-side facts/logs.",
                        steps=["run recipe preflight/smoke against the selected execution target"],
                        recommended=True,
                    ).to_dict()
                ],
            )
        )

    if execution_target == "local" and ffmpeg_users and not bool(env_data.get("has_ffmpeg")):
        ffmpeg = checks.get("ffmpeg") or {}
        issues.append(
            _issue(
                "ffmpeg_missing",
                blocking=True,
                finding=str(ffmpeg.get("finding") or "ffmpeg is missing"),
                impact=str(ffmpeg.get("impact") or "selected stages will fail"),
                affected=[req.stage for req in ffmpeg_users],
                source_check="ffmpeg",
                options=_option_dicts(ffmpeg),
            )
        )

    # An absent package drops its stages from the catalog entirely (``discover()['unavailable']``
    # reports it), so such a stage can never reach a recipe to be flagged here. A per-stage
    # package table would therefore cost maintenance for every new module while gating on a
    # condition it cannot observe. The remediation below still names the audio extra to install.

    available_secrets = set(env_data.get("available_secrets") or [])
    missing_secret_stages: dict[str, list[str]] = {}
    if execution_target == "local":
        for req in requirements:
            for secret in req.runtime_secrets:
                if secret not in available_secrets and secret not in req.satisfied_runtime_secrets:
                    missing_secret_stages.setdefault(secret, []).append(req.stage)
    for secret, affected in sorted(missing_secret_stages.items()):
        issues.append(
            _issue(
                "missing_secret",
                blocking=True,
                finding=f"required credential variable {secret!r} is not set",
                impact="the affected model stage cannot authenticate/download",
                affected=affected,
                source_check="runtime_secret",
                resolution_key=f"missing_secret:{secret}",
                options=[
                    RemediationOption(
                        id=f"set_{secret.lower()}_in_secret_manager",
                        kind="credential",
                        label=f"Configure {secret} outside the chat",
                        summary=(
                            f"Set {secret} in the execution shell/secret manager, without "
                            "sending its value to the host LLM."
                        ),
                        steps=[
                            f"configure {secret} in the approved shell/secret manager",
                            "rerun preflight; only presence is reported",
                        ],
                        recommended=True,
                        applies_to=sorted(set(affected)),
                    ).to_dict(),
                    RemediationOption(
                        id=f"replace_{secret.lower()}_stage",
                        kind="recipe_variant",
                        label="Choose a stage that does not need this credential",
                        summary="Replan only if an equivalent grounded capability exists in the catalog.",
                        steps=["compare applicable stage cards", "create and validate a new recipe"],
                        availability="conditional",
                        preserves_recipe=False,
                        tradeoffs=["may change model quality, language support, or outputs"],
                        applies_to=sorted(set(affected)),
                    ).to_dict(),
                ],
            )
        )

    if execution_target == "local" and not bool(env_data.get("python_supported", True)):
        python = checks.get("python") or {}
        issues.append(
            _issue(
                "unsupported_python",
                blocking=False,
                finding=str(python.get("finding") or "unsupported Python"),
                impact=str(python.get("impact") or "runtime behavior is unverified"),
                affected=stage_names,
                source_check="python",
                options=_option_dicts(python),
            )
        )

    disk = checks.get("disk") or {}
    if execution_target == "local" and disk_users and disk.get("status") == "warn":
        issues.append(
            _issue(
                "disk_capacity",
                blocking=False,
                finding=str(disk.get("finding") or "disk capacity is unverified"),
                impact=str(disk.get("impact") or "writes/downloads may fail"),
                affected=[req.stage for req in disk_users],
                confidence=str(disk.get("confidence") or "high"),
                source_check="disk",
                options=_option_dicts(disk),
            )
        )

    unknown_metadata = [req for req in requirements if not req.metadata_known]
    if unknown_metadata:
        issues.append(
            _issue(
                "environment_requirements_unknown",
                blocking=False,
                finding="some execution leaves could not expose complete environment metadata",
                impact="device/fallback applicability cannot be proven for those leaves",
                affected=[req.stage for req in unknown_metadata],
                confidence="unknown",
                source_check="stage_contract",
                options=[
                    RemediationOption(
                        id="inspect_stage_environment_contract",
                        kind="diagnostic",
                        label="Inspect the unresolved stage contract",
                        summary="Resolve/decompose the stage before claiming an environment or CPU path.",
                        steps=["run `describe` and inspect/decompose the affected stage"],
                        recommended=True,
                    ).to_dict()
                ],
            )
        )

    # Future machine-wide FAIL checks default safe: apply only when they say
    # they affect all execution and were not already handled above.
    handled_sources = {str(issue.get("source_check")) for issue in issues} | {
        "gpu",
        "cuda_driver_toolkit",
        "ffmpeg",
        "worker_env",
    }
    for check_id, check in checks.items():
        if (
            check.get("status") == "fail"
            and check_id not in handled_sources
            and "all" in set(check.get("capabilities") or [])
        ):
            issues.append(
                _issue(
                    check_id,
                    blocking=True,
                    finding=str(check.get("finding") or check_id),
                    impact=str(check.get("impact") or "recipe execution is unsafe"),
                    affected=stage_names,
                    confidence=str(check.get("confidence") or "unknown"),
                    source_check=check_id,
                    options=_option_dicts(check),
                )
            )

    choices = _choices(issues, requirements)
    blockers = [issue for issue in issues if issue.get("blocking")]
    unknowns = [issue for issue in issues if issue.get("confidence") == "unknown"]
    if blockers:
        status: DecisionStatus = "action_required"
        summary = (
            f"{len(blockers)} environment action(s) required before {operation}; "
            "no execution or automatic remediation was performed"
        )
    elif issues:
        status = "unknown" if unknowns and len(unknowns) == len(issues) else "degraded"
        summary = f"environment usable with {len(issues)} relevant warning(s)"
    else:
        status = "ready"
        summary = "environment is ready for the selected recipe"
    recommended = _recommended(choices)
    label = next(
        (str(option.get("label")) for option in choices if option.get("id") == recommended),
        "",
    )
    question = ""
    if blockers:
        question = (
            f"Recommended: {label}. Which available approach should we take before {operation}?"
            if label
            else (
                "No complete automatic remedy is proven. Which grounded "
                f"diagnostic/fix approach should we take before {operation}?"
            )
        )

    return EnvironmentDecision(
        status=status,
        operation=operation,
        execution_target=execution_target,
        can_execute=not blockers,
        decision_required=bool(blockers),
        prompt_user=bool(blockers),
        summary=summary,
        issues=issues,
        choices=choices,
        recommended=recommended,
        question=question,
        requirements=[req.to_dict() for req in requirements],
        doctor_status=str(report.get("status") or "ok"),
        ignored_machine_checks=sorted(
            check_id
            for check_id, check in checks.items()
            if check.get("status") != "ok" and check_id not in {str(issue.get("source_check")) for issue in issues}
        ),
    ).to_dict()


def verdict_issues(decision: dict[str, Any]) -> list[Issue]:
    """Project blocking preflight findings onto the stable Verdict gate surface."""
    out: list[Issue] = []
    for finding in decision.get("issues", []):
        if not finding.get("blocking"):
            continue
        affected = list(finding.get("affected_stages") or [])
        recommended = str(decision.get("recommended") or "")
        option = next(
            (item for item in decision.get("choices", []) if item.get("id") == recommended),
            None,
        )
        fix = (
            str(option.get("summary"))
            if isinstance(option, dict)
            else "review environment_decision choices and rerun preflight after the user-approved change"
        )
        out.append(
            Issue(
                code=str(finding.get("code") or "environment_action_required"),
                severity="error",
                message=f"{finding.get('finding')}: {finding.get('impact')}",
                stage=", ".join(affected) if affected else None,
                fix=fix,
                escalate_to="user",
            )
        )
    return out


def _safe_error_text(error: str) -> str:
    return redact_secret_text(str(error or "")[:_SAFE_ERROR_LIMIT])


def diagnose_failure(  # noqa: C901, PLR0913 - explicit taxonomy and applicability branches
    error: str,
    *,
    stages: list[Any] | None = None,
    env: EnvProfile | Any | None = None,  # noqa: ANN401
    operation: str = "run",
    phase: str = "runtime",
    execution_target: ExecutionTarget = "local",
    attempted_actions: list[str] | None = None,
) -> dict[str, Any]:
    """Classify a failure and return grounded recovery/diagnostic choices."""
    from nemo_curator.audio_agent.failures import classify

    safe = _safe_error_text(error)
    failure = classify(safe)
    preflight = environment_preflight(
        stages or [],
        env,
        operation=operation,
        execution_target=execution_target,
    )
    options = [dict(item) for item in preflight.get("choices", [])]
    preflight_blockers = {
        str(issue.get("resolution_key") or issue.get("code"))
        for issue in preflight.get("issues", [])
        if issue.get("blocking")
    }
    code = str(failure.get("code") or "unknown_failure")
    requirements = _execution_requirements(stages or [])
    asr_runtime_stages = {
        req.stage
        for req in requirements
        if req.stage
        in {
            "ASRStage",
            "NeMoASRAlignerStage",
            "SplitASRAlignJoinStage",
        }
    }
    if code == "asr_decoder_cuda_graph" and not asr_runtime_stages:
        # The signature is a general CUDA PTX/NVRTC failure even though its
        # original taxonomy entry was authored for the common NeMo ASR case.
        # With a concrete non-ASR recipe, remove ASR/CTC-specific guidance.
        failure = {
            **failure,
            "code": "cuda_runtime_jit_failure",
            "likely_cause": ("a runtime-compiled CUDA/PTX kernel is incompatible with the active driver/toolkit path"),
            "auto_fix": (
                "run `doctor --json` in the same execution context and choose a supported driver/framework combination"
            ),
            "user_guidance": (
                "inspect the affected non-ASR stage and CUDA evidence; do not apply an ASR decoder workaround"
            ),
        }
        code = "cuda_runtime_jit_failure"
    if code == "unknown_failure":
        options.append(
            RemediationOption(
                id="collect_minimal_diagnostics",
                kind="diagnostic",
                label="Collect minimal diagnostics",
                summary="Gather fresh doctor output and the failing phase/stage logs without guessing a fix.",
                steps=[
                    "run `doctor --json` in the same launch environment",
                    "capture the failing phase/stage and its bounded verbose log",
                    "do not repeat an already-failed action unchanged",
                ],
                recommended=True,
                verify=["a known cause is established or the issue is escalated with evidence"],
            ).to_dict()
        )
        status: DecisionStatus = "unknown"
    else:
        auto_fix = str(failure.get("auto_fix") or "").strip()
        guidance = str(failure.get("user_guidance") or "").strip()
        if code == "asr_decoder_cuda_graph":
            # The taxonomy's prose names CTC as one possible workaround, but
            # recipe applicability is narrower: only hybrid alignment stages,
            # never unrelated GPU stages or pure-TDT transcription.
            if not any(item.get("id") == "inspect_asr_cuda_stack" for item in options):
                options.append(
                    RemediationOption(
                        id="inspect_asr_cuda_stack",
                        kind="diagnostic",
                        label="Verify the ASR CUDA stack",
                        summary=(
                            "Confirm driver/PyTorch CUDA compatibility and the exact "
                            "selected ASR stage/checkpoint before choosing a fix."
                        ),
                        steps=["run `doctor --json` in the failing execution environment"],
                        recommended=not bool(preflight.get("recommended")),
                        verify=["recipe-aware preflight identifies an applicable host/environment fix"],
                    ).to_dict()
                )
            ctc = _ctc_choice(requirements)
            if ctc is not None and not any(item.get("id") == ctc.get("id") for item in options):
                options.append(ctc)
        elif auto_fix or guidance:
            kind: Literal[
                "host_change",
                "environment_change",
                "launch_change",
                "recipe_variant",
                "credential",
                "diagnostic",
            ] = (
                "environment_change"
                if failure.get("layer") == "env"
                else "recipe_variant"
                if failure.get("layer") in {"recipe", "data"}
                else "diagnostic"
            )
            options.append(
                RemediationOption(
                    id=f"recover_{code}",
                    kind=kind,
                    label=f"Address {code.replace('_', ' ')}",
                    summary=str(failure.get("likely_cause") or code),
                    steps=[step for step in (auto_fix, guidance) if step],
                    recommended=not bool(preflight.get("recommended")),
                    verify=["rerun preflight", "retry only with a bounded smoke"],
                ).to_dict()
            )
        status = "action_required"

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    safe_attempted_actions = [
        _safe_error_text(str(action)) for action in (attempted_actions or []) if str(action).strip()
    ]
    attempted = {action.strip().lower() for action in safe_attempted_actions}
    for option in options:
        option_id = str(option.get("id") or "")
        if option_id and option_id not in seen:
            item = dict(option)
            item["remaining_blockers"] = sorted(preflight_blockers - set(item.get("resolves") or []))
            if item["remaining_blockers"]:
                item["recommended"] = False
            if option_id.lower() in attempted or str(item.get("label") or "").strip().lower() in attempted:
                item["availability"] = "unavailable"
                item["reason"] = "the user reports this action was already attempted"
                item["recommended"] = False
            deduped.append(item)
            seen.add(option_id)
    preflight_recommended = str(preflight.get("recommended") or "")
    if preflight_recommended and not any(
        item.get("id") == preflight_recommended
        and item.get("availability") == "available"
        and not item.get("remaining_blockers")
        for item in deduped
    ):
        preflight_recommended = ""
    recommended = preflight_recommended or next(
        (
            str(item.get("id"))
            for item in deduped
            if item.get("recommended")
            and item.get("availability") == "available"
            and not item.get("remaining_blockers")
        ),
        None,
    )
    return {
        "status": status,
        "operation": operation,
        "phase": phase,
        "failure": failure,
        "environment_decision": preflight,
        "attempted_actions": safe_attempted_actions,
        "choices": deduped,
        "recommended": recommended,
        "decision_required": True,
        "prompt_user": True,
        "question": (
            "Review the evidence and available approaches; which user-approved "
            "diagnostic or remediation should we take next?"
        ),
        "host_directive": (
            "Explain the observed failure and confidence, correlate it with the live "
            "environment and selected recipe, recommend only a grounded applicable option, "
            "and ask before changing anything. Do not expose credentials, invent a cause, "
            "or repeat an attempted action unchanged."
        ),
    }
