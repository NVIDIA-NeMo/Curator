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

"""Ray execution, resource planning, and stage lifecycle."""

from __future__ import annotations

import contextlib
import copy
import functools
import math
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from nemo_curator.stages.audio.audio_agent import _verb_inputs, _verb_outputs


def _is_streaming_infeasible(err: Exception) -> bool:
    m = str(err).lower()
    return "streaming mode" in m and any(k in m for k in ("not enough gpu", "batch mode", "requires"))


def _automatic_retry_reset_hook(
    stages: list[Any],
    *,
    extra_reset: Callable[[], Any] | None = None,
    staged_outputs: list[_verb_outputs._StagedFileOutput] | None = None,
) -> Callable[[], None] | None:
    """Compose stage-owned reset hooks with existing pipeline finalizer cleanup."""
    stage_resets = [reset for stage in stages if callable(reset := getattr(stage, "reset_for_retry", None))]
    if not stage_resets and extra_reset is None:
        return None

    def _reset() -> None:
        for reset in stage_resets:
            reset()
        if staged_outputs:
            from fsspec.core import url_to_fs

            from nemo_curator.stages.audio.common import ManifestWriterStage

            for output in staged_outputs:
                if isinstance(output.stage, ManifestWriterStage):
                    fs, source = url_to_fs(output.final_path)
                    _, destination = url_to_fs(output.staged_path)
                    if fs.exists(source):
                        fs.cp_file(source, destination)
                    else:
                        with fs.open(destination, "wb"):
                            pass
        if extra_reset is not None:
            extra_reset()

    return _reset


def _run_pipeline_autofallback(  # noqa: PLR0913
    stages: list[Any],
    mode: str,
    caller_executor: Any,  # noqa: ANN401
    *,
    checkpoint_path: str | None = None,
    note: Any = None,  # noqa: ANN401
    before_retry: Any = None,  # noqa: ANN401
) -> tuple[list[Any] | None, str]:
    """Run in the planned mode; on a runtime streaming-infeasibility error (e.g. a
    composite stage hides inner GPU stages, so the planner undercounted the concurrent
    GPU reservation), retry once in batch -- Xenna's own recommended remedy.
    """
    executor = caller_executor if caller_executor is not None else _make_executor(mode)
    try:
        results = _run_pipeline(stages, executor, checkpoint_path=checkpoint_path)
        return results, mode  # noqa: TRY300
    except Exception as e:
        if caller_executor is None and mode != "batch" and _is_streaming_infeasible(e):
            if callable(note):
                note("streaming infeasible at runtime -> retried in batch")
            if callable(before_retry):
                # Reset hook for writers whose lifecycle is NOT truncate-in-setup() (the ALM
                # prepare/finalize pair), so two attempts are never merged. Standard sinks
                # truncate in setup(), which the retry re-invokes, so their shards are replaced
                # without this -- hence before_retry is None for a plain writer pipeline.
                before_retry()
            results = _run_pipeline(
                stages,
                _make_executor("batch"),
                checkpoint_path=checkpoint_path,
            )
            return results, "batch"
        raise


def _stops_a_head_it_started(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Guarantee that a Ray head bootstrapped by this call is stopped if the call raises.

    ``run`` and ``smoke`` stop their own head on every path they ANTICIPATE: a planning
    error, an infeasible plan, a normal return. What they cannot cover is the long tail
    between execution and that final stop -- re-binding the dataset, publishing artifacts,
    assembling the report -- which is not inside any handler. An unexpected error there
    escaped past the stop and left a live local head, with its ownership record and temp
    directory, behind in a process that had already given up on the work. The next call then
    found ``RAY_ADDRESS`` pointing at that head and refused as ambiguous, so one unrelated
    failure made every subsequent run refuse until the process was restarted.

    Ownership is module state in ``_ray``, so what needs stopping is knowable from outside
    the verb: anything owned on the way out that was not owned on the way in was started
    here. A successful return has already stopped and cleared it, so this finds nothing to
    do -- which is why it can wrap the verb whole instead of re-indenting its body.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        from nemo_curator.stages.audio.audio_agent._ray import owns_cluster, shutdown_cluster

        owned_before = owns_cluster()
        try:
            return fn(*args, **kwargs)
        except BaseException:  # KeyboardInterrupt too: Ctrl-C must not orphan a live head
            if not owned_before and owns_cluster():
                # No address argument: ownership is the whole question, and it has just been
                # answered. ``shutdown_cluster`` re-checks it and stops what this process
                # owns, with the same refusal gates as every other teardown path.
                with contextlib.suppress(Exception):
                    shutdown_cluster()
            raise

    return wrapper


def _bootstrap_ray() -> str:
    """Ensure a Ray cluster (opt-in) and return its address; sets RAY_ADDRESS."""
    from nemo_curator.stages.audio.audio_agent._ray import ensure_cluster

    return ensure_cluster()


def _shutdown_owned_ray(address: str | None) -> bool | None:
    """Stop only a local Ray head this process created.

    ``None`` means the address was external/not owned, ``True`` means cleanup
    completed, and ``False`` means an owned head could not be stopped safely.
    """
    if not address:
        return None
    from nemo_curator.stages.audio.audio_agent._ray import owns_cluster, shutdown_cluster

    if not owns_cluster(address):
        return None
    return shutdown_cluster(address)


def _resource_environment(
    env: Any,  # noqa: ANN401
    address: str | None,
    execution_target: str,
) -> Any:  # noqa: ANN401
    """Return only capacity facts that belong to the selected executor."""
    if execution_target != "custom_executor":
        return _apply_ray_cluster_capacity(env, address)
    target_env = copy.deepcopy(env)
    target_env.has_gpu = False
    target_env.gpu_count = 0
    target_env.gpu_names = []
    target_env.gpu_mem_gb = 0.0
    target_env.gpu_visibility = "unknown"
    target_env.total_cpus = 0
    target_env.total_ram_gb = 0.0
    target_env.free_disk_gb = None
    target_env.notes = list(getattr(target_env, "notes", []) or [])
    target_env.notes.append(
        "custom executor capacity is caller-owned and unverified; driver capacity was not substituted"
    )
    return target_env


def _adapt_resource_plan_for_target(
    plan: Any,  # noqa: ANN401
    *,
    execution_target: str,
    operation: str,
) -> None:
    """Keep target uncertainty explicit without making bounded verification impossible."""
    if not isinstance(getattr(plan, "notes", None), list):
        plan.notes = []
    escalations = list(getattr(plan, "escalations", []) or [])
    if execution_target == "custom_executor":
        if escalations:
            plan.notes.append("custom_executor_unverified_capacity=" + "; ".join(escalations))
        plan.escalations = []
        plan.feasible = True
        plan.notes.append("custom executor owns scheduling; capacity is verified by its bounded execution")
        return
    # Dormant by design, not by oversight: the planner currently reports unknown VRAM as a
    # NOTE and never escalates on it ("VRAM is intentionally NOT escalated here" --
    # ``planner.py``), so no live plan reaches this with such an escalation. It stays because
    # the rule it encodes outlives that choice: a bounded smoke is how VRAM gets MEASURED on a
    # remote cluster, so refusing the smoke for want of the measurement would close the only
    # door to it. A run is a different matter and still escalates.
    unknown_vram = [item for item in escalations if item.startswith("GPU VRAM capacity is unknown;")]
    if (
        execution_target == "external_ray"
        and operation == "smoke"
        and escalations
        and len(unknown_vram) == len(escalations)
    ):
        plan.notes.extend("bounded_remote_smoke=" + item for item in unknown_vram)
        plan.escalations = []
        plan.feasible = True


def _apply_ray_cluster_capacity(env: Any, address: str | None) -> Any:  # noqa: ANN401
    """Overlay driver probes with the resources the executor can schedule."""
    if not address:
        return env
    try:
        from nemo_curator.stages.audio.audio_agent._ray import cluster_resources

        resources = cluster_resources(address)
    except Exception as exc:
        msg = (
            "Ray cluster capacity could not be verified; refusing to substitute "
            f"driver resources ({type(exc).__name__}: {exc})"
        )
        raise RuntimeError(msg) from exc

    cpus = float(resources.get("CPU", 0.0) or 0.0)
    gpus = float(resources.get("GPU", 0.0) or 0.0)
    memory = float(resources.get("memory", 0.0) or 0.0)
    if not math.isfinite(cpus) or cpus <= 0:
        msg = f"Ray cluster at {address!r} reported no positive finite CPU capacity"
        raise RuntimeError(msg)
    if not math.isfinite(gpus) or gpus < 0:
        msg = f"Ray cluster at {address!r} reported invalid GPU capacity {gpus!r}"
        raise RuntimeError(msg)
    if not math.isfinite(memory) or memory < 0:
        msg = f"Ray cluster at {address!r} reported invalid memory capacity {memory!r}"
        raise RuntimeError(msg)
    env.total_cpus = cpus
    env.gpu_count = max(0, int(gpus))
    env.has_gpu = env.gpu_count > 0
    if memory > 0:
        env.total_ram_gb = round(memory / (1024**3), 1)

    if not _verb_inputs._ray_address_is_local(address):
        # Ray exposes GPU count, not per-device VRAM/name. Driver GPU details
        # must not be projected onto a remote or unresolved cluster.
        env.gpu_mem_gb = 0.0
        env.gpu_names = []
    env.notes.append(f"resource planning bound to Ray cluster capacity at {address}: CPU={cpus:g}, GPU={gpus:g}")
    return env


def _run_pipeline(stages: list[Any], executor: Any, *, checkpoint_path: str | None = None) -> list[Any] | None:  # noqa: ANN401
    from nemo_curator.pipeline import Pipeline

    executor_config = getattr(executor, "config", None)
    if isinstance(executor_config, Mapping) and executor_config.get("ignore_failures"):
        msg = "Audio agent execution requires ignore_failures=False to prove output completeness"
        raise ValueError(msg)
    pipeline = Pipeline(name="audio_agent_run", stages=list(stages))
    # Keep the verb's stdout pure JSON: backend/worker logs (Ray forwards them to the
    # driver's stdout, e.g. verbose NeMo output) go to stderr during execution, so a
    # host parsing the CLI's stdout never sees them interleaved with the result.
    with contextlib.redirect_stdout(sys.stderr):
        from nemo_curator.stages.audio import agent as foundation
        from nemo_curator.stages.audio.common import ManifestWriterStage

        pipeline.build()
        writer_identity = (
            foundation.pipeline_identity(pipeline.stages)
            if checkpoint_path is not None and any(isinstance(stage, ManifestWriterStage) for stage in pipeline.stages)
            else None
        )
        prepared = []
        try:
            for stage in pipeline.stages:
                prepare = getattr(stage, "prepare_on_driver", None)
                if callable(prepare):
                    if isinstance(stage, ManifestWriterStage):
                        prepare(checkpoint_path=checkpoint_path, pipeline_identity=writer_identity)
                    else:
                        prepare(checkpoint_path=checkpoint_path)
                    prepared.append(stage)
            with _raise_on_failed_tasks(pipeline.stages):
                result = pipeline.run(executor, checkpoint_path=checkpoint_path)
            _finalize_audio_stages(pipeline.stages)
        except Exception:
            _abort_prepared_audio_stages(prepared)
            raise
        return result


@dataclass
class _CompleteOutputBatch:
    """Serializable guard around a configured stage's original batch method."""

    method: Callable[..., Any]
    owner: Any
    stage: Any
    stage_name: str

    def __call__(self, tasks: list[Any]) -> list[Any]:
        from nemo_curator.stages.audio.utils import begin_resource_probe, resource_probe_metrics
        from nemo_curator.tasks import FailedTask

        probe_enabled = getattr(self.stage, "RESOURCE_PROBE", False)
        gpu_probe_started = begin_resource_probe(self.stage) if probe_enabled else False
        input_size = sum(task.num_items for task in tasks) if probe_enabled else 0
        started = time.perf_counter()
        result = list(self.method(self.owner, tasks) if self.owner is not None else self.method(tasks))
        elapsed = time.perf_counter() - started
        if probe_enabled:
            metrics = resource_probe_metrics(
                gpu_probe_started=gpu_probe_started,
                process_time=elapsed,
                num_items=input_size,
            )
            # Explicit stage metrics take precedence over observational probes.
            metrics.update(self.stage._consume_custom_metrics())
            self.stage._log_metrics(metrics)
        if any(isinstance(task, FailedTask) for task in result):
            msg = f"Pipeline recorded failed tasks in stage {self.stage_name}; partial outputs are not complete"
            raise RuntimeError(msg)
        return result


@contextlib.contextmanager
def _raise_on_failed_tasks(stages: list[Any]) -> Iterator[None]:
    """Check stage outputs before shared adapters remove failure sentinels.

    Store the unbound function and its owner separately: pickling a bound method
    after replacing that method would otherwise resolve back to this guard.
    """
    prior = [(stage, "process_batch" in vars(stage), stage.process_batch) for stage in stages]
    try:
        for stage, _had_override, method in prior:
            stage.process_batch = _CompleteOutputBatch(
                method=getattr(method, "__func__", method),
                owner=getattr(method, "__self__", None),
                stage=stage,
                stage_name=stage.name,
            )
        yield
    finally:
        for stage, had_override, method in prior:
            if had_override:
                stage.process_batch = method
            else:
                vars(stage).pop("process_batch", None)


def _finalize_audio_stages(stages: list[Any]) -> None:
    errors = []
    for stage in stages:
        finalize = getattr(stage, "finalize", None)
        if callable(finalize):
            try:
                finalize()
            except Exception as exc:  # noqa: BLE001 - attempt every durable output finalizer
                errors.append(exc)
    if errors:
        raise errors[0]


def _abort_prepared_audio_stages(stages: list[Any]) -> None:
    from loguru import logger

    for stage in stages:
        abort = getattr(stage, "abort_on_driver", None)
        if callable(abort):
            try:
                abort()
            except Exception as exc:  # noqa: BLE001 - preserve the original execution failure
                logger.error(f"Audio stage {stage.name} driver cleanup failed: {exc}")


def _calibration_for_run(
    explicit: dict[str, Any] | None,
    config_hash: str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Pick the measurements to plan a run with, and say where they came from.

    Returns ``(calibration, provenance_note)``. What the caller passed always wins: the store
    is a fallback for the flag nobody remembered, never an override of the one they did pass.
    The note records the substitution on the resource plan, because a run must never be
    planned from evidence its operator cannot see. Whether each stored entry is actually
    *applied* stays the planner's call -- it drops any measurement stamped with a different
    machine, and says so in its own notes.
    """
    if explicit is not None:
        return explicit, None
    from nemo_curator.stages.audio.audio_agent import calibration_store

    stored = calibration_store.load(config_hash)
    if not stored:
        return None, None
    measured_at = stored.get("created_at")
    return stored, (
        "calibration: none passed; using the per-stage measurements a prior smoke of this "
        f"exact recipe stored{f' at {measured_at}' if measured_at else ''} "
        "(pass --calibration to override)"
    )


def _plan_resources(  # noqa: ANN202
    stages: list[Any],
    env_obj: Any,  # noqa: ANN401
    data_profile: dict[str, Any] | None,
    calibration: dict[str, Any] | None = None,
):
    """Run the deterministic resource planner over the built stages (1C.1 + 1C.2 calibration)."""
    from nemo_curator.stages.audio import agent as foundation
    from nemo_curator.stages.audio.audio_agent import planner

    # Explicit CPU recipes keep their CPU resource request and get conservative
    # worker sizing. A GPU reservation is never rewritten after confirmation;
    # an unavailable GPU remains an infeasible plan that requires a new recipe.
    worker_notes = planner.pin_cpu_model_workers(stages)
    contracts: list[Any] = []
    for st in stages:
        try:
            contracts.append(foundation.build_contract(st))
        except Exception:  # noqa: BLE001 - a stage that can't describe itself gets conservative defaults
            contracts.append(None)
    rplan = planner.plan(stages, contracts, env_obj, data_profile, calibration=calibration)
    for note in worker_notes:
        rplan.notes.append("worker_sizing: " + note)
    return rplan


def _make_executor(mode: str) -> Any:  # noqa: ANN401
    """A XennaExecutor with the planned execution_mode; None -> the pipeline default."""
    try:
        from nemo_curator.backends.xenna import XennaExecutor

        return XennaExecutor({"execution_mode": mode})
    except Exception:  # noqa: BLE001 - fall back to the pipeline's default executor (streaming)
        return None
