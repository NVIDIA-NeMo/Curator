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
# ruff: noqa: C901, PLR0912, PLR0915

"""Build terminal audio pipeline performance summaries."""

from __future__ import annotations

import contextlib
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

    from nemo_curator.tasks import Task
    from nemo_curator.utils.performance_utils import StagePerfStats

_GPU_SAMPLE_KEYS = frozenset({"gpu_util_pct", "gpu_mem_used_pct"})
DEFAULT_PERCENTILES = (50, 95)


def _gpu_sample_base(key: str) -> str:
    return key.split("::", 1)[0]


def _normalized_gpu_uuid(value: object) -> str:
    """Match backend-emitted metric suffixes without importing backend code."""
    text = value.decode() if isinstance(value, bytes) else str(value)
    return text.strip().lower().removeprefix("gpu-")


def seconds_to_hours(seconds: float) -> float:
    return float(seconds) / 3600.0


def bytes_to_mb(value: float) -> float:
    return float(value) / (1024.0 * 1024.0)


def add_ratio(entry: dict[str, Any], name: str, numerator: float, denominator: float) -> None:
    if numerator > 0 and denominator > 0:
        entry[name] = float(numerator) / float(denominator)


def _percentile(values: list[float], p: int) -> float:
    rank = (len(values) - 1) * p / 100.0
    low = int(rank)
    high = min(low + 1, len(values) - 1)
    return values[low] + (rank - low) * (values[high] - values[low])


def summarize_samples(
    values: Iterable[float],
    name: str,
    percentiles: Iterable[int] = DEFAULT_PERCENTILES,
) -> dict[str, float]:
    samples = sorted(float(value) for value in values)
    return {f"{name}_p{p}": _percentile(samples, p) for p in percentiles} if samples else {}


def audio_hours_per_gpu_hour(audio_seconds: float, gpu_seconds: float) -> float | None:
    if audio_seconds <= 0 or gpu_seconds <= 0:
        return None
    return audio_seconds / gpu_seconds


def estimate_wallclock_s(total_process_time_s: float, actor_count: float | None = None) -> float | None:
    if actor_count and actor_count > 0:
        return total_process_time_s / actor_count
    return total_process_time_s if total_process_time_s > 0 else None


@dataclass
class AudioStageSamples:
    """Per-invocation sample lists used for percentile derivation.

    Populated once per dedup'd invocation; only the accumulator writes these.
    """

    invocation_process_times_s: list[float] = field(default_factory=list)
    actor_idle_times_s: list[float] = field(default_factory=list)
    items_processed_per_invocation: list[float] = field(default_factory=list)
    batch_sizes: list[float] = field(default_factory=list)
    audio_duration_s_per_invocation: list[float] = field(default_factory=list)

    def add(self, perf: StagePerfStats) -> None:
        """Record one dedup'd invocation's per-call samples.

        GPU util is sampled per device and accumulated separately, so it is
        intentionally absent here -- these are actor/stage scalars only.
        """
        self.invocation_process_times_s.append(float(perf.process_time))
        self.actor_idle_times_s.append(float(perf.actor_idle_time))
        self.items_processed_per_invocation.append(float(perf.num_items_processed))

        custom = perf.custom_metrics or {}
        # Batch size proxy; collapses to 1 for single-task-per-invocation stages.
        batch_size = (
            custom.get("utterances_input")
            or custom.get("input_count")
            or custom.get("input_tasks")
            or perf.num_items_processed
        )
        with contextlib.suppress(TypeError, ValueError):
            self.batch_sizes.append(float(batch_size))

        audio_s = custom.get("audio_duration_s") or custom.get("audio_duration") or 0.0
        try:
            audio_s_f = float(audio_s)
        except (TypeError, ValueError):
            audio_s_f = 0.0
        if audio_s_f > 0:
            self.audio_duration_s_per_invocation.append(audio_s_f)

    def summarize(self, percentiles: tuple[int, ...] = DEFAULT_PERCENTILES) -> dict[str, float]:
        """Render the percentile-derived view (only populated keys)."""
        out: dict[str, float] = {}
        out.update(summarize_samples(self.invocation_process_times_s, "invocation_process_time_s", percentiles))
        out.update(summarize_samples(self.actor_idle_times_s, "queue_wait_s", percentiles))
        out.update(summarize_samples(self.batch_sizes, "batch_size", percentiles))
        out.update(summarize_samples(self.audio_duration_s_per_invocation, "audio_duration_s", percentiles))
        return out


@dataclass
class AudioStageCallerContext:
    """Optional caller-provided fields the accumulator cannot derive itself.

    A writer with NVML/DCGM/autoscaler snapshots passes these to populate the
    GPU/actor fields; defaults cause those fields to be omitted.
    """

    actor_count_samples: list[float] = field(default_factory=list)
    gpu_util_pct_samples: list[float] = field(default_factory=list)
    gpu_hours: float = 0.0
    setup_time_s_total: float = 0.0
    wallclock_s: float | None = None  # overrides estimate if provided


def _valid_audio_duration(data: object, duration_key: str) -> float | None:
    if not isinstance(data, dict) or duration_key not in data:
        return None
    raw_duration = data[duration_key]
    if isinstance(raw_duration, bool):
        return None
    try:
        seconds = float(raw_duration)
    except (TypeError, ValueError):
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def _build_stage_summary(  # noqa: PLR0913
    stage_totals: dict[str, float],
    custom_totals: dict[str, float],
    samples: AudioStageSamples | None = None,
    caller_context: AudioStageCallerContext | None = None,
    stage_identity: dict[str, Any] | None = None,
    actor_breakdown: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Render one stage's summary in the proposed pipeline-perf shape.

    Combines framework scalar totals, the dedup'd custom-metric superset,
    per-invocation sample percentiles, and caller-provided GPU/actor context.
    """
    entry: dict[str, Any] = {
        "total_process_time_s": stage_totals.get("process_time", 0.0),
        "total_actor_idle_time_s": stage_totals.get("actor_idle_time", 0.0),
        "total_items_processed": stage_totals.get("num_items_processed", 0.0),
        "invocation_count": stage_totals.get("invocation_count", 0.0),
    }

    invocation_count = stage_totals.get("invocation_count", 0.0)
    total_time = stage_totals.get("process_time", 0.0)
    total_items = stage_totals.get("num_items_processed", 0.0)

    custom_sums = {
        key: float(value)
        for key, value in custom_totals.items()
        if isinstance(value, (int, float, bool)) and value != 0 and _gpu_sample_base(key) not in _GPU_SAMPLE_KEYS
    }
    # A measured empty boundary is different from an unavailable boundary.
    # Preserve these keys when a producer explicitly emitted zero.
    for boundary_key in (
        "pipeline_input_rows",
        "pipeline_input_audio_s",
        "pipeline_input_duration_rows",
        "pipeline_output_rows",
        "pipeline_output_audio_s",
        "pipeline_output_duration_rows",
    ):
        if boundary_key in custom_totals:
            custom_sums[boundary_key] = float(custom_totals[boundary_key])

    add_ratio(entry, "avg_invocation_time_s", total_time, invocation_count)
    add_ratio(entry, "throughput_items_per_s", total_items, total_time)

    # caller context: wallclock + GPU + actor
    ctx = caller_context or AudioStageCallerContext()
    actor_count_p50 = None
    if ctx.actor_count_samples:
        actor_count_p50 = summarize_samples(ctx.actor_count_samples, "actor_count").get("actor_count_p50")
    identity_actor_count = float((stage_identity or {}).get("actor_count", 0.0) or 0.0)
    wallclock_actor_count = actor_count_p50 or identity_actor_count or None

    wallclock_s = (
        ctx.wallclock_s
        if ctx.wallclock_s is not None
        else estimate_wallclock_s(
            total_process_time_s=total_time,
            actor_count=wallclock_actor_count,
        )
    )
    if ctx.gpu_hours > 0:
        entry["gpu_hours"] = ctx.gpu_hours
    if ctx.setup_time_s_total > 0:
        entry["setup_time_s_total"] = ctx.setup_time_s_total
    entry.update(summarize_samples(ctx.actor_count_samples, "actor_count"))
    entry.update(summarize_samples(ctx.gpu_util_pct_samples, "gpu_util_pct"))

    # Identity-driven topology + per-actor scheduling breakdown (keyed by
    # actor_id for GPU and CPU stages). Hardware gpu_hours/device_name deferred
    # to the NVML/DCGM proposal.
    if stage_identity:
        entry.update(stage_identity)
    if actor_breakdown:
        entry["per_actor"] = actor_breakdown

    if not custom_sums and not samples:
        return entry

    if custom_sums:
        entry["custom_metrics_sum"] = custom_sums

    if samples is not None:
        entry.update(samples.summarize())

    # ----- audio-domain throughput composites -----
    def metric(*names: str) -> float:
        return next(
            (
                float(custom_totals[name])
                for name in names
                if isinstance(custom_totals.get(name), (int, float, bool)) and custom_totals[name] != 0
            ),
            0.0,
        )

    audio_seconds = metric("audio_duration_s", "audio_duration", "duration")
    inference_time = metric("inference_time_s", "inference_time")
    output_tokens = metric("output_tokens")
    output_chars = metric("output_chars")
    waveform_mb = bytes_to_mb(metric("waveform_bytes"))

    # Both default to the audio duration the stage saw; filter stages may
    # override audio_hours_out via custom_metrics.
    if audio_seconds > 0:
        entry["audio_hours_in"] = seconds_to_hours(audio_seconds)
        entry["audio_hours_out"] = seconds_to_hours(audio_seconds)

    gpu_count = float((stage_identity or {}).get("gpu_count", 0.0) or 0.0)
    gpu_seconds = ctx.gpu_hours * 3600.0 if ctx.gpu_hours > 0 else (wallclock_s or 0.0) * gpu_count
    if gpu_seconds > 0:
        ah_per_gpu_h = audio_hours_per_gpu_hour(audio_seconds, gpu_seconds)
        if ah_per_gpu_h is not None:
            entry["audio_hours_per_gpu_hour"] = ah_per_gpu_h

    # Two efficiency views: overall (audio per total process-time, incl. overhead)
    # and inference-only. inference_compute_fraction is the model-vs-overhead share.
    add_ratio(entry, "throughput_audio_s_per_process_s", audio_seconds, total_time)
    add_ratio(entry, "throughput_audio_s_per_inference_s", audio_seconds, inference_time)
    add_ratio(entry, "inference_compute_fraction", inference_time, total_time)
    add_ratio(entry, "avg_audio_s_per_item", audio_seconds, total_items)
    add_ratio(entry, "throughput_output_tokens_per_process_s", output_tokens, total_time)
    add_ratio(entry, "throughput_output_tokens_per_inference_s", output_tokens, inference_time)
    add_ratio(entry, "throughput_output_chars_per_process_s", output_chars, total_time)
    add_ratio(entry, "throughput_output_chars_per_inference_s", output_chars, inference_time)
    add_ratio(entry, "throughput_waveform_mb_per_process_s", waveform_mb, total_time)
    adapter_calls = metric("adapter_inference_calls")
    adapter_items = metric("adapter_inference_items")
    if adapter_calls > 0:
        entry["adapter_inference_call_count"] = adapter_calls
        entry["adapter_inference_items"] = adapter_items
    add_ratio(
        entry,
        "avg_adapter_inference_batch_size",
        adapter_items,
        adapter_calls,
    )
    add_ratio(
        entry,
        "avg_audio_s_per_adapter_inference_call",
        audio_seconds,
        adapter_calls,
    )
    add_ratio(
        entry,
        "adapter_inference_calls_per_stage_invocation",
        adapter_calls,
        invocation_count,
    )

    # ----- pipeline-structure ratios -----
    add_ratio(entry, "output_tasks_per_input_task", metric("output_tasks"), metric("input_tasks"))

    # Generic item-fate aliases: populate from whichever stage-specific
    # counter is non-zero.
    items_skipped = metric("utterances_skipped", "skipped_short")
    items_filtered = metric("utterances_filtered", "filtered_count")
    items_recovered = metric("utterances_recovered")
    if items_skipped > 0:
        entry["items_skipped"] = items_skipped
    if items_filtered > 0:
        entry["items_filtered"] = items_filtered
    if items_recovered > 0:
        entry["items_recovered"] = items_recovered
    if output_tokens > 0:
        entry["output_tokens"] = output_tokens

    # filter/tagging stages: per-input-utterance ratios
    utterances_input = metric("utterances_input", "input_tasks")
    if utterances_input > 0:
        for metric_name in (
            "utterances_selected",
            "utterances_skipped",
            "utterances_processed",
            "utterances_eligible",
            "utterances_restored",
            "utterances_kept_as_is",
            "utterances_filtered",
            "utterances_newly_flagged",
            "utterances_recovered",
            "pnc_rejected",
            "empty_after_regex",
            "wrong_language",
            "low_probability",
        ):
            value = metric(metric_name)
            add_ratio(entry, f"{metric_name}_per_input_utterance", value, utterances_input)

    return entry


@dataclass(repr=False)
class AudioPerformanceSummary:
    """Accumulate and summarise audio task performance metrics.

    Writer-independent: a terminal stage calls ``record_task`` per output task,
    then writes ``build_summary()`` wherever its output contract requires.
    """

    duration_key: str = "duration"
    _stage_totals: dict[str, dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(float)),
    )
    _stage_custom_totals: dict[str, dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(float)),
    )
    _stage_samples: dict[str, AudioStageSamples] = field(
        default_factory=lambda: defaultdict(AudioStageSamples),
    )
    _stage_names: dict[str, str] = field(default_factory=dict)
    _stage_window_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    _seen_perf_invocations: set[str] = field(default_factory=set)
    # Per-(stage, actor) scheduling breakdown for any record with a resolved
    # actor_id (GPU and CPU stages). GPU actors also carry physical address +
    # NVML util/mem percentiles.
    _stage_actor_samples: dict[str, dict[str, AudioStageSamples]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(AudioStageSamples)),
    )
    _stage_actor_items: dict[str, dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(float)),
    )
    _stage_actor_audio_s: dict[str, dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(float)),
    )
    _stage_actor_location: dict[str, dict[str, dict[str, Any]]] = field(
        default_factory=lambda: defaultdict(dict),
    )
    # Per-GPU NVML samples nested stage -> actor -> address ("<host>:<idx>"),
    # rolled up under each actor's ``gpus`` block. ``_gpu_unit_meta`` holds
    # per-address metadata (gpu_index, gpu_uuid).
    _stage_actor_gpu_util: dict[str, dict[str, dict[str, list[float]]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
    )
    _stage_actor_gpu_mem: dict[str, dict[str, dict[str, list[float]]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
    )
    _gpu_unit_meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    # _stage_gpus: per-actor addresses ("<host>:<idx,idx>"); _stage_gpu_units:
    # individual devices ("<host>:<idx>") so gpu_count is true under tensor-parallel.
    _stage_gpus: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    _stage_gpu_units: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    _stage_actors: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    _actor_node: dict[str, str] = field(default_factory=dict)
    _total_utterances: int = 0
    _duration_utterances: int = 0
    _total_audio_seconds: float = 0.0
    _dataset_names: set[str] = field(default_factory=set)
    _output_column_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _wall_start_s: float = field(default_factory=time.perf_counter)

    @property
    def total_utterances(self) -> int:
        return self._total_utterances

    @property
    def total_audio_seconds(self) -> float:
        return self._total_audio_seconds

    @property
    def duration_utterances(self) -> int:
        return self._duration_utterances

    # -----------------------------------------------------------------------
    # Recording
    # -----------------------------------------------------------------------

    def record_task(self, task: Task, *, include_stage_perf: bool = True) -> None:
        """Record one audio task and optionally its attached stage perf chain."""
        audio_seconds = _valid_audio_duration(getattr(task, "data", None), self.duration_key)
        self._total_utterances += 1
        self._duration_utterances += int(audio_seconds is not None)
        self._total_audio_seconds += audio_seconds or 0.0
        dataset_name = str(getattr(task, "dataset_name", "") or "").strip()
        if dataset_name:
            self._dataset_names.add(dataset_name)
        data = getattr(task, "data", None)
        if isinstance(data, dict):
            for key in data:
                self._output_column_counts[str(key)] += 1

        if include_stage_perf:
            self.record_stage_perf(getattr(task, "_stage_perf", []) or [])

    @staticmethod
    def _fingerprint_perf(perf: StagePerfStats) -> str:
        """Deterministic fingerprint of a ``StagePerfStats`` value tuple.

        Fallback dedup key when ``invocation_id`` is unset: the same record is
        seen once per emitted downstream task, so an N-task invocation would be
        counted N times. Collisions (distinct invocations with byte-equal
        timings and custom metrics) are not a practical concern.
        """
        custom = sorted((perf.custom_metrics or {}).items())
        return repr(
            (
                perf.stage_name,
                getattr(perf, "stage_id", ""),
                getattr(perf, "actor_id", ""),
                getattr(perf, "node_id", ""),
                getattr(perf, "gpu_id", ""),
                getattr(perf, "physical_address", ""),
                round(perf.process_time, 9),
                round(perf.actor_idle_time, 9),
                perf.num_items_processed,
                round(float(getattr(perf, "window_start_s", 0.0)), 9),
                round(float(getattr(perf, "window_end_s", 0.0)), 9),
                tuple((k, round(float(v), 9)) for k, v in custom),
            )
        )

    def record_stage_perf(self, stage_perf_list: list[StagePerfStats]) -> None:
        """Accumulate ``StagePerfStats``, deduplicating repeat sightings.

        Dedup key is ``invocation_id`` when wired, else a synthetic value-tuple
        fingerprint. After dedup, each record feeds stage scalar totals,
        custom-metric sums, and per-invocation samples (for p50/p95).
        """
        for perf in stage_perf_list:
            if not all(
                hasattr(perf, attr)
                for attr in ("stage_name", "process_time", "actor_idle_time", "num_items_processed")
            ):
                # Legacy/custom stages sometimes attach dictionaries. Perf is
                # optional observability, so malformed records must not break
                # terminal manifest writing.
                continue
            stage_key = str(getattr(perf, "stage_id", "") or perf.stage_name)
            self._stage_names.setdefault(stage_key, str(perf.stage_name))
            invocation_id = getattr(perf, "invocation_id", "") or self._fingerprint_perf(perf)
            invocation_id = f"{stage_key}:{invocation_id}"
            if invocation_id in self._seen_perf_invocations:
                continue
            self._seen_perf_invocations.add(invocation_id)

            totals = self._stage_totals[stage_key]
            totals["process_time"] += perf.process_time
            totals["actor_idle_time"] += perf.actor_idle_time
            totals["num_items_processed"] += perf.num_items_processed
            totals["invocation_count"] += 1

            for key, value in (perf.custom_metrics or {}).items():
                if _gpu_sample_base(key) in _GPU_SAMPLE_KEYS:
                    continue
                if isinstance(value, (int, float, bool)):
                    self._stage_custom_totals[stage_key][key] += float(value)

            self._stage_samples[stage_key].add(perf)
            window_start_s = float(getattr(perf, "window_start_s", 0.0) or 0.0)
            window_end_s = float(getattr(perf, "window_end_s", 0.0) or 0.0)
            if window_end_s >= window_start_s > 0:
                previous = self._stage_window_bounds.get(stage_key)
                self._stage_window_bounds[stage_key] = (
                    min(previous[0], window_start_s) if previous else window_start_s,
                    max(previous[1], window_end_s) if previous else window_end_s,
                )
            self._record_actor_breakdown(perf, stage_key)

    def _record_actor_breakdown(self, perf: StagePerfStats, stage_key: str) -> None:
        """Accumulate the per-(stage, actor) scheduling breakdown.

        Keyed by ``actor_id`` so every actor-backed stage (GPU or CPU) reports
        per-actor metrics; GPU actors also contribute their physical address and
        device units. No-op for records without a resolved ``actor_id``.
        """
        actor_id = (getattr(perf, "actor_id", "") or "").strip()
        if not actor_id:
            return
        node_id = (getattr(perf, "node_id", "") or "").strip()
        self._stage_actors[stage_key].add(actor_id)
        if node_id:
            self._actor_node.setdefault(actor_id, node_id)
        self._stage_actor_samples[stage_key][actor_id].add(perf)
        self._stage_actor_items[stage_key][actor_id] += float(perf.num_items_processed)
        custom = perf.custom_metrics or {}
        audio_s = custom.get("audio_duration_s") or custom.get("audio_duration") or 0.0
        with contextlib.suppress(TypeError, ValueError):
            self._stage_actor_audio_s[stage_key][actor_id] += float(audio_s)
        # GPU topology: physical address + device units (gpu_count true under TP).
        physical_address = (getattr(perf, "physical_address", "") or "").strip()
        host = physical_address.rsplit(":", 1)[0] if physical_address else (node_id or "node")
        if physical_address:
            self._stage_gpus[stage_key].add(physical_address)
            for idx in getattr(perf, "gpu_indices", None) or ():
                self._stage_gpu_units[stage_key].add(f"{host}:{idx}")
        self._record_gpu_samples(stage_key, actor_id, host, perf)
        location = self._actor_location_fields(perf)
        if location:
            self._stage_actor_location[stage_key][actor_id] = location

    def _record_gpu_samples(self, stage_name: str, actor_id: str, host: str, perf: StagePerfStats) -> None:
        """Fold per-GPU NVML samples (``<base>::<uuid>``) onto a physical address.

        Maps each sample's normalized UUID back to the actor's physical GPU index
        (via parallel ``gpu_indices``/``gpu_uuids``) so it lands on the canonical
        ``<host>:<idx>`` address; unmappable UUIDs fall back to ``<host>:<uuid>``.
        """
        custom = perf.custom_metrics or {}
        if not any(_gpu_sample_base(k) in _GPU_SAMPLE_KEYS for k in custom):
            return
        gpu_indices = list(getattr(perf, "gpu_indices", None) or [])
        gpu_uuids = list(getattr(perf, "gpu_uuids", None) or [])
        uuid_to_index = {_normalized_gpu_uuid(u): idx for u, idx in zip(gpu_uuids, gpu_indices, strict=False)}
        uuid_to_raw = {_normalized_gpu_uuid(u): u for u in gpu_uuids}
        for key, value in custom.items():
            base = _gpu_sample_base(key)
            if base not in _GPU_SAMPLE_KEYS or "::" not in key:
                continue
            try:
                sample = float(value)
            except (TypeError, ValueError):
                continue
            uuid_key = key.split("::", 1)[1]
            index = uuid_to_index.get(uuid_key)
            address = f"{host}:{index}" if index is not None else f"{host}:{uuid_key}"
            self._stage_gpu_units[stage_name].add(address)
            target = self._stage_actor_gpu_util if base == "gpu_util_pct" else self._stage_actor_gpu_mem
            target[stage_name][actor_id][address].append(sample)
            meta = self._gpu_unit_meta.setdefault(address, {})
            if index is not None and "gpu_index" not in meta:
                meta["gpu_index"] = int(index)
            if uuid_key in uuid_to_raw and "gpu_uuid" not in meta:
                meta["gpu_uuid"] = uuid_to_raw[uuid_key]

    @staticmethod
    def _actor_location_fields(perf: StagePerfStats) -> dict[str, Any]:
        """Additive per-actor metadata (GPU actors carry physical address).

        ``node_id`` is folded in by the builder, not here.
        """
        block: dict[str, Any] = {}
        physical_address = getattr(perf, "physical_address", "") or ""
        pod_ip = getattr(perf, "pod_ip", "") or ""
        hostname = getattr(perf, "hostname", "") or ""
        gpu_indices = getattr(perf, "gpu_indices", None) or []
        gpu_uuids = getattr(perf, "gpu_uuids", None) or []
        if physical_address:
            block["physical_address"] = physical_address
        if pod_ip:
            block["pod_ip"] = pod_ip
        if hostname:
            block["hostname"] = hostname
        if gpu_indices:
            block["gpu_indices"] = [int(idx) for idx in gpu_indices]
        if gpu_uuids:
            block["gpu_uuids"] = list(gpu_uuids)
        return block

    # -----------------------------------------------------------------------
    # Building the published summary
    # -----------------------------------------------------------------------

    def _stage_identity_meta(self, stage_key: str) -> dict[str, Any]:
        """Topology labels for a stage: gpu_addresses, gpu_count, actor_count.

        ``gpu_count`` counts distinct physical devices (a TP actor on 2 GPUs
        counts as 2). Keys are omitted for stages without resolved identity.
        """
        meta: dict[str, Any] = {}
        stage_name = self._stage_names.get(stage_key, stage_key)
        if stage_key != stage_name:
            meta["stage_name"] = stage_name
        addresses = sorted(self._stage_gpus.get(stage_key, set()))
        if addresses:
            meta["gpu_addresses"] = addresses
            meta["gpu_count"] = float(len(self._stage_gpu_units.get(stage_key, addresses)))
        actors = self._stage_actors.get(stage_key, set())
        if actors:
            meta["actor_count"] = float(len(actors))
        return meta

    def _build_per_actor(self, stage_name: str) -> dict[str, dict[str, Any]]:
        """Per-actor scheduling breakdown for a stage (GPU and CPU alike).

        Keyed by ``actor_id``; empty when no actor identity was resolved. Each
        entry carries node_id, items_processed, audio_hours_in, and
        batch_size/queue_wait percentiles. GPU actors also carry physical_address,
        gpu_indices/gpu_uuids, and a nested ``gpus`` map of per-device NVML
        percentiles (only when the worker ran a GPU sampler).
        """
        actor_samples = self._stage_actor_samples.get(stage_name, {})
        if not actor_samples:
            return {}
        per_actor: dict[str, dict[str, Any]] = {}
        for actor_id in sorted(actor_samples):
            block: dict[str, Any] = {}
            node_id = self._actor_node.get(actor_id)
            if node_id:
                block["node_id"] = node_id
            items = self._stage_actor_items.get(stage_name, {}).get(actor_id, 0.0)
            if items:
                block["items_processed"] = items
            audio_s = self._stage_actor_audio_s.get(stage_name, {}).get(actor_id, 0.0)
            if audio_s > 0:
                block["audio_hours_in"] = seconds_to_hours(audio_s)
            summary = actor_samples[actor_id].summarize()
            for key in ("batch_size_p50", "batch_size_p95", "queue_wait_s_p50", "queue_wait_s_p95"):
                if key in summary:
                    block[key] = summary[key]
            location = self._stage_actor_location.get(stage_name, {}).get(actor_id)
            if location:
                block.update(location)
            gpus = self._build_actor_gpus(stage_name, actor_id)
            if gpus:
                block["gpus"] = gpus
            per_actor[actor_id] = block
        return per_actor

    def _build_actor_gpus(self, stage_name: str, actor_id: str) -> dict[str, dict[str, Any]]:
        """Per-physical-GPU NVML breakdown for one actor, keyed by ``<host>:<idx>``.

        Each device carries gpu_index/gpu_uuid metadata and util/mem percentiles
        from its own samples. Empty when the actor ran no GPU sampler.
        """
        util_by_addr = self._stage_actor_gpu_util.get(stage_name, {}).get(actor_id, {})
        mem_by_addr = self._stage_actor_gpu_mem.get(stage_name, {}).get(actor_id, {})
        addresses = sorted(set(util_by_addr) | set(mem_by_addr))
        gpus: dict[str, dict[str, Any]] = {}
        for address in addresses:
            block: dict[str, Any] = dict(self._gpu_unit_meta.get(address, {}))
            block.update(summarize_samples(util_by_addr.get(address, []), "gpu_util_pct"))
            block.update(summarize_samples(mem_by_addr.get(address, []), "gpu_mem_used_pct"))
            gpus[address] = block
        return gpus

    def build_stage_summaries(
        self,
        stage_caller_context: dict[str, AudioStageCallerContext] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Build per-stage aggregate summaries from accumulated metrics."""
        ctx_by_stage = stage_caller_context or {}
        result: dict[str, dict[str, Any]] = {}
        for stage_key, totals in self._stage_totals.items():
            context = ctx_by_stage.get(stage_key)
            if context is None:
                context = ctx_by_stage.get(self._stage_names.get(stage_key, stage_key))
            window_bounds = self._stage_window_bounds.get(stage_key)
            if window_bounds:
                context = replace(
                    context or AudioStageCallerContext(),
                    wallclock_s=max(window_bounds[1] - window_bounds[0], 0.0),
                )
            result[stage_key] = _build_stage_summary(
                dict(totals),
                dict(self._stage_custom_totals.get(stage_key, {})),
                samples=self._stage_samples.get(stage_key),
                caller_context=context,
                stage_identity=self._stage_identity_meta(stage_key),
                actor_breakdown=self._build_per_actor(stage_key),
            )
            display_name = self._stage_names.get(stage_key, stage_key)
            if stage_key != display_name:
                result[stage_key]["stage_id"] = stage_key
                result[stage_key]["stage_name"] = display_name
        return result

    def build_summary(  # noqa: PLR0913
        self,
        *,
        extra_stage_summaries: dict[str, dict[str, Any]] | None = None,
        wall_time_s: float | None = None,
        run_id: str | None = None,
        executor: str | None = None,
        pipeline_metadata: dict[str, Any] | None = None,
        stage_caller_context: dict[str, AudioStageCallerContext] | None = None,
    ) -> dict[str, Any]:
        """Build the full audio pipeline performance summary.

        Top-level fields match the pipeline-perf shape (run_id, executor,
        input_hours, output_hours, rows_in, rows_out, stages). Input values
        come from the first stage that declares the standard input-boundary
        counters; output values come from the last stage that declares the
        standard output-boundary counters.
        """
        resolved_wall_time_s = (
            max(time.perf_counter() - self._wall_start_s, 0.0) if wall_time_s is None else max(wall_time_s, 0.0)
        )
        stages_summary = self.build_stage_summaries(stage_caller_context)
        if extra_stage_summaries:
            stages_summary.update(extra_stage_summaries)

        rows_in, input_audio_s, input_duration_rows = self._pipeline_boundary(
            stages_summary,
            rows_key="pipeline_input_rows",
            audio_key="pipeline_input_audio_s",
            duration_rows_key="pipeline_input_duration_rows",
        )
        rows_out, output_audio_s, output_duration_rows = self._pipeline_boundary(
            stages_summary,
            rows_key="pipeline_output_rows",
            audio_key="pipeline_output_audio_s",
            duration_rows_key="pipeline_output_duration_rows",
            reverse=True,
        )
        if rows_out is None and self._total_utterances > 0:
            rows_out = float(self._total_utterances)
            output_audio_s = self._total_audio_seconds
            output_duration_rows = float(self._duration_utterances)

        input_complete = (
            rows_in is not None
            and input_audio_s is not None
            and input_duration_rows is not None
            and (rows_in == 0 or input_duration_rows >= rows_in)
        )
        output_complete = (
            rows_out is not None
            and output_audio_s is not None
            and output_duration_rows is not None
            and (rows_out == 0 or output_duration_rows >= rows_out)
        )
        input_hours = seconds_to_hours(input_audio_s) if input_complete and input_audio_s is not None else None
        output_hours = seconds_to_hours(output_audio_s) if output_complete and output_audio_s is not None else None

        summary: dict[str, Any] = {
            # proposed-structure top-level
            "run_id": run_id or "",
            "executor": executor or "",
            "input_hours": input_hours,
            "output_hours": output_hours,
            "rows_in": rows_in,
            "rows_out": rows_out,
            "input_duration_rows": input_duration_rows,
            "output_duration_rows": output_duration_rows,
            "total_audio_seconds": output_audio_s if output_complete else None,
            "total_audio_hours": output_hours,
            "pipeline_wall_time_s": resolved_wall_time_s,
            "perf_invocations_counted": len(self._seen_perf_invocations),
            "pipeline": dict(pipeline_metadata or {}),
            "dataset_names": sorted(self._dataset_names),
            "output_schema": {
                "columns": sorted(self._output_column_counts),
                "column_row_counts": dict(sorted(self._output_column_counts.items())),
            },
            "stages": stages_summary,
        }

        # Cluster-level rollup (scheduling only). Hardware rollups are deferred
        # to the NVML/DCGM proposal; only identity-derivable fields emitted here.
        pipeline_throughput: dict[str, Any] = {}
        if resolved_wall_time_s > 0 and output_complete and output_audio_s is not None and output_audio_s > 0:
            pipeline_throughput["audio_hours_per_wallclock_hour"] = seconds_to_hours(
                output_audio_s
            ) / seconds_to_hours(resolved_wall_time_s)
        all_addresses = sorted({addr for addrs in self._stage_gpus.values() for addr in addrs})
        if all_addresses:
            all_units = {unit for units in self._stage_gpu_units.values() for unit in units}
            pipeline_throughput["gpu_addresses"] = all_addresses
            pipeline_throughput["gpu_count"] = float(len(all_units or all_addresses))
            if output_complete and output_audio_s is not None:
                gpu_seconds = resolved_wall_time_s * pipeline_throughput["gpu_count"]
                pipeline_gpu_efficiency = audio_hours_per_gpu_hour(output_audio_s, gpu_seconds)
                if pipeline_gpu_efficiency is not None:
                    pipeline_throughput["audio_hours_per_gpu_hour"] = pipeline_gpu_efficiency
        if pipeline_throughput:
            summary["pipeline_throughput"] = pipeline_throughput

        return summary

    @staticmethod
    def _pipeline_boundary(
        stages_summary: dict[str, dict[str, Any]],
        *,
        rows_key: str,
        audio_key: str,
        duration_rows_key: str,
        reverse: bool = False,
    ) -> tuple[float | None, float | None, float | None]:
        """Return rows/audio from the first matching stage in traversal order."""
        stage_values = list(stages_summary.values())
        if reverse:
            stage_values.reverse()
        for stage_summary in stage_values:
            custom = stage_summary.get("custom_metrics_sum")
            if not isinstance(custom, dict) or rows_key not in custom:
                continue
            rows = float(custom.get(rows_key, 0.0) or 0.0)
            if audio_key not in custom:
                return rows, None, 0.0
            audio_s = float(custom.get(audio_key, 0.0) or 0.0)
            duration_rows = float(custom.get(duration_rows_key, 0.0) or 0.0) if duration_rows_key in custom else rows
            return rows, audio_s, duration_rows
        return None, None, None
