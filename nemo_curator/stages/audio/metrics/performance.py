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

"""Reusable audio pipeline performance summary helpers.

Audio stages emit counters/timings via ``_log_metrics()``; backends attach
them to ``Task._stage_perf`` as ``StagePerfStats``. Terminal stages feed those
into ``AudioPerformanceSummary`` to build the published ``perf_summary.json``.

Key types: ``AudioStageMetrics`` (superset of every scalar custom metric),
``AudioStageSamples`` (per-invocation samples for percentiles),
``AudioStageCallerContext`` (GPU/actor fields not derivable from perf stats),
and ``AudioPerformanceSummary`` (the dedup'ing accumulator). All
post-processing lives in ``performance_utils.py``; this file only collects.
"""

from __future__ import annotations

import contextlib
import time
from collections import defaultdict
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

from nemo_curator.stages.audio.metrics.performance_utils import (
    add_ratio,
    audio_hours_per_gpu_hour,
    bytes_to_mb,
    estimate_wallclock_s,
    seconds_to_hours,
    summarize_samples,
)
from nemo_curator.utils.gpu_sampler import norm_uuid

if TYPE_CHECKING:
    from nemo_curator.tasks import Task
    from nemo_curator.utils.performance_utils import StagePerfStats

# GPU-util metrics ride custom_metrics as ``<base>::<uuid>``, sampled per GPU
# and summarized as percentiles -- excluded from scalar totals so they are
# never summed into a meaningless aggregate.
_GPU_SAMPLE_KEYS = frozenset({"gpu_util_pct", "gpu_mem_used_pct"})
_MAX_CUSTOM_METRIC_KEYS = frozenset(
    {"expected_stage_gpu_count", "expected_stage_worker_count", "expected_worker_gpu_count"}
)


def _gpu_sample_base(key: str) -> str:
    """Base metric name of a (possibly UUID-namespaced) GPU sample key."""
    return key.split("::", 1)[0]


@dataclass
class AudioStageMetrics:
    """Superset of every scalar custom metric the audio pipeline emits.

    Stages populate only relevant fields via ``_log_metrics``; the accumulator
    sums them and rebuilds an ``AudioStageMetrics`` per stage. Default 0.0 means
    "not emitted"; ``to_dict()`` strips zeros so JSON only carries populated
    keys. Adding a metric is one field here plus the producer's ``_log_metrics``.
    """

    # ----- universal counters -----
    input_tasks: float = 0.0
    output_tasks: float = 0.0
    # Actor-pattern fix for stages the framework's num_items_processed cannot
    # count (e.g. discovery synthesises work from config, so input is seen as 0).
    total_items_emitted: float = 0.0

    # ----- audio volume scalars -----
    audio_duration_s: float = 0.0
    # Legacy aliases for older stages; new stages emit ``audio_duration_s``.
    audio_duration: float = 0.0
    duration: float = 0.0
    input_duration: float = 0.0
    filtered_dur: float = 0.0
    waveform_bytes: float = 0.0
    # "Truthful bytes loaded" for stages that load data themselves (tar reader)
    # where framework's input_data_size_mb is unavailable. Producer-opt-in.
    bytes_loaded: float = 0.0

    # ----- text/transcript output -----
    output_chars: float = 0.0
    output_tokens: float = 0.0
    turn1_output_tokens: float = 0.0
    turn2_output_tokens: float = 0.0

    # ----- inference timing -----
    inference_time_s: float = 0.0
    inference_time: float = 0.0  # legacy alias
    adapter_inference_calls: float = 0.0
    adapter_inference_items: float = 0.0

    # ----- model-side internal timers / counters -----
    model_turn1_prep_time_s: float = 0.0
    model_turn1_generation_time_s: float = 0.0
    model_turn2_prep_time_s: float = 0.0
    model_turn2_generation_time_s: float = 0.0
    model_turn1_valid_inputs: float = 0.0
    model_turn2_valid_inputs: float = 0.0
    model_utterances_skipped_preprocess: float = 0.0

    # ----- shard reader (NemoTarShardReaderStage) -----
    input_shards: float = 0.0
    output_utterances: float = 0.0
    utterances_emitted: float = 0.0
    manifest_entries: float = 0.0
    manifest_read_time_s: float = 0.0
    tar_members_seen: float = 0.0
    audio_members_decoded: float = 0.0
    tar_open_time_s: float = 0.0
    audio_decode_time_s: float = 0.0
    reader_total_time_s: float = 0.0
    corrupt_audio_count: float = 0.0
    duration_filtered_count: float = 0.0
    utterance_limit_hit: float = 0.0

    # ----- shard discovery -----
    corpora_seen: float = 0.0
    shards_seen: float = 0.0
    shards_emitted: float = 0.0
    shards_skipped_completed: float = 0.0
    discovery_time_s: float = 0.0

    # ----- inference / filter / tagging utterance accounting -----
    utterances_input: float = 0.0
    utterances_processed: float = 0.0
    utterances_skipped: float = 0.0
    utterances_selected: float = 0.0
    utterances_eligible: float = 0.0
    utterances_restored: float = 0.0
    utterances_kept_as_is: float = 0.0
    utterances_filtered: float = 0.0
    utterances_newly_flagged: float = 0.0
    utterances_recovered: float = 0.0

    # ----- text-filter rejection reasons -----
    pnc_rejected: float = 0.0
    empty_after_regex: float = 0.0
    wrong_language: float = 0.0
    low_probability: float = 0.0

    # ----- preserve-by-value / generic batch filter -----
    input_count: float = 0.0
    output_count: float = 0.0
    filtered_count: float = 0.0

    # ----- manifest reader -----
    manifests_read: float = 0.0
    entries_read: float = 0.0

    # ----- ALM data overlap / builder -----
    filter_time: float = 0.0
    input_windows: float = 0.0
    output_windows: float = 0.0
    segments_processed: float = 0.0
    windows_created: float = 0.0

    # ----- audio split / merge / resample / NeMo ASR align -----
    splits_produced: float = 0.0
    splits_joined: float = 0.0
    words_aligned: float = 0.0
    segments_merged: float = 0.0
    skipped_conversion: float = 0.0
    entries_processed: float = 0.0
    files_transcribed: float = 0.0
    process_time: float = 0.0  # legacy custom timer some stages emit

    # ----- speaker diarization -----
    segments_detected: float = 0.0
    overlap_segments_detected: float = 0.0
    speakers_detected: float = 0.0

    # ----- VAD -----
    vad_segments_detected: float = 0.0
    skipped_short: float = 0.0

    # ----- sharded manifest writer (ShardedManifestWriterStage) -----
    writer_process_calls: float = 0.0
    writer_invocation_count: float = 0.0
    writer_items_processed: float = 0.0
    manifest_write_time_s: float = 0.0
    done_marker_write_time_s: float = 0.0
    perf_write_time_s: float = 0.0

    # forward-compat: any emitted scalar this dataclass doesn't know
    extras: dict[str, float] = field(default_factory=dict)

    @classmethod
    def known_field_names(cls) -> set[str]:
        return {f.name for f in fields(cls) if f.name != "extras"}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AudioStageMetrics:
        known = cls.known_field_names()
        kwargs: dict[str, float] = {}
        extras: dict[str, float] = {}
        for k, v in (d or {}).items():
            if not isinstance(v, (int, float, bool)):
                continue
            fv = float(v)
            if k in known:
                kwargs[k] = fv
            else:
                extras[k] = fv
        return cls(extras=extras, **kwargs)

    def to_dict(self) -> dict[str, float]:
        """Serialise only populated fields (zeros are omitted)."""
        out: dict[str, float] = {}
        for f in fields(self):
            if f.name == "extras":
                continue
            v = getattr(self, f.name)
            if v != 0.0:
                out[f.name] = v
        out.update({k: v for k, v in self.extras.items() if v != 0.0})
        return out


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

    def summarize(self, percentiles: tuple[int, ...] = (50, 95)) -> dict[str, float]:
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


def serialize_stage_perf(stage_perf_list: list[StagePerfStats]) -> list[dict[str, Any]]:
    """Serialise a task's stage performance chain to JSON-friendly dicts."""
    result: list[dict[str, Any]] = []
    for perf in stage_perf_list:
        entry: dict[str, Any] = {
            "invocation_id": getattr(perf, "invocation_id", ""),
            "stage_name": perf.stage_name,
            "process_time": perf.process_time,
            "actor_idle_time": perf.actor_idle_time,
            "num_items_processed": perf.num_items_processed,
        }
        # Identity labels (best-effort; empty when unresolved).
        for identity_field in ("actor_id", "node_id", "gpu_id", "physical_address", "pod_ip", "hostname"):
            identity_value = getattr(perf, identity_field, "")
            if identity_value:
                entry[identity_field] = identity_value
        gpu_indices = getattr(perf, "gpu_indices", None) or []
        gpu_uuids = getattr(perf, "gpu_uuids", None) or []
        if gpu_indices:
            entry["gpu_indices"] = [int(idx) for idx in gpu_indices]
        if gpu_uuids:
            entry["gpu_uuids"] = list(gpu_uuids)
        if perf.custom_metrics:
            entry["custom_metrics"] = dict(perf.custom_metrics)
        result.append(entry)
    return result


def _task_audio_seconds(task: Task, duration_key: str) -> float:
    data = getattr(task, "data", {})
    if not isinstance(data, dict):
        return 0.0
    try:
        seconds = float(data.get(duration_key, 0.0))
    except (TypeError, ValueError):
        return 0.0
    return seconds if seconds > 0 else 0.0


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

    metrics = AudioStageMetrics.from_dict(custom_totals)
    custom_sums = metrics.to_dict()

    # Actor-pattern stages lack framework num_items_processed; fall back to
    # total_items_emitted to keep throughput ratios meaningful.
    if total_items == 0.0 and metrics.total_items_emitted > 0:
        total_items = metrics.total_items_emitted
        entry["total_items_processed"] = total_items
    if metrics.total_items_emitted > 0:
        entry["total_items_emitted"] = metrics.total_items_emitted

    add_ratio(entry, "avg_invocation_time_s", total_time, invocation_count)
    add_ratio(entry, "throughput_items_per_s", total_items, total_time)

    # caller context: wallclock + GPU + actor
    ctx = caller_context or AudioStageCallerContext()
    actor_count_p50 = None
    if ctx.actor_count_samples:
        actor_count_p50 = summarize_samples(ctx.actor_count_samples, "actor_count").get("actor_count_p50")

    wallclock_s = (
        ctx.wallclock_s
        if ctx.wallclock_s is not None
        else estimate_wallclock_s(
            total_process_time_s=total_time,
            actor_count=actor_count_p50,
        )
    )
    if wallclock_s is not None and wallclock_s > 0:
        entry["wallclock_s"] = wallclock_s

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

    expected_gpu_count = custom_sums.get("expected_stage_gpu_count", 0.0)
    if expected_gpu_count > 0:
        active_gpu_count = float(entry.get("gpu_count", 0.0) or 0.0)
        missing_gpu_count = max(0.0, expected_gpu_count - active_gpu_count)
        entry["expected_gpu_count"] = expected_gpu_count
        entry["active_sampled_gpu_count"] = active_gpu_count
        entry["missing_or_unattributed_gpu_count"] = missing_gpu_count
        entry["active_sampled_gpu_fraction"] = active_gpu_count / expected_gpu_count
    expected_worker_count = custom_sums.get("expected_stage_worker_count", 0.0)
    if expected_worker_count > 0:
        entry["expected_worker_count"] = expected_worker_count
    expected_worker_gpu_count = custom_sums.get("expected_worker_gpu_count", 0.0)
    if expected_worker_gpu_count > 0:
        entry["expected_worker_gpu_count"] = expected_worker_gpu_count

    if not custom_sums and not samples:
        return entry

    if custom_sums:
        entry["custom_metrics_sum"] = custom_sums

    if samples is not None:
        entry.update(samples.summarize())

    # ----- audio-domain throughput composites -----
    audio_seconds = metrics.audio_duration_s or metrics.audio_duration or metrics.duration
    inference_time = metrics.inference_time_s or metrics.inference_time
    output_tokens = metrics.output_tokens
    output_chars = metrics.output_chars
    waveform_mb = bytes_to_mb(metrics.waveform_bytes)
    bytes_loaded_mb = bytes_to_mb(metrics.bytes_loaded)

    # Both default to the audio duration the stage saw; filter stages may
    # override audio_hours_out via custom_metrics.
    if audio_seconds > 0:
        entry["audio_hours_in"] = seconds_to_hours(audio_seconds)
        entry["audio_hours_out"] = seconds_to_hours(audio_seconds)

    if wallclock_s and actor_count_p50:
        gpu_seconds = wallclock_s * actor_count_p50
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
    add_ratio(entry, "throughput_bytes_loaded_mb_per_process_s", bytes_loaded_mb, total_time)
    if metrics.adapter_inference_calls > 0:
        entry["adapter_inference_call_count"] = metrics.adapter_inference_calls
        entry["adapter_inference_items"] = metrics.adapter_inference_items
    add_ratio(
        entry,
        "avg_adapter_inference_batch_size",
        metrics.adapter_inference_items,
        metrics.adapter_inference_calls,
    )
    add_ratio(
        entry,
        "avg_audio_s_per_adapter_inference_call",
        audio_seconds,
        metrics.adapter_inference_calls,
    )
    add_ratio(
        entry,
        "adapter_inference_calls_per_stage_invocation",
        metrics.adapter_inference_calls,
        invocation_count,
    )

    # ----- pipeline-structure ratios -----
    add_ratio(entry, "output_tasks_per_input_task", metrics.output_tasks, metrics.input_tasks)
    utterances_emitted = metrics.utterances_emitted or metrics.output_utterances
    add_ratio(entry, "utterances_emitted_per_input_shard", utterances_emitted, metrics.input_shards)

    # Generic item-fate aliases: populate from whichever stage-specific
    # counter is non-zero.
    items_skipped = metrics.utterances_skipped or metrics.model_utterances_skipped_preprocess or metrics.skipped_short
    items_filtered = (
        metrics.utterances_filtered
        or metrics.filtered_count
        or metrics.duration_filtered_count
        or metrics.shards_skipped_completed
    )
    items_recovered = metrics.utterances_recovered
    if items_skipped > 0:
        entry["items_skipped"] = items_skipped
    if items_filtered > 0:
        entry["items_filtered"] = items_filtered
    if items_recovered > 0:
        entry["items_recovered"] = items_recovered
    if output_tokens > 0:
        entry["output_tokens"] = output_tokens

    # filter/tagging stages: per-input-utterance ratios
    utterances_input = metrics.utterances_input or metrics.input_tasks
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
            value = getattr(metrics, metric_name, 0.0)
            add_ratio(entry, f"{metric_name}_per_input_utterance", value, utterances_input)

    return entry


@dataclass
class AudioPerformanceSummary:
    """Accumulate and summarise audio task performance metrics.

    Writer-independent: a terminal stage calls ``record_task`` per output task,
    then writes ``build_summary()`` wherever its output contract requires.
    """

    duration_key: str = "duration"
    _stage_totals: dict[str, dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(float)),
        repr=False,
    )
    _stage_custom_totals: dict[str, dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(float)),
        repr=False,
    )
    _stage_samples: dict[str, AudioStageSamples] = field(
        default_factory=lambda: defaultdict(AudioStageSamples),
        repr=False,
    )
    _seen_perf_invocations: set[str] = field(default_factory=set, repr=False)
    # Per-(stage, actor) scheduling breakdown for any record with a resolved
    # actor_id (GPU and CPU stages). GPU actors also carry physical address +
    # NVML util/mem percentiles.
    _stage_actor_samples: dict[str, dict[str, AudioStageSamples]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(AudioStageSamples)),
        repr=False,
    )
    _stage_actor_items: dict[str, dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(float)),
        repr=False,
    )
    _stage_actor_audio_s: dict[str, dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(float)),
        repr=False,
    )
    _stage_actor_location: dict[str, dict[str, dict[str, Any]]] = field(
        default_factory=lambda: defaultdict(dict),
        repr=False,
    )
    # Per-GPU NVML samples nested stage -> actor -> address ("<host>:<idx>"),
    # rolled up under each actor's ``gpus`` block. ``_gpu_unit_meta`` holds
    # per-address metadata (gpu_index, gpu_uuid).
    _stage_actor_gpu_util: dict[str, dict[str, dict[str, list[float]]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        repr=False,
    )
    _stage_actor_gpu_mem: dict[str, dict[str, dict[str, list[float]]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        repr=False,
    )
    _gpu_unit_meta: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)
    # _stage_gpus: per-actor addresses ("<host>:<idx,idx>"); _stage_gpu_units:
    # individual devices ("<host>:<idx>") so gpu_count is true under tensor-parallel.
    _stage_gpus: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set), repr=False)
    _stage_gpu_units: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set), repr=False)
    _stage_actors: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set), repr=False)
    _actor_node: dict[str, str] = field(default_factory=dict, repr=False)
    _shard_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int), repr=False)
    _shard_audio_seconds: dict[str, float] = field(default_factory=lambda: defaultdict(float), repr=False)
    _total_utterances: int = field(default=0, repr=False)
    _total_audio_seconds: float = field(default=0.0, repr=False)
    _wall_start_s: float = field(default_factory=time.perf_counter, repr=False)

    @property
    def total_utterances(self) -> int:
        return self._total_utterances

    @property
    def shard_keys(self) -> list[str]:
        return sorted(self._shard_counts)

    def shard_count(self, shard_key: str) -> int:
        return self._shard_counts.get(shard_key, 0)

    def reset_wall_timer(self) -> None:
        self._wall_start_s = time.perf_counter()

    # -----------------------------------------------------------------------
    # Recording
    # -----------------------------------------------------------------------

    def record_task(self, task: Task, shard_key: str | None = None, *, include_stage_perf: bool = True) -> None:
        """Record one audio task and optionally its attached stage perf chain."""
        audio_seconds = _task_audio_seconds(task, self.duration_key)
        self._total_utterances += 1
        self._total_audio_seconds += audio_seconds

        if shard_key is not None:
            self._shard_counts[shard_key] += 1
            self._shard_audio_seconds[shard_key] += audio_seconds

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
                getattr(perf, "actor_id", ""),
                getattr(perf, "node_id", ""),
                getattr(perf, "gpu_id", ""),
                getattr(perf, "physical_address", ""),
                round(perf.process_time, 9),
                round(perf.actor_idle_time, 9),
                perf.num_items_processed,
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
            invocation_id = getattr(perf, "invocation_id", "") or self._fingerprint_perf(perf)
            if invocation_id in self._seen_perf_invocations:
                continue
            self._seen_perf_invocations.add(invocation_id)

            totals = self._stage_totals[perf.stage_name]
            totals["process_time"] += perf.process_time
            totals["actor_idle_time"] += perf.actor_idle_time
            totals["num_items_processed"] += perf.num_items_processed
            totals["invocation_count"] += 1

            for key, value in (perf.custom_metrics or {}).items():
                if _gpu_sample_base(key) in _GPU_SAMPLE_KEYS:
                    continue
                if isinstance(value, (int, float, bool)):
                    if key in _MAX_CUSTOM_METRIC_KEYS:
                        self._stage_custom_totals[perf.stage_name][key] = max(
                            self._stage_custom_totals[perf.stage_name].get(key, 0.0),
                            float(value),
                        )
                    else:
                        self._stage_custom_totals[perf.stage_name][key] += float(value)

            self._stage_samples[perf.stage_name].add(perf)
            self._record_actor_breakdown(perf)

    def _record_actor_breakdown(self, perf: StagePerfStats) -> None:
        """Accumulate the per-(stage, actor) scheduling breakdown.

        Keyed by ``actor_id`` so every actor-backed stage (GPU or CPU) reports
        per-actor metrics; GPU actors also contribute their physical address and
        device units. No-op for records without a resolved ``actor_id``.
        """
        stage_name = perf.stage_name
        actor_id = (getattr(perf, "actor_id", "") or "").strip()
        if not actor_id:
            return
        node_id = (getattr(perf, "node_id", "") or "").strip()
        self._stage_actors[stage_name].add(actor_id)
        if node_id:
            self._actor_node.setdefault(actor_id, node_id)
        self._stage_actor_samples[stage_name][actor_id].add(perf)
        self._stage_actor_items[stage_name][actor_id] += float(perf.num_items_processed)
        custom = perf.custom_metrics or {}
        audio_s = custom.get("audio_duration_s") or custom.get("audio_duration") or 0.0
        with contextlib.suppress(TypeError, ValueError):
            self._stage_actor_audio_s[stage_name][actor_id] += float(audio_s)
        # GPU topology: physical address + device units (gpu_count true under TP).
        physical_address = (getattr(perf, "physical_address", "") or "").strip()
        host = physical_address.rsplit(":", 1)[0] if physical_address else (node_id or "node")
        if physical_address:
            self._stage_gpus[stage_name].add(physical_address)
            for idx in getattr(perf, "gpu_indices", None) or ():
                self._stage_gpu_units[stage_name].add(f"{host}:{idx}")
        self._record_gpu_samples(stage_name, actor_id, host, perf)
        location = self._actor_location_fields(perf)
        if location:
            self._stage_actor_location[stage_name][actor_id] = location

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
        uuid_to_index = {norm_uuid(u): idx for u, idx in zip(gpu_uuids, gpu_indices, strict=False)}
        uuid_to_raw = {norm_uuid(u): u for u in gpu_uuids}
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

    def _stage_identity_meta(self, stage_name: str) -> dict[str, Any]:
        """Topology labels for a stage: gpu_addresses, gpu_count, actor_count.

        ``gpu_count`` counts distinct physical devices (a TP actor on 2 GPUs
        counts as 2). Keys are omitted for stages without resolved identity.
        """
        meta: dict[str, Any] = {}
        addresses = sorted(self._stage_gpus.get(stage_name, set()))
        if addresses:
            meta["gpu_addresses"] = addresses
            meta["gpu_count"] = float(len(self._stage_gpu_units.get(stage_name, addresses)))
        actors = self._stage_actors.get(stage_name, set())
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
        return {
            stage_name: _build_stage_summary(
                dict(totals),
                dict(self._stage_custom_totals.get(stage_name, {})),
                samples=self._stage_samples.get(stage_name),
                caller_context=ctx_by_stage.get(stage_name),
                stage_identity=self._stage_identity_meta(stage_name),
                actor_breakdown=self._build_per_actor(stage_name),
            )
            for stage_name, totals in self._stage_totals.items()
        }

    def build_summary(
        self,
        *,
        extra_stage_summaries: dict[str, dict[str, Any]] | None = None,
        wall_time_s: float | None = None,
        run_id: str | None = None,
        executor: str | None = None,
        stage_caller_context: dict[str, AudioStageCallerContext] | None = None,
    ) -> dict[str, Any]:
        """Build the full audio pipeline performance summary.

        Top-level fields match the proposed pipeline-perf shape (run_id,
        executor, input_hours, output_hours, rows_in, rows_out, stages).
        Backward-compat keys (total_utterances, total_audio_seconds, shards,
        etc.) are preserved verbatim for the protocol-doc baseline tables.
        """
        resolved_wall_time_s = (
            max(time.perf_counter() - self._wall_start_s, 0.0) if wall_time_s is None else max(wall_time_s, 0.0)
        )
        stages_summary = self.build_stage_summaries(stage_caller_context)
        if extra_stage_summaries:
            stages_summary.update(extra_stage_summaries)

        # Derive top-level input_hours from the first stage that has audio volume.
        # Derive rows_in by priority so discovery's synthetic input_tasks=1 does
        # not mask reader-level row counts.
        input_hours = 0.0
        rows_in_by_key = {
            "manifest_entries": 0.0,
            "output_utterances": 0.0,
            "input_shards": 0.0,
            "input_tasks": 0.0,
        }
        for stage_dict in stages_summary.values():
            if input_hours == 0.0 and "audio_hours_in" in stage_dict:
                input_hours = stage_dict["audio_hours_in"]
            cm = stage_dict.get("custom_metrics_sum", {})
            for key, value in rows_in_by_key.items():
                if value == 0.0:
                    rows_in_by_key[key] = float(cm.get(key, 0.0) or 0.0)

        rows_in = next((value for value in rows_in_by_key.values() if value > 0.0), 0.0)

        output_hours = seconds_to_hours(self._total_audio_seconds)
        rows_out = float(self._total_utterances)

        summary: dict[str, Any] = {
            # proposed-structure top-level
            "run_id": run_id or "",
            "executor": executor or "",
            "input_hours": input_hours,
            "output_hours": output_hours,
            "rows_in": rows_in,
            "rows_out": rows_out,
            # backward-compat top-level (protocol-doc baselines)
            "total_utterances": self._total_utterances,
            "total_audio_seconds": self._total_audio_seconds,
            "total_audio_hours": output_hours,
            "writer_wall_time_s": resolved_wall_time_s,
            "pipeline_audio_s_per_wall_s": (
                self._total_audio_seconds / resolved_wall_time_s if resolved_wall_time_s > 0 else 0.0
            ),
            "pipeline_utterances_per_wall_s": (
                self._total_utterances / resolved_wall_time_s if resolved_wall_time_s > 0 else 0.0
            ),
            "perf_invocations_counted": len(self._seen_perf_invocations),
            "shards": {
                shard: {
                    "utterances": count,
                    "audio_seconds": self._shard_audio_seconds.get(shard, 0.0),
                    "audio_hours": self._shard_audio_seconds.get(shard, 0.0) / 3600.0,
                }
                for shard, count in sorted(self._shard_counts.items())
            },
            "stages": stages_summary,
        }

        # Cluster-level rollup (scheduling only). Hardware rollups are deferred
        # to the NVML/DCGM proposal; only identity-derivable fields emitted here.
        pipeline_throughput: dict[str, Any] = {}
        if resolved_wall_time_s > 0 and self._total_audio_seconds > 0:
            pipeline_throughput["audio_hours_per_wallclock_hour"] = seconds_to_hours(
                self._total_audio_seconds
            ) / seconds_to_hours(resolved_wall_time_s)
        all_addresses = sorted({addr for addrs in self._stage_gpus.values() for addr in addrs})
        if all_addresses:
            all_units = {unit for units in self._stage_gpu_units.values() for unit in units}
            pipeline_throughput["gpu_addresses"] = all_addresses
            pipeline_throughput["gpu_count"] = float(len(all_units or all_addresses))
        if pipeline_throughput:
            summary["pipeline_throughput"] = pipeline_throughput

        return summary
