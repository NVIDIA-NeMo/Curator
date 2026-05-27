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

"""Reusable audio pipeline performance summary helpers.

Audio stages emit processor-specific counters and timings via
``ProcessingStage._log_metrics()``. Backends attach those to
``Task._stage_perf`` as ``StagePerfStats``. Terminal audio stages (the
writer) consume those chains and call into ``AudioPerformanceSummary``
to build the published ``perf_summary.json``.

Architecture:

  * ``AudioStageMetrics`` -- typed superset dataclass of EVERY scalar
    custom metric any audio stage may emit. Stages populate only what
    is relevant; unknown keys are preserved in ``extras``.
  * ``AudioStageSamples`` -- per-invocation sample lists used to derive
    p50/p95 percentiles (batch sizes, audio durations, queue waits).
  * ``AudioStageCallerContext`` -- caller-provided fields that cannot
    be derived from ``StagePerfStats`` alone (GPU-hours, GPU
    utilisation percentiles, actor-count percentiles, setup-time).
  * ``AudioPerformanceSummary`` -- accumulator. Dedupes repeat sightings
    of the same ``StagePerfStats`` (via framework
    ``StagePerfStats.invocation_id`` when wired; otherwise a synthetic
    value-tuple fingerprint), collects samples, and renders the final
    summary dict matching the proposed pipeline-perf shape.

All post-processing (percentiles, ratios, unit conversions, wallclock
estimation) lives in ``performance_utils.py`` -- this file only does
collection + orchestration.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field, fields
from typing import Any

from nemo_curator.stages.audio.metrics.performance_utils import (
    add_ratio,
    audio_hours_per_gpu_hour,
    bytes_to_mb,
    estimate_wallclock_s,
    seconds_to_hours,
    summarize_samples,
)
from nemo_curator.tasks import Task
from nemo_curator.utils.performance_utils import StagePerfStats


# ===========================================================================
# Superset dataclass: every custom metric any audio stage may emit
# ===========================================================================

@dataclass
class AudioStageMetrics:
    """Superset of every scalar custom metric the audio pipeline emits.

    Stages populate only the fields relevant to them via
    ``_log_metrics({"field_name": value, ...})``. The accumulator sums
    them and rebuilds an ``AudioStageMetrics`` per stage at summary
    time. Default 0.0 means "stage did not emit"; ``to_dict()`` strips
    those automatically so the published JSON only carries the
    populated keys.

    Adding a new audio-pipeline metric is a single-field edit here +
    the corresponding ``_log_metrics`` call on the producer.
    """

    # ----- universal counters (any audio stage may emit) -----
    input_tasks: float = 0.0
    output_tasks: float = 0.0
    # ``total_items_emitted`` is the actor-pattern fix for stages that
    # the framework's ``num_items_processed`` cannot count
    # (e.g. NemoTarShardDiscoveryStage: synthesises work from config,
    # so the framework sees 0 input items). Stages that synthesise
    # downstream tasks should set this themselves.
    total_items_emitted: float = 0.0

    # ----- audio volume scalars -----
    audio_duration_s: float = 0.0
    # Legacy aliases preserved for backward-compat across older audio
    # stages (whisperx_vad, pyannote, common, split, resample). New
    # stages should emit ``audio_duration_s``.
    audio_duration: float = 0.0
    duration: float = 0.0
    input_duration: float = 0.0
    filtered_dur: float = 0.0
    waveform_bytes: float = 0.0
    # New audio-side counter for the "truthful bytes loaded" view that
    # framework's ``input_data_size_mb`` cannot provide for stages
    # which load data themselves (e.g. tar reader). Producer-opt-in.
    bytes_loaded: float = 0.0

    # ----- text/transcript output -----
    output_chars: float = 0.0
    output_tokens: float = 0.0
    turn1_output_tokens: float = 0.0
    turn2_output_tokens: float = 0.0

    # ----- inference timing -----
    inference_time_s: float = 0.0
    # Legacy alias for older inference stages; new code should emit
    # ``inference_time_s``.
    inference_time: float = 0.0

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
    process_time: float = 0.0  # legacy custom timer some stages emit alongside framework

    # ----- speaker diarization -----
    segments_detected: float = 0.0
    overlap_segments_detected: float = 0.0
    speakers_detected: float = 0.0

    # ----- VAD -----
    vad_segments_detected: float = 0.0
    skipped_short: float = 0.0

    # ----- sharded manifest writer (ShardedManifestWriterStage) -----
    writer_process_calls: float = 0.0
    manifest_write_time_s: float = 0.0
    done_marker_write_time_s: float = 0.0
    perf_write_time_s: float = 0.0

    # ----- forward-compat: any emitted scalar this dataclass doesn't know -----
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


# ===========================================================================
# Per-invocation sample lists for p50/p95 percentile computation
# ===========================================================================

@dataclass
class AudioStageSamples:
    """Per-invocation sample lists used for percentile derivation.

    Populated once per dedup'd invocation. Empty by default; only the
    accumulator writes to these.
    """

    invocation_process_times_s: list[float] = field(default_factory=list)
    actor_idle_times_s: list[float] = field(default_factory=list)
    items_processed_per_invocation: list[float] = field(default_factory=list)
    batch_sizes: list[float] = field(default_factory=list)
    audio_duration_s_per_invocation: list[float] = field(default_factory=list)

    def add(self, perf: StagePerfStats) -> None:
        """Record one dedup'd invocation's per-call samples."""
        self.invocation_process_times_s.append(float(perf.process_time))
        self.actor_idle_times_s.append(float(perf.actor_idle_time))
        self.items_processed_per_invocation.append(float(perf.num_items_processed))

        custom = perf.custom_metrics or {}
        # Batch size proxy: stages that handle a list of tasks-per-invocation
        # report this as ``utterances_input`` or ``input_count``. For
        # single-task-per-invocation stages this collapses to 1.
        batch_size = (
            custom.get("utterances_input")
            or custom.get("input_count")
            or custom.get("input_tasks")
            or perf.num_items_processed
        )
        try:
            self.batch_sizes.append(float(batch_size))
        except (TypeError, ValueError):
            pass

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


# ===========================================================================
# Caller-provided context (GPU / actor data the audio summary can't derive)
# ===========================================================================

@dataclass
class AudioStageCallerContext:
    """Caller-provided fields the audio accumulator cannot derive itself.

    Optional. A writer that has access to NVML / DCGM / Xenna autoscaler
    snapshots can pass these in to populate the GPU- and actor-related
    fields of the proposed pipeline-perf shape. Leaving them at default
    causes those fields to be omitted from the published summary.
    """

    actor_count_samples: list[float] = field(default_factory=list)
    gpu_util_pct_samples: list[float] = field(default_factory=list)
    gpu_hours: float = 0.0
    setup_time_s_total: float = 0.0
    wallclock_s: float | None = None  # overrides estimate if provided


# ===========================================================================
# Per-task serialiser (preserved for backward-compat with existing callers)
# ===========================================================================

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


# ===========================================================================
# Per-stage summary builder
# ===========================================================================

def _build_stage_summary(
    stage_totals: dict[str, float],
    custom_totals: dict[str, float],
    samples: AudioStageSamples | None = None,
    caller_context: AudioStageCallerContext | None = None,
) -> dict[str, Any]:
    """Render one stage's summary in the proposed pipeline-perf shape.

    Combines:
      * Framework scalar totals (process_time, actor_idle_time,
        num_items_processed, invocation_count).
      * Custom-metric superset (``AudioStageMetrics``) summed across
        dedup'd invocations.
      * Per-invocation sample percentiles
        (``AudioStageSamples.summarize()``).
      * Caller-provided GPU / actor / setup-time context
        (``AudioStageCallerContext``).
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

    # A4 fix: actor-pattern stages cannot get the framework's
    # num_items_processed populated, so fall back to total_items_emitted
    # to keep throughput ratios meaningful.
    if total_items == 0.0 and metrics.total_items_emitted > 0:
        total_items = metrics.total_items_emitted
        entry["total_items_processed"] = total_items
    if metrics.total_items_emitted > 0:
        entry["total_items_emitted"] = metrics.total_items_emitted

    add_ratio(entry, "avg_invocation_time_s", total_time, invocation_count)
    add_ratio(entry, "throughput_items_per_s", total_items, total_time)

    # ----- caller context: wallclock + GPU + actor -----
    ctx = caller_context or AudioStageCallerContext()
    invocation_times = samples.invocation_process_times_s if samples else []
    actor_count_p50 = None
    if ctx.actor_count_samples:
        actor_count_p50 = summarize_samples(ctx.actor_count_samples, "actor_count").get("actor_count_p50")

    wallclock_s = ctx.wallclock_s if ctx.wallclock_s is not None else estimate_wallclock_s(
        total_process_time_s=total_time,
        actor_count=actor_count_p50,
        invocation_times=invocation_times,
    )
    if wallclock_s is not None and wallclock_s > 0:
        entry["wallclock_s"] = wallclock_s

    if ctx.gpu_hours > 0:
        entry["gpu_hours"] = ctx.gpu_hours
    if ctx.setup_time_s_total > 0:
        entry["setup_time_s_total"] = ctx.setup_time_s_total
    entry.update(summarize_samples(ctx.actor_count_samples, "actor_count"))
    entry.update(summarize_samples(ctx.gpu_util_pct_samples, "gpu_util_pct"))

    if not custom_sums and not samples:
        return entry

    if custom_sums:
        entry["custom_metrics_sum"] = custom_sums

    # ----- per-invocation percentile derivation -----
    if samples is not None:
        entry.update(samples.summarize())

    # ----- audio-domain throughput composites -----
    audio_seconds = metrics.audio_duration_s or metrics.audio_duration or metrics.duration
    inference_time = metrics.inference_time_s or metrics.inference_time
    output_tokens = metrics.output_tokens
    output_chars = metrics.output_chars
    waveform_mb = bytes_to_mb(metrics.waveform_bytes)
    bytes_loaded_mb = bytes_to_mb(metrics.bytes_loaded)

    # audio_hours_in/out: per-stage view of the proposed structure.
    # By default both are the audio duration the stage saw; filter
    # stages may override audio_hours_out via custom_metrics if they
    # want to publish a different out-view.
    if audio_seconds > 0:
        entry["audio_hours_in"] = seconds_to_hours(audio_seconds)
        entry["audio_hours_out"] = seconds_to_hours(audio_seconds)

    if wallclock_s and actor_count_p50:
        gpu_seconds = wallclock_s * actor_count_p50
        ah_per_gpu_h = audio_hours_per_gpu_hour(audio_seconds, gpu_seconds)
        if ah_per_gpu_h is not None:
            entry["audio_hours_per_gpu_hour"] = ah_per_gpu_h

    add_ratio(entry, "throughput_audio_s_per_process_s", audio_seconds, total_time)
    add_ratio(entry, "throughput_audio_s_per_inference_s", audio_seconds, inference_time)
    add_ratio(entry, "avg_audio_s_per_item", audio_seconds, total_items)
    add_ratio(entry, "throughput_output_tokens_per_process_s", output_tokens, total_time)
    add_ratio(entry, "throughput_output_tokens_per_inference_s", output_tokens, inference_time)
    add_ratio(entry, "throughput_output_chars_per_process_s", output_chars, total_time)
    add_ratio(entry, "throughput_output_chars_per_inference_s", output_chars, inference_time)
    add_ratio(entry, "throughput_waveform_mb_per_process_s", waveform_mb, total_time)
    add_ratio(entry, "throughput_bytes_loaded_mb_per_process_s", bytes_loaded_mb, total_time)

    # ----- pipeline-structure ratios -----
    add_ratio(entry, "output_tasks_per_input_task", metrics.output_tasks, metrics.input_tasks)
    utterances_emitted = metrics.utterances_emitted or metrics.output_utterances
    add_ratio(entry, "utterances_emitted_per_input_shard", utterances_emitted, metrics.input_shards)

    # ----- proposed-structure item-fate aliases -----
    # The proposed run-report uses generic ``items_skipped`` /
    # ``items_filtered`` / ``items_recovered`` regardless of stage type;
    # populate from whichever stage-specific counter is non-zero.
    items_skipped = (
        metrics.utterances_skipped
        or metrics.model_utterances_skipped_preprocess
        or metrics.skipped_short
    )
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

    # ----- filter/tagging stages: per-input-utterance ratios -----
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


# ===========================================================================
# Accumulator
# ===========================================================================

@dataclass
class AudioPerformanceSummary:
    """Accumulate and summarise audio task performance metrics.

    Independent of any writer implementation. A terminal audio stage
    calls ``record_task`` for each output task it sees, then writes
    ``build_summary()`` to wherever its output contract requires.
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

        Used as a fallback dedup key when the backend has not populated
        ``StagePerfStats.invocation_id``. The audio pipeline observes
        the same ``StagePerfStats`` once per emitted downstream task,
        so an actor invocation that emits N output tasks would
        otherwise be counted N times. Two genuinely distinct
        invocations producing byte-equal timings AND byte-equal
        custom-metric tuples is astronomically unlikely, so collisions
        are not a practical concern.
        """
        custom = sorted((perf.custom_metrics or {}).items())
        return repr((
            perf.stage_name,
            round(perf.process_time, 9),
            round(perf.actor_idle_time, 9),
            perf.num_items_processed,
            tuple((k, round(float(v), 9)) for k, v in custom),
        ))

    def record_stage_perf(self, stage_perf_list: list[StagePerfStats]) -> None:
        """Accumulate ``StagePerfStats``, deduplicating repeat sightings.

        Dedup key:
          1. ``StagePerfStats.invocation_id`` when the backend wires it
             (preferred -- fixes the dedup at the framework layer).
          2. Synthetic value-tuple fingerprint (audio-pipeline fallback;
             works without framework changes).

        After dedup, the per-invocation perf record contributes to:
          * stage scalar totals (process_time, actor_idle_time,
            num_items_processed, invocation_count).
          * stage custom-metric sums (audio_duration_s, output_tokens,
            etc. -- the superset).
          * stage per-invocation samples (used downstream for p50/p95).
        """
        for perf in stage_perf_list:
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
                if isinstance(value, (int, float, bool)):
                    self._stage_custom_totals[perf.stage_name][key] += float(value)

            self._stage_samples[perf.stage_name].add(perf)

    # -----------------------------------------------------------------------
    # Building the published summary
    # -----------------------------------------------------------------------

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
            )
            for stage_name, totals in self._stage_totals.items()
        }

    def build_summary(  # noqa: PLR0913
        self,
        *,
        extra_stage_summaries: dict[str, dict[str, Any]] | None = None,
        wall_time_s: float | None = None,
        run_id: str | None = None,
        executor: str | None = None,
        stage_caller_context: dict[str, AudioStageCallerContext] | None = None,
    ) -> dict[str, Any]:
        """Build the full audio pipeline performance summary.

        Top-level fields match the proposed pipeline-perf shape:
          ``run_id``, ``executor``, ``input_hours``, ``output_hours``,
          ``rows_in``, ``rows_out``, ``stages``.

        Backward-compat keys (``total_utterances``,
        ``total_audio_seconds``, ``total_audio_hours``,
        ``writer_wall_time_s``, ``pipeline_audio_s_per_wall_s``,
        ``pipeline_utterances_per_wall_s``, ``perf_invocations_counted``,
        ``shards``) are preserved verbatim so the protocol-doc baseline
        tables continue to read.
        """
        resolved_wall_time_s = (
            max(time.perf_counter() - self._wall_start_s, 0.0) if wall_time_s is None else max(wall_time_s, 0.0)
        )
        stages_summary = self.build_stage_summaries(stage_caller_context)
        if extra_stage_summaries:
            stages_summary.update(extra_stage_summaries)

        # Derive top-level input_hours / rows_in from the first stage that
        # has them populated (typically the reader / discovery).
        input_hours = 0.0
        rows_in = 0.0
        for stage_dict in stages_summary.values():
            if input_hours == 0.0 and "audio_hours_in" in stage_dict:
                input_hours = stage_dict["audio_hours_in"]
            cm = stage_dict.get("custom_metrics_sum", {})
            if rows_in == 0.0:
                rows_in = float(cm.get("input_tasks", 0.0) or cm.get("input_shards", 0.0))
            if input_hours and rows_in:
                break

        output_hours = seconds_to_hours(self._total_audio_seconds)
        rows_out = float(self._total_utterances)

        summary: dict[str, Any] = {
            # ----- proposed-structure top-level -----
            "run_id": run_id or "",
            "executor": executor or "",
            "input_hours": input_hours,
            "output_hours": output_hours,
            "rows_in": rows_in,
            "rows_out": rows_out,
            # ----- backward-compat top-level (protocol-doc baselines) -----
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
        return summary
