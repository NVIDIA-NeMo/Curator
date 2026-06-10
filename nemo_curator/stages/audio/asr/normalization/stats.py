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

"""Streaming transcript-quality statistics for normalized ASR manifests."""

from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.audio.asr.normalization.transcript import _RESOURCE_ROOT, _load_alphabet, resolve_lang
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata


@dataclass
class _StatsBucket:
    total_transcripts: int = 0
    valid_transcripts: int = 0
    invalid_transcripts: int = 0
    dropped_invalid: int = 0
    total_duration_seconds: float = 0.0
    valid_duration_seconds: float = 0.0
    invalid_duration_seconds: float = 0.0
    total_chars: int = 0
    known_chars: Counter[str] = field(default_factory=Counter)
    unknown_char_count: int = 0
    unknown_chars: Counter[str] = field(default_factory=Counter)
    split_counts: dict[str, dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: {"total": 0, "valid": 0, "invalid": 0})
    )
    split_duration_seconds: dict[str, dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: {"total": 0.0, "valid": 0.0, "invalid": 0.0})
    )


@dataclass(frozen=True)
class _BucketUpdate:
    text: str
    duration: float
    known_chars: Counter[str]
    unknown_chars: dict[str, int]
    transcript_error: bool
    split: str
    dropped: int


@dataclass
class TranscriptStatsStage(ProcessingStage[AudioTask, AudioTask]):
    """Collect transcript-validity stats while streaming ``AudioTask`` objects.

    The stage writes global aggregate statistics, full summaries for each
    language/source pair under ``by_language``, and full language-level totals
    under ``by_language_overall``.
    """

    name: str = "transcript_stats"
    text_key: str = "text"
    lang_key: str = "lang"
    source_key: str = "source"
    duration_key: str = "duration"
    split_key: str = "split_type"
    unknown_chars_key: str = "unknown_chars"
    transcript_error_key: str = "transcript_error"
    drop_invalid: bool = False
    log_top_n_unknown_chars: int = 50
    output_summary_path: str | None = None
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def __post_init__(self) -> None:
        self._summary_handle = None
        self._total_transcripts = 0
        self._valid_transcripts = 0
        self._invalid_transcripts = 0
        self._dropped_invalid = 0
        self._total_duration_seconds = 0.0
        self._valid_duration_seconds = 0.0
        self._invalid_duration_seconds = 0.0
        self._total_chars = 0
        self._known_chars: Counter[str] = Counter()
        self._languages: set[str] = set()
        self._unknown_char_count = 0
        self._unknown_chars: Counter[str] = Counter()
        self._split_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "valid": 0, "invalid": 0})
        self._split_duration_seconds: dict[str, dict[str, float]] = defaultdict(
            lambda: {"total": 0.0, "valid": 0.0, "invalid": 0.0}
        )
        self._language_stats: dict[str, _StatsBucket] = defaultdict(_StatsBucket)
        self._language_source_stats: dict[str, dict[str, _StatsBucket]] = defaultdict(
            lambda: defaultdict(_StatsBucket)
        )

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.text_key, self.lang_key, self.unknown_chars_key, self.transcript_error_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def num_workers(self) -> int | None:
        return 1

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_FANOUT_STAGE: False}

    def xenna_stage_spec(self) -> dict[str, Any]:
        return {"num_workers": 1}

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        if self.output_summary_path:
            parent = os.path.dirname(self.output_summary_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            self._summary_handle = open(self.output_summary_path, "w", encoding="utf-8")  # noqa: SIM115

    def process(self, task: AudioTask) -> AudioTask | None:
        start = time.perf_counter()
        text = str(task.data.get(self.text_key, ""))
        lang = resolve_lang(str(task.data[self.lang_key]))
        self._languages.add(lang)
        source = str(task.data.get(self.source_key, "unknown") or "unknown")
        duration = float(task.data.get(self.duration_key, 0.0) or 0.0)
        unknown_chars = _coerce_unknown_chars(task.data.get(self.unknown_chars_key, {}))
        unknown_char_set = set(unknown_chars)
        transcript_error = bool(task.data.get(self.transcript_error_key, bool(unknown_chars)))
        split = str(task.data.get(self.split_key, "unknown"))

        self._total_transcripts += 1
        self._total_duration_seconds += duration
        self._total_chars += len(text)
        known_chars = Counter(char for char in text if char not in unknown_char_set)
        self._known_chars.update(known_chars)
        self._unknown_chars.update(unknown_chars)
        unknown_count = sum(unknown_chars.values())
        self._unknown_char_count += unknown_count

        split_counts = self._split_counts[split]
        split_durations = self._split_duration_seconds[split]
        split_counts["total"] += 1
        split_durations["total"] += duration
        if transcript_error:
            self._invalid_transcripts += 1
            self._invalid_duration_seconds += duration
            split_counts["invalid"] += 1
            split_durations["invalid"] += duration
        else:
            self._valid_transcripts += 1
            self._valid_duration_seconds += duration
            split_counts["valid"] += 1
            split_durations["valid"] += duration

        dropped = int(transcript_error and self.drop_invalid)
        self._dropped_invalid += dropped
        update = _BucketUpdate(
            text=text,
            duration=duration,
            known_chars=known_chars,
            unknown_chars=unknown_chars,
            transcript_error=transcript_error,
            split=split,
            dropped=dropped,
        )
        _update_bucket(self._language_stats[lang], update)
        _update_bucket(self._language_source_stats[lang][source], update)
        self._log_metrics(self._metrics_snapshot(process_time=time.perf_counter() - start))
        self._write_summary()
        if dropped:
            return None
        return task

    def summary(self) -> dict[str, Any]:
        valid_rate = self._valid_transcripts / self._total_transcripts if self._total_transcripts else 0.0
        invalid_rate = self._invalid_transcripts / self._total_transcripts if self._total_transcripts else 0.0
        unique_known_chars = len(self._known_chars)
        unique_unknown_chars = len(self._unknown_chars)
        unique_known_char_rate = unique_known_chars / self._total_chars if self._total_chars else 0.0
        unique_unknown_char_rate = unique_unknown_chars / self._total_chars if self._total_chars else 0.0
        alpha_minus_known_chars = []
        if len(self._languages) == 1:
            language = next(iter(self._languages))
            alpha_minus_known_chars = sorted(
                _load_alphabet(_RESOURCE_ROOT / language / "alphabet.txt") - set(self._known_chars)
            )
        split_hours = {
            split: {key: seconds / 3600 for key, seconds in durations.items()}
            for split, durations in self._split_duration_seconds.items()
        }
        return {
            "total_transcripts": self._total_transcripts,
            "valid_transcripts": self._valid_transcripts,
            "invalid_transcripts": self._invalid_transcripts,
            "dropped_invalid": self._dropped_invalid,
            "emitted_transcripts": self._total_transcripts - self._dropped_invalid,
            "valid_transcript_rate": valid_rate,
            "invalid_transcript_rate": invalid_rate,
            "total_duration_hours": self._total_duration_seconds / 3600,
            "valid_duration_hours": self._valid_duration_seconds / 3600,
            "invalid_duration_hours": self._invalid_duration_seconds / 3600,
            "total_chars": self._total_chars,
            "unique_known_chars": unique_known_chars,
            "unique_known_char_rate": unique_known_char_rate,
            "unique_unknown_chars": unique_unknown_chars,
            "unique_unknown_char_rate": unique_unknown_char_rate,
            "alpha_minus_known_chars": alpha_minus_known_chars,
            "split_counts": dict(self._split_counts),
            "split_hours": split_hours,
            "by_language": {
                lang: {
                    source: _bucket_summary(bucket, lang)
                    for source, bucket in sorted(source_buckets.items(), key=lambda item: item[0])
                }
                for lang, source_buckets in sorted(self._language_source_stats.items(), key=lambda item: item[0])
            },
            "by_language_overall": {
                lang: _bucket_summary(bucket, lang)
                for lang, bucket in sorted(self._language_stats.items(), key=lambda item: item[0])
            },
        }

    def teardown(self) -> None:
        logger.info(self.format_summary())
        if self._summary_handle is not None:
            self._summary_handle.close()
            self._summary_handle = None

    def _write_summary(self) -> None:
        if not self.output_summary_path:
            return
        if self._summary_handle is None:
            self.setup_on_node()
        self._summary_handle.seek(0)
        self._summary_handle.write(json.dumps(self.summary(), ensure_ascii=False, indent=2) + "\n")
        self._summary_handle.truncate()
        self._summary_handle.flush()

    def format_summary(self) -> str:
        summary = self.summary()
        lines = [
            f"[{self.name}] Transcript normalization summary",
            "  per_language_source:",
        ]
        for lang, source_summaries in summary["by_language"].items():
            for source, source_summary in source_summaries.items():
                lines.extend(_format_summary_block(f"lang={lang} source={source}", source_summary, indent="    "))
        lines.append("  per_language_overall:")
        for lang, language_summary in summary["by_language_overall"].items():
            lines.extend(_format_summary_block(f"lang={lang} overall", language_summary, indent="    "))
        lines.append("  global:")
        lines.extend(_format_summary_block("all languages/sources", summary, indent="    "))
        return "\n".join(lines)

    def _metrics_snapshot(self, process_time: float) -> dict[str, float | int]:
        summary = self.summary()
        return {
            "input_tasks": summary["total_transcripts"],
            "emitted_tasks": summary["emitted_transcripts"],
            "dropped_invalid": summary["dropped_invalid"],
            "valid_transcripts": summary["valid_transcripts"],
            "invalid_transcripts": summary["invalid_transcripts"],
            "total_duration_hours": summary["total_duration_hours"],
            "valid_duration_hours": summary["valid_duration_hours"],
            "invalid_duration_hours": summary["invalid_duration_hours"],
            "total_chars": summary["total_chars"],
            "unique_known_chars": summary["unique_known_chars"],
            "unique_known_char_rate": summary["unique_known_char_rate"],
            "unique_unknown_chars": summary["unique_unknown_chars"],
            "unique_unknown_char_rate": summary["unique_unknown_char_rate"],
            "process_time": process_time,
        }


def _coerce_unknown_chars(value: Any) -> dict[str, int]:  # noqa: ANN401
    if isinstance(value, dict):
        return {str(char): int(count) for char, count in value.items()}
    return {}


def _format_summary_block(label: str, summary: dict[str, Any], *, indent: str) -> list[str]:
    split_hours = _round_nested_floats(summary["split_hours"])
    return [
        f"{indent}{label}",
        (
            f"{indent}  transcripts: total={summary['total_transcripts']} "
            f"valid={summary['valid_transcripts']} ({summary['valid_transcript_rate']:.2%}) "
            f"invalid={summary['invalid_transcripts']} ({summary['invalid_transcript_rate']:.2%})"
        ),
        (
            f"{indent}  hours: total={summary['total_duration_hours']:.2f} "
            f"valid_after_filter={summary['valid_duration_hours']:.2f} "
            f"invalid_removed={summary['invalid_duration_hours']:.2f}"
        ),
        f"{indent}  split_hours: {split_hours}",
        (
            f"{indent}  chars: total={summary['total_chars']} "
            f"unique_known={summary['unique_known_chars']} "
            f"unique_known_rate={summary['unique_known_char_rate']:.2%} "
            f"unique_unknown={summary['unique_unknown_chars']} "
            f"unique_unknown_rate={summary['unique_unknown_char_rate']:.2%}"
        ),
        f"{indent}  alpha_minus_known_chars: {summary['alpha_minus_known_chars']}",
        f"{indent}  split_counts: {summary['split_counts']}",
    ]


def _update_bucket(bucket: _StatsBucket, update: _BucketUpdate) -> None:
    bucket.total_transcripts += 1
    bucket.dropped_invalid += update.dropped
    bucket.total_duration_seconds += update.duration
    bucket.total_chars += len(update.text)
    bucket.known_chars.update(update.known_chars)
    bucket.unknown_chars.update(update.unknown_chars)
    bucket.unknown_char_count += sum(update.unknown_chars.values())

    split_counts = bucket.split_counts[update.split]
    split_durations = bucket.split_duration_seconds[update.split]
    split_counts["total"] += 1
    split_durations["total"] += update.duration
    if update.transcript_error:
        bucket.invalid_transcripts += 1
        bucket.invalid_duration_seconds += update.duration
        split_counts["invalid"] += 1
        split_durations["invalid"] += update.duration
    else:
        bucket.valid_transcripts += 1
        bucket.valid_duration_seconds += update.duration
        split_counts["valid"] += 1
        split_durations["valid"] += update.duration


def _bucket_summary(bucket: _StatsBucket, lang: str) -> dict[str, Any]:
    valid_rate = bucket.valid_transcripts / bucket.total_transcripts if bucket.total_transcripts else 0.0
    invalid_rate = bucket.invalid_transcripts / bucket.total_transcripts if bucket.total_transcripts else 0.0
    unique_known_chars = len(bucket.known_chars)
    unique_unknown_chars = len(bucket.unknown_chars)
    unique_known_char_rate = unique_known_chars / bucket.total_chars if bucket.total_chars else 0.0
    unique_unknown_char_rate = unique_unknown_chars / bucket.total_chars if bucket.total_chars else 0.0
    split_hours = {
        split: {key: seconds / 3600 for key, seconds in durations.items()}
        for split, durations in bucket.split_duration_seconds.items()
    }
    return {
        "total_transcripts": bucket.total_transcripts,
        "valid_transcripts": bucket.valid_transcripts,
        "invalid_transcripts": bucket.invalid_transcripts,
        "dropped_invalid": bucket.dropped_invalid,
        "emitted_transcripts": bucket.total_transcripts - bucket.dropped_invalid,
        "valid_transcript_rate": valid_rate,
        "invalid_transcript_rate": invalid_rate,
        "total_duration_hours": bucket.total_duration_seconds / 3600,
        "valid_duration_hours": bucket.valid_duration_seconds / 3600,
        "invalid_duration_hours": bucket.invalid_duration_seconds / 3600,
        "total_chars": bucket.total_chars,
        "unique_known_chars": unique_known_chars,
        "unique_known_char_rate": unique_known_char_rate,
        "unique_unknown_chars": unique_unknown_chars,
        "unique_unknown_char_rate": unique_unknown_char_rate,
        "alpha_minus_known_chars": sorted(
            _load_alphabet(_RESOURCE_ROOT / lang / "alphabet.txt") - set(bucket.known_chars)
        ),
        "split_counts": {split: dict(counts) for split, counts in bucket.split_counts.items()},
        "split_hours": split_hours,
    }


def _round_nested_floats(value: Any, ndigits: int = 2) -> Any:  # noqa: ANN401
    if isinstance(value, dict):
        return {key: _round_nested_floats(nested_value, ndigits) for key, nested_value in value.items()}
    if isinstance(value, float):
        return round(value, ndigits)
    return value
