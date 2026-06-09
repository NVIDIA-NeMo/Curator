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

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.audio.asr.normalization.transcript import _RESOURCE_ROOT, _load_alphabet, resolve_lang
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


@dataclass
class TranscriptStatsStage(ProcessingStage[AudioTask, AudioTask]):
    """Collect transcript-validity stats while streaming ``AudioTask`` objects."""

    name: str = "transcript_stats"
    text_key: str = "text"
    lang_key: str = "lang"
    duration_key: str = "duration"
    split_key: str = "split_type"
    unknown_chars_key: str = "unknown_chars"
    transcript_error_key: str = "transcript_error"
    drop_invalid: bool = False
    log_top_n_unknown_chars: int = 50
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def __post_init__(self) -> None:
        self._total_transcripts = 0
        self._valid_transcripts = 0
        self._invalid_transcripts = 0
        self._dropped_invalid = 0
        self._total_duration_seconds = 0.0
        self._valid_duration_seconds = 0.0
        self._invalid_duration_seconds = 0.0
        self._total_chars = 0
        self._known_chars: Counter[str] = Counter()
        self._language: str | None = None
        self._unknown_char_count = 0
        self._unknown_chars: Counter[str] = Counter()
        self._split_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "valid": 0, "invalid": 0})
        self._split_duration_seconds: dict[str, dict[str, float]] = defaultdict(
            lambda: {"total": 0.0, "valid": 0.0, "invalid": 0.0}
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

    def process(self, task: AudioTask) -> AudioTask | None:
        start = time.perf_counter()
        text = str(task.data.get(self.text_key, ""))
        lang = resolve_lang(str(task.data[self.lang_key]))
        if self._language is None:
            self._language = lang
        elif lang != self._language:
            msg = f"{self.name} expects one language per dataset, got {self._language!r} and {lang!r}"
            raise ValueError(msg)
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
        self._log_metrics(self._metrics_snapshot(process_time=time.perf_counter() - start))
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
        if self._language is not None:
            alpha_minus_known_chars = sorted(
                _load_alphabet(_RESOURCE_ROOT / self._language / "alphabet.txt") - set(self._known_chars)
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
        }

    def teardown(self) -> None:
        logger.info(self.format_summary())

    def format_summary(self) -> str:
        summary = self.summary()
        return "\n".join(
            [
                f"[{self.name}] Transcript normalization summary",
                (
                    f"  transcripts: total={summary['total_transcripts']} "
                    f"valid={summary['valid_transcripts']} ({summary['valid_transcript_rate']:.2%}) "
                    f"invalid={summary['invalid_transcripts']} ({summary['invalid_transcript_rate']:.2%})"
                ),
                (
                    f"  hours: total={summary['total_duration_hours']:.2f} "
                    f"valid_after_filter={summary['valid_duration_hours']:.2f} "
                    f"invalid_removed={summary['invalid_duration_hours']:.2f}"
                ),
                f"  split_hours_before_after_filter: {summary['split_hours']}",
                (
                    f"  chars: total={summary['total_chars']} "
                    f"unique_known={summary['unique_known_chars']} "
                    f"unique_known_rate={summary['unique_known_char_rate']:.2%} "
                    f"unique_unknown={summary['unique_unknown_chars']} "
                    f"unique_unknown_rate={summary['unique_unknown_char_rate']:.2%}"
                ),
                f"  alpha_minus_known_chars: {summary['alpha_minus_known_chars']}",
                f"  split_counts: {summary['split_counts']}",
            ]
        )

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
