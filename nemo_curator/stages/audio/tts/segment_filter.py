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

# ruff: noqa: RUF001  (charset regexes intentionally contain Cyrillic/Greek letters and typographic quotes)

"""Common GranaryHiFi segment filter.

Segments inside a sample row are kept only if every condition passes (``combine="all"``) or any passes
(``combine="any"``). Failing segments are removed from ``segments``; the sample row itself is always kept, even
with an empty segment list. With ``drop=False`` segments are kept and the ids of failed conditions are written to
``reasons_key`` instead (for retention analysis).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nemo_curator.stages.audio.tts.fields import MISSING, get_dotted, set_dotted
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

SUPPORTED_OPERATORS = frozenset(
    {
        "eq",
        "ne",
        "gt",
        "ge",
        "lt",
        "le",
        "exists",
        "not_exists",
        "is_null",
        "not_null",
        "regex",
        "not_regex",
        "in",
        "not_in",
    }
)
REGEX_MATCH_MODES = frozenset({"fullmatch", "match", "search"})
_OPERATOR_ALIASES = {"==": "eq", "equals": "eq", "!=": "ne", "not_equals": "ne"}
_NUMERIC_OPERATORS = frozenset({"gt", "ge", "lt", "le"})

# Characters allowed after the language letters in ``pnc_text``.
CHARSET_PUNCTUATION = r"""\s\.,!?;:…'"‘’‚“”„«»‹›\(\)\[\]\{\}¡¿–—-"""
LETTERS_BY_LANGUAGE: dict[str, str] = {
    "bg": "А-Яа-яЁё",
    "cs": "A-Za-zÁáČčĎďÉéĚěÍíŇňÓóŘřŠšŤťÚúŮůÝýŽž",
    "da": "A-Za-zÆæØøÅå",
    "de": "A-Za-zÄäÖöÜüẞß",
    "el": "Α-Ωα-ωΆάΈέΉήΊίΌόΎύΏώΪϊΫϋΐΰΣς",
    "en": "A-Za-z",
    "es": "A-Za-zÁáÉéÍíÓóÚúÜüÑñ",
    "et": "A-Za-zŠšŽžÕõÄäÖöÜü",
    "fi": "A-Za-zÅåÄäÖöŠšŽž",
    "fr": "A-Za-zÀ-ÖØ-öø-ÿŒœÆæÇç",
    "hr": "A-Za-zČčĆćĐđŠšŽž",
    "hu": "A-Za-zÁáÉéÍíÓóÖöŐőÚúÜüŰű",
    "it": "A-Za-zÀàÈèÉéÌìÍíÎîÒòÓóÙùÚú",
    "lt": "A-Za-zĄąČčĘęĖėĮįŠšŲųŪūŽž",
    "lv": "A-Za-zĀāČčĒēĢģĪīĶķĻļŅņŠšŪūŽž",
    "mt": "A-Za-zĊċĠġĦħŻż",
    "nl": "A-Za-zÁáÉéÍíÓóÚúÀàÈèËëÏïÖöÜü",
    "pl": "A-Za-zĄąĆćĘęŁłŃńÓóŚśŹźŻż",
    "pt": "A-Za-zÁáÀàÂâÃãÇçÉéÊêÍíÓóÔôÕõÚúÜü",
    "ro": "A-Za-zĂăÂâÎîȘșŞşȚțŢţ",
    "ru": "А-Яа-яЁё",
    "sk": "A-Za-zÁáÄäČčĎďÉéÍíĹĺĽľŇňÓóÔôŔŕŠšŤťÚúÝýŽž",
    "sl": "A-Za-zČčŠšŽž",
    "sv": "A-Za-zÅåÄäÖöÉé",
    "uk": "А-ЩЬЮЯҐЄІЇа-щьюяґєіїЁё",
}
PNC_TEXT_CHARSET_BY_LANGUAGE: dict[str, str] = {
    language: f"^[{letters}{CHARSET_PUNCTUATION}]+$" for language, letters in LETTERS_BY_LANGUAGE.items()
}

# Per-source filter thresholds.
FILTER_THRESHOLD_PRESETS: dict[str, dict[str, float]] = {
    "default": {
        "sed_weighted_speech_rate": 0.5,
        "sed_weighted_model_confidence": 0.2,
        "sed_weighted_classifier_confidence": 0.13,
        "diarized_speech_rate": 0.7,
        "num_speakers": 1,
        "utmos_v2_score": 1.0,
    },
    "ytc": {
        "sed_weighted_speech_rate": 0.5,
        "sed_weighted_model_confidence": 0.15,
        "sed_weighted_classifier_confidence": 0.1,
        "diarized_speech_rate": 0.6,
        "num_speakers": 1,
        "utmos_v2_score": 1.5,
    },
    "voxpopuli": {
        "sed_weighted_speech_rate": 0.5,
        "sed_weighted_model_confidence": 0.15,
        "sed_weighted_classifier_confidence": 0.1,
        "diarized_speech_rate": 0.6,
        "num_speakers": 1,
        "utmos_v2_score": 1.5,
    },
    "librilight": {
        "sed_weighted_speech_rate": 0.4,
        "sed_weighted_model_confidence": 0.2,
        "sed_weighted_classifier_confidence": 0.1,
        "diarized_speech_rate": 0.2,
        "num_speakers": 1,
        "utmos_v2_score": 2.5,
    },
}
FILTER_THRESHOLD_PRESETS["yodas"] = dict(FILTER_THRESHOLD_PRESETS["default"])


@dataclass(frozen=True)
class FilterCondition:
    """One filter condition on a dotted ``field`` of a segment (or of the sample row with ``scope="sample"``)."""

    id: str
    field: str
    operator: str
    scope: str = "segment"
    value: Any = None
    pattern: str | None = None
    pattern_by_language: dict[str, str] | None = None
    match_mode: str = "fullmatch"

    def __post_init__(self) -> None:
        operator = _OPERATOR_ALIASES.get(self.operator.lower(), self.operator.lower())
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "scope", self.scope.lower())
        object.__setattr__(self, "match_mode", self.match_mode.lower())
        if self.pattern_by_language:
            object.__setattr__(
                self, "pattern_by_language", {str(k).lower(): str(v) for k, v in self.pattern_by_language.items()}
            )
        if not self.field.strip():
            msg = f"Filter condition {self.id} requires field"
            raise ValueError(msg)
        if self.scope not in {"segment", "sample"}:
            msg = f"Unsupported filter scope for {self.id}: {self.scope}"
            raise ValueError(msg)
        if operator not in SUPPORTED_OPERATORS:
            msg = f"Unsupported filter operator for {self.id}: {operator}"
            raise ValueError(msg)
        if self.match_mode not in REGEX_MATCH_MODES:
            msg = f"Unsupported regex match_mode for {self.id}: {self.match_mode}"
            raise ValueError(msg)
        self._validate_values()

    def _validate_values(self) -> None:
        operator = self.operator
        if operator in {"regex", "not_regex"}:
            if self.pattern is None and not self.pattern_by_language:
                msg = f"Regex filter condition {self.id} requires pattern or pattern_by_language"
                raise ValueError(msg)
            for candidate in [self.pattern, *(self.pattern_by_language or {}).values()]:
                if candidate is None:
                    continue
                try:
                    re.compile(candidate)
                except re.error as exc:
                    msg = f"Invalid regex for filter condition {self.id}: {exc}"
                    raise ValueError(msg) from exc
        if operator in {"in", "not_in"} and not isinstance(self.value, (list, tuple, set, frozenset)):
            msg = f"Filter condition {self.id} with operator {operator} requires a list value"
            raise TypeError(msg)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], index: int = 0) -> FilterCondition:
        if not isinstance(raw, dict):
            msg = f"filter condition {index} must be a mapping"
            raise TypeError(msg)
        pattern = raw.get("pattern")
        return cls(
            id=str(raw.get("id") or f"condition_{index:02d}"),
            field=str(raw.get("field") or "").strip(),
            operator=str(raw.get("operator") or "eq"),
            scope=str(raw.get("scope", "segment")),
            value=raw.get("value"),
            pattern=None if pattern is None else str(pattern),
            pattern_by_language=raw.get("pattern_by_language") or None,
            match_mode=str(raw.get("match_mode", "fullmatch")),
        )


def parse_conditions(raw_conditions: list[FilterCondition | dict[str, Any]]) -> list[FilterCondition]:
    """Normalize ``FilterCondition`` objects or condition dicts into ``FilterCondition`` s."""
    if not isinstance(raw_conditions, (list, tuple)) or not raw_conditions:
        msg = "filter conditions must be a non-empty list"
        raise ValueError(msg)
    conditions = [
        raw if isinstance(raw, FilterCondition) else FilterCondition.from_dict(raw, index)
        for index, raw in enumerate(raw_conditions)
    ]
    ids = [condition.id for condition in conditions]
    duplicates = sorted({cid for cid in ids if ids.count(cid) > 1})
    if duplicates:
        msg = f"Duplicate filter condition id(s): {duplicates}"
        raise ValueError(msg)
    return conditions


def default_conditions(  # noqa: PLR0913
    preset: str = "default",
    *,
    diarization_key: str = "diarization",
    speaker_cluster_key: str = "speaker_cluster",
    sed_key: str = "sed",
    utmos_key: str = "utmosv2",
    pnc_text_key: str = "GranaryV2.pnc_text",
    min_duration: float = 3.0,
    speaker_id_field: str | None = None,
    speaker_id_scope: str | None = None,
) -> list[FilterCondition]:
    """Default filter conditions with the source preset's thresholds.

    Presets: ``default``, ``ytc``, ``yodas``, ``voxpopuli``, ``librilight``. LibriLight reads the speaker id from the
    sample-level ``speaker_id`` field; other presets use ``<speaker_cluster_key>.speaker_id``.
    """
    key = preset.lower()
    if key not in FILTER_THRESHOLD_PRESETS:
        msg = f"Unknown filter preset {preset!r}; expected one of {sorted(FILTER_THRESHOLD_PRESETS)}"
        raise ValueError(msg)
    thresholds = FILTER_THRESHOLD_PRESETS[key]
    if speaker_id_field is None:
        speaker_id_field = "speaker_id" if key == "librilight" else f"{speaker_cluster_key}.speaker_id"
    if speaker_id_scope is None:
        speaker_id_scope = "sample" if key == "librilight" else "segment"
    return [
        FilterCondition("min_duration", "duration", "ge", value=min_duration),
        FilterCondition("pnc_text_has_letter", pnc_text_key, "regex", pattern=r"[^\W\d_]", match_mode="search"),
        FilterCondition(
            "sed_weighted_speech_rate",
            f"{sed_key}.weighted_speech_rate",
            "ge",
            value=thresholds["sed_weighted_speech_rate"],
        ),
        FilterCondition(
            "sed_weighted_model_confidence",
            f"{sed_key}.weighted_model_confidence",
            "ge",
            value=thresholds["sed_weighted_model_confidence"],
        ),
        FilterCondition(
            "sed_weighted_classifier_confidence",
            f"{sed_key}.weighted_classifier_confidence",
            "ge",
            value=thresholds["sed_weighted_classifier_confidence"],
        ),
        FilterCondition(
            "diarized_speech_rate",
            f"{diarization_key}.diarized_speech_rate",
            "ge",
            value=thresholds["diarized_speech_rate"],
        ),
        FilterCondition("num_speakers", f"{diarization_key}.num_speakers", "eq", value=thresholds["num_speakers"]),
        FilterCondition("speaker_id_not_null", speaker_id_field, "not_null", scope=speaker_id_scope),
        FilterCondition("utmos_v2_score", f"{utmos_key}.score", "ge", value=thresholds["utmos_v2_score"]),
        FilterCondition(
            "pnc_text_charset",
            pnc_text_key,
            "regex",
            pattern_by_language=dict(PNC_TEXT_CHARSET_BY_LANGUAGE),
            match_mode="fullmatch",
        ),
    ]


def context_conditions(context_key: str = "context") -> list[FilterCondition]:
    """Conditions that keep only segments with a usable context."""
    return [
        FilterCondition("context_error", f"{context_key}.error", "is_null"),
        FilterCondition("context_recording_id", f"{context_key}.context_recording_id", "not_null"),
        FilterCondition("context_audio_offset", f"{context_key}.context_audio_offset", "not_null"),
        FilterCondition("context_audio_bytes", f"{context_key}.context_audio_bytes", "not_null"),
    ]


def safe_name(value: str) -> str:
    """Counter-safe condition id."""
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_") or "condition"


def logical_language(value: str) -> str:
    """Leading letters, lowercased (``bg000`` -> ``bg``, ``en_x`` -> ``en``)."""
    match = re.match(r"([A-Za-z]+)", value)
    return match.group(1).lower() if match else value.lower()


def numeric_compare(value: Any, threshold: Any, operator: str) -> bool:  # noqa: ANN401, PLR0911
    """Compare a value to a threshold; missing, non-finite or boolean values fail."""
    if isinstance(value, bool) or isinstance(threshold, bool):
        return False
    try:
        numeric_value = float(value)
        numeric_threshold = float(threshold)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(numeric_value) or not math.isfinite(numeric_threshold):
        return False
    if operator == "gt":
        return numeric_value > numeric_threshold
    if operator == "ge":
        return numeric_value >= numeric_threshold
    if operator == "lt":
        return numeric_value < numeric_threshold
    return numeric_value <= numeric_threshold


def regex_matches(pattern: str, value: str, match_mode: str) -> bool:
    """Match ``value`` against ``pattern`` with ``search`` or ``fullmatch`` semantics."""
    compiled = re.compile(pattern)
    if match_mode == "search":
        return compiled.search(value) is not None
    if match_mode == "match":
        return compiled.match(value) is not None
    return compiled.fullmatch(value) is not None


def evaluate_condition(  # noqa: C901, PLR0911
    condition: FilterCondition, payload: dict[str, Any], language: str
) -> tuple[bool, bool]:
    """Return ``(passed, missing)``.

    A missing field fails every operator except ``not_exists``; numeric operators cast with ``float`` and fail on
    bools / non-finite values; regex conditions without a pattern for ``language`` (and no default) fail.
    """
    value = get_dotted(payload, condition.field, MISSING)
    missing = value is MISSING
    operator = condition.operator
    if operator == "exists":
        return not missing, missing
    if operator == "not_exists":
        return missing, missing
    if missing:
        return False, True
    if operator == "is_null":
        return value is None, False
    if operator == "not_null":
        return value is not None, False
    if operator == "eq":
        return value == condition.value, False
    if operator == "ne":
        return value != condition.value, False
    if operator in _NUMERIC_OPERATORS:
        return numeric_compare(value, condition.value, operator), False
    if operator == "in":
        return value in condition.value, False
    if operator == "not_in":
        return value not in condition.value, False
    pattern = (condition.pattern_by_language or {}).get(language, condition.pattern)
    if pattern is None or value is None:
        return False, False
    matched = regex_matches(pattern, str(value), condition.match_mode)
    return (matched if operator == "regex" else not matched), False


@dataclass
class SegmentFilterStage(ProcessingStage[AudioTask, AudioTask]):
    """Filter Granary segments by declarative conditions.

    Args:
        conditions: ``FilterCondition`` objects or condition dicts. Defaults to ``default_conditions(preset)``.
        preset: Threshold preset used when ``conditions`` is empty: default / ytc / yodas / voxpopuli / librilight.
        combine: ``all`` (every condition must pass) or ``any``.
        language: Fixed language for per-language regexes (overrides the row).
        language_keys: Sample-row fields holding the language (first non-empty wins).
        manifest_path_key: Optional row field with the source manifest path; its parent directory name is used as
            the language (e.g. ``.../bg000/manifest.jsonl``) when no language field is set.
        drop: Remove failing segments. With ``drop=False`` all segments are kept and failed condition ids are
            written to ``reasons_key`` (an empty list for passing segments).
    """

    conditions: list[Any] = field(default_factory=list)
    preset: str = "default"
    combine: str = "all"
    language: str | None = None
    language_keys: tuple[str, ...] = ("source_lang", "language", "lang")
    manifest_path_key: str | None = None
    drop: bool = True
    reasons_key: str = "filter_failed_conditions"
    name: str = "SegmentFilter"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    batch_size: int = 64

    def __post_init__(self) -> None:
        super().__init__()
        self.combine = self.combine.lower()
        if self.combine not in {"all", "any"}:
            msg = "combine must be 'all' or 'any'"
            raise ValueError(msg)
        self._conditions = parse_conditions(self.conditions or default_conditions(self.preset))
        self._needs_language = any(c.pattern_by_language for c in self._conditions)
        self.counters: Counter[str] = Counter()

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], ["segments"]

    def language_for_row(self, row: dict[str, Any]) -> str:
        if self.language:
            return logical_language(self.language)
        for key in self.language_keys:
            value = get_dotted(row, key)
            if value:
                return logical_language(str(value))
        if self.manifest_path_key:
            path = get_dotted(row, self.manifest_path_key)
            if path:
                return logical_language(Path(str(path)).parent.name)
        if self._needs_language and row.get("segments"):
            # every per-language regex would fail and silently drop the whole row
            msg = (
                f"SegmentFilterStage: no language for row (looked at {list(self.language_keys)}"
                f"{' and ' + self.manifest_path_key if self.manifest_path_key else ''}); set `language`"
            )
            raise ValueError(msg)
        return ""

    def evaluate_segment(
        self, segment: dict[str, Any], row: dict[str, Any], language: str, stats: Counter[str]
    ) -> list[str]:
        """Return the ids of failed conditions, updating per-condition passed/failed/missing counters."""
        results: list[bool] = []
        failed: list[str] = []
        for condition in self._conditions:
            payload = row if condition.scope == "sample" else segment
            passed, missing = evaluate_condition(condition, payload, language)
            results.append(passed)
            prefix = f"condition_{safe_name(condition.id)}"
            stats[f"{prefix}_{'passed' if passed else 'failed'}"] += 1
            if missing:
                stats[f"{prefix}_missing"] += 1
            if not passed:
                failed.append(condition.id)
        keep = all(results) if self.combine == "all" else any(results)
        return [] if keep else failed

    def filter_row(self, row: dict[str, Any]) -> Counter[str]:
        stats: Counter[str] = Counter()
        segments = row.get("segments")
        if not isinstance(segments, list):
            return stats
        language = self.language_for_row(row)
        kept: list[Any] = []
        for segment in segments:
            stats["input_segments"] += 1
            if not isinstance(segment, dict):
                stats["filtered_segments"] += 1
                if not self.drop:
                    kept.append(segment)
                continue
            failed = self.evaluate_segment(segment, row, language, stats)
            if not failed:
                stats["kept_segments"] += 1
            else:
                stats["filtered_segments"] += 1
            if self.drop:
                if not failed:
                    kept.append(segment)
            else:
                set_dotted(segment, self.reasons_key, failed)
                kept.append(segment)
        row["segments"] = kept
        if stats["kept_segments"] == 0:
            stats["samples_without_kept_segments"] += 1
        return stats

    def process(self, task: AudioTask) -> AudioTask:
        stats = self.filter_row(task.data)
        self.counters.update(stats)
        self._log_metrics({key: float(value) for key, value in stats.items()})
        return task

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if len(tasks) == 0:
            return []
        totals: Counter[str] = Counter()
        for task in tasks:
            totals.update(self.filter_row(task.data))
        self.counters.update(totals)
        self._log_metrics({key: float(value) for key, value in totals.items()})
        return list(tasks)
