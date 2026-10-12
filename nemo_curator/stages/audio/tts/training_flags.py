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

"""Training-time validity flags for GranaryHiFi cuts.

The training loader keeps a cut when::

    valid_tn_raw_transcript and valid_ipa and valid_context_audio_text
    and not early_cut_off and context_audio_text_overlap < 0.5

Texts are the spoken-form ``GranaryV2.tn_raw`` of the target and of its context segment.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Any

from nemo_curator.stages.audio.tts.audio_io import TAR_BYTES_FIELD, TAR_MEMBER_FIELD, TAR_OFFSET_FIELD, TN_RAW_FIELD
from nemo_curator.stages.audio.tts.context import compute_text_overlap
from nemo_curator.stages.audio.tts.fields import get_dotted, set_dotted
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

EARLY_CUT_OFF_DECISIONS = frozenset({"DROP", "REVIEW"})


def valid_tn_raw_transcript(value: Any) -> bool:  # noqa: ANN401
    """Non-empty, has a letter, and consists only of letters, whitespace and punctuation (Unicode P*).

    Digits and symbols left unexpanded by text normalization make the transcript invalid.
    """
    if not isinstance(value, str):
        return False
    normalized = unicodedata.normalize("NFC", value)
    if not normalized.strip() or not any(char.isalpha() for char in normalized):
        return False
    return all(char.isalpha() or char.isspace() or unicodedata.category(char).startswith("P") for char in normalized)


def valid_ipa(value: Any, error: Any = None) -> bool:  # noqa: ANN401
    """Whether an IPA annotation holds a non-empty string and no error."""
    return isinstance(value, str) and bool(value.strip()) and not error


def is_early_cut_off(decision: Any) -> bool:  # noqa: ANN401
    """``DROP`` (MFA-confirmed) and ``REVIEW`` (undecided / unsupported) both exclude the cut."""
    return isinstance(decision, str) and decision.strip().upper() in EARLY_CUT_OFF_DECISIONS


def ipa_value(annotation: Any) -> tuple[str | None, Any]:  # noqa: ANN401
    """``(ipa, error)`` from an IPA annotation dict ``{ipa, error}`` or a plain string."""
    if isinstance(annotation, dict):
        ipa = annotation.get("ipa")
        return (ipa if isinstance(ipa, str) else None), annotation.get("error")
    if isinstance(annotation, str):
        return annotation, None
    return None, None


def compute_flags(
    target_text: str | None,
    ipa: str | None,
    ipa_error: Any,  # noqa: ANN401
    decision: Any,  # noqa: ANN401
    context_text: str | None,
) -> dict[str, Any]:
    if target_text is None or context_text is None:
        overlap = None
    else:
        overlap = round(compute_text_overlap(target_text, context_text), 4)
    return {
        "valid_tn_raw_transcript": valid_tn_raw_transcript(target_text),
        "valid_ipa": valid_ipa(ipa, ipa_error),
        "early_cut_off": is_early_cut_off(decision),
        "valid_context_audio_text": valid_tn_raw_transcript(context_text),
        "context_audio_text_overlap": overlap,
        "context_tn_raw": context_text,
    }


def passes_training_filter(flags: dict[str, Any], max_context_overlap: float = 0.5) -> bool:
    """The GranaryHiFi training-loader predicate on the flags written by ``TrainingFlagsStage``."""
    overlap = flags.get("context_audio_text_overlap")
    return (
        flags.get("valid_tn_raw_transcript") is True
        and flags.get("valid_ipa") is True
        and flags.get("valid_context_audio_text") is True
        and flags.get("early_cut_off") is False
        and overlap is not None
        and float(overlap) < max_context_overlap
    )


def _text(value: Any) -> str | None:  # noqa: ANN401
    return value if isinstance(value, str) else None


@dataclass
class TrainingFlagsStage(ProcessingStage[AudioTask, AudioTask]):
    """Write ``valid_tn_raw_transcript`` / ``valid_ipa`` / ``early_cut_off`` / ``valid_context_audio_text`` /
    ``context_audio_text_overlap`` (+ ``context_tn_raw``) for every segment under ``output_key``.

    The context's ``tn_raw`` is taken from the context annotation (``context_audio_tn_raw``, written by
    ``ContextSelectionStage``) or else looked up in the same sample row by tar member (and byte range when present).
    Rows are never dropped; segments without a context get ``valid_context_audio_text=False`` and overlap ``None``.
    """

    output_key: str = "tts_flags"
    text_key: str = TN_RAW_FIELD
    ipa_key: str = "ipa"
    early_cut_off_key: str = "early_cut_off_detection"
    context_key: str = "context"
    name: str = "TrainingFlags"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    batch_size: int = 64

    def __post_init__(self) -> None:
        super().__init__()

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.output_key.split(".")[0]]

    def _context_text(self, context: Any, segments: list[dict[str, Any]]) -> str | None:  # noqa: ANN401
        if not isinstance(context, dict) or context.get("context_recording_id") is None:
            return None
        if "context_audio_tn_raw" in context:
            return _text(context.get("context_audio_tn_raw"))
        member = str(context["context_recording_id"])
        offset, size = context.get("context_audio_offset"), context.get("context_audio_bytes")
        matches = [segment for segment in segments if str(get_dotted(segment, TAR_MEMBER_FIELD)) == member]
        if len(matches) > 1 and offset is not None and size is not None:
            matches = [
                segment
                for segment in matches
                if get_dotted(segment, TAR_OFFSET_FIELD) == offset and get_dotted(segment, TAR_BYTES_FIELD) == size
            ]
        return _text(get_dotted(matches[0], self.text_key)) if matches else None

    def flags_for_segment(self, segment: dict[str, Any], segments: list[dict[str, Any]]) -> dict[str, Any]:
        ipa, ipa_error = ipa_value(get_dotted(segment, self.ipa_key))
        decision = get_dotted(segment, f"{self.early_cut_off_key}.decision")
        context_text = self._context_text(get_dotted(segment, self.context_key), segments)
        return compute_flags(_text(get_dotted(segment, self.text_key)), ipa, ipa_error, decision, context_text)

    def annotate_row(self, row: dict[str, Any]) -> None:
        raw_segments = row.get("segments")
        segments = (
            [segment for segment in raw_segments if isinstance(segment, dict)]
            if isinstance(raw_segments, list)
            else [row]
        )
        passing = 0
        for segment in segments:
            flags = self.flags_for_segment(segment, segments)
            passing += passes_training_filter(flags)
            set_dotted(segment, self.output_key, flags)
        self._log_metrics({"segments": float(len(segments)), "training_filter_pass": float(passing)})

    def process(self, task: AudioTask) -> AudioTask:
        self.annotate_row(task.data)
        return task

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if len(tasks) == 0:
            return []
        for task in tasks:
            self.annotate_row(task.data)
        return list(tasks)
