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

"""WER / CER computation stage."""

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from nemo.collections.asr.metrics.wer import word_error_rate_detail
from nemo_text_processing.text_normalization import Normalizer

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask

_WER_METRIC_KEYS = (
    "wer",
    "cer",
    "start_cer",
    "end_cer",
    "wer_pnc",
    "cer_pnc",
    "char_rate",
    "word_rate",
)
_WER_SKIP_REASONS = ("missing_configured_text_key", "empty_reference")
_WER_ERROR_SKIP_REASON_PREFIX = "wer_error: "


@dataclass
class ComputeWERStage(ProcessingStage[AudioTask, AudioTask]):
    """
    Stage that computes Word Error Rate (WER), CER, edge CER, and optionally PNC WER/CER.
    This stage cleans the text and normalizes it using NeMo text processing (numbers to words, etc).

    Operates on segments within each entry (audio_segment["hypothesis_text_key"] vs audio_segment["reference_text_key"]).
    If "segments" is not in the data entry, the stage will compute WER, CER, edge CER, and optionally PNC WER/CER for the entire entry.
    Entries or segments missing configured text keys are annotated with a metric skip reason. Missing keys are summarized
    in one warning per configured-key signature on each worker so other segments can continue processing without flooding
    the worker logs.

    Args:
        language: Language of the text. Defaults to "en".
        hypothesis_text_key: Key to the hypothesis text. Defaults to "text".
        reference_text_key: Key to the reference text. Defaults to "text_ref".
        num_words_threshold: Number of words to use for normalization. Defaults to 200.
        num_words_look_back: Number of words to look back for normalization. Defaults to 5.
        compute_pnc_wer: Whether to compute PNC WER/CER. Defaults to False.
        pnc_chars: Punctuation characters to use for normalization. Defaults to special punctuation string.
        edge_length: Length of the edge to compute CER. Defaults to 12.
        segments_key: Key for the segments in the manifest. Defaults to "segments".

    Returns:
        The same data as in the input data, but with WER, CER, edge CER, and optionally PNC WER/CER added to each segment.
    """

    language: str = "en"
    hypothesis_text_key: str = "text"
    reference_text_key: str = "text_ref"
    num_words_threshold: int = 200
    num_words_look_back: int = 5
    compute_pnc_wer: bool = False
    pnc_chars: str = "،؟.、？¿!,?।"  # noqa: RUF001
    edge_length: int = 12

    segments_key: str = "segments"

    # Stage metadata
    name: str = "ComputeWER"

    # Internal state
    _normalizer: Any = field(default=None, repr=False)
    _warned_missing_key_signatures: set[tuple[tuple[str, str], ...]] = field(
        default_factory=set,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.num_words_look_back >= self.num_words_threshold:
            msg = (
                f"num_words_look_back ({self.num_words_look_back}) must be less than "
                f"num_words_threshold ({self.num_words_threshold})"
            )
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], ["metrics"]

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        """Setup stage."""
        self._warned_missing_key_signatures.clear()
        if self._normalizer is None:
            self._normalizer = Normalizer(input_case="cased", lang=self.language.lower())

    def normalize_text(self, text: str) -> str:
        """Normalize text using NeMo text processing (numbers to words, etc)."""
        text = text.replace("<unk>", "").replace("|", "").replace("⁇", "").replace("<", "").replace(">", "")
        text = re.sub(r"\s+", " ", text)
        words = text.split()
        if len(words) <= self.num_words_threshold:
            normalized_text = self._normalizer.normalize(text, verbose=False, punct_post_process=False)
        else:
            final = ""
            shorter_strings = []
            prev_string = []
            remainder_start = 0
            t = self.num_words_threshold

            for i in range(int(len(words) / t)):
                chunk_start = i * t
                chunk_end = chunk_start + t
                if chunk_end < len(words) and any(c.isdigit() for c in words[chunk_end]):
                    shorter_strings.append(
                        " ".join(prev_string + words[chunk_start : chunk_end - self.num_words_look_back])
                    )
                    prev_string = words[chunk_end - self.num_words_look_back : chunk_end]
                else:
                    shorter_strings.append(" ".join(prev_string + words[chunk_start:chunk_end]))
                    prev_string = []
                remainder_start = chunk_end

            shorter_strings.append(" ".join(prev_string + words[remainder_start:]))

            for chunk in shorter_strings:
                final = final + self._normalizer.normalize(chunk, verbose=False, punct_post_process=False) + " "

            normalized_text = final.strip()

        return normalized_text

    def strip_spaces_before_punctuations(self, text: str) -> str:
        """Strip spaces before punctuation characters."""
        return re.sub(f"(\\w)\\s+([{self.pnc_chars}])", r"\1\2", text)

    def normalize_and_clean_text(self, text: str) -> tuple[str, str]:
        """Normalize and clean text. Returns (cleaned_with_punct, cleaned_without_punct)."""
        normalized_text = self.normalize_text(text)
        cleaned_text_with_punct = self.clean_text(normalized_text, retain_pncs=True)
        cleaned_text = self.clean_text(normalized_text, retain_pncs=False)
        return cleaned_text_with_punct, cleaned_text

    def clean_text(self, text: str, retain_pncs: bool = True) -> str:
        """Clean text by removing invalid characters."""
        invalid_chars = '/*":=_-{|}~¨«·»¡¿…‧‹›≪≫!:;ː→'  # noqa: RUF001
        if retain_pncs:
            replace_with_space = list(invalid_chars)
            replace_with_blank = list('`¨´‘“”`ʻ‘“"‘”')  # noqa: RUF001
        else:
            replace_with_space = list(invalid_chars + self.pnc_chars)
            replace_with_blank = list('`¨´‘’“”`ʻ‘’“-"‘”')  # noqa: RUF001
            text = text.lower()

        replace_with_apos = list("‘’ʻ‘’‘’’")  # noqa: RUF001
        text = text.strip()

        for i in replace_with_blank:
            text = text.replace(i, "")
        for i in replace_with_space:
            text = text.replace(i, " ")
        for i in replace_with_apos:
            text = text.replace(i, "'")

        if retain_pncs:
            text = self.strip_spaces_before_punctuations(text)

        return " ".join(text.split())

    def get_char_rate(self, text: str, duration: float) -> float:
        """Calculate character rate (chars per second)."""
        num_chars = len(text.replace(" ", ""))
        return round(num_chars / duration, 2) if duration > 0 else 0.0

    def get_word_rate(self, text: str, duration: float) -> float:
        """Calculate word rate (words per second)."""
        num_words = len(text.split())
        return round(num_words / duration, 2) if duration > 0 else 0.0

    def _missing_text_keys(self, audio_segment: dict[str, Any]) -> list[tuple[str, str]]:
        """Return the configured text fields that are absent from an entry or segment."""
        configured_keys = (
            ("hypothesis_text_key", self.hypothesis_text_key),
            ("reference_text_key", self.reference_text_key),
        )
        return [(field_name, key) for field_name, key in configured_keys if key not in audio_segment]

    def _reset_wer_metrics(self, audio_segment: dict[str, Any]) -> dict[str, Any]:
        """Remove results and skip state owned by this stage before recomputing."""
        metrics = audio_segment.setdefault("metrics", {})
        for metric_key in _WER_METRIC_KEYS:
            metrics.pop(metric_key, None)
        skip_reason = metrics.get("metric_skip_reason")
        if skip_reason in _WER_SKIP_REASONS or (
            isinstance(skip_reason, str) and skip_reason.startswith(_WER_ERROR_SKIP_REASON_PREFIX)
        ):
            metrics.pop("metric_skip_reason")
        return metrics

    def _mark_missing_text_keys(self, audio_segment: dict[str, Any]) -> None:
        """Clear stale WER results and annotate an item whose configured text keys are absent."""
        metrics = self._reset_wer_metrics(audio_segment)
        metrics["metric_skip_reason"] = "missing_configured_text_key"

    def _warn_missing_text_keys(
        self,
        missing_key_counts: Counter[tuple[str, str]],
        *,
        skipped_items: int,
        item_name: str,
        task_id: str = "",
    ) -> None:
        """Emit one worker-local warning for each missing configured-key signature."""
        signature = tuple(sorted(missing_key_counts))
        if signature in self._warned_missing_key_signatures:
            return
        self._warned_missing_key_signatures.add(signature)

        item_label = item_name if skipped_items == 1 else f"{item_name}s"
        task_context = f" in task '{task_id}'" if task_id else ""
        details = "; ".join(
            f"{field_name}={key!r} missing from {count} {item_name if count == 1 else f'{item_name}s'}"
            for (field_name, key), count in missing_key_counts.items()
        )
        logger.warning(
            f"[{self.name}] skipped WER computation for {skipped_items} {item_label}{task_context}: "
            f"configured text key(s) missing ({details}). Further warnings for this configured-key signature "
            "are suppressed on this worker."
        )

    def get_wer(self, audio_segment: dict[str, Any]) -> None:
        """Compute WER, CER, edge CER, and optionally PNC WER/CER per segment."""
        start = audio_segment.get("start", 0)
        end = audio_segment.get("end", audio_segment.get("duration", 0))
        duration = end - start

        metrics = self._reset_wer_metrics(audio_segment)
        missing_text_keys = self._missing_text_keys(audio_segment)
        if missing_text_keys:
            metrics["metric_skip_reason"] = "missing_configured_text_key"
            self._warn_missing_text_keys(Counter(missing_text_keys), skipped_items=1, item_name="audio segment")
            return

        hypothesis_pnc, hypothesis_clean = self.normalize_and_clean_text(audio_segment[self.hypothesis_text_key])
        reference_pnc, reference_clean = self.normalize_and_clean_text(audio_segment[self.reference_text_key])

        if not reference_clean:
            metrics["wer"] = None
            metrics["cer"] = None
            metrics["metric_skip_reason"] = "empty_reference"
            audio_segment["metrics"] = metrics
            return

        metrics["char_rate"] = self.get_char_rate(audio_segment[self.hypothesis_text_key], duration)
        metrics["word_rate"] = self.get_word_rate(audio_segment[self.hypothesis_text_key], duration)

        wer_val, tokens, ins_rate, del_rate, sub_rate = word_error_rate_detail(
            hypotheses=[hypothesis_clean],
            references=[reference_clean],
            use_cer=False,
        )
        metrics["wer"] = {
            "wer": round(wer_val, 4),
            "tokens": tokens,
            "ins_rate": round(ins_rate, 4),
            "del_rate": round(del_rate, 4),
            "sub_rate": round(sub_rate, 4),
        }

        cer_val, tokens, ins_rate, del_rate, sub_rate = word_error_rate_detail(
            hypotheses=[hypothesis_clean],
            references=[reference_clean],
            use_cer=True,
        )
        metrics["cer"] = {
            "cer": round(cer_val, 4),
            "tokens": tokens,
            "ins_rate": round(ins_rate, 4),
            "del_rate": round(del_rate, 4),
            "sub_rate": round(sub_rate, 4),
        }

        (start_cer, tokens, ins_rate, del_rate, sub_rate) = word_error_rate_detail(
            hypotheses=[hypothesis_clean[: self.edge_length]],
            references=[reference_clean[: self.edge_length]],
            use_cer=True,
        )
        metrics["start_cer"] = {
            "cer": round(start_cer, 4),
            "tokens": tokens,
            "ins_rate": round(ins_rate, 4),
            "del_rate": round(del_rate, 4),
            "sub_rate": round(sub_rate, 4),
        }

        end_cer, tokens, ins_rate, del_rate, sub_rate = word_error_rate_detail(
            hypotheses=[hypothesis_clean[-self.edge_length :]],
            references=[reference_clean[-self.edge_length :]],
            use_cer=True,
        )
        metrics["end_cer"] = {
            "cer": round(end_cer, 4),
            "tokens": tokens,
            "ins_rate": round(ins_rate, 4),
            "del_rate": round(del_rate, 4),
            "sub_rate": round(sub_rate, 4),
        }

        if self.compute_pnc_wer:
            (
                wer_pnc,
                tokens_pnc,
                ins_rate_pnc,
                del_rate_pnc,
                sub_rate_pnc,
            ) = word_error_rate_detail(
                hypotheses=[hypothesis_pnc],
                references=[reference_pnc],
                use_cer=False,
            )
            metrics["wer_pnc"] = {
                "wer": round(wer_pnc, 4),
                "tokens": tokens_pnc,
                "ins_rate": round(ins_rate_pnc, 4),
                "del_rate": round(del_rate_pnc, 4),
                "sub_rate": round(sub_rate_pnc, 4),
            }

            (
                cer_pnc,
                tokens_pnc,
                ins_rate_pnc,
                del_rate_pnc,
                sub_rate_pnc,
            ) = word_error_rate_detail(
                hypotheses=[hypothesis_pnc],
                references=[reference_pnc],
                use_cer=True,
            )
            metrics["cer_pnc"] = {
                "cer": round(cer_pnc, 4),
                "tokens": tokens_pnc,
                "ins_rate": round(ins_rate_pnc, 4),
                "del_rate": round(del_rate_pnc, 4),
                "sub_rate": round(sub_rate_pnc, 4),
            }

        audio_segment["metrics"] = metrics

    def process(self, task: AudioTask) -> AudioTask:
        """Compute WER, CER, edge CER, and optionally PNC WER/CER per segment."""
        data_entry = task.data
        if self.segments_key in data_entry:
            missing_key_counts: Counter[tuple[str, str]] = Counter()
            skipped_segments = 0
            for audio_segment in data_entry[self.segments_key]:
                missing_text_keys = self._missing_text_keys(audio_segment)
                if missing_text_keys:
                    self._mark_missing_text_keys(audio_segment)
                    missing_key_counts.update(missing_text_keys)
                    skipped_segments += 1
                    continue
                try:
                    self.get_wer(audio_segment)
                except (KeyError, ValueError) as ex:
                    logger.warning(f"[{self.name}] skipping segment in {task.task_id}: {ex}")
                    metrics = self._reset_wer_metrics(audio_segment)
                    metrics["metric_skip_reason"] = f"{_WER_ERROR_SKIP_REASON_PREFIX}{ex}"
            if skipped_segments:
                self._warn_missing_text_keys(
                    missing_key_counts,
                    skipped_items=skipped_segments,
                    item_name="segment",
                    task_id=task.task_id,
                )
        else:
            missing_text_keys = self._missing_text_keys(data_entry)
            if missing_text_keys:
                self._mark_missing_text_keys(data_entry)
                self._warn_missing_text_keys(
                    Counter(missing_text_keys),
                    skipped_items=1,
                    item_name="entry",
                    task_id=task.task_id,
                )
            else:
                self.get_wer(data_entry)
        return task


@dataclass
class GetPairwiseWerStage(ProcessingStage[AudioTask, AudioTask]):
    """Compute pairwise word-error-rate (WER) as a percentage for each pair of text and pred_text.

    WER is measured between ``data[self.text_key]`` and ``data[self.pred_text_key]``
    and stored as a percentage (e.g. 5.0 means 5% WER).

    Args:
        text_key: Key for the utterance transcript. Defaults to "text".
        pred_text_key: Key for the ASR predictions. Defaults to "pred_text".
        wer_key: Key to store the computed WER percentage. Defaults to "wer_pct".
    """

    name: str = "GetPairwiseWerStage"
    text_key: str = "text"
    pred_text_key: str = "pred_text"
    wer_key: str = "wer_pct"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.text_key, self.pred_text_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.text_key, self.pred_text_key, self.wer_key]

    def process(self, task: AudioTask) -> AudioTask:
        """Compute WER percentage between hypothesis and reference text."""
        hypothesis = task.data.get(self.pred_text_key)
        reference = task.data.get(self.text_key)
        if hypothesis is None or reference is None:
            return task
        wer_val, _, _, _, _ = word_error_rate_detail(
            hypotheses=[hypothesis],
            references=[reference],
            use_cer=False,
        )
        task.data[self.wer_key] = round(wer_val * 100.0, 4)
        return task
