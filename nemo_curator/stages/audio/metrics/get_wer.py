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


import re
import time

import editdistance
from nemo.collections.asr.metrics.wer import word_error_rate_detail

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask

try:
    from nemo_text_processing.text_normalization.normalize import Normalizer

    NEMO_TEXT_PROCESSING_AVAILABLE = True
except ImportError:
    NEMO_TEXT_PROCESSING_AVAILABLE = False
from dataclasses import dataclass, field


def get_wer(text: str, pred_text: str) -> float:
    text_words = text.split()
    pred_text_words = pred_text.split()
    word_dist = editdistance.eval(text_words, pred_text_words)

    num_words = len(text_words)
    return round(word_dist / num_words * 100.0, 2)


def get_cer(text: str, pred_text: str) -> float:
    char_dist = editdistance.eval(text, pred_text)
    num_chars = len(text)
    return round(char_dist / num_chars * 100.0, 2)


def get_charrate(text: str, duration: float) -> float:
    num_chars = len(text)
    return round(num_chars / duration, 2)


def get_wordrate(text: str, duration: float) -> float:
    num_words = len(text.split())
    return round(num_words / duration, 2)


@dataclass
class GetPairwiseWerStage(ProcessingStage[AudioTask, AudioTask]):
    """Count pairwise word-error-rate (WER) * 100% for each pair of text and pred_text.

    WER is measured between ``data[self.text_key]`` and ``data[self.pred_text_key]``.

    Args:
        text_key: Key for the utterance transcript. Defaults to "text".
        pred_text_key: Key for the ASR predictions. Defaults to "pred_text".
        wer_key: Key to store the computed WER. Defaults to "wer".
    """

    name: str = "GetPairwiseWerStage"
    text_key: str = "text"
    pred_text_key: str = "pred_text"
    wer_key: str = "wer"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.text_key, self.pred_text_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.text_key, self.pred_text_key, self.wer_key]

    def process(self, task: AudioTask) -> AudioTask:
        task.data[self.wer_key] = get_wer(task.data[self.text_key], task.data[self.pred_text_key])
        return task


@dataclass
class ComputeNormalizedWERMetricsStage(ProcessingStage[AudioTask, AudioTask]):
    """
    This stage computes the Word Error Rate (WER) between two text keys.
    It normalizes the text and computes the WER, CER, punctuation-normalized WER and CER, edge CER metrics.

    Example:
        .. code-block:: yaml
            - _target_: nemo_curator.stages.audio.metrics.ComputeNormalizedWERMetricsStage
              language: en
              hypothesis_text_key: text
              reference_text_key: reference_text
              num_words_threshold: 200
              num_words_look_back: 5
              compute_pnc_wer: true
              pnc_chars: ",.!?"
    """

    language: str = "en"
    segments_key: str = "segments"
    hypothesis_text_key: str = "text"
    reference_text_key: str = "text"
    num_words_threshold: int = 200
    num_words_look_back: int = 5
    compute_pnc_wer: bool = False
    pnc_chars: str = ",.!?"
    edge_length: int = 12

    # Stage metadata
    name: str = "ComputeNormalizedWERMetrics"

    normalizer: Normalizer = field(default=None, repr=False)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.segments_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.segments_key]

    def setup(self, _: WorkerMetadata | None = None) -> None:
        """Setup stage."""
        if self.normalizer is None:
            self.normalizer = Normalizer(input_case="cased", lang=self.language.lower())

    def normalize_text(self, text: str) -> str:
        """
        Normalize the text using nemo text processing library which can convert numbers to words, etc. If the text is longer than num_words_threshold,
        it will be split into chunks of num_words_threshold size and then normalized.
        If the text contains numbers, it will be split before the number and the last few words will be kept for context.
        If the text does not contain numbers, it will be split into chunks of num_words_threshold size and then normalized.
        The normalized text is returned.
        """
        text = text.replace("<unk>", "").replace("|", "").replace("⁇", "").replace("<", "").replace(">", "")
        text = re.sub(r"\s+", " ", text)
        words = text.split()
        if len(words) <= self.num_words_threshold:
            if self.normalizer is not None:
                normalized_text = self.normalizer.normalize(text, verbose=False, punct_post_process=False)
            else:
                normalized_text = text
        else:
            final = ""
            shorter_strings = []
            prev_string = []
            i = 0

            # Process text in chunks of num_words_threshold size
            for i in range(int(len(words) / self.num_words_threshold) - 1):
                # Check if next word contains numbers
                if any(c.isdigit() for c in words[i * self.num_words_threshold + self.num_words_threshold]):
                    # If number found, split before it and keep last few words for context
                    shorter_strings.append(
                        " ".join(
                            prev_string
                            + words[
                                i * self.num_words_threshold : i * self.num_words_threshold
                                + self.num_words_threshold
                                - self.num_words_look_back
                            ]
                        )
                    )
                    prev_string = words[
                        i * self.num_words_threshold + self.num_words_threshold - self.num_words_look_back : i
                        * self.num_words_threshold
                        + self.num_words_threshold
                    ]
                else:
                    # If no number, take full chunk and clear context
                    shorter_strings.append(
                        " ".join(
                            prev_string
                            + words[
                                i * self.num_words_threshold : i * self.num_words_threshold + self.num_words_threshold
                            ]
                        )
                    )
                    prev_string = []

            # Add remaining words to the list
            shorter_strings.append(" ".join(prev_string + words[i * self.num_words_threshold :]))

            # Normalize each chunk and combine into final text
            for i in shorter_strings:
                if self.normalizer is not None:
                    final = final + self.normalizer.normalize(i, verbose=False, punct_post_process=False) + " "
                else:
                    final = final + i + " "

            normalized_text = final.strip()

        return normalized_text

    def strip_spaces_before_punctuations(self, text: str) -> str:
        """
        Strip spaces before punctuation characters.
        """
        return re.sub(f"(\\w)\\s+([{self.pnc_chars}])", r"\1\2", text)

    def normalize_and_clean_text(self, text: str) -> tuple[str, str]:
        """
        Normalize and clean the text.
        Returns:
            Tuple[str, str]: The cleaned text containing punctuations and capitalizations and the cleaned text without punctuations and capitalizations.
        """
        normalized_text = self.normalize_text(text)
        cleaned_text_with_punct = self.clean_text(normalized_text, retain_pncs=True)
        cleaned_text = self.clean_text(normalized_text, retain_pncs=False)
        return cleaned_text_with_punct, cleaned_text

    def clean_text(self, text: str, retain_pncs: bool = True) -> str:
        """
        Clean the text by removing invalid characters and replacing them with spaces or blanks.
        Args:
            text (str): The text to clean
            retain_pncs (bool): Whether to retain punctuation and capitalization characters in the text. Defaults to True.
        Returns:
            str: The cleaned text containing punctuations and capitalizations.
        """
        invalid_chars = '/*":=_-{|}~¨«·»¡¿…‧<>≪≫!:;→'
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
        """
        Calculate the character rate in characters per minute
        """
        num_chars = len(text.replace(" ", ""))
        return round((num_chars) / duration, 2) if duration > 0 else 0.0

    def get_word_rate(self, text: str, duration: float) -> float:
        """
        Calculate the word rate in words per minute
        """
        num_words = len(text.split())
        return round((num_words) / duration, 2) if duration > 0 else 0.0

    def process(self, task: AudioTask) -> AudioTask:
        """
        Process a single dataset entry to compute WER, CER, punctuation-normalized WER and CER, edge CER metrics.
        """
        t0 = time.perf_counter()
        data_entry = task.data

        for segment in data_entry[self.segments_key]:
            duration = segment["end"] - segment["start"]
            metrics = segment.get("metrics", {})

            if self.hypothesis_text_key not in segment or self.reference_text_key not in segment:
                continue

            # Normalize and clean the hypothesis and reference text
            hypothesis_pnc, hypothesis_clean = self.normalize_and_clean_text(segment[self.hypothesis_text_key])
            reference_pnc, reference_clean = self.normalize_and_clean_text(segment[self.reference_text_key])

            # Add character and word rate metrics
            metrics["char_rate"] = self.get_char_rate(segment[self.hypothesis_text_key], duration)
            metrics["word_rate"] = self.get_word_rate(segment[self.hypothesis_text_key], duration)

            # Add WER metrics
            wer, tokens, ins_rate, del_rate, sub_rate = word_error_rate_detail(
                hypotheses=[hypothesis_clean],
                references=[reference_clean],
                use_cer=False,
            )

            metrics["wer"] = {
                "wer": round(wer, 4),
                "tokens": tokens,
                "ins_rate": round(ins_rate, 4),
                "del_rate": round(del_rate, 4),
                "sub_rate": round(sub_rate, 4),
            }

            # Add CER metrics
            cer, tokens, ins_rate, del_rate, sub_rate = word_error_rate_detail(
                hypotheses=[hypothesis_clean],
                references=[reference_clean],
                use_cer=True,
            )

            metrics["cer"] = {
                "cer": round(cer, 4),
                "tokens": tokens,
                "ins_rate": round(ins_rate, 4),
                "del_rate": round(del_rate, 4),
                "sub_rate": round(sub_rate, 4),
            }

            # Add edge CER metrics
            start_cer, tokens, ins_rate, del_rate, sub_rate = word_error_rate_detail(
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

            # Add punctuation-normalized WER and CER metrics
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

            segment["metrics"] = metrics

        self._log_metrics(
            {
                "process_time": time.perf_counter() - t0,
            }
        )

        return task
