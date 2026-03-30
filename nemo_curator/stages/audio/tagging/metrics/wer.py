# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from dataclasses import dataclass, field
from typing import Any

from nemo.collections.asr.metrics.wer import word_error_rate_detail
from nemo_text_processing.text_normalization import Normalizer

from nemo_curator.stages.audio.common import LegacySpeechStage
from nemo_curator.tasks import AudioBatch


@dataclass
class ComputeWERStage(LegacySpeechStage):
    """
    Stage that computes Word Error Rate (WER), CER, edge CER, and optionally PNC WER/CER.

    Operates on segments within each entry (segment["hypothesis_text_key"] vs segment["reference_text_key"]).

    Args:
        language: Language of the text. Defaults to "en".
        hypothesis_text_key: Key to the hypothesis text. Defaults to "text".
        reference_text_key: Key to the reference text. Defaults to "text".
        num_words_threshold: Number of words to use for normalization. Defaults to 200.
        num_words_look_back: Number of words to look back for normalization. Defaults to 5.
        compute_pnc_wer: Whether to compute PNC WER/CER. Defaults to False.
        pnc_chars: Punctuation characters to use for normalization. Defaults to special punctuation string.
        edge_length: Length of the edge to compute CER. Defaults to 12.

    Returns:
        The same data as in the input data, but with WER, CER, edge CER, and optionally PNC WER/CER added to each segment.
    """

    language: str = "en"
    hypothesis_text_key: str = "text"
    reference_text_key: str = "text"
    num_words_threshold: int = 200
    num_words_look_back: int = 5
    compute_pnc_wer: bool = False
    pnc_chars: str = "،؟.、？¿!,?।"  # noqa: RUF001
    edge_length: int = 12

    # Stage metadata
    name: str = "ComputeWER"

    # Internal state
    _normalizer: Any = field(default=None, repr=False)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], ["segments"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], ["segments", "metrics"]

    def setup(self, worker_metadata: Any = None) -> None:  # noqa: ARG002, ANN401
        """Setup stage."""
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
                if any(c.isdigit() for c in words[chunk_end]):
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

    def process_dataset_entry(self, data_entry: dict[str, Any]) -> list[AudioBatch]:
        """Compute WER, CER, edge CER, and optionally PNC WER/CER per segment."""
        if self._normalizer is None:
            self.setup()

        if "segments" not in data_entry:
            return [AudioBatch(data=[data_entry])]

        for segment in data_entry["segments"]:
            duration = segment["end"] - segment["start"]

            if self.hypothesis_text_key not in segment or self.reference_text_key not in segment:
                continue

            metrics = segment.get("metrics", {})

            hypothesis_pnc, hypothesis_clean = self.normalize_and_clean_text(segment[self.hypothesis_text_key])
            reference_pnc, reference_clean = self.normalize_and_clean_text(segment[self.reference_text_key])

            metrics["char_rate"] = self.get_char_rate(segment[self.hypothesis_text_key], duration)
            metrics["word_rate"] = self.get_word_rate(segment[self.hypothesis_text_key], duration)

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

            segment["metrics"] = metrics

        return [AudioBatch(data=[data_entry])]
