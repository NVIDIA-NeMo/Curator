# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata


@dataclass
class WhisperHallucinationStage(ProcessingStage[AudioTask, AudioTask]):
    """Detect common Whisper hallucination patterns and flag entries.

    Five checks are applied:
    - Repeated n-grams: low lexical diversity (unique-word ratio <= threshold).
    - Long word: an abnormally long word or a word much longer than its neighbours.
    - Frequent single phrase: the full transcript matches a known hallucination phrase.
    - Low char rate: word-chars / duration <= char_rate_threshold (sparse text over long audio).
    - High char rate: word-chars / duration > max_char_rate (impossible speech rate; short audio
      with dense confabulated text, e.g. Whisper generating a full sentence over 0.1 s).

    If any check triggers and ``skip_me`` is still empty, it is set to
    ``"Hallucination"`` (an already non-empty value is preserved).
    """

    name: str = "WhisperHallucination"
    common_hall_file: str = ""
    unique_words_threshold: float = 0.4
    long_word_threshold: int = 25
    long_word_rel_threshold: float = 3.0
    char_rate_threshold: float = 4.0
    max_char_rate: float = 40.0
    duration_key: str = "duration"
    text_key: str = "pred_text"
    skip_me_key: str = "skip_me"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    _phrases: set[str] = field(default_factory=set, init=False, repr=False)
    _setup_called: bool = field(default=False, init=False, repr=False)
    _n_processed: int = field(default=0, init=False, repr=False)
    _n_flagged: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.common_hall_file:
            msg = "common_hall_file is required for WhisperHallucinationStage"
            raise ValueError(msg)

    @staticmethod
    def _strip_frequency_count(line: str) -> str:
        """Strip optional trailing frequency count from a phrase line.

        Community hallucination lists often include counts, e.g.
        ``"Thank you 1297"`` → ``"Thank you"``.
        """
        parts = line.rsplit(maxsplit=1)
        has_trailing_count = len(parts) > 1 and parts[1].lstrip("-").isdigit()
        if has_trailing_count:
            return parts[0]
        return line

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        with open(self.common_hall_file, encoding="utf-8") as f:
            phrases = {self._strip_frequency_count(line.strip()) for line in f if line.strip()}
        self._phrases = phrases
        self._setup_called = True
        logger.info(f"WhisperHallucinationStage: loaded {len(phrases)} phrases from {self.common_hall_file}")

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.text_key, self.skip_me_key, self.duration_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.skip_me_key]

    def _repeated_ngrams(self, words: list[str]) -> bool:
        if not words:
            return False
        return len(set(words)) / len(words) <= self.unique_words_threshold

    def _long_word(self, words: list[str]) -> bool:
        if not words:
            return False
        lengths = sorted(len(w) for w in words)
        if lengths[-1] >= self.long_word_threshold:
            return True
        if len(lengths) > 1 and lengths[-2] > 0:
            return (lengths[-1] - lengths[-2]) / lengths[-2] >= self.long_word_rel_threshold
        return False

    # Phrases shorter than this are matched exactly; longer ones also match as prefixes.
    _PREFIX_MATCH_MIN_LEN: int = 8

    def _frequent_single_word(self, text: str) -> bool:
        cleaned = text.strip().replace(".", "").replace("?", "").replace("!", "")
        if cleaned in self._phrases:
            return True
        return any(
            len(phrase) >= self._PREFIX_MATCH_MIN_LEN and cleaned.startswith(phrase) for phrase in self._phrases
        )

    def _low_char_rate(self, words: list[str], duration: float) -> bool:
        if duration <= 0:
            return False
        chars = sum(len(w) for w in words)
        return chars / duration <= self.char_rate_threshold

    def _high_char_rate(self, words: list[str], duration: float) -> bool:
        if duration <= 0:
            return False
        chars = sum(len(w) for w in words)
        return chars / duration > self.max_char_rate

    def process(self, task: AudioTask) -> AudioTask:
        if not self._setup_called:
            logger.warning(
                f"WhisperHallucinationStage ({self.name}): setup() was not called before process(). "
                "Calling setup() now — check that your executor invokes setup() on each worker."
            )
            self.setup()
        text = task.data[self.text_key]
        if not isinstance(text, str):
            return task
        words = text.split()
        duration = task.data.get(self.duration_key, 0.0) or 0.0

        if not words:
            self._n_processed += 1
            if not task.data[self.skip_me_key]:
                task.data[self.skip_me_key] = "Empty text"
                self._n_flagged += 1
            return task

        repeated = self._repeated_ngrams(words)
        long_w = self._long_word(words)
        phrase = self._frequent_single_word(text)
        low_rate = self._low_char_rate(words, duration)
        high_rate = self._high_char_rate(words, duration)

        self._n_processed += 1
        if repeated or long_w or phrase or low_rate or high_rate:
            self._n_flagged += 1
            reasons = [
                name
                for name, hit in [
                    ("repeated_ngrams", repeated),
                    ("long_word", long_w),
                    ("phrase_match", phrase),
                    ("low_char_rate", low_rate),
                    ("high_char_rate", high_rate),
                ]
                if hit
            ]
            logger.debug(f"[{self.name}] flagged ({','.join(reasons)}) dur={duration:.2f}s: {text[:80]!r}")
            if not task.data[self.skip_me_key]:
                task.data[self.skip_me_key] = "Hallucination"
        return task

    def teardown(self) -> None:
        logger.info(f"[{self.name}] done — processed={self._n_processed}, flagged={self._n_flagged}")
