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

from dataclasses import dataclass, field

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


@dataclass
class WhisperHallucinationStage(ProcessingStage[AudioTask, AudioTask]):
    """Detect common Whisper hallucination patterns and flag entries.

    Four checks are applied:
    - Repeated n-grams: low lexical diversity (unique-word ratio <= threshold).
    - Long word: an abnormally long word or a word much longer than its neighbours.
    - Frequent single phrase: the full transcript matches a known hallucination phrase.
    - High char rate: word-chars / duration > max_char_rate (impossible speech rate; short audio
      with dense confabulated text, e.g. Whisper generating a full sentence over 0.1 s).

    When flagged, ``_skip_me`` is set to ``"Hallucination:{name}"`` to track
    which stage instance produced the flag.  By default an already non-empty
    value is preserved; set ``overwrite=True`` to allow overwriting (used by
    the ASR recovery hallucination re-check).
    """

    common_hall_file: str = ""
    unique_words_threshold: float = 0.4
    long_word_threshold: int = 25
    long_word_rel_threshold: float = 3.0
    max_char_rate: float = 40.0
    duration_key: str = "duration"
    text_key: str = "pred_text"
    skip_me_key: str = "_skip_me"
    overwrite: bool = False
    recovery_value: str = ""
    name: str = "WhisperHallucination"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    _phrases: set[str] = field(default_factory=set, init=False, repr=False)
    _setup_called: bool = field(default=False, init=False, repr=False)
    _n_processed: int = field(default=0, init=False, repr=False)
    _n_flagged: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.common_hall_file:
            msg = "common_hall_file is required for WhisperHallucinationStage"
            raise ValueError(msg)

    def setup(self, _worker_metadata: object | None = None) -> None:
        with open(self.common_hall_file, encoding="utf-8") as f:
            phrases = {line.strip() for line in f if line.strip()}
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
            len(phrase) >= self._PREFIX_MATCH_MIN_LEN and cleaned.startswith(phrase)
            for phrase in self._phrases
        )

    def _high_char_rate(self, words: list[str], duration: float) -> bool:
        if duration <= 0:
            return False
        chars = sum(len(w) for w in words)
        return chars / duration > self.max_char_rate

    def _process_single(self, task: AudioTask) -> AudioTask:
        current_flag = str(task.data.get(self.skip_me_key, ""))
        if not self.overwrite and current_flag:
            return task
        text = task.data[self.text_key]
        if not isinstance(text, str) or not text.strip():
            return task
        words = text.split()
        duration = task.data.get(self.duration_key, 0.0) or 0.0

        repeated = self._repeated_ngrams(words)
        long_w = self._long_word(words)
        phrase = self._frequent_single_word(text)
        high_rate = self._high_char_rate(words, duration)

        self._n_processed += 1
        is_hallucinated = repeated or long_w or phrase or high_rate
        was_flagged = current_flag.startswith("Hallucination")
        if is_hallucinated:
            self._n_flagged += 1
            reasons = [
                name
                for name, hit in [
                    ("repeated_ngrams", repeated),
                    ("long_word", long_w),
                    ("phrase_match", phrase),
                    ("high_char_rate", high_rate),
                ]
                if hit
            ]
            logger.debug(
                f"[{self.name}] flagged ({','.join(reasons)}) dur={duration:.2f}s: {text[:80]!r}"
            )
            if was_flagged or not current_flag:
                task.data[self.skip_me_key] = f"Hallucination:{self.name}"
        elif self.overwrite and was_flagged:
            task.data[self.skip_me_key] = self.recovery_value
        return task

    def process(self, task: AudioTask) -> AudioTask:
        if not self._setup_called:
            logger.warning(
                f"WhisperHallucinationStage ({self.name}): setup() was not called before process(). "
                "Calling setup() now — check that your executor invokes setup() on each worker."
            )
            self.setup()
        return self._process_single(task)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if not self._setup_called:
            logger.warning(
                f"WhisperHallucinationStage ({self.name}): setup() was not called before process_batch(). "
                "Calling setup() now — check that your executor invokes setup() on each worker."
            )
            self.setup()
        return [self._process_single(task) for task in tasks]

    def teardown(self) -> None:
        logger.info(
            f"[{self.name}] done — processed={self._n_processed}, flagged={self._n_flagged}"
        )
