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

"""Rule-based detection of common ASR hallucination patterns.

No model and no GPU: these are cheap lexical heuristics meant to run over
transcripts that an ASR stage already produced, flagging rows that look
confabulated rather than transcribed.
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

# Languages that legitimately produce very long single words, either by
# agglutination or by compounding. The relative long-word check is disabled
# for these languages; the absolute threshold remains caller-configurable.
AGGLUTINATIVE_COMPOUNDING_LANGS: frozenset[str] = frozenset(
    {
        "fi",
        "hu",
        "et",  # Uralic agglutinative
        "de",
        "nl",
        "da",
        "sv",
        "no",  # Germanic compounding
    }
)


def _set_note(task_data: dict[str, Any], stage_name: str, value: str, notes_key: str) -> None:
    """Record this stage's decision under its own key in the notes dict.

    Keying by stage name lets downstream analysis query per-stage decisions
    without parsing a concatenated string, and keeps two instances of the same
    filter from overwriting each other.
    """
    notes = task_data.get(notes_key)
    if not isinstance(notes, dict):
        notes = {}
        task_data[notes_key] = notes
    notes[stage_name] = value


@dataclass
class WhisperHallucinationStage(ProcessingStage[AudioTask, AudioTask]):
    """Flag transcripts that show common ASR hallucination patterns.

    Four independent checks run, and any single hit flags the row:

    - **Repeated n-grams**: the unique-word ratio falls to or below
      ``unique_words_threshold``, the signature of a decoder stuck in a loop.
    - **Long word**: a word at or over the absolute length bar, or one much
      longer than the next-longest word by ``long_word_rel_threshold``.
    - **Phrase match**: the whole transcript is a known hallucination phrase
      from ``common_hall_file`` (think "Thank you." over silence).
    - **High char rate**: characters per second exceed ``max_char_rate``, an
      impossible speaking rate, which catches dense text confabulated over a
      very short clip.

    When flagged, ``skip_me_key`` is set to ``"Hallucination:{name}"`` so the
    originating instance stays identifiable. An existing non-hallucination flag
    from some other filter is never clobbered. Setting ``overwrite=True`` turns
    the stage into a re-check that can also *clear* a previous hallucination
    flag when a second-pass transcript now looks clean, which is how an ASR
    recovery pass promotes its result.

    ``recovery_value`` is a caller-selected note written only when
    ``overwrite=True`` clears an existing hallucination flag. For example,
    ``recovery_value="Recovered:ASR"`` produces the note ``"recovered:asr"``.
    Leaving it empty produces ``"recovered"``.

    Run the bundled YAML from the Curator repository root over a JSONL manifest
    that already contains ``pred_text``, ``_skipme``, and ``duration``:

    .. code-block:: bash

        python nemo_curator/config/run.py \\
            --config-path ../../tutorials/audio/whisper_hallucination \\
            --config-name pipeline \\
            manifest_path=/absolute/path/to/asr_manifest.jsonl \\
            output_path=/absolute/path/to/filtered_manifest.jsonl
    """

    common_hall_file: str = ""
    unique_words_threshold: float = 0.4
    long_word_threshold: int = 25
    long_word_rel_threshold: float = 3.0
    max_char_rate: float = 40.0
    language_key: str = "language"
    duration_key: str = "duration"
    text_key: str = "pred_text"
    skip_me_key: str = "_skipme"
    notes_key: str = "additional_notes"
    overwrite: bool = False
    recovery_value: str = ""
    name: str = "WhisperHallucination"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    _phrases: set[str] = field(default_factory=set, init=False, repr=False)
    _setup_called: bool = field(default=False, init=False, repr=False)
    _n_processed: int = field(default=0, init=False, repr=False)
    _n_flagged: int = field(default=0, init=False, repr=False)

    # Phrases shorter than this are matched exactly; longer ones also match
    # as prefixes. Keep this out of the dataclass constructor/YAML surface:
    # lowering it can make common short phrases match unrelated transcripts.
    _PREFIX_MATCH_MIN_LEN: ClassVar[int] = 8

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
        return [], [self.skip_me_key, self.notes_key]

    def _repeated_ngrams(self, words: list[str]) -> bool:
        if not words:
            return False
        return len(set(words)) / len(words) <= self.unique_words_threshold

    def _long_word(self, words: list[str], threshold: int | None = None, skip_relative: bool = False) -> bool:
        if not words:
            return False
        effective_threshold = threshold if threshold is not None else self.long_word_threshold
        lengths = sorted(len(w) for w in words)
        if lengths[-1] >= effective_threshold:
            return True
        if skip_relative:
            return False
        if len(lengths) > 1 and lengths[-2] > 0:
            return (lengths[-1] - lengths[-2]) / lengths[-2] >= self.long_word_rel_threshold
        return False

    def _frequent_single_word(self, text: str) -> bool:
        cleaned = text.strip().replace(".", "").replace("?", "").replace("!", "")
        if cleaned in self._phrases:
            return True
        return any(
            len(phrase) >= self._PREFIX_MATCH_MIN_LEN and cleaned.startswith(phrase) for phrase in self._phrases
        )

    def _high_char_rate(self, words: list[str], duration: float) -> bool:
        if duration <= 0:
            return False
        chars = sum(len(w) for w in words)
        return chars / duration > self.max_char_rate

    def _is_agglutinative(self, task: AudioTask) -> bool:
        lang = str(task.data.get(self.language_key, "")).lower().strip()
        return lang in AGGLUTINATIVE_COMPOUNDING_LANGS

    def _process_single(self, task: AudioTask) -> AudioTask:
        current_flag = str(task.data.get(self.skip_me_key, ""))
        if not self.overwrite and current_flag:
            _set_note(task.data, self.name, "skipped (flagged)", self.notes_key)
            return task
        text = task.data[self.text_key]
        if not isinstance(text, str) or not text.strip():
            _set_note(task.data, self.name, "skipped (empty text)", self.notes_key)
            return task
        words = text.split()
        duration = task.data.get(self.duration_key, 0.0) or 0.0

        is_agglutinative = self._is_agglutinative(task)
        repeated = self._repeated_ngrams(words)
        long_w = self._long_word(words, threshold=self.long_word_threshold, skip_relative=is_agglutinative)
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
            logger.debug(f"[{self.name}] flagged ({','.join(reasons)}) dur={duration:.2f}s: {text[:80]!r}")
            if was_flagged or not current_flag:
                task.data[self.skip_me_key] = f"Hallucination:{self.name}"
            _set_note(task.data, self.name, f"hallucination ({', '.join(reasons)})", self.notes_key)
        elif self.overwrite and was_flagged:
            task.data[self.skip_me_key] = ""
            recovery_note = self.recovery_value.lower() if self.recovery_value else "recovered"
            _set_note(task.data, self.name, recovery_note, self.notes_key)
        else:
            _set_note(task.data, self.name, "passed", self.notes_key)
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
        logger.info(f"[{self.name}] done — processed={self._n_processed}, flagged={self._n_flagged}")
