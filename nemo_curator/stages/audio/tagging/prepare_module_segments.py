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

"""
Prepare Module Segments Stage.
Merges adjacent same-speaker segments and splits by duration, punctuation, and bandwidth.
"""

import hashlib
import random
from dataclasses import dataclass
from typing import Any

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask

from .merge_alignment_diarization import MergeAlignmentDiarizationStage
from .utils import add_non_speaker_segments


@dataclass
class PrepareModuleSegmentsStage(ProcessingStage[AudioTask, AudioTask]):
    """
    Stage that prepares segments for TTS or ASR by merging and splitting based on
    duration, punctuation, and bandwidth.

    Merges adjacent same-speaker segments, then splits by max duration, pauses, terminal punctuation, and bandwidth changes.

    Args:
        module: Target module: "tts" (single-speaker segments) or "asr" (multi-speaker ok).
        min_duration: Minimum segment duration in seconds. Defaults to 5.0.
        max_duration: Maximum segment duration in seconds. Defaults to 20.0.
        max_pause: Max pause between words to stay in same segment (TTS).
        text_key: Key for segment text. Defaults to "text".
        words_key: Key for word-level alignments in segments. Defaults to "words".
        terminal_punct_marks: Punctuation that ends an utterance (e.g. ".!?"). Defaults to CJK/Latin punct string.
        full_utterance_ratio: Ratio of content to segment at terminal punctuation (0-1).
        punctuation_split_only: If True, split only at punctuation; else also by duration. Defaults to False.
        name: Stage name for logging and output files. Defaults to "PrepareModuleSegments".


    Returns:
        The same data as in the input manifest, but with the new segments added to the metadata.
    """

    module: str = "tts"
    min_duration: float = 5.0
    max_duration: float = 20.0
    max_pause: float = 2.0
    text_key: str = "text"
    words_key: str = "words"
    terminal_punct_marks: str = ".!?。？？！。"  # noqa: RUF001
    full_utterance_ratio: float = 1.0
    punctuation_split_only: bool = False

    name: str = "PrepareModuleSegments"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], ["segments", "duration"]

    def __post_init__(self):
        if self.module not in ("tts", "asr"):
            msg = "Module must be either 'tts' or 'asr'"
            raise ValueError(msg)
        self._rng = random.Random()  # noqa: S311

    def get_words_list_from_all_segments(self, metadata: dict[str, Any]) -> list[dict[str, Any]]:
        """This method gets the words list from all the speaker segments

        Args:
            metadata: A dictionary containing the metadata of the audio file

        Returns:
            A list of words with the following fields:
                - word: The word
                - start: The start time of the word
                - end: The end time of the word
                - speaker: The speaker of the word
                - pesq_squim: The PESQ score of the word if available
                - stoi_squim: The STOI score of the word if available
                - sisdr_squim: The SI-SDR score of the word if available
                - bandwidth: The bandwidth of the word if available
        """
        segments = metadata["segments"]
        audio_duration = metadata["duration"]

        if "overlap_segments" not in metadata:
            add_non_speaker_segments(segments, audio_duration)
            alignment = metadata.get("alignment", [])
            MergeAlignmentDiarizationStage.align_words_to_segments(alignment, segments, self.text_key, self.words_key)

        words = []
        for segment in segments:
            if self.text_key not in segment or (segment[self.text_key] or "").strip() == "":
                continue

            if self.words_key in segment:
                for word in segment[self.words_key]:
                    new_word = dict(word)
                    new_word["speaker"] = segment["speaker"]
                    if "metrics" in segment:
                        m = segment["metrics"]
                        new_word["stoi_squim"] = m.get("stoi_squim") if isinstance(m, dict) else None
                        new_word["sisdr_squim"] = m.get("sisdr_squim") if isinstance(m, dict) else None
                        new_word["pesq_squim"] = m.get("pesq_squim") if isinstance(m, dict) else None
                        new_word["bandwidth"] = m.get("bandwidth") if isinstance(m, dict) else None
                    else:
                        new_word["stoi_squim"] = None
                        new_word["sisdr_squim"] = None
                        new_word["pesq_squim"] = None
                        new_word["bandwidth"] = None
                    words.append(new_word)
            else:
                logger.debug("Found no words in segment")

        return words

    def is_valid_segment(self, segment: dict[str, Any]) -> bool:
        """Return False if segment is a single over-long word or has no text."""
        words = segment.get("words", [])
        if len(words) == 1:
            w = words[0]
            if (w.get("end", 0) - w.get("start", 0)) > self.max_duration:
                return False
        sentence = " ".join([w.get("word", "") for w in words])
        return bool(sentence and sentence.strip())

    def split_segment_by_duration(  # noqa: C901
        self, segment: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Split one segment by duration, pause, and bandwidth (TTS) or duration only (ASR)."""
        words = segment["words"]
        current_segment = {
            "speaker": segment["speaker"],
            "start": segment["start"],
            "end": segment["end"],
            "words": [],
        }
        segments_out = []
        rand_max_duration = (
            self.max_duration
            if self.module == "tts"
            else self._rng.randint(int(self.min_duration), int(self.max_duration))
        )

        for word in words:
            if not current_segment["words"]:
                current_segment = {
                    "speaker": word["speaker"],
                    "start": word["start"],
                    "end": word["end"],
                    "words": [word],
                }
                continue

            # break the current segment if the duration is greater than the max duration and start a new segment
            if (word["end"] - current_segment["start"]) > rand_max_duration:
                if self.is_valid_segment(current_segment):
                    segments_out.append(current_segment)
                current_segment = {
                    "speaker": segment["speaker"],
                    "start": word["start"],
                    "end": word["end"],
                    "words": [word],
                }
                if self.module == "asr":
                    rand_max_duration = self._rng.randint(int(self.min_duration), int(self.max_duration))
                continue
            # break the current segment if the pause is greater than the max pause and start a new segment
            if (
                self.module == "tts"
                and (word["start"] - current_segment["end"] > self.max_pause)
                and (current_segment["end"] - current_segment["start"] >= self.min_duration)
            ):
                if self.is_valid_segment(current_segment):
                    segments_out.append(current_segment)
                current_segment = {
                    "speaker": segment["speaker"],
                    "start": word["start"],
                    "end": word["end"],
                    "words": [word],
                }
                continue
            # break the current segment if the bandwidth is different and start a new segment
            if (
                self.module == "tts"
                and current_segment["words"]
                and word.get("bandwidth") != current_segment["words"][-1].get("bandwidth")
                and (current_segment["end"] - current_segment["start"] >= self.min_duration)
            ):
                if self.is_valid_segment(current_segment):
                    segments_out.append(current_segment)
                current_segment = {
                    "speaker": word["speaker"],
                    "start": word["start"],
                    "end": word["end"],
                    "words": [word],
                }
                continue
            # add the word to the current segment
            current_segment["words"].append(word)
            current_segment["end"] = word["end"]

        if current_segment["words"] and self.is_valid_segment(current_segment):
            segments_out.append(current_segment)

        return segments_out

    def split_segment_by_punctuation(  # noqa: C901, PLR0912, PLR0915
        self, segment: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Split segment at terminal punctuation; fallback to duration split if none, or when over max_duration."""
        words = segment["words"]
        split_points = [
            i for i, word in enumerate(words) if word.get("word") and word["word"][-1] in self.terminal_punct_marks
        ]
        segments_out = []

        if not split_points:
            if self.punctuation_split_only:
                return segments_out
            return self.split_segment_by_duration(segment)

        group_word_start = 0
        current_end = 0
        new_split_points = []

        while current_end < len(split_points):
            end_idx = split_points[current_end]
            current_duration = words[end_idx]["end"] - words[group_word_start]["start"]

            if current_duration < self.min_duration:
                next_end = current_end + 1
                while (
                    next_end < len(split_points)
                    and (words[split_points[next_end]]["end"] - words[group_word_start]["start"]) <= self.max_duration
                ):
                    next_end += 1

                if next_end > current_end + 1:
                    chosen = split_points[next_end - 1]
                    new_split_points.append(chosen)
                    group_word_start = chosen + 1
                    current_end = next_end
                else:
                    chosen = split_points[current_end]
                    new_split_points.append(chosen)
                    group_word_start = chosen + 1
                    current_end += 1
            else:
                chosen = split_points[current_end]
                new_split_points.append(chosen)
                group_word_start = chosen + 1
                current_end += 1

        total_duration = 0
        split_start_index = 0
        for split_end_index in new_split_points:
            total_duration += words[split_end_index]["end"] - words[split_start_index]["start"]
            split_start_index = split_end_index + 1

        required_full_utterance_duration = self.full_utterance_ratio * total_duration

        start = 0
        current_full_utterance_duration = 0
        for end in new_split_points:
            duration = words[end]["end"] - words[start]["start"]
            current_full_utterance_duration += duration

            is_full_utterance_reached = (
                self.full_utterance_ratio < 1.0 and current_full_utterance_duration > required_full_utterance_duration
            )

            if not is_full_utterance_reached:
                sub_segment = {
                    "speaker": segment.get("speaker"),
                    "start": words[start]["start"],
                    "end": words[end]["end"],
                    "words": words[start : end + 1],
                }
            else:
                end = new_split_points[-1]  # noqa: PLW2901
                sub_segment = {
                    "speaker": segment.get("speaker"),
                    "start": words[start]["start"],
                    "end": words[end]["end"],
                    "words": words[start : end + 1],
                }

            if is_full_utterance_reached or duration > self.max_duration:
                segments_out.extend(self.split_segment_by_duration(sub_segment))
            elif self.is_valid_segment(sub_segment):
                segments_out.append(sub_segment)

            start = end + 1
            if is_full_utterance_reached:
                break

        if start < len(words):
            remaining_segment = {
                "speaker": segment["speaker"],
                "start": words[start]["start"],
                "end": words[-1]["end"],
                "words": words[start:],
            }
            segments_out.extend(self.split_segment_by_duration(remaining_segment))

        return segments_out

    def add_new_segments_to_metadata(self, metadata: dict[str, Any], new_segments: list[dict[str, Any]]) -> None:
        """Write new segment list into metadata with text, words, and metrics keys."""
        segments = []
        for new_segment in new_segments:
            if self.module == "tts":
                speaker = new_segment["speaker"]
            else:
                unique_speakers = dict.fromkeys(w["speaker"] for w in new_segment["words"])
                speaker = ",".join(unique_speakers)

            seg = {
                "speaker": speaker,
                "start": new_segment["start"],
                "end": new_segment["end"],
                self.text_key: " ".join(w["word"] for w in new_segment["words"]),
                self.words_key: [
                    {"word": w["word"], "start": w["start"], "end": w["end"]} for w in new_segment["words"]
                ],
                "metrics": {
                    "pesq_squim": [w.get("pesq_squim") for w in new_segment["words"]],
                    "stoi_squim": [w.get("stoi_squim") for w in new_segment["words"]],
                    "sisdr_squim": [w.get("sisdr_squim") for w in new_segment["words"]],
                    "bandwidth": [w.get("bandwidth") for w in new_segment["words"]],
                },
            }
            segments.append(seg)

        metadata["segments"] = segments

    def prepare_asr_segments(self, words: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
        """Prepare ASR segments (multi-speaker per segment allowed)."""
        new_segments = []
        if words:
            current_segment = {
                "speaker": None,
                "start": words[0]["start"],
                "end": words[-1]["end"],
                "words": words,
            }
            new_segments = self.split_segment_by_punctuation(current_segment)
        self.add_new_segments_to_metadata(metadata, new_segments)

    def prepare_tts_segments(self, words: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
        """Prepare TTS segments (single speaker per segment)."""
        new_segments = []
        speaker_segments = []
        current_segment = {"speaker": None, "start": None, "end": None, "words": []}

        for word in words:
            if current_segment["speaker"] is None:
                current_segment = {
                    "speaker": word["speaker"],
                    "start": word["start"],
                    "end": word["end"],
                    "words": [word],
                }
            elif word["speaker"] != current_segment["speaker"]:
                speaker_segments.append(current_segment)
                current_segment = {
                    "speaker": word["speaker"],
                    "start": word["start"],
                    "end": word["end"],
                    "words": [word],
                }
            else:
                current_segment["words"].append(word)
                current_segment["end"] = word["end"]

        if current_segment["words"]:
            speaker_segments.append(current_segment)

        for speaker_segment in speaker_segments:
            if speaker_segment["speaker"] in ("no-speaker", None):
                continue
            new_segments.extend(self.split_segment_by_punctuation(speaker_segment))

        self.add_new_segments_to_metadata(metadata, new_segments)

    def process(self, task: AudioTask) -> AudioTask:
        """Process one entry: build words from segments, then prepare TTS or ASR segments."""
        data_entry = task.data
        entry_id = data_entry.get("audio_filepath", data_entry.get("audio_item_id", ""))
        seed = int(hashlib.md5(entry_id.encode()).hexdigest()[:8], 16)  # noqa: S324
        self._rng.seed(seed)
        try:
            if "segments" not in data_entry:
                logger.info(f"[{self.name}] No segments in metadata for: {data_entry.get('audio_filepath', '')}")
                return task

            words = self.get_words_list_from_all_segments(data_entry)
            if self.module == "asr":
                self.prepare_asr_segments(words, data_entry)
            elif self.module == "tts":
                self.prepare_tts_segments(words, data_entry)
        except Exception as e:  # noqa: BLE001
            logger.error(f"[{self.name}] Error processing entry: {e}")

        return task
