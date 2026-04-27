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
Merge Alignment and Diarization Stage.
"""

import time
from dataclasses import dataclass

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask


@dataclass
class MergeAlignmentDiarizationStage(ProcessingStage[AudioTask, AudioTask]):
    """
    Stage that merges alignment and diarization information.

    Takes a jsonl data containing both alignment and diarization information
    and merges the alignment info into the diarization segments.

    Args:
        text_key: Key to add text to segments
        words_key: Key to add word alignments to segments

    Returns:
        The same data as in the input manifest, but with alignment information merged into
        the diarization segments.
    Example:
        .. code-block:: yaml

            - _target_: nemo_curator.stages.audio.tagging.merge_alignment_diarization.MergeAlignmentDiarizationStage
              text_key: "text"
              words_key: "words"
    """

    # Output keys
    text_key: str = "text"
    words_key: str = "words"

    # Stage metadata
    name: str = "MergeAlignmentDiarization"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], ["alignment", "segments"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], ["alignment", "segments"]

    @staticmethod
    def align_words_to_segments(
        alignment: list[dict],
        segments: list[dict],
        text_key: str,
        words_key: str,
    ) -> None:
        """
        Align words to segments based on timestamps.

        Iterates through the alignment and finds words that belong in each segment,
        joining them together to form the text for the segment.

        Args:
            alignment: List of words with start and end times
            segments: List of segments with start and end times
            text_key: Key to add text to segment
            words_key: Key to add words to segment

        Returns:
            None

        Alignment example:
        [
            {
                "word": "Hello",
                "start": 0.0,
                "end": 1.0
            },...
        ]

        Segments example:
        [
            {
                "speaker": "speaker1",
                "start": 0.0,
                "end": 3.0
            },...
        ]

        Output:
        [
            {
                "speaker": "speaker1",
                "start": 0.0,
                "end": 3.0,
                "text": "Hello there",
                "words": [
                    {
                        "word": "Hello",
                        "start": 0.0,
                        "end": 1.0
                    },
                    {
                        "word": "there",
                        "start": 1.0,
                        "end": 3.0
                    },...
                ]
            },...
        ]
        """
        last_word_idx = 0
        alignment = sorted(alignment, key=lambda x: x.get("start", 0))
        segments = sorted(segments, key=lambda x: x.get("start", 0))

        if len(alignment) > 0 and len(segments) > 0:
            for i, segment in enumerate(segments):
                words_in_segment = []

                while last_word_idx < len(alignment):
                    word = alignment[last_word_idx]
                    word_start = word.get("start", 0)
                    word_end = word.get("end", 0)

                    if word_start >= segment.get("end", 0):
                        break

                    if word_start >= segment.get("start", 0) and word_end <= segment.get("end", 0):
                        words_in_segment.append(word)
                        last_word_idx += 1
                    else:
                        # Check overlap with current segment
                        current_overlap = max(
                            0,
                            min(word_end, segment.get("end", 0)) - max(word_start, segment.get("start", 0)),
                        )

                        # Check overlap with next segment if exists
                        if i < len(segments) - 1:
                            next_segment = segments[i + 1]
                            next_overlap = max(
                                0,
                                min(word_end, next_segment.get("end", 0))
                                - max(word_start, next_segment.get("start", 0)),
                            )
                        else:
                            next_overlap = 0

                        # Assign based on overlap comparison
                        if current_overlap >= next_overlap and current_overlap > 0:
                            words_in_segment.append(word)
                            last_word_idx += 1
                        elif next_overlap > current_overlap:
                            break
                        else:
                            logger.debug(
                                f"Word '{word.get('word', '')}' at [{word_start:.3f}, {word_end:.3f}] "
                                f"falls in gap between segments; skipping."
                            )
                            last_word_idx += 1

                    if last_word_idx == len(alignment):
                        break

                segment[text_key] = " ".join([x.get("word", "") for x in words_in_segment])
                segment[words_key] = words_in_segment

    def process(self, task: AudioTask) -> AudioTask:
        """Process entry to merge alignment and diarization."""
        t0 = time.perf_counter()
        data_entry = task.data
        alignment = data_entry.get("alignment", [])
        segments = data_entry.get("segments", [])

        if alignment and segments:
            self.align_words_to_segments(alignment, segments, self.text_key, self.words_key)

        self._log_metrics(
            {
                "process_time": time.perf_counter() - t0,
                "segments_merged": len(segments),
                "words_aligned": len(alignment),
            }
        )
        return task
