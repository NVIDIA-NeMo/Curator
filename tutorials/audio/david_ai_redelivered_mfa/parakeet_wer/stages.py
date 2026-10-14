# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom Curator stages for segment extraction and per-segment WER."""

from __future__ import annotations

import hashlib
import re
import unicodedata
import wave
from dataclasses import dataclass, field
from pathlib import Path

from num2words import num2words

from nemo_curator.stages.audio.inference.asr.asr_nemo import InferenceAsrNemoStage
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

_NUMBER_TOKEN = re.compile(r"^\d+$")
_WORD_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


@dataclass
class ParallelInferenceAsrNemoStage(InferenceAsrNemoStage):
    """NeMo ASR stage with an explicit number of one-GPU workers."""

    worker_count: int = 1
    name: str = "ParakeetSegmentASR"

    def num_workers(self) -> int:
        return self.worker_count


def normalize_wer_text(text: str) -> str:
    """Normalize reference and ASR text consistently before WER."""
    folded = "".join(
        character
        for character in unicodedata.normalize("NFKD", text.casefold())
        if unicodedata.category(character) != "Mn"
    )
    words: list[str] = []
    for token in _WORD_TOKEN.findall(folded):
        if _NUMBER_TOKEN.fullmatch(token):
            spoken = str(num2words(int(token), lang="en"))
            words.extend(_WORD_TOKEN.findall(spoken.casefold()))
        else:
            words.append(token)
    return " ".join(words)


def word_error_counts(reference: str, hypothesis: str) -> dict[str, int | float | None]:
    """Compute Levenshtein WER and S/D/I counts for normalized text."""
    ref_words = normalize_wer_text(reference).split()
    hyp_words = normalize_wer_text(hypothesis).split()
    if not ref_words:
        return {
            "reference_words": 0,
            "hypothesis_words": len(hyp_words),
            "substitutions": 0,
            "deletions": 0,
            "insertions": len(hyp_words),
            "errors": len(hyp_words),
            "wer_pct": 0.0 if not hyp_words else None,
        }

    previous = [(index, 0, 0, index) for index in range(len(hyp_words) + 1)]
    for ref_index, ref_word in enumerate(ref_words, start=1):
        current = [(ref_index, 0, ref_index, 0)]
        for hyp_index, hyp_word in enumerate(hyp_words, start=1):
            if ref_word == hyp_word:
                current.append(previous[hyp_index - 1])
                continue
            sub = previous[hyp_index - 1]
            delete = previous[hyp_index]
            insert = current[hyp_index - 1]
            candidates = (
                (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),
                (delete[0] + 1, delete[1], delete[2] + 1, delete[3]),
                (insert[0] + 1, insert[1], insert[2], insert[3] + 1),
            )
            current.append(min(candidates, key=lambda item: (item[0], item[3], item[2], item[1])))
        previous = current

    errors, substitutions, deletions, insertions = previous[-1]
    return {
        "reference_words": len(ref_words),
        "hypothesis_words": len(hyp_words),
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "errors": errors,
        "wer_pct": round(100.0 * errors / len(ref_words), 6),
    }


@dataclass
class SegmentClipExtractionStage(ProcessingStage[AudioTask, AudioTask]):
    """Extract exact manifest intervals from masked mono 16 kHz WAVs."""

    scratch_dir: str = ""
    sample_rate: int = 16000
    minimum_clip_duration: float = 0.1
    name: str = "ExtractMaskedSegment"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    batch_size: int = 1

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], ["audio_filepath", "start", "end"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], ["segment_audio_filepath", "source_audio_filepath", "clip_start", "clip_end"]

    def process(self, task: AudioTask) -> AudioTask:
        source = Path(task.data["audio_filepath"])
        start = float(task.data["start"])
        end = float(task.data["end"])
        if end <= start:
            msg = f"invalid segment interval: {start} >= {end}"
            raise ValueError(msg)

        digest = hashlib.sha256(
            f"{task.data['recording_id']}:{task.data['segment_index']}:{start}:{end}".encode()
        ).hexdigest()[:20]
        destination = Path(self.scratch_dir) / f"{digest}.wav"
        destination.parent.mkdir(parents=True, exist_ok=True)

        with wave.open(str(source), "rb") as reader:
            if reader.getframerate() != self.sample_rate or reader.getnchannels() != 1:
                msg = (
                    f"masked audio must be mono {self.sample_rate} Hz WAV: "
                    f"{source} is {reader.getnchannels()}ch/{reader.getframerate()}Hz"
                )
                raise ValueError(msg)
            total_frames = reader.getnframes()
            start_frame = min(total_frames, max(0, round(start * self.sample_rate)))
            end_frame = min(total_frames, round(end * self.sample_rate))
            minimum_frames = max(1, round(self.minimum_clip_duration * self.sample_rate))
            if end_frame - start_frame < minimum_frames:
                center_frame = round((start + end) * self.sample_rate / 2)
                start_frame = max(0, center_frame - minimum_frames // 2)
                end_frame = min(total_frames, start_frame + minimum_frames)
                start_frame = max(0, end_frame - minimum_frames)
            reader.setpos(start_frame)
            frames = reader.readframes(max(0, end_frame - start_frame))
            sample_width = reader.getsampwidth()
            compression_type = reader.getcomptype()
            compression_name = reader.getcompname()

        with wave.open(str(destination), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(sample_width)
            writer.setframerate(self.sample_rate)
            writer.setcomptype(compression_type, compression_name)
            writer.writeframes(frames)

        task.data["source_audio_filepath"] = str(source)
        task.data["segment_audio_filepath"] = str(destination)
        task.data["clip_start"] = round(start_frame / self.sample_rate, 6)
        task.data["clip_end"] = round(end_frame / self.sample_rate, 6)
        return task


@dataclass
class SegmentWERStage(ProcessingStage[AudioTask, AudioTask]):
    """Compute normalized per-segment WER and remove the temporary clip."""

    reference_key: str = "text_raw"
    hypothesis_key: str = "pred_text"
    cleanup_clips: bool = True
    name: str = "ComputeSegmentWER"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    batch_size: int = 1

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.reference_key, self.hypothesis_key, "segment_audio_filepath"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], ["wer_pct", "wer_details", "audio_filepath"]

    def process(self, task: AudioTask) -> AudioTask:
        clip = Path(task.data["segment_audio_filepath"])
        try:
            details = word_error_counts(
                str(task.data[self.reference_key]),
                str(task.data[self.hypothesis_key]),
            )
            task.data["wer_details"] = details
            task.data["wer_pct"] = details["wer_pct"]
            task.data["audio_filepath"] = task.data["source_audio_filepath"]
            return task
        finally:
            if self.cleanup_clips:
                clip.unlink(missing_ok=True)
