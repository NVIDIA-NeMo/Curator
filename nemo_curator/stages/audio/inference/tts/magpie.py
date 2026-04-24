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
Magpie TTS Stage

Generates audio files from transcripts using the Magpie TTS model.
Follows the exact pattern from NeMo Curator:
https://github.com/NVIDIA-NeMo/Curator/blob/main/nemo_curator/stages/audio/common.py

"""

import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar

import soundfile as sf
from loguru import logger
from nemo.collections.tts.models import MagpieTTSModel

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


@dataclass
class MagpieTTSStage(ProcessingStage[AudioTask, AudioTask]):
    """
    Stage for generating audio files from transcripts using the Magpie TTS model.

    Takes a manifest containing transcripts and languages, and generates audio files using the Magpie TTS model.

    """

    # Processing parameters
    generated_audio_dir: str
    model_name: str = "nvidia/magpie_tts_multilingual_357m"
    cache_dir: str | None = None

    # Key names
    text_key: str = "text"
    language_key: str = "language"
    generated_audio_filepath_key: str = "generated_audio_filepath"

    # Stage metadata
    name: str = "MagpieTTS"
    speaker_map: ClassVar[dict[str, int]] = {"John": 0, "Sofia": 1, "Aria": 2, "Jason": 3, "Leo": 4}
    speaker: str = "Sofia"
    speaker_idx: ClassVar[int] = speaker_map["Sofia"]
    _tts_model: Any = field(default=None, repr=False)
    resources: Resources = field(default_factory=lambda: Resources(gpus=1))

    @property
    def _device(self) -> str:
        """Derive device from resources configuration."""
        return "cuda" if self.resources.requires_gpu else "cpu"

    def setup_on_node(
        self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None
    ) -> None:
        if self._tts_model:
            return
        try:
            kwargs: dict[str, Any] = {"model_name": self.model_name, "return_model_file": True}
            if self.cache_dir is not None:
                kwargs["cache_dir"] = self.cache_dir
            MagpieTTSModel.from_pretrained(**kwargs)
        except Exception as e:
            msg = f"Failed to download {self.model_name}"
            raise RuntimeError(msg) from e

    def setup(self, _: WorkerMetadata | None = None) -> None:
        if not self._tts_model:
            try:
                kwargs: dict[str, Any] = {"model_name": self.model_name}
                if self.cache_dir is not None:
                    kwargs["cache_dir"] = self.cache_dir
                self._tts_model = MagpieTTSModel.from_pretrained(**kwargs)
            except Exception as e:
                msg = f"Failed to load {self.model_name}"
                raise RuntimeError(msg) from e
        self._tts_model = self._tts_model.to(self._device)
        logger.info(f"[{self.name}] Initialized MagpieTTS on {self._device}")

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.text_key, self.language_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [
            self.text_key,
            self.language_key,
            self.generated_audio_filepath_key,
        ]

    def generate_audio_file(self, transcript: str, language: str) -> str:
        """
        Generate an audio file from the transcript.
        """
        audio, audio_len = self._tts_model.do_tts(
            transcript, language=language, apply_TN=False, speaker_index=self.speaker_idx
        )
        return audio, audio_len

    def process(self, task: AudioTask) -> AudioTask:
        """
        Process a single task by generating an audio file from the transcript.
        """
        t0 = time.perf_counter()
        data_entry = task.data
        if self.text_key not in data_entry:
            msg = "Text is required"
            raise ValueError(msg)

        if self.language_key not in data_entry:
            msg = "Language is required"
            raise ValueError(msg)

        audio, audio_len = self.generate_audio_file(data_entry[self.text_key], data_entry[self.language_key])
        text_hash = hashlib.sha256(data_entry[self.text_key].encode()).hexdigest()[:16]
        audio_filepath = os.path.join(self.generated_audio_dir, f"{text_hash}.wav")
        sf.write(audio_filepath, audio.cpu().squeeze().numpy(), self._tts_model.output_sample_rate)

        # Update metadata
        data_entry[self.generated_audio_filepath_key] = audio_filepath

        duration_s = audio_len.item() / self._tts_model.output_sample_rate
        self._log_metrics(
            {
                "process_time": time.perf_counter() - t0,
                "duration": max(duration_s, 0.0),
            }
        )
        return task
