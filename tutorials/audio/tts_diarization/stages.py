# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utterance-level adaptation of the existing Chatterbox stage for the paper."""

from pathlib import Path

import numpy as np
import soundfile as sf

from nemo_curator.stages.audio.tts.chatterbox_tts import ChatterboxTTSStage
from nemo_curator.tasks import AudioTask


class PaperChatterboxTTSStage(ChatterboxTTSStage):
    """Use preassigned per-utterance reference audio with the native TTS stage.

    Inherits model loading, isolated runtime, cache verification, synthesis,
    normalization and resampling. Each input has utterance_id, speaker, text
    and reference_audio. The utterance ID supplies an internal cache identity
    and per-utterance exaggeration draw; it does not identify a conversation.
    References shorter than min_reference_duration seconds and failed synthesis
    raise errors. Successful tasks retain source metadata and the actual prompt.
    """

    def __init__(self, output_audio_dir: str, min_reference_duration: float = 5.0, **kwargs: object):
        super().__init__(
            output_audio_dir=output_audio_dir,
            reference_voices_dataset=str(Path(output_audio_dir).resolve()),
            **kwargs,
        )
        if not np.isfinite(min_reference_duration) or min_reference_duration <= 0:
            msg = "min_reference_duration must be finite and positive"
            raise ValueError(msg)
        self.min_reference_duration = min_reference_duration
        self._reference_audio = ""

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], ["utterance_id", "speaker", "text", "reference_audio"]

    def _load_reference_audio_files(self) -> None:
        """References are assigned in the job manifest rather than discovered."""

    def _assign_reference(self, speaker: str, conversation_id: str) -> tuple[str, str]:  # noqa: ARG002
        return self._reference_audio, speaker

    def _reference_content_hash(self, speaker: str, conversation_id: str) -> str:  # noqa: ARG002
        return self._hash_file_content(self._reference_audio)

    def _generate_turn_audio(self, text: str, reference_wav: str, conversation_id: str) -> np.ndarray:
        audio = super()._generate_turn_audio(text, reference_wav, conversation_id)
        if audio is None:
            msg = "Chatterbox did not generate an utterance"
            raise RuntimeError(msg)
        return audio

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        results = []
        for task in tasks:
            data = task.data
            for key in ("utterance_id", "speaker", "text", "reference_audio"):
                if not isinstance(data.get(key), str) or not data[key].strip():
                    msg = f"Paper TTS job requires nonempty {key}"
                    raise ValueError(msg)
            reference = Path(data["reference_audio"]).expanduser().resolve()
            if sf.info(reference).duration < self.min_reference_duration:
                msg = f"Reference is shorter than {self.min_reference_duration} seconds: {reference}"
                raise ValueError(msg)
            self._reference_audio = str(reference)
            data["reference_audio"] = str(reference)
            data["conversation_id"] = data["utterance_id"]
            data["utterance"] = data["text"]
            for key in ("audio_filepath", "duration", "waveform", "sample_rate", "reference_voice"):
                data.pop(key, None)
            super().process_batch([task])
            if not data.get("audio_filepath") or data.get("duration", 0) <= 0:
                msg = f"TTS failed for utterance {data['utterance_id']}"
                raise RuntimeError(msg)
            data.pop("conversation_id", None)
            results.append(task)
        return results
