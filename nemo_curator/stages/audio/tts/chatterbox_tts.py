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

"""Chatterbox TTS stage for multi-speaker conversation audio generation."""

from __future__ import annotations

import atexit
import glob
import hashlib
import os
import random
import shutil
import tempfile
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.tts import ChatterboxTTS
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

SUPPORTED_LANGUAGES = frozenset({
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it",
    "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
})


class ChatterboxTTSStage(ProcessingStage[AudioTask, AudioTask]):
    """Generate audio for conversation turns using ChatterboxTTS.

    Supports both English-only (``ChatterboxTTS``) and multilingual
    (``ChatterboxMultilingualTTS``) models. When ``language`` is ``None``,
    the English model is used; otherwise the multilingual model is loaded
    for the specified language code.

    Each input ``AudioTask`` represents one conversation turn with fields
    ``utterance`` (or ``text``), ``speaker``, and ``conversation_id``.
    The output ``AudioTask`` is enriched with ``audio_filepath`` and
    ``duration``.

    Speaker voices are assigned from a reference dataset and stay
    consistent within a conversation. Reference audio can optionally be
    cleaned of silences using paired RTTM files.

    Args:
        output_audio_dir: Directory for generated WAV files.
        reference_voices_dataset: Root path containing reference audio.
            Supports ``wavs/<dialog>/<speaker>.wav`` layout (with optional
            ``rttms/`` siblings) and MLS layout ``<spk>/<book>/<seg>.flac``.
        language: ISO 639-1 language code, or ``None`` for English-only model.
        device: Torch device string.
        max_reference_duration: Maximum seconds of reference speech to use.
        sample_rate: Output WAV sample rate (Chatterbox default 24000).
        cfg_weight: Classifier-free guidance weight.
        exaggeration: Emotion exaggeration. A single float for a fixed value,
            or a ``[min, max]`` list to randomly vary per conversation.
        temperature: Sampling temperature.
        repetition_penalty: Repetition penalty (default higher for multilingual).
        min_p: Min-p sampling parameter.
        top_p: Top-p sampling parameter.
        normalize_audio: Whether to normalise output volume.
        normalize_level: Target loudness in dB.
    """

    name = "ChatterboxTTSStage"
    resources = Resources(gpus=1)

    def __init__(
        self,
        output_audio_dir: str,
        reference_voices_dataset: str,
        language: str | None = None,
        device: str = "cuda",
        max_reference_duration: float = 60.0,
        sample_rate: int = 24000,
        cfg_weight: float = 0.5,
        exaggeration: float | list[float] = 0.5,
        temperature: float = 0.8,
        repetition_penalty: float | None = None,
        min_p: float = 0.05,
        top_p: float = 1.0,
        normalize_audio: bool = True,
        normalize_level: float = -20.0,
    ):
        super().__init__()

        self.output_audio_dir = output_audio_dir
        self.reference_voices_dataset = reference_voices_dataset
        self.language = language
        self.device = device
        self.max_reference_duration = max_reference_duration
        self.sample_rate = sample_rate
        self.cfg_weight = cfg_weight
        self.temperature = temperature
        self.min_p = min_p
        self.top_p = top_p
        self.normalize_audio = normalize_audio
        self.normalize_level = normalize_level

        if language is not None and language.lower() not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{language}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_LANGUAGES))}"
            )
        if language is not None:
            self.language = language.lower()

        if repetition_penalty is not None:
            self.repetition_penalty = repetition_penalty
        else:
            self.repetition_penalty = 2.0 if language else 1.2

        if isinstance(exaggeration, (list, tuple)) and len(exaggeration) == 2:
            self.exaggeration_range: tuple[float, float] | None = tuple(exaggeration)
            self.exaggeration: float = float(exaggeration[0])
        else:
            self.exaggeration_range = None
            self.exaggeration = float(exaggeration)

        self.model = None
        self.reference_wavs_list: list[str] | None = None
        self._reference_layout: str = "wavs"
        self._speaker_audio_map: dict[str, list[str]] = {}

        self.speaker_to_reference: dict[str, str] = {}
        self._speaker_to_original_wav: dict[str, str] = {}
        self.speaker_to_ref_id: dict[str, str] = {}
        self.conversation_exaggeration: dict[str, float] = {}

        self.temp_dir: str | None = None
        self._rng = random.Random()

    def setup(self, worker_metadata: Any = None) -> None:  # noqa: ARG002
        """Load the TTS model and discover reference audio files."""
        os.makedirs(self.output_audio_dir, exist_ok=True)
        self._init_temp_dir()
        self._load_model()
        self._load_reference_audio_files()

    def teardown(self) -> None:
        """Release model and clean up temp files."""
        self.model = None
        self.speaker_to_reference.clear()
        self._speaker_to_original_wav.clear()
        self.speaker_to_ref_id.clear()
        self.conversation_exaggeration.clear()
        self._cleanup_temp_dir()

    def _init_temp_dir(self) -> None:
        if self.temp_dir is None or not os.path.exists(self.temp_dir):
            self.temp_dir = tempfile.mkdtemp(prefix="chatterbox_ref_")
            atexit.register(self._cleanup_temp_dir)

    def _cleanup_temp_dir(self) -> None:
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception:
                pass
            self.temp_dir = None

    def _load_model(self) -> None:
        """Load ChatterboxTTS or ChatterboxMultilingualTTS."""
        if self.language:
            os.environ["TRANSFORMERS_ATTN_IMPLEMENTATION"] = "eager"
            try:
                import chatterbox.models.t3.llama_configs as _llama_cfgs
                for _cfg_dict in _llama_cfgs.LLAMA_CONFIGS.values():
                    _cfg_dict["attn_implementation"] = "eager"
            except (ImportError, AttributeError):
                pass

            self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
            logger.info(f"Loaded ChatterboxMultilingualTTS (language={self.language})")
        else:
            self.model = ChatterboxTTS.from_pretrained(device=self.device)
            logger.info("Loaded ChatterboxTTS (English)")

    def _load_reference_audio_files(self) -> None:
        """Discover reference audio files in wavs/ or MLS layout."""
        wav_pattern = os.path.join(self.reference_voices_dataset, "wavs", "*", "*.wav")
        self.reference_wavs_list = glob.glob(wav_pattern)

        if self.reference_wavs_list:
            self._reference_layout = "wavs"
            logger.info(f"Found {len(self.reference_wavs_list)} reference files (wavs/ layout)")
            return

        flac_pattern = os.path.join(self.reference_voices_dataset, "*", "*", "*.flac")
        self.reference_wavs_list = glob.glob(flac_pattern)

        if self.reference_wavs_list:
            self._reference_layout = "mls"
            self._speaker_audio_map = {}
            for fpath in self.reference_wavs_list:
                speaker_id = fpath.split(os.sep)[-3]
                self._speaker_audio_map.setdefault(speaker_id, []).append(fpath)
            logger.info(
                f"Found {len(self.reference_wavs_list)} reference files "
                f"from {len(self._speaker_audio_map)} speakers (MLS layout)"
            )
            return

        raise ValueError(
            f"No reference audio found in {self.reference_voices_dataset}. "
            f"Expected wavs/*/*.wav or */*/*.flac"
        )

    def _process_audio_with_rttm(self, audio_filepath: str, rttm_filepath: str) -> str:
        """Strip silences using RTTM speech segments, up to max_reference_duration."""
        if not os.path.exists(rttm_filepath):
            return audio_filepath

        try:
            audio, sr = ta.load(audio_filepath)

            speech_segments: list[tuple[float, float]] = []
            with open(rttm_filepath, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5 and parts[0] == "SPEAKER":
                        start = float(parts[3])
                        dur = float(parts[4])
                        speech_segments.append((start, start + dur))

            if not speech_segments:
                return audio_filepath

            speech_segments.sort()
            chunks: list[torch.Tensor] = []
            total_dur = 0.0

            for start, end in speech_segments:
                if total_dur >= self.max_reference_duration:
                    break
                s_sample = int(start * sr)
                e_sample = int(end * sr)
                seg = audio[:, s_sample:e_sample]
                seg_dur = seg.shape[1] / sr

                if total_dur + seg_dur > self.max_reference_duration:
                    remaining = self.max_reference_duration - total_dur
                    seg = seg[:, : int(remaining * sr)]

                chunks.append(seg)
                total_dur += seg.shape[1] / sr

            if not chunks:
                return audio_filepath

            processed = torch.cat(chunks, dim=1)
            unique_name = hashlib.md5(audio_filepath.encode()).hexdigest()[:8] + "_" + os.path.basename(audio_filepath)
            out_path = os.path.join(self.temp_dir, unique_name)
            ta.save(out_path, processed, sr)
            return out_path

        except Exception as e:
            logger.warning(f"RTTM processing failed for {audio_filepath}: {e}")
            return audio_filepath

    def _get_reference_audio_wavs(self, already_taken: set[str]) -> tuple[str, str]:
        """Select a reference WAV, optionally clean with RTTM.

        Returns:
            Tuple of (processed_path, original_path). The original path is
            needed for deduplication since RTTM processing changes the path.
        """
        available = [
            w for w in self.reference_wavs_list
            if w not in already_taken
        ]
        if not available:
            available = self.reference_wavs_list

        selected = self._rng.choice(available)

        parts = selected.split(os.sep)
        dialog_id = parts[-2]
        speaker_id = os.path.splitext(parts[-1])[0]
        rttm_path = os.path.join(
            self.reference_voices_dataset, "rttms", dialog_id, f"{speaker_id}.rttm"
        )
        processed = self._process_audio_with_rttm(selected, rttm_path)
        return processed, selected

    def _get_reference_audio_mls(self, already_taken_ids: set[str]) -> tuple[str, str]:
        """Select an MLS speaker, concatenate segments as reference."""
        available = [
            s for s in self._speaker_audio_map if s not in already_taken_ids
        ]
        if not available:
            available = list(self._speaker_audio_map.keys())

        chosen = self._rng.choice(available)
        files = list(self._speaker_audio_map[chosen])
        self._rng.shuffle(files)

        chunks: list[torch.Tensor] = []
        total_dur = 0.0
        last_sr = 16000

        for fpath in files:
            if total_dur >= self.max_reference_duration:
                break
            try:
                audio, sr = ta.load(fpath)
                last_sr = sr
                seg_dur = audio.shape[1] / sr
                if total_dur + seg_dur > self.max_reference_duration:
                    remaining = self.max_reference_duration - total_dur
                    audio = audio[:, : int(remaining * sr)]
                chunks.append(audio)
                total_dur += audio.shape[1] / sr
            except Exception as e:
                logger.warning(f"Failed to load {fpath}: {e}")

        if not chunks:
            return files[0], chosen

        concatenated = torch.cat(chunks, dim=1)
        out_path = os.path.join(self.temp_dir, f"ref_{chosen}.wav")
        ta.save(out_path, concatenated, last_sr)
        return out_path, chosen

    def _assign_reference(self, speaker: str, conversation_id: str) -> tuple[str, str]:
        """Get or assign a reference audio file for a speaker in a conversation.

        Returns:
            Tuple of (ref_path, ref_id) where ref_id is a stable identifier
            for the reference voice (MLS speaker ID or dialog/speaker tag).
        """
        key = f"{conversation_id}_{speaker}"

        if key in self.speaker_to_reference:
            return self.speaker_to_reference[key], self.speaker_to_ref_id[key]

        if self._reference_layout == "mls":
            already_taken_ids = {
                self.speaker_to_ref_id[k]
                for k in self.speaker_to_ref_id
                if k.startswith(f"{conversation_id}_")
            }
            ref_path, ref_id = self._get_reference_audio_mls(already_taken_ids)
        else:
            already_taken = {
                orig for k, orig in self._speaker_to_original_wav.items()
                if k.startswith(f"{conversation_id}_")
            }
            ref_path, original_wav = self._get_reference_audio_wavs(already_taken)
            self._speaker_to_original_wav[key] = original_wav
            parts = original_wav.split(os.sep)
            ref_id = f"{parts[-2]}/{os.path.splitext(parts[-1])[0]}"

        self.speaker_to_reference[key] = ref_path
        self.speaker_to_ref_id[key] = ref_id
        return ref_path, ref_id

    def _get_exaggeration(self, conversation_id: str) -> float:
        """Get consistent exaggeration for a conversation (random range support)."""
        if self.exaggeration_range is None:
            return self.exaggeration

        if conversation_id not in self.conversation_exaggeration:
            lo, hi = self.exaggeration_range
            self.conversation_exaggeration[conversation_id] = self._rng.uniform(lo, hi)
        return self.conversation_exaggeration[conversation_id]

    def _generate_turn_audio(
        self, text: str, reference_wav: str, conversation_id: str
    ) -> np.ndarray:
        """Run ChatterboxTTS inference for a single turn."""
        try:
            exag = self._get_exaggeration(conversation_id)
            generate_kwargs: dict[str, Any] = {
                "audio_prompt_path": reference_wav,
                "cfg_weight": self.cfg_weight,
                "exaggeration": exag,
                "temperature": self.temperature,
                "repetition_penalty": self.repetition_penalty,
                "min_p": self.min_p,
                "top_p": self.top_p,
            }
            if self.language:
                generate_kwargs["language_id"] = self.language

            with torch.inference_mode():
                wav = self.model.generate(text, **generate_kwargs)

            if self.normalize_audio:
                wav = self._normalize_audio(wav)

            return wav.squeeze(0).cpu().numpy()
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            return np.zeros(self.sample_rate * 2)

    def _normalize_audio(self, wav: torch.Tensor) -> torch.Tensor:
        """RMS-based normalisation with clipping protection."""
        rms = torch.sqrt(torch.mean(wav ** 2))
        if rms < 1e-10:
            return wav
        current_db = 20 * torch.log10(rms + 1e-8)
        gain = 10 ** ((self.normalize_level - current_db) / 20)
        normalised = wav * gain
        peak = torch.max(torch.abs(normalised))
        if peak > 1.0:
            normalised = normalised / peak * 0.99
        return normalised

    @staticmethod
    def _output_filename(conversation_id: str, speaker: str, text: str) -> str:
        """Deterministic filename: ``{conv_id_hash}_{speaker}_{text_hash}.wav``."""
        conv_hash = hashlib.md5(conversation_id.encode("utf-8")).hexdigest()[:12]
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
        return f"{conv_hash}_{speaker}_{text_hash}.wav"

    def process(self, task: AudioTask) -> AudioTask:
        """Generate audio for a single conversation turn."""
        return self.process_batch([task])[0]

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        """Generate audio for a batch of conversation turns.

        Each turn is synthesised independently (TTS is autoregressive and
        not easily vectorised), but batching allows the model to stay warm
        across turns and avoids repeated setup overhead.
        """
        if not tasks:
            return []

        output_tasks: list[AudioTask] = []
        for task in tasks:
            data = task.data

            text = data.get("utterance") or data.get("text", "")
            if not text or not text.strip():
                logger.warning(f"Skipping task {task.task_id}: no text")
                output_tasks.append(task)
                continue

            text = text.strip()
            speaker = data.get("speaker", "unknown")
            conversation_id = data.get("conversation_id", "unknown")

            reference_wav, ref_id = self._assign_reference(speaker, conversation_id)

            filename = self._output_filename(conversation_id, speaker, text)
            audio_path = os.path.join(self.output_audio_dir, filename)

            if os.path.exists(audio_path):
                audio_data, _ = sf.read(audio_path)
            else:
                audio_data = self._generate_turn_audio(
                    text, reference_wav, conversation_id
                )
                sf.write(audio_path, audio_data, self.sample_rate)

            duration = len(audio_data) / self.sample_rate

            out_data = dict(data)
            out_data["audio_filepath"] = audio_path
            out_data["duration"] = duration
            out_data["reference_voice"] = ref_id

            output_tasks.append(
                AudioTask(
                    data=out_data,
                    task_id=task.task_id,
                    dataset_name=task.dataset_name,
                )
            )

            logger.info(
                f"[TTS] {conversation_id[:8]}/{speaker}: "
                f"{duration:.2f}s -> {filename}"
            )

        return output_tasks
