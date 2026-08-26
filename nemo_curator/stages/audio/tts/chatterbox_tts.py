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

import contextlib
import glob
import hashlib
import json
import math
import os
import shutil
import tempfile
import uuid
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import soundfile as sf
import torch
import torchaudio as ta
from huggingface_hub import snapshot_download
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata

SUPPORTED_LANGUAGES = frozenset(
    {
        "ar",
        "da",
        "de",
        "el",
        "en",
        "es",
        "fi",
        "fr",
        "he",
        "hi",
        "it",
        "ja",
        "ko",
        "ms",
        "nl",
        "no",
        "pl",
        "pt",
        "ru",
        "sv",
        "sw",
        "tr",
        "zh",
    }
)

# chatterbox-tts is not a Curator dependency because it hard-pins
# transformers==5.2.0 and torch==2.6.0 (incompatible with Curator's
# transformers>=4.56,<5.0 / torch==2.10.0), which would make uv lock and the
# audio extras unresolvable. It is installed at runtime into an isolated Ray
# virtualenv via the stage's ``runtime_env`` instead (see ``_CHATTERBOX_RUNTIME_ENV``).
_CHATTERBOX_PIP_SPEC = "chatterbox-tts==0.1.7"
# chatterbox's resemble-perth watermarker imports pkg_resources at runtime.
# pkg_resources ships with setuptools but was REMOVED in setuptools>=81, and Ray's
# isolated virtualenv (unlike a uv --seed venv) does not seed setuptools at all.
# Without setuptools<81 here, perth.PerthImplicitWatermarker silently becomes None
# and ChatterboxTTS() raises "'NoneType' object is not callable".
_CHATTERBOX_SETUPTOOLS_SPEC = "setuptools<81"
# Ray clones the whole base venv before installing chatterbox's pins on top, so
# the base env's torchvision (built against Curator's own torch) is left behind
# unchanged. It is ABI-incompatible with chatterbox's torch==2.6.0 (custom-op
# registration in transformers' vision utils raises "operator torchvision::nms
# does not exist" as soon as anything -- including plain LlamaModel -- triggers
# transformers' lazy import of image_utils), so it must be pinned to the release
# matching torch 2.6.0 alongside chatterbox, not left for pip to skip.
_CHATTERBOX_TORCHVISION_SPEC = "torchvision==0.21.0"
_CHATTERBOX_RUNTIME_ENV: dict[str, Any] = {
    # pip_check=False: chatterbox's pinned transformers/torch differ from the
    # cloned base venv; we only need them consistent inside this isolated env.
    "pip": {
        "packages": [_CHATTERBOX_PIP_SPEC, _CHATTERBOX_SETUPTOOLS_SPEC, _CHATTERBOX_TORCHVISION_SPEC],
        "pip_check": False,
    },
}

_CHATTERBOX_REPO_ID = "ResembleAI/chatterbox"
# Pin the model repository independently of the Python package.  Loading a
# moving ``main`` revision would make cache entries irreproducible even with an
# exact chatterbox-tts version.
_CHATTERBOX_MODEL_REVISION = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
_ENGLISH_MODEL_FILES = (
    "ve.safetensors",
    "t3_cfg.safetensors",
    "s3gen.safetensors",
    "tokenizer.json",
    "conds.pt",
)
_MULTILINGUAL_MODEL_FILES = (
    "ve.pt",
    "t3_mtl23ls_v2.safetensors",
    "s3gen.pt",
    "grapheme_mtl_merged_expanded_v1.json",
    "conds.pt",
    "Cangjie5_TC.json",
)

# The multilingual model requires eager attention. Loading it mutates
# process-global state (this env var and chatterbox's LLAMA_CONFIGS), which
# teardown() restores so other transformer stages in the same worker are
# unaffected. ``_UNSET`` marks state that was absent before we touched it.
_ATTN_ENV = "TRANSFORMERS_ATTN_IMPLEMENTATION"
_UNSET = object()

# Chatterbox's S3Gen decoder always synthesises at this rate; both
# ChatterboxTTS and ChatterboxMultilingualTTS set ``self.sr`` to it after
# loading (see their ``__init__``). Used only as a defensive fallback if a
# model object doesn't expose ``.sr`` -- the real source of truth at runtime
# is always ``self.model.sr``, never the user-configured ``self.sample_rate``.
_CHATTERBOX_NATIVE_SR = 24000

# Bump whenever a code change alters what audio a given cache manifest
# produces in a way the manifest fields don't otherwise capture (e.g. a
# generation-algorithm change). This invalidates every existing cache entry
# by changing the filename hash, forcing regeneration instead of silently
# reusing stale audio.
_CACHE_SCHEMA_VERSION = 1
_REFERENCE_PREPROCESSING_SCHEMA_VERSION = 1


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

    Voice/exaggeration assignment is a deterministic, stateless hash of
    ``(conversation_id, speaker)`` (see ``_assign_reference``), not a
    stateful/random pick. This matters for multi-GPU and multi-node runs:
    Ray Data/Xenna instantiate several independent actor copies of this
    one-GPU stage and distribute batch-size-one turns across them in
    whatever order they arrive, so a given conversation's turns can be
    processed by different actors/nodes. Deriving the assignment purely from
    ``(conversation_id, speaker)`` guarantees every actor computes the same
    voice for the same character, regardless of which actor/node handles a
    turn or in what order turns arrive -- unlike a random or history-based
    pick, which would only stay consistent within a single actor's memory.

    Args:
        output_audio_dir: Directory for generated WAV files.
        reference_voices_dataset: Root path containing reference audio.
            Supports ``wavs/<dialog>/<speaker>.wav`` layout (with optional
            ``rttms/`` siblings) and MLS layout ``<spk>/<book>/<seg>.flac``.
        language: ISO 639-1 language code, or ``None`` for English-only model.
        device: Torch device string.
        cache_dir: HuggingFace cache directory for Chatterbox model weights.
        max_reference_duration: Maximum seconds of reference speech to use.
        sample_rate: Output WAV sample rate. Chatterbox always synthesises at
            24000 Hz internally; if this differs, output is resampled to it.
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
    # Turns are synthesised serially in process_batch (Chatterbox generate()
    # is single-text and per-voice-conditioned), so one task per batch.
    batch_size = 1
    # Run in an isolated Ray virtualenv that pip-installs chatterbox-tts, so its
    # transformers==5.2.0 / torch==2.6.0 pins never collide with Curator's main
    # environment or uv.lock. To reuse a pre-provisioned environment (e.g. one
    # where chatterbox is already installed) disable this with
    # ``ChatterboxTTSStage(...).with_(runtime_env={})``.
    runtime_env: ClassVar[dict[str, Any] | None] = _CHATTERBOX_RUNTIME_ENV

    def __init__(  # noqa: PLR0913
        self,
        output_audio_dir: str,
        reference_voices_dataset: str,
        language: str | None = None,
        device: str = "cuda",
        cache_dir: str | None = None,
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
        self.cache_dir = cache_dir
        self.max_reference_duration = max_reference_duration
        self.sample_rate = sample_rate
        self.cfg_weight = cfg_weight
        self.temperature = temperature
        self.min_p = min_p
        self.top_p = top_p
        self.normalize_audio = normalize_audio
        self.normalize_level = normalize_level

        if language is not None and language.lower() not in SUPPORTED_LANGUAGES:
            msg = f"Unsupported language '{language}'. Supported: {', '.join(sorted(SUPPORTED_LANGUAGES))}"
            raise ValueError(msg)
        if language is not None:
            self.language = language.lower()

        _multilingual_default_penalty = 2.0
        if repetition_penalty is not None:
            self.repetition_penalty = repetition_penalty
        else:
            self.repetition_penalty = _multilingual_default_penalty if language else 1.2

        _exag_range_len = 2
        if isinstance(exaggeration, (list, tuple)) and len(exaggeration) == _exag_range_len:
            self.exaggeration_range: tuple[float, float] | None = tuple(exaggeration)
            self.exaggeration: float = float(exaggeration[0])
        else:
            self.exaggeration_range = None
            self.exaggeration = float(exaggeration)

        self.model = None
        self._model_snapshot_dir: str | None = None
        self.reference_wavs_list: list[str] | None = None
        self._reference_layout: str = "wavs"
        self._speaker_audio_map: dict[str, list[str]] = {}

        self.speaker_to_reference: dict[str, str] = {}
        self._speaker_to_original_wav: dict[str, str] = {}
        self.speaker_to_ref_id: dict[str, str] = {}
        self.speaker_to_ref_content_hash: dict[str, str] = {}
        self.conversation_exaggeration: dict[str, float] = {}

        self.temp_dir: str | None = None

        # Saved process-global state for restoration in teardown().
        self._global_state_modified = False
        self._prev_attn_env: str | None = None
        self._llama_cfg_restore: list[tuple[dict, Any]] = []

    def inputs(self) -> tuple[list[str], list[str]]:
        """Required task data keys.

        The transcript is read from ``utterance``; ``text`` is accepted as a
        fallback when ``utterance`` is absent.
        """
        return [], ["conversation_id", "speaker", "utterance"]

    def outputs(self) -> tuple[list[str], list[str]]:
        """Data keys added to each task."""
        return [], ["audio_filepath", "duration", "reference_voice"]

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        """Pre-download Chatterbox weights on the node so workers load from cache."""
        try:
            self._pre_download_model_weights()
        except Exception:  # noqa: BLE001
            logger.warning("Chatterbox model pre-download in setup_on_node failed; will retry in setup().")

    def _pre_download_model_weights(self) -> None:
        """Download Chatterbox checkpoint files from HuggingFace."""
        model_files = _MULTILINGUAL_MODEL_FILES if self.language else _ENGLISH_MODEL_FILES
        self._model_snapshot_dir = snapshot_download(
            repo_id=_CHATTERBOX_REPO_ID,
            repo_type="model",
            revision=_CHATTERBOX_MODEL_REVISION,
            allow_patterns=list(model_files),
            cache_dir=self.cache_dir,
            token=os.getenv("HF_TOKEN"),
        )
        logger.info(f"Pre-downloaded Chatterbox weights from {_CHATTERBOX_REPO_ID}@{_CHATTERBOX_MODEL_REVISION}")

    def setup(self, worker_metadata: object = None) -> None:  # noqa: ARG002
        """Load the TTS model and discover reference audio files."""
        os.makedirs(self.output_audio_dir, exist_ok=True)
        self._init_temp_dir()
        self._load_model()
        self._load_reference_audio_files()

    def teardown(self) -> None:
        """Release model, restore global state, and clean up temp files."""
        self.model = None
        self.speaker_to_reference.clear()
        self._speaker_to_original_wav.clear()
        self.speaker_to_ref_id.clear()
        self.speaker_to_ref_content_hash.clear()
        self.conversation_exaggeration.clear()
        self._restore_global_state()
        self._cleanup_temp_dir()

    def _restore_global_state(self) -> None:
        """Undo the process-global mutations made by the multilingual model load."""
        if not self._global_state_modified:
            return

        if self._prev_attn_env is None:
            os.environ.pop(_ATTN_ENV, None)
        else:
            os.environ[_ATTN_ENV] = self._prev_attn_env

        for cfg, prev in self._llama_cfg_restore:
            if prev is _UNSET:
                cfg.pop("attn_implementation", None)
            else:
                cfg["attn_implementation"] = prev

        self._llama_cfg_restore = []
        self._prev_attn_env = None
        self._global_state_modified = False

    def _init_temp_dir(self) -> None:
        if self.temp_dir is None or not os.path.exists(self.temp_dir):
            self.temp_dir = tempfile.mkdtemp(prefix="chatterbox_ref_")

    def _cleanup_temp_dir(self) -> None:
        if self.temp_dir and os.path.exists(self.temp_dir):
            with contextlib.suppress(OSError):
                shutil.rmtree(self.temp_dir)
            self.temp_dir = None

    def _load_model(self) -> None:
        """Load ChatterboxTTS or ChatterboxMultilingualTTS."""
        # setup_on_node() normally populated this exact snapshot.  Download it
        # here only as a retry if node setup failed; from_local() below never
        # consults the default HF cache or performs a second download.
        if self._model_snapshot_dir is None:
            self._pre_download_model_weights()

        if self.language:
            from chatterbox.models.t3 import llama_configs as _llama_cfgs
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            self._prev_attn_env = os.environ.get(_ATTN_ENV, None)
            self._llama_cfg_restore = [
                (cfg, cfg.get("attn_implementation", _UNSET)) for cfg in _llama_cfgs.LLAMA_CONFIGS.values()
            ]
            self._global_state_modified = True

            os.environ[_ATTN_ENV] = "eager"
            for cfg in _llama_cfgs.LLAMA_CONFIGS.values():
                cfg["attn_implementation"] = "eager"

            self.model = ChatterboxMultilingualTTS.from_local(
                self._model_snapshot_dir,
                device=self.device,
            )
            logger.info(f"Loaded ChatterboxMultilingualTTS (language={self.language})")
        else:
            from chatterbox.tts import ChatterboxTTS

            self.model = ChatterboxTTS.from_local(self._model_snapshot_dir, device=self.device)
            logger.info("Loaded ChatterboxTTS (English)")

    def _load_reference_audio_files(self) -> None:
        """Discover reference audio files in wavs/ or MLS layout.

        Lists are sorted so every actor/node derives an identical ordering
        from the same dataset on disk: ``glob.glob()`` order is filesystem-
        dependent (and can differ across NFS-mounted nodes), and the
        deterministic voice assignment in ``_assign_reference`` relies on
        indexing into these lists identically everywhere.
        """
        wav_pattern = os.path.join(self.reference_voices_dataset, "wavs", "*", "*.wav")
        self.reference_wavs_list = sorted(glob.glob(wav_pattern))

        if self.reference_wavs_list:
            self._reference_layout = "wavs"
            logger.info(f"Found {len(self.reference_wavs_list)} reference files (wavs/ layout)")
            return

        flac_pattern = os.path.join(self.reference_voices_dataset, "*", "*", "*.flac")
        self.reference_wavs_list = sorted(glob.glob(flac_pattern))

        if self.reference_wavs_list:
            self._reference_layout = "mls"
            self._speaker_audio_map = {}
            for fpath in self.reference_wavs_list:
                speaker_id = fpath.split(os.sep)[-3]
                self._speaker_audio_map.setdefault(speaker_id, []).append(fpath)
            for files in self._speaker_audio_map.values():
                files.sort()
            logger.info(
                f"Found {len(self.reference_wavs_list)} reference files "
                f"from {len(self._speaker_audio_map)} speakers (MLS layout)"
            )
            return

        msg = f"No reference audio found in {self.reference_voices_dataset}. Expected wavs/*/*.wav or */*/*.flac"
        raise ValueError(msg)

    def _process_audio_with_rttm(self, audio_filepath: str, rttm_filepath: str) -> str:  # noqa: C901, PLR0912
        """Strip silences using RTTM speech segments, up to max_reference_duration."""
        if not os.path.exists(rttm_filepath):
            return audio_filepath

        try:
            audio, sr = ta.load(audio_filepath)

            speech_segments: list[tuple[float, float]] = []
            audio_duration = audio.shape[1] / sr
            with open(rttm_filepath, encoding="utf-8") as f:
                for line_number, line in enumerate(f, start=1):
                    parts = line.strip().split()
                    _min_rttm_fields = 5
                    if len(parts) >= _min_rttm_fields and parts[0] == "SPEAKER":
                        try:
                            start = float(parts[3])
                            dur = float(parts[4])
                        except ValueError:
                            logger.warning(
                                f"Skipping malformed RTTM record {rttm_filepath}:{line_number}: invalid start/duration"
                            )
                            continue
                        if not (math.isfinite(start) and math.isfinite(dur)) or start < 0 or dur <= 0:
                            logger.warning(
                                f"Skipping malformed RTTM record {rttm_filepath}:{line_number}: start and duration must be finite, start >= 0, duration > 0"
                            )
                            continue
                        if start >= audio_duration:
                            logger.warning(
                                f"Skipping out-of-bounds RTTM record {rttm_filepath}:{line_number}: start {start} is beyond {audio_duration:.3f}s audio"
                            )
                            continue
                        speech_segments.append((start, min(start + dur, audio_duration)))

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
            unique_name = (
                hashlib.md5(audio_filepath.encode(), usedforsecurity=False).hexdigest()[:8]
                + "_"
                + os.path.basename(audio_filepath)
            )
            out_path = os.path.join(self.temp_dir, unique_name)
            ta.save(out_path, processed, sr)
        except (OSError, RuntimeError) as e:
            logger.warning(f"RTTM processing failed for {audio_filepath}: {e}")
            return audio_filepath
        else:
            return out_path

    @staticmethod
    def _stable_digest(key: str) -> bytes:
        """SHA-256 digest of ``key``, used as a source of deterministic pseudo-randomness.

        Unlike Python's ``hash()`` (salted per-process via ``PYTHONHASHSEED``)
        or ``random.Random()`` (per-instance, order-dependent state), this is
        a pure function of ``key``: identical on every actor, process, node,
        and Python version. Voice/exaggeration assignment must be derived
        this way so that the same ``(conversation_id, speaker)`` always
        resolves to the same value regardless of which multi-GPU/multi-node
        worker happens to process a given turn, and regardless of arrival
        order (Ray Data / Xenna fan this stage out across several actor
        copies and work-steal batch-size-one turns across them; each actor
        has independent memory, so any history- or RNG-based assignment can
        diverge across actors -- see stage docstring).
        """
        return hashlib.sha256(key.encode("utf-8")).digest()

    @classmethod
    def _stable_index(cls, key: str, modulus: int) -> int:
        """Deterministic index in ``[0, modulus)`` derived from ``key``."""
        return int.from_bytes(cls._stable_digest(key)[:8], "big") % modulus

    @classmethod
    def _stable_unit_interval(cls, key: str) -> float:
        """Deterministic float in ``[0, 1)`` derived from ``key``."""
        return int.from_bytes(cls._stable_digest(key)[8:16], "big") / 2**64

    def _get_reference_audio_wavs(self, key: str) -> tuple[str, str]:
        """Deterministically select a reference WAV for ``key``, optionally clean with RTTM.

        Returns:
            Tuple of (processed_path, original_path). The original path is
            needed for deduplication since RTTM processing changes the path.
        """
        selected = self.reference_wavs_list[self._stable_index(key, len(self.reference_wavs_list))]

        parts = selected.split(os.sep)
        dialog_id = parts[-2]
        speaker_id = os.path.splitext(parts[-1])[0]
        rttm_path = os.path.join(self.reference_voices_dataset, "rttms", dialog_id, f"{speaker_id}.rttm")
        processed = self._process_audio_with_rttm(selected, rttm_path)
        return processed, selected

    def _get_reference_audio_mls(self, key: str) -> tuple[str, str]:
        """Deterministically select an MLS speaker, concatenate segments as reference."""
        speaker_ids = sorted(self._speaker_audio_map)
        chosen = speaker_ids[self._stable_index(key, len(speaker_ids))]
        # Deterministic "shuffle": order by a stable hash keyed on (chosen, file),
        # not by insertion/glob order, so every actor concatenates segments
        # identically for the same speaker.
        files = sorted(
            self._speaker_audio_map[chosen],
            key=lambda f: self._stable_index(f"{chosen}::{f}", 2**32),
        )

        chunks: list[torch.Tensor] = []
        total_dur = 0.0
        # Rate of the first successfully loaded segment; every later segment is
        # resampled to it before concatenation. MLS segments for one speaker
        # are not guaranteed to share a sample rate, and torch.cat'ing raw
        # tensors recorded at different rates -- then saving under a single
        # header rate -- would silently play the mismatched segments back at
        # the wrong speed/pitch.
        ref_sr: int | None = None

        for fpath in files:
            if total_dur >= self.max_reference_duration:
                break
            try:
                audio, sr = ta.load(fpath)
                if ref_sr is None:
                    ref_sr = sr
                elif sr != ref_sr:
                    audio = ta.functional.resample(audio, orig_freq=sr, new_freq=ref_sr)
                seg_dur = audio.shape[1] / ref_sr
                if total_dur + seg_dur > self.max_reference_duration:
                    remaining = self.max_reference_duration - total_dur
                    audio = audio[:, : int(remaining * ref_sr)]
                chunks.append(audio)
                total_dur += audio.shape[1] / ref_sr
            except (OSError, RuntimeError) as e:
                logger.warning(f"Failed to load {fpath}: {e}")

        if not chunks or ref_sr is None:
            return files[0], chosen

        try:
            concatenated = torch.cat(chunks, dim=1)
            out_path = os.path.join(self.temp_dir, f"ref_{chosen}.wav")
            ta.save(out_path, concatenated, ref_sr)
        except (RuntimeError, OSError) as e:
            logger.warning(f"MLS concatenation failed for speaker {chosen}: {e}")
            return files[0], chosen
        else:
            return out_path, chosen

    def _assign_reference(self, speaker: str, conversation_id: str) -> tuple[str, str]:
        """Get or assign a reference audio file for a speaker in a conversation.

        The assignment is a pure, deterministic function of ``key`` (see
        ``_stable_index``): it does NOT depend on which other speakers/refs
        this actor has already handed out, or on the order turns arrive in.
        This is required for correctness under multi-GPU/multi-node fan-out,
        where Ray Data/Xenna spin up several independent actor copies of this
        one-GPU stage and distribute batch-size-one turns across them --
        each copy has its own memory, so any "avoid reusing a reference
        already taken in this conversation" bookkeeping that only looks at
        *this actor's* history would let the same (conversation_id, speaker)
        land on different actors and receive different voices. The
        per-instance dicts below are pure memoization caches (a speedup for
        repeated calls within one actor), not sources of the decision.

        Trade-off: because assignment no longer coordinates across speakers,
        two different speakers in the same conversation can (rarely) hash to
        the same reference voice. That's a minor cosmetic risk; the critical
        invariant -- the same character always gets the same voice, on any
        actor, any node, any order -- is now guaranteed instead of best-effort.

        Returns:
            Tuple of (ref_path, ref_id) where ref_id is a stable identifier
            for the reference voice (MLS speaker ID or dialog/speaker tag).
        """
        key = f"{conversation_id}::{speaker}"

        if key in self.speaker_to_reference:
            return self.speaker_to_reference[key], self.speaker_to_ref_id[key]

        if self._reference_layout == "mls":
            ref_path, ref_id = self._get_reference_audio_mls(key)
            self.speaker_to_ref_content_hash[key] = self._hash_mls_speaker_content(ref_id)
        else:
            ref_path, original_wav = self._get_reference_audio_wavs(key)
            self._speaker_to_original_wav[key] = original_wav
            parts = original_wav.split(os.sep)
            ref_id = f"{parts[-2]}/{os.path.splitext(parts[-1])[0]}"
            self.speaker_to_ref_content_hash[key] = self._hash_wavs_reference_content(original_wav)

        self.speaker_to_reference[key] = ref_path
        self.speaker_to_ref_id[key] = ref_id
        return ref_path, ref_id

    def _reference_content_hash(self, speaker: str, conversation_id: str) -> str:
        """Content-identity hash for the reference voice assigned to this key.

        Hashed from the *source* reference file(s) on disk (not the
        per-actor RTTM-cleaned/concatenated temp copy), so replacing the
        underlying reference audio invalidates the cache even though
        ``reference_id`` (a path-derived label) stays the same, and so the
        hash is identical across actors regardless of each actor's own temp
        directory. Assumes ``_assign_reference`` has already been called for
        this key (as ``process_batch`` always does); falls back to an empty
        string otherwise rather than raising.
        """
        return self.speaker_to_ref_content_hash.get(f"{conversation_id}::{speaker}", "")

    @staticmethod
    def _hash_file_content(path: str) -> str:
        """SHA-256 of a file's bytes, used as a reference-content identity check."""
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _hash_wavs_reference_content(self, original_wav: str) -> str:
        """Identity of WAV bytes plus the RTTM-driven preprocessing input."""
        dialog_id = os.path.basename(os.path.dirname(original_wav))
        speaker_id = os.path.splitext(os.path.basename(original_wav))[0]
        rttm_path = os.path.join(self.reference_voices_dataset, "rttms", dialog_id, f"{speaker_id}.rttm")
        identity = {
            "preprocessing_schema_version": _REFERENCE_PREPROCESSING_SCHEMA_VERSION,
            "wav_sha256": self._hash_file_content(original_wav),
            "rttm_present": os.path.isfile(rttm_path),
            "rttm_sha256": self._hash_file_content(rttm_path) if os.path.isfile(rttm_path) else None,
        }
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _hash_mls_speaker_content(self, speaker_id: str) -> str:
        """Content identity for all reference clips belonging to an MLS speaker.

        Hashes every file for the speaker (not just the subset that happens
        to fit under ``max_reference_duration``), so any edit to that
        speaker's reference pool invalidates the cache even if the specific
        files used for a given turn didn't change.
        """
        file_hashes = [self._hash_file_content(f) for f in self._speaker_audio_map[speaker_id]]
        return hashlib.sha256("".join(file_hashes).encode("utf-8")).hexdigest()

    def _get_exaggeration(self, conversation_id: str) -> float:
        """Get consistent exaggeration for a conversation (random range support).

        Deterministic (see ``_stable_index``/``_assign_reference`` docstring):
        every actor derives the same value for a given ``conversation_id``.
        """
        if self.exaggeration_range is None:
            return self.exaggeration

        if conversation_id not in self.conversation_exaggeration:
            lo, hi = self.exaggeration_range
            frac = self._stable_unit_interval(f"exaggeration::{conversation_id}")
            self.conversation_exaggeration[conversation_id] = lo + frac * (hi - lo)
        return self.conversation_exaggeration[conversation_id]

    def _generate_turn_audio(self, text: str, reference_wav: str, conversation_id: str) -> np.ndarray:
        """Run ChatterboxTTS inference for a single turn.

        The model always synthesises at its own native rate (``self.model.sr``,
        typically 24 kHz), independent of ``self.sample_rate``. If the two
        differ, the raw samples are resampled to ``self.sample_rate`` here --
        otherwise the caller would write unchanged samples into a WAV header
        claiming a different rate, changing playback speed/pitch and
        corrupting any duration computed from the configured rate.
        """
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

                native_sr = getattr(self.model, "sr", _CHATTERBOX_NATIVE_SR)
                if native_sr != self.sample_rate:
                    wav = ta.functional.resample(wav, orig_freq=native_sr, new_freq=self.sample_rate)

            if self.normalize_audio:
                wav = self._normalize_audio(wav)

            return wav.squeeze(0).cpu().numpy()
        except Exception as e:  # noqa: BLE001 -- graceful fallback: any failure yields silence
            logger.error(f"TTS generation failed: {e}")
            return np.zeros(self.sample_rate * 2)

    def _normalize_audio(self, wav: torch.Tensor) -> torch.Tensor:
        """RMS-based normalisation with clipping protection."""
        _silence_threshold = 1e-10
        rms = torch.sqrt(torch.mean(wav**2))
        if rms < _silence_threshold:
            return wav
        current_db = 20 * torch.log10(rms + 1e-8)
        gain = 10 ** ((self.normalize_level - current_db) / 20)
        normalised = wav * gain
        peak = torch.max(torch.abs(normalised))
        if peak > 1.0:
            normalised = normalised / peak * 0.99
        return normalised

    def _cache_manifest(  # noqa: PLR0913
        self,
        *,
        conversation_id: str,
        speaker: str,
        text: str,
        ref_id: str,
        ref_content_hash: str,
        exaggeration: float,
    ) -> dict[str, Any]:
        """Canonical manifest of every effective generation input.

        Used both to derive the cache filename hash and as the sidecar
        written next to each cached WAV so a filename-hash cache hit can be
        independently validated before being trusted (see
        ``_read_cached_audio_if_valid``). Must include every parameter that
        can change the resulting audio -- a knob left out here can make a
        real config change (e.g. ``language``) silently return stale cached
        audio from a previous, different configuration.
        """
        return {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "model_repo_id": _CHATTERBOX_REPO_ID,
            "model_revision": _CHATTERBOX_MODEL_REVISION,
            "model_class": "ChatterboxMultilingualTTS" if self.language else "ChatterboxTTS",
            "language": self.language,
            "conversation_id": conversation_id,
            "speaker": speaker,
            "text": text,
            "reference_id": ref_id,
            "reference_content_hash": ref_content_hash,
            "max_reference_duration": self.max_reference_duration,
            "cfg_weight": self.cfg_weight,
            "exaggeration": exaggeration,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
            "min_p": self.min_p,
            "top_p": self.top_p,
            "normalize_audio": self.normalize_audio,
            "normalize_level": self.normalize_level,
            "sample_rate": self.sample_rate,
        }

    @staticmethod
    def _hash_manifest(manifest: dict[str, Any]) -> str:
        """Stable hash of a cache manifest (key order doesn't affect the result)."""
        canonical = json.dumps(manifest, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def _output_filename(cls, manifest: dict[str, Any]) -> str:
        """Deterministic filename derived from the full generation manifest.

        ``conv_hash``/``text_hash`` keep filenames grep-able by conversation
        and roughly stable for the same text; ``config_hash`` covers every
        other manifest field (voice, language, sampling params, ...) so any
        change to effective generation inputs produces a new filename
        instead of colliding with a previous, differently-configured run.
        """
        conv_hash = hashlib.md5(manifest["conversation_id"].encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
        text_hash = hashlib.md5(manifest["text"].encode("utf-8"), usedforsecurity=False).hexdigest()[:10]
        config_hash = cls._hash_manifest(manifest)[:16]
        return f"{conv_hash}_{manifest['speaker']}_{text_hash}_{config_hash}.wav"

    @staticmethod
    def _sidecar_path(audio_path: str) -> str:
        return f"{os.path.splitext(audio_path)[0]}.json"

    def _read_cached_audio_if_valid(self, audio_path: str, manifest: dict[str, Any]) -> tuple[np.ndarray, int] | None:
        """Return cached audio only if a matching sidecar confirms it's valid.

        A filename-hash hit is NOT trusted on its own: it's cross-checked
        against a sidecar JSON containing the exact manifest used to
        generate that file. Missing, corrupt, or mismatched sidecars
        (including legacy cache entries written before this sidecar existed)
        are treated as a cache miss and trigger regeneration, rather than
        risking a hash collision or a stale entry being served silently.
        """
        if not os.path.exists(audio_path):
            return None

        sidecar_path = self._sidecar_path(audio_path)
        try:
            with open(sidecar_path, encoding="utf-8") as f:
                cached_manifest = json.load(f)
        except (OSError, json.JSONDecodeError):
            logger.warning(f"No valid cache sidecar for {audio_path}; regenerating.")
            return None

        if cached_manifest != manifest:
            logger.warning(f"Cache sidecar for {audio_path} doesn't match current config; regenerating.")
            return None

        try:
            # Use the WAV's own embedded rate for downstream duration math,
            # not ``self.sample_rate`` -- they should always agree since
            # ``sample_rate`` is part of the manifest above, but trusting the
            # file's actual rate is a cheap, direct guard against ever
            # mis-reporting duration for an existing cache entry.
            audio_data, file_sr = sf.read(audio_path)
        except (OSError, RuntimeError) as e:
            logger.warning(f"Failed to read cached audio {audio_path}: {e}; regenerating.")
            return None
        return audio_data, file_sr

    def _publish_cache_entry(self, audio_path: str, audio_data: np.ndarray, manifest: dict[str, Any]) -> None:
        """Write the WAV and its sidecar atomically (temp file + rename).

        Ensures no process ever observes a partially written cache entry
        (e.g. a truncated WAV from a crash mid-write) as a valid cache hit.
        The sidecar is published before the WAV, so by the time
        ``os.path.exists(audio_path)`` becomes true for another
        reader/retry, its sidecar is already there to validate against.
        """
        unique = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
        tmp_audio = f"{audio_path}.tmp-{unique}"
        sidecar_path = self._sidecar_path(audio_path)
        tmp_sidecar = f"{sidecar_path}.tmp-{unique}"
        try:
            # format="WAV" explicitly: the temp filename's ".tmp-<suffix>"
            # ending would otherwise defeat soundfile's extension sniffing.
            sf.write(tmp_audio, audio_data, self.sample_rate, format="WAV")
            with open(tmp_sidecar, "w", encoding="utf-8") as f:
                json.dump(manifest, f, sort_keys=True)
            os.replace(tmp_sidecar, sidecar_path)
            os.replace(tmp_audio, audio_path)
        finally:
            for tmp in (tmp_audio, tmp_sidecar):
                if os.path.exists(tmp):
                    with contextlib.suppress(OSError):
                        os.remove(tmp)

    def process(self, task: AudioTask) -> AudioTask:
        """Not supported; use ``process_batch()`` for TTS inference."""
        msg = f"[{self.name}] is a GPU/batched inference stage. Use process_batch() instead."
        raise NotImplementedError(msg)

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

            try:
                # Reference selection includes file I/O and RTTM preprocessing,
                # so it belongs inside the per-task boundary as much as cache I/O.
                reference_wav, ref_id = self._assign_reference(speaker, conversation_id)
                ref_content_hash = self._reference_content_hash(speaker, conversation_id)
                exaggeration = self._get_exaggeration(conversation_id)

                manifest = self._cache_manifest(
                    conversation_id=conversation_id,
                    speaker=speaker,
                    text=text,
                    ref_id=ref_id,
                    ref_content_hash=ref_content_hash,
                    exaggeration=exaggeration,
                )
                filename = self._output_filename(manifest)
                audio_path = os.path.join(self.output_audio_dir, filename)
                cached = self._read_cached_audio_if_valid(audio_path, manifest)
                if cached is not None:
                    audio_data, audio_sr = cached
                else:
                    audio_data = self._generate_turn_audio(text, reference_wav, conversation_id)
                    audio_sr = self.sample_rate
                    self._publish_cache_entry(audio_path, audio_data, manifest)
            except Exception as e:  # noqa: BLE001 -- isolate a bad reference/record to one task
                logger.error(f"TTS processing failed for task {task.task_id}: {e}")
                output_tasks.append(task)
                continue

            duration = len(audio_data) / audio_sr

            task.data["audio_filepath"] = audio_path
            task.data["duration"] = duration
            task.data["reference_voice"] = ref_id
            output_tasks.append(task)

            logger.info(f"[TTS] {conversation_id[:8]}/{speaker}: {duration:.2f}s -> {filename}")

        return output_tasks
