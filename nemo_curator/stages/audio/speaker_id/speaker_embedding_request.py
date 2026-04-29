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

"""Extract speaker embeddings from a JSONL manifest with audio file paths.

Counterpart of :mod:`~nemo_curator.stages.audio.request.prepare_omni_request`
which builds chat messages for an LLM.  This stage instead runs a NeMo
``EncDecSpeakerLabelModel`` locally on each audio file referenced
in the JSONL rows, and persists the resulting speaker-embedding vectors.
"""

from __future__ import annotations

import io
import os
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import soundfile as sf
import torch
from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch


def _is_remote_storage_path(value: str) -> bool:
    if not value or not isinstance(value, str):
        return False
    s = value.strip()
    return s.startswith(("s3://", "ais://"))


@dataclass
class SpeakerEmbeddingRequestStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Read audio file paths from a JSONL-backed :class:`DocumentBatch`, run
    a speaker model, and add an ``embedding`` column with speaker-embedding vectors.

    Optionally saves all embeddings to a single ``.npz`` / ``.pt`` file.

    This is the speaker-embedding counterpart of
    :class:`~nemo_curator.stages.audio.request.prepare_omni_request.PrepareOmniRequestStage`,
    which builds OpenAI chat messages instead.

    Supports the same audio sources as the Omni stage:

    * **local files** -- ``audio_filepath`` points to a file on disk.
    * **tar members** -- when ``input_tar`` is set, ``audio_filepath`` values
      are member names inside the tar archive.
    * **S3 / AIS** -- ``audio_filepath`` starts with ``s3://`` or ``ais://``.

    Args:
        model_name: Pretrained NeMo speaker model name.
        audio_filepath_key: Column name for audio file paths in the JSONL rows.
        input_tar: Optional tar archive containing audio referenced by rows.
        target_sample_rate: Resample audio to this rate (default model expects 16 kHz).
        output_path: If set, save all embeddings to this file after processing.
        output_format: ``"npz"`` or ``"pt"`` (only used when ``output_path`` is set).
        s3cfg: Path to AIS/S3 config for remote storage access.
    """

    name: str = "SpeakerEmbeddingRequestStage"
    model_name: str = "nvidia/speakerverification_en_titanet_large"
    cache_dir: str | None = None
    speaker_model: Any | None = field(default=None, repr=False)
    audio_filepath_key: str = "audio_filepath"
    input_tar: str = ""
    target_sample_rate: int = 16000
    output_path: str = ""
    output_format: Literal["npz", "pt"] = "npz"
    s3cfg: str = ""
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpu_memory_gb=4.0))
    batch_size: int = 1

    _tar_cache: dict[str, bytes] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.model_name and self.speaker_model is None:
            msg = "Either model_name or speaker_model is required."
            raise ValueError(msg)

    # ------------------------------------------------------------------
    # Model loading (same pattern as InferenceAsrNemoStage)
    # ------------------------------------------------------------------

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        if self.speaker_model is not None:
            return
        import nemo.collections.asr as nemo_asr

        try:
            kwargs: dict[str, Any] = {"model_name": self.model_name, "return_model_file": True}
            if self.cache_dir is not None:
                kwargs["cache_dir"] = self.cache_dir
            nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(**kwargs)
        except Exception:
            logger.info(f"Could not pre-cache {self.model_name}; will download on first use")

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if self.speaker_model is not None:
            self.speaker_model.eval()
            return
        import nemo.collections.asr as nemo_asr

        map_location = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs: dict[str, Any] = {"model_name": self.model_name, "map_location": map_location}
        if self.cache_dir is not None:
            kwargs["cache_dir"] = self.cache_dir
        self.speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(**kwargs)
        self.speaker_model.eval()

        if self.s3cfg:
            from nemo_curator.stages.audio.request.prepare_omni_request import _init_ais_client

            _init_ais_client(self.s3cfg)

    # ------------------------------------------------------------------
    # Audio loading (local / tar / S3)
    # ------------------------------------------------------------------

    def _load_tar_cache(self) -> dict[str, bytes]:
        if self._tar_cache is not None:
            return self._tar_cache
        tar_path = Path(self.input_tar.strip()).expanduser().resolve()
        if not tar_path.is_file():
            msg = f"input_tar is not a file: {tar_path}"
            raise FileNotFoundError(msg)
        self._tar_cache = {}
        with tarfile.open(tar_path, "r:*") as tf:
            for member in tf.getmembers():
                if member.isfile():
                    stream = tf.extractfile(member)
                    if stream is not None:
                        self._tar_cache[member.name] = stream.read()
                        basename = Path(member.name).name
                        if basename not in self._tar_cache:
                            self._tar_cache[basename] = self._tar_cache[member.name]
        return self._tar_cache

    def _load_audio_bytes(self, filepath: str) -> np.ndarray:
        """Load audio from any source and return a mono float32 array at target_sample_rate."""
        filepath = filepath.strip()

        if self.input_tar.strip():
            cache = self._load_tar_cache()
            key = filepath
            if key not in cache:
                key = Path(filepath).name
            if key not in cache:
                msg = f"Audio member {filepath!r} not found in tar"
                raise FileNotFoundError(msg)
            audio, sr = sf.read(io.BytesIO(cache[key]), dtype="float32")
        elif _is_remote_storage_path(filepath):
            from nemo_curator.stages.audio.request.prepare_omni_request import _read_remote_bytes

            raw = _read_remote_bytes(filepath)
            audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
        else:
            path = Path(filepath).expanduser().resolve()
            if not path.is_file():
                msg = f"Audio file not found: {filepath}"
                raise FileNotFoundError(msg)
            audio, sr = sf.read(str(path), dtype="float32")

        if audio.ndim > 1:
            audio = audio[:, 0]

        if sr != self.target_sample_rate:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sample_rate)

        return audio.astype(np.float32)

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        device = self.speaker_model.device
        signal = torch.tensor(audio[np.newaxis, :], device=device, dtype=torch.float32)
        signal_len = torch.tensor([audio.shape[0]], device=device)
        _, emb = self.speaker_model.forward(input_signal=signal, input_signal_length=signal_len)
        return emb.squeeze().cpu().numpy()

    def _save_embeddings(self, cut_ids: list[str], embeddings: np.ndarray) -> None:
        if not self.output_path:
            return
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        if self.output_format == "pt":
            torch.save(
                {"cut_ids": cut_ids, "embeddings": torch.from_numpy(embeddings)},
                self.output_path,
            )
        else:
            np.savez(
                self.output_path,
                cut_ids=np.array(cut_ids, dtype=object),
                embeddings=embeddings,
            )
        logger.info(f"Saved {len(cut_ids)} embeddings to {self.output_path}")

    # ------------------------------------------------------------------
    # Stage entry point
    # ------------------------------------------------------------------

    def process(self, input_batch: DocumentBatch) -> DocumentBatch:
        df = input_batch.to_pandas().copy()
        embeddings_list: list[np.ndarray | None] = []

        # Optional in-memory waveforms travelling alongside rows.  The
        # JsonlReader → DocumentBatch path strips them; the AIS-streamed
        # path keeps them in task metadata.  Check the column first.
        has_waveform_col = "waveform" in df.columns
        has_sr_col = "sample_rate" in df.columns

        for idx, row in df.iterrows():
            filepath = row.get(self.audio_filepath_key, "")
            wav = row.get("waveform") if has_waveform_col else None
            try:
                if wav is not None and not (isinstance(wav, float) and np.isnan(wav)):
                    sr = int(row["sample_rate"]) if has_sr_col else self.target_sample_rate
                    audio = np.asarray(wav, dtype=np.float32)
                    if audio.ndim > 1:
                        audio = audio[:, 0] if audio.shape[1] < audio.shape[0] else audio[0]
                    if sr != self.target_sample_rate:
                        import librosa

                        audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sample_rate)
                    audio = audio.astype(np.float32)
                else:
                    if not filepath or (isinstance(filepath, float) and np.isnan(filepath)):
                        logger.warning(f"Row {idx}: missing waveform and {self.audio_filepath_key}, skipping")
                        embeddings_list.append(None)
                        continue
                    audio = self._load_audio_bytes(str(filepath))
                emb = self._extract_embedding(audio)
                embeddings_list.append(emb)
            except Exception as exc:
                logger.warning(f"Row {idx}: failed to extract embedding for {filepath!r}: {exc}")
                embeddings_list.append(None)

        df["embedding"] = embeddings_list

        if self.output_path:
            valid_mask = [e is not None for e in embeddings_list]
            valid_ids = df.index[valid_mask].tolist()
            valid_filepaths = [str(df.at[i, self.audio_filepath_key]) for i in valid_ids]
            valid_embs = [e for e in embeddings_list if e is not None]
            if valid_embs:
                self._save_embeddings(valid_filepaths, np.stack(valid_embs))

        return DocumentBatch(
            data=df,
            dataset_name=input_batch.dataset_name,
            task_id=input_batch.task_id,
        )
