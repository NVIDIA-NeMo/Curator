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

"""UTMOSv2 Mean Opinion Score (MOS) scoring stage.

Supports three audio input modes:

1. **In-memory waveform** — when the entry contains a ``waveform`` key
   (numpy array) and ``sample_rate`` (int), as produced by
   ``NemoTarShardReaderStage``.  The waveform is resampled to
   ``self.sample_rate`` and written as a temp WAV for UTMOSv2.  No file
   downloads or disk I/O beyond the temp dir.  This is the preferred mode
   for large-scale NeMo tarred datasets streamed from AIS/S3.

2. **Local or remote file path** — when the entry has an
   ``audio_filepath`` pointing to a local file or a remote URL
   (``s3://``, ``ais://``, ``http(s)://``).  Remote files are downloaded;
   non-WAV formats are converted via ffmpeg.

3. **Mixed** — a ``DocumentBatch`` may contain entries with waveforms and
   entries with file paths; each is handled appropriately.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import soundfile as sf
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask, DocumentBatch

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata

_REMOTE_PREFIXES = ("s3://", "ais://", "gs://", "http://", "https://")
_NON_WAV_EXTENSIONS = {".opus", ".m4a", ".flac", ".mp3", ".ogg", ".webm", ".aac"}


def _is_remote(path: str) -> bool:
    return any(path.startswith(p) for p in _REMOTE_PREFIXES)


def _download_remote(remote_path: str, dest_dir: str) -> str | None:
    """Download a remote audio file to *dest_dir*.

    Supports ``s3://`` / ``ais://`` via the ``ais`` CLI (falls back to
    ``aws s3 cp``) and ``http(s)://`` via ``curl``.
    """
    basename = remote_path.rstrip("/").rsplit("/", 1)[-1]
    local_path = os.path.join(dest_dir, basename)

    if remote_path.startswith(("s3://", "ais://")):
        ais_bin = shutil.which("ais") or shutil.which("ais-iad")
        if ais_bin:
            ret = subprocess.run(
                [ais_bin, "object", "get", remote_path, local_path],
                capture_output=True, timeout=300,
            )
            if ret.returncode == 0 and os.path.exists(local_path):
                return local_path
            logger.warning(f"ais get failed for {remote_path}: {ret.stderr.decode()[:200]}")

        if remote_path.startswith("s3://"):
            ret = subprocess.run(
                ["aws", "s3", "cp", remote_path, local_path, "--quiet"],
                capture_output=True, timeout=300,
            )
            if ret.returncode == 0 and os.path.exists(local_path):
                return local_path
            logger.warning(f"aws s3 cp failed for {remote_path}: {ret.stderr.decode()[:200]}")

    elif remote_path.startswith(("http://", "https://")):
        ret = subprocess.run(
            ["curl", "-sfL", "-o", local_path, remote_path],
            capture_output=True, timeout=300,
        )
        if ret.returncode == 0 and os.path.exists(local_path):
            return local_path
        logger.warning(f"curl failed for {remote_path}")

    return None


def _convert_to_wav(src: str, dest_wav: str, sample_rate: int = 16000) -> bool:
    """Convert any ffmpeg-supported audio format to 16-bit PCM WAV."""
    ret = subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-ar", str(sample_rate), "-ac", "1",
         "-sample_fmt", "s16", dest_wav],
        capture_output=True, timeout=120,
    )
    return ret.returncode == 0 and os.path.exists(dest_wav)


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio array. Uses librosa if available, else linear interp."""
    if orig_sr == target_sr:
        return audio
    try:
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        ratio = target_sr / orig_sr
        indices = np.round(np.arange(0, len(audio), 1 / ratio)).astype(int)
        indices = indices[indices < len(audio)]
        return audio[indices]


@dataclass
class GetUtmosv2ScoreStage(ProcessingStage[DocumentBatch | AudioTask, DocumentBatch | AudioTask]):
    """Compute UTMOSv2 Mean Opinion Score (MOS) for audio.

    Accepts ``DocumentBatch`` (from ``JsonlReader``) or ``AudioTask``
    (from ``NemoTarShardReaderStage``).

    Audio source priority per entry:

    1. In-memory ``waveform`` + ``sample_rate`` keys (NeMo tar streaming).
    2. ``audio_filepath`` pointing to a local or remote file.

    Args:
        audio_filepath_key: Key pointing to the audio path (mode 2).
        waveform_key: Key for in-memory waveform numpy array (mode 1).
        sample_rate_key: Key for the waveform's sample rate (mode 1).
        audio_root: Optional root prepended to relative file paths.
        score_key: Output key for the predicted MOS score.
        inference_batch_size: Batch size for ``model.predict()``.
        num_repetitions: Random-crop repetitions for test-time augmentation.
        sample_rate: Target sample rate (default 16000).
        predict_dataset: UTMOSv2 data-domain ID.
        name: Stage identifier.
        resources: Compute resources (default 1 GPU).
    """

    audio_filepath_key: str = "audio_filepath"
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"
    audio_root: str = ""
    score_key: str = "utmosv2_score"
    inference_batch_size: int = 16
    num_repetitions: int = 1
    sample_rate: int = 16000
    predict_dataset: str = "sarulab"
    name: str = "GetUtmosv2ScoreStage"
    resources: Resources = field(default_factory=lambda: Resources(gpus=1.0))

    _model: Any = field(default=None, init=False, repr=False)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.score_key]

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        import utmosv2
        self._model = utmosv2.create_model(pretrained=True)

    # ------------------------------------------------------------------
    # WAV preparation: waveform or file path → temp WAV file
    # ------------------------------------------------------------------

    def _waveform_to_wav(
        self, waveform: np.ndarray, orig_sr: int, wav_path: str,
    ) -> bool:
        """Write an in-memory waveform to a WAV file at ``self.sample_rate``."""
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        waveform = _resample(waveform.astype(np.float32), orig_sr, self.sample_rate)
        try:
            sf.write(wav_path, waveform, self.sample_rate, subtype="PCM_16")
            return True
        except Exception as e:
            logger.warning(f"Failed to write WAV: {e}")
            return False

    def _filepath_to_wav(
        self, audio_path: str, wav_path: str,
    ) -> bool:
        """Resolve a file path (local/remote, any format) to a WAV file."""
        # Remote: download first
        if _is_remote(audio_path):
            dl_dir = os.path.dirname(wav_path)
            local_file = _download_remote(audio_path, dl_dir)
            if local_file is None:
                logger.warning(f"Failed to download: {audio_path}")
                return False
            audio_path = local_file

        ext = Path(audio_path).suffix.lower()
        if ext in _NON_WAV_EXTENSIONS:
            return _convert_to_wav(audio_path, wav_path, self.sample_rate)

        if os.path.exists(audio_path):
            os.symlink(os.path.abspath(audio_path), wav_path)
            return True

        logger.warning(f"Audio file not found: {audio_path}")
        return False

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_dir(self, wav_dir: str) -> list[float]:
        """Score all WAV files in a directory."""
        results = self._model.predict(
            input_dir=wav_dir,
            batch_size=self.inference_batch_size,
            num_repetitions=self.num_repetitions,
            predict_dataset=self.predict_dataset,
            num_workers=0,
            verbose=False,
        )
        return [r["predicted_mos"] for r in results]

    # ------------------------------------------------------------------
    # Main process
    # ------------------------------------------------------------------

    def process(self, task: DocumentBatch | AudioTask) -> DocumentBatch | AudioTask:
        if isinstance(task, DocumentBatch):
            entries = task.data.to_dict("records")
        elif isinstance(task, AudioTask):
            entries = [task.data]
        else:
            raise TypeError(f"Unsupported task type: {type(task)}")

        root = Path(self.audio_root) if self.audio_root else None

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_dir = os.path.join(tmpdir, "wavs")
            os.makedirs(wav_dir)

            valid_indices: list[int] = []
            wav_count = 0

            for i, entry in enumerate(entries):
                wav_path = os.path.join(wav_dir, f"{wav_count:08d}.wav")
                ok = False

                # Mode 1: in-memory waveform (from NemoTarShardReaderStage)
                waveform = entry.get(self.waveform_key)
                if waveform is not None and isinstance(waveform, np.ndarray):
                    orig_sr = int(entry.get(self.sample_rate_key, self.sample_rate))
                    ok = self._waveform_to_wav(waveform, orig_sr, wav_path)
                else:
                    # Mode 2: file path (local or remote)
                    fp = entry.get(self.audio_filepath_key)
                    if fp:
                        if _is_remote(fp):
                            audio_path = fp
                        elif root is not None and not Path(fp).is_absolute():
                            audio_path = str(root / fp)
                        else:
                            audio_path = fp
                        ok = self._filepath_to_wav(audio_path, wav_path)

                if ok:
                    valid_indices.append(i)
                    wav_count += 1
                else:
                    if waveform is None and not entry.get(self.audio_filepath_key):
                        logger.warning(
                            f"Entry {i}: no '{self.waveform_key}' or "
                            f"'{self.audio_filepath_key}'"
                        )
                    entry[self.score_key] = float("nan")

            if valid_indices:
                scores = self._score_dir(wav_dir)
                for idx, score in zip(valid_indices, scores):
                    entries[idx][self.score_key] = round(float(score), 4)

        # Strip waveform from output (large, not serializable to JSONL)
        for entry in entries:
            entry.pop(self.waveform_key, None)

        if isinstance(task, AudioTask):
            task.data.update(entries[0])
            return task

        return DocumentBatch(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=pd.DataFrame(entries),
            _stage_perf=task._stage_perf,
        )
