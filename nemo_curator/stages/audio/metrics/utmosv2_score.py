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

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioBatch, DocumentBatch

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata


@dataclass
class GetUtmosv2ScoreStage(ProcessingStage[DocumentBatch | AudioBatch, AudioBatch]):
    """Compute UTMOSv2 Mean Opinion Score (MOS) for audio files.

    Delegates entirely to the public ``model.predict(input_dir=...)`` API
    from `UTMOSv2 <https://github.com/sarulab-speech/UTMOSv2>`_.
    Internally UTMOSv2's Dataset tiles short audio (``np.tile``) and
    random-crops each sample to a fixed length, so every tensor in a
    DataLoader batch is the same size with **no zero-padding**.

    The model is loaded once per worker in ``setup()`` and reused across
    tasks.  Multiple ``num_repetitions`` average out the random cropping to
    produce stable scores (test-time augmentation).

    Accepts both ``DocumentBatch`` (e.g. from ``JsonlReader``) and
    ``AudioBatch`` as input.

    Args:
        audio_filepath_key: Key in each data entry pointing to the WAV path.
        audio_root: Optional root directory prepended to relative
            ``audio_filepath`` values.  Leave empty when paths are absolute.
        score_key: Key where the predicted MOS will be stored.
        inference_batch_size: Batch size passed to ``model.predict()``.
        num_repetitions: Number of random-crop repetitions to average
            (test-time augmentation).  Higher values produce more stable
            scores at the cost of proportionally more compute.
        predict_dataset: UTMOSv2 data-domain ID used for prediction.
            Valid values: ``"sarulab"`` (default), ``"bvcc"``, ``"somos"``,
            ``"blizzard2008"`` through ``"blizzard2011"``.
        name: Stage identifier.
        resources: Compute resources.  Defaults to 1 GPU.
    """

    audio_filepath_key: str = "audio_filepath"
    audio_root: str = ""
    score_key: str = "utmosv2_score"
    inference_batch_size: int = 16
    num_repetitions: int = 1
    predict_dataset: str = "sarulab"
    name: str = "GetUtmosv2ScoreStage"
    resources: Resources = field(default_factory=lambda: Resources(gpus=1.0))

    _model: Any = field(default=None, init=False, repr=False)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        import utmosv2

        self._model = utmosv2.create_model(pretrained=True)

    def _score_files(self, file_paths: list[str]) -> list[float]:
        """Score audio files via ``model.predict(input_dir=...)``.

        Creates a temporary directory of symlinks so that files from
        arbitrary locations can be fed to UTMOSv2's ``input_dir`` API,
        which builds a proper Dataset + DataLoader (tile per sample,
        random-crop to fixed length, batch via collation — no padding).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, fp in enumerate(file_paths):
                os.symlink(os.path.abspath(fp), os.path.join(tmpdir, f"{i:08d}.wav"))

            results = self._model.predict(
                input_dir=tmpdir,
                batch_size=self.inference_batch_size,
                num_repetitions=self.num_repetitions,
                predict_dataset=self.predict_dataset,
                num_workers=0,
                verbose=False,
            )

        return [r["predicted_mos"] for r in results]

    def process(self, task: DocumentBatch | AudioBatch) -> AudioBatch:
        if isinstance(task, DocumentBatch):
            entries = task.data.to_dict("records")
        elif isinstance(task, AudioBatch):
            entries = list(task.data)
        else:
            raise TypeError(f"Unsupported task type: {type(task)}")

        valid_paths: list[str] = []
        valid_indices: list[int] = []
        root = Path(self.audio_root) if self.audio_root else None

        for i, entry in enumerate(entries):
            fp = entry.get(self.audio_filepath_key)
            if not fp:
                logger.warning(f"Missing audio_filepath key in entry {i}")
                entry[self.score_key] = float("nan")
                continue
            resolved = Path(fp) if root is None or Path(fp).is_absolute() else root / fp
            if resolved.exists():
                valid_paths.append(str(resolved))
                valid_indices.append(i)
            else:
                logger.warning(f"Audio file not found: {resolved}")
                entry[self.score_key] = float("nan")

        if valid_paths:
            scores = self._score_files(valid_paths)
            for idx, score in zip(valid_indices, scores):
                entries[idx][self.score_key] = round(float(score), 4)

        return AudioBatch(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            filepath_key=self.audio_filepath_key,
            data=entries,
            _stage_perf=task._stage_perf,
        )
