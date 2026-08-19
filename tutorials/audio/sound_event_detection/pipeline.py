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

"""End-to-end Sound Event Detection (SED) labeling pipeline.

Chains the two SED stages added in this branch into a single :class:`Pipeline`:

1. ``_LoadWaveformStage`` -- read each ``audio_filepath`` into an in-memory
   mono waveform (SoundFile, channel-last layout as ``SEDInferenceStage`` expects).
2. ``SEDInferenceStage`` -- run the AudioSet-pretrained CNN14 model, producing a
   per-frame ``(T, 527)`` probability matrix in task data.
3. ``SEDPostprocessingStage`` -- convert framewise probabilities into labeled
   ``sed_events`` (speech / music / noise / ... spans).
4. ``_KeepEventsStage`` -- drop the large transient arrays so the result is
   JSON-serializable, then write one line per utterance with its ``sed_events``.

The CNN14 checkpoint is NOT downloaded automatically -- pass ``--checkpoint``
pointing at a PANNs ``Cnn14_DecisionLevelMax`` ``.pth``.

Example::

    python pipeline.py \\
        --input-manifest /data/manifest.jsonl \\
        --checkpoint     /models/Cnn14_DecisionLevelMax_mAP=0.385.pth \\
        --output-dir     /data/sed_out \\
        --threshold      0.5
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.common import ManifestReader
from nemo_curator.stages.audio.inference import SEDInferenceStage
from nemo_curator.stages.audio.io.convert import AudioToDocumentStage
from nemo_curator.stages.audio.postprocessing import SEDPostprocessingStage
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata


@dataclass
class _LoadWaveformStage(ProcessingStage[AudioTask, AudioTask]):
    """Read ``audio_filepath`` into an in-memory mono waveform + sample rate.

    SoundFile returns channel-last data (``(samples,)`` mono or ``(samples, ch)``),
    which is exactly the layout ``SEDInferenceStage`` expects to mono-mix.
    """

    audio_filepath_key: str = "audio_filepath"
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"

    name: str = "LoadWaveform"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        import soundfile

        self._soundfile = soundfile

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.audio_filepath_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.waveform_key, self.sample_rate_key]

    def process(self, task: AudioTask) -> AudioTask | None:
        path = task.data.get(self.audio_filepath_key)
        if not path:
            logger.warning(f"Task {task.task_id} has no {self.audio_filepath_key!r}; dropping")
            return None
        try:
            waveform, sample_rate = self._soundfile.read(path, dtype="float32")
        except (RuntimeError, OSError) as exc:
            logger.warning(f"Failed to read {path}: {exc}; dropping")
            return None
        task.data[self.waveform_key] = waveform
        task.data[self.sample_rate_key] = sample_rate
        return task


@dataclass
class _KeepEventsStage(ProcessingStage[AudioTask, AudioTask]):
    """Drop the transient waveform / framewise arrays so the task is serializable."""

    drop_keys: tuple[str, ...] = ("waveform", "_sed_framewise", "sed_fps", "sed_valid_frames")

    name: str = "KeepEvents"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: AudioTask) -> AudioTask:
        for key in self.drop_keys:
            task.data.pop(key, None)
        return task


def create_sed_pipeline(args: argparse.Namespace) -> Pipeline:
    """Build the SED labeling pipeline."""
    pipeline = Pipeline(
        name="sound_event_detection",
        description="Load audio, run CNN14 SED inference, and label sound events.",
    )

    pipeline.add_stage(ManifestReader(manifest_path=args.input_manifest))
    pipeline.add_stage(_LoadWaveformStage())
    pipeline.add_stage(
        SEDInferenceStage(
            checkpoint_path=args.checkpoint,
            model_type=args.model_type,
            sample_rate=args.sample_rate,
        ).with_(batch_size=args.batch_size, resources=Resources(cpus=1.0, gpus=args.gpus))
    )
    pipeline.add_stage(
        SEDPostprocessingStage(
            threshold=args.threshold,
            min_duration_sec=args.min_duration_sec,
            emit_subcategories=args.emit_subcategories,
        )
    )
    pipeline.add_stage(_KeepEventsStage())
    pipeline.add_stage(AudioToDocumentStage())
    pipeline.add_stage(JsonlWriter(path=args.output_dir, write_kwargs={"force_ascii": False}))

    return pipeline


def main(args: argparse.Namespace) -> None:
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.verbose else "INFO")

    pipeline = create_sed_pipeline(args)
    logger.info(pipeline.describe())

    logger.info("Starting SED pipeline execution...")
    pipeline.run(XennaExecutor())
    logger.info("SED pipeline completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input-manifest", type=str, required=True, help="JSONL manifest with 'audio_filepath' per line"
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="PANNs Cnn14_DecisionLevelMax .pth checkpoint")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for the labeled output manifest")
    parser.add_argument("--model-type", type=str, default="Cnn14_DecisionLevelMax", help="CNN14 variant")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Model target sample rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Waveforms per GPU forward pass")
    parser.add_argument("--gpus", type=float, default=1.0, help="GPUs to request for the inference stage")
    parser.add_argument("--threshold", type=float, default=0.5, help="Event-detection probability threshold")
    parser.add_argument("--min-duration-sec", type=float, default=0.3, help="Drop events shorter than this")
    parser.add_argument(
        "--emit-subcategories", action="store_true", help="Label per AudioSet class instead of superclass"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    main(parser.parse_args())
