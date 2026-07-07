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

"""
Speaker separation stage using NeMo SortFormer diarization model.

Performs speaker diarization and separates audio by speaker,
creating separate AudioTask outputs for each speaker.

Example:
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.audio.segmentation import SpeakerSeparationStage
    from nemo_curator.stages.resources import Resources

    pipeline = Pipeline(name="speaker_pipeline")
    pipeline.add_stage(
        SpeakerSeparationStage(exclude_overlaps=True, min_duration=0.8)
        .with_(resources=Resources(cpus=1.0, gpus=1.0))
    )
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from loguru import logger
from pydub import AudioSegment

try:
    from nemo.collections.asr.models import SortformerEncLabelModel
except ImportError:
    SortformerEncLabelModel = None

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.audio._agent_ready import AgentReady, Gates, IOSpec, StageContract
from nemo_curator.stages.audio._residency import resolve_audio
from nemo_curator.stages.audio.segmentation.speaker_separation_module.speaker_sep import SpeakerSeparator
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


def _pydub_to_waveform_sr(seg: AudioSegment) -> tuple[torch.Tensor, int]:
    """Convert PyDub AudioSegment to (waveform, sample_rate). Output is canonical format only."""
    max_val = float(1 << (8 * seg.sample_width - 1))
    samples = np.array(seg.get_array_of_samples(), dtype=np.float32) / max_val
    if seg.channels > 1:
        samples = samples.reshape((-1, seg.channels)).mean(axis=1)
    return torch.from_numpy(samples).unsqueeze(0), seg.frame_rate


@dataclass
class SpeakerSeparationStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """
    Speaker separation stage using NeMo SortFormer diarization model.

    Separates audio by speaker and creates separate AudioTask outputs
    for each speaker's segments. Downloads the NeMo model from
    HuggingFace Hub (nvidia/diar_sortformer_4spk-v1).

    Args:
        model_path: HuggingFace model ID or path to NeMo diarization model
        exclude_overlaps: Whether to exclude overlapping speaker regions
        min_duration: Minimum segment duration in seconds
        gap_threshold: Gap threshold for merging speaker segments
        buffer_time: Buffer time around speaker segments
        audio_filepath_key: Key in data dict for the input audio file path.
        waveform_key: Key in data dict for the in-memory waveform tensor.
        sample_rate_key: Key in data dict for the waveform sample rate.
        speaker_id_key: Key where each child task's speaker id is written.
        num_speakers_key: Key where the detected speaker count is written.
        duration_key: Key where each child's speech duration in seconds is written.
        diar_segments_key: Key where each child's diarization segments are written.
        input_residency: Which input to use — "waveform" (in-memory only), "file"
            (audio_filepath only), or "auto" (waveform first, file fallback; default).

    Note:
        Per-speaker child tasks DROP the parent's audio_filepath (it points at the
        full multi-speaker file) and carry ``original_file`` for provenance instead;
        downstream stages consume the per-speaker waveform.

        GPU assignment is handled by the executor via _resources.
        Use .with_(resources=Resources(gpus=X)) to configure GPU allocation.
    """

    model_path: str = "nvidia/diar_sortformer_4spk-v1"
    exclude_overlaps: bool = True
    min_duration: float = 0.8
    gap_threshold: float = 0.1
    buffer_time: float = 0.5
    audio_filepath_key: str = "audio_filepath"
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"
    speaker_id_key: str = "speaker_id"
    num_speakers_key: str = "num_speakers"
    duration_key: str = "duration"
    diar_segments_key: str = "diar_segments"
    input_residency: str = "auto"

    name: str = "SpeakerSeparation"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpus=1.0))

    def __post_init__(self):
        super().__init__()
        self._separator = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [
            self.waveform_key,
            self.sample_rate_key,
            self.speaker_id_key,
            self.num_speakers_key,
            self.duration_key,
            self.diar_segments_key,
        ]

    def describe(self) -> StageContract:
        return StageContract(
            reads_one_of=[
                IOSpec(data_keys=[self.waveform_key, self.sample_rate_key], accepts=["waveform"]),
                IOSpec(data_keys=[self.audio_filepath_key], accepts=["file"]),
            ],
            writes=IOSpec(
                data_keys=[
                    self.waveform_key,
                    self.sample_rate_key,
                    self.speaker_id_key,
                    self.num_speakers_key,
                    self.duration_key,
                    self.diar_segments_key,
                    "original_file",
                ],
                produces=["tensor"],
            ),
            # children drop the parent's audio_filepath (and blob keys)
            preserves_upstream_keys=False,
            cardinality="1:N fan-out",
            # One child per detected speaker; speaker_id is the per-child key that
            # identifies which slice of the iteration a child is (role-resolvable).
            iteration_key=self.speaker_id_key,
            gates=Gates(requires_gpu=self.resources.gpus > 0, requires_internet_first_run=True),
        )

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}

    def setup_on_node(self, _node_info: Any = None, _worker_metadata: Any = None) -> None:  # noqa: ANN401
        try:
            SortformerEncLabelModel.from_pretrained(self.model_path)
        except Exception:  # noqa: BLE001
            logger.warning("Model pre-download in setup_on_node failed; will retry in setup().")

    def setup(self, _: WorkerMetadata | None = None) -> None:
        self._initialize_separator()

    def teardown(self) -> None:
        if self._separator is not None:
            del self._separator
            self._separator = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def _check_gpu_availability(gpus: float) -> None:
        if gpus > 0 and not torch.cuda.is_available():
            msg = (
                "Resources request GPU (gpus > 0) but CUDA is not available. "
                "Either set resources=Resources(gpus=0) for CPU-only or install CUDA."
            )
            raise RuntimeError(msg)

    def _initialize_separator(self) -> None:
        if self._separator is None:
            self._check_gpu_availability(self._resources.gpus)
            try:
                use_gpu = self._resources.gpus > 0 and torch.cuda.is_available()

                separator_config = {
                    "speaker_model_path": self.model_path,
                    "speaker_gap_threshold": self.gap_threshold,
                    "speaker_exclude_overlaps": self.exclude_overlaps,
                    "speaker_min_duration": self.min_duration,
                    "speaker_buffer_time": self.buffer_time,
                    "use_gpu": use_gpu,
                }

                self._separator = SpeakerSeparator(
                    model_name=self.model_path,
                    config=separator_config,
                )

                logger.info(f"NeMo speaker separator loaded from HuggingFace: {self.model_path}")
            except ImportError as e:
                logger.error(f"Failed to import speaker separation module: {e}")
                raise
            except Exception as e:
                logger.error(f"Failed to load speaker separator: {e}")
                raise

    # Keys dropped from the parent task when building per-speaker child tasks.
    # "audio"/"waveform" are non-serializable blobs replaced by per-speaker audio.
    # "duration"/"num_samples" describe the parent file, not the speaker segment;
    # each child gets its own duration from the diarization result.
    _INHERITED_DROP_KEYS = frozenset({"audio", "waveform", "duration", "num_samples"})

    def _build_speaker_tasks(
        self,
        speaker_audio_data: dict,
        item: dict,
        task: AudioTask,
    ) -> list[AudioTask]:
        """Build AudioTask list from speaker audio data."""
        results: list[AudioTask] = []
        num_speakers = len(speaker_audio_data)
        for speaker_id, result in speaker_audio_data.items():
            if result.duration < self.min_duration:
                logger.debug(f"Skipping {speaker_id}: duration {result.duration:.2f}s < {self.min_duration}s")
                continue
            spk_waveform, spk_sr = _pydub_to_waveform_sr(result.audio)
            # Drop the parent's file path(s) too: they point at the FULL
            # multi-speaker file, so a file-preferring downstream stage would
            # process the whole file per speaker instead of this speaker's
            # extracted waveform. With the path gone, downstream resolves the
            # per-speaker waveform (input_residency="auto") instead.
            drop_keys = {
                *self._INHERITED_DROP_KEYS,
                self.waveform_key,
                self.duration_key,
                self.audio_filepath_key,
                "audio_filepath",
            }
            speaker_data = {
                **{k: v for k, v in item.items() if k not in drop_keys},
                self.waveform_key: spk_waveform,
                self.sample_rate_key: spk_sr,
                self.speaker_id_key: speaker_id,
                self.num_speakers_key: num_speakers,
                self.duration_key: result.duration,
                self.diar_segments_key: result.diar_segments,
                # Source identity must survive the audio_filepath drop above —
                # TimestampMapper (and any provenance consumer) reads original_file.
                "original_file": item.get("original_file")
                or item.get(self.audio_filepath_key)
                or item.get("audio_filepath")
                or "unknown",
            }
            spk_task = AudioTask(
                data=speaker_data,
                dataset_name=task.dataset_name,
                _metadata=dict(task._metadata or {}),
                _stage_perf=list(task._stage_perf),
            )
            results.append(spk_task)
        return results

    def process(self, task: AudioTask) -> list[AudioTask]:
        """
        Separate audio by speaker.

        Returns:
            List of AudioTask objects, one per speaker.
        """
        if self._separator is None:
            msg = "Speaker separator failed to initialize. Cannot process audio."
            raise RuntimeError(msg)

        item = dict(task.data)
        waveform = None
        results: list[AudioTask] = []

        try:
            audio_result = resolve_audio(
                item,
                residency=self.input_residency,  # type: ignore[arg-type]
                audio_filepath_key=self.audio_filepath_key,
                waveform_key=self.waveform_key,
                sample_rate_key=self.sample_rate_key,
            )
            if audio_result is None:
                return []
            waveform, sample_rate = audio_result

            speaker_audio_data = self._separator.get_speaker_audio_data(
                waveform,
                sample_rate=sample_rate,
                gap_threshold=self.gap_threshold,
                exclude_overlaps=self.exclude_overlaps,
                min_duration=self.min_duration,
                buffer_time=self.buffer_time,
            )

            if len(speaker_audio_data) == 0:
                logger.warning("No speakers detected")
                return []

            logger.info(f"Detected {len(speaker_audio_data)} speakers")
            results = self._build_speaker_tasks(speaker_audio_data, item, task)

        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            msg = (
                "CUDA out of memory processing audio. "
                "Consider splitting long audio files into shorter segments, "
                "using a GPU with more memory, or setting resources=Resources(gpus=0) for CPU mode."
            )
            raise RuntimeError(msg) from e
        except Exception as e:  # noqa: BLE001
            logger.exception(f"[SpeakerSeparation] Failed to process task {task.task_id}: {e}")
        finally:
            if waveform is not None:
                del waveform
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return results
