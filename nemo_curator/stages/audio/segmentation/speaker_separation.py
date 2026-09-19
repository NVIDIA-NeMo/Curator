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

import contextlib
import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast

import numpy as np
import soundfile as sf
import torch
from loguru import logger
from pydub import AudioSegment

try:
    from nemo.collections.asr.models import SortformerEncLabelModel
except ImportError:
    SortformerEncLabelModel = None

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.audio._agent._agent_ready import AgentReady, Gates, IOSpec, StageContract, StaticHints
from nemo_curator.stages.audio._agent._residency import (
    InputResidency,
    accepts_for_residency,
    resolve_audio,
    validate_audio_key_configuration,
    validate_input_residency,
    write_audio_stable,
)
from nemo_curator.stages.audio.common import ensure_mono, ensure_waveform_2d
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


def _bounded_speaker_stem(stem: str, speaker_id: str) -> str:
    """Fit the content-addressed WAV name within a POSIX NAME_MAX component."""
    suffix = f"_{speaker_id}_{'0' * 16}.wav"
    budget = 255 - len(suffix.encode("utf-8"))
    encoded = stem.encode("utf-8")
    if len(encoded) <= budget:
        return stem
    digest = hashlib.sha256(encoded).hexdigest()[:12]
    marker = f"~{digest}"
    encoded = encoded[: budget - len(marker)]
    while True:
        try:
            prefix = encoded.decode("utf-8")
            break
        except UnicodeDecodeError:
            encoded = encoded[:-1]
    return f"{prefix}{marker}"


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
        waveform_key: Key in data dict for the in-memory waveform tensor.
        sample_rate_key: Key in data dict for the waveform sample rate.
        audio_filepath_key: Key in data dict for the input audio file path.
        speaker_id_key: Key where each child task's speaker id is written.
        num_speakers_key: Key where the detected speaker count is written.
        duration_key: Key where each child's speech duration in seconds is written.
        diar_segments_key: Key where each child's diarization segments are written.
        original_file_key: Key carrying the source file path for provenance.
        input_residency: Which input to use — "waveform" (in-memory only), "file"
            (audio_filepath only), or "auto" (waveform first, file fallback; default).
        keep_waveform_in_task: Keep each per-speaker waveform in the task (default True,
            today's behavior). Set False to emit only on-disk paths (requires write_to_disk).
        write_to_disk: Also write each per-speaker track to ``separated_audio_dir`` and set
            ``audio_filepath_key`` to it, so file-based downstream stages can consume it.
            Defaults to False (in-memory only, unchanged).
        separated_audio_dir: Directory for per-speaker WAVs (required when write_to_disk=True).

    Note:
        By default (write_to_disk=False) per-speaker child tasks retain the parent's
        audio_filepath for legacy provenance and carry ``original_file`` explicitly;
        downstream consumes the per-speaker waveform.
        With write_to_disk=True, each child instead gets its own audio_filepath.

        GPU assignment is handled by the executor via _resources.
        Use .with_(resources=Resources(gpus=X)) to configure GPU allocation.
    """

    model_path: str = "nvidia/diar_sortformer_4spk-v1"
    exclude_overlaps: bool = True
    min_duration: float = 0.8
    gap_threshold: float = 0.1
    buffer_time: float = 0.5

    name: str = "SpeakerSeparation"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpus=1.0))

    # New agent/residency fields follow every legacy positional field.
    audio_filepath_key: str = "audio_filepath"
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"
    speaker_id_key: str = "speaker_id"
    num_speakers_key: str = "num_speakers"
    duration_key: str = "duration"
    diar_segments_key: str = "diar_segments"
    original_file_key: str = "original_file"
    input_residency: InputResidency = "auto"
    # Output residency (both default to today's behavior: in-memory waveform only, no disk).
    keep_waveform_in_task: bool = True
    write_to_disk: bool = False
    separated_audio_dir: str | None = None

    AGENT_STATIC: ClassVar[StaticHints] = StaticHints(gates=Gates(requires_internet_first_run=True))

    def __post_init__(self):
        super().__init__()
        self._separator = None
        validate_input_residency(self.input_residency, stage_name=self.name)
        metadata_output_keys = {
            "speaker_id_key": self.speaker_id_key,
            "num_speakers_key": self.num_speakers_key,
            "duration_key": self.duration_key,
            "diar_segments_key": self.diar_segments_key,
            "original_file_key": self.original_file_key,
        }
        validate_audio_key_configuration(
            self.name,
            input_keys={
                "audio_filepath_key": self.audio_filepath_key,
                "waveform_key": self.waveform_key,
                "sample_rate_key": self.sample_rate_key,
            },
            output_keys=metadata_output_keys,
        )
        validate_audio_key_configuration(
            self.name,
            input_keys={},
            output_keys={
                **metadata_output_keys,
                "waveform_key": self.waveform_key,
                "sample_rate_key": self.sample_rate_key,
            },
        )
        if not (self.keep_waveform_in_task or self.write_to_disk):
            msg = "At least one of keep_waveform_in_task or write_to_disk must be True"
            raise ValueError(msg)
        if self.write_to_disk and not self.separated_audio_dir:
            msg = "separated_audio_dir is required when write_to_disk=True"
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        outs: list[str] = []
        if self.keep_waveform_in_task:
            outs.append(self.waveform_key)
        outs.append(self.sample_rate_key)
        outs.extend([self.speaker_id_key, self.num_speakers_key, self.duration_key, self.diar_segments_key])
        outs.append(self.original_file_key)
        if self.write_to_disk:
            outs.append(self.audio_filepath_key)
        return [], outs

    def _removed_output_keys(self) -> list[str]:
        """Return parent fields that are absent rather than replaced in each child."""
        removes = {"audio", "num_samples"}
        if not self.keep_waveform_in_task:
            removes.add(self.waveform_key)
        if self.waveform_key != "waveform" or not self.keep_waveform_in_task:
            removes.add("waveform")
        if self.write_to_disk and self.audio_filepath_key != "audio_filepath":
            removes.add("audio_filepath")
        if self.sample_rate_key != "sample_rate":
            removes.add("sample_rate")
        if self.duration_key != "duration":
            removes.add("duration")
        return sorted(removes)

    def describe(self) -> StageContract:
        forms = accepts_for_residency(self.input_residency)
        reads_one_of = []
        if "waveform" in forms:
            reads_one_of.append(IOSpec(data_keys=[self.waveform_key, self.sample_rate_key], accepts=["waveform"]))
        if "file" in forms:
            reads_one_of.append(IOSpec(data_keys=[self.audio_filepath_key], accepts=["file"]))
        writes: list[str] = []
        produces: list[str] = []
        if self.keep_waveform_in_task:
            writes.append(self.waveform_key)
            produces.append("tensor")
        writes.append(self.sample_rate_key)
        writes.extend(
            [
                self.speaker_id_key,
                self.num_speakers_key,
                self.duration_key,
                self.diar_segments_key,
                self.original_file_key,
            ]
        )
        if self.write_to_disk:
            writes.append(self.audio_filepath_key)
            produces.append("disk")
        invalidates = [] if self.write_to_disk else list(dict.fromkeys([self.audio_filepath_key, "audio_filepath"]))
        return StageContract(
            reads_one_of=reads_one_of,
            writes=IOSpec(data_keys=writes, produces=produces),
            preserves_upstream_keys=True,
            cardinality="1:N fan-out",
            # One child per detected speaker; speaker_id is the per-child key that
            # identifies which slice of the iteration a child is (role-resolvable).
            iteration_key=self.speaker_id_key,
            gates=Gates(
                requires_gpu=self.resources.requires_gpu,
                requires_internet_first_run=True,
                writes_to_disk=self.write_to_disk,
                output_path_params=["separated_audio_dir"],
                # Diarization runs on one file's audio, and ``num_speakers`` counts the speakers
                # found in THAT file. Unlike SplitLongAudioStage, a shared output directory is
                # still safe here: ``write_audio_stable`` names each per-speaker WAV after a
                # digest of its own samples rather than after the source basename.
                per_row_independent=True,
            ),
            removes_keys=self._removed_output_keys(),
            invalidates_keys=invalidates,
        )

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

    def _write_speaker_wav(
        self,
        waveform: torch.Tensor,
        sr: int,
        original_file: str,
        speaker_id: str,
        *,
        output_dir: str | None = None,
    ) -> str:
        """Write one per-speaker waveform to ``separated_audio_dir`` and return the path."""
        stem = os.path.splitext(os.path.basename(str(original_file)))[0] or "audio"
        stem = _bounded_speaker_stem(stem, str(speaker_id))
        return write_audio_stable(
            waveform,
            sr,
            output_dir=output_dir or self.separated_audio_dir,
            stem=stem,
            tag=str(speaker_id),
        )

    def _persist_speaker_wavs(
        self,
        pending: list[tuple[dict[str, Any], torch.Tensor, int, str]],
    ) -> None:
        """Stage every WAV and publish immutable outputs without clobbering existing files."""
        output_dir = str(self.separated_audio_dir)
        os.makedirs(output_dir, exist_ok=True)
        staging_dir = tempfile.mkdtemp(prefix=".speaker-separation-", dir=output_dir)
        try:
            staged: list[tuple[dict[str, Any], str, str]] = []
            for speaker_data, waveform, sample_rate, speaker_id in pending:
                staged_path = self._write_speaker_wav(
                    waveform,
                    sample_rate,
                    speaker_data[self.original_file_key],
                    speaker_id,
                    output_dir=staging_dir,
                )
                final_path = os.path.join(output_dir, os.path.basename(staged_path))
                staged.append((speaker_data, staged_path, final_path))

            for speaker_data, staged_path, final_path in staged:
                with contextlib.suppress(FileExistsError):
                    # The staging directory is inside output_dir, so a hard link is an
                    # atomic no-clobber publish. A pre-existing content-addressed output
                    # is already the complete desired artifact and must not be replaced.
                    os.link(staged_path, final_path)
                speaker_data[self.audio_filepath_key] = final_path
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)

    def _prepare_speaker_outputs(
        self,
        speaker_audio_data: dict,
        item: dict,
    ) -> list[tuple[dict[str, Any], torch.Tensor, int, str]]:
        """Convert separator output into child dictionaries and persistence inputs."""
        pending: list[tuple[dict[str, Any], torch.Tensor, int, str]] = []
        num_speakers = len(speaker_audio_data)
        for speaker_id, result in speaker_audio_data.items():
            if result.duration < self.min_duration:
                logger.debug(f"Skipping {speaker_id}: duration {result.duration:.2f}s < {self.min_duration}s")
                continue
            spk_waveform, spk_sr = _pydub_to_waveform_sr(result.audio)
            drop_keys = {
                *self._INHERITED_DROP_KEYS,
                self.waveform_key,
                self.duration_key,
                self.sample_rate_key,
                "sample_rate",
            }
            if self.write_to_disk:
                drop_keys.update({self.audio_filepath_key, "audio_filepath"})
            original_file = self._original_file(item)
            speaker_data = {
                **{k: v for k, v in item.items() if k not in drop_keys},
                self.speaker_id_key: speaker_id,
                self.num_speakers_key: num_speakers,
                self.duration_key: result.duration,
                self.diar_segments_key: result.diar_segments,
                self.original_file_key: original_file,
                self.sample_rate_key: spk_sr,
            }
            # Output residency: keep the in-memory waveform (default) and/or persist a
            # per-speaker WAV and point audio_filepath at it (opt-in write_to_disk).
            if self.keep_waveform_in_task:
                speaker_data[self.waveform_key] = spk_waveform
            pending.append((speaker_data, spk_waveform, spk_sr, speaker_id))
        return pending

    def _original_file(self, item: dict[str, Any]) -> Any:  # noqa: ANN401
        for key in dict.fromkeys([self.original_file_key, self.audio_filepath_key, "audio_filepath"]):
            if key in item:
                return item[key]
        return "unknown"

    @staticmethod
    def _build_speaker_tasks(
        pending: list[tuple[dict[str, Any], torch.Tensor, int, str]],
        task: AudioTask,
    ) -> list[AudioTask]:
        """Wrap prepared speaker dictionaries as independent AudioTask children."""
        results: list[AudioTask] = []
        for speaker_data, _waveform, _sample_rate, _speaker_id in pending:
            spk_task = AudioTask(
                data=speaker_data,
                dataset_name=task.dataset_name,
                _metadata=dict(task._metadata or {}),
                _stage_perf=list(task._stage_perf),
            )
            results.append(spk_task)
        return results

    def _resolve_audio(self, item: dict[str, Any]) -> tuple[torch.Tensor, int] | None:
        """Resolve mono input while preserving the legacy resident-waveform semantics."""
        if self.input_residency != "file":
            resident = item.get(self.waveform_key)
            if resident is not None:
                sample_rate = item.get(self.sample_rate_key)
                if sample_rate is None:
                    if self.input_residency == "waveform":
                        return None
                    path = item.get(self.audio_filepath_key)
                    if not path:
                        return None
                    expanded = os.path.expanduser(str(path))
                    if not os.path.exists(expanded):
                        return None
                    # The pre-residency stage read only the file header in this
                    # case; it did not replace the caller's resident samples.
                    sample_rate = sf.info(expanded).samplerate
                return ensure_mono(ensure_waveform_2d(resident)), int(sample_rate)

        resolved = resolve_audio(
            item,
            residency=self.input_residency,  # type: ignore[arg-type]
            audio_filepath_key=self.audio_filepath_key,
            waveform_key=self.waveform_key,
            sample_rate_key=self.sample_rate_key,
        )
        if resolved is None:
            return None
        waveform, sample_rate = resolved
        return ensure_mono(waveform), sample_rate

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
        speaker_audio_data: dict[str, Any] | None = None
        pending: list[tuple[dict[str, Any], torch.Tensor, int, str]] | None = None

        try:
            audio_result = self._resolve_audio(item)
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
            pending = self._prepare_speaker_outputs(speaker_audio_data, item)

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
            return []
        finally:
            if waveform is not None:
                del waveform
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        prepared = cast("list[tuple[dict[str, Any], torch.Tensor, int, str]]", pending)
        if self.write_to_disk:
            self._persist_speaker_wavs(prepared)
        return self._build_speaker_tasks(prepared, task)
