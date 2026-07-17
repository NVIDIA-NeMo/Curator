# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Metadata planning, bounded segment decoding, and parent reassembly."""

from __future__ import annotations

import copy
import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from fsspec.core import url_to_fs

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.audio.inference.batch_policy import BatchPolicy
from nemo_curator.stages.audio.model_input_segmentation import plan_audio_segments
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask, DispatchBatchTask, EmptyTask

if TYPE_CHECKING:
    import numpy as np

    from nemo_curator.backends.base import WorkerMetadata

SEGMENT_DESCRIPTOR_KEY = "_curator_audio_segment"
PARENT_ID_KEY = "_curator_parent_id"
PARENT_REPOSITORY_KEY = "_curator_parent_repository"
SEGMENT_INDEX_KEY = "_curator_segment_index"
SEGMENT_COUNT_KEY = "_curator_segment_count"


@dataclass(frozen=True)
class AudioSegmentDescriptor:
    """A payload-free file range that can be decoded independently."""

    audio_filepath: str
    start_s: float
    duration_s: float


class AudioSegmentDecoder(Protocol):
    """Optional seam for replacing inline file decode in a future PR."""

    def decode(self, descriptor: AudioSegmentDescriptor) -> tuple[np.ndarray, int]:
        """Return one bounded mono waveform and its sample rate."""


@dataclass
class SoundFileSegmentDecoder:
    """Decode exactly one descriptor; never read the complete source corpus."""

    storage_options: dict[str, Any] | None = None

    def decode(self, descriptor: AudioSegmentDescriptor) -> tuple[np.ndarray, int]:
        import soundfile

        fs, resolved = url_to_fs(descriptor.audio_filepath, **(self.storage_options or {}))
        with fs.open(resolved, "rb") as stream, soundfile.SoundFile(stream) as audio:
            sample_rate = int(audio.samplerate)
            start_frame = max(0, round(descriptor.start_s * sample_rate))
            frame_count = max(0, round(descriptor.duration_s * sample_rate))
            audio.seek(min(start_frame, len(audio)))
            waveform = audio.read(frames=frame_count, dtype="float32", always_2d=True)
        return waveform.mean(axis=1), sample_rate


class _ParentMetadataRepository:
    """Global-bucketing-owned metadata store with one copy per parent."""

    def __init__(self) -> None:
        self.parents: dict[str, dict[str, Any]] = {}

    def put_many(self, parents: dict[str, dict[str, Any]]) -> None:
        for parent_id, parent in parents.items():
            existing = self.parents.get(parent_id)
            if existing is not None and existing != parent:
                msg = f"Parent metadata conflict for {parent_id}"
                raise ValueError(msg)
            if existing is None:
                self.parents[parent_id] = copy.deepcopy(parent)

    def get_parent(self, parent_id: str) -> dict[str, Any] | None:
        parent = self.parents.get(parent_id)
        return None if parent is None else copy.deepcopy(parent)


_LOCAL_PARENT_REPOSITORIES: dict[str, _ParentMetadataRepository] = {}


def _safe_actor_suffix(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _local_parent_repository(repository_id: str) -> _ParentMetadataRepository:
    return _LOCAL_PARENT_REPOSITORIES.setdefault(repository_id, _ParentMetadataRepository())


def _ray_parent_repository(repository_id: str) -> Any:  # noqa: ANN401
    import ray

    actor_name = f"curator_audio_parent_repository_{_safe_actor_suffix(repository_id)}"
    return (
        ray.remote(_ParentMetadataRepository)
        .options(name=actor_name, get_if_exists=True, lifetime="detached")
        .remote()
    )


def _kill_named_actor(actor_name: str) -> None:
    import ray

    if not ray.is_initialized():
        return
    try:
        actor = ray.get_actor(actor_name)
    except ValueError:
        return
    ray.kill(actor)


@dataclass
class GlobalAudioManifestPlannerStage(ProcessingStage[EmptyTask, DispatchBatchTask]):
    """Read all manifest metadata once and emit globally packed dispatches."""

    manifest_path: str | list[str]
    owner_stage: str = "FastConformerDispatchStage"
    max_model_input_duration_s: float = 600.0
    batch_policy: BatchPolicy = field(default_factory=BatchPolicy)
    audio_filepath_key: str = "audio_filepath"
    duration_key: str = "duration"
    storage_options: dict[str, Any] | None = None
    dataset_name: str = "audio"
    repository_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    name: str = "GlobalAudioManifestPlannerStage"
    batch_size: int = 1
    is_resumable: bool = False

    def __post_init__(self) -> None:
        if not self.manifest_path:
            msg = "GlobalAudioManifestPlannerStage.manifest_path is required"
            raise ValueError(msg)
        if not self.owner_stage:
            msg = "GlobalAudioManifestPlannerStage.owner_stage is required"
            raise ValueError(msg)
        if not self.batch_policy.enabled:
            msg = "Global audio planning requires BatchPolicy.enabled=true"
            raise ValueError(msg)
        if not self.repository_id:
            msg = "GlobalAudioManifestPlannerStage.repository_id is required"
            raise ValueError(msg)

    def num_workers(self) -> int | None:
        return 1

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}

    def process(self, task: EmptyTask) -> list[DispatchBatchTask]:
        del task
        segments: list[AudioTask] = []
        costs: list[float] = []
        parents: dict[str, dict[str, Any]] = {}
        for manifest_index, path in enumerate(self._manifest_paths()):
            fs, resolved = url_to_fs(path, **(self.storage_options or {}))
            with fs.open(resolved, "r", encoding="utf-8") as stream:
                for row_index, line in enumerate(stream):
                    if not line.strip():
                        continue
                    parent = json.loads(line)
                    planned = self._plan_parent(parent, manifest_index, row_index, costs)
                    segments.extend(planned)
                    parents[str(planned[0].data[PARENT_ID_KEY])] = parent

        self._store_parents(parents)

        signature = self.batch_policy.dispatch_signature()
        packed = self.batch_policy.bucketize_with_costs(segments, costs)
        return [
            DispatchBatchTask(
                dataset_name=self.dataset_name,
                data=batch_items,
                batch_id=f"global-audio:{sequence_index}",
                owner_stage=self.owner_stage,
                sequence_index=sequence_index,
                bucket_index=bucket_index,
                total_cost=sum(batch_costs),
                item_costs=tuple(batch_costs),
                cost_unit="seconds",
                policy_signature=signature,
            )
            for sequence_index, (bucket_index, batch_items, batch_costs) in enumerate(packed)
        ]

    def _store_parents(self, parents: dict[str, dict[str, Any]]) -> None:
        import ray

        if ray.is_initialized():
            ray.get(_ray_parent_repository(self.repository_id).put_many.remote(parents))
        else:
            _local_parent_repository(self.repository_id).put_many(parents)

    def cleanup_run_resources(self) -> None:
        suffix = _safe_actor_suffix(self.repository_id)
        _kill_named_actor(f"curator_audio_parent_assembly_{suffix}")
        _kill_named_actor(f"curator_audio_parent_repository_{suffix}")
        _LOCAL_PARENT_REPOSITORIES.pop(self.repository_id, None)

    def _manifest_paths(self) -> list[str]:
        return [self.manifest_path] if isinstance(self.manifest_path, str) else list(self.manifest_path)

    def _plan_parent(
        self,
        parent: dict[str, Any],
        manifest_index: int,
        row_index: int,
        costs: list[float],
    ) -> list[AudioTask]:
        if "waveform" in parent:
            msg = "Global metadata planning refuses inline waveform payloads"
            raise ValueError(msg)
        audio_filepath = parent.get(self.audio_filepath_key)
        duration = parent.get(self.duration_key)
        if not isinstance(audio_filepath, str) or not audio_filepath:
            msg = f"Manifest row {manifest_index}:{row_index} has no {self.audio_filepath_key!r}"
            raise ValueError(msg)
        if not isinstance(duration, (int, float)) or isinstance(duration, bool) or duration <= 0:
            msg = f"Manifest row {manifest_index}:{row_index} has invalid {self.duration_key!r}: {duration!r}"
            raise ValueError(msg)

        parent_id = f"{manifest_index}:{row_index}"
        plans = plan_audio_segments(float(duration), self.max_model_input_duration_s)
        tasks: list[AudioTask] = []
        for plan in plans:
            costs.append(plan.duration_s)
            tasks.append(
                AudioTask(
                    dataset_name=self.dataset_name,
                    data={
                        "audio_filepath": audio_filepath,
                        "duration": plan.duration_s,
                        "segment_start_s": plan.start_s,
                        "segment_duration_s": plan.duration_s,
                        SEGMENT_DESCRIPTOR_KEY: AudioSegmentDescriptor(
                            audio_filepath=audio_filepath,
                            start_s=plan.start_s,
                            duration_s=plan.duration_s,
                        ),
                        PARENT_ID_KEY: parent_id,
                        PARENT_REPOSITORY_KEY: self.repository_id,
                        SEGMENT_INDEX_KEY: plan.index,
                        SEGMENT_COUNT_KEY: plan.count,
                    },
                )
            )
        return tasks


class _ParentAssemblyState:
    """Idempotent segment accumulator with acknowledged retry replay."""

    def __init__(self, pred_text_key: str, parent_repository: Any) -> None:  # noqa: ANN401
        self.pred_text_key = pred_text_key
        self.parent_repository = parent_repository
        self.pending: dict[str, dict[int, dict[str, Any]]] = {}
        self.expected_counts: dict[str, int] = {}
        self.completed: dict[str, dict[int, dict[str, Any]]] = {}
        self.operations: dict[str, tuple[dict[str, Any], tuple[AudioTask, ...]]] = {}

    def add_many(
        self,
        tasks: list[AudioTask],
        acknowledge_operation_ids: tuple[str, ...] = (),
    ) -> list[AudioTask]:
        for operation_id in acknowledge_operation_ids:
            self.operations.pop(operation_id, None)

        outputs: list[AudioTask] = []
        seen_in_call: set[str] = set()
        for task in tasks:
            operation_id = self.operation_id(task)
            snapshot = copy.deepcopy(dict(task.data))
            cached = self.operations.get(operation_id)
            if cached is not None:
                cached_snapshot, cached_outputs = cached
                if cached_snapshot != snapshot:
                    msg = f"Conflicting retry for segment operation {operation_id}"
                    raise ValueError(msg)
                if operation_id not in seen_in_call:
                    outputs.extend(copy.deepcopy(list(cached_outputs)))
                seen_in_call.add(operation_id)
                continue

            result = self._add_new(task, snapshot)
            cached_outputs = () if result is None else (result,)
            self.operations[operation_id] = (snapshot, cached_outputs)
            if result is not None:
                outputs.append(result)
            seen_in_call.add(operation_id)
        return outputs

    def _add_new(self, task: AudioTask, snapshot: dict[str, Any]) -> AudioTask | None:
        parent_id = str(snapshot[PARENT_ID_KEY])
        index = int(snapshot[SEGMENT_INDEX_KEY])
        count = int(snapshot[SEGMENT_COUNT_KEY])
        if count <= 0 or index < 0 or index >= count:
            msg = f"Invalid segment {index}/{count} for parent {parent_id}"
            raise ValueError(msg)

        completed_parts = self.completed.get(parent_id)
        if completed_parts is not None:
            completed_snapshot = completed_parts.get(index)
            if completed_snapshot != snapshot:
                msg = f"Conflicting segment {index} delivered after parent {parent_id} completed"
                raise ValueError(msg)
            return None

        expected_count = self.expected_counts.setdefault(parent_id, count)
        if expected_count != count:
            msg = f"Conflicting segment counts for parent {parent_id}: {expected_count} and {count}"
            raise ValueError(msg)

        parts = self.pending.setdefault(parent_id, {})
        existing = parts.get(index)
        if existing is not None:
            if existing != snapshot:
                msg = f"Conflicting duplicate segment {index} for parent {parent_id}"
                raise ValueError(msg)
            return None
        parts[index] = snapshot
        if len(parts) != count:
            return None

        missing = set(range(count)) - parts.keys()
        if missing:
            msg = f"Parent {parent_id} is missing segment indices {sorted(missing)}"
            raise RuntimeError(msg)
        parent = self._parent_data(parent_id)
        if parent is None:
            msg = f"Parent metadata repository has no row for {parent_id}"
            raise RuntimeError(msg)
        ordered = [parts[segment_index] for segment_index in range(count)]
        parent[self.pred_text_key] = " ".join(
            str(segment[self.pred_text_key]).strip() for segment in ordered if str(segment[self.pred_text_key]).strip()
        )
        self.completed[parent_id] = copy.deepcopy(parts)
        del self.pending[parent_id]
        del self.expected_counts[parent_id]
        return AudioTask(
            dataset_name=task.dataset_name,
            data=parent,
            _metadata=dict(task._metadata),
            _stage_perf=list(task._stage_perf),
        )

    def _parent_data(self, parent_id: str) -> dict[str, Any] | None:
        getter = self.parent_repository.get_parent
        if hasattr(getter, "remote"):
            import ray

            return ray.get(getter.remote(parent_id))
        return getter(parent_id)

    def missing_segments(self) -> dict[str, tuple[int, ...]]:
        return {
            parent_id: tuple(sorted(set(range(count)) - self.pending.get(parent_id, {}).keys()))
            for parent_id, count in self.expected_counts.items()
        }

    def assert_complete(self) -> None:
        missing = self.missing_segments()
        if missing:
            msg = f"Incomplete parent segment sets: {missing}"
            raise RuntimeError(msg)

    @staticmethod
    def operation_id(task: AudioTask) -> str:
        return f"{task.data[PARENT_ID_KEY]}:segment:{task.data[SEGMENT_INDEX_KEY]}"


@dataclass
class GlobalAudioParentAssemblerStage(ProcessingStage[AudioTask, AudioTask]):
    """Reassemble globally reordered segments through one job-scoped state owner."""

    pred_text_key: str = "pred_text"
    name: str = "GlobalAudioParentAssemblerStage"
    batch_size: int = 64
    is_resumable: bool = False
    _states: dict[str, _ParentAssemblyState] = field(init=False, default_factory=dict, repr=False)
    _actors: dict[str, Any] = field(init=False, default_factory=dict, repr=False)
    _pending_ack_ids: dict[str, tuple[str, ...]] = field(init=False, default_factory=dict, repr=False)

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        return None

    def process(self, task: AudioTask) -> AudioTask | None:
        results = self.process_batch([task])
        return results[0] if results else None

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        import ray

        grouped: dict[str, list[AudioTask]] = {}
        for task in tasks:
            repository_id = task.data.get(PARENT_REPOSITORY_KEY)
            if not isinstance(repository_id, str) or not repository_id:
                msg = f"Audio segment is missing {PARENT_REPOSITORY_KEY}"
                raise ValueError(msg)
            grouped.setdefault(repository_id, []).append(task)

        outputs: list[AudioTask] = []
        for repository_id, repository_tasks in grouped.items():
            acknowledgements = self._pending_ack_ids.get(repository_id, ())
            if ray.is_initialized():
                actor = self._assembly_actor(repository_id)
                results = ray.get(
                    actor.add_many.remote(
                        repository_tasks,
                        acknowledge_operation_ids=acknowledgements,
                    )
                )
            else:
                state = self._states.setdefault(
                    repository_id,
                    _ParentAssemblyState(self.pred_text_key, _local_parent_repository(repository_id)),
                )
                results = state.add_many(repository_tasks, acknowledge_operation_ids=acknowledgements)
            self._pending_ack_ids[repository_id] = tuple(
                dict.fromkeys(_ParentAssemblyState.operation_id(task) for task in repository_tasks)
            )
            outputs.extend(results)
        return outputs

    def _assembly_actor(self, repository_id: str) -> Any:  # noqa: ANN401
        import ray

        actor = self._actors.get(repository_id)
        if actor is not None:
            return actor
        actor_name = f"curator_audio_parent_assembly_{_safe_actor_suffix(repository_id)}"
        actor = (
            ray.remote(_ParentAssemblyState)
            .options(name=actor_name, get_if_exists=True, lifetime="detached")
            .remote(self.pred_text_key, _ray_parent_repository(repository_id))
        )
        self._actors[repository_id] = actor
        return actor
