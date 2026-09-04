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

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import pandas as pd
from fsspec.core import url_to_fs
from loguru import logger

from nemo_curator.stages.audio._agent._agent_ready import AgentReady, Gates, IOSpec, StageContract, StaticHints
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask, DocumentBatch

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata

_NON_SERIALIZABLE_KEYS = frozenset(
    {
        "waveform",
        "audio",
        "audio_data",
        "audio_array",
        "segments",
    }
)
_DROP_VALUE = object()


def _is_tensor(v: object) -> bool:
    """Check if a value is a torch.Tensor without importing torch at module level."""
    return type(v).__name__ == "Tensor" and type(v).__module__.startswith("torch")


class AudioToDocumentStage(AgentReady, ProcessingStage[AudioTask, DocumentBatch]):
    """Convert AudioTask entries into DocumentBatch DataFrames.

    Overrides ``process_batch`` to aggregate an entire batch of
    ``AudioTask`` objects into a single multi-row ``DocumentBatch``,
    avoiding the overhead of many single-row DataFrames.  Set
    ``batch_size`` to control how many audio entries land in each
    DataFrame (default 64).

    Non-serializable keys (torch tensors, raw audio arrays) are
    stripped before building the DataFrame as a safety net, even if
    upstream stages failed to clean them up.
    """

    name = "AudioToDocumentStage"
    BATCH_ONLY = True  # process() raises; only process_batch is implemented (agent-discovery hint)
    batch_size: int = 64

    def __init__(
        self,
        batch_size: int = 64,
        keep_keys: list[str] | None = None,
        drop_keys: tuple[str, ...] = (),
        serialize_segments: bool = False,
        segments_key: str = "segments",
    ) -> None:
        self.batch_size = batch_size
        self.keep_keys = keep_keys
        self.drop_keys = drop_keys
        self.serialize_segments = serialize_segments
        self.segments_key = segments_key

    def process(self, task: AudioTask) -> DocumentBatch:
        msg = "AudioToDocumentStage only supports process_batch"
        raise NotImplementedError(msg)

    def describe(self) -> StageContract:
        return StageContract(
            reads=IOSpec(data_keys=[]),
            writes=IOSpec(data_keys=[]),
            cardinality="N:1",
            # Strips tensors/audio blobs while building the DataFrame, so its
            # output is serialization-safe — the sanctioned sink to place before
            # a JSON writer when a resident tensor may be present.
            # Packs rows into one batch for downstream throughput; the values are untouched.
            gates=Gates(sanitizes_output=True, per_row_independent=True),
            description="Aggregate AudioTasks into a DocumentBatch, stripping tensors/audio blobs (JSON/disk-safe).",
        )

    def _sanitize_nested(self, value: object) -> object:
        """Strip tensors/audio blobs from nested structures before DataFrame conversion."""
        if _is_tensor(value):
            return _DROP_VALUE
        if isinstance(value, dict):
            cleaned = {}
            for k, v in value.items():
                if k in _NON_SERIALIZABLE_KEYS:
                    continue
                nested = self._sanitize_nested(v)
                if nested is not _DROP_VALUE:
                    cleaned[k] = nested
            return cleaned
        if isinstance(value, list):
            cleaned_list = []
            for item in value:
                nested = self._sanitize_nested(item)
                if nested is not _DROP_VALUE:
                    cleaned_list.append(nested)
            return cleaned_list
        return value

    def _sanitize(self, data: dict) -> dict:
        """Remove non-serializable keys and any remaining tensor values."""
        cleaned = {}
        keys = self.keep_keys if self.keep_keys is not None else data.keys()
        for k, v in data.items():
            if k not in keys or k in self.drop_keys:
                continue
            if k in _NON_SERIALIZABLE_KEYS or k == self.segments_key:
                if k == self.segments_key and self.serialize_segments:
                    cleaned[k] = self._sanitize_nested(v)
                continue
            if _is_tensor(v):
                logger.warning(
                    f"[AudioToDocumentStage] Dropping non-serializable "
                    f"key {k!r} (torch.Tensor) before DataFrame conversion"
                )
                continue
            cleaned[k] = v
        return cleaned

    def process_batch(self, tasks: list[AudioTask]) -> list[DocumentBatch]:
        if len(tasks) == 0:
            return []
        df = pd.DataFrame([self._sanitize(t.data) for t in tasks])
        perf = []
        for t in tasks:
            perf.extend(t._stage_perf)
        return [
            DocumentBatch(
                data=df,
                dataset_name=",".join(dict.fromkeys(t.dataset_name for t in tasks)),
                _stage_perf=perf,
            )
        ]


@dataclass
class DocumentBatchJsonlWriterStage(AgentReady, ProcessingStage[DocumentBatch, DocumentBatch]):
    """Append every row in a DocumentBatch to one JSONL manifest.

    This is the task-type-compatible terminal sink for
    :class:`AudioToDocumentStage`. The output file is truncated once in
    ``setup()`` (called on the driver), while ``setup_on_node()`` only
    creates its parent directory. Within one run, successive batches append
    to the same file. The input ``DocumentBatch`` is returned unchanged so
    its dataset name, metadata, and performance records are preserved.

    Supports local and cloud paths via fsspec.

    Args:
        output_path: Destination JSONL path (local or cloud).
    """

    output_path: str
    name: str = "document_batch_jsonl_writer"

    AGENT_STATIC: ClassVar[StaticHints] = StaticHints(
        gates=Gates(
            writes_to_disk=True,
            output_path_params=["output_path"],
            lifecycle_side_effects=True,
            requires_serializable_input=True,
            per_row_independent=True,
        ),
        description="Write each DocumentBatch row to one JSONL manifest",
    )

    def __post_init__(self) -> None:
        if not self.output_path:
            msg = "output_path is required for DocumentBatchJsonlWriterStage"
            raise ValueError(msg)

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        """Truncate the output once on the driver before processing starts."""
        self._fs, self._path = url_to_fs(self.output_path)
        parent_dir = "/".join(self._path.split("/")[:-1])
        if parent_dir:
            self._fs.makedirs(parent_dir, exist_ok=True)
        with self._fs.open(self._path, "w", encoding="utf-8"):
            pass
        logger.info(f"DocumentBatchJsonlWriterStage: writing to {self.output_path}")

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        """Ensure the parent exists on each node without truncating."""
        self._fs, self._path = url_to_fs(self.output_path)
        parent_dir = "/".join(self._path.split("/")[:-1])
        if parent_dir:
            self._fs.makedirs(parent_dir, exist_ok=True)

    def process(self, task: DocumentBatch) -> DocumentBatch:
        dataframe = task.to_pandas()
        if not dataframe.empty:
            with self._fs.open(self._path, "a", encoding="utf-8") as stream:
                dataframe.to_json(
                    stream,
                    orient="records",
                    lines=True,
                    force_ascii=False,
                )
        return task

    def num_workers(self) -> int | None:
        return 1

    def describe(self) -> StageContract:
        return StageContract(
            gates=Gates(
                writes_to_disk=True,
                output_path_params=["output_path"],
                lifecycle_side_effects=True,
                requires_serializable_input=True,
                # Appends each batch's rows as they arrive; a row's line is its own contents.
                # Which lines the manifest ends up holding is a fact about the run, and a delta
                # merge rewrites exactly that. Also stated in AGENT_STATIC above -- describe()
                # was the view a delta reads, and it was the one left silent.
                per_row_independent=True,
            ),
        )
