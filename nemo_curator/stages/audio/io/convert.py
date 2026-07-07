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

import pandas as pd
from loguru import logger

from nemo_curator.stages.audio._agent_ready import AgentReady, Gates, IOSpec, StageContract
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask, DocumentBatch

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
            gates=Gates(sanitizes_output=True),
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
