# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text-volume measurements for benchmarks with one output task per input batch."""

from dataclasses import dataclass

import pyarrow as pa

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.deduplication.id_generator import IdGeneratorBase
from nemo_curator.tasks import DocumentBatch, FileGroupTask

VOLUME_KEYS = ("num_input_documents", "num_output_documents", "input_text_bytes", "output_text_bytes")


@dataclass
class MeasureText(ProcessingStage[DocumentBatch, DocumentBatch]):
    boundary: str
    text_field: str = "text"

    def __post_init__(self):
        self.name = f"measure_{self.boundary}_text"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.text_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.text_field]

    def process(self, task: DocumentBatch) -> DocumentBatch:
        texts = task.data[self.text_field]
        if isinstance(texts, pa.ChunkedArray):
            texts = texts.to_pylist()
        text_bytes = 0
        for text in texts:
            if not isinstance(text, str):
                msg = "Text measurement requires non-null strings in the text column"
                raise TypeError(msg)
            text_bytes += len(text.encode("utf-8"))
        task._metadata = {
            **task._metadata,
            "text_volume": {
                **task._metadata.get("text_volume", {}),
                f"num_{self.boundary}_documents": task.num_items,
                f"{self.boundary}_text_bytes": text_bytes,
            },
        }
        return task


def collect_text_metrics(tasks: list[FileGroupTask], *, preserve_document_count: bool = False) -> dict[str, int]:
    if not tasks:
        msg = "No completed document batches to measure"
        raise ValueError(msg)
    metrics = dict.fromkeys(VOLUME_KEYS, 0)
    # Empty batches must remain tasks through the reader/filter/writer graph.
    for task in tasks:
        counts = task._metadata["text_volume"]
        if preserve_document_count and counts["num_input_documents"] != counts["num_output_documents"]:
            msg = "Normalization changed the document count"
            raise ValueError(msg)
        if counts["num_output_documents"] > counts["num_input_documents"]:
            msg = "Filtering expanded the document count"
            raise ValueError(msg)
        for key in metrics:
            metrics[key] += counts[key]
    return metrics


def count_registry_documents(path: str | None) -> int | None:
    """Count actually allocated document IDs, not files, from a completed workflow."""
    if path is None:
        return None
    registry = IdGeneratorBase.from_disk(path)
    count = 0
    previous_end = -1
    for start, end in sorted(registry.batch_registry.values()):
        if not isinstance(start, int) or not isinstance(end, int) or start <= previous_end or end < start - 1:
            msg = "Invalid or overlapping document ranges in ID registry"
            raise ValueError(msg)
        count += end - start + 1
        previous_end = end
    return count
