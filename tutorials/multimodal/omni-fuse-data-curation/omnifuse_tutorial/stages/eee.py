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

"""Expert Embedding Engine stage."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources

from omnifuse_tutorial.compat.curator import make_document_batch, records_from_task
from omnifuse_tutorial.config.models import ExperimentConfig
from omnifuse_tutorial.data.io import write_json, write_npy
from omnifuse_tutorial.eee.backends import BackendFactory, backend_factory
from omnifuse_tutorial.eee.results import EmbeddingBundle

logger = logging.getLogger(__name__)


@dataclass
class EEEEmbeddingStage(ProcessingStage[Any, Any]):
    config: ExperimentConfig | None = None
    backend_factory_fn: BackendFactory = backend_factory
    name: str = "EEEEmbedding"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpus=1.0))
    sharded_outputs: bool = False
    worker_count: int | None = None
    slots_per_actor: int = 1
    _backend: Any = field(default=None, init=False, repr=False)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["pair_id", "sns_annotation", "modality"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def xenna_stage_spec(self) -> dict[str, Any]:
        spec: dict[str, Any] = {"slots_per_actor": self.slots_per_actor}
        if self.worker_count is not None:
            spec["num_workers"] = self.worker_count
        return spec

    def process(self, task: Any) -> Any:
        if self.config is None:
            raise ValueError("EEEEmbeddingStage requires config")
        records = records_from_task(task)
        backend = self._ensure_backend()
        experts = list(self.config.eee.experts)
        embeddings: dict[str, list[list[float]]] = {}
        embedding_errors: list[dict[str, str]] = []
        for expert in experts:
            rows: list[list[float]] = []
            for record in records:
                for side in ("raw", "annotation"):
                    row, error = _embed_with_fallback(
                        backend=backend,
                        record=record,
                        expert=expert,
                        side=side,
                        embedding_dim=self.config.eee.embedding_dim,
                        continue_on_error=self.config.eee.continue_on_error,
                    )
                    rows.append(row)
                    if error is not None:
                        embedding_errors.append(error)
            embeddings[expert] = rows

        bundle = EmbeddingBundle(
            pair_ids=[record["pair_id"] for record in records],
            modalities=[record["modality"] for record in records],
            records=records,
            experts=experts,
            embeddings=embeddings,
        )

        metadata = dict(getattr(task, "_metadata", {}) or {})
        output_dir = self._output_dir(metadata)
        for expert, rows in embeddings.items():
            safe_name = expert.replace("-", "_")
            write_npy(output_dir / f"{safe_name}_interleaved.npy", rows)
            write_npy(output_dir / f"{safe_name}_raw.npy", rows[0::2])
            write_npy(output_dir / f"{safe_name}_annotation.npy", rows[1::2])
        errors_path = write_json(output_dir / "errors.json", embedding_errors)
        metadata_path = write_json(
            output_dir / "metadata.json",
            {
                "pair_ids": bundle.pair_ids,
                "modalities": bundle.modalities,
                "experts": bundle.experts,
                "embedding_dim": bundle.embedding_dim,
                "error_count": len(embedding_errors),
                "errors_path": str(errors_path),
            },
        )

        if not self.sharded_outputs:
            metadata["embedding_bundle"] = bundle
        metadata["embedding_metadata_path"] = str(metadata_path)
        metadata["embedding_errors_path"] = str(errors_path)
        return make_document_batch(
            task_id=f"{task.task_id}_eee",
            dataset_name=task.dataset_name,
            records=records,
            metadata=metadata,
            stage_perf=getattr(task, "_stage_perf", []),
        )

    def teardown(self) -> None:
        if self._backend is not None:
            unload = getattr(self._backend, "unload", None)
            if callable(unload):
                unload()
        self._backend = None

    def _ensure_backend(self) -> Any:
        if self.config is None:
            raise ValueError("EEEEmbeddingStage requires config")
        if self._backend is None:
            self._backend = self.backend_factory_fn(self.config.eee, self.config.runtime)
        return self._backend

    def _output_dir(self, metadata: dict[str, Any]) -> Any:
        if self.config is None:
            raise ValueError("EEEEmbeddingStage requires config")
        if not self.sharded_outputs:
            return self.config.run_dir / "embeddings"
        shard_index = metadata.get("shard_index")
        if not isinstance(shard_index, int):
            raise ValueError("Sharded EEE tasks require integer shard_index metadata")
        shard_dir = self.config.run_dir / "embeddings" / "shards" / f"{shard_index:05d}"
        metadata["embedding_shard_dir"] = str(shard_dir)
        return shard_dir


def _embed_with_fallback(
    *,
    backend: Any,
    record: dict[str, Any],
    expert: str,
    side: str,
    embedding_dim: int,
    continue_on_error: bool,
) -> tuple[list[float], dict[str, str] | None]:
    embed = backend.embed_raw if side == "raw" else backend.embed_annotation
    try:
        return embed(record, expert), None
    except Exception as exc:
        if not continue_on_error:
            raise
        pair_id = str(record.get("pair_id", "unknown"))
        logger.warning(
            "[%s] %s %s embedding failed, using zero vector: %s: %s",
            pair_id,
            expert,
            side,
            type(exc).__name__,
            exc,
        )
        return [0.0] * embedding_dim, {
            "pair_id": pair_id,
            "modality": str(record.get("modality", "unknown")),
            "expert": expert,
            "side": side,
            "type": type(exc).__name__,
            "message": str(exc)[:1000],
        }
