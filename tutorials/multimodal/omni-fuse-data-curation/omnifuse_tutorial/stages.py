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

"""NeMo Curator stages: read pairs, SNS, EEE, projection training, and datablend ranking."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch

from omnifuse_tutorial.data import (
    cosine,
    expert_file,
    load_records,
    make_batch,
    raw_text,
    task_records,
    with_records,
    write_json,
    write_jsonl,
    write_npy,
)

if TYPE_CHECKING:
    from pathlib import Path

    from omnifuse_tutorial.config import ExperimentConfig

logger = logging.getLogger(__name__)


@dataclass
class _Stage(ProcessingStage[DocumentBatch, DocumentBatch]):
    config: ExperimentConfig
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []


@dataclass
class _ModelStage(_Stage):
    """Keeps its models resident across tasks; `sharded` writes per-shard outputs for the parallel step."""

    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpus=1.0))
    sharded: bool = False
    workers: int | None = None
    _engine: Any = field(default=None, init=False, repr=False)

    def num_workers(self) -> int | None:
        return self.workers

    def xenna_stage_spec(self) -> dict[str, Any]:
        return {"slots_per_actor": 1}

    def _output_dir(self, metadata: dict[str, Any], step: str) -> Path:
        directory = self.config.run_dir / step
        if not self.sharded:
            return directory
        if not isinstance(metadata.get("shard_index"), int):
            raise ValueError(f"Sharded {self.name} tasks require integer shard_index metadata")
        directory = directory / "shards" / f"{metadata['shard_index']:05d}"
        metadata[f"{step}_shard_dir"] = str(directory)
        return directory


@dataclass
class PairManifestReaderStage(_Stage):
    name = "PairManifestReader"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["pair_id", "raw_path", "annotation", "modality"]

    def process(self, task: Any) -> DocumentBatch:
        experiment = self.config.experiment_id
        records = load_records(self.config.data_pools)
        return make_batch(experiment, records, {"experiment_id": experiment})


@dataclass
class SNSStage(_ModelStage):
    name = "SNS"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["pair_id", "annotation", "modality"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["sns_annotation", "sns_raw_text"]

    def process(self, task: DocumentBatch) -> DocumentBatch:
        if self._engine is None:
            from omnifuse_tutorial.sns import SNS

            self._engine = SNS(self.config)
        results = [self._engine.process_record(record) for record in task_records(task)]
        metadata = dict(task._metadata)
        manifest = write_jsonl(self._output_dir(metadata, "sns") / "manifest.jsonl", [row for _, row in results])
        return with_records(task, [record for record, _ in results], {**metadata, "sns_manifest_path": str(manifest)})


@dataclass
class EEEEmbeddingStage(_ModelStage):
    name = "EEEEmbedding"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["pair_id", "sns_annotation", "modality"]

    def process(self, task: DocumentBatch) -> DocumentBatch:
        if self._engine is None:
            from omnifuse_tutorial.eee import EEEBackend

            self._engine = EEEBackend(self.config.eee, self.config.runtime)
        records, errors, metadata = task_records(task), [], dict(task._metadata)
        experts = list(self.config.eee.experts)
        embeddings = {
            expert: [self._embed(record, expert, side, errors) for record in records for side in ("raw", "annotation")]
            for expert in experts
        }
        directory = self._output_dir(metadata, "embeddings")
        for expert, rows in embeddings.items():
            for kind, part in (("interleaved", rows), ("raw", rows[0::2]), ("annotation", rows[1::2])):
                write_npy(expert_file(directory, expert, kind), part)
        errors_path = write_json(directory / "errors.json", errors)
        summary = {
            "pair_ids": [record["pair_id"] for record in records],
            "modalities": [record["modality"] for record in records],
            "experts": experts,
            "embedding_dim": self.config.eee.embedding_dim,
            "error_count": len(errors),
            "errors_path": str(errors_path),
        }
        metadata_path = write_json(directory / "metadata.json", summary)
        paths = {"embedding_metadata_path": str(metadata_path), "embedding_errors_path": str(errors_path)}
        return with_records(task, records, {**metadata, **paths})

    def _embed(self, record: dict[str, Any], expert: str, side: str, errors: list[dict[str, str]]) -> Any:
        """Embed one side of a pair; with `continue_on_error`, failures become zero vectors listed in errors.json."""
        try:
            return (self._engine.embed_raw if side == "raw" else self._engine.embed_annotation)(record, expert)
        except Exception as exc:
            if not self.config.eee.continue_on_error:
                raise
            pair_id, kind = str(record.get("pair_id", "unknown")), type(exc).__name__
            logger.warning("[%s] %s %s embedding failed, using zero vector: %s: %s", pair_id, expert, side, kind, exc)
            modality = str(record.get("modality", "unknown"))
            errors.append(
                {
                    "pair_id": pair_id,
                    "modality": modality,
                    "expert": expert,
                    "side": side,
                    "type": kind,
                    "message": str(exc)[:1000],
                }
            )
            return np.zeros(self.config.eee.embedding_dim)


@dataclass
class ProjectionTrainingStage(_Stage):
    name = "ProjectionTraining"

    def __post_init__(self) -> None:
        projection = self.config.projection
        self.resources = Resources(cpus=1.0, gpus=float(projection.num_gpus if projection.device != "cpu" else 0))

    def process(self, task: DocumentBatch) -> DocumentBatch:
        from omnifuse_tutorial.projection import train_projection

        embeddings, output = self.config.run_dir / "embeddings", self.config.run_dir / "projection"
        if not (embeddings / "metadata.json").exists():
            raise FileNotFoundError(f"Missing {embeddings / 'metadata.json'}. Run 2_embed.py first.")
        experts = json.loads((embeddings / "metadata.json").read_text(encoding="utf-8"))["experts"]
        interleaved = {expert: np.load(expert_file(embeddings, expert, "interleaved")) for expert in experts}
        if self.config.projection.save_weights_path is None:
            self.config.projection.save_weights_path = output / "model.pt"
        records = task_records(task)
        result = train_projection(
            self.config.projection, experts, interleaved, [str(record.get("modality", "")) for record in records]
        )
        paths = {
            "projected_embeddings_path": write_npy(output / "projected_embeddings.npy", result["projected"]),
            "annotation_embeddings_path": write_npy(output / "annotation_embeddings.npy", result["annotations"]),
            "projection_model_path": write_json(output / "model.json", result["model"]),
            "projection_loss_path": write_json(output / "loss_history.json", {"loss": result["loss"]}),
            "projection_metrics_path": write_json(output / "metrics.json", {"recall_at_10": result["recall"]}),
        }
        return with_records(task, records, {**task._metadata, **{key: str(path) for key, path in paths.items()}})


@dataclass
class DatablendRankingStage(_Stage):
    name = "DatablendRanking"

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["rank", "score"]

    def process(self, task: DocumentBatch) -> DocumentBatch:
        from omnifuse_tutorial.eee import EEEBackend

        projected_path, blend = self.config.run_dir / "projection" / "projected_embeddings.npy", self.config.datablend
        if not projected_path.exists():
            raise FileNotFoundError(f"Missing {projected_path}. Run 3_project.py first.")
        anchor = json.loads(projected_path.with_name("model.json").read_text(encoding="utf-8"))["anchor_expert"]
        query = EEEBackend(self.config.eee, self.config.runtime).embed_query(blend.query, anchor)
        ranked = []
        for index, (record, projected) in enumerate(zip(task_records(task), np.load(projected_path))):
            row = {
                "score": cosine(projected, query),
                "pair_id": record["pair_id"],
                "pool": record["pool"],
                "modality": record["modality"],
                "raw_path": record["raw_path"],
                "raw_text": raw_text(record),
                "annotation": record.get("sns_annotation") or record.get("annotation"),
                "original_annotation": record.get("annotation"),
                "source_index": index,
            }
            ranked.append({**row, "metadata": record.get("metadata", {})} if blend.include_metadata else row)
        ranked.sort(key=lambda row: row["score"], reverse=True)
        for rank, row in enumerate(ranked, start=1):
            row["rank"] = rank
        selected = ranked
        if blend.top_k is not None:
            selected = ranked[: blend.top_k]
        elif blend.blend_fraction is not None:
            selected = ranked[: max(1, int(len(ranked) * blend.blend_fraction))]
        output = self.config.run_dir / "datablend"
        paths = {
            "datablend_ranked_path": str(write_jsonl(output / "datablend_ranked.jsonl", ranked)),
            "datablend_topk_path": str(write_jsonl(output / "datablend_topk.jsonl", selected)),
        }
        return with_records(task, selected, {**task._metadata, **paths, "datablend_size": len(selected)})


def build_pipeline(config: ExperimentConfig) -> Pipeline:
    """The whole tutorial as one Curator pipeline; `.run()` returns the final task, whose metadata lists artifact paths."""
    stages = [PairManifestReaderStage, SNSStage, EEEEmbeddingStage, ProjectionTrainingStage, DatablendRankingStage]
    return Pipeline(
        name=f"omnifuse-{config.experiment_id}",
        description="Omni-Fuse SNS, EEE, projection, and datablend pipeline",
        stages=[stage(config) for stage in stages],
    )
