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

"""Projection training and inference."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from omnifuse_tutorial.config.models import ProjectionConfig
from omnifuse_tutorial.data.io import cosine_similarity
from omnifuse_tutorial.eee.results import EmbeddingBundle


@dataclass
class ProjectionResult:
    projected_raw: list[list[float]]
    annotation_embeddings: list[list[float]]
    expert_weights: dict[str, float]
    loss_history: list[float]
    recall_at_10: dict[str, float]
    model: dict[str, Any]


class ProjectionTrainer:
    def __init__(self, config: ProjectionConfig):
        self.config = config

    def train_and_project(self, bundle: EmbeddingBundle) -> ProjectionResult:
        if not bundle.records:
            raise ValueError("Cannot train projection on an empty bundle")
        if self.config.backend == "torch" or (self.config.backend == "auto" and _torch_available()):
            return self._train_torch_projection(bundle)
        return self._train_linear_projection(bundle)

    def _train_linear_projection(self, bundle: EmbeddingBundle) -> ProjectionResult:
        expert_scores = {
            expert: _mean_pair_similarity(bundle.raw_embeddings(expert), bundle.annotation_embeddings(expert))
            for expert in bundle.experts
        }
        weights = _softmax(expert_scores)
        projected = _weighted_sum(
            [bundle.raw_embeddings(expert) for expert in bundle.experts], [weights[e] for e in bundle.experts]
        )

        anchor_expert = "text-based" if "text-based" in bundle.experts else bundle.experts[0]
        annotations = bundle.annotation_embeddings(anchor_expert)
        loss = _contrastive_loss(projected, annotations, self.config.contrastive_temperature)
        recall = _recall_at_k(projected, annotations, k=min(10, len(projected)))
        model = {
            "type": "linear_expert_projection",
            "experts": bundle.experts,
            "expert_weights": weights,
            "embedding_dim": bundle.embedding_dim,
            "anchor_expert": anchor_expert,
        }
        return ProjectionResult(
            projected_raw=projected,
            annotation_embeddings=annotations,
            expert_weights=weights,
            loss_history=[loss],
            recall_at_10=recall,
            model=model,
        )

    def _train_torch_projection(self, bundle: EmbeddingBundle) -> ProjectionResult:
        import torch
        import torch.nn.functional as F

        torch.manual_seed(0)
        raw_inputs = torch.tensor(_concat_raw_expert_embeddings(bundle), dtype=torch.float32)
        anchor_expert = "text-based" if "text-based" in bundle.experts else bundle.experts[0]
        anchors = torch.tensor(bundle.annotation_embeddings(anchor_expert), dtype=torch.float32)
        modalities = [str(record.get("modality", "")) for record in bundle.records]

        model = _ProjectionMLP(
            input_dim=raw_inputs.shape[1],
            output_dim=anchors.shape[1],
            hidden_dim=self.config.hidden_layer_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        batch_size = max(1, min(self.config.batch_size, raw_inputs.shape[0]))
        loss_history: list[float] = []
        epochs = max(1, self.config.num_epochs)

        for _epoch in range(epochs):
            permutation = torch.randperm(raw_inputs.shape[0])
            epoch_losses: list[float] = []
            for start in range(0, raw_inputs.shape[0], batch_size):
                batch_idx = permutation[start : start + batch_size]
                batch_raw = raw_inputs[batch_idx]
                batch_anchors = anchors[batch_idx]
                batch_modalities = [modalities[int(idx)] for idx in batch_idx]

                projected = model(batch_raw)
                task_loss = _torch_contrastive_loss(projected, batch_anchors, self.config.contrastive_temperature)
                cluster_loss = _torch_cluster_bias_loss(projected, batch_modalities)
                scale_loss = _torch_scale_bias_loss(projected, batch_modalities)
                loss = (
                    self.config.contrastive_loss_weight * task_loss
                    + self.config.bias_loss_weight * cluster_loss
                    + self.config.scale_loss_weight * scale_loss
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
            loss_history.append(sum(epoch_losses) / len(epoch_losses))

        model.eval()
        with torch.no_grad():
            projected_tensor = F.normalize(model(raw_inputs), dim=1)
        projected = projected_tensor.cpu().tolist()
        annotations = F.normalize(anchors, dim=1).cpu().tolist()
        recall = _recall_at_k(projected, annotations, k=min(self.config.eval_recall_k, len(projected)))

        state_dict_path = None
        if self.config.save_weights_path:
            self.config.save_weights_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), self.config.save_weights_path)
            state_dict_path = str(self.config.save_weights_path)

        equal_weight = 1.0 / len(bundle.experts)
        metadata = {
            "type": "torch_mlp_projection",
            "experts": bundle.experts,
            "input_dim": int(raw_inputs.shape[1]),
            "embedding_dim": bundle.embedding_dim,
            "hidden_layer_size": self.config.hidden_layer_size,
            "num_layers": self.config.num_layers,
            "dropout": self.config.dropout,
            "anchor_expert": anchor_expert,
            "contrastive_loss_weight": self.config.contrastive_loss_weight,
            "bias_loss_weight": self.config.bias_loss_weight,
            "scale_loss_weight": self.config.scale_loss_weight,
            "contrastive_temperature": self.config.contrastive_temperature,
            "state_dict_path": state_dict_path,
        }
        return ProjectionResult(
            projected_raw=projected,
            annotation_embeddings=annotations,
            expert_weights=dict.fromkeys(bundle.experts, equal_weight),
            loss_history=loss_history,
            recall_at_10=recall,
            model=metadata,
        )


class _ProjectionMLP:
    def __new__(
        cls,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ):
        from torch import nn

        layers: list[nn.Module] = []
        current_dim = input_dim
        hidden_layers = max(0, num_layers - 1)
        for _ in range(hidden_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        return nn.Sequential(*layers)


def _mean_pair_similarity(raw: list[list[float]], annotations: list[list[float]]) -> float:
    if not raw or not annotations:
        return 0.0
    scores = [cosine_similarity(left, right) for left, right in zip(raw, annotations)]
    return sum(scores) / len(scores)


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    max_score = max(scores.values())
    exp_scores = {name: math.exp(score - max_score) for name, score in scores.items()}
    total = sum(exp_scores.values())
    if total == 0:
        equal = 1.0 / len(scores)
        return dict.fromkeys(scores, equal)
    return {name: value / total for name, value in exp_scores.items()}


def _weighted_sum(expert_matrices: list[list[list[float]]], weights: list[float]) -> list[list[float]]:
    if not expert_matrices:
        return []
    n_rows = len(expert_matrices[0])
    n_cols = len(expert_matrices[0][0]) if n_rows else 0
    output: list[list[float]] = []
    for row_idx in range(n_rows):
        row = [0.0] * n_cols
        for matrix, weight in zip(expert_matrices, weights):
            for col_idx, value in enumerate(matrix[row_idx]):
                row[col_idx] += weight * value
        output.append(row)
    return output


def _contrastive_loss(raw: list[list[float]], annotations: list[list[float]], temperature: float) -> float:
    if not raw:
        return 0.0
    losses: list[float] = []
    for idx, raw_vec in enumerate(raw):
        logits = [cosine_similarity(raw_vec, ann_vec) / temperature for ann_vec in annotations]
        max_logit = max(logits)
        log_sum_exp = max_logit + math.log(sum(math.exp(item - max_logit) for item in logits))
        losses.append(-logits[idx] + log_sum_exp)
    return sum(losses) / len(losses)


def _recall_at_k(raw: list[list[float]], annotations: list[list[float]], k: int) -> dict[str, float]:
    if not raw or not annotations:
        return {"annotation_to_raw": 0.0, "raw_to_annotation": 0.0, "average": 0.0}
    a2r_hits = 0
    for idx, ann_vec in enumerate(annotations):
        ranked = sorted(range(len(raw)), key=lambda raw_idx: cosine_similarity(ann_vec, raw[raw_idx]), reverse=True)
        if idx in ranked[:k]:
            a2r_hits += 1
    r2a_hits = 0
    for idx, raw_vec in enumerate(raw):
        ranked = sorted(
            range(len(annotations)), key=lambda ann_idx: cosine_similarity(raw_vec, annotations[ann_idx]), reverse=True
        )
        if idx in ranked[:k]:
            r2a_hits += 1
    a2r = a2r_hits / len(annotations)
    r2a = r2a_hits / len(raw)
    return {"annotation_to_raw": a2r, "raw_to_annotation": r2a, "average": (a2r + r2a) / 2.0}


def _concat_raw_expert_embeddings(bundle: EmbeddingBundle) -> list[list[float]]:
    rows: list[list[float]] = []
    expert_matrices = [bundle.raw_embeddings(expert) for expert in bundle.experts]
    for row_idx in range(len(bundle.records)):
        row: list[float] = []
        for matrix in expert_matrices:
            row.extend(matrix[row_idx])
        rows.append(row)
    return rows


def _torch_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return True


def _torch_contrastive_loss(projected: Any, anchors: Any, temperature: float) -> Any:
    import torch
    import torch.nn.functional as F

    projected = F.normalize(projected, dim=1)
    anchors = F.normalize(anchors, dim=1)
    logits = projected @ anchors.T / temperature
    labels = torch.arange(projected.shape[0], device=projected.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def _torch_cluster_bias_loss(projected: Any, modalities: list[str]) -> Any:
    import torch

    if len(set(modalities)) <= 1:
        return projected.new_tensor(0.0)
    overall = projected.mean(dim=0)
    losses = []
    for modality in sorted(set(modalities)):
        indices = [idx for idx, value in enumerate(modalities) if value == modality]
        centroid = projected[torch.tensor(indices, device=projected.device)].mean(dim=0)
        losses.append(torch.mean((centroid - overall) ** 2))
    return torch.stack(losses).mean()


def _torch_scale_bias_loss(projected: Any, modalities: list[str]) -> Any:
    import torch

    if len(set(modalities)) <= 1:
        return projected.new_tensor(0.0)
    overall = projected.mean(dim=0)
    overall_spread = torch.norm(projected - overall, dim=1).mean()
    losses = []
    for modality in sorted(set(modalities)):
        indices = [idx for idx, value in enumerate(modalities) if value == modality]
        modality_rows = projected[torch.tensor(indices, device=projected.device)]
        centroid = modality_rows.mean(dim=0)
        spread = torch.norm(modality_rows - centroid, dim=1).mean()
        losses.append((spread - overall_spread) ** 2)
    return torch.stack(losses).mean()
