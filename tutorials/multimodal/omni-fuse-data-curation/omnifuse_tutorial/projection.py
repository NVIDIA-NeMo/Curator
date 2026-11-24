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

"""Projection training: map concatenated expert embeddings of raw data into the annotation embedding space."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    import torch

    from omnifuse_tutorial.config import ProjectionConfig

LEARNING_RATE = 1e-3
CONTRASTIVE_LOSS_WEIGHT = 0.99
BIAS_LOSS_WEIGHT = 0.01
SCALE_LOSS_WEIGHT = 0.0001
CONTRASTIVE_TEMPERATURE = 0.07
NUM_LAYERS = 3
DROPOUT = 0.1
EVAL_RECALL_K = 10
LOG_EVERY_N_EPOCHS = 100


def train_projection(
    config: ProjectionConfig,
    experts: list[str],
    interleaved: dict[str, np.ndarray],
    modalities: list[str],
    weights_path: Path | None = None,
) -> dict[str, Any]:
    """MLP on per-expert interleaved (raw, annotation) rows: symmetric contrastive plus modality-bias losses."""
    import torch
    from torch.nn import functional

    if not modalities:
        msg = "Cannot train projection on an empty bundle"
        raise ValueError(msg)
    anchor = "text-based" if "text-based" in experts else experts[0]
    torch.manual_seed(0)
    device, gpu_count = _torch_device(config)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
    raw = np.concatenate([interleaved[expert][0::2] for expert in experts], axis=1)
    inputs = torch.tensor(raw, dtype=torch.float32)
    targets = torch.tensor(interleaved[anchor][1::2], dtype=torch.float32)
    layers, width = [], inputs.shape[1]
    for _ in range(max(0, NUM_LAYERS - 1)):
        layers += [torch.nn.Linear(width, config.hidden_layer_size), torch.nn.ReLU(), torch.nn.Dropout(DROPOUT)]
        width = config.hidden_layer_size
    base_model = torch.nn.Sequential(*layers, torch.nn.Linear(width, targets.shape[1])).to(device)
    model = torch.nn.DataParallel(base_model) if gpu_count > 1 else base_model
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    batch_size, epochs, count = max(1, min(config.batch_size, len(inputs))), max(1, config.num_epochs), len(inputs)
    print(
        f"Projection training: {count} records, {epochs} epochs, batch size {batch_size}, device {device}, GPUs {gpu_count}",
        flush=True,
    )
    started, losses = time.monotonic(), []
    for epoch in range(epochs):
        permutation, epoch_losses = torch.randperm(count), []
        for start in range(0, count, batch_size):
            index = permutation[start : start + batch_size]
            projected = model(inputs[index].to(device, non_blocking=True))
            batch_targets = targets[index].to(device, non_blocking=True)
            batch_modalities = [modalities[int(item)] for item in index]
            loss = (
                CONTRASTIVE_LOSS_WEIGHT * _contrastive_loss(projected, batch_targets, CONTRASTIVE_TEMPERATURE)
                + BIAS_LOSS_WEIGHT * _bias_loss(projected, batch_modalities, scale=False)
                + SCALE_LOSS_WEIGHT * _bias_loss(projected, batch_modalities, scale=True)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(sum(epoch_losses) / len(epoch_losses))
        if epoch == 0 or (epoch + 1) % LOG_EVERY_N_EPOCHS == 0 or epoch + 1 == epochs:
            elapsed = time.monotonic() - started
            print(f"Projection epoch {epoch + 1}/{epochs}: loss={losses[-1]:.6f}, elapsed={elapsed:.1f}s", flush=True)
    model.eval()
    with torch.no_grad():
        projected = torch.cat([model(inputs[i : i + batch_size].to(device)) for i in range(0, count, batch_size)])
        projected, annotations = (
            functional.normalize(projected, dim=1),
            functional.normalize(targets.to(device), dim=1),
        )
        recall = _recall_torch(annotations, projected, k=min(EVAL_RECALL_K, count), chunk=batch_size)
    state_dict_path = None
    if weights_path:
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(base_model.state_dict(), weights_path)
        state_dict_path = str(weights_path)
    model_info = {
        "type": "torch_mlp_projection",
        "experts": experts,
        "input_dim": int(inputs.shape[1]),
        "embedding_dim": int(interleaved[experts[0]].shape[1]),
        "hidden_layer_size": config.hidden_layer_size,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "anchor_expert": anchor,
        "contrastive_loss_weight": CONTRASTIVE_LOSS_WEIGHT,
        "bias_loss_weight": BIAS_LOSS_WEIGHT,
        "scale_loss_weight": SCALE_LOSS_WEIGHT,
        "contrastive_temperature": CONTRASTIVE_TEMPERATURE,
        "device": str(device),
        "num_gpus": gpu_count,
        "parallelism": "data_parallel" if gpu_count > 1 else "single_device",
        "state_dict_path": state_dict_path,
    }
    return {
        "projected": projected.cpu().numpy(),
        "annotations": annotations.cpu().numpy(),
        "model": model_info,
        "loss": losses,
        "recall": recall,
    }


def _torch_device(config: ProjectionConfig) -> tuple[torch.device, int]:
    """Use the GPUs Curator assigned to this worker (exposed through CUDA_VISIBLE_DEVICES)."""
    import torch

    if config.num_gpus == 0:
        return torch.device("cpu"), 0
    if not torch.cuda.is_available():
        msg = "Projection requested GPU training (num_gpus >= 1), but no CUDA device is visible"
        raise RuntimeError(msg)
    return torch.device("cuda:0"), min(config.num_gpus, torch.cuda.device_count())


def _contrastive_loss(projected: torch.Tensor, anchors: torch.Tensor, temperature: float) -> torch.Tensor:
    import torch
    from torch.nn import functional

    logits = functional.normalize(projected, dim=1) @ functional.normalize(anchors, dim=1).T / temperature
    labels = torch.arange(projected.shape[0], device=projected.device)
    return 0.5 * (functional.cross_entropy(logits, labels) + functional.cross_entropy(logits.T, labels))


def _bias_loss(projected: torch.Tensor, modalities: list[str], scale: bool) -> torch.Tensor:
    """Penalize per-modality centroid offsets (scale=False) or spread differences (scale=True) in a batch."""
    import torch

    if len(set(modalities)) <= 1:
        return projected.new_tensor(0.0)
    overall = projected.mean(dim=0)
    overall_spread = torch.norm(projected - overall, dim=1).mean() if scale else None
    losses = []
    for modality in sorted(set(modalities)):
        rows = projected[
            torch.tensor([i for i, value in enumerate(modalities) if value == modality], device=projected.device)
        ]
        centroid = rows.mean(dim=0)
        if scale:
            losses.append((torch.norm(rows - centroid, dim=1).mean() - overall_spread) ** 2)
        else:
            losses.append(torch.mean((centroid - overall) ** 2))
    return torch.stack(losses).mean()


def _recall_torch(annotations: torch.Tensor, projected: torch.Tensor, k: int, chunk: int) -> dict[str, float]:
    """Retrieval recall@k between matching rows, scoring `chunk` queries at a time so memory stays O(chunk x N)."""
    import torch

    def recall(queries: torch.Tensor, keys: torch.Tensor) -> float:
        hits = 0
        for start in range(0, len(queries), chunk):
            top = (queries[start : start + chunk] @ keys.T).topk(k, dim=1).indices
            targets = torch.arange(start, start + len(top), device=top.device).unsqueeze(1)
            hits += int((top == targets).any(dim=1).sum())
        return hits / len(queries)

    a2r, r2a = recall(annotations, projected), recall(projected, annotations)
    return {"annotation_to_raw": a2r, "raw_to_annotation": r2a, "average": (a2r + r2a) / 2.0}
