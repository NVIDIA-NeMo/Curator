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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from nemo_curator.models.clip import CLIPImageEmbeddings
from nemo_curator.stages.interleaved.stages import BaseInterleavedScoreFilterStage
from nemo_curator.stages.interleaved.utils import image_bytes_to_array
from nemo_curator.stages.resources import Resources

if TYPE_CHECKING:
    import numpy as np

    from nemo_curator.backends.base import NodeInfo, WorkerMetadata
    from nemo_curator.tasks import InterleavedBatch

DEFAULT_CLIP_MIN_SCORE: float = 0.15


def _sample_text_positions_and_texts(df: pd.DataFrame, sample_id: str | int) -> tuple[list[int], list[str]]:
    """Text ``position`` and stripped non-empty ``text_content``, same row order."""
    if "text_content" not in df.columns or "modality" not in df.columns or "position" not in df.columns:
        return [], []
    subset = df[(df["sample_id"] == sample_id) & (df["modality"] == "text")]
    if subset.empty:
        return [], []
    positions: list[int] = []
    contents: list[str] = []
    for _, row in subset.iterrows():
        raw = row["text_content"]
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            continue
        s = str(raw).strip()
        if not s:
            continue
        pos = row["position"]
        if pd.isna(pos):
            continue
        positions.append(int(pos))
        contents.append(s)
    return positions, contents


def _indices_and_decoded_images_from_rows(
    rows: list[tuple[int, bytes]], row_ok: pd.Series
) -> tuple[list[int], list[np.ndarray]]:
    """Decode image bytes per row; clear ``row_ok`` entries where decode fails."""
    indices: list[int] = []
    images: list[np.ndarray] = []
    for idx, b in rows:
        arr = image_bytes_to_array(b, row_index=idx)
        if arr is None:
            row_ok.loc[idx] = False
            continue
        indices.append(idx)
        images.append(arr)
    return indices, images


@dataclass
class InterleavedCLIPScoreAnnotatorStage(BaseInterleavedScoreFilterStage):
    """Add CLIP similarity dicts per image row as ``{name}_clip_scores``.

    For each image row, all text rows with the same ``sample_id`` form (image, text) pairs.
    The stored dict maps each text row's interleaved ``position`` to the CLIP similarity.
    Non-image rows and images with no text in the sample use ``<NA>``.
    """

    model_dir: str | None = None
    name: str = "interleaved_clip_score_annotator"
    resources: Resources = field(default_factory=lambda: Resources(gpu_memory_gb=20.0))

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        self._model = CLIPImageEmbeddings(self.model_dir)
        self._model.setup()

    def setup_on_node(self, node_info: NodeInfo, worker_metadata: WorkerMetadata) -> None:  # noqa: ARG002
        """Download the weights for the CLIP model on the node."""
        if self.model_dir is None:
            msg = "InterleavedCLIPScoreAnnotatorStage requires model_dir to be set"
            raise RuntimeError(msg)
        CLIPImageEmbeddings.download_weights_on_node(self.model_dir)

    def _clip_score_dicts_series(self, task: InterleavedBatch, df: pd.DataFrame) -> pd.Series:  # noqa: C901
        out = pd.Series(pd.NA, index=df.index, dtype=object)
        image_mask = df["modality"] == "image"
        if not image_mask.any():
            return out
        sample_id_to_rows: dict[str | int, list[tuple[int, bytes]]] = {}
        for idx, image_bytes in self.iter_materialized_bytes(task=task, df=df, row_mask=image_mask):
            if image_bytes is None:
                continue
            sample_id = df.loc[idx, "sample_id"]
            sample_id_to_rows.setdefault(sample_id, []).append((idx, image_bytes))
        decode_ok = pd.Series(True, index=df.index, dtype=bool)
        for sample_id, rows in sample_id_to_rows.items():
            text_positions, texts = _sample_text_positions_and_texts(df, sample_id)
            if not texts:
                for idx, _ in rows:
                    out.at[idx] = pd.NA  # noqa: PD008
                continue
            indices, images = _indices_and_decoded_images_from_rows(rows, decode_ok)
            if not images:
                continue
            try:
                img_emb = self._model(images)
            except Exception as e:  # noqa: BLE001
                logger.debug(
                    "CLIP score computation failed (indices={}): {}",
                    indices,
                    e,
                )
                continue
            text_emb = self._model.encode_text(texts)
            sim = img_emb @ text_emb.T
            num_t = len(text_positions)
            for i, idx in enumerate(indices):
                d: dict[int, float] = {}
                for j in range(num_t):
                    cell = sim[i, j]
                    d[text_positions[j]] = float(cell.item()) if hasattr(cell, "item") else float(cell)
                out.at[idx] = {int(k): float(v) for k, v in d.items()}  # noqa: PD008
        return out

    def annotation_columns(self, task: InterleavedBatch, df: pd.DataFrame) -> dict[str, pd.Series]:
        return {f"{self.name}_clip_scores": self._clip_score_dicts_series(task, df)}
