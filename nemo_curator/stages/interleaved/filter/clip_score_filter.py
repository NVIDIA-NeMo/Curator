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

from nemo_curator.models.clip import CLIPImageEmbeddings
from nemo_curator.stages.interleaved.stages import BaseInterleavedFilterStage
from nemo_curator.stages.interleaved.utils import image_bytes_to_array
from nemo_curator.stages.resources import Resources

if TYPE_CHECKING:
    import numpy as np

    from nemo_curator.backends.base import NodeInfo, WorkerMetadata
    from nemo_curator.tasks import InterleavedBatch

DEFAULT_CLIP_MIN_SCORE: float = 0.15


def _sample_texts_list_from_df(df: pd.DataFrame, sample_id: str) -> list[str]:
    """Return list of text_content from all text rows for the given sample_id (non-empty)."""
    if "text_content" not in df.columns or "modality" not in df.columns:
        return []
    subset = df[(df["sample_id"] == sample_id) & (df["modality"] == "text")]
    if subset.empty:
        return []
    return [s.strip() for s in subset["text_content"].dropna().astype(str).tolist() if s.strip()]


def _indices_and_decoded_images_from_rows(
    rows: list[tuple[int, bytes]], keep_mask: pd.Series
) -> tuple[list[int], list[np.ndarray]]:
    """Decode image bytes per row; clear keep_mask entries where decode fails."""
    indices: list[int] = []
    images: list[np.ndarray] = []
    for idx, b in rows:
        arr = image_bytes_to_array(b, row_index=idx)
        if arr is None:
            keep_mask.loc[idx] = False
            continue
        indices.append(idx)
        images.append(arr)
    return indices, images


@dataclass
class InterleavedCLIPScoreFilterStage(BaseInterleavedFilterStage):
    """Filter interleaved image rows by CLIP image-text relevance score.

    For each image row, all text rows with the same sample_id form (image, text)
    pairs. CLIP similarity is computed for each pair. An image is kept only if at
    least one pair has score >= min_score; otherwise it is dropped.
    """

    model_dir: str | None = None
    min_score: float = DEFAULT_CLIP_MIN_SCORE
    name: str = "interleaved_clip_score_filter"
    resources: Resources = field(default_factory=lambda: Resources(gpu_memory_gb=20.0))

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        self._model = CLIPImageEmbeddings(self.model_dir)
        self._model.setup()

    def setup_on_node(self, node_info: NodeInfo, worker_metadata: WorkerMetadata) -> None:  # noqa: ARG002
        """Download the weights for the CLIP model on the node."""
        if self.model_dir is None:
            msg = "InterleavedCLIPScoreFilterStage requires model_dir to be set"
            raise RuntimeError(msg)
        CLIPImageEmbeddings.download_weights_on_node(self.model_dir)

    def content_keep_mask(self, task: InterleavedBatch, df: pd.DataFrame) -> pd.Series:
        keep_mask = pd.Series(True, index=df.index, dtype=bool)
        image_mask = df["modality"] == "image"
        if not image_mask.any():
            return keep_mask

        sample_id_to_rows: dict[str, list[tuple[int, bytes]]] = {}
        for idx, image_bytes in self.iter_materialized_bytes(task=task, df=df, row_mask=image_mask):
            if image_bytes is None:
                keep_mask.loc[idx] = False
                continue
            sample_id = df.loc[idx, "sample_id"]
            sample_id_to_rows.setdefault(sample_id, []).append((idx, image_bytes))

        for sample_id, rows in sample_id_to_rows.items():
            texts = _sample_texts_list_from_df(df, sample_id)
            if not texts:
                for idx, _ in rows:
                    keep_mask.loc[idx] = False
                continue
            indices, images = _indices_and_decoded_images_from_rows(rows, keep_mask)
            if not images:
                continue
            img_emb = self._model(images)
            text_emb = self._model.encode_text(texts)
            scores = img_emb @ text_emb.T
            for i, idx in enumerate(indices):
                keep_mask.loc[idx] = (scores[i].max() >= self.min_score).item()

        return keep_mask
