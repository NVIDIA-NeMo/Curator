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

import pandas as pd

from nemo_curator.models.clip import CLIPImageEmbeddings
from nemo_curator.stages.multimodal.filter.blur_filter import _image_bytes_to_array
from nemo_curator.stages.multimodal.stages import BaseMultimodalFilterStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import MultiBatchTask


def _sample_texts_list_from_df(df: pd.DataFrame, sample_id: str) -> list[str]:
    """Return list of text_content from all text rows for the given sample_id (non-empty)."""
    if "text_content" not in df.columns or "modality" not in df.columns:
        return []
    subset = df[(df["sample_id"] == sample_id) & (df["modality"] == "text")]
    if subset.empty:
        return []
    return [s.strip() for s in subset["text_content"].dropna().astype(str).tolist() if s.strip()]


@dataclass
class MultimodalCLIPScoreFilterStage(BaseMultimodalFilterStage):
    """Filter multimodal image rows by CLIP image-text relevance score.

    For each image row, all text rows with the same sample_id form (image, text)
    pairs. CLIP similarity is computed for each pair. An image is kept only if at
    least one pair has score >= min_score; otherwise it is dropped.
    """

    model_dir: str | None = None
    min_score: float = 0.15
    image_content_types: tuple[str, ...] = ("image/jpeg", "image/jpg", "image/png")
    name: str = "multimodal_clip_score_filter"

    def __post_init__(self) -> None:
        self._model: CLIPImageEmbeddings | None = None
        self.resources = Resources(gpus=0.25)

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if self.model_dir is None:
            msg = "MultimodalCLIPScoreFilterStage requires model_dir to be set"
            raise RuntimeError(msg)
        CLIPImageEmbeddings.download_weights_on_node(self.model_dir)
        self._model = CLIPImageEmbeddings(self.model_dir)
        self._model.setup()

    def content_keep_mask(self, task: MultiBatchTask, df: pd.DataFrame) -> pd.Series:
        keep_mask = pd.Series(True, index=df.index, dtype=bool)
        image_mask = (df["modality"] == "image") & (df["content_type"].isin(self.image_content_types))
        if not image_mask.any():
            return keep_mask

        self._ensure_model()
        assert self._model is not None

        for idx, image_bytes in self.iter_materialized_bytes(task=task, df=df, row_mask=image_mask):
            if image_bytes is None:
                keep_mask.loc[idx] = False
                continue
            image = _image_bytes_to_array(image_bytes)
            if image is None:
                keep_mask.loc[idx] = False
                continue
            sample_id = df.loc[idx, "sample_id"]
            texts = _sample_texts_list_from_df(df, sample_id)
            if not texts:
                keep_mask.loc[idx] = False
                continue
            img_emb = self._model([image])
            text_emb = self._model.encode_text(texts)
            scores = img_emb @ text_emb.T
            from loguru import logger

            logger.info(f"scores: {scores}")
            max_score = scores.max()
            keep_mask.loc[idx] = (max_score >= self.min_score).item()
        return keep_mask
