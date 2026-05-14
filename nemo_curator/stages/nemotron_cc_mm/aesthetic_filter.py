"""LAION-aesthetic filter for ``InterleavedBatch`` (Pattern C — GPU stage).

Wraps Curator's :class:`CLIPImageEmbeddings` and :class:`AestheticScorer`
(LAION's ``sac+logos+ava1-l14-linearMSE`` MLP on CLIP-ViT-L/14 features —
the same model OmniCorpus cites) to score every image and drop rows that
fall below ``min_aesthetic_score``.  Default 3.7 matches OmniCorpus §3.1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd
import torch
from loguru import logger

from nemo_curator.models.aesthetics import AestheticScorer
from nemo_curator.models.clip import CLIPImageEmbeddings
from nemo_curator.stages.interleaved.utils.image_utils import image_bytes_to_array
from nemo_curator.stages.nemotron_cc_mm.text_filters import LoggingInterleavedFilterStage
from nemo_curator.stages.resources import Resources

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata
    from nemo_curator.tasks import InterleavedBatch


# Reuse the same cache dir as the NSFW filter — they share the CLIP weights.
DEFAULT_MODEL_DIR = "/home/aot/codebase/nemotron_cc_mm/data/models/curator"


@dataclass
class InterleavedAestheticFilter(LoggingInterleavedFilterStage):
    """Drop image rows whose LAION aesthetic score falls below the threshold.

    Parameters
    ----------
    model_dir:
        Local directory used to cache CLIP + aesthetic-scorer weights.
        Populated on first run via Hugging Face.
    min_aesthetic_score:
        Drop image rows with predicted aesthetic score strictly less than
        this value.  Default 3.7 (OmniCorpus §3.1, LAION-5B [88]).
    gpu_batch_size:
        Number of images per CLIP forward pass.  Default 64 keeps peak
        VRAM under ~6 GB on ViT-L/14.
    """

    model_dir: str = DEFAULT_MODEL_DIR
    min_aesthetic_score: float = 3.7
    gpu_batch_size: int = 64
    name: str = "interleaved_aesthetic_filter"
    resources: Resources = field(default_factory=lambda: Resources(gpu_memory_gb=8.0))

    def setup_on_node(
        self, node_info: NodeInfo, worker_metadata: WorkerMetadata
    ) -> None:  # noqa: ARG002
        """Download CLIP + aesthetic-scorer weights to ``model_dir`` once per node."""
        CLIPImageEmbeddings.download_weights_on_node(self.model_dir)
        AestheticScorer.download_weights_on_node(self.model_dir)

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        """Load CLIP + the aesthetic MLP onto the GPU for this worker."""
        self._clip = CLIPImageEmbeddings(self.model_dir)
        self._clip.setup()
        self._aesthetic = AestheticScorer(model_dir=self.model_dir)
        self._aesthetic.setup()

    def content_keep_mask(
        self, task: InterleavedBatch, df: pd.DataFrame
    ) -> pd.Series:
        keep_mask = pd.Series(True, index=df.index, dtype=bool)
        image_mask = df["modality"] == "image"
        if not image_mask.any():
            return keep_mask

        indices: list[int] = []
        images: list = []
        for idx, image_bytes in self.iter_materialized_bytes(
            task=task, df=df, row_mask=image_mask
        ):
            if image_bytes is None:
                keep_mask.loc[idx] = False
                continue
            arr = image_bytes_to_array(image_bytes, row_index=idx)
            if arr is None:
                keep_mask.loc[idx] = False
                continue
            indices.append(idx)
            images.append(arr)

        if not images:
            return keep_mask

        import numpy as np
        chunk = max(1, int(self.gpu_batch_size))
        score_chunks: list = []
        with torch.no_grad():
            for start in range(0, len(images), chunk):
                batch = images[start:start + chunk]
                emb = self._clip(batch)
                scores = self._aesthetic(emb)
                score_chunks.append(
                    scores.detach().to("cpu").float().numpy()
                )
                del emb, scores
                torch.cuda.empty_cache()
        scores_cpu = np.concatenate(score_chunks)

        n_dropped = 0
        for i, idx in enumerate(indices):
            score = float(scores_cpu[i])
            if score < self.min_aesthetic_score:
                keep_mask.loc[idx] = False
                n_dropped += 1

        if n_dropped and self.log_drops:
            logger.info(
                f"[{self.name}] low-aesthetic image rows: {n_dropped} dropped "
                f"out of {len(indices)} scored "
                f"(threshold < {self.min_aesthetic_score})"
            )

        return keep_mask
