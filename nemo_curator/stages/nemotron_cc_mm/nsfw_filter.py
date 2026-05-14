"""LAION-NSFW filter for ``InterleavedBatch`` (Pattern C — GPU stage).

Wraps Curator's :class:`CLIPImageEmbeddings` and :class:`NSFWScorer` to
produce per-image NSFW probabilities, then drops image rows whose score
crosses ``max_nsfw_score`` (default 0.8, matching OmniCorpus).

Per-image policy (Phase 1): only the offending image rows are dropped;
the rest of the doc is kept.  Per-doc policy (MINT-1T / OBELICS style)
can be added later by mapping any-NSFW-in-sample to the whole sample.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd
import torch
from loguru import logger

from nemo_curator.models.clip import CLIPImageEmbeddings
from nemo_curator.models.nsfw import NSFWScorer
from nemo_curator.stages.interleaved.utils.image_utils import image_bytes_to_array
from nemo_curator.stages.resources import Resources

from nemo_curator.stages.nemotron_cc_mm.text_filters import LoggingInterleavedFilterStage

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata
    from nemo_curator.tasks import InterleavedBatch


DEFAULT_MODEL_DIR = "/home/aot/codebase/nemotron_cc_mm/data/models/curator"


@dataclass
class InterleavedNSFWFilter(LoggingInterleavedFilterStage):
    """Drop image rows whose LAION-NSFW score exceeds ``max_nsfw_score``.

    Loads CLIP (for image embeddings) and the LAION NSFW MLP on each
    worker.  Decodes image bytes with OpenCV, computes embeddings in
    GPU-bounded chunks, then thresholds.

    Parameters
    ----------
    model_dir:
        Directory used by Curator to cache the CLIP + NSFW weights.
        Will be created and populated on first run.
    max_nsfw_score:
        Drop image rows with predicted NSFW probability strictly greater
        than this value.  Default 0.8 (OmniCorpus).
    gpu_batch_size:
        Number of images per CLIP forward pass.  Default 64 keeps peak
        VRAM under ~6 GB on ViT-L/14 even with long sequences.
    """

    model_dir: str = DEFAULT_MODEL_DIR
    max_nsfw_score: float = 0.8
    gpu_batch_size: int = 64
    name: str = "interleaved_nsfw_filter"
    resources: Resources = field(default_factory=lambda: Resources(gpu_memory_gb=8.0))

    def setup_on_node(
        self, node_info: NodeInfo, worker_metadata: WorkerMetadata
    ) -> None:  # noqa: ARG002
        """Download CLIP and NSFW model weights to ``model_dir`` once per node."""
        CLIPImageEmbeddings.download_weights_on_node(self.model_dir)
        NSFWScorer.download_weights_on_node(self.model_dir)

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        """Load both models onto the GPU for this worker."""
        self._clip = CLIPImageEmbeddings(self.model_dir)
        self._clip.setup()
        self._nsfw = NSFWScorer(model_dir=self.model_dir)
        self._nsfw.setup()

    def content_keep_mask(
        self, task: InterleavedBatch, df: pd.DataFrame
    ) -> pd.Series:
        keep_mask = pd.Series(True, index=df.index, dtype=bool)
        image_mask = df["modality"] == "image"
        if not image_mask.any():
            return keep_mask

        indices: list[int] = []
        images: list = []  # list of np.ndarray RGB
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

        # GPU-bounded batching to avoid OOM on large surviving image sets.
        # CLIP ViT-L/14's full activations can be ~50–100 MB per image
        # at fp32; one shot on tens of thousands of images would OOM.
        import numpy as np
        chunk = max(1, int(self.gpu_batch_size))
        score_chunks: list = []
        with torch.no_grad():
            for start in range(0, len(images), chunk):
                batch = images[start:start + chunk]
                emb = self._clip(batch)
                scores = self._nsfw(emb)
                score_chunks.append(
                    scores.detach().to("cpu").float().numpy()
                )
                # Free intermediate activations between chunks.
                del emb, scores
                torch.cuda.empty_cache()
        scores_cpu = np.concatenate(score_chunks)

        n_dropped = 0
        for i, idx in enumerate(indices):
            score = float(scores_cpu[i])
            if score > self.max_nsfw_score:
                keep_mask.loc[idx] = False
                n_dropped += 1

        if n_dropped and self.log_drops:
            logger.info(
                f"[{self.name}] NSFW image rows: {n_dropped} dropped "
                f"out of {len(indices)} scored "
                f"(threshold > {self.max_nsfw_score})"
            )

        return keep_mask
