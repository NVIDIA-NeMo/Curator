"""Image-byte filters for ``InterleavedBatch``.

Provides:
    * :class:`InterleavedGeometryFilter`   — drop image rows by pixel dims.
    * :class:`InterleavedImageCountFilter` — drop docs by image-row count.

Aspect-ratio filtering lives in Curator's
:class:`InterleavedAspectRatioFilterStage` and is composed via the
launcher directly; we do not reimplement it here.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from nemo_curator.stages.nemotron_cc_mm.text_filters import (
    BaseInterleavedSampleFilterStage,
    LoggingInterleavedFilterStage,
)

if TYPE_CHECKING:
    from nemo_curator.tasks import InterleavedBatch

try:
    from PIL import Image
    # Raise the default decompression-bomb ceiling.  Default is ~178 M
    # pixels; we accept up to ~500 M (legit hi-res images stay well
    # below this; anything above is dropped by the geometry bounds
    # anyway, so this just stops PIL from raising on dimension reads).
    Image.MAX_IMAGE_PIXELS = 500_000_000
except ImportError:
    Image = None  # type: ignore[assignment]


@dataclass
class InterleavedGeometryFilter(LoggingInterleavedFilterStage):
    """Drop image rows whose dimensions fall outside the configured bounds.

    Default ``[150, 20 000]`` per dimension (MINT-1T §2.3).  Images
    without ``binary_content`` (e.g. download failed earlier) are
    dropped — they have no dimensions we can verify.
    """

    min_width: int = 150
    min_height: int = 150
    max_width: int = 20_000
    max_height: int = 20_000
    name: str = "interleaved_geometry_filter"

    @staticmethod
    def _image_dims(image_bytes: bytes) -> tuple[int, int] | None:
        if Image is None:
            msg = (
                "Pillow is required for InterleavedGeometryFilter. "
                "Install dependency group `image_cpu`."
            )
            raise RuntimeError(msg)
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                return img.size  # (width, height)
        except Exception:  # noqa: BLE001
            # PIL raises many distinct exception types on bad input
            # (OSError, SyntaxError, ValueError, DecompressionBombError,
            # UnidentifiedImageError, …).  Any of them → drop this
            # image; never let one malformed file kill the batch.
            return None

    def content_keep_mask(
        self, task: InterleavedBatch, df: pd.DataFrame
    ) -> pd.Series:
        keep_mask = pd.Series(True, index=df.index, dtype=bool)
        image_mask = df["modality"] == "image"
        if not image_mask.any():
            return keep_mask
        for idx, image_bytes in self.iter_materialized_bytes(
            task=task, df=df, row_mask=image_mask
        ):
            if image_bytes is None:
                keep_mask.loc[idx] = False
                continue
            dims = self._image_dims(image_bytes)
            if dims is None:
                keep_mask.loc[idx] = False
                continue
            w, h = dims
            if not (
                self.min_width <= w <= self.max_width
                and self.min_height <= h <= self.max_height
            ):
                keep_mask.loc[idx] = False
        return keep_mask


@dataclass
class InterleavedImageCountFilter(BaseInterleavedSampleFilterStage):
    """Drop docs whose surviving image-row count is outside ``[min, max]``.

    Run this AFTER the image acquire/filter chain — by then each doc's
    image rows reflect what actually passed download + geometry + NSFW.
    Default ``[1, 30]``: text-only docs are dropped (we're building an
    interleaved dataset), and image-stuffed pages (sliders, galleries)
    are capped to avoid wildly skewing the corpus.
    """

    min_images: int = 1
    max_images: int = 30
    name: str = "interleaved_image_count_filter"

    def is_sample_ok(self, sample_id: str, group: pd.DataFrame) -> bool:
        n = int((group["modality"] == "image").sum())
        return self.min_images <= n <= self.max_images
