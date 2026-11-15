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

import io
from dataclasses import dataclass

import torch
from PIL import Image
from torchvision import transforms

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.image.diagnostics import decode_image_rgb
from nemo_curator.tasks import ImageBatch


@dataclass
class ImageDecodeStage(ProcessingStage[ImageBatch, ImageBatch]):
    """Decode ``image_bytes`` into native-resolution RGB ``image_data`` on CPU.

    Uses :func:`decode_image_rgb` (frame zero, EXIF orientation, alpha over white).
    Every image stays in the batch; ``metadata["image_decode_status"]`` is ``ok``,
    ``missing`` (no bytes), ``resource_limit`` (more than ``max_pixels``) or
    ``decode_error``. Encoded bytes are released after decoding.
    """

    max_pixels: int = 25_000_000
    name: str = "image_decode"

    def process(self, task: ImageBatch) -> ImageBatch:
        for image in task.data:
            raw, image.image_bytes, image.image_data = image.image_bytes, None, None
            status = "missing"
            if raw is not None:
                try:
                    with Image.open(io.BytesIO(raw)) as header:
                        too_large = header.width * header.height > self.max_pixels
                    status = "resource_limit" if too_large else "ok"
                    if not too_large:
                        image.image_data = decode_image_rgb(raw)
                except MemoryError:
                    raise
                except Exception:  # noqa: BLE001 - Pillow raises many exception types for corrupt files
                    status = "decode_error"
            image.metadata["image_decode_status"] = status
        return task


@dataclass
class ImageResizeStage(ProcessingStage[ImageBatch, ImageBatch]):
    """Resize the shorter side to ``size`` and center-crop to ``size`` x ``size`` on CPU.

    Uses the same bicubic, antialiased transform as
    :class:`~nemo_curator.models.clip.CLIPImageEmbeddings`, so GPU stages receive small,
    uniformly shaped uint8 arrays instead of native-resolution pixels.
    Place it after stages that need native pixels, such as image diagnostics.
    """

    size: int = 224
    name: str = "image_resize"

    def process(self, task: ImageBatch) -> ImageBatch:
        torch.set_num_threads(1)  # many CPU workers share the node
        resize = transforms.Compose(
            [
                transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.CenterCrop(self.size),
            ]
        )
        for image in task.data:
            if image.image_data is not None:
                pixels = resize(torch.from_numpy(image.image_data).permute(2, 0, 1))
                image.image_data = pixels.permute(1, 2, 0).contiguous().numpy()
        return task
