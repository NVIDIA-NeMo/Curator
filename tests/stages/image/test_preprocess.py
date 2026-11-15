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

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from nemo_curator.stages.image.preprocess import ImageDecodeStage, ImageResizeStage
from nemo_curator.tasks import ImageBatch, ImageObject


def _png(width: int, height: int) -> bytes:
    rng = np.random.default_rng(0)
    buffer = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (height, width, 3), dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_decode_statuses_and_releases_bytes() -> None:
    images = [
        ImageObject(image_id="ok", image_bytes=_png(40, 30)),
        ImageObject(image_id="big", image_bytes=_png(100, 100)),
        ImageObject(image_id="bad", image_bytes=b"not an image"),
        ImageObject(image_id="missing"),
    ]
    out = ImageDecodeStage(max_pixels=5000).process(ImageBatch(data=images, dataset_name="test"))
    assert [image.metadata["image_decode_status"] for image in out.data] == [
        "ok",
        "resource_limit",
        "decode_error",
        "missing",
    ]
    assert out.data[0].image_data.shape == (30, 40, 3)
    assert all(image.image_data is None for image in out.data[1:])
    assert all(image.image_bytes is None for image in out.data)


def test_resize_matches_clip_transform_and_skips_missing() -> None:
    rgb = np.random.default_rng(1).integers(0, 255, (300, 500, 3), dtype=np.uint8)
    out = ImageResizeStage(size=224).process(
        ImageBatch(data=[ImageObject(image_data=rgb), ImageObject()], dataset_name="test")
    )
    assert out.data[0].image_data.shape == (224, 224, 3)
    assert out.data[0].image_data.dtype == np.uint8
    assert out.data[1].image_data is None
    # Same pixels as the resize/crop half of CLIPImageEmbeddings' GPU transform.
    clip = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(224),
        ]
    )
    expected = clip(torch.from_numpy(rgb).permute(2, 0, 1)).permute(1, 2, 0).numpy()
    np.testing.assert_array_equal(out.data[0].image_data, expected)


def test_task_repr_does_not_format_image_arrays() -> None:
    # Ray Data calls repr() on every stage output; formatting arrays made GPU stages CPU-bound.
    batch = ImageBatch(
        data=[ImageObject(embedding=np.ones(768, np.float32), image_data=np.ones((224, 224, 3), np.uint8))] * 128,
        dataset_name="test",
    )
    assert repr(batch) == "ImageBatch(task_id=, dataset_name=test)"
