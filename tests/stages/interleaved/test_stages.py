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

"""Tests for the task-level ndarray cache on ``BaseInterleavedAnnotatorStage``."""

from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
import pyarrow as pa
from PIL import Image

from nemo_curator.stages.interleaved.filter.blur_filter import InterleavedBlurFilterStage
from nemo_curator.stages.interleaved.filter.qrcode_filter import InterleavedQRCodeFilterStage
from nemo_curator.tasks import InterleavedBatch
from nemo_curator.tasks.interleaved import INTERLEAVED_SCHEMA

from .conftest import make_row


def _jpeg_bytes(seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


def _image_task(jpeg_bytes_list: list[bytes]) -> InterleavedBatch:
    rows = [
        make_row(sample_id="s1", position=i, modality="image", binary_content=b)
        for i, b in enumerate(jpeg_bytes_list)
    ]
    table = pa.Table.from_pylist(rows, schema=INTERLEAVED_SCHEMA)
    return InterleavedBatch(task_id="t", dataset_name="d", data=table)


def test_iter_decoded_images_second_call_hits_cache() -> None:
    """A repeat call on the same task yields cached arrays without re-materializing."""
    task = _image_task([_jpeg_bytes(0)])
    stage = InterleavedQRCodeFilterStage()
    df = task.to_pandas()
    image_mask = df["modality"] == "image"

    first = list(stage.iter_decoded_images(task=task, df=df, row_mask=image_mask))
    assert len(first) == 1
    idx, arr = first[0]
    assert arr is not None
    assert task._metadata["_image_array_cache"][idx] is arr

    materialize = MagicMock(side_effect=AssertionError("cache hit must skip materialization"))
    with patch.object(stage, "iter_materialized_bytes", materialize):
        second = list(stage.iter_decoded_images(task=task, df=df, row_mask=image_mask))

    assert len(second) == 1
    assert second[0][0] == idx
    assert second[0][1] is arr
    materialize.assert_not_called()


def test_iter_decoded_images_cache_shared_across_stages() -> None:
    """Filter stages on the same task share the decoded-array cache via ``task._metadata``."""
    task = _image_task([_jpeg_bytes(0)])
    df = task.to_pandas()
    image_mask = df["modality"] == "image"

    producer = InterleavedBlurFilterStage()
    consumer = InterleavedQRCodeFilterStage()

    first = list(producer.iter_decoded_images(task=task, df=df, row_mask=image_mask))

    materialize = MagicMock(side_effect=AssertionError("cache hit must skip materialization"))
    with patch.object(consumer, "iter_materialized_bytes", materialize):
        second = list(consumer.iter_decoded_images(task=task, df=df, row_mask=image_mask))

    assert second[0][1] is first[0][1]
    materialize.assert_not_called()


def test_iter_decoded_images_mixed_hits_and_misses() -> None:
    """Pre-cached rows are yielded from cache; uncached rows fall through to decode."""
    task = _image_task([_jpeg_bytes(0), _jpeg_bytes(1)])
    df = task.to_pandas()
    image_mask = df["modality"] == "image"

    preloaded = np.zeros((4, 4, 3), dtype=np.uint8)
    task._metadata["_image_array_cache"] = {0: preloaded}

    results = dict(InterleavedQRCodeFilterStage().iter_decoded_images(task=task, df=df, row_mask=image_mask))

    assert results[0] is preloaded
    assert results[1] is not None
    assert results[1].shape == (32, 32, 3)
    assert task._metadata["_image_array_cache"][1] is results[1]
