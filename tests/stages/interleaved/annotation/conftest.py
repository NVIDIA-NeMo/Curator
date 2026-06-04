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

from io import BytesIO

import numpy as np
import pyarrow as pa
from PIL import Image

from nemo_curator.tasks import InterleavedBatch
from nemo_curator.tasks.interleaved import INTERLEAVED_SCHEMA


def interleaved_task(rows: list[dict]) -> InterleavedBatch:
    table = pa.Table.from_pylist(rows, schema=INTERLEAVED_SCHEMA)
    return InterleavedBatch(task_id="test", dataset_name="d", data=table)


def make_jpeg_bytes(width: int = 32, height: int = 32, sharp: bool = True) -> bytes:
    buf = BytesIO()
    if sharp:
        rng = np.random.default_rng()
        arr = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
    else:
        img = Image.new("RGB", (width, height), (128, 128, 128))
    img.save(buf, format="JPEG")
    return buf.getvalue()
