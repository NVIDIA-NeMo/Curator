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

import numpy as np

from nemo_curator.tasks import ImageBatch, ImageObject


def test_reprs_do_not_format_image_payloads() -> None:
    # Ray Data formats a repr for every stage output; printing arrays made GPU stages CPU-bound.
    image = ImageObject(
        image_id="a",
        image_bytes=b"x" * 10_000,
        image_data=np.ones((224, 224, 3), np.uint8),
        embedding=np.ones(768, np.float32),
    )
    assert repr(ImageBatch(data=[image] * 128, dataset_name="test")) == "ImageBatch(task_id=, dataset_name=test)"
    assert len(repr(image)) < 200
