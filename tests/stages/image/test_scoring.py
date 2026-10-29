# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from collections.abc import Callable
from typing import Never

import numpy as np
import pytest

from nemo_curator.stages.image.scoring import ImageScoringStage
from nemo_curator.tasks import ImageBatch, ImageObject


def test_scoring_keeps_identity_boundaries_and_failed_images():
    calls = []
    setups = []

    def factory() -> Callable[[list[np.ndarray]], list[float]]:
        setups.append(1)

        def predict(images: list[np.ndarray]) -> list[float]:
            calls.append([int(image[0, 0, 0]) for image in images])
            return [float(image.mean()) for image in images]

        return predict

    stage = ImageScoringStage(factory, model_inference_batch_size=2, remove_image_data=True)
    stage.setup()
    task = ImageBatch(
        dataset_name="test",
        data=[
            ImageObject(
                image_id=str(i),
                metadata={"url": str(i)},
                image_data=np.full((2, 3, 3), i, dtype=np.uint8) if i != 1 else None,
            )
            for i in range(5)
        ],
    )
    result = stage.process(task)
    assert calls == [[0], [2, 3], [4]]
    assert setups == [1]
    assert [image.image_id for image in result.data] == list("01234")
    assert [image.metadata["url"] for image in result.data] == list("01234")
    assert [image.metadata["image_score"] for image in result.data] == [0.0, None, 2.0, 3.0, 4.0]
    assert result.data[1].metadata["image_score_status"] == "decode_error"
    assert all(image.image_data is None for image in result.data)


@pytest.mark.parametrize("scores", [[], [float("nan")], [float("inf")], [1.0, 2.0]])
def test_invalid_model_output_fails_without_claiming_success(scores: list[float]):
    stage = ImageScoringStage(lambda: lambda _images: scores)
    stage.setup()
    task = ImageBatch(dataset_name="test", data=[ImageObject(image_data=np.zeros((2, 2, 3)))])
    with pytest.raises(ValueError, match="one finite scalar"):
        stage.process(task)
    assert "image_score_status" not in task.data[0].metadata


def test_model_failure_is_retryable_and_not_a_decode_failure():
    def fail(_images: list[np.ndarray]) -> Never:
        message = "device unavailable"
        raise RuntimeError(message)

    stage = ImageScoringStage(lambda: fail)
    stage.setup()
    task = ImageBatch(dataset_name="test", data=[ImageObject(image_data=np.zeros((2, 2, 3)))])
    with pytest.raises(RuntimeError, match="device unavailable"):
        stage.process(task)
    assert "image_score_status" not in task.data[0].metadata
