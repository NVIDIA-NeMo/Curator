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

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest  # pyright: ignore[reportMissingImports]

if TYPE_CHECKING:
    from pathlib import Path

from nemo_curator.stages.image.io.convert import ConvertImageBatchToDocumentBatchStage
from nemo_curator.tasks import DocumentBatch, ImageBatch, ImageObject


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(7)


@pytest.fixture
def image_batch_with_embeddings(rng: np.random.Generator, tmp_path: Path) -> ImageBatch:
    images: list[ImageObject] = []
    for i in range(3):
        embedding = rng.normal(size=(8,)).astype(np.float32)
        images.append(
            ImageObject(
                image_id=f"img_{i:03d}",
                image_path=str(tmp_path / f"img_{i:03d}.jpg"),
                image_data=rng.integers(0, 255, (16, 16, 3), dtype=np.uint8),
                embedding=embedding,
            )
        )
    return ImageBatch(
        data=images,
        dataset_name="ds_test",
        task_id="task_123",
        _metadata={"foo": "bar"},
        _stage_perf={"stage": 1.23},
    )


class TestConvertImageBatchToDocumentBatchStage:
    def test_default_fields_outputs_image_id_only(self, image_batch_with_embeddings: ImageBatch) -> None:
        stage = ConvertImageBatchToDocumentBatchStage()
        out = stage.process(image_batch_with_embeddings)

        assert isinstance(out, DocumentBatch)
        df = out.to_pandas()
        assert list(df.columns) == ["image_id"]
        assert df["image_id"].tolist() == [img.image_id for img in image_batch_with_embeddings.data]

        # Metadata and identifiers preserved
        assert out.task_id == f"{image_batch_with_embeddings.task_id}_{stage.name}"
        assert out.dataset_name == image_batch_with_embeddings.dataset_name
        assert out._metadata == image_batch_with_embeddings._metadata
        assert out._stage_perf == image_batch_with_embeddings._stage_perf

    def test_custom_fields_include_embeddings(self, image_batch_with_embeddings: ImageBatch) -> None:
        stage = ConvertImageBatchToDocumentBatchStage(fields=["image_id", "embedding"])
        out = stage.process(image_batch_with_embeddings)

        assert isinstance(out, DocumentBatch)
        df = out.to_pandas()
        assert list(df.columns) == ["image_id", "embedding"]
        assert df.shape[0] == 3

        # Validate that image_ids and embeddings are correctly propagated
        src_ids = [img.image_id for img in image_batch_with_embeddings.data]
        assert df["image_id"].tolist() == src_ids
        for i, img in enumerate(image_batch_with_embeddings.data):
            np.testing.assert_allclose(df.iloc[i]["embedding"], img.embedding)

    def test_empty_input_default_fields(self) -> None:
        stage = ConvertImageBatchToDocumentBatchStage()
        empty_batch = ImageBatch(data=[], dataset_name="ds", task_id="t0")
        out = stage.process(empty_batch)
        df = out.to_pandas()
        assert isinstance(out, DocumentBatch)
        assert list(df.columns) == ["image_id"]
        assert len(df) == 0

    def test_empty_input_with_custom_fields(self) -> None:
        stage = ConvertImageBatchToDocumentBatchStage(fields=["image_id", "embedding", "image_path"])
        empty_batch = ImageBatch(data=[], dataset_name="ds", task_id="t0")
        out = stage.process(empty_batch)
        df = out.to_pandas()
        assert list(df.columns) == ["image_id", "embedding", "image_path"]
        assert len(df) == 0

    def test_missing_attribute_yields_none(self, rng: np.random.Generator, tmp_path: Path) -> None:
        # Build images without the 'embedding' attribute
        images = [
            ImageObject(
                image_id="no_embed_1",
                image_path=str(tmp_path / "a.jpg"),
                image_data=rng.integers(0, 255, (8, 8, 3), dtype=np.uint8),
            ),
            ImageObject(
                image_id="no_embed_2",
                image_path=str(tmp_path / "b.jpg"),
                image_data=rng.integers(0, 255, (8, 8, 3), dtype=np.uint8),
            ),
        ]
        batch = ImageBatch(data=images, dataset_name="dsx", task_id="t1")
        stage = ConvertImageBatchToDocumentBatchStage(fields=["image_id", "embedding"])  # 'embedding' may be missing
        out = stage.process(batch)
        df = out.to_pandas()
        assert list(df.columns) == ["image_id", "embedding"]
        assert df.shape == (2, 2)
        assert df["embedding"].isna().all()
