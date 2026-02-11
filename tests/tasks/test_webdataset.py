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

import numpy as np

from nemo_curator.tasks.webdataset import WebDatasetBatch, WebDatasetSample


def test_sample_defaults() -> None:
    s = WebDatasetSample()
    assert s.key == ""
    assert s.components == {}
    assert s.metadata == {}
    assert s.shard_path == ""
    assert s.extensions == []


def test_sample_image_helpers() -> None:
    s = WebDatasetSample(key="img_001")
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    s.set_image(img, ext="jpg")
    assert s.get_image("jpg") is img
    assert s.get_image("png") is None


def test_sample_text_helpers() -> None:
    s = WebDatasetSample(key="doc_001")
    s.set_text("hello world")
    assert s.get_text("txt") == "hello world"
    assert s.get_text("json") is None


def test_sample_json_helpers() -> None:
    s = WebDatasetSample(key="meta_001")
    s.set_json({"label": 1})
    assert s.get_json("json") == {"label": 1}
    assert s.get_json("txt") is None


def test_sample_extensions_property() -> None:
    s = WebDatasetSample(key="multi", components={"jpg": b"img", "json": {"a": 1}, "txt": "cap"})
    exts = s.extensions
    assert set(exts) == {"jpg", "json", "txt"}


def test_sample_metadata() -> None:
    s = WebDatasetSample(key="m1", metadata={"score": 0.9, "source": "web"})
    assert s.metadata["score"] == 0.9


def test_batch_creation() -> None:
    samples = [
        WebDatasetSample(key=f"s{i}", components={"jpg": np.zeros((2, 2, 3), dtype=np.uint8)})
        for i in range(5)
    ]
    batch = WebDatasetBatch(task_id="b1", dataset_name="ds", data=samples)
    assert batch.num_items == 5
    assert batch.validate() is True


def test_batch_empty() -> None:
    batch = WebDatasetBatch(task_id="empty", dataset_name="ds", data=[])
    assert batch.num_items == 0
    assert batch.validate() is True


def test_batch_is_task_subclass() -> None:
    from nemo_curator.tasks.tasks import Task

    batch = WebDatasetBatch(task_id="t", dataset_name="d", data=[])
    assert isinstance(batch, Task)


def test_batch_repr() -> None:
    batch = WebDatasetBatch(task_id="t1", dataset_name="ds", data=[])
    assert "WebDatasetBatch" in repr(batch)
    assert "t1" in repr(batch)


def test_sample_multimodal() -> None:
    """A single sample can hold image + text + json simultaneously."""
    s = WebDatasetSample(key="multi_001")
    s.set_image(np.zeros((8, 8, 3), dtype=np.uint8))
    s.set_text("a caption")
    s.set_json({"width": 8, "height": 8})

    assert s.get_image() is not None
    assert s.get_text() == "a caption"
    assert s.get_json() == {"width": 8, "height": 8}
    assert set(s.extensions) == {"jpg", "txt", "json"}
