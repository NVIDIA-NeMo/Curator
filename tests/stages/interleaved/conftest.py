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

import json
import tarfile
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from nemo_curator.tasks import FileGroupTask


def build_multi_frame_tiff(n_frames: int, width: int = 64, height: int = 48) -> bytes:
    """Build a synthetic multi-frame TIFF with *n_frames* distinct frames.

    Each frame has a unique solid colour so downstream tests can verify that
    the correct frame was extracted.
    """
    frames = []
    for i in range(n_frames):
        r, g, b = (40 * i) % 256, (80 + 30 * i) % 256, (160 + 50 * i) % 256
        frames.append(Image.new("RGB", (width + i, height + i), (r, g, b)))
    buf = BytesIO()
    frames[0].save(buf, format="TIFF", save_all=True, append_images=frames[1:])
    return buf.getvalue()


def write_tar(tar_path: Path, members: dict[str, bytes]) -> str:
    """Write a tar archive with the given ``{member_name: payload}`` map."""
    with tarfile.open(tar_path, "w") as tf:
        for name, payload in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            tf.addfile(info, BytesIO(payload))
    return str(tar_path)


def task_for_tar(tar_path: str, task_id: str = "file_group_0", dataset_name: str = "mint_test") -> FileGroupTask:
    """Build a ``FileGroupTask`` wrapping a single tar path."""
    return FileGroupTask(
        task_id=task_id,
        dataset_name=dataset_name,
        data=[tar_path],
        _metadata={"source_files": [tar_path]},
    )


@pytest.fixture
def mint_like_tar(tmp_path: Path) -> tuple[str, str, bytes]:
    tar_path = tmp_path / "shard-00000.tar"
    sample_id = "abc123"
    payload = {
        "pdf_name": "doc.pdf",
        "url": "https://example.com/doc.pdf",
        "texts": ["hello", None, "world"],
        "images": ["page_0_image_1", None, "page_2_image_9"],
        "image_metadata": [{"page": 0}, {"page": 2}],
    }
    image_bytes = b"fake-image-bytes"
    with tarfile.open(tar_path, "w") as tf:
        json_blob = json.dumps(payload).encode("utf-8")
        json_info = tarfile.TarInfo(name=f"{sample_id}.json")
        json_info.size = len(json_blob)
        tf.addfile(json_info, BytesIO(json_blob))

        img_info = tarfile.TarInfo(name=f"{sample_id}.tiff")
        img_info.size = len(image_bytes)
        tf.addfile(img_info, BytesIO(image_bytes))
    return str(tar_path), sample_id, image_bytes


@pytest.fixture
def input_task(mint_like_tar: tuple[str, str, bytes]) -> FileGroupTask:
    tar_path, _, _ = mint_like_tar
    return task_for_tar(tar_path)
