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

"""Shared pytest fixtures and factory helpers for interleaved IO tests.

Factory helpers (non-fixture functions)
----------------------------------------
make_row           -- Build a single interleaved row dict with sensible defaults.
make_image_row     -- Build an image row dict with encoded source_ref.
make_image_task    -- Build an InterleavedBatch from a list of row dicts.
make_interleaved_batch -- Build a standard metadata+text+image batch.
write_tar          -- Write a tar archive from a ``{name: bytes}`` map.
task_for_tar       -- Wrap a tar path in a FileGroupTask.
build_multi_frame_tiff -- Generate a synthetic multi-frame TIFF.

Fixtures
---------
mint_like_tar      -- MINT-1T-style tar with JSON payload + TIFF image.
input_task         -- FileGroupTask wrapping the mint_like_tar fixture.
single_row_table   -- Single-row PyArrow table (text modality).
single_row_task    -- InterleavedBatch wrapping single_row_table.
"""

import json
import tarfile
from io import BytesIO
from pathlib import Path

import pyarrow as pa
import pytest
from PIL import Image

from nemo_curator.tasks import FileGroupTask, InterleavedBatch
from nemo_curator.tasks.interleaved import INTERLEAVED_SCHEMA

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


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


def build_jpeg_in_tiff(n_frames: int = 2, width: int = 32, height: int = 32) -> bytes:
    """Build a multi-frame TIFF stored with JPEG compression (TIFF+JPEG, like MINT-1T PDFs).

    Each frame has a distinct solid colour.  Saving RGBA with JPEG compression
    is the exact scenario that previously caused pixel corruption in
    ``_extract_tiff_frame`` (JPEG does not support alpha — using it for an RGBA
    frame produced wrong pixel values and a corrupted alpha channel).
    """
    frames = []
    for i in range(n_frames):
        r, g, b, a = (50 * i) % 256, (100 + 40 * i) % 256, (150 + 30 * i) % 256, 255
        frames.append(Image.new("RGBA", (width, height), (r, g, b, a)))
    buf = BytesIO()
    frames[0].save(buf, format="TIFF", compression="jpeg", save_all=True, append_images=frames[1:])
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


# ---------------------------------------------------------------------------
# Row / batch factories
# ---------------------------------------------------------------------------


def _make_base_row(sample_id: str, position: int, modality: str, content_type: str) -> dict:
    """Internal: build the fixed fields shared by all row factories."""
    return {
        "sample_id": sample_id,
        "position": position,
        "modality": modality,
        "content_type": content_type,
        "text_content": None,
        "binary_content": None,
        "source_ref": None,
        "materialize_error": None,
    }


def make_text_row(
    sample_id: str = "s1",
    position: int = 0,
    text_content: str | None = None,
    source_ref: str | None = None,
) -> dict:
    """Build a ``text`` modality row dict.

    Example::

        make_text_row("doc1", position=0, text_content="hello")
    """
    row = _make_base_row(sample_id, position, "text", "text/plain")
    row["text_content"] = text_content
    row["source_ref"] = source_ref
    return row


def make_metadata_row(
    sample_id: str = "s1",
    source_ref: str | None = None,
) -> dict:
    """Build a ``metadata`` modality row dict (position is always -1).

    Example::

        make_metadata_row("doc1")
    """
    row = _make_base_row(sample_id, -1, "metadata", "application/json")
    row["source_ref"] = source_ref
    return row


def make_row(
    sample_id: str = "s1",
    position: int = 0,
    modality: str = "text",
    content_type: str | None = None,
    **overrides: object,
) -> dict:
    """Build a single interleaved row dict with sensible defaults.

    For the common cases prefer the focused factories ``make_text_row``,
    ``make_metadata_row``, or ``make_image_row``.  Use ``make_row`` when you
    need an arbitrary combination of fields.

    *content_type* defaults to the conventional MIME type for the given
    *modality* when not supplied explicitly.  Extra keyword arguments override
    any default field value.

    Example::

        make_row("doc1", 0, "text", text_content="hello")
        make_row("doc1", -1, "metadata")
        make_row("doc1", 1, "image", binary_content=b"bytes", content_type="image/png")
    """
    _mime = {"text": "text/plain", "image": "image/jpeg", "metadata": "application/json"}
    row = _make_base_row(sample_id, position, modality, content_type or _mime.get(modality, ""))
    row.update(overrides)
    return row


def make_image_row(
    path: str | None,
    member: str | None = None,
    byte_offset: int | None = None,
    byte_size: int | None = None,
    content_type: str = "image/jpeg",
) -> dict:
    """Build an image row dict with an encoded *source_ref*.

    Defaults to ``sample_id="s1"``, ``position=0``, and
    ``content_type="image/jpeg"``.

    Example::

        make_image_row(path="/data/shard.tar", member="img.jpg")
        make_image_row(path="/data/range.bin", member="img.tiff", byte_offset=512, byte_size=1024)
        make_image_row(path="/data/img.tiff", content_type="image/tiff")
    """
    return {
        "sample_id": "s1",
        "position": 0,
        "modality": "image",
        "content_type": content_type,
        "text_content": None,
        "binary_content": None,
        "source_ref": InterleavedBatch.build_source_ref(
            path=path,
            member=member,
            byte_offset=byte_offset,
            byte_size=byte_size,
        ),
        "materialize_error": None,
    }


def make_image_task(rows: list[dict], metadata: dict | None = None) -> InterleavedBatch:
    """Create an ``InterleavedBatch`` from a list of row dicts.

    Uses ``INTERLEAVED_SCHEMA`` so all reserved columns are typed correctly.
    Primarily used by materialization and classify-rows tests.
    """
    table = pa.Table.from_pylist(rows, schema=INTERLEAVED_SCHEMA)
    return InterleavedBatch(task_id="test", dataset_name="d", data=table, _metadata=metadata or {})


def make_interleaved_batch(
    num_samples: int = 2,
    task_id: str = "test_batch",
    include_images: bool = True,
    schema: pa.Schema = INTERLEAVED_SCHEMA,
) -> InterleavedBatch:
    """Build a standard ``InterleavedBatch`` with one sample per *num_samples*.

    Each sample contains:
    * one ``metadata`` row  (position=-1)
    * one ``text`` row      (position=0)
    * one ``image`` row     (position=1) -- omitted when *include_images=False*

    The *schema* parameter lets callers pass an extended schema (e.g. with
    extra passthrough columns) while reusing this factory.

    Example::

        batch = make_interleaved_batch(num_samples=3)
        batch = make_interleaved_batch(num_samples=1, include_images=False)
    """
    rows = []
    for i in range(num_samples):
        sid = f"sample_{i}"
        rows.append(make_metadata_row(sid))
        rows.append(make_text_row(sid, position=0, text_content=f"Hello {i}"))
        if include_images:
            rows.append(make_row(sid, 1, "image", binary_content=b"fake-jpeg-bytes"))
    table = pa.Table.from_pylist(rows, schema=schema)
    return InterleavedBatch(
        task_id=task_id,
        dataset_name="test",
        data=table,
        _metadata={"source_files": ["test.tar"]},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mint_like_tar(tmp_path: Path) -> tuple[str, str, bytes]:
    """Create a MINT-1T-style tar with a JSON sidecar and a TIFF image.

    Returns ``(tar_path, sample_id, image_bytes)``.
    """
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
    """``FileGroupTask`` wrapping the ``mint_like_tar`` tar archive."""
    tar_path, _, _ = mint_like_tar
    return task_for_tar(tar_path)


@pytest.fixture
def single_row_table() -> pa.Table:
    """Single-row PyArrow table (text modality) with a realistic source_ref."""
    return pa.Table.from_pylist(
        [
            make_row(
                sample_id="s1",
                position=0,
                modality="text",
                text_content="hello",
                source_ref=json.dumps(
                    {"path": "/dataset/shard.tar", "member": "s1.json", "byte_offset": 10, "byte_size": 20}
                ),
            )
        ],
        schema=INTERLEAVED_SCHEMA,
    )


@pytest.fixture
def single_row_task(single_row_table: pa.Table) -> InterleavedBatch:
    """``InterleavedBatch`` wrapping the ``single_row_table`` fixture."""
    return InterleavedBatch(task_id="t1", dataset_name="d1", data=single_row_table)
