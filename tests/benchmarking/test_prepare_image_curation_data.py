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

import io
import json
import tarfile
from pathlib import Path

from PIL import Image

from benchmarking.data_prep.prepare_image_curation_data import (
    build_dataset,
    select_candidates,
    verify_dataset,
)


def _multi_frame_tiff() -> bytes:
    frames = [Image.new("RGB", (64, 48), color=color) for color in ("red", "blue")]
    output = io.BytesIO()
    frames[0].save(output, format="TIFF", save_all=True, append_images=frames[1:])
    for frame in frames:
        frame.close()
    return output.getvalue()


def _add_bytes(tf: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    tf.addfile(info, io.BytesIO(payload))


def _write_mint_fixture(input_path: Path) -> None:
    input_path.mkdir()
    metadata = [{"sha256": f"{value:064x}", "width": 64, "height": 48} for value in (1, 2, 3)]
    payload = {
        "images": ["bad.tiff", "frame_0", "frame_1"],
        "image_metadata": metadata,
        "texts": [None, None, None],
    }
    with tarfile.open(input_path / "source.tar", "w") as tf:
        _add_bytes(tf, "sample.json", json.dumps(payload).encode())
        _add_bytes(tf, "sample.tiff", _multi_frame_tiff())
        _add_bytes(tf, "bad.tiff", b"not a TIFF")


def test_select_build_and_verify_exact_unique_fixture(tmp_path: Path) -> None:
    input_path = tmp_path / "input"
    output_path = tmp_path / "output"
    output_path.mkdir()
    _write_mint_fixture(input_path)

    candidates, stats = select_candidates(input_path, num_candidates=3, min_side=32, max_pixels=1_000_000)

    assert len(candidates) == 3
    assert len({candidate.source_sha256 for candidate in candidates}) == 3
    assert stats["eligible"] == 3
    build_stats = build_dataset(
        input_path,
        output_path,
        candidates,
        num_images=2,
        images_per_tar=2,
        candidate_buffer_per_shard=1,
        jpeg_quality=95,
        workers=1,
    )
    verification = verify_dataset(output_path, num_images=2, images_per_tar=2)

    assert build_stats == {"images": 2, "decode_errors": 1, "reused_shards": 0}
    assert verification == {"num_shards": 1, "num_images": 2, "unique_images": 2}
    with tarfile.open(output_path / "000000.tar") as tf:
        assert [member.name for member in tf.getmembers()] == [f"{2:064x}.jpg", f"{3:064x}.jpg"]


def test_selection_deduplicates_source_sha256(tmp_path: Path) -> None:
    input_path = tmp_path / "input"
    _write_mint_fixture(input_path)
    duplicate_payload = {
        "images": ["only_image"],
        "image_metadata": [{"sha256": f"{1:064x}", "width": 64, "height": 48}],
    }
    with tarfile.open(input_path / "duplicate.tar", "w") as tf:
        _add_bytes(tf, "duplicate.json", json.dumps(duplicate_payload).encode())
        _add_bytes(tf, "duplicate.tiff", _multi_frame_tiff())

    candidates, _ = select_candidates(input_path, num_candidates=3, min_side=32, max_pixels=1_000_000)

    assert len(candidates) == 3
    assert len({candidate.source_sha256 for candidate in candidates}) == 3
