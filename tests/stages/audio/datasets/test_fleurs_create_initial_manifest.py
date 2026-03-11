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

import os
import sys
import types
from pathlib import Path
from typing import Any


def _import_stage_module() -> tuple[Any, Any]:
    # Inject a stub for optional dependency 'wget' to avoid import errors
    if "wget" not in sys.modules:
        sys.modules["wget"] = types.SimpleNamespace(download=lambda *_args, **_kwargs: None)
    from nemo_curator.stages.audio.datasets.fleurs.create_initial_manifest import (
        CreateInitialManifestFleursStage,
        get_fleurs_url_list,
    )

    return CreateInitialManifestFleursStage, get_fleurs_url_list


def test_get_fleurs_url_list_builds_urls() -> None:
    _, get_fleurs_url_list = _import_stage_module()
    urls = get_fleurs_url_list("hy_am", "dev")
    assert urls[0].endswith("/hy_am/dev.tsv")
    assert urls[1].endswith("/hy_am/audio/dev.tar.gz")


def test_process_transcript_parses_tsv(tmp_path: Path) -> None:
    stage_cls, _ = _import_stage_module()
    # Arrange: create fake dev.tsv and expected wav layout
    lang = "hy_am"
    split = "dev"
    raw_dir = tmp_path / "fleurs"
    audio_dir = raw_dir / split
    audio_dir.mkdir(parents=True)

    # two rows, one malformed that should be skipped
    tsv_path = raw_dir / f"{split}.tsv"
    lines = [
        "idx\tfile1.wav\thello world\n",
        "badline\n",
        "idx\tfile2.wav\tsecond\n",
    ]
    tsv_path.write_text("".join(lines), encoding="utf-8")

    # Create the expected audio files (names only needed for abspath join)
    (audio_dir / "file1.wav").write_bytes(b"")
    (audio_dir / "file2.wav").write_bytes(b"")

    stage = stage_cls(lang=lang, split=split, raw_data_dir=raw_dir.as_posix())

    # Act
    batches = stage.process_transcript(tsv_path.as_posix())

    # Assert batching behavior and content (default batch_size is 1, so expect 2 batches)
    assert len(batches) == 2
    b0, b1 = batches
    assert len(b0.data) == 1
    assert len(b1.data) == 1
    assert b0.data[0][stage.filepath_key].endswith(os.path.join(split, "file1.wav"))
    assert b0.data[0][stage.text_key] == "hello world"
    assert b1.data[0][stage.filepath_key].endswith(os.path.join(split, "file2.wav"))
    assert b1.data[0][stage.text_key] == "second"
