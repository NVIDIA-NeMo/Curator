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

import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest


def _import_stage_module() -> tuple[Any, Any]:
    # Inject a stub for optional dependency 'wget' to avoid import errors
    if "wget" not in sys.modules:
        sys.modules["wget"] = types.SimpleNamespace(download=lambda *_args, **_kwargs: None)
    from nemo_curator.stages.audio.datasets.fleurs.create_initial_manifest import (
        CreateInitialManifestFleursStage,
        get_fleurs_url_list,
    )

    return CreateInitialManifestFleursStage, get_fleurs_url_list


def test_ray_stage_spec(tmp_path: Path) -> None:
    from nemo_curator.backends.experimental.utils import RayStageSpecKeys

    stage_cls, _ = _import_stage_module()
    stage = stage_cls(lang="hy_am", split="dev", raw_data_dir=str(tmp_path / "fleurs"))
    spec = stage.ray_stage_spec()
    assert spec[RayStageSpecKeys.IS_FANOUT_STAGE] is True


def test_get_fleurs_url_list_builds_urls() -> None:
    _, get_fleurs_url_list = _import_stage_module()
    urls = get_fleurs_url_list("hy_am", "dev")
    assert urls[0].endswith("/hy_am/dev.tsv")
    assert urls[1].endswith("/hy_am/audio/dev.tar.gz")


def test_post_init_requires_lang(tmp_path: Path) -> None:
    stage_cls, _ = _import_stage_module()
    with pytest.raises(ValueError, match="lang is required"):
        stage_cls(lang="", split="dev", raw_data_dir=str(tmp_path))


def test_post_init_requires_split(tmp_path: Path) -> None:
    stage_cls, _ = _import_stage_module()
    with pytest.raises(ValueError, match="split is required"):
        stage_cls(lang="en_us", split="", raw_data_dir=str(tmp_path))


def test_post_init_requires_raw_data_dir() -> None:
    stage_cls, _ = _import_stage_module()
    with pytest.raises(ValueError, match="raw_data_dir is required"):
        stage_cls(lang="en_us", split="dev", raw_data_dir="")


def test_inputs_outputs(tmp_path: Path) -> None:
    stage_cls, _ = _import_stage_module()
    stage = stage_cls(lang="en_us", split="dev", raw_data_dir=str(tmp_path))
    assert stage.inputs() == ([], [])
    assert stage.outputs() == ([], ["audio_filepath", "text"])


def test_download_extract_files(tmp_path: Path) -> None:
    from unittest.mock import patch

    stage_cls, _ = _import_stage_module()
    stage = stage_cls(lang="en_us", split="dev", raw_data_dir=str(tmp_path / "fleurs"))

    with (
        patch("nemo_curator.stages.audio.datasets.fleurs.create_initial_manifest.download_file") as mock_dl,
        patch("nemo_curator.stages.audio.datasets.fleurs.create_initial_manifest.extract_archive") as mock_ext,
    ):
        stage.download_extract_files(str(tmp_path / "fleurs"))
        assert mock_dl.call_count == 2
        mock_ext.assert_called_once()


def test_process_end_to_end(tmp_path: Path) -> None:
    from unittest.mock import patch

    stage_cls, _ = _import_stage_module()
    raw_dir = tmp_path / "fleurs"
    raw_dir.mkdir()
    tsv_path = raw_dir / "dev.tsv"
    tsv_path.write_text("0\tfile1.wav\thello\n1\tfile2.wav\tworld\n", encoding="utf-8")
    audio_dir = raw_dir / "dev"
    audio_dir.mkdir()
    (audio_dir / "file1.wav").write_bytes(b"")
    (audio_dir / "file2.wav").write_bytes(b"")

    stage = stage_cls(lang="en_us", split="dev", raw_data_dir=str(raw_dir))
    from nemo_curator.tasks import _EmptyTask

    with (
        patch("nemo_curator.stages.audio.datasets.fleurs.create_initial_manifest.download_file"),
        patch("nemo_curator.stages.audio.datasets.fleurs.create_initial_manifest.extract_archive"),
    ):
        results = stage.process(_EmptyTask(task_id="empty", dataset_name="test", data=None))
    assert len(results) == 2
    assert results[0].data["text"] == "hello"
    assert results[1].data["text"] == "world"


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

    # Each valid TSV line produces one AudioTask
    assert len(batches) == 2
    b0, b1 = batches
    assert b0.data[stage.filepath_key].endswith(os.path.join(split, "file1.wav"))
    assert b0.data[stage.text_key] == "hello world"
    assert b1.data[stage.filepath_key].endswith(os.path.join(split, "file2.wav"))
    assert b1.data[stage.text_key] == "second"
