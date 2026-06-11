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
        get_fleurs_filenames,
    )

    return CreateInitialManifestFleursStage, get_fleurs_filenames


def test_ray_stage_spec(tmp_path: Path) -> None:
    from nemo_curator.backends.utils import RayStageSpecKeys

    stage_cls, _ = _import_stage_module()
    stage = stage_cls(lang="hy_am", split="dev", raw_data_dir=str(tmp_path / "fleurs"))
    spec = stage.ray_stage_spec()
    assert spec[RayStageSpecKeys.IS_FANOUT_STAGE] is True


def test_get_fleurs_filenames_builds_paths() -> None:
    _, get_fleurs_filenames = _import_stage_module()
    tsv_filename, audio_filename = get_fleurs_filenames("hy_am", "dev")
    assert tsv_filename == "data/hy_am/dev.tsv"
    assert audio_filename == "data/hy_am/audio/dev.tar.gz"


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


def test_language_data_dir_is_namespaced_per_language(tmp_path: Path) -> None:
    stage_cls, _ = _import_stage_module()
    raw = str(tmp_path / "fleurs")
    hy = stage_cls(lang="hy_am", split="dev", raw_data_dir=raw)
    en = stage_cls(lang="en_us", split="dev", raw_data_dir=raw)
    assert hy.language_data_dir() != en.language_data_dir()
    assert hy.language_data_dir().endswith(os.path.join("fleurs", "hy_am"))
    assert en.language_data_dir().endswith(os.path.join("fleurs", "en_us"))


def test_download_extract_files_stages_transcript(tmp_path: Path) -> None:
    from unittest.mock import patch

    stage_cls, _ = _import_stage_module()
    dst = tmp_path / "fleurs" / "en_us"
    hf_cache = tmp_path / "hf_cache"
    hf_cache.mkdir()
    hf_tsv = hf_cache / "dev.tsv"
    hf_tsv.write_text("0\tfile1.wav\thello\n", encoding="utf-8")

    stage = stage_cls(lang="en_us", split="dev", raw_data_dir=str(tmp_path / "fleurs"))

    with (
        patch(
            "nemo_curator.stages.audio.datasets.fleurs.create_initial_manifest.hf_hub_download",
            side_effect=[str(hf_tsv), str(hf_cache / "audio" / "dev.tar.gz")],
        ) as mock_dl,
        patch("nemo_curator.stages.audio.datasets.fleurs.create_initial_manifest.extract_archive") as mock_ext,
    ):
        tsv_path, audio_root = stage.download_extract_files(str(dst))
        assert mock_dl.call_count == 2
        mock_ext.assert_called_once()
        assert tsv_path == os.path.join(str(dst), "dev.tsv")
        assert audio_root == os.path.join(str(dst), "dev")
        assert os.path.isfile(tsv_path)


def test_process_downloads_once_when_missing(tmp_path: Path) -> None:
    from unittest.mock import patch

    stage_cls, _ = _import_stage_module()
    raw_dir = tmp_path / "fleurs"
    hf_cache = tmp_path / "hf_cache"
    hf_cache.mkdir()
    hf_tsv = hf_cache / "dev.tsv"
    hf_tsv.write_text("0\tfile1.wav\thello\n1\tfile2.wav\tworld\n", encoding="utf-8")

    stage = stage_cls(lang="en_us", split="dev", raw_data_dir=str(raw_dir))
    from nemo_curator.tasks import EmptyTask

    with (
        patch(
            "nemo_curator.stages.audio.datasets.fleurs.create_initial_manifest.hf_hub_download",
            side_effect=[str(hf_tsv), str(hf_cache / "audio" / "dev.tar.gz")],
        ) as mock_dl,
        patch("nemo_curator.stages.audio.datasets.fleurs.create_initial_manifest.extract_archive"),
    ):
        results = stage.process(EmptyTask(dataset_name="test", data=None))
    assert mock_dl.call_count == 2
    assert len(results) == 2
    assert results[0].data["text"] == "hello"
    assert results[1].data["text"] == "world"


def test_process_auto_download_reuses_when_present(tmp_path: Path) -> None:
    from unittest.mock import patch

    stage_cls, _ = _import_stage_module()
    raw_dir = _stage_prestaged_layout(tmp_path, lang="en_us", split="dev")
    stage = stage_cls(lang="en_us", split="dev", raw_data_dir=str(raw_dir))
    from nemo_curator.tasks import EmptyTask

    with patch(
        "nemo_curator.stages.audio.datasets.fleurs.create_initial_manifest.hf_hub_download",
    ) as mock_dl:
        results = stage.process(EmptyTask(dataset_name="test", data=None))

    mock_dl.assert_not_called()
    assert len(results) == 2


def _stage_prestaged_layout(tmp_path: Path, lang: str = "hy_am", split: str = "train") -> Path:
    """Create parent dir with per-language staging: <parent>/<lang>/<split>.tsv + <split>/."""
    parent = tmp_path / "fleurs"
    lang_dir = parent / lang
    audio_dir = lang_dir / split
    audio_dir.mkdir(parents=True)
    (lang_dir / f"{split}.tsv").write_text("0\tfile1.wav\thello\n1\tfile2.wav\tworld\n", encoding="utf-8")
    (audio_dir / "file1.wav").write_bytes(b"")
    (audio_dir / "file2.wav").write_bytes(b"")
    return parent


def test_locate_prestaged_files_success(tmp_path: Path) -> None:
    stage_cls, _ = _import_stage_module()
    raw_dir = _stage_prestaged_layout(tmp_path)
    stage = stage_cls(lang="hy_am", split="train", raw_data_dir=str(raw_dir), auto_download=False)

    tsv_path, audio_root = stage.locate_prestaged_files(stage.language_data_dir())
    assert tsv_path == os.path.join(str(raw_dir), "hy_am", "train.tsv")
    assert audio_root == os.path.join(str(raw_dir), "hy_am", "train")


def test_locate_prestaged_files_missing_transcript_raises(tmp_path: Path) -> None:
    stage_cls, _ = _import_stage_module()
    raw_dir = tmp_path / "fleurs"
    empty = raw_dir / "hy_am"
    empty.mkdir(parents=True)
    stage = stage_cls(lang="hy_am", split="train", raw_data_dir=str(raw_dir), auto_download=False)

    with pytest.raises(FileNotFoundError, match="transcript not found"):
        stage.locate_prestaged_files(stage.language_data_dir())


def test_process_no_download_reads_prestaged(tmp_path: Path) -> None:
    from unittest.mock import patch

    stage_cls, _ = _import_stage_module()
    raw_dir = _stage_prestaged_layout(tmp_path)
    stage = stage_cls(lang="hy_am", split="train", raw_data_dir=str(raw_dir), auto_download=False)

    from nemo_curator.tasks import EmptyTask

    with patch(
        "nemo_curator.stages.audio.datasets.fleurs.create_initial_manifest.hf_hub_download",
    ) as mock_dl:
        results = stage.process(EmptyTask(dataset_name="test", data=None))

    mock_dl.assert_not_called()
    assert len(results) == 2
    assert results[0].data["text"] == "hello"
    assert results[1].data["text"] == "world"


def _import_prep_module() -> types.ModuleType:
    """Import benchmarking/data_prep/prepare_fleurs_data.py (not on the default path)."""
    prep_dir = Path(__file__).resolve().parents[4] / "benchmarking" / "data_prep"
    if str(prep_dir) not in sys.path:
        sys.path.insert(0, str(prep_dir))
    import prepare_fleurs_data  # type: ignore[import-not-found]

    return prepare_fleurs_data


def test_prepare_fleurs_stage_dataset_does_not_recopy(tmp_path: Path) -> None:
    """Regression: stage_dataset must not re-copy the transcript onto itself."""
    from unittest.mock import patch

    _import_stage_module()
    prep = _import_prep_module()

    output_path = tmp_path / "fleurs"
    hf_cache = tmp_path / "hf_cache" / "audio"
    hf_cache.mkdir(parents=True)
    hf_tsv = hf_cache.parent / "train.tsv"
    hf_tsv.write_text("0\tfile1.wav\thello\n", encoding="utf-8")

    with (
        patch(
            "nemo_curator.stages.audio.datasets.fleurs.create_initial_manifest.hf_hub_download",
            side_effect=[str(hf_tsv), str(hf_cache / "train.tar.gz")],
        ),
        patch("nemo_curator.stages.audio.datasets.fleurs.create_initial_manifest.extract_archive"),
    ):
        ok = prep.stage_dataset(output_path, lang="hy_am", split="train", cache_dir=None)

    assert ok is True
    assert (output_path / "hy_am" / "train.tsv").is_file()


def test_process_transcript_parses_tsv(tmp_path: Path) -> None:
    stage_cls, _ = _import_stage_module()
    lang = "hy_am"
    split = "dev"
    raw_dir = tmp_path / "fleurs"
    audio_dir = raw_dir / split
    audio_dir.mkdir(parents=True)

    tsv_path = raw_dir / f"{split}.tsv"
    lines = [
        "idx\tfile1.wav\thello world\n",
        "badline\n",
        "idx\tfile2.wav\tsecond\n",
    ]
    tsv_path.write_text("".join(lines), encoding="utf-8")

    (audio_dir / "file1.wav").write_bytes(b"")
    (audio_dir / "file2.wav").write_bytes(b"")

    stage = stage_cls(lang=lang, split=split, raw_data_dir=raw_dir.as_posix())

    batches = stage.process_transcript(tsv_path.as_posix(), audio_dir.as_posix())

    assert len(batches) == 2
    b0, b1 = batches
    assert b0.data[stage.filepath_key].endswith(os.path.join(split, "file1.wav"))
    assert b0.data[stage.text_key] == "hello world"
    assert b1.data[stage.filepath_key].endswith(os.path.join(split, "file2.wav"))
    assert b1.data[stage.text_key] == "second"
