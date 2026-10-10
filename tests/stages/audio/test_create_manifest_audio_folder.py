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

"""Unit tests for CreateInitialManifestAudioFolderStage (generic local-folder source)."""

import os
from pathlib import Path

import pytest

from nemo_curator.backends import base as backend_base
from nemo_curator.backends.base import BaseStageAdapter
from nemo_curator.stages.audio.common import CreateInitialManifestAudioFolderStage
from nemo_curator.tasks import EmptyTask


def _touch(root: str, rel: str) -> None:
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "wb").close()  # placeholder; the stage only collects paths, never decodes


class TestCreateInitialManifestAudioFolderStage:
    def test_recursive_collects_audio_only_one_task_each(self, tmp_path) -> None:  # noqa: ANN001
        root = str(tmp_path)
        for rel in ["a.wav", "b.FLAC", "notes.txt", "sub/c.mp3"]:
            _touch(root, rel)
        tasks = CreateInitialManifestAudioFolderStage(data_dir=root).process(None)
        names = sorted(os.path.basename(t.data["audio_filepath"]) for t in tasks)
        assert names == ["a.wav", "b.FLAC", "c.mp3"]  # .txt excluded; subdir included; ext match is case-insensitive
        assert all(t.data["audio_item_id"] for t in tasks)
        assert all(os.path.isabs(t.data["audio_filepath"]) for t in tasks)

    def test_same_filename_in_two_folders_gets_two_ids(self, tmp_path) -> None:  # noqa: ANN001
        root = str(tmp_path)
        for rel in ["spk1/utt1.wav", "spk2/utt1.wav"]:
            _touch(root, rel)

        tasks = CreateInitialManifestAudioFolderStage(data_dir=root).process(None)
        ids = sorted(t.data["audio_item_id"] for t in tasks)

        assert ids == ["spk1__utt1~ewav", "spk2__utt1~ewav"], ids

    def test_flattened_path_and_literal_separator_get_distinct_ids(self, tmp_path) -> None:  # noqa: ANN001
        root = str(tmp_path)
        for rel in ["spk1/utt1.wav", "spk1__utt1.wav"]:
            _touch(root, rel)

        tasks = CreateInitialManifestAudioFolderStage(data_dir=root).process(None)
        ids_by_path = {
            os.path.relpath(task.data["audio_filepath"], root): task.data["audio_item_id"] for task in tasks
        }

        assert ids_by_path[os.path.join("spk1", "utt1.wav")] == "spk1__utt1~ewav"
        assert ids_by_path["spk1__utt1.wav"] == "spk1~u~uutt1~ewav"
        assert len(set(ids_by_path.values())) == 2

    def test_flat_folder_ids_remain_stable_with_extensions(self, tmp_path) -> None:  # noqa: ANN001
        """Extensions keep identity independent of other files selected by a scan."""
        root = str(tmp_path)
        for rel in ["a.wav", "b.wav"]:
            _touch(root, rel)

        tasks = CreateInitialManifestAudioFolderStage(data_dir=root).process(None)

        assert sorted(t.data["audio_item_id"] for t in tasks) == ["a~ewav", "b~ewav"]

    def test_same_stem_with_different_extensions_gets_two_ids(self, tmp_path) -> None:  # noqa: ANN001
        root = str(tmp_path)
        for rel in ["a.wav", "a.flac"]:
            _touch(root, rel)

        tasks = CreateInitialManifestAudioFolderStage(data_dir=root).process(None)
        ids_by_name = {os.path.basename(task.data["audio_filepath"]): task.data["audio_item_id"] for task in tasks}

        assert ids_by_name == {"a.flac": "a~eflac", "a.wav": "a~ewav"}
        assert len(set(ids_by_name.values())) == 2

    @pytest.mark.parametrize("suffix", [".flac", ".WAV", ".mp3"])
    def test_ids_survive_full_delta_and_bounded_scans(self, tmp_path: Path, suffix: str) -> None:
        wav = tmp_path / "a.wav"
        wav.touch()
        retained = CreateInitialManifestAudioFolderStage(str(tmp_path)).process(None)[0]
        added = tmp_path / f"a{suffix}"
        added.touch()
        full = CreateInitialManifestAudioFolderStage(str(tmp_path)).process(None)
        delta = CreateInitialManifestAudioFolderStage(str(tmp_path), include_files=[str(added)]).process(None)[0]
        by_path = {row.data["audio_filepath"]: row.data["audio_item_id"] for row in full}
        assert by_path[str(wav)] == retained.data["audio_item_id"]
        assert by_path[str(added)] == delta.data["audio_item_id"]
        assert by_path[str(wav)] != by_path[str(added)]
        bounded = CreateInitialManifestAudioFolderStage(str(tmp_path), max_samples=1).process(None)[0]
        assert bounded.data["audio_item_id"] == by_path[bounded.data["audio_filepath"]]

    def test_resume_does_not_skip_a_new_earlier_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        later = tmp_path / "b.wav"
        later.touch()
        source = CreateInitialManifestAudioFolderStage(str(tmp_path))
        source.is_source_stage = True
        adapter = BaseStageAdapter(source)
        parent = EmptyTask()
        parent.task_id = "0"
        first = adapter._post_process_task_ids([parent], source.process(parent))
        completed = first[0].get_source_id()
        earlier = tmp_path / "a.wav"
        earlier.touch()
        second = adapter._post_process_task_ids([parent], source.process(parent))
        monkeypatch.setattr(backend_base, "completed_resumability_sources", lambda _ids: {completed})
        monkeypatch.setattr(backend_base, "flush_resumability_deltas", lambda _deltas: None)
        survivors = adapter._source_counters(second)
        assert [row.data["audio_filepath"] for row in survivors] == [str(earlier)]
        assert second[1].get_source_id() == completed

    def test_non_recursive_and_max_samples(self, tmp_path) -> None:  # noqa: ANN001
        root = str(tmp_path)
        for rel in ["a.wav", "b.wav", "sub/c.wav"]:
            _touch(root, rel)
        tasks = CreateInitialManifestAudioFolderStage(data_dir=root, recursive=False, max_samples=1).process(None)
        assert len(tasks) == 1  # sub/ excluded (non-recursive), capped to 1
        assert tasks[0].data["audio_filepath"].endswith(".wav")

    def test_extension_filter(self, tmp_path) -> None:  # noqa: ANN001
        root = str(tmp_path)
        for rel in ["a.wav", "b.mp3"]:
            _touch(root, rel)
        tasks = CreateInitialManifestAudioFolderStage(data_dir=root, extensions=[".mp3"]).process(None)
        assert [os.path.basename(t.data["audio_filepath"]) for t in tasks] == ["b.mp3"]

    def test_missing_dir_returns_empty(self, tmp_path) -> None:  # noqa: ANN001
        tasks = CreateInitialManifestAudioFolderStage(data_dir=str(tmp_path / "nope")).process(None)
        assert tasks == []

    def test_requires_data_dir(self) -> None:
        with pytest.raises(ValueError):  # noqa: PT011
            CreateInitialManifestAudioFolderStage(data_dir="")

    def test_output_keys_must_be_distinct(self, tmp_path) -> None:  # noqa: ANN001
        with pytest.raises(ValueError, match="Output keys must be distinct"):
            CreateInitialManifestAudioFolderStage(
                data_dir=str(tmp_path),
                audio_filepath_key="audio",
                audio_item_id_key="audio",
            )

    def test_fanout_preserves_parent_provenance(self, tmp_path) -> None:  # noqa: ANN001
        root = str(tmp_path)
        _touch(root, "a.wav")
        parent = EmptyTask(_metadata={"trace": "seed"}, _stage_perf=["upstream"])

        [child] = CreateInitialManifestAudioFolderStage(data_dir=root).process(parent)

        assert all(child._metadata[key] == value for key, value in parent._metadata.items())
        assert "audio_folder_source_id" in child._metadata
        assert "audio_folder_source_id" not in parent._metadata
        assert child._stage_perf == parent._stage_perf
        assert child._stage_perf is not parent._stage_perf

    def test_contract_writes_filepath_and_no_disk_write(self) -> None:
        c = CreateInitialManifestAudioFolderStage(data_dir="/tmp").describe()  # noqa: S108
        assert "audio_filepath" in c.writes.data_keys
        assert c.gates.writes_to_disk is False  # references existing files; no disk write


def test_added_earlier_file_does_not_reuse_completed_source_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nemo_curator.backends.base import BaseStageAdapter
    from nemo_curator.tasks import EmptyTask

    (tmp_path / "b.wav").touch()
    stage = CreateInitialManifestAudioFolderStage(str(tmp_path))
    stage.is_source_stage = True
    adapter = BaseStageAdapter(stage)
    initial = adapter._post_process_task_ids([EmptyTask()], stage.process(None))
    completed_id = initial[0].get_source_id()
    (tmp_path / "a.wav").touch()
    rows = adapter._post_process_task_ids([EmptyTask()], stage.process(None))
    assert rows[1].get_source_id() == completed_id
    monkeypatch.setattr("nemo_curator.backends.base.completed_resumability_sources", lambda _: {completed_id})
    monkeypatch.setattr("nemo_curator.backends.base.flush_resumability_deltas", lambda _: None)
    survivors = adapter._source_counters(rows)
    assert [Path(t.data["audio_filepath"]).name for t in survivors] == ["a.wav"]


def test_full_delta_and_restricted_folder_ids_do_not_collide(tmp_path: Path) -> None:
    (tmp_path / "a.wav").touch()
    initial = CreateInitialManifestAudioFolderStage(str(tmp_path)).process(None)[0]
    (tmp_path / "a.flac").touch()
    delta = CreateInitialManifestAudioFolderStage(str(tmp_path), include_files=[str(tmp_path / "a.flac")]).process(
        None
    )
    full = CreateInitialManifestAudioFolderStage(str(tmp_path)).process(None)
    ids = {task.data["audio_filepath"]: task.data["audio_item_id"] for task in full}
    assert len(set(ids.values())) == 2
    assert initial.data["audio_item_id"] == ids[initial.data["audio_filepath"]]
    assert delta[0].data["audio_item_id"] == ids[delta[0].data["audio_filepath"]]
    assert initial.get_deterministic_id() != delta[0].get_deterministic_id()


def test_flat_folder_ids_include_the_extension(tmp_path) -> None:  # noqa: ANN001
    """Identity includes extension regardless of other files in the selected cohort."""
    root = str(tmp_path)
    for rel in ["a.wav", "b.wav"]:
        _touch(root, rel)

    tasks = CreateInitialManifestAudioFolderStage(data_dir=root).process(None)

    assert sorted(t.data["audio_item_id"] for t in tasks) == ["a~ewav", "b~ewav"]
