# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path

import pytest

from nemo_curator.stages.audio.audio_agent.profiler import profile_data
from nemo_curator.stages.audio.audio_agent.semantic_review import _data_profile_summary


@pytest.mark.parametrize("selection", [None, [], ["b.wav"]])
def test_manifest_profile_matches_reader_selection(tmp_path: Path, selection: list[str] | None) -> None:
    from nemo_curator.stages.audio.common import ManifestReaderStage
    from nemo_curator.tasks import FileGroupTask

    paths = {name: tmp_path / name for name in ("a.wav", "b.wav")}
    for path in paths.values():
        path.write_bytes(b"stat fixture")
    rows = [{"recording": str(paths[name]), "duration": i} for i, name in enumerate(("a.wav", "b.wav", "b.wav"))]
    manifest = tmp_path / "input.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    include = None if selection is None else [str(paths[name]) for name in selection]
    reader = ManifestReaderStage(include_files=include, include_files_key="recording")
    emitted = reader.process(FileGroupTask(dataset_name="test", data=[str(manifest)]))

    profile = profile_data(str(manifest), max_probe=0, audio_filepath_key="recording", include_files=include)

    assert profile.num_files == len(emitted)
    assert set(profile.inventory) == {Path(task.data["recording"]).name for task in emitted}
    assert not profile.source_errors


@pytest.mark.parametrize("max_probe", [0, 1])
@pytest.mark.parametrize("source_kind", ["manifest", "relative_manifest", "folder", "definition"])
def test_workspace_references_are_checked_before_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, max_probe: int, source_kind: str
) -> None:
    import builtins

    import soundfile

    from nemo_curator.stages.audio.audio_agent import profiler

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"must not be inspected")
    alias = workspace / "alias.wav"
    alias.symlink_to(outside)
    monkeypatch.setenv("AUDIO_AGENT_WORKSPACE", str(workspace))
    original_open = builtins.open
    original_stat_entry = profiler._stat_entry

    def checked_open(path: object, *args: object, **kwargs: object) -> object:
        if isinstance(path, (str, Path)):
            assert Path(path).resolve() != outside
        return original_open(path, *args, **kwargs)

    def checked_stat(path: str, **kwargs: str) -> tuple[str, bool]:
        assert Path(path).resolve() != outside
        return original_stat_entry(path, **kwargs)

    monkeypatch.setattr(builtins, "open", checked_open)
    monkeypatch.setattr(profiler, "_stat_entry", checked_stat)
    monkeypatch.setattr(soundfile, "info", lambda *_a, **_kw: pytest.fail("outside audio was probed"))
    kwargs = {}
    source = workspace
    if source_kind.endswith("manifest"):
        source = workspace / "input.jsonl"
        reference = str(outside) if source_kind == "manifest" else "alias.wav"
        source.write_text(json.dumps({"audio_filepath": reference}) + "\n")
        if source_kind == "relative_manifest":
            kwargs = {"audio_dir": str(workspace), "audio_path_resolution": "relative"}
    elif source_kind == "definition":
        alias.unlink()
        kwargs = {"identity_files": [str(outside)]}

    profile = profile_data(str(source), max_probe=max_probe, **kwargs)

    assert any("outside the allowed workspace" in error for error in profile.source_errors)
    assert not profile.stat_digest
    assert not profile.inventory


def test_folder_profile_does_not_claim_sibling_transcripts_absent(tmp_path: Path) -> None:
    sidecar = tmp_path / "labels.jsonl"
    sidecar.write_text('{"text": "a supplied transcript"}\n')
    before = sidecar.read_bytes()
    profile = profile_data(str(tmp_path), max_probe=0)
    assert profile.has_transcripts is False
    assert any("transcript_presence=not_inspected" in note for note in profile.notes)
    summary = _data_profile_summary(profile.to_dict())
    assert summary["notes"] == profile.notes
    assert sidecar.read_bytes() == before


def test_manifest_profile_still_reports_observed_transcript_values(tmp_path: Path) -> None:
    manifest = tmp_path / "labels.jsonl"
    manifest.write_text('{"text": "a supplied transcript"}\n')
    profile = profile_data(str(manifest), max_probe=0)
    assert profile.has_transcripts is True
    assert not any("transcript_presence=not_inspected" in note for note in profile.notes)
