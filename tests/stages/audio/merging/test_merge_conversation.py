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

"""Conversation placement, publication, and executor integration tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.backends.ray_data.adapter import RayDataStageAdapter
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.backends.xenna.adapter import XennaStageAdapter
from nemo_curator.stages.audio.common.manifest_reader import ManifestReaderStage
from nemo_curator.stages.audio.merging import MergeConversationSDPStage
from nemo_curator.tasks import AudioTask, FileGroupTask, group_tasks_by_data_key

SR = 8000


def _turn(  # noqa: PLR0913
    tmp_path: Path,
    index: int = 0,
    speaker: str = "A",
    duration: float = 1.0,
    amplitude: float = 0.2,
    sr: int = SR,
    segments: list[tuple[float, float]] | None = None,
    words: list[tuple[float, float, str]] | None = None,
    **fields: object,
) -> dict[str, Any]:
    path = tmp_path / f"turn_{index}.wav"
    sf.write(path, np.full(round(duration * sr), amplitude), sr, subtype="FLOAT")
    entry = {"audio_filepath": str(path), "speaker": speaker, "turn_index": index, **fields}
    if segments is not None:
        rttm = path.with_suffix(".rttm")
        rttm.write_text(
            "".join(f"SPEAKER turn 1 {start} {length} <NA> <NA> {speaker} <NA> <NA>\n" for start, length in segments)
        )
        entry["rttm_filepath"] = str(rttm)
    if words is not None:
        ctm = path.with_suffix(".ctm")
        ctm.write_text("".join(f"turn 1 {start} {length} {word}\n" for start, length, word in words))
        entry["ctm_filepath"] = str(ctm)
    return entry


def _parent(turns: object, conversation_id: str = "conversation") -> AudioTask:
    return AudioTask(data={"conversation_id": conversation_id, "turns": turns}, dataset_name="test")


def _stage(tmp_path: Path, **kwargs: float | bool) -> MergeConversationSDPStage:
    return MergeConversationSDPStage(str(tmp_path / "output"), **kwargs)


def _merge(tmp_path: Path, turns: object, **kwargs: float | bool) -> dict[str, Any]:
    result = _stage(tmp_path, **kwargs).process(_parent(turns))
    assert isinstance(result, AudioTask)
    return result.data


def _rows(path: str) -> list[list[str]]:
    return [line.split() for line in Path(path).read_text().splitlines()]


@pytest.mark.parametrize("backend", ["ray_data", "xenna"])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_manifest_conversations_survive_adapter_batches(tmp_path: Path, backend: str, batch_size: int) -> None:
    rows = []
    for length in (1, 2, 5):
        turns = [_turn(tmp_path, index=10 * length + i, speaker=str(i)) for i in range(length)]
        rows.append({"conversation_id": f"conversation_{length}", "turns": turns[::-1]})
    manifest = tmp_path / "conversations.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    source = FileGroupTask(data=[str(manifest)], dataset_name="test")
    tasks = ManifestReaderStage().process(source)
    stage = _stage(tmp_path).with_(batch_size=batch_size)
    if backend == "ray_data":
        adapter = RayDataStageAdapter(stage)
        assert adapter.batch_size == batch_size
    else:
        adapter = XennaStageAdapter(stage)
        assert adapter.stage_batch_size == batch_size
    output = []
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i : i + batch_size]
        if backend == "ray_data":
            output.extend(adapter._process_batch_internal({"item": np.array(batch, dtype=object)})["item"])
        else:
            output.extend(adapter.process_data(batch))
    assert len(output) == 3
    for task, length in zip(output, (1, 2, 5), strict=True):
        assert task.data["duration"] == length
        audio, sr = sf.read(task.data["audio_filepath"])
        assert sr == SR
        assert len(audio) == length * SR
        assert task.data["num_speakers"] == length


def test_preserves_parent_provenance(tmp_path: Path) -> None:
    task = _parent([_turn(tmp_path)])
    task._metadata = {"source_files": ["input.jsonl"]}
    original_id = task.task_id
    result = _stage(tmp_path).process(task)
    assert result is task
    assert result.task_id == original_id
    assert result._metadata == {"source_files": ["input.jsonl"]}


@pytest.mark.parametrize(
    "conversation_id", ["../escaped", "/absolute", "..", ".", "a/b", "a\\b", "", "two words", "a\x00b"]
)
def test_rejects_unsafe_conversation_ids_before_writes(tmp_path: Path, conversation_id: str) -> None:
    with pytest.raises(ValueError, match="conversation_id"):
        _stage(tmp_path).process(_parent([_turn(tmp_path)], conversation_id))
    assert not (tmp_path / "output").exists()


def test_rejects_flat_turn_tasks(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="turns"):
        _stage(tmp_path).process(AudioTask(data=_turn(tmp_path), dataset_name="test"))


def test_rejects_mixed_ids_before_writes(tmp_path: Path) -> None:
    turns = [_turn(tmp_path, conversation_id="conversation"), _turn(tmp_path, index=1, conversation_id="other")]
    with pytest.raises(ValueError, match="conversation_id"):
        _stage(tmp_path).process(_parent(turns))
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "turns",
    [
        None,
        {},
        [None],
        [{"speaker": ""}],
        [{"speaker": "A B"}],
        [{"speaker": "A", "turn_index": True}],
        [{"speaker": "A", "turn_index": 1.2}],
    ],
)
def test_rejects_malformed_turns(tmp_path: Path, turns: object) -> None:
    with pytest.raises(ValueError, match=r"turns|speaker|turn_index"):
        _stage(tmp_path).process(_parent(turns))


def test_rejects_duplicate_turn_indices(tmp_path: Path) -> None:
    turns = [_turn(tmp_path), _turn(tmp_path, speaker="B")]
    with pytest.raises(ValueError, match="turn_index"):
        _stage(tmp_path).process(_parent(turns))


def test_empty_conversation_is_filtered(tmp_path: Path) -> None:
    assert _stage(tmp_path).process(_parent([])) == []
    assert _stage(tmp_path).process_batch([]) == []


def test_speaker_labels_cannot_escape_or_collide_with_aggregates(tmp_path: Path) -> None:
    labels = ["mixed", "multichannel", "all", "../../escaped", "/absolute", "a/b", "a_b", "a\\b"]
    result = _merge(
        tmp_path,
        [
            _turn(tmp_path, index=i, speaker=label, amplitude=0.1, words=[(0, 0.5, "word")])
            for i, label in enumerate(labels)
        ],
    )
    root = Path(result["audio_filepath"]).parent
    assert set(result["speaker_artifacts"]) == set(labels)
    assert set(result["speaker_references"]) == set(labels)
    paths = []
    for i, label in enumerate(labels):
        artifact = result["speaker_artifacts"][label]
        path = Path(artifact["audio_filepath"])
        assert path.is_relative_to(root / "speakers")
        audio, _ = sf.read(path)
        assert audio.ndim == 1
        assert np.mean(audio[i * SR : (i + 1) * SR]) == pytest.approx(0.1, abs=4e-5)
        assert len(_rows(artifact["rttm_filepath"])) == 1
        assert len(_rows(artifact["ctm_filepath"])) == 1
        assert artifact["channel"] == i
        paths.append(path)
    assert len(set(paths)) == len(labels)
    multichannel, _ = sf.read(root / "multichannel.wav", always_2d=True)
    assert multichannel.shape == (len(labels) * SR, len(labels))
    assert len(_rows(result["rttm_filepath"])) == len(labels)
    assert len(_rows(result["ctm_filepath"])) == len(labels)


@pytest.mark.parametrize("amplitude", [0.2, 0.8])
def test_returning_speaker_overlap_accumulates_without_clipping(tmp_path: Path, amplitude: float) -> None:
    turns = [
        _turn(tmp_path, duration=2, amplitude=amplitude),
        _turn(tmp_path, index=1, speaker="B", duration=0.5, amplitude=0.1, overlap=0.5),
        _turn(tmp_path, index=2, amplitude=2 * amplitude, overlap=0.5),
    ]
    result = _merge(tmp_path, turns)
    channel, _ = sf.read(result["speaker_artifacts"]["A"]["audio_filepath"])
    gain = min(1, 0.99 / (3 * amplitude))
    assert channel[SR // 2] == pytest.approx(amplitude * gain, abs=4e-5)
    assert channel[7 * SR // 4] == pytest.approx(3 * amplitude * gain, abs=4e-5)
    assert np.max(np.abs(channel)) <= 0.99001
    mixed, _ = sf.read(result["audio_filepath"])
    assert np.max(np.abs(mixed)) <= 0.99001
    rows = _rows(result["rttm_filepath"])
    assert [float(row[3]) for row in rows] == [0, 1.5, 1.5]
    assert result["duration"] == 2.5


@pytest.mark.parametrize("middle", ["rate", "unreadable", "missing", "empty", "nonfinite"])
def test_invalid_audio_fails_whole_conversation_without_publication(tmp_path: Path, middle: str) -> None:
    turns = [
        _turn(tmp_path),
        _turn(tmp_path, index=1, speaker="B", sr=16000 if middle == "rate" else SR),
        _turn(tmp_path, index=2),
    ]
    path = Path(turns[1]["audio_filepath"])
    if middle == "unreadable":
        path.write_text("invalid WAV")
    elif middle == "missing":
        path.unlink()
    elif middle == "empty":
        sf.write(path, np.array([]), SR)
    elif middle == "nonfinite":
        sf.write(path, np.array([np.nan]), SR, subtype="FLOAT")
    task = _parent(turns)
    original = copy.deepcopy(task.data)
    assert _stage(tmp_path).process(task) == []
    assert task.data == original
    assert not (tmp_path / "output" / "conversation").exists()


def test_full_audio_fallback_advances_annotations_and_keeps_ctm(tmp_path: Path) -> None:
    turns = [
        _turn(tmp_path, words=[(0.1, 0.2, "hello")], reference_voice="voice"),
        _turn(tmp_path, index=1, speaker="B", segments=[(0, 1)], words=[(0.2, 0.2, "world")]),
    ]
    result = _merge(tmp_path, turns, seglst_offset=0)
    assert result["duration"] == 2
    assert result["mfa_fallback"] is True
    assert [float(row[3]) for row in _rows(result["rttm_filepath"])] == [0, 1]
    assert [float(row[2]) for row in _rows(result["ctm_filepath"])] == pytest.approx([0.1, 1.2])
    seglst = json.loads(Path(result["seglst_filepath"]).read_text())
    assert [(row["start_time"], row["end_time"]) for row in seglst] == [(0, 1), (1, 2)]
    assert result["speaker_references"]["A"]["reference_voice"] == "voice"


def test_rttm_clipping_and_removed_silence_share_ctm_mapping(tmp_path: Path) -> None:
    turn = _turn(
        tmp_path,
        duration=4,
        segments=[(-0.1, 0.6), (0.4, 0.6), (3.5, 2)],
        words=[(-0.1, 0.3, "begin"), (2, 0.2, "removed"), (3.8, 1, "end")],
        text="begin end",
    )
    result = _merge(tmp_path, [turn], max_intra_turn_pause=0.1, seglst_offset=0)
    assert result["duration"] == 1.5
    rttm = _rows(result["rttm_filepath"])
    assert [(float(row[3]), float(row[4])) for row in rttm] == [(0, 1), (1, 0.5)]
    ctm = _rows(result["ctm_filepath"])
    assert [row[4] for row in ctm] == ["begin", "end"]
    assert [(float(row[2]), float(row[3])) for row in ctm] == pytest.approx([(0, 0.2), (1.3, 0.2)])
    assert result["mfa_fallback"] is False


def test_short_pause_is_retained_in_audio_and_annotations(tmp_path: Path) -> None:
    turn = _turn(tmp_path, duration=2, segments=[(0.25, 0.5), (1, 0.5)], words=[(1.1, 0.2, "word")])
    result = _merge(tmp_path, [turn], max_intra_turn_pause=0.3)
    assert result["duration"] == 1.25
    assert float(_rows(result["rttm_filepath"])[1][3]) == 0.75
    assert float(_rows(result["ctm_filepath"])[0][2]) == 0.85


@pytest.mark.parametrize("randomize", [False, True])
def test_pause_cap_and_randomization_use_actual_audio_clock(tmp_path: Path, randomize: bool) -> None:
    turns = [_turn(tmp_path, segments=[(0, 1)]), _turn(tmp_path, index=1, speaker="B", overlap=-10, segments=[(0, 1)])]
    result = _merge(tmp_path, turns, max_pause_duration=0.75, randomize_pauses=randomize)
    start = float(_rows(result["rttm_filepath"])[1][3])
    assert 1.3 <= start <= 1.75
    if not randomize:
        assert start == 1.75
    assert result["duration"] == pytest.approx(start + 1, abs=1e-6)
    audio, _ = sf.read(result["speaker_artifacts"]["B"]["audio_filepath"])
    assert np.flatnonzero(audio)[0] == round(start * SR)


def test_no_ctm_is_reported_as_empty_path(tmp_path: Path) -> None:
    result = _merge(tmp_path, [_turn(tmp_path)])
    assert result["ctm_filepath"] == ""
    assert result["speaker_artifacts"]["A"]["ctm_filepath"] == ""


def test_same_id_rerun_refuses_existing_artifacts(tmp_path: Path) -> None:
    stage = _stage(tmp_path)
    first = stage.process(_parent([_turn(tmp_path, speaker="Alice", words=[(0, 0.5, "old")])]))
    root = Path(first.data["audio_filepath"]).parent
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert stage.process(_parent([_turn(tmp_path, index=1, speaker="Bob")])) == []
    assert stage.process(_parent([{"speaker": "Bob", "turn_index": 0, "audio_filepath": "/missing.wav"}])) == []
    after = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert before == after


def test_symlink_destination_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "output"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "conversation").symlink_to(outside, target_is_directory=True)
    assert _stage(tmp_path).process(_parent([_turn(tmp_path)])) == []
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("failure", ["audio", "seglst"])
def test_write_failure_does_not_publish_partial_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    stage = _stage(tmp_path)
    turns = [_turn(tmp_path)]

    def fail(*_args: object, **_kwargs: object) -> None:
        msg = "disk write failed"
        raise OSError(msg)

    if failure == "audio":
        monkeypatch.setattr(sf, "write", fail)
    else:
        monkeypatch.setattr(stage, "_generate_seglst", fail)
    task = _parent(turns)
    assert stage.process(task) == []
    assert "turns" in task.data
    assert not (tmp_path / "output" / "conversation").exists()
    assert list((tmp_path / "output").iterdir()) == []


@pytest.mark.parametrize("name", ["max_pause_duration", "max_intra_turn_pause", "seglst_offset"])
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_rejects_invalid_timing_configuration(tmp_path: Path, name: str, value: float) -> None:
    with pytest.raises(ValueError, match=name):
        _stage(tmp_path, **{name: value})


@pytest.mark.parametrize("executor_class", [RayDataExecutor, XennaExecutor])
def test_complete_conversations_through_executor(tmp_path: Path, executor_class: type) -> None:
    tasks = [
        _parent([_turn(tmp_path, index=100 * length + i, duration=0.05) for i in range(length)], f"executor_{length}")
        for length in (1, 3, 7)
    ]
    executor = executor_class({"execution_mode": "batch"}) if executor_class is XennaExecutor else executor_class()
    output = executor.execute([_stage(tmp_path)], initial_tasks=tasks)
    assert len(output) == len(tasks)
    by_id = {task.data["conversation_id"]: task.data for task in output}
    for length in (1, 3, 7):
        result = by_id[f"executor_{length}"]
        assert sf.info(result["audio_filepath"]).frames == 400 * length
        assert result["duration"] == pytest.approx(0.05 * length)
        assert len(_rows(result["rttm_filepath"])) == length


@pytest.mark.parametrize("annotation", ["missing", "malformed", "outside"])
def test_unusable_rttm_falls_back_to_complete_audio(tmp_path: Path, annotation: str) -> None:
    turn = _turn(tmp_path, words=[(0.1, 0.2, "kept")])
    path = tmp_path / "bad.rttm"
    turn["rttm_filepath"] = str(path)
    if annotation == "malformed":
        path.write_text("SPEAKER f 1 nan 1\nSPEAKER f 1 0 -1\nSPEAKER f 1 nope 1\nshort\n")
    elif annotation == "outside":
        path.write_text("SPEAKER f 1 2 1\n")
    result = _merge(tmp_path, [turn])
    assert result["duration"] == 1
    assert result["mfa_fallback"] is True
    assert _rows(result["ctm_filepath"])[0][4] == "kept"


def test_stereo_turn_is_downmixed_before_placement(tmp_path: Path) -> None:
    turn = _turn(tmp_path, segments=[(0, 1)])
    sf.write(turn["audio_filepath"], np.tile([0.1, 0.3], (SR, 1)), SR, subtype="FLOAT")
    result = _merge(tmp_path, [turn])
    mono, _ = sf.read(result["audio_filepath"])
    assert mono.shape == (SR,)
    assert np.mean(mono) == pytest.approx(0.2, abs=4e-5)
    assert result["mfa_fallback"] is False


def test_words_crossing_cut_pause_stay_within_retained_audio(tmp_path: Path) -> None:
    turn = _turn(tmp_path, duration=3, segments=[(0, 0.5), (2, 0.5)], words=[(0.25, 2, "crossing")])
    result = _merge(tmp_path, [turn], max_intra_turn_pause=0)
    word = _rows(result["ctm_filepath"])[0]
    assert float(word[2]) == 0.25
    assert float(word[3]) == 0.5
    assert result["duration"] == 1


@pytest.mark.parametrize("overlap", [100, 1e308])
def test_clamped_overlap_uses_rounded_samples_for_all_sidecars(tmp_path: Path, overlap: float) -> None:
    turns = [
        _turn(tmp_path, duration=0.100125, segments=[(0, 1)]),
        _turn(
            tmp_path, index=1, speaker="B", duration=0.01, overlap=overlap, segments=[(0, 1)], words=[(0, 1, "word")]
        ),
    ]
    result = _merge(tmp_path, turns, seglst_offset=0)
    assert result["duration"] == 801 / SR
    assert float(_rows(result["rttm_filepath"])[1][3]) == 721 / SR
    assert float(_rows(result["ctm_filepath"])[0][2]) == 721 / SR
    assert float(_rows(result["ctm_filepath"])[0][3]) == 80 / SR


def test_mfa_skipped_marks_fallback_even_with_valid_rttm(tmp_path: Path) -> None:
    result = _merge(tmp_path, [_turn(tmp_path, segments=[(0, 1)], mfa_skipped=True)])
    assert result["mfa_fallback"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [("audio_filepath", None), ("rttm_filepath", []), ("ctm_filepath", 0), ("overlap", float("nan"))],
)
def test_rejects_invalid_turn_fields_before_writes(tmp_path: Path, field: str, value: object) -> None:
    turn = _turn(tmp_path)
    turn[field] = value
    with pytest.raises(ValueError, match=field):
        _stage(tmp_path).process(_parent([turn]))
    assert not (tmp_path / "output").exists()


def test_large_finite_annotation_times_are_clipped_before_sample_conversion(tmp_path: Path) -> None:
    turn = _turn(tmp_path, segments=[(1e308, 1)], words=[(0.25, 1e308, "clipped"), (1e308, 1, "outside")])
    result = _merge(tmp_path, [turn])
    assert result["mfa_fallback"] is True
    assert _rows(result["ctm_filepath"]) == [["conversation", "1", "0.250000", "0.750000", "clipped"]]


@pytest.mark.parametrize("backend", ["ray_data", "xenna"])
def test_grouped_conversations_keep_good_groups_after_bad_audio(tmp_path: Path, backend: str) -> None:
    tasks = []
    for i, conversation in enumerate(("a-good", "b-bad", "c-good")):
        entry = _turn(tmp_path, index=i, conversation_id=conversation)
        if conversation == "b-bad":
            Path(entry["audio_filepath"]).write_text("not a wave file")
        tasks.append(AudioTask(data=entry, dataset_name="test"))
    groups = group_tasks_by_data_key(tasks, "conversation_id")
    stage = _stage(tmp_path)
    if backend == "ray_data":
        output = RayDataStageAdapter(stage)._process_batch_internal({"item": np.array(groups, dtype=object)})["item"]
    else:
        output = XennaStageAdapter(stage).process_data(groups)
    assert [task.data["conversation_id"] for task in output] == ["a-good", "c-good"]
    assert not (tmp_path / "output" / "b-bad").exists()
    assert all("member_metadata" in task._metadata for task in output)


def test_incomplete_generated_conversation_is_not_published(tmp_path: Path) -> None:
    entry = _turn(tmp_path, conversation_id="conversation", conversation_num_turns=2)
    group = group_tasks_by_data_key([AudioTask(data=entry)], "conversation_id")[0]
    assert _stage(tmp_path).process(group) == []
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("rates", [(8000, 16000), (16000, 8000)])
def test_mismatched_sample_rates_are_rejected_in_both_orders(tmp_path: Path, rates: tuple[int, int]) -> None:
    turns = [_turn(tmp_path, index=i, sr=rate) for i, rate in enumerate(rates)]
    assert _stage(tmp_path).process(_parent(turns)) == []
    assert not (tmp_path / "output").exists()
