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

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
import pytest
import soundfile as sf

import nemo_curator.stages.audio.io.nemo_speech_writer as writer_module
from nemo_curator.stages.audio.io._nemo_speech_state import register_nemo_speech_shard
from nemo_curator.stages.audio.io.nemo_speech_writer import (
    NeMoSpeechWriterStage,
    finalize_nemo_speech_output,
)
from nemo_curator.tasks import AudioTask

_SAMPLE_RATE = 16_000
_SHARD_KEY = "corpus/manifests/shard_000"


def _tone(duration: float = 0.25) -> np.ndarray:
    samples = np.arange(round(duration * _SAMPLE_RATE), dtype=np.float32)
    return (0.2 * np.sin(2 * np.pi * 440 * samples / _SAMPLE_RATE)).astype(np.float32)


def _task(
    *,
    input_id: str = "0",
    source: str = "s3://speech-bucket/set_a/clip.wav",
    shard_key: str = _SHARD_KEY,
    shard_total: int = 1,
    output_slot: str | None = None,
    **overrides: object,
) -> AudioTask:
    waveform = _tone()
    data = {
        "audio_filepath": source,
        "original_file": source,
        "sample_rate": _SAMPLE_RATE,
        "sampling_rate": _SAMPLE_RATE,
        "duration_sec": len(waveform) / _SAMPLE_RATE,
        "start_ms": 0,
        "waveform": waveform,
    }
    data.update(overrides)
    if data["waveform"] is None:
        del data["waveform"]
    return AudioTask(
        dataset_name="test",
        data=data,
        _metadata={
            "_shard_key": shard_key,
            "_shard_total": shard_total,
            "_shard_input_id": input_id,
            "_shard_output_id": output_slot or f"row:{input_id}:{data.get('start_ms', 0)}",
        },
    )


def _manifest_path(output_dir: Path, shard_key: str = _SHARD_KEY) -> Path:
    return output_dir / f"{shard_key}.jsonl"


def _marker_path(output_dir: Path, shard_key: str = _SHARD_KEY) -> Path:
    return output_dir / f"{shard_key}.jsonl.done"


def _receipt_paths(output_dir: Path) -> list[Path]:
    receipt_root = output_dir / ".nemo_curator" / "nemo_speech_rows"
    return sorted(receipt_root.rglob("*.json"))


def _read_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _writer(output_dir: Path) -> NeMoSpeechWriterStage:
    writer = NeMoSpeechWriterStage(output_dir=str(output_dir))
    writer.setup()
    return writer


def test_writes_real_decodable_opus_then_finalizes_manifest_and_marker(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    writer = _writer(output_dir)
    task = _task(text="hello", score=np.float32(0.75))

    assert writer.process(task) is None

    relative_audio = PurePosixPath(task.data["output_audio_filepath"])
    opus_path = output_dir / relative_audio
    decoded, sample_rate = sf.read(opus_path, dtype="float32")
    assert sample_rate == _SAMPLE_RATE
    assert decoded.ndim == 1
    assert len(decoded) > 0
    assert np.max(np.abs(decoded)) > 0.01

    manifest_path = _manifest_path(output_dir)
    marker_path = _marker_path(output_dir)
    assert len(_receipt_paths(output_dir)) == 1
    owner_paths = list((output_dir / ".nemo_curator" / "nemo_speech_audio_owners").rglob("*.json"))
    assert len(owner_paths) == 2
    assert {json.loads(path.read_text(encoding="utf-8"))["audio_filepath"] for path in owner_paths} == {
        relative_audio.as_posix()
    }
    assert not manifest_path.exists()
    assert not marker_path.exists()

    assert finalize_nemo_speech_output(str(output_dir)) == [str(manifest_path)]
    rows = _read_manifest(manifest_path)
    manifest_audio = relative_audio.relative_to(PurePosixPath(_SHARD_KEY).parent)
    assert rows == [
        {
            "audio_filepath": manifest_audio.as_posix(),
            "duration": 0.25,
            "offset": 0.0,
            "original_audio_filepath": "s3://speech-bucket/set_a/clip.wav",
            "sample_rate": _SAMPLE_RATE,
            "sampling_rate": _SAMPLE_RATE,
            "score": 0.75,
            "text": "hello",
        }
    ]
    assert json.loads(marker_path.read_text(encoding="utf-8")) == {
        "completed_inputs": 1,
        "expected_inputs": 1,
        "manifest_rows": 1,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "shard_key": _SHARD_KEY,
        "version": 1,
    }


def test_saved_segment_resets_offset_and_preserves_absolute_source_extent(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    writer = _writer(output_dir)
    task = _task(offset=10.0, start_ms=2_000, end_ms=3_500)

    writer.process(task)
    finalize_nemo_speech_output(str(output_dir))

    row = _read_manifest(_manifest_path(output_dir))[0]
    assert row["offset"] == 0.0
    assert row["original_offset"] == 12.0
    assert row["original_end"] == 13.5


def test_saved_offset_only_row_preserves_source_offset(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    writer = _writer(output_dir)
    writer.process(_task(offset=10.0, start_ms=None))
    finalize_nemo_speech_output(str(output_dir))

    row = _read_manifest(_manifest_path(output_dir))[0]
    assert row["offset"] == 0.0
    assert row["original_offset"] == 10.0


def test_rewriting_saved_segment_keeps_inherited_source_extent(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    writer = _writer(output_dir)
    writer.process(
        _task(
            source="/previous/output/clip.opus",
            original_audio_filepath="s3://speech-bucket/original.wav",
            original_offset=12.0,
            offset=0.0,
            start_ms=1_000,
            end_ms=1_250,
        )
    )
    finalize_nemo_speech_output(str(output_dir))

    row = _read_manifest(_manifest_path(output_dir))[0]
    assert row["original_audio_filepath"] == "s3://speech-bucket/original.wav"
    assert row["original_offset"] == 13.0
    assert row["original_end"] == 13.25


def test_placeholder_rows_need_no_waveform_or_opus(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    writer = _writer(output_dir)
    writer.process(
        _task(
            input_id="silent",
            shard_total=2,
            waveform=None,
            vad_empty=True,
            source_lang="en",
        )
    )
    writer.process(
        _task(
            input_id="broken",
            source="s3://speech-bucket/set_a/broken.wav",
            shard_total=2,
            waveform=None,
            read_error=True,
            audio_too_long=True,
            duration=42.5,
        )
    )

    assert not list(output_dir.rglob("*.opus"))
    assert not _marker_path(output_dir).exists()
    finalize_nemo_speech_output(str(output_dir))

    rows = _read_manifest(_manifest_path(output_dir))
    vad_row = next(row for row in rows if row.get("vad_empty"))
    error_row = next(row for row in rows if row.get("read_error"))
    assert vad_row["audio_filepath"] == ""
    assert vad_row["duration"] == 0.0
    assert vad_row["source_lang"] == "en"
    assert error_row["audio_filepath"] == ""
    assert error_row["duration"] == 42.5
    assert error_row["audio_too_long"] is True


@pytest.mark.parametrize("unsafe_path", ["../escape.opus", "/absolute/escape.opus", "..\\escape.opus"])
def test_rejects_unsafe_preset_audio_paths(tmp_path: Path, unsafe_path: str) -> None:
    output_dir = tmp_path / "output"
    writer = _writer(output_dir)
    task = _task(output_audio_filepath=unsafe_path)

    with pytest.raises(ValueError, match="safe relative path"):
        writer.process(task)

    assert not (tmp_path / "escape.opus").exists()
    assert not _receipt_paths(output_dir)


def test_rejects_unsafe_shard_key(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    writer = _writer(output_dir)

    with pytest.raises(ValueError, match="Unsafe NeMo shard key"):
        writer.process(_task(shard_key="../outside"))

    assert not _receipt_paths(output_dir)


def test_rejects_output_path_through_symlink_outside_root(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    writer = _writer(output_dir)
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (output_dir / "corpus").symlink_to(outside_dir, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        writer.process(_task(output_audio_filepath=f"{_SHARD_KEY}/audio/clip.opus"))

    assert not list(outside_dir.rglob("*.opus"))
    assert not _receipt_paths(output_dir)


def test_distinct_absolute_sources_with_same_basename_get_safe_unique_paths(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    writer = _writer(output_dir)
    tasks = [
        _task(input_id="0", source="/mnt/set_a/clip.wav", shard_total=2),
        _task(input_id="1", source="/mnt/set_b/clip.wav", shard_total=2),
    ]

    for task in tasks:
        writer.process(task)

    relative_paths = [PurePosixPath(task.data["output_audio_filepath"]) for task in tasks]
    assert relative_paths[0] != relative_paths[1]
    assert all(not path.is_absolute() and ".." not in path.parts for path in relative_paths)
    assert all((output_dir / path).is_file() for path in relative_paths)
    finalize_nemo_speech_output(str(output_dir))
    assert {row["audio_filepath"] for row in _read_manifest(_manifest_path(output_dir))} == {
        path.relative_to(PurePosixPath(_SHARD_KEY).parent).as_posix() for path in relative_paths
    }


def test_incomplete_shard_prevents_publication_for_every_shard(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    writer = _writer(output_dir)
    complete_key = "a_complete/shard"
    incomplete_key = "z_incomplete/shard"
    writer.process(_task(shard_key=complete_key))
    writer.process(_task(shard_key=incomplete_key, shard_total=2))

    with pytest.raises(ValueError, match="incomplete: saw 1 of 2 inputs"):
        finalize_nemo_speech_output(str(output_dir))

    for shard_key in (complete_key, incomplete_key):
        assert not _manifest_path(output_dir, shard_key).exists()
        assert not _marker_path(output_dir, shard_key).exists()
    assert not list(output_dir.rglob("*.tmp"))


def test_registered_shard_with_all_outputs_dropped_is_rejected(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    register_nemo_speech_shard(output_dir, _SHARD_KEY, expected_inputs=1)

    with pytest.raises(ValueError, match="incomplete: saw 0 of 1 inputs"):
        finalize_nemo_speech_output(str(output_dir))

    assert not _manifest_path(output_dir).exists()
    assert not _marker_path(output_dir).exists()


def test_conflicting_registered_shard_totals_are_rejected(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    writer = _writer(output_dir)
    writer.process(_task(shard_total=1))

    with pytest.raises(ValueError, match="Conflicting NeMo speech shard registration"):
        writer.process(_task(input_id="1", shard_total=2))


def test_replay_is_idempotent_for_receipt_audio_manifest_and_marker(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    writer = _writer(output_dir)
    task = _task()

    writer.process(task)
    opus_path = output_dir / task.data["output_audio_filepath"]
    original_bytes = opus_path.read_bytes()
    writer.process(task)

    assert len(_receipt_paths(output_dir)) == 1
    assert len(list(output_dir.rglob("*.opus"))) == 1
    assert opus_path.read_bytes() == original_bytes
    assert finalize_nemo_speech_output(str(output_dir)) == [str(_manifest_path(output_dir))]
    original_manifest = _manifest_path(output_dir).read_bytes()
    original_marker = _marker_path(output_dir).read_bytes()

    assert finalize_nemo_speech_output(str(output_dir)) == []
    assert _manifest_path(output_dir).read_bytes() == original_manifest
    assert _marker_path(output_dir).read_bytes() == original_marker
    assert len(_read_manifest(_manifest_path(output_dir))) == 1


def test_fanout_never_marks_complete_until_explicit_finalize(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    writer = _writer(output_dir)
    first = _task(input_id="recording-0", start_ms=0)
    second = _task(input_id="recording-0", start_ms=1_000)

    writer.process(first)
    assert not _marker_path(output_dir).exists()
    assert not _manifest_path(output_dir).exists()
    writer.process(second)
    assert len(_receipt_paths(output_dir)) == 2
    assert not _marker_path(output_dir).exists()
    assert not _manifest_path(output_dir).exists()

    finalize_nemo_speech_output(str(output_dir))
    marker = json.loads(_marker_path(output_dir).read_text(encoding="utf-8"))
    assert marker["expected_inputs"] == 1
    assert marker["completed_inputs"] == 1
    assert marker["manifest_rows"] == 2
    assert len(_read_manifest(_manifest_path(output_dir))) == 2


def test_finalize_rejects_missing_audio_even_with_orphan_partial_temp(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    writer = _writer(output_dir)
    task = _task()
    writer.process(task)
    opus_path = output_dir / task.data["output_audio_filepath"]
    partial_path = opus_path.with_name(f".{opus_path.name}.interrupted.tmp")
    partial_path.write_bytes(opus_path.read_bytes()[:32])
    opus_path.unlink()

    with pytest.raises(ValueError, match="missing or invalid Opus"):
        finalize_nemo_speech_output(str(output_dir))

    assert partial_path.is_file()
    assert not _manifest_path(output_dir).exists()
    assert not _marker_path(output_dir).exists()


def test_process_replaces_corrupted_existing_opus_on_replay(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    writer = _writer(output_dir)
    task = _task()
    writer.process(task)
    opus_path = output_dir / task.data["output_audio_filepath"]
    opus_path.write_bytes(b"truncated-opus")

    writer.process(task)

    decoded, sample_rate = sf.read(opus_path, dtype="float32")
    assert sample_rate == _SAMPLE_RATE
    assert len(decoded) > 0
    finalize_nemo_speech_output(str(output_dir))
    assert _marker_path(output_dir).is_file()


def test_unbound_equal_duration_opus_is_replaced_and_cannot_be_published(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    writer = _writer(output_dir)
    relative = PurePosixPath(_SHARD_KEY, "audio", "preset.opus")
    opus_path = output_dir / relative
    opus_path.parent.mkdir(parents=True)
    sf.write(opus_path, -_tone(), _SAMPLE_RATE, format="OGG", subtype="OPUS")
    foreign_bytes = opus_path.read_bytes()
    task = _task(output_audio_filepath=relative.as_posix())

    writer.process(task)

    assert opus_path.read_bytes() != foreign_bytes
    sf.write(opus_path, -_tone(), _SAMPLE_RATE, format="OGG", subtype="OPUS")
    assert writer_module._valid_opus(opus_path, _SAMPLE_RATE, len(_tone()))
    with pytest.raises(ValueError, match="matching content binding"):
        finalize_nemo_speech_output(str(output_dir))


def test_replay_repairs_and_finalizer_rejects_duration_truncated_opus(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    writer = _writer(output_dir)
    waveform = _tone(duration=3.0)
    task = _task(waveform=waveform, duration_sec=3.0)
    writer.process(task)
    opus_path = output_dir / task.data["output_audio_filepath"]
    complete_bytes = opus_path.read_bytes()

    opus_path.write_bytes(complete_bytes[: len(complete_bytes) // 2])
    assert sf.info(opus_path).frames < len(waveform)
    assert not writer_module._valid_opus(opus_path, _SAMPLE_RATE, len(waveform))

    writer.process(task)
    assert writer_module._valid_opus(opus_path, _SAMPLE_RATE, len(waveform))

    repaired_bytes = opus_path.read_bytes()
    opus_path.write_bytes(repaired_bytes[: len(repaired_bytes) // 2])
    with pytest.raises(ValueError, match="missing or invalid Opus"):
        finalize_nemo_speech_output(str(output_dir))
    assert not _marker_path(output_dir).exists()


def test_validation_rejects_opus_missing_a_middle_page(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    writer = _writer(output_dir)
    waveform = _tone(duration=5.0)
    task = _task(waveform=waveform, duration_sec=5.0)
    writer.process(task)
    opus_path = output_dir / task.data["output_audio_filepath"]
    payload = opus_path.read_bytes()

    pages: list[tuple[int, int]] = []
    offset = 0
    while offset < len(payload):
        assert payload[offset : offset + 4] == b"OggS"
        segment_count = payload[offset + 26]
        body_offset = offset + 27 + segment_count
        page_end = body_offset + sum(payload[offset + 27 : body_offset])
        pages.append((offset, page_end))
        offset = page_end
    assert len(pages) >= 4
    page_start, page_end = pages[len(pages) // 2]
    opus_path.write_bytes(payload[:page_start] + payload[page_end:])

    assert not writer_module._valid_opus(opus_path, _SAMPLE_RATE, len(waveform))
    monkeypatch.setattr(writer_module.sf, "info", lambda _path: (_ for _ in ()).throw(RuntimeError("disabled")))
    assert not writer_module._valid_opus(opus_path, _SAMPLE_RATE, len(waveform))


def test_finalize_rejects_corrupted_final_opus(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    writer = _writer(output_dir)
    task = _task()
    writer.process(task)
    opus_path = output_dir / task.data["output_audio_filepath"]
    opus_path.write_bytes(b"not-an-opus-file")

    with pytest.raises(ValueError, match="missing or invalid Opus"):
        finalize_nemo_speech_output(str(output_dir))

    assert not _manifest_path(output_dir).exists()
    assert not _marker_path(output_dir).exists()


def test_manifest_only_preserves_original_reference_without_inventing_opus(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    writer = NeMoSpeechWriterStage(output_dir=str(output_dir), save_audio=False)
    writer.setup()
    source = "s3://speech-bucket/set_a/clip.wav"

    writer.process(
        _task(
            source=source,
            waveform=None,
            sample_rate=8_000,
            sampling_rate=8_000,
            offset=10.0,
            start_ms=2_000,
            end_ms=3_500,
        )
    )
    finalize_nemo_speech_output(str(output_dir))

    assert not list(output_dir.rglob("*.opus"))
    row = _read_manifest(_manifest_path(output_dir))[0]
    assert row["audio_filepath"] == source
    assert row["sample_rate"] == 8_000
    assert row["offset"] == 12.0
    assert row["original_end"] == 13.5


@pytest.mark.parametrize("output_dir", ["relative/output", "~/output", "s3://bucket/output"])
def test_rejects_non_absolute_or_uri_output_dirs(output_dir: str) -> None:
    with pytest.raises(ValueError, match="absolute local path"):
        NeMoSpeechWriterStage(output_dir=output_dir)

    with pytest.raises(ValueError, match="absolute local path"):
        finalize_nemo_speech_output(output_dir)


def test_preset_audio_path_must_be_in_dedicated_shard_audio_namespace(tmp_path: Path) -> None:
    writer = _writer(tmp_path / "output")

    with pytest.raises(ValueError, match="must be below"):
        writer.process(_task(output_audio_filepath=f"{_SHARD_KEY}.jsonl"))


def test_two_outputs_cannot_claim_the_same_preset_audio_path(tmp_path: Path) -> None:
    writer = _writer(tmp_path / "output")
    preset = f"{_SHARD_KEY}/audio/shared.opus"
    writer.process(_task(input_id="0", shard_total=2, output_slot="child-a", output_audio_filepath=preset))

    with pytest.raises(ValueError, match="Conflicting NeMo speech outputs claim"):
        writer.process(_task(input_id="1", shard_total=2, output_slot="child-b", output_audio_filepath=preset))


def test_same_input_fanout_without_offsets_uses_framework_child_identity(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    writer = _writer(output_dir)
    first = _task(input_id="recording", output_slot="source_0")
    second = _task(input_id="recording", output_slot="source_1")

    writer.process(first)
    writer.process(second)
    finalize_nemo_speech_output(str(output_dir))

    assert first.data["output_audio_filepath"] != second.data["output_audio_filepath"]
    assert len(_read_manifest(_manifest_path(output_dir))) == 2


def test_saved_fanout_is_sorted_by_source_offset(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    writer = _writer(output_dir)
    writer.process(_task(input_id="recording", start_ms=10_000, output_slot="late"))
    writer.process(_task(input_id="recording", start_ms=2_000, output_slot="early"))
    finalize_nemo_speech_output(str(output_dir))

    assert [row["original_offset"] for row in _read_manifest(_manifest_path(output_dir))] == [2.0, 10.0]


@pytest.mark.parametrize(
    "task_id",
    [
        "r0123456789abcdef0123456789abcdef",
        "r0123456789abcdef0123456789abcdef_0",
        "r0123456789abcdef0123456789abcdef_0_2",
    ],
)
def test_random_framework_task_ancestry_requires_explicit_output_slot(tmp_path: Path, task_id: str) -> None:
    output_dir = tmp_path / "output"
    writer = _writer(output_dir)
    task = _task()
    task._metadata.pop("_shard_output_id")
    task.task_id = task_id

    with pytest.raises(ValueError, match="non-deterministic framework task_id"):
        writer.process(task)

    assert not _receipt_paths(output_dir)


def test_random_framework_task_descendant_uses_explicit_output_slot(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    writer = _writer(output_dir)
    task = _task(output_slot="stable-child")
    task.task_id = "r0123456789abcdef0123456789abcdef_0"

    writer.process(task)

    receipt = json.loads(_receipt_paths(output_dir)[0].read_text(encoding="utf-8"))
    assert receipt["output_slot"] == "stable-child"


@pytest.mark.parametrize(
    "task_id",
    [
        "source_r0123456789abcdef0123456789abcdef_0",
        "prefixr0123456789abcdef0123456789abcdef_0",
        "r0123456789abcdef0123456789abcdefx_0",
    ],
)
def test_legitimate_framework_task_ids_containing_random_shape_are_accepted(
    tmp_path: Path,
    task_id: str,
) -> None:
    output_dir = tmp_path / "output"
    writer = _writer(output_dir)
    task = _task()
    task._metadata.pop("_shard_output_id")
    task.task_id = task_id

    writer.process(task)

    receipt = json.loads(_receipt_paths(output_dir)[0].read_text(encoding="utf-8"))
    assert receipt["output_slot"] == task_id


def test_replay_replaces_stale_placeholder_receipt_for_same_output_slot(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    writer = _writer(output_dir)
    writer.process(_task(waveform=None, read_error=True, output_slot="stable-child"))
    writer.process(_task(output_slot="stable-child", text="recovered"))

    assert len(_receipt_paths(output_dir)) == 2
    finalize_nemo_speech_output(str(output_dir))
    rows = _read_manifest(_manifest_path(output_dir))
    assert len(rows) == 1
    assert rows[0]["text"] == "recovered"
    assert "read_error" not in rows[0]


def test_replay_never_downgrades_success_to_placeholder_for_same_output_slot(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    writer = _writer(output_dir)
    writer.process(_task(output_slot="stable-child", text="durable"))
    writer.process(_task(waveform=None, read_error=True, output_slot="stable-child"))

    assert len(_receipt_paths(output_dir)) == 2
    finalize_nemo_speech_output(str(output_dir))
    rows = _read_manifest(_manifest_path(output_dir))
    assert len(rows) == 1
    assert rows[0]["text"] == "durable"
    assert "read_error" not in rows[0]


def test_republication_hides_stale_marker_before_manifest_replace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    writer = _writer(output_dir)
    writer.process(_task(waveform=None, read_error=True, output_slot="stable-child"))
    finalize_nemo_speech_output(str(output_dir))
    assert _marker_path(output_dir).is_file()

    writer.process(_task(output_slot="stable-child", text="recovered"))

    def fail_marker_write(*_args: object, **_kwargs: object) -> None:
        message = "injected marker publication failure"
        raise OSError(message)

    with monkeypatch.context() as context:
        context.setattr(writer_module, "write_json_atomically", fail_marker_write)
        with pytest.raises(OSError, match="injected marker"):
            finalize_nemo_speech_output(str(output_dir))

    assert not _marker_path(output_dir).exists()
    assert _read_manifest(_manifest_path(output_dir))[0]["text"] == "recovered"
    assert finalize_nemo_speech_output(str(output_dir)) == [str(_manifest_path(output_dir))]
    assert _marker_path(output_dir).is_file()


def test_audio_ownership_records_are_hash_sharded(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    writer = _writer(output_dir)
    writer.process(_task())

    owner_root = output_dir / ".nemo_curator" / "nemo_speech_audio_owners"
    owner_paths = list(owner_root.rglob("*.json"))
    assert len(owner_paths) == 2
    for path in owner_paths:
        relative_parts = path.relative_to(owner_root).parts
        assert len(relative_parts) == 3
        assert len(relative_parts[0]) == 2
        assert len(relative_parts[1]) == 2
        assert all(len(digest) == 64 for digest in Path(relative_parts[2]).stem.split("."))


def test_ffprobe_validation_fallback_accepts_real_opus(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    writer = _writer(output_dir)
    task = _task()
    writer.process(task)
    opus_path = output_dir / task.data["output_audio_filepath"]

    monkeypatch.setattr(writer_module.sf, "info", lambda _path: (_ for _ in ()).throw(RuntimeError("disabled")))

    assert writer_module._valid_opus(opus_path, _SAMPLE_RATE, len(_tone()))


def test_receipt_directory_leaf_symlink_is_rejected(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    writer = _writer(output_dir)
    row_root = output_dir / ".nemo_curator" / "nemo_speech_rows"
    row_root.rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    row_root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        writer.process(_task())

    assert not list(outside.rglob("*.json"))
