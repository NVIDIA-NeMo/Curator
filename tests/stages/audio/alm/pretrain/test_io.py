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

"""Stage-level tests for ``nemo_curator.stages.audio.alm.pretrain.io``.

Covers the input-manifest reader and the two per-replica writer stages
(snippet manifest, per-task metrics).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nemo_curator.stages.audio.alm.pretrain import (
    PretrainMetricsAggregatorStage,
    ReadLongFormManifestStage,
    SnippetManifestWriterStage,
)
from nemo_curator.stages.audio.alm.pretrain.utils import _PRETRAIN_META_KEY
from nemo_curator.tasks import AudioTask, _EmptyTask


def _make_audio_task(data: dict | None = None) -> AudioTask:
    return AudioTask(dataset_name="ds", data=data or {})


def _ts(start: float, end: float, text: str = "x", text_itn: str | None = None) -> dict:
    return {
        "speaker": "A",
        "start": start,
        "end": end,
        "text": text,
        "text_ITN": text_itn if text_itn is not None else text,
        "words": [],
    }


@pytest.fixture
def manifest_path(tmp_path: Path) -> Path:
    """Write a small input manifest with two rows (one valid, one missing path)."""
    p = tmp_path / "in.jsonl"
    rows = [
        {"id": "A", "audio_filepath": "./a.wav", "segments": [_ts(0, 5, "hi")]},
        {"id": "B", "segments": []},  # missing audio_filepath -- should be warned & skipped
    ]
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


# ----------------------------------------------------------------------
# ReadLongFormManifestStage
# ----------------------------------------------------------------------


class TestReadLongFormManifestStage:
    def test_emits_one_task_per_valid_row(self, tmp_path: Path, manifest_path: Path) -> None:
        stage = ReadLongFormManifestStage(input_manifest=str(manifest_path), audio_dir=str(tmp_path))
        out = stage.process(_EmptyTask(dataset_name="empty", data=None))
        assert len(out) == 1
        assert out[0].data["id"] == "A"

    def test_resolves_audio_path_against_audio_dir(self, tmp_path: Path, manifest_path: Path) -> None:
        stage = ReadLongFormManifestStage(input_manifest=str(manifest_path), audio_dir="/data")
        out = stage.process(_EmptyTask(dataset_name="empty", data=None))
        assert out[0].data["audio_filepath"] == "/data/a.wav"

    def test_missing_manifest_raises(self, tmp_path: Path) -> None:
        stage = ReadLongFormManifestStage(input_manifest=str(tmp_path / "nope.jsonl"), audio_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            stage.process(_EmptyTask(dataset_name="empty", data=None))

    def test_skips_rows_missing_id(self, tmp_path: Path) -> None:
        # `id` is required: the metrics aggregator keys per-source records on
        # it and snippet ids embed it.  Missing/empty ids must be filtered
        # at read time rather than silently propagated as `line_N` fallbacks.
        p = tmp_path / "in.jsonl"
        rows = [
            {"audio_filepath": "./a.wav", "segments": []},  # no id
            {"id": "", "audio_filepath": "./b.wav", "segments": []},  # empty id
            {"id": "  ", "audio_filepath": "./c.wav", "segments": []},  # whitespace id
            {"id": "OK", "audio_filepath": "./d.wav", "segments": []},
        ]
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        stage = ReadLongFormManifestStage(input_manifest=str(p), audio_dir="/data")
        out = stage.process(_EmptyTask(dataset_name="empty", data=None))
        assert [t.data["id"] for t in out] == ["OK"]

    def test_skips_duplicate_ids(self, tmp_path: Path) -> None:
        # Duplicate ids silently collapse per_original metrics and can
        # collide tar member names (snippet_id embeds the source id).
        # First occurrence wins; later duplicates are skipped with a warning.
        p = tmp_path / "in.jsonl"
        rows = [
            {"id": "A", "audio_filepath": "./a.wav", "segments": []},
            {"id": "A", "audio_filepath": "./a2.wav", "segments": []},
            {"id": "B", "audio_filepath": "./b.wav", "segments": []},
        ]
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        stage = ReadLongFormManifestStage(input_manifest=str(p), audio_dir="/data")
        out = stage.process(_EmptyTask(dataset_name="empty", data=None))
        assert [t.data["id"] for t in out] == ["A", "B"]
        # The kept "A" row is the first one, not the duplicate.
        assert out[0].data["audio_filepath"] == "/data/a.wav"

    def test_basename_mode_rejects_duplicate_basenames(self, tmp_path: Path) -> None:
        # In basename mode, two different source rows whose audio_filepath
        # values share a basename would silently route to the same audio.
        # Surface that as a hard error rather than corrupt outputs.
        p = tmp_path / "in.jsonl"
        rows = [
            {"id": "A", "audio_filepath": "./shard1/foo.wav", "segments": []},
            {"id": "B", "audio_filepath": "./shard2/foo.wav", "segments": []},
        ]
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        stage = ReadLongFormManifestStage(input_manifest=str(p), audio_dir="/data")
        with pytest.raises(ValueError, match="duplicate audio basename"):
            stage.process(_EmptyTask(dataset_name="empty", data=None))

    def test_relative_mode_preserves_subdirectories(self, tmp_path: Path) -> None:
        # 'relative' joins audio_dir / audio_filepath verbatim and so does
        # NOT trip the duplicate-basename guard.
        p = tmp_path / "in.jsonl"
        rows = [
            {"id": "A", "audio_filepath": "shard1/foo.wav", "segments": []},
            {"id": "B", "audio_filepath": "shard2/foo.wav", "segments": []},
        ]
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        stage = ReadLongFormManifestStage(input_manifest=str(p), audio_dir="/data", audio_path_resolution="relative")
        out = stage.process(_EmptyTask(dataset_name="empty", data=None))
        assert [t.data["audio_filepath"] for t in out] == [
            "/data/shard1/foo.wav",
            "/data/shard2/foo.wav",
        ]

    def test_as_is_mode_returns_value_unchanged(self, tmp_path: Path) -> None:
        p = tmp_path / "in.jsonl"
        rows = [
            {"id": "A", "audio_filepath": "/absolute/a.wav", "segments": []},
            {"id": "B", "audio_filepath": "relative/b.wav", "segments": []},
        ]
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        stage = ReadLongFormManifestStage(input_manifest=str(p), audio_dir="/ignored", audio_path_resolution="as_is")
        out = stage.process(_EmptyTask(dataset_name="empty", data=None))
        assert [t.data["audio_filepath"] for t in out] == [
            "/absolute/a.wav",
            "relative/b.wav",
        ]

    def test_unknown_resolution_mode_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "in.jsonl"
        with p.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"id": "A", "audio_filepath": "./a.wav", "segments": []}) + "\n")
        stage = ReadLongFormManifestStage(input_manifest=str(p), audio_dir="/data", audio_path_resolution="not_a_mode")
        with pytest.raises(ValueError, match="unknown audio_path_resolution"):
            stage.process(_EmptyTask(dataset_name="empty", data=None))


# ----------------------------------------------------------------------
# SnippetManifestWriterStage
# ----------------------------------------------------------------------


class TestSnippetManifestWriterStage:
    def test_writes_non_stub_to_shard(self, tmp_path: Path) -> None:
        out_path = str(tmp_path / "out.jsonl")
        stage = SnippetManifestWriterStage(output_path=out_path)
        stage.setup_on_node()
        stage.setup()
        t = _make_audio_task({"id": "X", "snippet_id": "X_0.000_3.000", "duration": 3.0})
        stage.process(t)
        # Shard exists, final file does not yet
        shard = stage._shard_path
        assert shard is not None
        assert Path(shard).exists()
        assert not Path(out_path).exists()
        text = Path(shard).read_text(encoding="utf-8").strip()
        row = json.loads(text)
        assert row["snippet_id"] == "X_0.000_3.000"

    def test_skips_stub_tasks(self, tmp_path: Path) -> None:
        stage = SnippetManifestWriterStage(output_path=str(tmp_path / "out.jsonl"))
        stage.setup_on_node()
        stage.setup()
        stub = _make_audio_task({"id": "X", "snippet_id": None})
        stage.process(stub)
        # No shard file produced (we never opened it for write)
        shard = stage._shard_path
        assert shard is not None
        assert not Path(shard).exists()


# ----------------------------------------------------------------------
# PretrainMetricsAggregatorStage
# ----------------------------------------------------------------------


class TestPretrainMetricsAggregatorStage:
    def test_writes_one_jsonl_record_per_task(self, tmp_path: Path) -> None:
        out_path = str(tmp_path / "metrics.json")
        stage = PretrainMetricsAggregatorStage(output_path=out_path)
        stage.setup()

        task1 = _make_audio_task({"id": "A", "snippet_id": "A_0_5", "duration": 5.0, "segments": [_ts(0, 5, "x")]})
        task1._metadata[_PRETRAIN_META_KEY] = {
            "original_seg_count": 10,
            "original_seg_duration": 100.0,
            "dropped_empty": 1,
            "dropped_overlap": 2,
            "dropped_too_long": 0,
            "dropped_too_short": 0,
            "dropped_no_text": 0,
            "filtered_repetition_texts": ["repeat one", "repeat two"],
        }
        task2 = _make_audio_task({"id": "A", "snippet_id": "A_5_12", "duration": 7.0, "segments": [_ts(0, 7, "y")]})
        task2._metadata[_PRETRAIN_META_KEY] = task1._metadata[_PRETRAIN_META_KEY]
        stub = _make_audio_task({"id": "B", "snippet_id": None, "duration": 0.0, "segments": []})
        stub._metadata[_PRETRAIN_META_KEY] = {
            "original_seg_count": 3,
            "original_seg_duration": 30.0,
            "dropped_empty": 0,
            "dropped_overlap": 0,
            "dropped_too_long": 3,
            "dropped_too_short": 0,
            "dropped_no_text": 0,
        }

        stage.process(task1)
        stage.process(task2)
        stage.process(stub)

        shard = stage._shard_path
        assert shard is not None
        # Aggregator writes JSONL incrementally in process() (no teardown reliance);
        # expect three lines: two non-stub snippets + one stub.
        lines = Path(shard).read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3
        records = [json.loads(line) for line in lines]
        ids = [r["id"] for r in records]
        assert ids == ["A", "A", "B"]
        # Non-stub records carry per-snippet output info; stubs are flagged.
        assert records[0]["is_stub"] is False
        assert records[0]["out_segments"] == 1
        assert records[0]["out_duration_sec"] == pytest.approx(5.0)
        assert records[1]["out_duration_sec"] == pytest.approx(7.0)
        assert records[2]["is_stub"] is True
        assert records[2]["out_segments"] == 0
        # filtered_texts is emitted exactly once per id (first occurrence).
        assert records[0]["filtered_texts"] == ["repeat one", "repeat two"]
        assert "filtered_texts" not in records[1]
        assert records[2]["filtered_texts"] == []
