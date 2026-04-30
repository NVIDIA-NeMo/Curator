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

"""Stage-level tests for the audio ALM pretrain pipeline.

Exercises every stage end-to-end at the ``process()`` level (no Ray /
soundfile / torch needed thanks to the dry-run extractor mode).  Also
covers the ``prepare_*`` / ``finalize_*`` helpers across simulated
multi-replica shards.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nemo_curator.stages.audio.alm.pretrain import (
    OverlapFilterStage,
    PretrainMetricsAggregatorStage,
    ReadLongFormManifestStage,
    SnippetCutPlannerStage,
    SnippetExtractionStage,
    SnippetManifestWriterStage,
    finalize_audio_pretrain_outputs,
    prepare_audio_pretrain_outputs,
)
from nemo_curator.stages.audio.alm.pretrain.stages import (
    _PLAN_DATA_KEY,
    _PRETRAIN_META_KEY,
    _make_shard_path,
)
from nemo_curator.tasks import AudioTask, _EmptyTask

# ----------------------------------------------------------------------
# Test fixtures
# ----------------------------------------------------------------------


def _make_audio_task(data: dict | None = None, *, task_id: str = "t1") -> AudioTask:
    return AudioTask(task_id=task_id, dataset_name="ds", data=data or {})


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
        stage.__post_init__()
        out = stage.process(_EmptyTask(task_id="empty", dataset_name="empty", data=None))
        assert len(out) == 1
        assert out[0].data["id"] == "A"

    def test_resolves_audio_path_against_audio_dir(self, tmp_path: Path, manifest_path: Path) -> None:
        stage = ReadLongFormManifestStage(input_manifest=str(manifest_path), audio_dir="/data")
        stage.__post_init__()
        out = stage.process(_EmptyTask(task_id="empty", dataset_name="empty", data=None))
        assert out[0].data["audio_filepath"] == "/data/a.wav"

    def test_missing_manifest_raises(self, tmp_path: Path) -> None:
        stage = ReadLongFormManifestStage(input_manifest=str(tmp_path / "nope.jsonl"), audio_dir=str(tmp_path))
        stage.__post_init__()
        with pytest.raises(FileNotFoundError):
            stage.process(_EmptyTask(task_id="empty", dataset_name="empty", data=None))

    def test_post_init_validates_required_args(self) -> None:
        with pytest.raises(ValueError, match="input_manifest"):
            ReadLongFormManifestStage(input_manifest="", audio_dir="/data").__post_init__()
        with pytest.raises(ValueError, match="audio_dir"):
            ReadLongFormManifestStage(input_manifest="a", audio_dir="").__post_init__()


# ----------------------------------------------------------------------
# OverlapFilterStage
# ----------------------------------------------------------------------


class TestOverlapFilterStage:
    def test_drops_empty_then_overlap(self) -> None:
        segs = [
            _ts(0, 3, "a"),  # ok
            {"start": 5, "end": 6, "text": "", "text_ITN": "", "words": []},  # empty
            _ts(10, 15, "b"),  # overlap pair
            _ts(13, 18, "c"),  # overlap pair
            _ts(20, 23, "d"),  # ok
        ]
        task = _make_audio_task({"id": "X", "segments": segs})
        stage = OverlapFilterStage()
        out = stage.process(task)
        # Kept: only "a" and "d"
        assert [s["text"] for s in out.data["segments"]] == ["a", "d"]
        meta = out._metadata[_PRETRAIN_META_KEY]
        assert meta["original_seg_count"] == 5
        assert meta["dropped_empty"] == 1
        assert meta["dropped_overlap"] == 2
        assert meta["kept_after_filter_count"] == 2

    def test_no_segments_metadata_initialized(self) -> None:
        task = _make_audio_task({"id": "X", "segments": []})
        stage = OverlapFilterStage()
        out = stage.process(task)
        meta = out._metadata[_PRETRAIN_META_KEY]
        assert meta["original_seg_count"] == 0
        assert meta["dropped_empty"] == 0
        assert meta["dropped_overlap"] == 0


# ----------------------------------------------------------------------
# SnippetCutPlannerStage
# ----------------------------------------------------------------------


class TestSnippetCutPlannerStage:
    def test_writes_plan_and_drop_counts(self) -> None:
        segs = [_ts(0, 5, "a"), _ts(5, 10, "b")]
        task = _make_audio_task({"id": "X", "segments": segs})
        stage = SnippetCutPlannerStage(max_duration_sec=20.0, min_duration_sec=0.5, max_segment_gap_in_snippet=30.0)
        out = stage.process(task)
        plan = out.data[_PLAN_DATA_KEY]
        assert len(plan) == 1
        assert (plan[0]["start"], plan[0]["end"]) == (0, 10)
        meta = out._metadata[_PRETRAIN_META_KEY]
        assert meta["planned_snippets"] == 1
        assert meta["dropped_too_long"] == 0
        assert meta["dropped_too_short"] == 0
        assert meta["dropped_no_text"] == 0

    def test_invalid_args_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_duration"):
            SnippetCutPlannerStage(max_duration_sec=-1.0).__post_init__()
        with pytest.raises(ValueError, match="min_duration"):
            SnippetCutPlannerStage(min_duration_sec=-1.0).__post_init__()
        with pytest.raises(ValueError, match="min_duration_sec must be <="):
            SnippetCutPlannerStage(max_duration_sec=5.0, min_duration_sec=10.0).__post_init__()
        with pytest.raises(ValueError, match="max_segment_gap_in_snippet"):
            SnippetCutPlannerStage(max_segment_gap_in_snippet=-0.1).__post_init__()


# ----------------------------------------------------------------------
# SnippetExtractionStage (dry-run)
# ----------------------------------------------------------------------


class TestSnippetExtractionStageDryRun:
    def test_emits_one_task_per_planned_snippet_no_audio_io(self, tmp_path: Path) -> None:
        snippet1 = {"start": 0.0, "end": 5.0, "segments": [_ts(0.0, 5.0, "hi", "Hi.")]}
        snippet2 = {"start": 5.0, "end": 12.5, "segments": [_ts(6.0, 12.0, "world", "World.")]}
        task = _make_audio_task(
            {
                "id": "X",
                "audio_filepath": "/missing/source.wav",
                _PLAN_DATA_KEY: [snippet1, snippet2],
                "text": "WHOLE",
                "audio_sample_rate": 22050,
                "audio_num_channels": 2,
                "audio_size": 999,
                "actual_duration": 100.0,
                "proposed_duration": 100.0,
                "alignment": "STALE",
            }
        )
        stage = SnippetExtractionStage(output_dir=str(tmp_path / "snips"), dry_run=True)
        stage.__post_init__()
        out = stage.process(task)
        assert len(out) == 2

        s0 = out[0].data
        # Snippet ID + path pattern
        assert s0["snippet_id"] == "X_0.000_5.000"
        assert s0["audio_filepath"].endswith("X_0.000_5.000.flac")
        assert s0["duration"] == pytest.approx(5.0)
        # Field cleanup
        assert "alignment" not in s0
        assert "audio_size" not in s0
        # Audio-property fields updated
        assert s0["audio_sample_rate"] == 16000
        assert s0["audio_num_channels"] == 1
        assert s0["actual_duration"] == pytest.approx(5.0)
        assert s0["proposed_duration"] == pytest.approx(5.0)
        # Top-level text recomputed (uses text_ITN -> "Hi.")
        assert s0["text"] == "Hi."
        # Segments relativized
        assert s0["segments"][0]["start"] == pytest.approx(0.0)

    def test_zero_planned_emits_stub(self, tmp_path: Path) -> None:
        task = _make_audio_task({"id": "X", _PLAN_DATA_KEY: []})
        stage = SnippetExtractionStage(output_dir=str(tmp_path / "snips"), dry_run=True)
        stage.__post_init__()
        out = stage.process(task)
        assert len(out) == 1
        assert out[0].data["snippet_id"] is None

    def test_invalid_output_format_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="output_format"):
            SnippetExtractionStage(output_dir=str(tmp_path), output_format="m4a").__post_init__()
        with pytest.raises(ValueError, match="target_sample_rate"):
            SnippetExtractionStage(output_dir=str(tmp_path), target_sample_rate=0).__post_init__()


# ----------------------------------------------------------------------
# SnippetManifestWriterStage
# ----------------------------------------------------------------------


class TestSnippetManifestWriterStage:
    def test_writes_non_stub_to_shard(self, tmp_path: Path) -> None:
        out_path = str(tmp_path / "out.jsonl")
        stage = SnippetManifestWriterStage(output_path=out_path)
        stage.__post_init__()
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
        stage.__post_init__()
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
        stage.__post_init__()
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


# ----------------------------------------------------------------------
# prepare + finalize end-to-end across multiple shards
# ----------------------------------------------------------------------


class TestPrepareAndFinalize:
    def test_finalize_merges_manifest_and_metrics(self, tmp_path: Path) -> None:
        manifest = str(tmp_path / "snippets.jsonl")
        metrics = str(tmp_path / "metrics.json")

        # Two writer shards
        s1 = _make_shard_path(manifest, "jsonl")
        s2 = _make_shard_path(manifest, "jsonl")
        with open(s1, "w") as f:
            f.write(json.dumps({"snippet_id": "a", "duration": 5.0}) + "\n")
            f.write(json.dumps({"snippet_id": "b", "duration": 12.0}) + "\n")
        with open(s2, "w") as f:
            f.write(json.dumps({"snippet_id": "c", "duration": 31.0}) + "\n")

        # Two metrics shards covering the same id (one record per task,
        # JSONL — matching what the aggregator writes in process()).
        record_template = {
            "id": "vid1",
            "in_segments": 10,
            "in_duration_sec": 100.0,
            "dropped": {"empty": 1, "overlap": 0, "too_long": 0, "too_short": 0, "no_text": 0},
            "is_stub": False,
            "out_segments": 3,
        }
        m1 = _make_shard_path(metrics, "jsonl")
        m2 = _make_shard_path(metrics, "jsonl")
        with open(m1, "w") as f:
            f.writelines(json.dumps({**record_template, "out_duration_sec": dur}) + "\n" for dur in (5.0, 12.0))
        with open(m2, "w") as f:
            f.write(json.dumps({**record_template, "out_duration_sec": 31.0}) + "\n")

        finalize_audio_pretrain_outputs(manifest, metrics)

        # Manifest concatenated
        lines = Path(manifest).read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3
        # Metrics combined
        summary = json.loads(Path(metrics).read_text(encoding="utf-8"))
        assert summary["num_input_audios"] == 1
        assert summary["num_output_snippets"] == 3
        assert summary["output_total_duration_sec"] == pytest.approx(48.0)
        assert summary["snippet_duration_histogram_30s"] == {"0-30": 2, "30-60": 1}
        assert list(tmp_path.glob("*.shard-*")) == []

    def test_prepare_removes_only_matching_shards(self, tmp_path: Path) -> None:
        manifest = str(tmp_path / "snippets.jsonl")
        metrics = str(tmp_path / "metrics.json")
        # Plant a stale shard for both (both are JSONL — metrics shards are
        # JSONL too, since the aggregator writes per-task records).
        for path in (manifest, metrics):
            Path(_make_shard_path(path, "jsonl")).touch()
        unrelated = tmp_path / "other.txt"
        unrelated.write_text("keep me", encoding="utf-8")

        prepare_audio_pretrain_outputs(manifest, metrics)

        assert list(tmp_path.glob("*.shard-*")) == []
        assert unrelated.exists()
