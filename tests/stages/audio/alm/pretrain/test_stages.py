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
    SnippetRepetitionFilterStage,
    finalize_audio_pretrain_outputs,
    prepare_audio_pretrain_outputs,
)
from nemo_curator.stages.audio.alm.pretrain.utils import (
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
        stage = SnippetExtractionStage(
            output_dir=str(tmp_path / "snips"),
            output_audio_tar_path=str(tmp_path / "snips.tar"),
            dry_run=True,
        )
        stage.__post_init__()
        out = stage.process(task)
        assert len(out) == 2

        s0 = out[0].data
        # Snippet ID + path pattern (WebDataset-friendly: dashes between
        # fields, underscores instead of decimal points so the resulting
        # filename has only one `.` before the extension).
        assert s0["snippet_id"] == "X-0_000-5_000"
        # In tar mode `audio_filepath` is the tar-internal basename,
        # not a filesystem path -- no slashes.
        assert s0["audio_filepath"] == "X-0_000-5_000.flac"
        assert s0["duration"] == pytest.approx(5.0)
        # Field cleanup
        assert "alignment" not in s0
        assert "audio_size" not in s0
        # Audio-property fields updated
        assert s0["audio_sample_rate"] == 16000
        assert s0["audio_num_channels"] == 1
        assert s0["actual_duration"] == pytest.approx(5.0)
        assert s0["proposed_duration"] == pytest.approx(5.0)
        # Top-level text recomputed from each segment's `text` field
        # (text_ITN is unreliable in real data and is no longer consulted).
        assert s0["text"] == "hi"
        # Segments relativized
        assert s0["segments"][0]["start"] == pytest.approx(0.0)

    def test_zero_planned_emits_stub(self, tmp_path: Path) -> None:
        task = _make_audio_task({"id": "X", _PLAN_DATA_KEY: []})
        stage = SnippetExtractionStage(
            output_dir=str(tmp_path / "snips"),
            output_audio_tar_path=str(tmp_path / "snips.tar"),
            dry_run=True,
        )
        stage.__post_init__()
        out = stage.process(task)
        assert len(out) == 1
        assert out[0].data["snippet_id"] is None

    def test_invalid_output_format_rejected(self, tmp_path: Path) -> None:
        tar_path = str(tmp_path / "snips.tar")
        with pytest.raises(ValueError, match="output_format"):
            SnippetExtractionStage(
                output_dir=str(tmp_path), output_audio_tar_path=tar_path, output_format="m4a"
            ).__post_init__()
        with pytest.raises(ValueError, match="target_sample_rate"):
            SnippetExtractionStage(
                output_dir=str(tmp_path), output_audio_tar_path=tar_path, target_sample_rate=0
            ).__post_init__()
        with pytest.raises(ValueError, match="output_audio_tar_path"):
            SnippetExtractionStage(output_dir=str(tmp_path)).__post_init__()


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


# ----------------------------------------------------------------------
# prepare + finalize end-to-end across multiple shards
# ----------------------------------------------------------------------


class TestPrepareAndFinalize:
    def test_finalize_merges_manifest_and_metrics(self, tmp_path: Path) -> None:
        manifest = str(tmp_path / "snippets.jsonl")
        metrics = str(tmp_path / "metrics.json")
        tar_path = str(tmp_path / "snippets.tar")

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

        finalize_audio_pretrain_outputs(manifest, metrics, tar_path)

        # Manifest concatenated
        lines = Path(manifest).read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3
        # Metrics combined
        summary = json.loads(Path(metrics).read_text(encoding="utf-8"))
        assert summary["num_input_audios"] == 1
        assert summary["num_output_snippets"] == 3
        assert summary["output_total_duration_sec"] == pytest.approx(48.0)
        assert summary["snippet_duration_histogram_30s"] == {"0-30": 2, "30-60": 1}
        # The new examples key is always present, empty when no records carry it.
        assert summary["dropped_repetition_examples"] == []
        assert list(tmp_path.glob("*.shard-*")) == []

    def test_finalize_caps_filtered_examples_globally(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import nemo_curator.stages.audio.alm.pretrain.finalize as m

        monkeypatch.setattr(m, "_MAX_FILTERED_TEXT_EXAMPLES", 5)
        manifest = str(tmp_path / "snippets.jsonl")
        metrics = str(tmp_path / "metrics.json")
        tar_path = str(tmp_path / "snippets.tar")

        record_template = {
            "in_segments": 10,
            "in_duration_sec": 100.0,
            "dropped": {"empty": 0, "overlap": 0, "too_long": 0, "too_short": 0, "no_text": 0, "repetition": 4},
            "is_stub": True,
            "out_segments": 0,
            "out_duration_sec": 0.0,
        }
        shard = _make_shard_path(metrics, "jsonl")
        with open(shard, "w") as f:
            # Three sources, each contributing 4 filtered texts → 12 total,
            # capped to 5 globally.
            for src_idx in range(3):
                rec = {
                    **record_template,
                    "id": f"vid{src_idx}",
                    "filtered_texts": [f"src{src_idx}-text{i}" for i in range(4)],
                }
                f.write(json.dumps(rec) + "\n")

        finalize_audio_pretrain_outputs(manifest, metrics, tar_path)

        summary = json.loads(Path(metrics).read_text(encoding="utf-8"))
        examples = summary["dropped_repetition_examples"]
        assert len(examples) == 5
        # First-encountered-wins: vid0's four texts plus the first of vid1's.
        assert examples == [
            "src0-text0",
            "src0-text1",
            "src0-text2",
            "src0-text3",
            "src1-text0",
        ]

    def test_finalize_merges_tar_shards_sorted(self, tmp_path: Path) -> None:
        import io
        import tarfile

        manifest = str(tmp_path / "snippets.jsonl")
        metrics = str(tmp_path / "metrics.json")
        tar_path = str(tmp_path / "snippets.tar")

        # Two tar shards, in unsorted order across the two files
        s1 = _make_shard_path(tar_path, "tar")
        s2 = _make_shard_path(tar_path, "tar")
        with tarfile.open(s1, "w") as t:
            for name, body in (("c.flac", b"CCC"), ("a.flac", b"AAA")):
                ti = tarfile.TarInfo(name=name)
                ti.size = len(body)
                t.addfile(ti, io.BytesIO(body))
        with tarfile.open(s2, "w") as t:
            for name, body in (("d.flac", b"DDDD"), ("b.flac", b"BB")):
                ti = tarfile.TarInfo(name=name)
                ti.size = len(body)
                t.addfile(ti, io.BytesIO(body))

        finalize_audio_pretrain_outputs(manifest, metrics, tar_path)

        # Final tar exists, contains all 4 members, sorted
        with tarfile.open(tar_path, "r") as t:
            names = t.getnames()
            assert names == ["a.flac", "b.flac", "c.flac", "d.flac"]
            payloads = {n: t.extractfile(n).read() for n in names}
        assert payloads == {"a.flac": b"AAA", "b.flac": b"BB", "c.flac": b"CCC", "d.flac": b"DDDD"}
        # Tar shards cleaned up
        assert sorted(tmp_path.glob("snippets.tar.shard-*.tar")) == []

    def test_finalize_drops_manifest_rows_missing_from_tar(self, tmp_path: Path) -> None:
        """Manifest reconciliation: rows whose tar member is missing get dropped
        and surfaced as `dropped.missing_audio` in the merged metrics."""
        import io
        import tarfile

        import numpy as np
        import soundfile as sf

        manifest = str(tmp_path / "snippets.jsonl")
        metrics = str(tmp_path / "metrics.json")
        tar_path = str(tmp_path / "snippets.tar")

        # Manifest writer shard with 3 entries; tar shard has only 2 of those
        # members (the third member's data was "truncated" -- simulated by
        # never adding it to the tar at all).
        ms = _make_shard_path(manifest, "jsonl")
        with open(ms, "w") as f:
            for sid in ("X-0_000-1_000", "X-1_000-2_000", "X-2_000-3_000"):
                f.write(
                    json.dumps(
                        {"id": "X", "snippet_id": sid, "audio_filepath": f"{sid}.flac", "duration": 1.0}
                    )
                    + "\n"
                )
        # Metrics shard so finalize writes a merged metrics.json that
        # _patch_metrics_with_reconcile_drops can update.
        ms_metrics = _make_shard_path(metrics, "jsonl")
        with open(ms_metrics, "w") as f:
            for _ in range(3):
                f.write(
                    json.dumps(
                        {
                            "id": "X",
                            "in_segments": 1,
                            "in_duration_sec": 3.0,
                            "dropped": {},
                            "is_stub": False,
                            "out_segments": 1,
                            "out_duration_sec": 1.0,
                        }
                    )
                    + "\n"
                )
        # Synthesize a real, decodable FLAC body so the kept members survive
        # the header/duration check; only the missing member should be dropped.
        flac_buf = io.BytesIO()
        sf.write(flac_buf, np.zeros(160, dtype=np.float32), 16000, format="FLAC")
        flac_bytes = flac_buf.getvalue()
        ts = _make_shard_path(tar_path, "tar")
        with tarfile.open(ts, "w") as t:
            for sid in ("X-0_000-1_000", "X-2_000-3_000"):
                ti = tarfile.TarInfo(name=f"{sid}.flac")
                ti.size = len(flac_bytes)
                t.addfile(ti, io.BytesIO(flac_bytes))

        finalize_audio_pretrain_outputs(manifest, metrics, tar_path)

        # Manifest reduced to 2 rows matching the tar's members
        kept_paths = [json.loads(line)["audio_filepath"] for line in Path(manifest).read_text().splitlines() if line]
        assert sorted(kept_paths) == ["X-0_000-1_000.flac", "X-2_000-3_000.flac"]
        # Metrics summary records the reconcile drop.
        summary = json.loads(Path(metrics).read_text(encoding="utf-8"))
        assert summary["dropped"]["missing_audio"] == 1
        # corrupted_audio is only added when non-zero; absent here.
        assert "corrupted_audio" not in summary["dropped"]

    def test_finalize_drops_manifest_rows_with_unreadable_audio(self, tmp_path: Path) -> None:
        """Manifest reconciliation: rows whose tar member fails the audio
        header/duration check get dropped (e.g. truncated payload from a
        worker killed mid-write) and counted under
        `dropped.corrupted_audio` in the merged metrics."""
        import io
        import tarfile

        import numpy as np
        import soundfile as sf

        manifest = str(tmp_path / "snippets.jsonl")
        metrics = str(tmp_path / "metrics.json")
        tar_path = str(tmp_path / "snippets.tar")

        # Three entries: two with valid FLAC payloads, one with bogus bytes
        # (header unreadable -> sf.info raises -> row dropped).
        ms = _make_shard_path(manifest, "jsonl")
        with open(ms, "w") as f:
            for sid in ("X-0_000-1_000", "X-1_000-2_000", "X-2_000-3_000"):
                f.write(
                    json.dumps(
                        {"id": "X", "snippet_id": sid, "audio_filepath": f"{sid}.flac", "duration": 1.0}
                    )
                    + "\n"
                )
        ms_metrics = _make_shard_path(metrics, "jsonl")
        with open(ms_metrics, "w") as f:
            for _ in range(3):
                f.write(
                    json.dumps(
                        {
                            "id": "X",
                            "in_segments": 1,
                            "in_duration_sec": 3.0,
                            "dropped": {},
                            "is_stub": False,
                            "out_segments": 1,
                            "out_duration_sec": 1.0,
                        }
                    )
                    + "\n"
                )

        flac_buf = io.BytesIO()
        sf.write(flac_buf, np.zeros(160, dtype=np.float32), 16000, format="FLAC")
        flac_bytes = flac_buf.getvalue()

        ts = _make_shard_path(tar_path, "tar")
        with tarfile.open(ts, "w") as t:
            # Two readable members.
            for sid in ("X-0_000-1_000", "X-2_000-3_000"):
                ti = tarfile.TarInfo(name=f"{sid}.flac")
                ti.size = len(flac_bytes)
                t.addfile(ti, io.BytesIO(flac_bytes))
            # One member that's in the tar but whose payload won't decode.
            bogus = b"NOT_A_FLAC_FILE"
            ti = tarfile.TarInfo(name="X-1_000-2_000.flac")
            ti.size = len(bogus)
            t.addfile(ti, io.BytesIO(bogus))

        finalize_audio_pretrain_outputs(manifest, metrics, tar_path)

        kept_paths = [
            json.loads(line)["audio_filepath"]
            for line in Path(manifest).read_text().splitlines()
            if line
        ]
        assert sorted(kept_paths) == ["X-0_000-1_000.flac", "X-2_000-3_000.flac"]
        summary = json.loads(Path(metrics).read_text(encoding="utf-8"))
        assert summary["dropped"]["corrupted_audio"] == 1
        # missing_audio is only added when non-zero; absent here.
        assert "missing_audio" not in summary["dropped"]

    def test_prepare_removes_only_matching_shards(self, tmp_path: Path) -> None:
        manifest = str(tmp_path / "snippets.jsonl")
        metrics = str(tmp_path / "metrics.json")
        tar_path = str(tmp_path / "snippets.tar")
        # Plant a stale shard for each output type. Manifest and metrics
        # shards are JSONL; tar shards are TAR.
        for path in (manifest, metrics):
            Path(_make_shard_path(path, "jsonl")).touch()
        Path(_make_shard_path(tar_path, "tar")).touch()
        unrelated = tmp_path / "other.txt"
        unrelated.write_text("keep me", encoding="utf-8")

        prepare_audio_pretrain_outputs(manifest, metrics, tar_path)

        assert list(tmp_path.glob("*.shard-*")) == []
        assert unrelated.exists()


# ----------------------------------------------------------------------
# SnippetRepetitionFilterStage
# ----------------------------------------------------------------------


def _build_tiny_word_tokenizer(tmp_dir: Path, words: list[str]) -> Path:
    """Save a WordLevel HF fast tokenizer covering ``words`` to ``tmp_dir``."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"[UNK]": 0}
    for i, w in enumerate(words, start=1):
        vocab[w] = i
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok_dir = tmp_dir / "tok"
    tok_dir.mkdir()
    tok.save(str(tok_dir / "tokenizer.json"))
    (tok_dir / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", "model_max_length": 4096}),
        encoding="utf-8",
    )
    return tok_dir


class TestSnippetRepetitionFilterStage:
    @pytest.fixture
    def tokenizer_dir(self, tmp_path: Path) -> Path:
        pytest.importorskip("transformers")
        pytest.importorskip("tokenizers")
        return _build_tiny_word_tokenizer(
            tmp_path,
            ["thank", "you", "for", "watching", "please", "subscribe", "the", "quick", "brown", "fox", "hi"],
        )

    def _make_task_with_plan(self, plan: list[dict]) -> AudioTask:
        task = AudioTask(task_id="t1", dataset_name="ds", data={_PLAN_DATA_KEY: plan})
        task._metadata = {}
        return task

    def test_drops_repetitive_snippet(self, tokenizer_dir: Path) -> None:
        stage = SnippetRepetitionFilterStage(tokenizer_path=str(tokenizer_dir))
        stage.__post_init__()
        stage.setup()

        repeat = "thank you for watching " * 10
        plan = [{"start": 0.0, "end": 30.0, "segments": [_ts(0.0, 30.0, repeat)]}]
        out = stage.process(self._make_task_with_plan(plan))
        assert out.data[_PLAN_DATA_KEY] == []
        meta = out._metadata[_PRETRAIN_META_KEY]
        assert meta["dropped_repetition"] == 1
        assert meta["kept_after_repetition_filter"] == 0
        # The dropped text is captured (un-colorized) for the metrics summary.
        assert meta["filtered_repetition_texts"] == [repeat.strip()]

    def test_keeps_non_repetitive_snippet(self, tokenizer_dir: Path) -> None:
        stage = SnippetRepetitionFilterStage(tokenizer_path=str(tokenizer_dir))
        stage.__post_init__()
        stage.setup()

        plan = [
            {"start": 0.0, "end": 5.0, "segments": [_ts(0.0, 5.0, "the quick brown fox")]},
        ]
        out = stage.process(self._make_task_with_plan(plan))
        assert len(out.data[_PLAN_DATA_KEY]) == 1
        meta = out._metadata[_PRETRAIN_META_KEY]
        assert meta["dropped_repetition"] == 0
        assert meta["kept_after_repetition_filter"] == 1

    def test_keeps_short_snippet_without_enough_tokens_for_ngram(self, tokenizer_dir: Path) -> None:
        # ngram_n=4 but text tokenizes to 1 token -> no n-grams to evaluate, kept
        stage = SnippetRepetitionFilterStage(tokenizer_path=str(tokenizer_dir), ngram_n=4)
        stage.__post_init__()
        stage.setup()

        plan = [{"start": 0.0, "end": 1.0, "segments": [_ts(0.0, 1.0, "hi")]}]
        out = stage.process(self._make_task_with_plan(plan))
        assert len(out.data[_PLAN_DATA_KEY]) == 1
        assert out._metadata[_PRETRAIN_META_KEY]["dropped_repetition"] == 0

    def test_filters_only_repetitive_snippets_in_a_mixed_plan(self, tokenizer_dir: Path) -> None:
        stage = SnippetRepetitionFilterStage(tokenizer_path=str(tokenizer_dir))
        stage.__post_init__()
        stage.setup()

        plan = [
            {"start": 0.0, "end": 5.0, "segments": [_ts(0.0, 5.0, "the quick brown fox")]},
            {"start": 5.0, "end": 35.0, "segments": [_ts(5.0, 35.0, "thank you for watching " * 10)]},
            {"start": 35.0, "end": 36.0, "segments": [_ts(35.0, 36.0, "hi")]},
        ]
        out = stage.process(self._make_task_with_plan(plan))
        kept_texts = [s["segments"][0]["text"] for s in out.data[_PLAN_DATA_KEY]]
        assert kept_texts == ["the quick brown fox", "hi"]
        assert out._metadata[_PRETRAIN_META_KEY]["dropped_repetition"] == 1
        assert out._metadata[_PRETRAIN_META_KEY]["kept_after_repetition_filter"] == 2
        # The override of planned_snippets reflects the post-filter count.
        assert out._metadata[_PRETRAIN_META_KEY]["planned_snippets"] == 2

    def test_post_init_validates(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="tokenizer_path"):
            SnippetRepetitionFilterStage(tokenizer_path="").__post_init__()
        with pytest.raises(ValueError, match="ngram_n"):
            SnippetRepetitionFilterStage(tokenizer_path=str(tmp_path), ngram_n=0).__post_init__()
        with pytest.raises(ValueError, match="ngram_max_count"):
            SnippetRepetitionFilterStage(tokenizer_path=str(tmp_path), ngram_max_count=0).__post_init__()

    def test_per_source_example_cap(self, tokenizer_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The per-source filtered_repetition_texts list is capped."""
        import nemo_curator.stages.audio.alm.pretrain.planning as m

        monkeypatch.setattr(m, "_MAX_FILTERED_TEXT_EXAMPLES", 3)
        stage = SnippetRepetitionFilterStage(tokenizer_path=str(tokenizer_dir))
        stage.__post_init__()
        stage.setup()

        repeat = "thank you for watching " * 10
        plan = [{"start": float(i), "end": float(i) + 30.0, "segments": [_ts(0.0, 30.0, repeat)]} for i in range(7)]
        out = stage.process(self._make_task_with_plan(plan))
        meta = out._metadata[_PRETRAIN_META_KEY]
        # All 7 are counted as dropped, but only the first 3 texts are retained.
        assert meta["dropped_repetition"] == 7
        assert len(meta["filtered_repetition_texts"]) == 3

    def test_process_is_idempotent_under_re_execution(self, tokenizer_dir: Path) -> None:
        """Calling process() twice on the same source must not double-count.

        Ray Data may fan a stage out across multiple blocks for the same
        upstream task, so process() can run more than once per source. The
        per-source counters and example list must be assignment-based so
        the second run overwrites rather than appends.
        """
        stage = SnippetRepetitionFilterStage(tokenizer_path=str(tokenizer_dir))
        stage.__post_init__()
        stage.setup()

        repeat = "thank you for watching " * 10
        # Build TWO tasks with the same plan; process() runs once per task.
        task1 = self._make_task_with_plan(
            [{"start": 0.0, "end": 30.0, "segments": [_ts(0.0, 30.0, repeat)]}]
        )
        task2 = self._make_task_with_plan(
            [{"start": 0.0, "end": 30.0, "segments": [_ts(0.0, 30.0, repeat)]}]
        )
        # First-pass result.
        out1 = stage.process(task1)
        meta1 = out1._metadata[_PRETRAIN_META_KEY]
        first_count = meta1["dropped_repetition"]
        first_texts = list(meta1["filtered_repetition_texts"])
        # Second pass on the same metadata dict (simulates re-execution).
        # We feed it a task that ALREADY carries the prior-pass metadata.
        task2._metadata[_PRETRAIN_META_KEY] = dict(meta1)
        out2 = stage.process(task2)
        meta2 = out2._metadata[_PRETRAIN_META_KEY]
        # Counters identical (overwrite semantics, not append).
        assert meta2["dropped_repetition"] == first_count
        assert meta2["filtered_repetition_texts"] == first_texts
