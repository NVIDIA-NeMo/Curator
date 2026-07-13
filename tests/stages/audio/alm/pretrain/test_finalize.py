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

"""Tests for ``prepare_audio_pretrain_outputs`` / ``finalize_audio_pretrain_outputs``.

Exercises the driver-side helpers that:

* clean up stale shard files before a run, and
* merge the per-replica manifest / metrics / tar shards into the final
  outputs, including reconciliation of manifest rows against the merged
  tar (dropped ``missing_audio`` / ``corrupted_audio`` are surfaced in
  the metrics summary).
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from nemo_curator.stages.audio.alm.pretrain import (
    finalize_audio_pretrain_outputs,
    prepare_audio_pretrain_outputs,
)
from nemo_curator.stages.audio.alm.pretrain.utils import _make_shard_path


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

    def test_finalize_preserves_existing_tar_when_shards_have_no_members(self, tmp_path: Path) -> None:
        manifest = str(tmp_path / "snippets.jsonl")
        metrics = str(tmp_path / "metrics.json")
        tar_path = str(tmp_path / "snippets.tar")

        with tarfile.open(tar_path, "w") as t:
            body = b"previous-good"
            ti = tarfile.TarInfo(name="previous.flac")
            ti.size = len(body)
            t.addfile(ti, io.BytesIO(body))

        Path(_make_shard_path(tar_path, "tar")).touch()

        finalize_audio_pretrain_outputs(manifest, metrics, tar_path)

        with tarfile.open(tar_path, "r") as t:
            names = t.getnames()
            assert names == ["previous.flac"]
            payload = t.extractfile("previous.flac").read()
        assert payload == b"previous-good"

    def test_finalize_drops_manifest_rows_missing_from_tar(self, tmp_path: Path) -> None:
        """Manifest reconciliation: rows whose tar member is missing get dropped
        and surfaced as `dropped.missing_audio` in the merged metrics."""
        manifest = str(tmp_path / "snippets.jsonl")
        metrics = str(tmp_path / "metrics.json")
        tar_path = str(tmp_path / "snippets.tar")

        # Manifest writer shard with 3 entries; tar shard has only 2 of those
        # members (the third member's data was "truncated" -- simulated by
        # never adding it to the tar at all).
        ms = _make_shard_path(manifest, "jsonl")
        with open(ms, "w") as f:
            f.writelines(
                json.dumps(
                    {
                        "id": "X",
                        "snippet_id": sid,
                        "audio_filepath": f"{sid}.flac",
                        "duration": 1.0,
                        "segments": [{"start": 0.0, "end": 1.0, "text": "x"}],
                    }
                )
                + "\n"
                for sid in ("X-0_000-1_000", "X-1_000-2_000", "X-2_000-3_000")
            )
        # Metrics shard so finalize writes a merged metrics.json that
        # _patch_metrics_post_reconcile can update.
        ms_metrics = _make_shard_path(metrics, "jsonl")
        with open(ms_metrics, "w") as f:
            f.writelines(
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
                for _ in range(3)
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
        # Output totals are rebuilt from the post-reconcile manifest, so the
        # dropped row no longer counts toward the snippet/segment/duration
        # tallies or the per-original out_* fields.
        assert summary["num_output_snippets"] == 2
        assert summary["output_total_segments"] == 2
        assert summary["output_total_duration_sec"] == 2.0
        assert summary["snippet_duration_histogram_30s"] == {"0-30": 2}
        per_x = next(e for e in summary["per_original"] if e["id"] == "X")
        assert per_x["out_snippets"] == 2
        assert per_x["out_segments"] == 2
        assert per_x["out_duration_sec"] == 2.0

    def test_finalize_reconciles_custom_audio_filepath_key(self, tmp_path: Path) -> None:
        """A non-default audio filepath key should reconcile against that key,
        not silently drop every manifest row by reading ``audio_filepath``."""
        manifest = str(tmp_path / "snippets.jsonl")
        metrics = str(tmp_path / "metrics.json")
        tar_path = str(tmp_path / "snippets.tar")
        audio_key = "audio_path"

        sid = "X-0_000-1_000"
        ms = _make_shard_path(manifest, "jsonl")
        with open(ms, "w") as f:
            f.write(
                json.dumps(
                    {
                        "id": "X",
                        "snippet_id": sid,
                        audio_key: f"{sid}.flac",
                        "duration": 1.0,
                        "segments": [{"start": 0.0, "end": 1.0, "text": "x"}],
                    }
                )
                + "\n"
            )

        ms_metrics = _make_shard_path(metrics, "jsonl")
        with open(ms_metrics, "w") as f:
            f.write(
                json.dumps(
                    {
                        "id": "X",
                        "in_segments": 1,
                        "in_duration_sec": 1.0,
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
            ti = tarfile.TarInfo(name=f"{sid}.flac")
            ti.size = len(flac_bytes)
            t.addfile(ti, io.BytesIO(flac_bytes))

        finalize_audio_pretrain_outputs(manifest, metrics, tar_path, audio_filepath_key=audio_key)

        rows = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line]
        assert len(rows) == 1
        assert rows[0][audio_key] == f"{sid}.flac"
        summary = json.loads(Path(metrics).read_text(encoding="utf-8"))
        assert "missing_audio" not in summary["dropped"]
        assert summary["num_output_snippets"] == 1

    def test_finalize_drops_manifest_rows_with_unreadable_audio(self, tmp_path: Path) -> None:
        """Manifest reconciliation: rows whose tar member fails the audio
        header/duration check get dropped (e.g. truncated payload from a
        worker killed mid-write) and counted under
        `dropped.corrupted_audio` in the merged metrics."""
        manifest = str(tmp_path / "snippets.jsonl")
        metrics = str(tmp_path / "metrics.json")
        tar_path = str(tmp_path / "snippets.tar")

        # Three entries: two with valid FLAC payloads, one with bogus bytes
        # (header unreadable -> sf.info raises -> row dropped).
        ms = _make_shard_path(manifest, "jsonl")
        with open(ms, "w") as f:
            f.writelines(
                json.dumps(
                    {
                        "id": "X",
                        "snippet_id": sid,
                        "audio_filepath": f"{sid}.flac",
                        "duration": 1.0,
                        "segments": [{"start": 0.0, "end": 1.0, "text": "x"}],
                    }
                )
                + "\n"
                for sid in ("X-0_000-1_000", "X-1_000-2_000", "X-2_000-3_000")
            )
        ms_metrics = _make_shard_path(metrics, "jsonl")
        with open(ms_metrics, "w") as f:
            f.writelines(
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
                for _ in range(3)
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
        # Output totals reflect the post-reconcile manifest, not the
        # pre-reconcile shard records.
        assert summary["num_output_snippets"] == 2
        assert summary["output_total_duration_sec"] == 2.0
        per_x = next(e for e in summary["per_original"] if e["id"] == "X")
        assert per_x["out_snippets"] == 2
        assert per_x["out_duration_sec"] == 2.0

    def test_finalize_rebuilds_per_original_when_all_snippets_dropped(self, tmp_path: Path) -> None:
        """Reconcile drops every snippet of source ``Y`` (no tar members) but
        keeps both of source ``X``.  Post-rebuild, ``X`` keeps its 2 snippets
        and ``Y`` reads 0 across the board, even though both sources had
        non-stub metrics shard records before reconcile."""
        manifest = str(tmp_path / "snippets.jsonl")
        metrics = str(tmp_path / "metrics.json")
        tar_path = str(tmp_path / "snippets.tar")

        ms = _make_shard_path(manifest, "jsonl")
        with open(ms, "w") as f:
            f.writelines(
                json.dumps(
                    {
                        "id": pid,
                        "snippet_id": sid,
                        "audio_filepath": f"{sid}.flac",
                        "duration": dur,
                        "segments": [{"start": 0.0, "end": dur, "text": "x"}],
                    }
                )
                + "\n"
                for pid, sid, dur in (
                    ("X", "X-0_000-1_000", 1.0),
                    ("X", "X-1_000-2_000", 1.0),
                    ("Y", "Y-0_000-1_000", 1.0),
                )
            )
        ms_metrics = _make_shard_path(metrics, "jsonl")
        with open(ms_metrics, "w") as f:
            f.writelines(
                json.dumps(
                    {
                        "id": pid,
                        "in_segments": 1,
                        "in_duration_sec": 1.0,
                        "dropped": {},
                        "is_stub": False,
                        "out_segments": 1,
                        "out_duration_sec": 1.0,
                    }
                )
                + "\n"
                for pid in ("X", "X", "Y")
            )

        flac_buf = io.BytesIO()
        sf.write(flac_buf, np.zeros(160, dtype=np.float32), 16000, format="FLAC")
        flac_bytes = flac_buf.getvalue()
        ts = _make_shard_path(tar_path, "tar")
        with tarfile.open(ts, "w") as t:
            # Only X's snippets are written to the tar; Y is missing entirely.
            for sid in ("X-0_000-1_000", "X-1_000-2_000"):
                ti = tarfile.TarInfo(name=f"{sid}.flac")
                ti.size = len(flac_bytes)
                t.addfile(ti, io.BytesIO(flac_bytes))

        finalize_audio_pretrain_outputs(manifest, metrics, tar_path)

        summary = json.loads(Path(metrics).read_text(encoding="utf-8"))
        assert summary["dropped"]["missing_audio"] == 1
        # X kept both snippets, Y was fully dropped.
        assert summary["num_output_snippets"] == 2
        per_x = next(e for e in summary["per_original"] if e["id"] == "X")
        per_y = next(e for e in summary["per_original"] if e["id"] == "Y")
        assert per_x["out_snippets"] == 2
        assert per_x["out_duration_sec"] == 2.0
        assert per_y["out_snippets"] == 0
        assert per_y["out_duration_sec"] == 0.0

    def test_finalize_skips_metrics_shard_rows_missing_id(self, tmp_path: Path) -> None:
        manifest = str(tmp_path / "snippets.jsonl")
        metrics = str(tmp_path / "metrics.json")
        tar_path = str(tmp_path / "snippets.tar")

        ms_metrics = _make_shard_path(metrics, "jsonl")
        with open(ms_metrics, "w") as f:
            f.write(json.dumps({"out_duration_sec": 9.0, "is_stub": False}) + "\n")
            f.write(
                json.dumps(
                    {
                        "id": "X",
                        "in_segments": 1,
                        "in_duration_sec": 1.0,
                        "dropped": {},
                        "is_stub": False,
                        "out_segments": 1,
                        "out_duration_sec": 1.0,
                    }
                )
                + "\n"
            )

        finalize_audio_pretrain_outputs(manifest, metrics, tar_path)

        summary = json.loads(Path(metrics).read_text(encoding="utf-8"))
        assert summary["num_input_audios"] == 1
        assert summary["per_original"][0]["id"] == "X"

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
