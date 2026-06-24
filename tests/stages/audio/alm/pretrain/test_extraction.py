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

"""Stage-level tests for ``nemo_curator.stages.audio.alm.pretrain.extraction``.

Two complementary scenarios:

* ``TestSnippetExtractionStageReal`` generates a short synthesized sine
  WAV (mono and stereo variants), feeds it through
  ``SnippetExtractionStage`` with a hand-rolled snippet plan, and checks
  the resulting tar shard's audio members match expected sample rate,
  channel count, and duration.
* ``TestSnippetExtractionStageDryRun`` exercises ``dry_run=True``: no
  audio I/O, no resampling, no tar writes -- only manifest metadata is
  emitted.  Useful as a fast path that doesn't need real audio fixtures.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from nemo_curator.stages.audio.alm.pretrain import SnippetExtractionStage
from nemo_curator.stages.audio.alm.pretrain.utils import _PLAN_DATA_KEY
from nemo_curator.tasks import AudioTask


def _open_member_as_audio(tar_path: str, member_name: str) -> tuple[np.ndarray, int, int]:
    """Read ``member_name`` from ``tar_path`` and return (data, sample_rate, channels)."""
    with tarfile.open(tar_path, "r") as t:
        f = t.extractfile(member_name)
        assert f is not None, f"member {member_name!r} not in {tar_path}"
        data, sr = sf.read(io.BytesIO(f.read()), always_2d=True)
    return data, sr, data.shape[1]


def _make_wav(path: Path, duration_sec: float, sample_rate: int, channels: int = 1) -> None:
    n = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n, endpoint=False, dtype=np.float32)
    mono = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    data = mono if channels == 1 else np.stack([mono] * channels, axis=-1)
    sf.write(str(path), data, sample_rate, subtype="PCM_16")


def _seg(start: float, end: float) -> dict:
    return {"speaker": "A", "start": start, "end": end, "text": "x", "text_ITN": "x", "words": []}


def _task_with_plan(audio_path: Path, plan: list[dict], extras: dict | None = None) -> AudioTask:
    data = {"id": "X", "audio_filepath": str(audio_path), _PLAN_DATA_KEY: plan}
    if extras:
        data.update(extras)
    return AudioTask(dataset_name="ds", data=data)


# ----------------------------------------------------------------------
# Real extraction
# ----------------------------------------------------------------------


class TestSnippetExtractionStageReal:
    @staticmethod
    def _make_stage(tmp_path: Path, *, output_format: str = "wav") -> tuple[SnippetExtractionStage, Path]:
        tar_path = tmp_path / "snips.tar"
        stage = SnippetExtractionStage(
            output_dir=str(tmp_path / "snips"),
            output_audio_tar_path=str(tar_path),
            target_sample_rate=16000,
            output_format=output_format,
        )
        stage.__post_init__()
        stage.setup_on_node()
        stage.setup()
        return stage, tar_path

    def test_writes_one_member_per_planned_snippet(self, tmp_path: Path) -> None:
        src = tmp_path / "src.wav"
        _make_wav(src, duration_sec=10.0, sample_rate=16000)

        plan = [
            {"start": 0.0, "end": 3.0, "segments": [_seg(0.0, 3.0)]},
            {"start": 4.0, "end": 8.0, "segments": [_seg(4.0, 8.0)]},
        ]
        stage, _tar_path = self._make_stage(tmp_path, output_format="wav")

        out = stage.process(_task_with_plan(src, plan))
        stage.teardown()
        assert len(out) == 2

        # Per-replica tar shard exists and has 2 members
        shards = sorted(tmp_path.glob("snips.tar.shard-*.tar"))
        assert len(shards) == 1
        with tarfile.open(str(shards[0]), "r") as t:
            members = sorted(t.getnames())
        assert members == sorted(o.data["audio_filepath"] for o in out)
        assert len(members) == 2

        # First member: 3.0s @ 16k = 48000 frames
        data, sr, channels = _open_member_as_audio(str(shards[0]), members[0])
        assert sr == 16000
        assert channels == 1
        assert data.shape[0] == pytest.approx(48000, abs=2)

    def test_resamples_when_source_rate_differs(self, tmp_path: Path) -> None:
        src = tmp_path / "src22k.wav"
        _make_wav(src, duration_sec=4.0, sample_rate=22050)

        plan = [{"start": 0.0, "end": 2.0, "segments": [_seg(0.0, 2.0)]}]
        stage, _tar = self._make_stage(tmp_path, output_format="wav")

        out = stage.process(_task_with_plan(src, plan))
        stage.teardown()
        assert len(out) == 1

        shards = sorted(tmp_path.glob("snips.tar.shard-*.tar"))
        assert len(shards) == 1
        member = out[0].data["audio_filepath"]
        data, sr, channels = _open_member_as_audio(str(shards[0]), member)
        assert sr == 16000
        assert channels == 1
        # 2s @ 16k = 32000 frames (allow tiny resample rounding)
        assert data.shape[0] == pytest.approx(32000, abs=4)

    def test_channel_averages_to_mono_for_stereo_source(self, tmp_path: Path) -> None:
        src = tmp_path / "stereo.wav"
        _make_wav(src, duration_sec=2.0, sample_rate=16000, channels=2)

        plan = [{"start": 0.0, "end": 1.0, "segments": [_seg(0.0, 1.0)]}]
        stage, _tar = self._make_stage(tmp_path, output_format="wav")

        out = stage.process(_task_with_plan(src, plan))
        stage.teardown()
        shards = sorted(tmp_path.glob("snips.tar.shard-*.tar"))
        member = out[0].data["audio_filepath"]
        _data, _sr, channels = _open_member_as_audio(str(shards[0]), member)
        assert channels == 1

    def test_emitted_metadata_uses_tar_basename(self, tmp_path: Path) -> None:
        src = tmp_path / "src.wav"
        _make_wav(src, duration_sec=5.0, sample_rate=16000)

        plan = [{"start": 1.0, "end": 4.0, "segments": [_seg(1.0, 4.0)]}]
        stage, _tar = self._make_stage(tmp_path, output_format="flac")

        # Source row has audio_sample_rate / audio_num_channels populated, so
        # the extractor's conditional updates fire.
        out = stage.process(_task_with_plan(src, plan, extras={"audio_sample_rate": 22050, "audio_num_channels": 2}))
        stage.teardown()
        d = out[0].data
        # Snippet ID + tar-internal basename (no slashes, no directory prefix)
        assert d["snippet_id"] == "X-1_000-4_000"
        assert d["audio_filepath"] == "X-1_000-4_000.flac"
        # Metadata reflects post-cut audio (overwritten because the source had these keys)
        assert d["audio_sample_rate"] == 16000
        assert d["audio_num_channels"] == 1
        # Duration approximately matches plan (within one frame at target sr)
        assert d["duration"] == pytest.approx(3.0, abs=1.0 / 16000)

    def test_missing_source_emits_stub(self, tmp_path: Path) -> None:
        plan = [{"start": 0.0, "end": 1.0, "segments": [_seg(0.0, 1.0)]}]
        task = _task_with_plan(tmp_path / "does_not_exist.wav", plan)
        stage, _tar = self._make_stage(tmp_path, output_format="wav")
        out = stage.process(task)
        stage.teardown()
        assert len(out) == 1
        assert out[0].data["snippet_id"] is None
        # The shard was still opened (setup ran) but contains no members.
        shards = sorted(tmp_path.glob("snips.tar.shard-*.tar"))
        assert len(shards) == 1
        with tarfile.open(str(shards[0]), "r") as t:
            assert t.getnames() == []

    def test_dry_run_writes_no_tar(self, tmp_path: Path) -> None:
        plan = [{"start": 0.0, "end": 1.0, "segments": [_seg(0.0, 1.0)]}]
        stage = SnippetExtractionStage(
            output_dir=str(tmp_path / "snips"),
            output_audio_tar_path=str(tmp_path / "snips.tar"),
            target_sample_rate=16000,
            output_format="flac",
            dry_run=True,
        )
        stage.setup_on_node()
        stage.setup()
        out = stage.process(_task_with_plan(tmp_path / "missing.wav", plan))
        stage.teardown()
        assert len(out) == 1
        # Manifest entry uses tar-internal basename even in dry-run
        assert out[0].data["audio_filepath"] == "X-0_000-1_000.flac"
        # No tar file or tar shards on disk
        assert not (tmp_path / "snips.tar").exists()
        assert sorted(tmp_path.glob("snips.tar.shard-*.tar")) == []


# ----------------------------------------------------------------------
# Dry-run extraction (no audio I/O, no tar writes)
# ----------------------------------------------------------------------


class TestSnippetExtractionStageDryRun:
    def test_emits_one_task_per_planned_snippet_no_audio_io(self, tmp_path: Path) -> None:
        snippet1 = {"start": 0.0, "end": 5.0, "segments": [_seg(0.0, 5.0)]}
        snippet2 = {"start": 5.0, "end": 12.5, "segments": [_seg(6.0, 12.0)]}
        extras = {
            "text": "WHOLE",
            "audio_sample_rate": 22050,
            "audio_num_channels": 2,
            "audio_size": 999,
            "actual_duration": 100.0,
            "proposed_duration": 100.0,
            "alignment": "STALE",
        }
        task = _task_with_plan(Path("/missing/source.wav"), [snippet1, snippet2], extras=extras)
        stage = SnippetExtractionStage(
            output_dir=str(tmp_path / "snips"),
            output_audio_tar_path=str(tmp_path / "snips.tar"),
            dry_run=True,
        )
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
        assert s0["text"] == "x"
        # Segments relativized
        assert s0["segments"][0]["start"] == pytest.approx(0.0)

    def test_zero_planned_emits_stub(self, tmp_path: Path) -> None:
        task = _task_with_plan(Path("/missing/source.wav"), [])
        stage = SnippetExtractionStage(
            output_dir=str(tmp_path / "snips"),
            output_audio_tar_path=str(tmp_path / "snips.tar"),
            dry_run=True,
        )
        out = stage.process(task)
        assert len(out) == 1
        assert out[0].data["snippet_id"] is None

    def test_invalid_output_format_rejected(self, tmp_path: Path) -> None:
        tar_path = str(tmp_path / "snips.tar")
        with pytest.raises(ValueError, match="output_format"):
            SnippetExtractionStage(output_dir=str(tmp_path), output_audio_tar_path=tar_path, output_format="m4a")
        with pytest.raises(ValueError, match="target_sample_rate"):
            SnippetExtractionStage(output_dir=str(tmp_path), output_audio_tar_path=tar_path, target_sample_rate=0)
