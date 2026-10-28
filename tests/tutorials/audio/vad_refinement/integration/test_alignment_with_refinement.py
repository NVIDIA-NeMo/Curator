# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
import soundfile as sf

from tutorials.audio.vad_refinement import alignment_with_refinement as pipeline
from tutorials.audio.vad_refinement.rectify_vad_boundaries import VADPredictions, _marked_words

pytest.importorskip("nemo", reason="Physical-shard integration tests require the Curator NeMo audio environment")


class AlignerFixture:
    def __init__(self, root: Path, *, missing: bool = False) -> None:
        from nemo_curator.stages.audio.tagging.inference.nemo_asr_align import NeMoASRAlignerStage

        self.root = root
        self.missing = missing
        self.stage = NeMoASRAlignerStage(decoder_type="ctc", use_cuda_graphs=False)
        self.stage._asr_model = SimpleNamespace(
            cfg=SimpleNamespace(preprocessor=SimpleNamespace(window_stride=0.0125)),
            transcribe=self.transcribe,
        )

    def process_batch(self, tasks: list) -> list:
        return self.stage.process_batch(tasks)

    def transcribe(self, paths: list[str], **_kwargs: object) -> list[SimpleNamespace]:
        assert all(Path(path).is_file() for path in paths)
        stamps = (
            []
            if self.missing
            else [
                {"start_offset": 1, "end_offset": 3, "word": "one"},
                {"start_offset": 7, "end_offset": 9, "word": "two"},
            ]
        )
        return [SimpleNamespace(timestamp={"word": stamps}, word_confidence=None) for _ in paths]


class DetectorFixture:
    metadata: ClassVar[dict] = {"name": "synthetic-activity"}

    def __call__(self, path: Path) -> VADPredictions:
        samples, rate = sf.read(path)
        return VADPredictions(silero=((0, len(samples) / rate),), secondary=((0, len(samples) / rate),))


def _input_manifest(root: Path, *, count: int = 2) -> Path:
    rows = []
    for index in range(count):
        path = root / str(index) / "same.wav"
        path.parent.mkdir(parents=True)
        sf.write(path, np.concatenate([np.zeros(8000), np.ones(16000) * 0.25]), 16000, subtype="PCM_16")
        rows.append(
            {
                "audio_filepath": str(path),
                "text": "one two",
                "offset": 0.5,
                "duration": 1.0,
                "sampling_rate": 16000,
                "speaker": "same-speaker",
            }
        )
    manifest = root / "source.jsonl"
    pipeline._write_jsonl(manifest, rows)
    return manifest


def test_main_aligner_tar_refinement_roundtrip_after_relocation(tmp_path: Path) -> None:
    source = _input_manifest(tmp_path / "inputs")
    root = tmp_path / "work"
    aligned = pipeline.align_tasks(
        pipeline.input_tasks(None, source), root, AlignerFixture(root / "grids"), batch_size=2
    )
    rows = list(pipeline.read_rows(aligned[0]))
    assert len({row["segment_id"] for row in rows}) == 2
    assert all(row["offset"] == 0 and row["source_offset"] == 0.5 for row in rows)
    assert all(row["duration"] == 1.0 for row in rows)
    samples, _ = sf.read(rows[0]["audio_filepath"])
    np.testing.assert_array_equal(samples, np.full(16000, 0.25))
    originals = [_marked_words(Path(row["mfa_textgrid"])) for row in rows]
    pipeline.pack_shards(aligned, root / "shards", records_per_shard=1)
    relocated = tmp_path / "relocated"
    shutil.move(root / "shards", relocated)
    # The shard must be sufficient: remove original MFA and decoded clip files.
    shutil.rmtree(root)
    first = pipeline.refine_shards(relocated, tmp_path / "result", DetectorFixture(), worker_count=2, worker_index=0)
    second = pipeline.refine_shards(relocated, tmp_path / "result", DetectorFixture(), worker_count=2, worker_index=1)
    assert len(first) == len(second) == 1
    assert first[0].parent != second[0].parent
    for index, manifest in enumerate(sorted(first + second)):
        (row,) = pipeline.read_rows(manifest)
        assert row["offset"] == 0
        assert row["source_offset"] == 0.5
        words = _marked_words(Path(row["vad_refined_textgrid"]))
        assert [word[2] for word in words] == ["one", "two"]
        for before, after in zip(originals[index], words, strict=True):
            assert after[0] <= before[0]
            assert after[1] >= before[1]
        assert words[0][1] == words[1][0] == pytest.approx(0.5)
        assert row["alignment_source"] == "curator_nemo_asr"
        audit = json.loads(Path(row["refinement_audit"]).read_text())
        assert audit["anchor_source"] == "curator_nemo_asr"
        assert "mfa" not in audit["refinement_algorithm"]["weights"]
        assert Path(row["mfa_textgrid"]).is_file()
        marker = json.loads((manifest.parent / "complete.json").read_text())
        assert marker["rows"] == 1


def test_missing_asr_words_prevents_publishing_aligned_batch(tmp_path: Path) -> None:
    source = _input_manifest(tmp_path / "inputs", count=1)
    root = tmp_path / "work"
    with pytest.raises(ValueError, match="no word timestamps"):
        pipeline.align_tasks(
            pipeline.input_tasks(None, source), root, AlignerFixture(root / "grids", missing=True), batch_size=1
        )
    assert not list((root / "aligned").glob("*.jsonl"))


def test_supplied_transcript_is_preserved_separately_from_asr(tmp_path: Path) -> None:
    source = _input_manifest(tmp_path / "inputs", count=1)
    (row,) = pipeline.read_rows(source)
    row["text"] = "a different supplied transcript"
    pipeline._write_jsonl(source, [row])
    root = tmp_path / "work"
    (manifest,) = pipeline.align_tasks(
        pipeline.input_tasks(None, source), root, AlignerFixture(root / "grids"), batch_size=1
    )
    (aligned,) = pipeline.read_rows(manifest)
    assert aligned["source_text"] == "a different supplied transcript"
    assert aligned["text"] == "one two"


def test_overlapping_main_timestamps_are_reported_without_repair(tmp_path: Path) -> None:
    row = {"duration": 1.0, "segment_id": "clip", "text": "original"}
    result = {
        "text": "one two",
        "alignment": [
            {"word": "one", "start": 0.1, "end": 0.7},
            {"word": "two", "start": 0.6, "end": 0.9},
        ],
    }
    with pytest.raises(ValueError, match="Invalid Curator word timestamp"):
        pipeline.write_alignment(row, result, tmp_path)
    assert not list(tmp_path.rglob("*.TextGrid"))


def test_non_tar_reader_batches_preserve_original_row_identity(tmp_path: Path) -> None:
    source = _input_manifest(tmp_path / "inputs", count=3)
    batches = list(pipeline.input_tasks(None, source, batch_size=1))
    assert [len(tasks) for _, tasks in batches] == [1, 1, 1]
    assert [tasks[0]._metadata["_shard_input_id"] for _, tasks in batches] == ["0", "1", "2"]
    root = tmp_path / "work"
    aligned = pipeline.align_tasks(batches, root, AlignerFixture(root / "grids"), batch_size=1)
    assert len(aligned) == 3
    rows = [row for manifest in aligned for row in pipeline.read_rows(manifest)]
    assert len({row["segment_id"] for row in rows}) == 3


def test_alignment_workers_own_disjoint_batches(tmp_path: Path) -> None:
    source = _input_manifest(tmp_path / "inputs", count=6)
    root = tmp_path / "work"
    outputs = []
    for worker_index in range(2):
        outputs.append(
            pipeline.align_tasks(
                pipeline.input_tasks(None, source, batch_size=1),
                root,
                AlignerFixture(root / "grids"),
                batch_size=1,
                worker_count=2,
                worker_index=worker_index,
            )
        )
    assert len(outputs[0]) + len(outputs[1]) == 6
    assert not set(outputs[0]) & set(outputs[1])


def test_duplicate_aligned_ids_do_not_publish_shard(tmp_path: Path) -> None:
    source = _input_manifest(tmp_path / "inputs", count=1)
    root = tmp_path / "work"
    manifests = pipeline.align_tasks(
        pipeline.input_tasks(None, source), root, AlignerFixture(root / "grids"), batch_size=1
    )
    (row,) = pipeline.read_rows(manifests[0])
    pipeline._write_jsonl(manifests[0], [row, row])
    with pytest.raises(ValueError, match="Duplicate aligned clip"):
        pipeline.pack_shards(manifests, root / "shards", records_per_shard=2)
    assert not (root / "shards/shard-000000").exists()


def test_missing_tar_audio_does_not_mark_shard_complete(tmp_path: Path) -> None:
    source = _input_manifest(tmp_path / "inputs", count=1)
    root = tmp_path / "work"
    aligned = pipeline.align_tasks(
        pipeline.input_tasks(None, source), root, AlignerFixture(root / "grids"), batch_size=1
    )
    (manifest,) = pipeline.pack_shards(aligned, root / "shards", records_per_shard=1)
    (row,) = pipeline.read_rows(manifest)
    row["audio_filepath"] = "missing.wav"
    pipeline._write_jsonl(manifest, [row])
    with pytest.raises((ValueError, RuntimeError), match=r"decode|manifest|matching"):
        pipeline.refine_shards(root / "shards", root / "result", DetectorFixture())
    assert not list((root / "result").rglob("complete.json"))
