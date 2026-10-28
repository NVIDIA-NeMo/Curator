# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
import sys
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
import soundfile as sf

from tutorials.audio.vad_refinement import alignment_with_refinement as pipeline
from tutorials.audio.vad_refinement.rectify_vad_boundaries import VADPredictions, _marked_words

pytest.importorskip("nemo", reason="Physical-shard integration tests require the Curator NeMo audio environment")


class AlignerFixture:
    def __init__(self, root: Path, *, missing: bool = False) -> None:
        from nemo_curator.stages.audio.alignment import MFAAlignmentStage

        root.mkdir(parents=True, exist_ok=True)
        executable = root / "fixture_mfa"
        script = """import sys, os
from pathlib import Path
from textgrid import TextGrid, IntervalTier
assert sys.argv[1] == 'align'
assert str(Path(sys.argv[0]).parent) == os.environ['PATH'].split(os.pathsep)[0]
assert '--beam' in sys.argv
assert '-j' in sys.argv
corpus, output = Path(sys.argv[2]), Path(sys.argv[5])
for transcript in corpus.rglob('*.txt'):
    assert transcript.read_text().strip()
    destination = output / transcript.parent.name / (transcript.stem + '.TextGrid')
    destination.parent.mkdir(parents=True, exist_ok=True)
    grid = TextGrid(minTime=0, maxTime=1)
    tier = IntervalTier(name='words', minTime=0, maxTime=1)
    tier.add(0.1000068, 0.3000032, 'one')
    tier.add(0.7000068, 0.9000032, 'two')
    grid.append(tier)
    grid.write(str(destination))
"""
        if missing:
            script = "import sys; sys.exit(0)\n"
        executable.write_text(f"#!{sys.executable}\n" + script)
        executable.chmod(0o755)
        self.stage = MFAAlignmentStage(
            output_dir=str(root / "outputs"),
            mfa_command=str(executable),
            dictionary="fixture-dictionary",
            acoustic_model="fixture-model",
            num_jobs=1,
            g2p_model="",
            create_rttm=False,
            create_ctm=False,
            mfa_root_dir=str(root / "state"),
            copy_models_to_local=False,
        )
        self.stage.setup_on_node()
        self.stage.setup()

    def process_batch(self, tasks: list) -> list:
        return self.stage.process_batch(tasks)


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


def test_montreal_tar_refinement_roundtrip_after_relocation(tmp_path: Path) -> None:
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
        assert words[0][1] == words[1][0] == pytest.approx(0.500005)
        assert row["alignment_source"] == "montreal_forced_aligner"
        audit = json.loads(Path(row["refinement_audit"]).read_text())
        assert audit["anchor_source"] == "montreal_forced_aligner"
        assert "mfa" not in audit["refinement_algorithm"]["weights"]
        assert Path(row["mfa_textgrid"]).is_file()
        marker = json.loads((manifest.parent / "complete.json").read_text())
        assert marker["rows"] == 1


def test_missing_mfa_output_prevents_publishing_aligned_batch(tmp_path: Path) -> None:
    source = _input_manifest(tmp_path / "inputs", count=1)
    root = tmp_path / "work"
    with pytest.raises(ValueError, match="no word alignment"):
        pipeline.align_tasks(
            pipeline.input_tasks(None, source), root, AlignerFixture(root / "grids", missing=True), batch_size=1
        )
    assert not list((root / "aligned").glob("*.jsonl"))


def test_supplied_transcript_is_not_replaced(tmp_path: Path) -> None:
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
    assert aligned["text"] == "a different supplied transcript"


def test_mfa_textgrid_is_copied_without_rewriting(tmp_path: Path) -> None:
    from textgrid import IntervalTier, TextGrid

    source = tmp_path / "source.TextGrid"
    grid = TextGrid(minTime=0, maxTime=1)
    tier = IntervalTier(name="words", minTime=0, maxTime=1)
    tier.add(0.1000068, 0.9000032, "word")
    grid.append(tier)
    grid.write(str(source))
    row = {"segment_id": "clip", "text": "original"}
    pipeline.write_alignment(row, {"textgrid_filepath": str(source)}, tmp_path / "output")
    assert Path(row["mfa_textgrid"]).read_bytes() == source.read_bytes()
    assert row["text"] == "original"
    assert row["alignment_source"] == "montreal_forced_aligner"


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
