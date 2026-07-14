# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import wave
from pathlib import Path

from build_lhotse_variants import VARIANTS, segment_matches_variant
from manifest import (
    SegmentTaskConfig,
    build_segment_tasks,
    build_wer_distribution,
    segment_alignments,
    write_pipeline_outputs,
)
from stages import (
    ParallelInferenceAsrNemoStage,
    SegmentClipExtractionStage,
    normalize_wer_text,
    word_error_counts,
)

from nemo_curator.tasks import AudioTask

ROOT = Path(__file__).resolve().parents[2]


def test_normalization_and_wer_above_one_hundred_percent() -> None:
    assert normalize_wer_text("Café costs 2 dollars!") == "cafe costs two dollars"

    details = word_error_counts("hello", "hello extra words")

    assert details["insertions"] == 2
    assert details["wer_pct"] == 200.0


def test_fastmss_alignment_is_clipped_and_made_segment_relative() -> None:
    words = [(0.5, 1.2, "first"), (1.8, 2.4, "second"), (3.0, 3.5, "outside")]

    absolute, relative = segment_alignments(words, 1.0, 2.0)

    assert absolute == [
        {"word": "first", "start": 1.0, "end": 1.2},
        {"word": "second", "start": 1.8, "end": 2.0},
    ]
    assert relative == [
        {"symbol": "first", "start": 0.0, "duration": 0.2},
        {"symbol": "second", "start": 0.8, "duration": 0.2},
    ]


def test_distribution_reports_bounded_recommended_threshold() -> None:
    rows = [{"wer_pct": value} for value in (0.0, 5.0, 10.0, 25.0, 50.0, 150.0)]

    report = build_wer_distribution(rows, applied_threshold_pct=100.0)

    assert report["segments_with_wer"] == 6
    assert 25.0 <= report["recommended_threshold_pct"] <= 100.0
    assert sum(bucket["count"] for bucket in report["histogram"]) == 6


def test_lhotse_wer_variants_are_nested_and_require_alignment() -> None:
    aligned = {"alignment": [{"symbol": "word"}]}

    assert segment_matches_variant({**aligned, "wer_pct": 0.0}, VARIANTS[0])
    assert segment_matches_variant({**aligned, "wer_pct": 0.0}, VARIANTS[1])
    assert segment_matches_variant({**aligned, "wer_pct": 0.0}, VARIANTS[2])
    assert not segment_matches_variant({**aligned, "wer_pct": 5.0}, VARIANTS[0])
    assert segment_matches_variant({**aligned, "wer_pct": 5.0}, VARIANTS[1])
    assert segment_matches_variant({**aligned, "wer_pct": 5.0}, VARIANTS[2])
    assert not segment_matches_variant({**aligned, "wer_pct": 100.01}, VARIANTS[2])
    assert not segment_matches_variant({"alignment": [], "wer_pct": 0.0}, VARIANTS[0])


def test_output_writer_filters_high_wer_and_missing_alignment(tmp_path: Path) -> None:
    base = {
        "audio_filepath": "/synthetic/masked.wav",
        "start": 1.0,
        "end": 2.0,
        "duration": 1.0,
        "text": "reference",
        "text_raw": "Reference",
        "pred_text": "reference",
        "session_id": "session",
        "speaker_id": "speaker",
        "recording_id": "recording",
        "fastmss_textgrid": "/synthetic/recording_fastmss.TextGrid",
        "words": [{"word": "reference", "start": 1.1, "end": 1.5}],
    }
    tasks = [
        AudioTask(data={**base, "segment_index": 0, "wer_pct": 0.0, "alignment": [{"symbol": "reference"}]}),
        AudioTask(data={**base, "segment_index": 1, "wer_pct": 150.0, "alignment": [{"symbol": "bad"}]}),
        AudioTask(data={**base, "segment_index": 2, "wer_pct": 0.0, "alignment": []}),
    ]

    report = write_pipeline_outputs(
        tasks,
        output_dir=tmp_path,
        threshold_pct=100.0,
        require_fastmss_alignment=True,
    )

    manifest_rows = [
        json.loads(line) for line in (tmp_path / "manifests" / "recording.jsonl").read_text().splitlines()
    ]
    audit_rows = [json.loads(line) for line in (tmp_path / "segments_with_wer.jsonl").read_text().splitlines()]
    assert report["kept_segments"] == 1
    assert [row["segment_index"] for row in manifest_rows] == [0]
    assert audit_rows[1]["rejection_reasons"] == ["wer_above_threshold"]
    assert audit_rows[2]["rejection_reasons"] == ["missing_fastmss_alignment"]


def test_segment_clip_extraction_uses_exact_interval(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    with wave.open(str(source), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x01\x00" * 32000)
    task = AudioTask(
        data={
            "audio_filepath": str(source),
            "recording_id": "recording",
            "segment_index": 0,
            "start": 0.5,
            "end": 1.25,
        }
    )

    result = SegmentClipExtractionStage(scratch_dir=str(tmp_path / "scratch")).process(task)

    with wave.open(result.data["segment_audio_filepath"], "rb") as reader:
        assert reader.getframerate() == 16000
        assert reader.getnchannels() == 1
        assert reader.getnframes() == 12000


def test_ultrashort_segment_clip_is_padded_to_model_minimum(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    with wave.open(str(source), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x01\x00" * 32000)
    task = AudioTask(
        data={
            "audio_filepath": str(source),
            "recording_id": "recording",
            "segment_index": 0,
            "start": 1.0,
            "end": 1.001,
        }
    )

    result = SegmentClipExtractionStage(scratch_dir=str(tmp_path / "scratch")).process(task)

    with wave.open(result.data["segment_audio_filepath"], "rb") as reader:
        assert reader.getnframes() == 1600
    assert result.data["start"] == 1.0
    assert result.data["end"] == 1.001
    assert round(result.data["clip_end"] - result.data["clip_start"], 6) == 0.1


def test_build_segment_tasks_joins_manifest_masked_audio_and_fastmss(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    masked_dir = tmp_path / "audio_16k_masked"
    textgrid_dir = tmp_path / "textgrids"
    session_dir = data_root / "session"
    session_dir.mkdir(parents=True)
    masked_dir.mkdir()
    textgrid_dir.mkdir()
    (session_dir / "machine_generated_transcript.json").write_text(
        json.dumps(
            {
                "transcript": [
                    {"speaker": "speaker", "start": 0.0, "end": 1.0, "text": "first"},
                    {"speaker": "speaker", "start": 1.0, "end": 2.0, "text": "second"},
                ]
            }
        ),
        encoding="utf-8",
    )
    recording = "speaker_session_postprocessed"
    (masked_dir / f"{recording}.wav").touch()
    (textgrid_dir / f"{recording}_fastmss.TextGrid").write_text(
        """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 2
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 2
        intervals: size = 2
        intervals [1]:
            xmin = 0.1
            xmax = 0.5
            text = "first"
        intervals [2]:
            xmin = 1.1
            xmax = 1.6
            text = "second"
""",
        encoding="utf-8",
    )

    tasks = build_segment_tasks(
        SegmentTaskConfig(
            data_root=data_root,
            masked_audio_dir=masked_dir,
            textgrid_dir=textgrid_dir,
            sessions_file=None,
            shard_count=1,
            shard_index=0,
        )
    )

    assert len(tasks) == 2
    assert tasks[0].data["recording_id"] == recording
    assert tasks[0].data["alignment"] == [{"symbol": "first", "start": 0.1, "duration": 0.4}]
    assert tasks[1].data["alignment"] == [{"symbol": "second", "start": 0.1, "duration": 0.5}]


def test_local_two_gpu_launcher_uses_one_xenna_cluster() -> None:
    launcher = (ROOT / "parakeet_wer" / "run_local_2gpu.sh").read_text(encoding="utf-8")

    assert 'GPU_IDS="${GPU_IDS:-0,1}"' in launcher
    assert 'CUDA_VISIBLE_DEVICES="$GPU_0,$GPU_1"' in launcher
    assert "SHARD_COUNT=1" in launcher
    assert "SHARD_INDEX=0" in launcher
    assert "ASR_WORKERS=2" in launcher
    assert 'SCRATCH_DIR="$SCRATCH_DIR"' in launcher
    assert "ASRModel.from_pretrained" in launcher
    assert 'export NEMO_CACHE_DIR="$MODEL_CACHE_DIR"' in launcher
    assert "cache_dir=" not in launcher
    assert "analyze_wer_distribution.py" in launcher
    assert "local_2gpu.log" in launcher


def test_cluster_launcher_builds_and_merges_lhotse_after_array() -> None:
    cluster_dir = ROOT / "parakeet_wer" / "cluster"
    launcher = (cluster_dir / "run_multinode.sh").read_text(encoding="utf-8")
    node = (cluster_dir / "run_node.sh").read_text(encoding="utf-8")
    merge = (cluster_dir / "merge_outputs.sh").read_text(encoding="utf-8")

    assert '--dependency "afterok:$ARRAY_JOB_ID"' in launcher
    assert "BUILD_LHOTSE=1 \\" in node
    assert "merge_lhotse_variants.py" in merge
    assert "analyze_wer_distribution.py" in merge
    assert "scp " not in launcher + node + merge
    assert "rsync " not in launcher + node + merge


def test_parallel_asr_stage_requests_two_workers() -> None:
    stage = ParallelInferenceAsrNemoStage(model_name="synthetic-model", worker_count=2)

    assert stage.num_workers() == 2
    assert stage.resources.gpus == 0
