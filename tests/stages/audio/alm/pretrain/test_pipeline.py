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

"""End-to-end pipeline test on a small committed sample of long-form
diarized audio manifest rows.

Drives every stage in sequence (no Ray) in dry-run mode -- the test
doesn't need the source audios to exist on disk, only the manifest --
and verifies the final manifest + metrics summary against the spec.

The fixture ``sample_long_form_manifest.jsonl`` next to this file is a
10-row excerpt of real production data, large enough to exercise the
algorithm against realistic segment counts and timestamp jitter without
checking in the multi-hundred-MB source manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

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
from nemo_curator.tasks import _EmptyTask

_LOCAL_MANIFEST = Path(__file__).resolve().parent / "sample_long_form_manifest.jsonl"
_NUM_ROWS = 10
_MAX_DURATION_SEC = 30.0
_TARGET_SR = 16000


def _first_n_rows(path: Path, n: int) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for raw in f:
            if len(rows) >= n:
                break
            stripped = raw.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _run_pipeline_inline(  # noqa: PLR0913
    *,
    input_manifest: Path,
    audio_dir: Path,
    output_dir: Path,
    output_manifest: Path,
    metrics_path: Path,
    max_duration_sec: float,
    target_sample_rate: int,
) -> None:
    """Run every stage in sequence on the same Python process (no executor)."""
    prepare_audio_pretrain_outputs(str(output_manifest), str(metrics_path))

    reader = ReadLongFormManifestStage(input_manifest=str(input_manifest), audio_dir=str(audio_dir))
    reader.__post_init__()
    overlap = OverlapFilterStage()
    planner = SnippetCutPlannerStage(max_duration_sec=max_duration_sec)
    planner.__post_init__()
    extractor = SnippetExtractionStage(
        output_dir=str(output_dir),
        target_sample_rate=target_sample_rate,
        output_format="flac",
        dry_run=True,
    )
    extractor.__post_init__()
    extractor.setup_on_node()
    extractor.setup()
    writer = SnippetManifestWriterStage(output_path=str(output_manifest))
    writer.__post_init__()
    writer.setup_on_node()
    writer.setup()
    aggregator = PretrainMetricsAggregatorStage(output_path=str(metrics_path))
    aggregator.__post_init__()
    aggregator.setup_on_node()
    aggregator.setup()

    for input_task in reader.process(_EmptyTask(task_id="e", dataset_name="e", data=None)):
        filtered = overlap.process(input_task)
        planned = planner.process(filtered)
        for snippet_task in extractor.process(planned):
            writer.process(snippet_task)
            aggregator.process(snippet_task)

    finalize_audio_pretrain_outputs(str(output_manifest), str(metrics_path))


class TestPipelineEndToEndOnLocalManifest:
    """Drive the pipeline on the first 10 rows of the local ``test.jsonl``.

    Asserts on the resulting manifest + metrics summary -- not on
    specific row counts (those depend on the user's data) but on the
    structural invariants the spec requires.
    """

    def test_dry_run_produces_valid_outputs(self, tmp_path: Path) -> None:
        rows = _first_n_rows(_LOCAL_MANIFEST, _NUM_ROWS)
        assert len(rows) == _NUM_ROWS, f"expected {_NUM_ROWS} rows from {_LOCAL_MANIFEST}, got {len(rows)}"

        # Stage the input manifest in a tmp dir; audio_dir stays empty (dry-run).
        in_manifest = tmp_path / "in.jsonl"
        with in_manifest.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        audio_dir = tmp_path / "audios"
        audio_dir.mkdir()
        output_dir = tmp_path / "snippets"
        output_manifest = tmp_path / "snippets.jsonl"
        metrics_path = tmp_path / "metrics.json"

        _run_pipeline_inline(
            input_manifest=in_manifest,
            audio_dir=audio_dir,
            output_dir=output_dir,
            output_manifest=output_manifest,
            metrics_path=metrics_path,
            max_duration_sec=_MAX_DURATION_SEC,
            target_sample_rate=_TARGET_SR,
        )

        # Output files exist
        assert output_manifest.exists()
        assert metrics_path.exists()

        # All shards merged + cleaned up
        leftover = sorted(tmp_path.glob("**/*.shard-*"))
        assert leftover == [], f"shard files not cleaned up: {leftover}"

        # Manifest: at least one snippet (10 long-form audios with many
        # segments each won't all be filtered to zero in a sane spec).
        with output_manifest.open(encoding="utf-8") as f:
            snippet_rows = [json.loads(line) for line in f if line.strip()]
        assert snippet_rows, "expected at least one emitted snippet"

        self._assert_snippet_row_schema(snippet_rows, source_ids={r["id"] for r in rows})

        # Metrics: structural sanity
        summary = json.loads(metrics_path.read_text(encoding="utf-8"))
        self._assert_metrics_summary_schema(
            summary, expected_input_audios=_NUM_ROWS, expected_output_snippets=len(snippet_rows)
        )

    @staticmethod
    def _assert_snippet_row_schema(rows: list[dict], source_ids: set[str]) -> None:
        for row in rows:
            assert row["id"] in source_ids, f"snippet id {row['id']!r} not in source manifest"
            assert isinstance(row["snippet_id"], str)
            assert row["snippet_id"].startswith(row["id"] + "_")
            assert row["audio_filepath"].endswith(".flac")
            assert isinstance(row["duration"], (int, float))
            assert row["duration"] > 0
            # Audio-property fields point to the snippet, not the source
            assert row["audio_sample_rate"] == _TARGET_SR
            assert row["audio_num_channels"] == 1
            # Source-only fields removed
            assert "alignment" not in row
            assert "audio_size" not in row
            assert "resampled_audio_filepath" not in row
            assert row.get("swift_audio_filepath") == ""
            # Segments are relativized: every segment starts at >= 0 and ends
            # within the snippet duration (allow a small float tolerance).
            assert isinstance(row["segments"], list)
            assert row["segments"], "snippet should not be empty (no_text was filtered)"
            # `relativize_segments` clamps every timestamp into
            # [0, snippet_duration] so downstream consumers never have to
            # cope with real-world data jitter at segment / word boundaries.
            for seg in row["segments"]:
                assert 0 <= seg["start"] <= row["duration"]
                assert seg["start"] <= seg["end"] <= row["duration"]
                for w in seg.get("words") or []:
                    assert 0 <= w["start"] <= row["duration"]
                    assert w["start"] <= w["end"] <= row["duration"]
            # Top-level text is recomputed from snippet segments
            assert "text" in row
            # Concatenation should be non-empty (no_text was already filtered)
            assert row["text"].strip()

    @staticmethod
    def _assert_metrics_summary_schema(
        summary: dict, expected_input_audios: int, expected_output_snippets: int
    ) -> None:
        assert summary["num_input_audios"] == expected_input_audios
        assert summary["num_output_snippets"] == expected_output_snippets
        assert isinstance(summary["input_total_segments"], int)
        assert isinstance(summary["input_total_duration_sec"], (int, float))
        assert isinstance(summary["output_total_segments"], int)
        assert isinstance(summary["output_total_duration_sec"], (int, float))
        assert set(summary["dropped"]) == {"empty", "overlap", "too_long", "too_short", "no_text"}
        for v in summary["dropped"].values():
            assert isinstance(v, int)
            assert v >= 0
        # Histogram: keys are 30s-wide bin labels, values are non-negative ints.
        for label, count in summary["snippet_duration_histogram_30s"].items():
            assert "-" in label
            assert isinstance(count, int)
            assert count >= 0
        # per_original carries one entry per source audio that the aggregator saw.
        per_original = summary["per_original"]
        assert isinstance(per_original, list)
        assert len(per_original) == expected_input_audios
        for entry in per_original:
            assert {
                "id",
                "in_segments",
                "in_duration_sec",
                "dropped",
                "out_snippets",
                "out_segments",
                "out_duration_sec",
            } <= set(entry)
