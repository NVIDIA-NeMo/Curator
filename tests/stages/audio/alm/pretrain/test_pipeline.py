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

"""End-to-end pipeline tests.

Two scenarios:

* ``TestPipelineEndToEndOnLocalManifest`` drives every stage in
  sequence (no Ray) in dry-run mode against a committed 10-row sample
  of real production manifest data; the source audios don't need to
  exist on disk in dry-run.

* ``TestPipelineEndToEndTarMode`` synthesizes a tiny manifest backed by
  generated WAV files and runs the pipeline in real-extraction mode,
  verifying that the merged audio tar contains exactly the expected
  members (one per manifest snippet row) sorted lexicographically and
  that each member is decodable.

The fixture ``sample_long_form_manifest.jsonl`` next to this file is a
10-row excerpt of real production data, large enough to exercise the
algorithm against realistic segment counts and timestamp jitter without
checking in the multi-hundred-MB source manifest.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from tokenizers import Tokenizer, models, pre_tokenizers

from nemo_curator.stages.audio.alm.pretrain import (
    OverlapFilterStage,
    PretrainMetricsAggregatorStage,
    ReadLongFormManifestStage,
    SnippetCutPlannerStage,
    SnippetExtractionStage,
    SnippetManifestWriterStage,
    SnippetRepetitionFilterStage,
    build_audio_pretrain_pipeline,
    finalize_audio_pretrain_outputs,
    prepare_audio_pretrain_outputs,
)
from nemo_curator.tasks import EmptyTask

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


def _build_tiny_tokenizer_dir(tmp_dir: Path) -> Path:
    """Save a permissive WordLevel HF fast tokenizer to ``tmp_dir`` for tests.

    Vocab is unimportant for end-to-end tests because the manifest
    fixture has no obvious repetition; the OOV ``[UNK]`` mass tokenizes
    real words to a single id, but n-gram detection still works on
    those ids without producing false positives at default thresholds.
    """
    vocab = {"[UNK]": 0}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))  # noqa: S106  -- tokenizer special token, not a credential
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok_dir = tmp_dir / "tok"
    tok_dir.mkdir()
    tok.save(str(tok_dir / "tokenizer.json"))
    (tok_dir / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", "model_max_length": 4096}),
        encoding="utf-8",
    )
    return tok_dir


# Stage classes the factory must wire up, in order, for the inline driver
# below to make sense.  Asserting on this list also doubles as a unit test
# against `build_audio_pretrain_pipeline`'s structure.
_EXPECTED_STAGE_TYPES = (
    ReadLongFormManifestStage,
    OverlapFilterStage,
    SnippetCutPlannerStage,
    SnippetRepetitionFilterStage,
    SnippetExtractionStage,
    SnippetManifestWriterStage,
    PretrainMetricsAggregatorStage,
)


def _run_pipeline_inline(  # noqa: PLR0913
    *,
    input_manifest: Path,
    audio_dir: Path,
    output_dir: Path,
    output_manifest: Path,
    output_audio_tar_path: Path,
    metrics_path: Path,
    max_duration_sec: float,
    target_sample_rate: int,
    tokenizer_path: Path,
    dry_run: bool = True,
) -> None:
    """Drive the factory-built pipeline stage-by-stage in-process (no Ray).

    Builds the pipeline via :func:`build_audio_pretrain_pipeline` so the
    factory's stage list / ordering is exercised end-to-end, then walks
    ``pipeline.stages`` and dispatches the data flow inline.  Skips the
    real executor since these tests don't need (or want) Ray.
    """
    pipeline = build_audio_pretrain_pipeline(
        input_manifest=str(input_manifest),
        audio_dir=str(audio_dir),
        output_dir=str(output_dir),
        output_manifest_path=str(output_manifest),
        output_audio_tar_path=str(output_audio_tar_path),
        metrics_path=str(metrics_path),
        max_duration_sec=max_duration_sec,
        tokenizer_path=str(tokenizer_path),
        target_sample_rate=target_sample_rate,
        output_format="flac",
        dry_run=dry_run,
    )
    actual_types = tuple(type(s) for s in pipeline.stages)
    assert actual_types == _EXPECTED_STAGE_TYPES, (
        f"build_audio_pretrain_pipeline returned unexpected stage list: {actual_types}"
    )
    reader, overlap, planner, rep_filter, extractor, writer, aggregator = pipeline.stages

    prepare_audio_pretrain_outputs(str(output_manifest), str(metrics_path), str(output_audio_tar_path))

    # Drive the per-stage lifecycle hooks Ray would normally invoke.
    rep_filter.setup_on_node()
    rep_filter.setup()
    extractor.setup_on_node()
    extractor.setup()
    writer.setup_on_node()
    writer.setup()
    aggregator.setup_on_node()
    aggregator.setup()

    for input_task in reader.process(EmptyTask(dataset_name="e", data=None)):
        filtered = overlap.process(input_task)
        planned = planner.process(filtered)
        rep_filtered = rep_filter.process(planned)
        for snippet_task in extractor.process(rep_filtered):
            writer.process(snippet_task)
            aggregator.process(snippet_task)
    extractor.teardown()

    finalize_audio_pretrain_outputs(str(output_manifest), str(metrics_path), str(output_audio_tar_path))


class TestPipelineEndToEndOnLocalManifest:
    """Drive the pipeline on the first 10 rows of the local ``test.jsonl``.

    Asserts on the resulting manifest + metrics summary -- not on
    specific row counts (those depend on the user's data) but on the
    structural invariants the spec requires.
    """

    def test_dry_run_produces_valid_outputs(self, tmp_path: Path) -> None:
        pytest.importorskip("transformers")
        pytest.importorskip("tokenizers")
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
        output_audio_tar = tmp_path / "snippets.tar"
        metrics_path = tmp_path / "metrics.json"
        tokenizer_dir = _build_tiny_tokenizer_dir(tmp_path)

        _run_pipeline_inline(
            input_manifest=in_manifest,
            audio_dir=audio_dir,
            output_dir=output_dir,
            output_manifest=output_manifest,
            output_audio_tar_path=output_audio_tar,
            metrics_path=metrics_path,
            max_duration_sec=_MAX_DURATION_SEC,
            target_sample_rate=_TARGET_SR,
            tokenizer_path=tokenizer_dir,
        )

        # Output files exist; tar file is NOT produced in dry-run.
        assert output_manifest.exists()
        assert metrics_path.exists()
        assert not output_audio_tar.exists()

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
            assert row["snippet_id"].startswith(row["id"] + "-")
            assert row["audio_filepath"].endswith(".flac")
            # In tar mode `audio_filepath` is the tar-internal basename only.
            assert "/" not in row["audio_filepath"]
            assert row["audio_filepath"] == f"{row['snippet_id']}.flac"
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
        assert set(summary["dropped"]) == {"empty", "overlap", "too_long", "too_short", "no_text", "repetition"}
        for v in summary["dropped"].values():
            assert isinstance(v, int)
            assert v >= 0
        # Filtered-snippet example texts are always surfaced (possibly empty).
        examples = summary["dropped_repetition_examples"]
        assert isinstance(examples, list)
        for ex in examples:
            assert isinstance(ex, str)
        # The cap matches the count of repetition-filtered snippets.
        assert len(examples) <= summary["dropped"]["repetition"]
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


class TestPipelineEndToEndTarMode:
    """End-to-end real-extraction test with the audio tar output enabled.

    Synthesizes two short WAV source audios with a hand-rolled segment
    list, runs the full pipeline in non-dry-run mode, and verifies that
    the final merged tar at ``output_audio_tar_path`` contains exactly
    the snippets the manifest references, sorted lexicographically, and
    that every member decodes back to the expected sample rate +
    duration.
    """

    @staticmethod
    def _make_wav(path: Path, duration_sec: float, sample_rate: int) -> None:
        n = int(duration_sec * sample_rate)
        t = np.linspace(0, duration_sec, n, endpoint=False, dtype=np.float32)
        mono = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        sf.write(str(path), mono, sample_rate, subtype="PCM_16")

    @staticmethod
    def _seg(start: float, end: float, text: str) -> dict:
        return {"speaker": "A", "start": start, "end": end, "text": text, "text_ITN": text, "words": []}

    def test_full_pipeline_writes_merged_tar_with_expected_members(self, tmp_path: Path) -> None:
        pytest.importorskip("transformers")
        pytest.importorskip("tokenizers")
        pytest.importorskip("torchaudio")

        audio_dir = tmp_path / "audios"
        audio_dir.mkdir()
        # Two source files, 12s each. Each yields 2 snippets at max-duration 5s
        # (planner packs greedy, snippet boundaries fall on segment boundaries).
        for src_id in ("A", "B"):
            self._make_wav(audio_dir / f"{src_id}.wav", duration_sec=12.0, sample_rate=16000)

        rows = []
        for src_id in ("A", "B"):
            rows.append(
                {
                    "id": src_id,
                    "audio_filepath": str(audio_dir / f"{src_id}.wav"),
                    # 4 segments of ~3s each, no overlaps, < 30s default gap
                    "segments": [
                        self._seg(0.0, 3.0, f"{src_id} hello"),
                        self._seg(3.0, 6.0, f"{src_id} world"),
                        self._seg(6.0, 9.0, f"{src_id} foo"),
                        self._seg(9.0, 12.0, f"{src_id} bar"),
                    ],
                }
            )

        in_manifest = tmp_path / "in.jsonl"
        with in_manifest.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        output_dir = tmp_path / "snippets"
        output_manifest = tmp_path / "snippets.jsonl"
        output_audio_tar = tmp_path / "snippets.tar"
        metrics_path = tmp_path / "metrics.json"
        tokenizer_dir = _build_tiny_tokenizer_dir(tmp_path)

        _run_pipeline_inline(
            input_manifest=in_manifest,
            audio_dir=audio_dir,
            output_dir=output_dir,
            output_manifest=output_manifest,
            output_audio_tar_path=output_audio_tar,
            metrics_path=metrics_path,
            max_duration_sec=5.0,
            target_sample_rate=_TARGET_SR,
            tokenizer_path=tokenizer_dir,
            dry_run=False,
        )

        # Final tar exists; shards cleaned up
        assert output_audio_tar.exists()
        assert sorted(tmp_path.glob("snippets.tar.shard-*.tar")) == []

        # Manifest's audio_filepath set == tar member set
        with output_manifest.open(encoding="utf-8") as f:
            snippet_rows = [json.loads(line) for line in f if line.strip()]
        assert snippet_rows, "expected at least one snippet emitted"
        manifest_members = {row["audio_filepath"] for row in snippet_rows}

        with tarfile.open(str(output_audio_tar), "r") as t:
            tar_member_names = t.getnames()
            # Members are sorted lexicographically by the merger
            assert tar_member_names == sorted(tar_member_names)
            assert set(tar_member_names) == manifest_members
            # Every member is a readable audio file at the target SR
            for ti in t.getmembers():
                assert ti.isreg()
                f = t.extractfile(ti)
                assert f is not None
                data, sr = sf.read(io.BytesIO(f.read()), always_2d=True)
                assert sr == _TARGET_SR
                assert data.shape[1] == 1
                assert data.shape[0] > 0
