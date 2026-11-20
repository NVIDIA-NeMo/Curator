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

Covers the two per-replica writer stages (snippet manifest, per-task
metrics).  The input manifest is read by ``ManifestReader``; its long-form
options are tested in ``tests/stages/audio/common/test_manifest_reader.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nemo_curator.stages.audio._agent._agent_registry import build_contract, static_contract
from nemo_curator.stages.audio._agent._conformance import assert_agent_ready
from nemo_curator.stages.audio.alm.pretrain import (
    PretrainMetricsAggregatorStage,
    SnippetManifestWriterStage,
)
from nemo_curator.stages.audio.alm.pretrain.utils import _PRETRAIN_META_KEY
from nemo_curator.tasks import AudioTask


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


# ----------------------------------------------------------------------
# SnippetManifestWriterStage
# ----------------------------------------------------------------------


class TestSnippetManifestWriterStage:
    def test_new_key_parameter_preserves_legacy_positional_arguments(self, tmp_path: Path) -> None:
        stage = SnippetManifestWriterStage(str(tmp_path / "out.jsonl"), "legacy-name", 7)

        assert stage.name == "legacy-name"
        assert stage.batch_size == 7
        assert stage.snippet_id_key == "snippet_id"

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

    def test_custom_snippet_key_controls_stub_discrimination(self, tmp_path: Path) -> None:
        stage = SnippetManifestWriterStage(
            output_path=str(tmp_path / "out.jsonl"),
            snippet_id_key="clip_id",
        )
        stage.setup()
        stage.process(_make_audio_task({"id": "X", "clip_id": None}))
        shard = stage._shard_path
        assert shard is not None
        assert not Path(shard).exists()

        stage.process(_make_audio_task({"id": "X", "clip_id": "X-0-1"}))
        row = json.loads(Path(shard).read_text(encoding="utf-8"))
        assert row["clip_id"] == "X-0-1"

    def test_contract_validation_and_agent_ready(self, tmp_path: Path) -> None:
        stage = SnippetManifestWriterStage(
            output_path=str(tmp_path / "out.jsonl"),
            snippet_id_key="clip_id",
        )
        assert stage.validate_input(_make_audio_task({})) is True
        assert stage.validate_input(_make_audio_task({"clip_id": "X-0-1"})) is True
        assert stage.process_batch([_make_audio_task({})])[0].data == {}
        contract = assert_agent_ready(
            stage,
            lambda: _make_audio_task({"id": "X", "clip_id": "X-0-1", "duration": 1.0}),
            expected_cardinality="1:1",
            available_keys={"clip_id"},
            setup=True,
        )
        assert contract.reads.data_keys == ["clip_id"]
        assert contract.gates.per_row_independent is False
        assert static_contract(SnippetManifestWriterStage).gates.per_row_independent is False


# ----------------------------------------------------------------------
# PretrainMetricsAggregatorStage
# ----------------------------------------------------------------------


class TestPretrainMetricsAggregatorStage:
    def test_new_key_parameters_preserve_legacy_positional_arguments(self, tmp_path: Path) -> None:
        stage = PretrainMetricsAggregatorStage(str(tmp_path / "metrics.json"), "legacy-name", 7)

        assert stage.name == "legacy-name"
        assert stage.batch_size == 7
        assert stage.id_key == "id"
        assert stage.snippet_id_key == "snippet_id"
        assert stage.segments_key == "segments"
        assert stage.duration_key == "duration"

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

    def test_custom_input_keys_keep_canonical_metrics_schema(self, tmp_path: Path) -> None:
        stage = PretrainMetricsAggregatorStage(
            output_path=str(tmp_path / "metrics.json"),
            id_key="source_id",
            snippet_id_key="clip_id",
            segments_key="turns",
            duration_key="clip_duration",
        )
        stage.setup()
        task = _make_audio_task(
            {
                "source_id": "A",
                "clip_id": "A-0-5",
                "turns": [_ts(0, 5)],
                "clip_duration": 5.0,
            }
        )
        task._metadata[_PRETRAIN_META_KEY] = {}
        stage.process(task)

        shard = stage._shard_path
        assert shard is not None
        record = json.loads(Path(shard).read_text(encoding="utf-8"))
        assert record["id"] == "A"
        assert record["is_stub"] is False
        assert record["out_segments"] == 1
        assert record["out_duration_sec"] == pytest.approx(5.0)
        assert "source_id" not in record
        assert "clip_id" not in record

    def test_contract_validation_agent_ready_and_static_hints(self, tmp_path: Path) -> None:
        stage = PretrainMetricsAggregatorStage(
            output_path=str(tmp_path / "metrics.json"),
            id_key="source_id",
            snippet_id_key="clip_id",
            segments_key="turns",
            duration_key="clip_duration",
        )
        assert stage.validate_input(_make_audio_task({})) is True
        valid_data = {
            "source_id": "A",
            "clip_id": "A-0-5",
            "turns": [_ts(0, 5)],
            "clip_duration": 5.0,
        }
        assert stage.validate_input(_make_audio_task(valid_data)) is True
        assert stage.process_batch([_make_audio_task({})])[0].data == {}

        def fixture() -> AudioTask:
            task = _make_audio_task(dict(valid_data))
            task._metadata[_PRETRAIN_META_KEY] = {}
            return task

        contract = assert_agent_ready(
            stage,
            fixture,
            expected_cardinality="1:1",
            available_keys=set(valid_data),
            setup=True,
        )
        assert contract.reads.data_keys == []
        assert contract.optional_reads.data_keys == ["source_id", "clip_id", "turns", "clip_duration"]
        assert contract.metadata_reads == [_PRETRAIN_META_KEY]

        static = static_contract(PretrainMetricsAggregatorStage)
        configured = build_contract(stage)
        assert static.gates == configured.gates
