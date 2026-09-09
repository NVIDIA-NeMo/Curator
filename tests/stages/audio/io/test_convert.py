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

import json

import pandas as pd
import pytest
import torch

from nemo_curator.stages.audio._agent._agent_registry import build_contract
from nemo_curator.stages.audio._agent._planning import validate_pipeline
from nemo_curator.stages.audio.io.convert import AudioToDocumentStage
from nemo_curator.tasks import AudioTask, DocumentBatch


def test_audio_to_document_stage_process_raises() -> None:
    entry = AudioTask(
        dataset_name="ds",
        data={"audio_filepath": "/a.wav", "text": "hello"},
    )

    stage = AudioToDocumentStage()
    with pytest.raises(NotImplementedError, match="only supports process_batch"):
        stage.process(entry)


def test_process_batch_aggregates_into_single_dataframe() -> None:
    tasks = [AudioTask(dataset_name="ds", data={"audio_filepath": f"/{i}.wav", "text": f"text{i}"}) for i in range(5)]

    stage = AudioToDocumentStage()
    result = stage.process_batch(tasks)

    assert len(result) == 1
    doc = result[0]
    assert isinstance(doc, DocumentBatch)
    assert isinstance(doc.data, pd.DataFrame)
    assert len(doc.data) == 5
    assert list(doc.data["audio_filepath"]) == ["/0.wav", "/1.wav", "/2.wav", "/3.wav", "/4.wav"]
    assert list(doc.data["text"]) == ["text0", "text1", "text2", "text3", "text4"]
    assert doc.dataset_name == "ds"


def test_process_batch_empty() -> None:
    stage = AudioToDocumentStage()
    result = stage.process_batch([])
    assert result == []


def test_process_batch_preserves_stage_perf() -> None:
    tasks = [
        AudioTask(dataset_name="ds", data={"audio_filepath": "/a.wav"}, _stage_perf=["perf1"]),
        AudioTask(dataset_name="ds", data={"audio_filepath": "/b.wav"}, _stage_perf=["perf2"]),
    ]
    stage = AudioToDocumentStage()
    result = stage.process_batch(tasks)
    assert result[0]._stage_perf == ["perf1", "perf2"]


def test_process_batch_deduplicates_dataset_names() -> None:
    tasks = [
        AudioTask(dataset_name="ds_a", data={"audio_filepath": "/a.wav"}),
        AudioTask(dataset_name="ds_b", data={"audio_filepath": "/b.wav"}),
        AudioTask(dataset_name="ds_a", data={"audio_filepath": "/c.wav"}),
    ]
    stage = AudioToDocumentStage()
    result = stage.process_batch(tasks)
    assert result[0].dataset_name == "ds_a,ds_b"


def test_process_batch_single_task() -> None:
    task = AudioTask(dataset_name="ds", data={"audio_filepath": "/x.wav", "text": "hi"})
    stage = AudioToDocumentStage()
    result = stage.process_batch([task])
    assert len(result) == 1
    assert len(result[0].data) == 1
    assert result[0].data.iloc[0]["text"] == "hi"


class TestAudioToDocumentSerializationBoundary:
    """Nothing that cannot be JSON-encoded may cross into a DocumentBatch.

    Lifted from tests/stages/audio/test_agent_simulation_pipelines.py, which held the only
    coverage of this boundary. It drives one stage, so it belongs here.
    """

    def test_a_tensor_is_stripped_and_segments_are_opt_in(self) -> None:
        task = AudioTask(
            dataset_name="t",
            data={
                "audio_filepath": "src.wav",
                "text": "hello world",
                "waveform": torch.randn(1, 8000),
                "segments": [{"start": 0.0, "end": 0.5, "text": "hello"}],
            },
        )

        row = AudioToDocumentStage().process_batch([task])[0].to_pandas().iloc[0].to_dict()
        assert "waveform" not in row, "a tensor must never reach the document boundary"
        assert "segments" not in row, "segments are dropped unless asked for"
        assert row["text"] == "hello world"
        json.dumps(row)  # raises if anything non-serializable leaked

        kept = AudioToDocumentStage(serialize_segments=True).process_batch([task])[0]
        seg_row = kept.to_pandas().iloc[0].to_dict()
        assert "waveform" not in seg_row, "the tensor stays stripped even when segments are kept"
        assert seg_row["segments"][0]["text"] == "hello"

    def test_nested_tensor_is_removed_without_dropping_json_safe_field_names(self) -> None:
        task = AudioTask(
            dataset_name="t",
            data={
                "audio_filepath": "src.wav",
                "custom": {
                    "audio": "a JSON-safe description",
                    "segments": [{"text": "nested metadata"}],
                    "embedding": torch.zeros(2),
                },
            },
        )

        row = AudioToDocumentStage().process_batch([task])[0].to_pandas().iloc[0].to_dict()

        assert row["custom"] == {
            "audio": "a JSON-safe description",
            "segments": [{"text": "nested metadata"}],
        }
        json.dumps(row)

    def test_non_json_value_in_a_tuple_is_named_and_removed(self, caplog: pytest.LogCaptureFixture) -> None:
        task = AudioTask(
            dataset_name="t",
            data={"audio_filepath": "src.wav", "custom": ("kept", object())},
        )

        with caplog.at_level("WARNING"):
            row = AudioToDocumentStage().process_batch([task])[0].to_pandas().iloc[0].to_dict()

        assert row["custom"] == ["kept"]
        assert "custom[1]" in caplog.text
        json.dumps(row)


def test_configured_projection_is_visible_to_planning_and_runtime() -> None:
    stage = AudioToDocumentStage(keep_keys=["audio_filepath"])
    contract = build_contract(stage)

    assert contract.preserves_upstream_keys is False
    assert contract.reads.data_keys == ["audio_filepath"]
    assert contract.writes.data_keys == ["audio_filepath"]

    report = validate_pipeline(
        [stage],
        initial_keys={"audio_filepath", "text"},
        initial_roles={"audio_filepath", "transcript"},
    )
    assert "text" not in report.produced_keys

    batch = stage.process_batch(
        [AudioTask(dataset_name="d", data={"audio_filepath": "/a.wav", "text": "hi"})]
    )[0]
    assert list(batch.to_pandas().columns) == ["audio_filepath"]


def test_drop_keys_override_keep_keys_in_contract_and_runtime() -> None:
    stage = AudioToDocumentStage(
        keep_keys=["audio_filepath", "text"],
        drop_keys=("text",),
    )

    contract = stage.describe()
    batch = stage.process_batch(
        [AudioTask(dataset_name="d", data={"audio_filepath": "/a.wav", "text": "hi"})]
    )[0]

    assert contract.writes.data_keys == ["audio_filepath"]
    assert list(batch.to_pandas().columns) == ["audio_filepath"]


def test_projection_that_keeps_nothing_is_rejected() -> None:
    stage = AudioToDocumentStage(keep_keys=[])

    with pytest.raises(ValueError, match="kept no columns"):
        stage.process_batch(
            [
                AudioTask(dataset_name="d", data={"audio_filepath": "/a.wav"}),
                AudioTask(dataset_name="d", data={"audio_filepath": "/b.wav"}),
            ]
        )
