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

import pandas as pd
import pytest

from nemo_curator.stages.audio.io.convert import AudioToDocumentStage
from nemo_curator.tasks import AudioTask, DocumentBatch


def test_audio_to_document_stage_process_raises() -> None:
    entry = AudioTask(
        task_id="t1",
        dataset_name="ds",
        data={"audio_filepath": "/a.wav", "text": "hello"},
    )

    stage = AudioToDocumentStage()
    with pytest.raises(NotImplementedError, match="only supports process_batch"):
        stage.process(entry)


def test_process_batch_aggregates_into_single_dataframe() -> None:
    tasks = [
        AudioTask(task_id=f"t{i}", dataset_name="ds", data={"audio_filepath": f"/{i}.wav", "text": f"text{i}"})
        for i in range(5)
    ]

    stage = AudioToDocumentStage()
    result = stage.process_batch(tasks)

    assert len(result) == 1
    doc = result[0]
    assert isinstance(doc, DocumentBatch)
    assert isinstance(doc.data, pd.DataFrame)
    assert len(doc.data) == 5
    assert list(doc.data["audio_filepath"]) == ["/0.wav", "/1.wav", "/2.wav", "/3.wav", "/4.wav"]
    assert list(doc.data["text"]) == ["text0", "text1", "text2", "text3", "text4"]
    assert doc.task_id == "t0,t1,t2,t3,t4"
    assert doc.dataset_name == "ds"


def test_process_batch_empty() -> None:
    stage = AudioToDocumentStage()
    result = stage.process_batch([])
    assert result == []


def test_process_batch_preserves_stage_perf() -> None:
    tasks = [
        AudioTask(task_id="t1", dataset_name="ds", data={"audio_filepath": "/a.wav"}, _stage_perf=["perf1"]),
        AudioTask(task_id="t2", dataset_name="ds", data={"audio_filepath": "/b.wav"}, _stage_perf=["perf2"]),
    ]
    stage = AudioToDocumentStage()
    result = stage.process_batch(tasks)
    assert result[0]._stage_perf == ["perf1", "perf2"]


def test_process_batch_deduplicates_dataset_names() -> None:
    tasks = [
        AudioTask(task_id="t1", dataset_name="ds_a", data={"audio_filepath": "/a.wav"}),
        AudioTask(task_id="t2", dataset_name="ds_b", data={"audio_filepath": "/b.wav"}),
        AudioTask(task_id="t3", dataset_name="ds_a", data={"audio_filepath": "/c.wav"}),
    ]
    stage = AudioToDocumentStage()
    result = stage.process_batch(tasks)
    assert result[0].dataset_name == "ds_a,ds_b"


def test_process_batch_single_task() -> None:
    task = AudioTask(task_id="only", dataset_name="ds", data={"audio_filepath": "/x.wav", "text": "hi"})
    stage = AudioToDocumentStage()
    result = stage.process_batch([task])
    assert len(result) == 1
    assert len(result[0].data) == 1
    assert result[0].data.iloc[0]["text"] == "hi"


def test_process_batch_serializes_constructor_sample_key() -> None:
    task = AudioTask(
        task_id="only",
        dataset_name="ds",
        data={"audio_filepath": "/x.wav", "text": "hi"},
        sample_key="sample-serialized",
    )
    stage = AudioToDocumentStage()
    result = stage.process_batch([task])

    assert result[0].data.iloc[0]["sample_key"] == "sample-serialized"
