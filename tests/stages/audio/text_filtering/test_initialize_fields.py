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

from nemo_curator.stages.audio.text_filtering.initialize_fields import InitializeFieldsStage
from nemo_curator.tasks import AudioTask


def test_preserves_v1_state_and_initializes_granary_fields() -> None:
    stage = InitializeFieldsStage(pipeline_notes={"primary_model": "qwen"})
    task = AudioTask(data={"text": "original", "_skipme": "v1 reason", "shard_id": "3"})

    result = stage.process(task)

    assert result.data["granary_v1_prediction"] == "original"
    assert result.data["_skipme"] == ""
    assert result.data["source_lang"] == "en"
    assert result.data["shard_id"] == 3
    assert result.data["additional_notes"] == {
        "v1_skipme": "v1 reason",
        "primary_model": "qwen",
    }


def test_drops_stale_fields_without_touching_pipeline_data() -> None:
    stage = InitializeFieldsStage()
    task = AudioTask(
        data={
            "text": "original",
            "answer": "stale",
            "target_lang": "de",
            "pred_text": "prediction",
            "duration": 3.5,
        }
    )

    result = stage.process(task)

    assert "answer" not in result.data
    assert "target_lang" not in result.data
    assert result.data["pred_text"] == "prediction"
    assert result.data["duration"] == 3.5


def test_custom_keys_and_batch_processing() -> None:
    stage = InitializeFieldsStage(
        original_text_key="source",
        granary_v1_key="v1",
        skip_me_key="skip",
        source_lang_key="lang",
        default_source_lang="fr",
        drop_keys=["remove"],
    )
    tasks = [
        AudioTask(data={"source": "one", "remove": True}),
        AudioTask(data={"source": "two", "lang": "de"}),
    ]

    results = stage.process_batch(tasks)

    assert [task.data["v1"] for task in results] == ["one", "two"]
    assert [task.data["lang"] for task in results] == ["fr", "de"]
    assert all(task.data["skip"] == "" for task in results)
    assert "remove" not in results[0].data


def test_non_numeric_shard_id_is_preserved() -> None:
    task = AudioTask(data={"shard_id": "corpus-a/0001"})

    InitializeFieldsStage().process(task)

    assert task.data["shard_id"] == "corpus-a/0001"
