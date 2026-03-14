# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_curator.stages.audio.io.convert import AudioToDocumentStage
from nemo_curator.tasks import AudioEntry, DocumentBatch


def test_audio_to_document_stage_converts_entry() -> None:
    entry = AudioEntry(
        task_id="t1",
        dataset_name="ds",
        data={"audio_filepath": "/a.wav", "text": "hello"},
    )

    stage = AudioToDocumentStage()
    doc = stage.process(entry)

    assert isinstance(doc, DocumentBatch)
    assert isinstance(doc.data, pd.DataFrame)
    assert list(doc.data.columns) == ["audio_filepath", "text"]
    assert len(doc.data) == 1
    assert doc.task_id == entry.task_id
    assert doc.dataset_name == entry.dataset_name
