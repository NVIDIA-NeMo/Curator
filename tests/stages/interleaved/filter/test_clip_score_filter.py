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

import pytest

from nemo_curator.stages.interleaved.filter.clip_score_filter import InterleavedCLIPScoreFilterStage

from .conftest import interleaved_task


def test_clip_score_filter_requires_model_dir() -> None:
    # Stage only calls _ensure_model() when there is at least one image row.
    rows = [
        {
            "sample_id": "s1",
            "position": 0,
            "modality": "image",
            "content_type": "image/jpeg",
            "text_content": None,
            "binary_content": b"fake",
            "source_ref": None,
            "materialize_error": None,
        },
    ]
    task = interleaved_task(rows)
    stage = InterleavedCLIPScoreFilterStage(model_dir=None)
    with pytest.raises(RuntimeError, match="model_dir"):
        stage.process(task)
