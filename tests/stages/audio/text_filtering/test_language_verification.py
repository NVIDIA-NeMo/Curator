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


# ruff: noqa: INP001

from nemo_curator.stages.audio.text_filtering import LanguageVerificationStage
from nemo_curator.tasks import AudioTask


def _run(prediction: str, **extra: object) -> dict:
    task = AudioTask(data={"llm_language_prediction": prediction, "source_lang": "de", **extra}, task_id="t")
    return LanguageVerificationStage().process(task).data


def test_matching_language_passes() -> None:
    data = _run("German-de")
    assert not data.get("_skipme")
    assert data["additional_notes"]["LLMLanguageVerification"].startswith("passed")


def test_wrong_language_is_flagged() -> None:
    assert _run("French-fr")["_skipme"].startswith("Wrong language")


def test_existing_skip_flag_is_kept() -> None:
    assert _run("French-fr", _skipme="Hallucination")["_skipme"] == "Hallucination"
