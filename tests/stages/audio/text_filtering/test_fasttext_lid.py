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

from unittest.mock import MagicMock, patch

import pytest

from nemo_curator.stages.audio.text_filtering.fasttext_lid import FastTextLIDStage
from nemo_curator.tasks import AudioTask

_TEXT_KEY = "cleaned_text"
_SKIP_KEY = "_skip_me"
_LANG_KEY = "source_lang"


def _make_stage(label: str, prob: float, **kwargs: object) -> FastTextLIDStage:
    """Create a stage with a mocked fasttext model that returns the given label/prob."""
    stage = FastTextLIDStage(
        model_path="/fake/model.bin", text_key=_TEXT_KEY, skip_me_key=_SKIP_KEY,
        source_lang_key=_LANG_KEY, **kwargs,
    )
    mock_score = MagicMock()
    mock_score.item.return_value = prob
    mock_model = MagicMock()
    mock_model.predict.return_value = ([[f"__label__{label.lower()}"]], [[mock_score]])
    stage._model = mock_model
    return stage


def test_correct_lang_and_high_prob_passes() -> None:
    stage = _make_stage("EN", 0.95)
    task = AudioTask(data={_TEXT_KEY: "hello world", _SKIP_KEY: "", _LANG_KEY: "en"})
    result = stage.process(task)
    assert result.data[_SKIP_KEY] == ""


def test_wrong_lang_sets_skip_me() -> None:
    stage = _make_stage("FR", 0.95)
    task = AudioTask(data={_TEXT_KEY: "bonjour monde", _SKIP_KEY: "", _LANG_KEY: "en"})
    result = stage.process(task)
    assert result.data[_SKIP_KEY].startswith("Wrong language")


def test_low_prob_sets_skip_me() -> None:
    stage = _make_stage("EN", 0.5)
    task = AudioTask(data={_TEXT_KEY: "hello world", _SKIP_KEY: "", _LANG_KEY: "en"})
    result = stage.process(task)
    assert result.data[_SKIP_KEY].startswith("Low probability of language")


def test_correct_lang_exactly_at_threshold_passes() -> None:
    stage = _make_stage("EN", 0.8)
    task = AudioTask(data={_TEXT_KEY: "hello world", _SKIP_KEY: "", _LANG_KEY: "en"})
    result = stage.process(task)
    assert result.data[_SKIP_KEY] == ""


def test_default_min_lang_prob_is_80_percent() -> None:
    stage = FastTextLIDStage(model_path="/fake/model.bin")
    assert stage.min_lang_prob == 0.8


def test_empty_text_sets_skip_me_without_calling_model() -> None:
    stage = FastTextLIDStage(
        model_path="/fake/model.bin", text_key=_TEXT_KEY, skip_me_key=_SKIP_KEY,
    )
    mock_model = MagicMock()
    stage._model = mock_model
    task = AudioTask(data={_TEXT_KEY: "   ", _SKIP_KEY: ""})
    result = stage.process(task)
    assert result.data[_SKIP_KEY].startswith("Empty text")
    mock_model.predict.assert_not_called()


def test_preserves_existing_skip_me_reason() -> None:
    stage = _make_stage("FR", 0.95)
    task = AudioTask(data={_TEXT_KEY: "bonjour monde", _SKIP_KEY: "Hallucination", _LANG_KEY: "en"})
    result = stage.process(task)
    assert result.data[_SKIP_KEY] == "Hallucination"


def test_preserves_existing_skip_me_on_empty_text() -> None:
    stage = FastTextLIDStage(
        model_path="/fake/model.bin", text_key=_TEXT_KEY, skip_me_key=_SKIP_KEY,
    )
    mock_model = MagicMock()
    stage._model = mock_model
    task = AudioTask(data={_TEXT_KEY: "   ", _SKIP_KEY: "Hallucination"})
    result = stage.process(task)
    assert result.data[_SKIP_KEY] == "Hallucination"


def test_invalid_model_path_raises() -> None:
    stage = FastTextLIDStage(model_path="nonexistent.bin")
    with pytest.raises(ValueError, match="not a valid file path"):
        stage._resolve_model_path()


def test_non_string_text_returns_task_unchanged() -> None:
    stage = _make_stage("EN", 0.95)
    task = AudioTask(data={_TEXT_KEY: None, _SKIP_KEY: ""})
    result = stage.process(task)
    assert result.data[_SKIP_KEY] == ""


def test_requires_model_path() -> None:
    with pytest.raises(ValueError, match="model_path is required"):
        FastTextLIDStage(model_path="")


def test_known_model_name_checks_cache(tmp_path: object) -> None:
    stage = FastTextLIDStage(model_path="lid.176.ftz")
    with (
        patch("nemo_curator.stages.audio.text_filtering.fasttext_lid._DEFAULT_CACHE_DIR", str(tmp_path)),
        patch("urllib.request.urlretrieve") as mock_dl,
    ):
        mock_dl.side_effect = lambda _url, path: open(path, "w").close()
        resolved = stage._resolve_model_path()
    assert resolved.endswith("lid.176.ftz")


def test_wrong_lang_takes_precedence_over_low_prob() -> None:
    stage = _make_stage("FR", 0.1)
    task = AudioTask(data={_TEXT_KEY: "bonjour monde", _SKIP_KEY: "", _LANG_KEY: "en"})
    result = stage.process(task)
    assert result.data[_SKIP_KEY].startswith("Wrong language")
