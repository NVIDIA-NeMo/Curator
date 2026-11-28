# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_curator.models.audio.text_lid import INDICLID_LABEL_TO_LANGUAGE
from nemo_curator.stages.audio.text_filtering import TextLanguageIDStage
from nemo_curator.tasks import AudioTask

LID176_EXACT = ["as", "bn", "gu", "hi", "kn", "mai", "ml", "mr", "ne", "or", "pa", "sa", "sd", "ta", "te", "ur"]
LID176_LABELS = [*LID176_EXACT, "gom", "en"] + [f"x{i}" for i in range(176 - len(LID176_EXACT) - 2)]
FASTTEXT = "nemo_curator.models.audio.text_lid.FastTextLIDAdapter"
INDICLID = "nemo_curator.models.audio.text_lid.IndicLIDAdapter"


class FakeModel:
    """fastText-shaped model that predicts a fixed label for each input text."""

    def __init__(self, labels: list[str], predict: Callable[[str], str]) -> None:
        self.labels = labels
        self.predict_label = predict
        self.calls: list[list[str]] = []

    def get_labels(self) -> list[str]:
        return [f"__label__{label}" for label in self.labels]

    def predict(self, texts: list[str], k: int) -> tuple[list[list[str]], list[list[float]]]:
        assert k == 1
        self.calls.append(list(texts))
        return [[f"__label__{self.predict_label(text)}"] for text in texts], [[0.9] for _ in texts]


@pytest.fixture
def model_file(tmp_path: Path) -> str:
    path = tmp_path / "model.bin"
    path.write_bytes(b"")
    return str(path)


def _install(monkeypatch: pytest.MonkeyPatch, model: FakeModel) -> None:
    monkeypatch.setitem(sys.modules, "fasttext", SimpleNamespace(load_model=lambda _path: model))


def _row(text: str, lang: str, **extra: str) -> AudioTask:
    return AudioTask(data={"pnc_text": text, "source_lang": lang, **extra})


def _stage(model_file: str, adapter_target: str = FASTTEXT, **kwargs) -> TextLanguageIDStage:
    return TextLanguageIDStage(
        adapter_target=adapter_target, adapter_kwargs={"model_path": model_file}, text_key="pnc_text", **kwargs
    )


def test_fasttext_writes_the_code_and_verifies_it(monkeypatch: pytest.MonkeyPatch, model_file: str) -> None:
    model = FakeModel(LID176_LABELS, lambda _text: "hi")
    _install(monkeypatch, model)
    tasks = _stage(model_file).process_batch([_row(" एक  दो\nतीन ", "hi"), _row("एक दो", "mr")])

    assert model.calls == [["एक दो तीन", "एक दो"]]

    assert [task.data["llm_language_prediction"] for task in tasks] == ["hi", "hi"]
    assert tasks[0].data["additional_notes"] == {
        "LanguageID": "applied (modified)",
        "LLMLanguageVerification": "passed (Hindi-hi)",
    }
    assert not tasks[0].data.get("_skipme")
    assert tasks[1].data["_skipme"] == "Wrong language:LLMLanguageVerification"


def test_flagged_and_empty_rows_skip_the_model(monkeypatch: pytest.MonkeyPatch, model_file: str) -> None:
    model = FakeModel(LID176_LABELS, lambda _text: "hi")
    _install(monkeypatch, model)
    flagged, empty = _stage(model_file).process_batch(
        [_row("एक", "hi", _skipme="Hallucination:primary"), _row("  ", "hi")]
    )

    assert model.calls == []
    assert flagged.data["llm_language_prediction"] == ""
    assert flagged.data["_skipme"] == "Hallucination:primary"
    assert flagged.data["additional_notes"]["LLMLanguageVerification"] == "skipped (already flagged)"
    assert empty.data["additional_notes"]["LanguageID"] == "skipped (empty)"
    assert empty.data["additional_notes"]["LLMLanguageVerification"] == "skipped (empty prediction)"


def test_languages_limit_the_stage_to_its_rows(monkeypatch: pytest.MonkeyPatch, model_file: str) -> None:
    model = FakeModel(LID176_LABELS, lambda _text: "hi")
    _install(monkeypatch, model)
    listed, other = _stage(model_file, languages=["HI"]).process_batch([_row("एक", "hi"), _row("هڪ", "sd")])

    assert model.calls == [["एक"]]
    assert listed.data["llm_language_prediction"] == "hi"
    assert other.data == {"pnc_text": "هڪ", "source_lang": "sd"}


def test_an_empty_language_list_loads_no_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "fasttext", None)
    stage = _stage("/missing/lid.176.bin", languages=[])
    stage.setup()
    (task,) = stage.process_batch([_row("एक", "hi")])

    assert task.data == {"pnc_text": "एक", "source_lang": "hi"}


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"adapter_target": ""}, ValueError, "adapter_target is required"),
        ({"languages": "hi"}, TypeError, "list of language codes"),
        ({"languages": [""]}, ValueError, "non-empty string"),
    ],
)
def test_invalid_settings_fail_early(model_file: str, kwargs: dict, error: type[Exception], match: str) -> None:
    with pytest.raises(error, match=match):
        TextLanguageIDStage(**{"adapter_target": FASTTEXT, "adapter_kwargs": {"model_path": model_file}, **kwargs})


def test_fasttext_rejects_a_language_without_an_exact_label(monkeypatch: pytest.MonkeyPatch, model_file: str) -> None:
    _install(monkeypatch, FakeModel(LID176_LABELS, lambda _text: "gom"))

    with pytest.raises(ValueError, match="route this language to IndicLID"):
        _stage(model_file).process_batch([_row("text", "kok")])


def test_fasttext_checks_the_lid176_inventory(monkeypatch: pytest.MonkeyPatch, model_file: str) -> None:
    _install(monkeypatch, FakeModel(LID176_LABELS[:-1], lambda _text: "hi"))

    with pytest.raises(ValueError, match=r"Expected 176 lid\.176 labels"):
        _stage(model_file).setup()


def test_indiclid_maps_script_labels_to_language_codes(monkeypatch: pytest.MonkeyPatch, model_file: str) -> None:
    _install(monkeypatch, FakeModel(list(INDICLID_LABEL_TO_LANGUAGE), lambda _text: "mni_Meti"))
    (task,) = _stage(model_file, INDICLID).process_batch([_row("ꯑꯃ\r\nꯑꯅꯤ", "mni")])

    assert task.data["llm_language_prediction"] == "mni"
    assert task.data["additional_notes"]["LLMLanguageVerification"] == "passed (Manipuri-mni)"
    assert not task.data.get("_skipme")


def test_indiclid_other_label_is_unparseable(monkeypatch: pytest.MonkeyPatch, model_file: str) -> None:
    _install(monkeypatch, FakeModel(list(INDICLID_LABEL_TO_LANGUAGE), lambda _text: "other"))
    (task,) = _stage(model_file, INDICLID).process_batch([_row("text", "as")])

    assert task.data["_skipme"] == "Unparseable language:LLMLanguageVerification"


def test_indiclid_checks_its_label_inventory(monkeypatch: pytest.MonkeyPatch, model_file: str) -> None:
    _install(monkeypatch, FakeModel([*INDICLID_LABEL_TO_LANGUAGE, "extra_Latn"], lambda _text: "hin_Deva"))

    with pytest.raises(ValueError, match="extra=\\['extra_Latn'\\]"):
        _stage(model_file, INDICLID).setup()


def test_missing_model_file_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install(monkeypatch, FakeModel(LID176_LABELS, lambda _text: "hi"))

    with pytest.raises(ValueError, match="does not exist"):
        _stage(str(tmp_path / "missing.bin")).setup()


def test_missing_fasttext_package_names_the_install(monkeypatch: pytest.MonkeyPatch, model_file: str) -> None:
    monkeypatch.setitem(sys.modules, "fasttext", None)

    with pytest.raises(ImportError, match=r"fasttext==0\.9\.3"):
        _stage(model_file).setup()
