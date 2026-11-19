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

from typing import Any, ClassVar
from unittest.mock import patch

import pytest

from nemo_curator.models.asr.base import ASRAdapter, ASRResult
from nemo_curator.models.audio.indic_conformer_per_language import IndicConformerPerLanguageASR

_TEMPLATE = "ai4bharat/indicconformer_stt_{lang}_hybrid_ctc_rnnt_large"


class _FakeHybrid:
    """Stands in for IndicConformerHybridASR and records what it was asked to do."""

    created: ClassVar[list["_FakeHybrid"]] = []

    def __init__(self, model_id: str, **kwargs: Any):  # noqa: ANN401
        self.model_id = model_id
        self.kwargs = kwargs
        self.loaded_with: int | None = None
        self.unloaded = False
        self.downloaded = False
        self.batches: list[list[dict[str, Any]]] = []
        _FakeHybrid.created.append(self)

    def download_weights_on_node(self) -> None:
        self.downloaded = True

    def load_model(self, *, num_gpus: int) -> None:
        self.loaded_with = num_gpus

    def unload_model(self) -> None:
        self.unloaded = True

    def transcribe_batch(self, items: list[dict[str, Any]]) -> list[ASRResult]:
        self.batches.append(items)
        return [ASRResult(text=f"{self.model_id}:{item['task_id']}") for item in items]


@pytest.fixture(autouse=True)
def fake_hybrid():
    _FakeHybrid.created = []
    with patch("nemo_curator.models.audio.indic_conformer_per_language.IndicConformerHybridASR", _FakeHybrid):
        yield


def _items(*pairs: tuple[str, str]) -> list[dict[str, Any]]:
    return [{"task_id": task_id, "language_code": language} for task_id, language in pairs]


def test_is_an_asr_adapter() -> None:
    assert isinstance(IndicConformerPerLanguageASR(_TEMPLATE), ASRAdapter)


@pytest.mark.parametrize("model_id", ["ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large", "/models/ta.nemo"])
def test_model_id_needs_a_language_placeholder(model_id: str) -> None:
    with pytest.raises(ValueError, match=r"\{lang\}"):
        IndicConformerPerLanguageASR(model_id)


def test_rejects_unknown_languages_and_bad_limit() -> None:
    with pytest.raises(ValueError, match="Unsupported IndicConformer languages"):
        IndicConformerPerLanguageASR(_TEMPLATE, languages=["xx"])
    with pytest.raises(ValueError, match="max_loaded_languages"):
        IndicConformerPerLanguageASR(_TEMPLATE, max_loaded_languages=0)


def test_rows_go_to_their_language_checkpoint_in_order() -> None:
    adapter = IndicConformerPerLanguageASR(_TEMPLATE, decode_mode="rnnt")
    adapter.load_model(num_gpus=1)

    results = adapter.transcribe_batch(_items(("a", "ta"), ("b", "bn"), ("c", "ta"), ("d", "bn")))

    assert [r.text.split("_")[2] + ":" + r.text.rsplit(":", 1)[1] for r in results] == ["ta:a", "bn:b", "ta:c", "bn:d"]
    assert {a.model_id for a in _FakeHybrid.created} == {_TEMPLATE.format(lang="ta"), _TEMPLATE.format(lang="bn")}
    assert all(a.loaded_with == 1 and a.kwargs == {"decode_mode": "rnnt"} for a in _FakeHybrid.created)
    assert [len(a.batches[0]) for a in _FakeHybrid.created] == [2, 2]


def test_each_checkpoint_loads_once() -> None:
    adapter = IndicConformerPerLanguageASR(_TEMPLATE)
    adapter.load_model(num_gpus=1)

    adapter.transcribe_batch(_items(("a", "ta")))
    adapter.transcribe_batch(_items(("b", "ta"), ("c", "te")))

    assert [a.model_id for a in _FakeHybrid.created] == [_TEMPLATE.format(lang="ta"), _TEMPLATE.format(lang="te")]
    assert len(_FakeHybrid.created[0].batches) == 2


def test_unsupported_or_missing_language_is_not_sent_to_a_checkpoint() -> None:
    adapter = IndicConformerPerLanguageASR(_TEMPLATE)
    adapter.load_model(num_gpus=1)

    results = adapter.transcribe_batch(_items(("a", "en"), ("b", ""), ("c", "ta")))

    assert [r.unsupported_language for r in results] == ["en", None, None]
    assert results[0].text == results[1].text == ""
    assert results[2].text.endswith(":c")
    assert len(_FakeHybrid.created) == 1


def test_least_recently_used_checkpoint_is_unloaded_over_the_limit() -> None:
    adapter = IndicConformerPerLanguageASR(_TEMPLATE, max_loaded_languages=2)
    adapter.load_model(num_gpus=1)

    adapter.transcribe_batch(_items(("a", "ta")))
    adapter.transcribe_batch(_items(("b", "te")))
    adapter.transcribe_batch(_items(("c", "ta")))  # ta becomes the most recent
    adapter.transcribe_batch(_items(("d", "bn")))  # evicts te

    by_language = {a.model_id: a for a in _FakeHybrid.created}
    assert by_language[_TEMPLATE.format(lang="te")].unloaded
    assert not by_language[_TEMPLATE.format(lang="ta")].unloaded


def test_unload_model_unloads_everything_loaded() -> None:
    adapter = IndicConformerPerLanguageASR(_TEMPLATE)
    adapter.load_model(num_gpus=1)
    adapter.transcribe_batch(_items(("a", "ta"), ("b", "te")))

    adapter.unload_model()

    assert all(a.unloaded for a in _FakeHybrid.created)


def test_transcribing_before_load_model_fails() -> None:
    with pytest.raises(RuntimeError, match="load_model"):
        IndicConformerPerLanguageASR(_TEMPLATE).transcribe_batch(_items(("a", "ta")))


def test_download_prefetches_only_the_configured_languages() -> None:
    IndicConformerPerLanguageASR(_TEMPLATE, languages=["ta", "te"]).download_weights_on_node()

    assert sorted(a.model_id for a in _FakeHybrid.created) == [
        _TEMPLATE.format(lang="ta"),
        _TEMPLATE.format(lang="te"),
    ]
    assert all(a.downloaded for a in _FakeHybrid.created)
