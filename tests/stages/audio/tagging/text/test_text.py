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

from collections.abc import Callable

import pytest

from nemo_curator.stages.audio.tagging.text.arabic_remove_diacritics import ArabicRemoveDiacriticsStage
from nemo_curator.stages.audio.tagging.text.chinese_conversion import ChineseConversionStage
from nemo_curator.stages.audio.tagging.text.pnc import PNCwithBERTStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

try:
    from nemo.collections.nlp.models import PunctuationCapitalizationModel  # noqa: F401

    _pnc_available = True
except (ImportError, ModuleNotFoundError):
    _pnc_available = False


class TestArabicRemoveDiacriticsStage:
    def test_removes_diacritics_from_entry(self) -> None:
        stage = ArabicRemoveDiacriticsStage(text_key="text")
        entry = {
            "text": "بِسْمِ اللَّهِ",
            "alignment": [
                {"word": "بِسْمِ", "start": 0.0, "end": 0.5},
                {"word": "اللَّهِ", "start": 0.5, "end": 1.0},
            ],
        }
        result = stage.process(AudioTask(data=entry))
        out = result.data
        assert out["text"] == "بسم الله"
        assert out["alignment"][0]["word"] == "بسم"
        assert out["alignment"][1]["word"] == "الله"

    def test_handles_split_metadata(self) -> None:
        stage = ArabicRemoveDiacriticsStage(text_key="text")
        entry = {
            "audio_filepath": "/a.wav",
            "split_metadata": [
                {
                    "text": "بِسْمِ",
                    "alignment": [{"word": "بِسْمِ", "start": 0.0, "end": 0.5}],
                },
                {
                    "text": "اللَّهِ",
                    "alignment": [{"word": "اللَّهِ", "start": 0.5, "end": 1.0}],
                },
            ],
        }
        result = stage.process(AudioTask(data=entry))
        out = result.data
        assert out["split_metadata"][0]["text"] == "بسم"
        assert out["split_metadata"][0]["alignment"][0]["word"] == "بسم"
        assert out["split_metadata"][1]["text"] == "الله"
        assert out["split_metadata"][1]["alignment"][0]["word"] == "الله"


@pytest.mark.skipif(not _pnc_available, reason="PunctuationCapitalizationModel requires nemo_toolkit <= 2.4.1")
class TestPNCwithBERTStage:
    def test_processes_segments(self, audio_task: Callable[..., AudioTask]) -> None:
        stage = PNCwithBERTStage(text_key="text", update_alignment=True, resources=Resources())
        stage.setup()
        task = audio_task(
            segments=[
                {
                    "text": "hello world",
                    "start": 0.0,
                    "end": 1.0,
                    "alignment": [
                        {"word": "hello", "start": 0.0, "end": 0.5},
                        {"word": "world", "start": 0.5, "end": 1.0},
                    ],
                },
            ],
        )
        result = stage.process(task)
        out = result.data
        assert out["segments"][0]["text"]
        assert out["segments"][0]["text"] == "Hello world."
        assert out["segments"][0]["alignment"][0]["word"] == "Hello"

    def test_processes_top_level_text(self) -> None:
        stage = PNCwithBERTStage(text_key="text", resources=Resources())
        stage.setup()
        entry = {"text": "hello world"}
        result = stage.process(AudioTask(data=entry))
        out = result.data
        assert out["text"]
        assert out["text"] == "Hello world."


class TestChineseConversionStage:
    def test_converts_traditional_to_simplified(self, audio_task: Callable[..., AudioTask]) -> None:
        stage = ChineseConversionStage(text_key="text", convert_type="t2s")
        stage.setup()
        task = audio_task(
            segments=[
                {"text": "漢字", "start": 0.0, "end": 1.0},
            ],
        )
        result = stage.process(task)
        out = result.data
        assert out["segments"][0]["text_simplified"] == "汉字"
        assert out["segments"][0]["text"] == "漢字"

    def test_segment_without_text_key_is_skipped(self, audio_task: Callable[..., AudioTask]) -> None:
        stage = ChineseConversionStage(text_key="text")
        stage.setup()
        task = audio_task(
            segments=[
                {"start": 0.0, "end": 1.0},
            ],
        )
        result = stage.process(task)
        out = result.data
        assert "text_simplified" not in out["segments"][0]
