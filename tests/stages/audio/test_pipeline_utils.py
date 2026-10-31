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

from nemo_curator.stages.audio.pipeline_utils import (
    normalize_language_code,
    resolve_indic_language_code,
    resolve_model_route,
    resolve_whisper_language_code,
)


@pytest.mark.parametrize(
    ("language", "primary", "recovery"),
    [
        ("en", "qwen_omni", "qwen_asr"),
        ("th", "qwen_asr", "whisper"),
        ("pl", "parakeet_v3", "whisper"),
        ("he", "whisper", "none"),
        ("hi", "parakeet_riva", "indic_monolingual"),
        ("ml", "indic_monolingual", "none"),
        ("ur", "indic_monolingual", "qwen_omni"),
    ],
)
def test_reference_language_routes(language: str, primary: str, recovery: str) -> None:
    route = resolve_model_route(language)

    assert route.primary == primary
    assert route.recovery == recovery
    assert route.has_recovery is (recovery != "none")


def test_explicit_model_overrides_win() -> None:
    route = resolve_model_route("English", primary="whisper", recovery="none")

    assert route.language == "en"
    assert route.primary == "whisper"
    assert route.recovery == "none"


def test_backend_language_normalization() -> None:
    assert normalize_language_code("Hindi") == "hi"
    assert resolve_indic_language_code("Malayalam") == "ml"
    assert resolve_indic_language_code("English") is None
    assert resolve_whisper_language_code("Filipino") == "tl"


def test_unknown_language_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        resolve_model_route("xx")
