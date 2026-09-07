# modality: text
# ruff: noqa: RUF001

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

from contextlib import suppress

import pandas as pd
import pytest

with suppress(ImportError):
    import cudf

with suppress(ImportError):
    from nemo_curator.stages.text.utils.text import normalize_text


@pytest.mark.gpu
class TestNormalizeText:
    def test_defaults_lowercase_and_normalize_space(self) -> None:
        text = cudf.Series(["  THE quick\tbrown\nfox  ", "CAFÉ\tÀ LA CRÈME"])

        result = normalize_text(text)

        assert result.to_arrow().to_pylist() == ["the quick brown fox", "café à la crème"]

    @pytest.mark.parametrize(
        ("lowercase", "normalize_space", "expected"),
        [
            (True, False, ["  mixed\tcase  "]),
            (False, True, ["MIXED Case"]),
            (False, False, ["  MIXED\tCase  "]),
        ],
    )
    def test_options(
        self,
        lowercase: bool,
        normalize_space: bool,
        expected: list[str],
    ) -> None:
        text = cudf.Series(["  MIXED\tCase  "])

        result = normalize_text(text, lowercase=lowercase, normalize_space=normalize_space)

        assert result.to_arrow().to_pylist() == expected
        if not lowercase and not normalize_space:
            assert result is text

    @pytest.mark.parametrize(
        ("value", "lowercase", "normalize_space", "expected"),
        [
            pytest.param("  CAFÉ\tÀ LA\nCRÈME  ", True, True, "café à la crème", id="french"),
            pytest.param("¡HOLA,\tSEÑOR!", True, False, "¡hola,\tseñor!", id="spanish-lowercase"),
            pytest.param("  ΓΕΙΆ\tΣΟΥ\nΚΌΣΜΕ  ", True, True, "γειά σου κόσμε", id="greek"),
            pytest.param("  ПРИВЕТ\tМИР,\nКАК ДЕЛА?  ", True, True, "привет мир, как дела?", id="russian"),
            pytest.param("  مرحبًا\tبالعالم،\nكيف حالك؟  ", True, True, "مرحبًا بالعالم، كيف حالك؟", id="arabic"),
            pytest.param(
                "  こんにちは\t世界。\nお元気ですか？  ",
                False,
                True,
                "こんにちは 世界。 お元気ですか？",
                id="japanese",
            ),
        ],
    )
    def test_multilingual_options(
        self,
        value: str,
        lowercase: bool,
        normalize_space: bool,
        expected: str,
    ) -> None:
        result = normalize_text(cudf.Series([value]), lowercase=lowercase, normalize_space=normalize_space)

        assert result.to_arrow().to_pylist() == [expected]

    @pytest.mark.parametrize("text_series", ["text", ["text"], pd.Series(["text"])])
    def test_rejects_non_cudf_series(self, text_series: object) -> None:
        with pytest.raises(TypeError, match=r"Expected text_series of type cudf\.Series"):
            normalize_text(text_series)
