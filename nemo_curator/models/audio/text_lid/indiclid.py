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

"""Text language-ID adapter for AI4Bharat IndicLID-FTN v1.0."""

from __future__ import annotations

from nemo_curator.models.audio.text_lid.fasttext import FastTextFormatLIDAdapter

# Labels as the released IndicLID-FTN v1.0 checkpoint spells them; Meti, Olch and Tamil are not ISO 15924 codes.
INDICLID_LABEL_TO_LANGUAGE: dict[str, str] = {
    "asm_Beng": "as",
    "ben_Beng": "bn",
    "brx_Deva": "brx",
    "doi_Deva": "doi",
    # Manifests and the language check use ``en``, as the LLM language-ID stage writes it.
    "eng_Latn": "en",
    "guj_Gujr": "gu",
    "hin_Deva": "hi",
    "kan_Knda": "kn",
    "kas_Arab": "ks",
    "kas_Deva": "ks",
    "kok_Deva": "kok",
    "mai_Deva": "mai",
    "mal_Mlym": "ml",
    "mar_Deva": "mr",
    "mni_Beng": "mni",
    "mni_Meti": "mni",
    "nep_Deva": "ne",
    "ori_Orya": "or",
    "other": "other",
    "pan_Guru": "pa",
    "san_Deva": "sa",
    "sat_Olch": "sat",
    "snd_Arab": "sd",
    "tam_Tamil": "ta",
    "tel_Telu": "te",
    "urd_Arab": "ur",
}


class IndicLIDAdapter(FastTextFormatLIDAdapter):
    """IndicLID-FTN; only carriage returns and line feeds become spaces, and both script labels of Kashmiri and
    Manipuri map to one language code."""

    def check_source_language(self, language: str) -> None:
        if language not in INDICLID_LABEL_TO_LANGUAGE.values():
            msg = f"IndicLID-FTN has no label mapping for source_lang={language!r}"
            raise ValueError(msg)

    def _validate_labels(self, labels: frozenset[str]) -> None:
        expected_labels = frozenset(INDICLID_LABEL_TO_LANGUAGE)
        if labels != expected_labels:
            msg = (
                "IndicLID-FTN label inventory mismatch: "
                f"missing={sorted(expected_labels - labels)}, extra={sorted(labels - expected_labels)}"
            )
            raise ValueError(msg)

    def _sanitize_text(self, text: str) -> str:
        return text.replace("\r", " ").replace("\n", " ")

    def _map_label_to_language(self, raw_label: str) -> str:
        return INDICLID_LABEL_TO_LANGUAGE[raw_label]
