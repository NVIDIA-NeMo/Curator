# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import re

logger = logging.getLogger(__name__)


# A/AC.261/L.12/Rev.1
un_doc_re = re.compile(r"A/[A-Z0-9][A-Z.0-9]*(/[A-Z0-9][A-Z.0-9]*)*")


def is_hallucination(text):
    """Detect mbart-large-50 hallucinations.

    Background: mbart-large-50 has been trained on a lot of UN documents,
    which contain many repeated phrases, and the model seems to recite
    them often.
    """
    if (
        # es
        text.startswith("El Comité recomienda")
        or text.startswith("En la misma sesión")
        or text.startswith("La Comisión")
        or text.startswith("El Grupo de Trabajo")
        # fr
        or text.startswith("Le Comité")
        or text.startswith("Les caractéristiques de l’environnement sont les suivantes")
    ):
        return True

    # UN document numbers
    if un_doc_re.match(text):
        return True
    # UN jargon
    if (
        # es
        "Asamblea General" in text or
        "Secretario General" in text or
        "Naciones Unidas" in text or
        "Comisión" in text or
        "Comité" in text or
        "Secretaría" in text or
        "Secretaría General" in text or
        # fr
        "Assemblée Générale" in text or
        "Secrétariat Général" in text or
        "Nations Unies" in text or
        "Commission" in text or
        "Comité" in text or
        "Secrétariat" in text or
        "Secrétariat général" in text or
        "Communauté européenne" in text or
        # pt
        "Assembleia Geral" in text or
        "Secretário Geral" in text or
        "Nações Unidas" in text or
        "Comissão" in text or
        "Comité" in text or
        "Secretaria" in text or
        "Secretaria Geral" in text or
        # it
        "Assemblea Generale" in text or
        "Segretario Generale" in text or
        "Nazioni Unite" in text or
        "Commissione" in text or
        "Comitato" in text or
        "Segreteria" in text or
        "Segreteria Generale" in text or
        # de
        "Generalversammlung" in text or
        "Generaldirektor" in text or
        "Vereinte Nationen" in text or
        "Kommission" in text or
        "Komitee" in text
    ):
        return True
    # latex hallucinations
    if "\\displaystyle" in text:
        return True
    return False


def is_short_phrase(sentence: str) -> bool:
    if sentence.startswith('The text is "'):
        return False
    return len(sentence) < 30 or len(sentence.split()) < 5


class TranslationFailed(Exception):
    pass


def translate_check(translator, sentence: str, **kwargs):
    text = translator(sentence, **kwargs)
    if is_hallucination(text):
        raise TranslationFailed(f"Hallucination: {sentence} -> {text}")
    elif len(text) > 5 * len(sentence):
        raise TranslationFailed(f"Long output: {sentence} -> {text}")
    return fix_anchors(sentence, text)


def translate_wrapped(translator, sentence: str, **kwargs):
    """Translate phrases wrapped in a sentence.

    Background: mbart-large-50 sometimes has trouble translating text that is not a proper sentence.
    """
    text = translate_check(translator, f'The text is "{sentence}".', **kwargs)

    quote_pairs = [
        '""', # double quotes
        "“”", # smart double quotes
        "«»", # french double quotes
        "„”", # german double quotes
        "「」", # chinese/japanese quotes
        "『』", # japanese quotes
    ]

    for open_quote, close_quote in quote_pairs:
        left_idx = text.find(open_quote)
        if left_idx != -1:
            right_idx = text.rfind(close_quote)
            if right_idx > left_idx:
                return text[left_idx + 1:right_idx]

    raise TranslationFailed(f"Failed to wrap: {sentence} -> {text}")


def fix_anchors(source: str, translation: str) -> str:
    """Fix broken anchors in translation.

    Background: mbart-large-50 may break anchors like <a1> by adding whitespace, etc.
    """
    source_anchors = re.findall(r"(\s*)<a(\d)+>(\s*)", source)
    for pre_ws, idx, post_ws in source_anchors:
        translation = re.sub(rf"\s*< ?[aA] ?{idx} ?>\s*", f"{pre_ws}<a{idx}>{post_ws}", translation)
    return translation


def mbart_heuristics(translator, sentence: str, **kwargs):
    """Apply heuristics around mbart hallucinations."""
    try:
        if len(sentence.strip()) == 1:
            return sentence
        # wrap short phrases in a sentence
        if is_short_phrase(sentence):
            return translate_wrapped(translator, sentence, **kwargs)
        try:
            return translate_check(translator, sentence, **kwargs)
        except TranslationFailed:
            return translate_wrapped(translator, sentence, **kwargs)
    except TranslationFailed as err:
        logger.warning("Translation failed: %s", err)
        return sentence
