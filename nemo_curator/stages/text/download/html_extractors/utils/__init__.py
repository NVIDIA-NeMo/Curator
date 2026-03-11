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

import justext


def get_stop_list_dict(languages: list[str] | None = None) -> dict[str, frozenset[str]]:
    if languages is None:
        languages = []

    # Name mapping for language names from CLD2 (values)
    # and jusText (keys)
    lang_map = {
        "Haitian": "HAITIAN_CREOLE",
        "Norwegian_Bokmal": "NORWEGIAN",
        "Norwegian_Nynorsk": "NORWEGIAN_N",
        "Waray_Waray": "WARAY_PHILIPPINES",
    }

    # List obtained from https://github.com/stopwords-iso/stopwords-ja
    from .ja_stopwords import ja_stopwords

    # List obtained from https://github.com/stopwords-iso/stopwords-th
    from .th_stopwords import th_stopwords

    # List obtained from https://github.com/stopwords-iso/stopwords-zh
    from .zh_stopwords import zh_stopwords

    custom_stopwords = {
        "THAI": th_stopwords,
        "CHINESE": zh_stopwords,
        "JAPANESE": ja_stopwords,
    }

    if len(languages) == 0:
        languages = list(justext.get_stoplists())

        # Remove Latin as it yields a lot of low quality documents
        languages.remove("Latin")

        # Manually add Thai, Chinese, and Japanese
        languages.append("THAI")
        languages.append("CHINESE")
        languages.append("JAPANESE")

    stop_list_dict = {}
    for language in languages:
        lang_key = lang_map[language] if language in lang_map else language.upper()

        if lang_key in custom_stopwords:
            stop_list_dict[lang_key] = custom_stopwords[lang_key]
        else:
            stop_list_dict[lang_key] = justext.get_stoplist(language)

    return stop_list_dict
