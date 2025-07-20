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

import re
from typing import Any

from ray_curator.stages.download.text import DocumentExtractor

# The iterator and extractor code are in large part taken
# from the Red-Pajama repo
# https://github.com/togethercomputer/RedPajama-Data/tree/main/data_prep/arxiv


class ArxivExtractor(DocumentExtractor):
    """Extracts text from Arxiv LaTeX files."""

    def __init__(self):
        super().__init__()

    def _build_non_arg_macros_dict(self, file_content: str) -> dict[str, str]:
        r"""function takes the content of a tex file and returns a dictionary
        that contains the definitions of all macros that do not use arguments.
        The dictionary is of the form {macro_name: macro_value}.

        @param file_content: the content of the tex file as a string.

        @return: dict
        """
        # regex for extracting \newcommand macros without arguments
        non_arg_nc_reg = re.compile(
            # this regex matches the following:
            # \newcommand{\macro_name}{macro_value}
            # \newcommand*{\macro_name}{macro_value}
            # where macro_name is only allowed to contain letters and numbers;
            # macro_value can contain any character.
            pattern=r"\\\bnewcommand\b\*?\{(\\[a-zA-Z0-9]+?)\}\{(.*?)\}$",
            flags=re.MULTILINE,
        )

        # regex for extracting \def macros without arguments
        non_arg_def_reg = re.compile(
            # this regex matches the following:
            # \def\macro_name{macro_value}
            # where macro_name is only allowed to contain letters and numbers;
            # macro_value can contain any character.
            pattern=r"\\def\s*(\\[a-zA-Z0-9]+?)\s*\{(.*?)\}$",
            flags=re.MULTILINE,
        )

        # Extract all user-defined LaTeX macros from the preamble
        macros = {}
        for reg in [non_arg_nc_reg, non_arg_def_reg]:
            for match in reg.finditer(file_content):
                # convert the macro name and value to a raw string that can be
                # used in re.sub
                macro_name = match.group(1).encode("unicode-escape").decode("utf-8")
                macro_val = match.group(2).encode("unicode-escape").decode("utf-8")

                macros[macro_name] = macro_val

        return macros

    def extract(self, record: dict[str, str]) -> dict[str, Any] | None:
        if len(record["content"]) == 0:
            return None

        # build dictionaries that contain the definitions of all macros in all tex
        # files. This is later used to expand all macros used in the text with
        # their definitions, so that consistency among different authors is
        # ensured.

        non_arg_macros = {}
        for file_content in record["content"]:
            non_arg_macros.update(self._build_non_arg_macros_dict(file_content))

        # TODO: macros that take arguments are not supported yet
        arg_macros = {}

        # join multiple latex files with a newline character
        try:
            cleaned_latex_file_str = "\n".join(
                self._clean_tex_file(
                    file_content=file_content,
                    arg_macros=arg_macros,
                    non_arg_macros=non_arg_macros,
                )
                for file_content in record["content"]
            )
        except Exception:  # noqa: BLE001
            return None

        # Don't return meta
        if (cleaned_latex_file_str is not None) and (len(cleaned_latex_file_str) > 0):
            return {"text": cleaned_latex_file_str}

        return None

    def input_columns(self) -> list[str]:
        return ["id", "source_id", "content"]

    def output_columns(self) -> list[str]:
        return ["text"]
