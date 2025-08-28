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

from nemo_curator.stages.text.download import DocumentExtractor

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

    def _clean_tex_file(self, file_content: str, arg_macros: dict[str, str], non_arg_macros: dict[str, str]) -> str:
        r"""function takes a tex file as input and returns a cleaned version. The
         cleaned version is a concatenation of the tex files with the
        following modifications:

        - remove all comments (i.e. all lines starting with %)
        - remove everything before the first section-like header
        - remove everything after the first occurrence of either \appendix or
            \bibliography
        - inline-expand definitions and macros

        @param file_content: the content of the tex file as a string.

        @return: cleaned tex file as a string
        """
        # find the first occurence of a \section-like header and replace everything
        # before it with an empty string. This matches the following pattern:
        #   \<section-type>[optional-args]{name}
        pattern = r"^(.*?)("
        pattern += r"\\\bchapter\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
        pattern += r"\\\bpart\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
        pattern += r"\\\bsection\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
        pattern += r"\\\bsubsection\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
        pattern += r"\\\bsubsubsection\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
        pattern += r"\\\bparagraph\b\*?(?:\[(.*?)\])?\{(.*?)\}"
        pattern += r"\\\bsubparagraph\b\*?(?:\[(.*?)\])?\{(.*?)\}"
        pattern += r")"

        # if no section like header is found, then we return an empty string
        if not re.search(pattern, file_content, flags=re.DOTALL):
            return ""

        # replace everything with the second group of the match (i.e. everything
        # after and including the section header)
        file_content = re.sub(
            pattern=pattern,
            repl=r"\2",
            string=file_content,
            flags=re.DOTALL,  # make sure that the dot matches also newlines
        )

        # remove all line comments
        file_content = re.sub(
            pattern=r"(?m)^%.*\n?",
            repl=r"",
            string=file_content,
            flags=re.MULTILINE,
        )

        # remove all in comments within a line
        file_content = re.sub(
            # pattern matches a "%" that is not preceded by a backslash (=comment)
            pattern=r"[^\\]%.+$",
            repl=r"",
            string=file_content,
            flags=re.MULTILINE,
        )

        # find the first occurence of either \appendix or \bibliography and
        # replace everything after it with an empty string
        pattern = r"("
        pattern += r"\\appendix|"
        pattern += r"\\begin\{references\}|"
        pattern += r"\\begin\{REFERENCES\}|"
        pattern += r"\\begin\{thebibliography\}|"
        pattern += r"\\bibliography\{.*\}"
        pattern += r").*$"

        file_content = re.sub(
            pattern=pattern,
            repl=r"",
            string=file_content,
            flags=re.DOTALL,  # make sure that the dot matches also newlines
        )

        # inline-expand all non-arg macros
        for macro_name, macro_value in non_arg_macros.items():
            file_content = re.sub(
                # make pattern grouped to make sure that the macro is not part
                # of a longer alphanumeric word
                pattern=r"(" + macro_name + r")" + r"([^a-zA-Z0-9])",
                # replace the macro with its value and add back the character that
                # was matched after the macro
                repl=macro_value + r"\2",
                string=file_content,
            )

        # inline-expand all macros that use args
        # TODO: inline-expand macros with args
        for _macro_name, _macro_value in arg_macros.items():
            pass

        return file_content

    def extract(self, record: dict[str, str]) -> dict[str, Any] | None:
        if "content" not in record or len(record["content"]) == 0:
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
