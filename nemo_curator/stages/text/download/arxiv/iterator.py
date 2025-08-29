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

import gzip
import os
import re
import tarfile
import tempfile
from collections.abc import Iterator
from typing import Any

from loguru import logger

from nemo_curator.stages.text.download import DocumentIterator
from nemo_curator.utils.file_utils import get_all_file_paths_under, tar_safe_extract

# The iterator and extractor code are in large part taken
# from the Red-Pajama repo
# https://github.com/togethercomputer/RedPajama-Data/tree/main/data_prep/arxiv


class ArxivIterator(DocumentIterator):
    """Processes downloaded Arxiv files and extracts article content."""

    def __init__(self, log_frequency: int = 1000):
        super().__init__()
        self._log_frequency = log_frequency
        self._counter = 0

    def _tex_proj_loader(self, file_or_dir_path: str) -> list[str] | None:
        r"""function to load the tex files from a tar file or a gzip file. The
        function will return a tuple containing a list of tex files and the
        timestamp of the project.

        @param file_or_dir_path: path to the tar file or the gzip file

        @return: tuple containing a list of tex files and the timestamp of the
            project
        """
        files_and_content = []

        try:
            # if it is a directory, open it as a tarfile
            with tarfile.open(file_or_dir_path) as sub_tf:
                for member in sub_tf.getmembers():
                    if member.name.endswith(".tex"):
                        file_content = sub_tf.extractfile(member).read()

                        try:
                            file_content = file_content.decode("utf-8")
                        except UnicodeDecodeError:
                            logger.debug(f"UnicodeDecodeError: {file_or_dir_path}")
                            return None

                        files_and_content.append(file_content)

        except tarfile.ReadError:
            # otherwise we try opening it as a gzip file
            try:
                with gzip.open(file_or_dir_path, "rb") as gz:
                    file_content = gz.read()
            except Exception as e:  # noqa: BLE001
                # all fails, we skip this file
                logger.debug(f"[ERROR] {e}: {file_or_dir_path}")
                return None

            try:
                file_content = file_content.decode("utf-8")
            except UnicodeDecodeError:
                logger.debug(f"UnicodeDecodeError: {file_or_dir_path}")
                return None

            files_and_content.append(file_content)

        except Exception as e:  # noqa: BLE001
            logger.debug(f"[ERROR] {e}: {file_or_dir_path}")
            return None

        return files_and_content

    def _format_arxiv_id(self, arxiv_id: str) -> str:
        r"""this function brings the raw arxiv-id into a format compliant with the
        specification from arxiv. This is used to create the url to the arxiv
        abstract page.

        - Format prior to March 2007:
            <archive>/YYMMNNN where N is a 3-digit number
        - Format after March 2007: <archive>/YYMM.NNNNN where N is a
          5 (or 6)-digit number

        References: https://info.arxiv.org/help/arxiv_identifier.html

        @param arxiv_id: raw arxiv id which can be in one of the
                         following formats:
                         - <archive><YY><MM><NNN>
                         - <YY><MM><NNNNN|NNNNNN>

        @return: formatted arxiv id
        """
        match = re.search(r"^([a-zA-Z-]*)([\d\.]+)$", arxiv_id)

        if match is None:
            msg = f"Invalid arxiv id: {arxiv_id}"
            raise ValueError(msg)

        if match.group(1) == "":
            return match.group(2)

        return f"{match.group(1)}/{match.group(2)}"

    def iterate(self, file_path: str) -> Iterator[dict[str, Any]]:
        self._counter = 0
        download_dir = os.path.split(file_path)[0]
        bname = os.path.split(file_path)[-1]

        with tempfile.TemporaryDirectory(dir=download_dir) as tmpdir, tarfile.open(file_path) as tf:
            # Use safe extraction instead of extractall to prevent path traversal attacks
            tar_safe_extract(tf, tmpdir)
            for _i, item in enumerate(get_all_file_paths_under(tmpdir, recurse_subdirectories=True)):
                if self._counter > 0 and self._counter % self._log_frequency == 0:
                    logger.info(f"Extracted {self._counter} papers from {file_path}")
                self._counter += 1

                tex_files = self._tex_proj_loader(item)
                arxiv_id = os.path.splitext(os.path.split(item)[-1])[0]

                # get the arxiv id in the correct format
                try:
                    clean_arxiv_id = self._format_arxiv_id(arxiv_id)
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"[WARNING] failed to format arxiv id {arxiv_id}; exception={e}")
                    clean_arxiv_id = arxiv_id

                if tex_files is None:
                    continue

                yield {"id": clean_arxiv_id, "source_id": f"{bname}", "content": tex_files}

    def output_columns(self) -> list[str]:
        return ["id", "source_id", "content"]
