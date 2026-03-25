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

import os.path
import tarfile

import requests
from platformdirs import user_cache_dir

from nemo_curator.stages.text.filters.doc_filter import DocumentFilter


class HistogramFilter(DocumentFilter):
    """Histogram filter used by the NLLB paper (https://arxiv.org/pdf/2207.04672). See p30 for details.

    The high-level idea of histogram filter can be described as a cheap version of language ID.
    Basically, it checks what ratio of characters in the data instance are included in the character historgrams collected from trusted data in the corresponding language.
    If the ratio is too low, then there is a good chance that there is a language ID mismatch and the data instance should be discarded.

    Written with reference to the original fairseq implementation at:
    https://github.com/facebookresearch/fairseq/blob/main/examples/m2m_100/process_data/clean_histogram.py.
    """

    def __init__(
        self,
        lang: str | None = "en",
        threshold: float | None = 0.8,
        cache_dir: str | None = "",
        threshold_char: str | None = "]",
    ):
        """Args:
        lang (str, optional): Expected language of the segment. This will decide which histogram will be loaded. Defaults to "en".
        threshold (float, optional): Threshold for ratio of characters in the histogram. Defaults to 0.8.
        cache_dir (str, optional): Cache dir download histogram files. Defaults to "".
        threshold_char (str, optional): Formatter character of the histogram files. You should not change this unless you rebuilt your own histogram. Defaults to "]".
        """
        super().__init__()
        self._lang = lang
        self._threshold = threshold
        self._cache_dir = cache_dir if cache_dir else user_cache_dir()
        self._threshold_char = threshold_char
        self._name = "histogram"

        if not os.path.isdir(os.path.join(self._cache_dir, "histograms")):
            self._download_histograms()

        self._read_hist()

    def _download_histograms(self) -> None:
        """Download and process histograms from default repo.

        Raises:
            requests.exceptions.RequestException: If download fails.
        """

        # Send a GET request to the URL
        response = requests.get("https://dl.fbaipublicfiles.com/m2m_100/histograms.tar.gz")  # noqa: S113

        # Check if the request was successful
        if response.status_code != 200:  # noqa: PLR2004
            msg = f"Failed to download histogram file. Status code: {response.status_code}"
            raise requests.exceptions.RequestException(msg)

        # Open a file to write the content
        os.makedirs(self._cache_dir, exist_ok=True)
        download_dest_path = os.path.join(self._cache_dir, "histograms.tar.gz")
        with open(download_dest_path, "wb") as file:
            file.write(response.content)

        extract_path = os.path.join(self._cache_dir, "histograms")
        with tarfile.open(download_dest_path, "r:gz") as tar:
            # Extract all the contents into the specified directory
            tar.extractall(path=extract_path)  # noqa: S202

    def _read_hist(self) -> None:
        """Load histogram files."""

        self._histogram = []
        with open(
            os.path.join(
                self._cache_dir,
                "histograms",
                "checkpoint",
                "edunov",
                "cc60_multilingual",
                "clean_hists",
                self._lang,
            )
        ) as f:
            for line in f:
                c = line[0]
                if c == self._threshold_char:
                    break
                self._histogram.append(c)
        self._histogram = set(self._histogram)

    def score_document(self, text: str) -> float:
        """Compute histogram token ratio of a text data instance according to the loaded histogram.

        Args:
            text (str): Text data instance.

        Returns:
            float: Ratio of tokens included in the histogram.
        """
        cnt = len([c for c in text.strip() if c in self._histogram])
        return 1 if cnt / len(text) > self._threshold else 0

    def keep_document(self, score: float) -> bool:
        return score == 1
