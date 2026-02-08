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

import os
from collections.abc import Iterator
from typing import Any

import requests
from loguru import logger

from nemo_curator.stages.text.download.base import (
    DocumentDownloader,
    DocumentDownloadExtractStage,
    DocumentExtractor,
    DocumentIterator,
    URLGenerator,
)
from nemo_curator.stages.text.filters import DocumentFilter
from nemo_curator.stages.text.modifiers import DocumentModifier


class TinyStoriesURLGenerator(URLGenerator):
    """
    The implementation of a URL generator. For this example, we only need to download a single file.
    """

    def __init__(self, split: str):
        self.split = split

    def generate_urls(self) -> list[str]:
        return [f"https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-{self.split}.txt"]


class TinyStoriesDownloader(DocumentDownloader):
    """
    The implementation of a document downloader. Fetches the given URL and populates the given file.
    """

    def _get_output_filename(self, url: str) -> str:
        """Generate output filename from URL.

        Args:
            url: URL to download

        Returns:
            Output filename (without directory path)
        """
        return os.path.basename(url)

    def _download_to_path(self, url: str, path: str) -> tuple[bool, str | None]:
        """Download URL to specified path.

        Args:
            url: URL to download
            path: Local path to save file

        Returns:
            Tuple of (success, error_message). If success is True, error_message should be None.
            If success is False, error_message should contain the error details.
        """
        logger.debug(f"Downloading TinyStories dataset from '{url}'...")
        response = requests.get(url)  # noqa: S113

        with open(path, "wb") as file:
            file.write(response.content)

        return True, None


class TinyStoriesIterator(DocumentIterator):
    """
    The implementation of an iterator defining how to itereate the raw dataset and fetch records.
    """

    def __init__(self):
        super().__init__()
        self.record_separator_token = "<|endoftext|>"  # noqa: S105

    def iterate(self, file_path: str) -> Iterator[tuple[dict[str, str], str]]:
        sample = []

        with open(file_path) as file:
            # Keep reading the lines from file until we encounter a separator token.
            for line in file:
                if line.strip() == self.record_separator_token:
                    if sample:
                        # Join the sample pieces together, then yield
                        sample = " ".join(sample)
                        yield {"text": sample}
                        # Clear out for the next sample
                        sample = []
                else:
                    sample.append(line.strip())

            # Yield the last sample (if any)
            if sample:
                sample = " ".join(sample)
                yield {"text": sample}

    def output_columns(self) -> list[str]:
        return ["text"]


class TinyStoriesExtractor(DocumentExtractor):
    """
    The implementation of a document extractor. For this example, it's a no-op.
    """

    def extract(self, record: dict[str, str]) -> dict[str, Any] | None:
        return record

    def input_columns(self) -> list[str]:
        return ["text"]

    def output_columns(self) -> list[str]:
        return ["text"]


class TinyStoriesDownloadExtractStage(DocumentDownloadExtractStage):
    """
    The implementation of the download and extraction stage. Combines the above into a single stage.
    """

    def __init__(
        self,
        download_dir: str,
        split: str,
        verbose: bool = True,
    ):
        self.name = "tinystories_download_extract_pipeline"
        self.split = split

        self.url_generator = TinyStoriesURLGenerator(split=split)
        self.downloader = TinyStoriesDownloader(
            download_dir=download_dir,
            verbose=verbose,
        )
        self.iterator = TinyStoriesIterator()
        self.extractor = TinyStoriesExtractor()

        super().__init__(
            url_generator=self.url_generator,
            downloader=self.downloader,
            iterator=self.iterator,
            extractor=self.extractor,
            add_filename_column=True,
        )

    def get_description(self) -> str:
        """Get a description of this composite stage."""
        return f"TinyStories pipeline on split '{self.split}'"


class QuotationUnifier(DocumentModifier):
    """
    A simple modifier that unifies the quotation marks in the documents.
    """

    def modify_document(self, text: str) -> str:
        """
        Modifies the given text by replacing left and right single quotes with normal single quotes,
        and replacing left and right double quotes with normal double quotes.

        Args:
            text (str): The text to be modified.

        Returns:
            str: The modified text.
        """
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        return text.replace("\u201c", '"').replace("\u201d", '"')


class IncompleteStoryFilter(DocumentFilter):
    """
    If the document doesn't end with a terminating punctuation mark, then discard.
    """

    def __init__(self):
        super().__init__()
        # Accepted story terminators.
        self._story_terminators = {".", "!", "?", '"', "”"}

    def score_document(self, text: str) -> bool:
        """
        Determines if a document's score is valid based on the last character of the text.

        Args:
            text (str): The document text.

        Returns:
            bool: True if the document's score is valid, False otherwise.
        """
        return text.strip()[-1] in self._story_terminators

    def keep_document(self, scores: bool) -> bool:
        return scores
