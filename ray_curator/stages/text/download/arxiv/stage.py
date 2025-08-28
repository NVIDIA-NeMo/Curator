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

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.download import DocumentDownloadExtractStage

from .download import ArxivDownloader
from .extract import ArxivExtractor
from .iterator import ArxivIterator
from .url_generation import ArxivUrlGenerator


class ArxivDownloadExtractStage(DocumentDownloadExtractStage):
    """Composite stage for downloading and processing Arxiv data.

    This pipeline:
    1. Generates Arxiv dump URLs
    2. Downloads Arxiv .tar files
    3. Extracts articles from the tar files
    4. Cleans and extracts text from LaTeX files
    """

    def __init__(  # noqa: PLR0913
        self,
        download_dir: str = "./arxiv_downloads",
        url_limit: int | None = None,
        record_limit: int | None = None,
        add_filename_column: bool | str = True,
        log_frequency: int = 1000,
        verbose: bool = False,
    ):
        """
        Download Arxiv tar files and extract the contained LaTeX projects.

        This function obtains a list of Arxiv tar file URLs (via get_arxiv_urls), downloads the tar files,
        and then extracts the contained LaTeX source files. The resulting documents (after extraction) are
        assembled into a DocumentDataset.

        Args:
            download_dir (str, optional):
                The directory where the raw downloaded tar files will be kept. Defaults to "./arxiv_downloads".
            url_limit (Optional[int], optional):
                Limits the maximum number of Arxiv tar file URLs to download and process.
                If None, all available URLs (from get_arxiv_urls) are processed.
            record_limit (Optional[int], optional):
                Limits the maximum number of records to extract from each tar file.
                If None, all available records are extracted.
            add_filename_column (bool | str, optional):
                If True, adds a column to the output DataFrame with the filename of the tar file.
                If a string, adds a column with the specified name. Defaults to True.
            log_frequency (int, optional):
                How often to log progress. Defaults to 1000.
            verbose (bool, optional):
                If True, prints verbose output. Defaults to False.
        Returns:
            DocumentBatch:
                A batch object containing the extracted documents.
        """

        # Create the URL generator
        self.url_generator = ArxivUrlGenerator()

        # Create the downloader
        self.downloader = ArxivDownloader(
            download_dir=download_dir,
            verbose=verbose,
        )

        # Create the iterator
        self.iterator = ArxivIterator(
            log_frequency=log_frequency,
        )

        # Create the extractor
        self.extractor = ArxivExtractor()

        # Initialize the parent composite stage
        super().__init__(
            url_generator=self.url_generator,
            downloader=self.downloader,
            iterator=self.iterator,
            extractor=self.extractor,
            url_limit=url_limit,
            record_limit=record_limit,
            add_filename_column=add_filename_column,
        )
        self._name = "arxiv_pipeline"

    def decompose(self) -> list[ProcessingStage]:
        """Decompose this composite stage into its constituent stages."""
        return self.stages

    def get_description(self) -> str:
        """Get a description of this composite stage."""
        return "Arxiv pipeline"
