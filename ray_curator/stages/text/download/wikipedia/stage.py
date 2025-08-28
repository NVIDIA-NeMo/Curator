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

from .download import WikipediaDownloader
from .extract import WikipediaExtractor
from .iterator import WikipediaIterator
from .url_generation import WikipediaUrlGenerator


class WikipediaDownloadExtractStage(DocumentDownloadExtractStage):
    """Composite stage for downloading and processing Wikipedia data.

    This pipeline:
    1. Generates Wikipedia dump URLs for the specified language and date
    2. Downloads Wikipedia .bz2 dump files
    3. Extracts articles from the dump files
    4. Cleans and extracts text from Wikipedia markup
    """

    def __init__(  # noqa: PLR0913
        self,
        language: str = "en",
        download_dir: str = "./wikipedia_downloads",
        dump_date: str | None = None,
        wikidumps_index_prefix: str = "https://dumps.wikimedia.org",
        verbose: bool = False,
        url_limit: int | None = None,
        record_limit: int | None = None,
        add_filename_column: bool | str = True,
        log_frequency: int = 1000,
    ):
        """
        Initialize the Wikipedia download and extract stage.

        Args:
            language: Language code for the Wikipedia dump (e.g., "en", "es", "fr")
            download_dir: Directory to store downloaded .bz2 files
            dump_date: Specific dump date in "YYYYMMDD" format (if None, uses latest)
            wikidumps_index_prefix: Base URL for Wikipedia dumps
            verbose: If True, enables verbose logging
            url_limit: Maximum number of dump URLs to process
            record_limit: Maximum number of articles to extract per file
            add_filename_column: Whether to add filename column to output
            log_frequency: How often to log progress during iteration
        """
        self.language = language
        self.download_dir = download_dir
        self.dump_date = dump_date
        self.wikidumps_index_prefix = wikidumps_index_prefix
        self.verbose = verbose
        self.log_frequency = log_frequency

        # Create the URL generator
        self.url_generator = WikipediaUrlGenerator(
            language=language,
            dump_date=dump_date,
            wikidumps_index_prefix=wikidumps_index_prefix,
        )

        # Create the downloader
        self.downloader = WikipediaDownloader(
            download_dir=download_dir,
            verbose=verbose,
        )

        # Create the iterator
        self.iterator = WikipediaIterator(
            language=language,
            log_frequency=log_frequency,
        )

        # Create the extractor
        self.extractor = WikipediaExtractor(
            language=language,
        )

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
        self._name = f"wikipedia_{self.language}_pipeline"

    def decompose(self) -> list[ProcessingStage]:
        """Decompose this composite stage into its constituent stages."""
        return self.stages

    def get_description(self) -> str:
        """Get a description of this composite stage."""
        date_str = self.dump_date if self.dump_date else "latest"
        return f"Wikipedia {self.language} pipeline for dump {date_str}"
