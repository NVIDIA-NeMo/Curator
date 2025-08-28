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

from pathlib import Path
from typing import Literal

import pytest

from nemo_curator.stages.text.download.base.download import DocumentDownloadStage
from nemo_curator.stages.text.download.base.extract import DocumentExtractStage
from nemo_curator.stages.text.download.base.iterator import DocumentIterateStage
from nemo_curator.stages.text.download.base.url_generation import URLGenerationStage
from nemo_curator.stages.text.download.common_crawl.download import CommonCrawlWARCDownloader
from nemo_curator.stages.text.download.common_crawl.extract import CommonCrawlHTMLExtractor
from nemo_curator.stages.text.download.common_crawl.stage import CommonCrawlDownloadExtractStage
from nemo_curator.stages.text.download.common_crawl.url_generation import (
    MainCommonCrawlUrlGenerator,
    NewsCommonCrawlUrlGenerator,
)
from nemo_curator.stages.text.download.common_crawl.warc_iterator import CommonCrawlWarcIterator
from nemo_curator.stages.text.download.html_extractors import JusTextExtractor, ResiliparseExtractor


class TestCommonCrawlDownloadExtractStage:
    """Test suite for CommonCrawlDownloadExtractStage."""

    @pytest.mark.parametrize(
        ("crawl_type", "start_snapshot", "end_snapshot"),
        [
            ("main", "2021-23", "2021-26"),  # YYYY-WW format for main
            ("news", "2021-04", "2021-10"),  # YYYY-MM format for news
        ],
    )
    def test_common_crawl_stage_decomposition(
        self, tmp_path: Path, crawl_type: Literal["main", "news"], start_snapshot: str, end_snapshot: str
    ) -> None:
        """Test that CommonCrawlDownloadExtractStage can be decomposed into constituent stages."""
        download_dir = str(tmp_path / "downloads")
        stage = CommonCrawlDownloadExtractStage(
            start_snapshot=start_snapshot,
            end_snapshot=end_snapshot,
            download_dir=download_dir,
            crawl_type=crawl_type,
            html_extraction="justext",
            url_limit=5,
        )

        # Decompose the stage
        stages = stage.decompose()

        # Should have 4 stages: URL generation, download, iterate, extract
        assert len(stages) == 4

        # Check stage types
        assert isinstance(stages[0], URLGenerationStage)
        assert isinstance(stages[1], DocumentDownloadStage)
        assert isinstance(stages[2], DocumentIterateStage)
        assert isinstance(stages[3], DocumentExtractStage)

        # Verify the correct URL generator is used based on crawl_type
        url_gen_stage = stages[0]
        if crawl_type == "main":
            assert isinstance(url_gen_stage.url_generator, MainCommonCrawlUrlGenerator)
        else:  # news
            assert isinstance(url_gen_stage.url_generator, NewsCommonCrawlUrlGenerator)

        # Verify downloader stage
        download_stage = stages[1]
        assert isinstance(download_stage.downloader, CommonCrawlWARCDownloader)

        # Verify iterator stage
        iterate_stage = stages[2]
        assert isinstance(iterate_stage.iterator, CommonCrawlWarcIterator)

        # Verify extractor stage
        extract_stage = stages[3]
        assert isinstance(extract_stage.extractor, CommonCrawlHTMLExtractor)

    def test_common_crawl_stage_name(self, tmp_path: Path) -> None:
        """Test that stage name is as expected."""
        download_dir = str(tmp_path / "downloads")

        # Test main crawl
        main_stage = CommonCrawlDownloadExtractStage(
            start_snapshot="2021-23",
            end_snapshot="2021-26",
            download_dir=download_dir,
            crawl_type="main",
        )
        assert main_stage.name == "common_crawl_main_pipeline"

        # Test news crawl
        news_stage = CommonCrawlDownloadExtractStage(
            start_snapshot="2021-04",
            end_snapshot="2021-10",
            download_dir=download_dir,
            crawl_type="news",
        )
        assert news_stage.name == "common_crawl_news_pipeline"

    def test_common_crawl_stage_description(self, tmp_path: Path) -> None:
        """Test that stage description is as expected."""
        download_dir = str(tmp_path / "downloads")

        # Test main crawl
        main_stage = CommonCrawlDownloadExtractStage(
            start_snapshot="2021-23",
            end_snapshot="2021-26",
            download_dir=download_dir,
            crawl_type="main",
        )
        description = main_stage.get_description()
        assert description == "Common Crawl main pipeline: 2021-23 to 2021-26"

        # Test news crawl
        news_stage = CommonCrawlDownloadExtractStage(
            start_snapshot="2021-04",
            end_snapshot="2021-10",
            download_dir=download_dir,
            crawl_type="news",
        )
        description = news_stage.get_description()
        assert description == "Common Crawl news pipeline: 2021-04 to 2021-10"

    def test_common_crawl_html_extraction_algorithms(self, tmp_path: Path) -> None:
        """Test different HTML extraction algorithms initialization."""
        download_dir = str(tmp_path / "downloads")

        # Test with string algorithm
        stage_justext = CommonCrawlDownloadExtractStage(
            start_snapshot="2021-04", end_snapshot="2021-10", download_dir=download_dir, html_extraction="justext"
        )

        # Get the HTML extractor stage (4th stage)
        stages = stage_justext.decompose()
        extract_stage = stages[3]
        assert isinstance(extract_stage, DocumentExtractStage)
        assert isinstance(extract_stage.extractor, CommonCrawlHTMLExtractor)
        assert isinstance(extract_stage.extractor.algorithm, JusTextExtractor)

        # Test with algorithm object and custom stop lists
        custom_stop_lists = {"en": frozenset(["the", "and", "or"])}
        stage_resiliparse = CommonCrawlDownloadExtractStage(
            start_snapshot="2021-04",
            end_snapshot="2021-10",
            download_dir=download_dir,
            html_extraction=ResiliparseExtractor(),
            stop_lists=custom_stop_lists,
        )

        stages = stage_resiliparse.decompose()
        extract_stage = stages[3]
        assert isinstance(extract_stage, DocumentExtractStage)
        assert isinstance(extract_stage.extractor, CommonCrawlHTMLExtractor)
        assert isinstance(extract_stage.extractor.algorithm, ResiliparseExtractor)
        assert extract_stage.extractor._stop_lists == custom_stop_lists

    def test_common_crawl_stage_without_extractor(self, tmp_path: Path) -> None:
        """Test stage creation without an extractor (should still have 4 stages with default extractor)."""
        download_dir = str(tmp_path / "downloads")

        stage = CommonCrawlDownloadExtractStage(
            start_snapshot="2021-23",
            end_snapshot="2021-26",
            download_dir=download_dir,
            crawl_type="main",
            html_extraction=None,  # No extractor specified
        )

        # Should still have 4 stages as extractor is created with default algorithm
        stages = stage.decompose()
        assert len(stages) == 4

        # The extractor should be created with default algorithm
        extract_stage = stages[3]
        assert isinstance(extract_stage, DocumentExtractStage)
        assert isinstance(extract_stage.extractor, CommonCrawlHTMLExtractor)
        assert isinstance(extract_stage.extractor.algorithm, JusTextExtractor)

    def test_common_crawl_stage_parameters_propagation(self, tmp_path: Path) -> None:
        """Test that parameters are properly propagated to constituent stages."""
        download_dir = str(tmp_path / "downloads")

        stage = CommonCrawlDownloadExtractStage(
            start_snapshot="2021-23",
            end_snapshot="2021-26",
            download_dir=download_dir,
            crawl_type="main",
            use_aws_to_download=False,
            verbose=True,
            url_limit=10,
            record_limit=100,
            add_filename_column="custom_filename",
        )

        stages = stage.decompose()

        # Check URL generation stage
        url_stage = stages[0]
        assert isinstance(url_stage, URLGenerationStage)
        assert url_stage.limit == 10

        # Check download stage
        download_stage = stages[1]
        assert isinstance(download_stage, DocumentDownloadStage)
        assert isinstance(download_stage.downloader, CommonCrawlWARCDownloader)
        assert download_stage.downloader._download_dir == download_dir
        assert download_stage.downloader.use_aws_to_download is False
        assert download_stage.downloader._verbose is True

        # Check iterate stage
        iterate_stage = stages[2]
        assert isinstance(iterate_stage, DocumentIterateStage)
        assert iterate_stage.record_limit == 100
        assert iterate_stage.filename_col == "custom_filename"

        # Check extract stage
        extract_stage = stages[3]
        assert isinstance(extract_stage, DocumentExtractStage)
        assert extract_stage.filename_col == "custom_filename"

    def test_common_crawl_stage_inputs_outputs(self, tmp_path: Path) -> None:
        """Test stage inputs and outputs specification."""
        download_dir = str(tmp_path / "downloads")

        stage = CommonCrawlDownloadExtractStage(
            start_snapshot="2021-23",
            end_snapshot="2021-26",
            download_dir=download_dir,
            crawl_type="main",
        )

        # The composite stage should have inputs/outputs from first and last stages
        inputs = stage.inputs()
        outputs = stage.outputs()

        # Should expect empty input (from URL generation stage)
        assert inputs == ([], [])

        # Should produce DocumentBatch with extracted text (from extract stage) + filename column
        assert outputs == (["data"], ["url", "warc_id", "source_id", "language", "text", "file_name"])

    def test_common_crawl_stage_initialization_validation(self, tmp_path: Path) -> None:
        """Test that stage initialization validates parameters correctly."""
        download_dir = str(tmp_path / "downloads")

        # Test valid initialization
        stage = CommonCrawlDownloadExtractStage(
            start_snapshot="2021-23",
            end_snapshot="2021-26",
            download_dir=download_dir,
            crawl_type="main",
        )
        assert stage.crawl_type == "main"
        assert stage.start_snapshot == "2021-23"
        assert stage.end_snapshot == "2021-26"

        # Test that stage stores the components
        assert stage.url_generator is not None
        assert stage.downloader is not None
        assert stage.iterator is not None
        assert stage.extractor is not None

    def test_common_crawl_stage_algorithm_kwargs(self, tmp_path: Path) -> None:
        """Test that algorithm kwargs are passed correctly."""
        download_dir = str(tmp_path / "downloads")

        algorithm_kwargs = {"length_low": 50, "stopwords_low": 0.25}
        stage = CommonCrawlDownloadExtractStage(
            start_snapshot="2021-23",
            end_snapshot="2021-26",
            download_dir=download_dir,
            crawl_type="main",
            html_extraction="justext",
            html_extraction_kwargs=algorithm_kwargs,
        )

        # The algorithm kwargs should be passed to the extractor
        # (Testing that the extractor is created with the right parameters)
        assert stage.extractor is not None
        assert isinstance(stage.extractor, CommonCrawlHTMLExtractor)

        # Check that the custom parameters were applied
        algorithm = stage.extractor.algorithm
        assert isinstance(algorithm, JusTextExtractor)
        assert algorithm.length_low == 50
        assert algorithm.stopwords_low == 0.25
