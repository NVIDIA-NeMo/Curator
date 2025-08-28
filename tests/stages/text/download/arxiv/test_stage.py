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
from unittest import mock

from nemo_curator.stages.text.download.arxiv.download import ArxivDownloader
from nemo_curator.stages.text.download.arxiv.extract import ArxivExtractor
from nemo_curator.stages.text.download.arxiv.iterator import ArxivIterator
from nemo_curator.stages.text.download.arxiv.stage import ArxivDownloadExtractStage
from nemo_curator.stages.text.download.arxiv.url_generation import ArxivUrlGenerator
from nemo_curator.stages.text.download.base.download import DocumentDownloadStage
from nemo_curator.stages.text.download.base.extract import DocumentExtractStage
from nemo_curator.stages.text.download.base.iterator import DocumentIterateStage
from nemo_curator.stages.text.download.base.url_generation import URLGenerationStage


class TestArxivDownloadExtractStage:
    """Test suite for ArxivDownloadExtractStage."""

    @mock.patch.object(ArxivDownloader, "_check_s5cmd_installed", return_value=True)
    def test_arxiv_stage_decomposition(self, tmp_path: Path) -> None:
        """Test that ArxivDownloadExtractStage can be decomposed into constituent stages."""
        download_dir = str(tmp_path / "downloads")
        stage = ArxivDownloadExtractStage(download_dir=download_dir)

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
        assert isinstance(url_gen_stage.url_generator, ArxivUrlGenerator)

        # Verify downloader stage
        download_stage = stages[1]
        assert isinstance(download_stage.downloader, ArxivDownloader)

        # Verify iterator stage
        iterate_stage = stages[2]
        assert isinstance(iterate_stage.iterator, ArxivIterator)

        # Verify extractor stage
        extract_stage = stages[3]
        assert isinstance(extract_stage.extractor, ArxivExtractor)

    @mock.patch.object(ArxivDownloader, "_check_s5cmd_installed", return_value=True)
    def test_arxiv_stage_name(self, tmp_path: Path) -> None:
        """Test that stage name is as expected."""
        download_dir = str(tmp_path / "downloads")

        stage = ArxivDownloadExtractStage(download_dir=download_dir)
        assert stage.name == "arxiv_pipeline"

    @mock.patch.object(ArxivDownloader, "_check_s5cmd_installed", return_value=True)
    def test_arxiv_stage_description(self, tmp_path: Path) -> None:
        """Test that stage description is as expected."""
        download_dir = str(tmp_path / "downloads")

        stage = ArxivDownloadExtractStage(download_dir=download_dir)
        description = stage.get_description()
        assert description == "Arxiv pipeline"

    @mock.patch.object(ArxivDownloader, "_check_s5cmd_installed", return_value=True)
    def test_arxiv_stage_parameters_propagation(self, tmp_path: Path) -> None:
        """Test that parameters are properly propagated to constituent stages."""
        download_dir = str(tmp_path / "downloads")

        stage = ArxivDownloadExtractStage(
            download_dir=download_dir,
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
        assert isinstance(download_stage.downloader, ArxivDownloader)
        assert download_stage.downloader._download_dir == download_dir
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

    @mock.patch.object(ArxivDownloader, "_check_s5cmd_installed", return_value=True)
    def test_arxiv_stage_inputs_outputs(self, tmp_path: Path) -> None:
        """Test stage inputs and outputs specification."""
        download_dir = str(tmp_path / "downloads")

        stage = ArxivDownloadExtractStage(download_dir=download_dir)

        # The composite stage should have inputs/outputs from first and last stages
        inputs = stage.inputs()
        outputs = stage.outputs()

        # Should expect empty input (from URL generation stage)
        assert inputs == ([], [])

        # Should produce DocumentBatch with extracted text (from extract stage) + filename column
        assert outputs == (["data"], ["text", "file_name"])

    @mock.patch.object(ArxivDownloader, "_check_s5cmd_installed", return_value=True)
    def test_arxiv_stage_initialization_validation(self, tmp_path: Path) -> None:
        """Test that stage initialization validates parameters correctly."""
        download_dir = str(tmp_path / "downloads")

        # Test valid initialization
        stage = ArxivDownloadExtractStage(download_dir=download_dir)

        # Test that stage stores the components
        assert stage.url_generator is not None
        assert stage.downloader is not None
        assert stage.iterator is not None
        assert stage.extractor is not None
