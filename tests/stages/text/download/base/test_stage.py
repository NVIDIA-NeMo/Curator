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

from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.download.base.download import DocumentDownloadStage
from nemo_curator.stages.text.download.base.extract import DocumentExtractStage
from nemo_curator.stages.text.download.base.iterator import DocumentIterateStage
from nemo_curator.stages.text.download.base.stage import DocumentDownloadExtractStage
from nemo_curator.stages.text.download.base.url_generation import URLGenerationStage

from .test_download import MockDocumentDownloader
from .test_extract import MockDocumentExtractor
from .test_iterator import MockDocumentIterator
from .test_url_generation import MockURLGenerator


class TestDocumentDownloadExtractStage:
    """Test class for DocumentDownloadExtractStage composite functionality."""

    def test_stage_initialization_with_extractor(self, tmp_path: Path) -> None:
        """Test that composite stage initializes correctly with extractor."""
        url_generator = MockURLGenerator()
        downloader = MockDocumentDownloader(str(tmp_path))
        iterator = MockDocumentIterator()
        extractor = MockDocumentExtractor()

        stage = DocumentDownloadExtractStage(
            url_generator=url_generator,
            downloader=downloader,
            iterator=iterator,
            extractor=extractor,
            url_limit=5,
            record_limit=10,
            add_filename_column=True,
        )

        # Check that all components are stored
        assert stage.url_generator is url_generator
        assert stage.downloader is downloader
        assert stage.iterator is iterator
        assert stage.extractor is extractor
        assert stage.url_limit == 5
        assert stage.record_limit == 10
        assert stage.add_filename_column is True

    def test_stage_initialization_without_extractor(self, tmp_path: Path) -> None:
        """Test that composite stage initializes correctly without extractor."""
        url_generator = MockURLGenerator()
        downloader = MockDocumentDownloader(str(tmp_path))
        iterator = MockDocumentIterator()

        stage = DocumentDownloadExtractStage(
            url_generator=url_generator,
            downloader=downloader,
            iterator=iterator,
            extractor=None,
            url_limit=None,
            record_limit=None,
            add_filename_column="source_file",
        )

        # Check that components are stored correctly
        assert stage.url_generator is url_generator
        assert stage.downloader is downloader
        assert stage.iterator is iterator
        assert stage.extractor is None
        assert stage.url_limit is None
        assert stage.record_limit is None
        assert stage.add_filename_column == "source_file"

    def test_stage_properties(self, tmp_path: Path) -> None:
        """Test that stage properties are correctly defined."""
        url_generator = MockURLGenerator()
        downloader = MockDocumentDownloader(str(tmp_path))
        iterator = MockDocumentIterator()
        extractor = MockDocumentExtractor()

        stage = DocumentDownloadExtractStage(
            url_generator=url_generator,
            downloader=downloader,
            iterator=iterator,
            extractor=extractor,
        )

        # Test stage name
        expected_name = "document_download_extract_mockurlgenerator_mockdocumentdownloader_composite"
        assert stage.name == expected_name

        # Test inputs and outputs (from first and last stages)
        assert stage.inputs() == ([], [])  # From URL generation stage
        assert stage.outputs() == (
            ["data"],
            ["id", "processed_text", "language", "char_count", "file_name"],
        )  # From extract stage

    def test_stage_properties_without_extractor(self, tmp_path: Path) -> None:
        """Test stage properties when no extractor is provided."""
        url_generator = MockURLGenerator()
        downloader = MockDocumentDownloader(str(tmp_path))
        iterator = MockDocumentIterator()

        stage = DocumentDownloadExtractStage(
            url_generator=url_generator,
            downloader=downloader,
            iterator=iterator,
            extractor=None,
        )

        # Should use iterator outputs when no extractor
        assert stage.outputs() == (["data"], ["id", "content", "metadata", "file_name"])

    def test_decompose_with_extractor(self, tmp_path: Path) -> None:
        """Test decomposition into constituent stages with extractor."""
        url_generator = MockURLGenerator()
        downloader = MockDocumentDownloader(str(tmp_path))
        iterator = MockDocumentIterator()
        extractor = MockDocumentExtractor()

        stage = DocumentDownloadExtractStage(
            url_generator=url_generator,
            downloader=downloader,
            iterator=iterator,
            extractor=extractor,
            url_limit=5,
            record_limit=10,
        )

        stages = stage.decompose()

        # Should have 4 stages: URL generation, download, iterate, extract
        assert len(stages) == 4

        # Check stage types and order
        assert isinstance(stages[0], URLGenerationStage)
        assert isinstance(stages[1], DocumentDownloadStage)
        assert isinstance(stages[2], DocumentIterateStage)
        assert isinstance(stages[3], DocumentExtractStage)

        # Check that parameters are propagated correctly
        url_stage = stages[0]
        assert url_stage.url_generator is url_generator
        assert url_stage.limit == 5

        download_stage = stages[1]
        assert download_stage.downloader is downloader

        iterate_stage = stages[2]
        assert iterate_stage.iterator is iterator
        assert iterate_stage.record_limit == 10

        extract_stage = stages[3]
        assert extract_stage.extractor is extractor

    def test_decompose_without_extractor(self, tmp_path: Path) -> None:
        """Test decomposition into constituent stages without extractor."""
        url_generator = MockURLGenerator()
        downloader = MockDocumentDownloader(str(tmp_path))
        iterator = MockDocumentIterator()

        stage = DocumentDownloadExtractStage(
            url_generator=url_generator,
            downloader=downloader,
            iterator=iterator,
            extractor=None,
        )

        stages = stage.decompose()

        # Should have 3 stages: URL generation, download, iterate (no extract)
        assert len(stages) == 3

        # Check stage types and order
        assert isinstance(stages[0], URLGenerationStage)
        assert isinstance(stages[1], DocumentDownloadStage)
        assert isinstance(stages[2], DocumentIterateStage)

    def test_get_description(self, tmp_path: Path) -> None:
        """Test that stage description is correctly generated."""
        url_generator = MockURLGenerator()
        downloader = MockDocumentDownloader(str(tmp_path))
        iterator = MockDocumentIterator()
        extractor = MockDocumentExtractor()

        stage = DocumentDownloadExtractStage(
            url_generator=url_generator,
            downloader=downloader,
            iterator=iterator,
            extractor=extractor,
        )

        description = stage.get_description()
        expected = "URL-Download-Iterate-Extract pipeline using MockURLGenerator and MockDocumentDownloader"
        assert description == expected

    def test_stage_parameter_propagation(self, tmp_path: Path) -> None:
        """Test that parameters are correctly propagated to constituent stages."""
        url_generator = MockURLGenerator()
        downloader = MockDocumentDownloader(str(tmp_path))
        iterator = MockDocumentIterator()
        extractor = MockDocumentExtractor()

        stage = DocumentDownloadExtractStage(
            url_generator=url_generator,
            downloader=downloader,
            iterator=iterator,
            extractor=extractor,
            url_limit=3,
            record_limit=5,
            add_filename_column="custom_file",
        )

        stages = stage.decompose()

        # Check URL generation stage
        url_stage = stages[0]
        assert isinstance(url_stage, URLGenerationStage)
        assert url_stage.limit == 3

        # Check iterate stage
        iterate_stage = stages[2]
        assert isinstance(iterate_stage, DocumentIterateStage)
        assert iterate_stage.record_limit == 5
        assert iterate_stage.filename_col == "custom_file"

        # Check extract stage
        extract_stage = stages[3]
        assert isinstance(extract_stage, DocumentExtractStage)
        assert extract_stage.filename_col == "custom_file"

    def test_stage_resources(self, tmp_path: Path) -> None:
        """Test that stage has appropriate resource requirements."""
        url_generator = MockURLGenerator()
        downloader = MockDocumentDownloader(str(tmp_path))
        iterator = MockDocumentIterator()

        stage = DocumentDownloadExtractStage(
            url_generator=url_generator,
            downloader=downloader,
            iterator=iterator,
            extractor=None,
        )

        # Should have default resources from base class
        assert isinstance(stage.resources, Resources)

    def test_stage_different_filename_column_types(self, tmp_path: Path) -> None:
        """Test stage behavior with different filename column configurations."""
        url_generator = MockURLGenerator()
        downloader = MockDocumentDownloader(str(tmp_path))
        iterator = MockDocumentIterator()
        extractor = MockDocumentExtractor()

        # Test with boolean True
        stage_bool = DocumentDownloadExtractStage(
            url_generator=url_generator,
            downloader=downloader,
            iterator=iterator,
            extractor=extractor,
            add_filename_column=True,
        )

        stages = stage_bool.decompose()
        iterate_stage = stages[2]
        extract_stage = stages[3]
        assert isinstance(iterate_stage, DocumentIterateStage)
        assert isinstance(extract_stage, DocumentExtractStage)
        assert iterate_stage.filename_col == "file_name"  # Default name
        assert extract_stage.filename_col == "file_name"

        # Test with boolean False
        stage_false = DocumentDownloadExtractStage(
            url_generator=url_generator,
            downloader=downloader,
            iterator=iterator,
            extractor=extractor,
            add_filename_column=False,
        )

        stages = stage_false.decompose()
        iterate_stage = stages[2]
        extract_stage = stages[3]
        assert isinstance(iterate_stage, DocumentIterateStage)
        assert isinstance(extract_stage, DocumentExtractStage)
        assert iterate_stage.add_filename_column is False
        assert extract_stage.add_filename_column is False

        # Test with custom string
        stage_custom = DocumentDownloadExtractStage(
            url_generator=url_generator,
            downloader=downloader,
            iterator=iterator,
            extractor=extractor,
            add_filename_column="source_path",
        )

        stages = stage_custom.decompose()
        iterate_stage = stages[2]
        extract_stage = stages[3]
        assert isinstance(iterate_stage, DocumentIterateStage)
        assert isinstance(extract_stage, DocumentExtractStage)
        assert iterate_stage.filename_col == "source_path"
        assert extract_stage.filename_col == "source_path"

    def test_stage_edge_cases(self, tmp_path: Path) -> None:
        """Test edge cases for the composite stage."""
        url_generator = MockURLGenerator(urls=[])  # Empty URLs
        downloader = MockDocumentDownloader(str(tmp_path))
        iterator = MockDocumentIterator(records_per_file=0)  # No records

        stage = DocumentDownloadExtractStage(
            url_generator=url_generator,
            downloader=downloader,
            iterator=iterator,
            extractor=None,
            url_limit=0,  # Zero limit
            record_limit=0,  # Zero limit
        )

        stages = stage.decompose()

        # Check that limits are correctly set
        url_stage = stages[0]
        iterate_stage = stages[2]
        assert isinstance(url_stage, URLGenerationStage)
        assert isinstance(iterate_stage, DocumentIterateStage)
        assert url_stage.limit == 0
        assert iterate_stage.record_limit == 0

        # Should still have all required stages
        assert len(stages) == 3

    @mock.patch("nemo_curator.stages.text.download.base.stage.URLGenerationStage")
    @mock.patch("nemo_curator.stages.text.download.base.stage.DocumentDownloadStage")
    @mock.patch("nemo_curator.stages.text.download.base.stage.DocumentIterateStage")
    @mock.patch("nemo_curator.stages.text.download.base.stage.DocumentExtractStage")
    def test_stage_initialization_mocking(
        self,
        mock_extract_stage: mock.Mock,
        mock_iterate_stage: mock.Mock,
        mock_download_stage: mock.Mock,
        mock_url_stage: mock.Mock,
        tmp_path: Path,
    ) -> None:
        """Test that stage initialization creates the correct stage instances."""
        url_generator = MockURLGenerator()
        downloader = MockDocumentDownloader(str(tmp_path))
        iterator = MockDocumentIterator()
        extractor = MockDocumentExtractor()

        _ = DocumentDownloadExtractStage(
            url_generator=url_generator,
            downloader=downloader,
            iterator=iterator,
            extractor=extractor,
            url_limit=5,
            record_limit=10,
            add_filename_column="test_file",
        )

        # Verify that each stage type was instantiated with correct parameters
        mock_url_stage.assert_called_once_with(
            url_generator=url_generator,
            limit=5,
        )

        mock_download_stage.assert_called_once_with(
            downloader=downloader,
        )

        mock_iterate_stage.assert_called_once_with(
            iterator=iterator,
            record_limit=10,
            add_filename_column="test_file",
        )

        mock_extract_stage.assert_called_once_with(
            extractor=extractor,
            add_filename_column="test_file",
        )
