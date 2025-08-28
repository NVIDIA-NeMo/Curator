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

from nemo_curator.stages.text.download.base.download import DocumentDownloadStage
from nemo_curator.stages.text.download.base.extract import DocumentExtractStage
from nemo_curator.stages.text.download.base.iterator import DocumentIterateStage
from nemo_curator.stages.text.download.base.url_generation import URLGenerationStage
from nemo_curator.stages.text.download.wikipedia.download import WikipediaDownloader
from nemo_curator.stages.text.download.wikipedia.extract import WikipediaExtractor
from nemo_curator.stages.text.download.wikipedia.iterator import WikipediaIterator
from nemo_curator.stages.text.download.wikipedia.stage import WikipediaDownloadExtractStage
from nemo_curator.stages.text.download.wikipedia.url_generation import WikipediaUrlGenerator


class TestWikipediaDownloadExtractStage:
    """Test suite for WikipediaDownloadExtractStage."""

    def test_wikipedia_stage_decomposition(self, tmp_path: Path):
        """Test that WikipediaDownloadExtractStage can be decomposed into constituent stages."""
        download_dir = str(tmp_path / "downloads")
        stage = WikipediaDownloadExtractStage(
            language="en",
            download_dir=download_dir,
            dump_date="20230501",
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

        # Verify URL generator
        url_gen_stage = stages[0]
        assert isinstance(url_gen_stage.url_generator, WikipediaUrlGenerator)

        # Verify downloader stage
        download_stage = stages[1]
        assert isinstance(download_stage.downloader, WikipediaDownloader)

        # Verify iterator stage
        iterate_stage = stages[2]
        assert isinstance(iterate_stage.iterator, WikipediaIterator)

        # Verify extractor stage
        extract_stage = stages[3]
        assert isinstance(extract_stage.extractor, WikipediaExtractor)

    def test_wikipedia_stage_name_default(self, tmp_path: Path):
        """Test that stage name is as expected for default language."""
        download_dir = str(tmp_path / "downloads")

        stage = WikipediaDownloadExtractStage(
            language="en",
            download_dir=download_dir,
        )
        assert stage.name == "wikipedia_en_pipeline"

    def test_wikipedia_stage_name_custom_language(self, tmp_path: Path):
        """Test that stage name is as expected for custom language."""
        download_dir = str(tmp_path / "downloads")

        stage = WikipediaDownloadExtractStage(
            language="es",
            download_dir=download_dir,
        )
        assert stage.name == "wikipedia_es_pipeline"

    def test_wikipedia_stage_description_with_latest_dump(self, tmp_path: Path):
        """Test stage description with latest dump."""
        download_dir = str(tmp_path / "downloads")

        stage = WikipediaDownloadExtractStage(
            language="en",
            download_dir=download_dir,
            dump_date=None,  # Latest dump
        )
        description = stage.get_description()
        assert description == "Wikipedia en pipeline for dump latest"

    def test_wikipedia_stage_description_with_specific_dump(self, tmp_path: Path):
        """Test stage description with specific dump date."""
        download_dir = str(tmp_path / "downloads")

        stage = WikipediaDownloadExtractStage(
            language="fr",
            download_dir=download_dir,
            dump_date="20230501",
        )
        description = stage.get_description()
        assert description == "Wikipedia fr pipeline for dump 20230501"

    def test_wikipedia_stage_initialization_default_values(self, tmp_path: Path):
        """Test initialization with default values."""
        download_dir = str(tmp_path / "downloads")

        stage = WikipediaDownloadExtractStage(
            language="en",
            download_dir=download_dir,
        )

        assert stage.language == "en"
        assert stage.download_dir == download_dir
        assert stage.dump_date is None
        assert stage.wikidumps_index_prefix == "https://dumps.wikimedia.org"
        assert stage.verbose is False
        assert stage.log_frequency == 1000

    def test_wikipedia_stage_initialization_custom_values(self, tmp_path: Path):
        """Test initialization with custom values."""
        download_dir = str(tmp_path / "downloads")

        stage = WikipediaDownloadExtractStage(
            language="es",
            download_dir=download_dir,
            dump_date="20230301",
            wikidumps_index_prefix="https://custom.dumps.org",
            verbose=True,
            url_limit=10,
            record_limit=100,
            add_filename_column="custom_filename",
            log_frequency=500,
        )

        assert stage.language == "es"
        assert stage.download_dir == download_dir
        assert stage.dump_date == "20230301"
        assert stage.wikidumps_index_prefix == "https://custom.dumps.org"
        assert stage.verbose is True
        assert stage.log_frequency == 500

    def test_wikipedia_stage_parameters_propagation(self, tmp_path: Path):
        """Test that parameters are properly propagated to constituent stages."""
        download_dir = str(tmp_path / "downloads")

        stage = WikipediaDownloadExtractStage(
            language="de",
            download_dir=download_dir,
            dump_date="20230601",
            wikidumps_index_prefix="https://custom.dumps.org",
            verbose=True,
            url_limit=15,
            record_limit=200,
            add_filename_column="source_file",
            log_frequency=250,
        )

        stages = stage.decompose()

        # Check URL generation stage
        url_stage = stages[0]
        assert isinstance(url_stage, URLGenerationStage)
        assert url_stage.limit == 15
        assert isinstance(url_stage.url_generator, WikipediaUrlGenerator)
        assert url_stage.url_generator.language == "de"
        assert url_stage.url_generator.dump_date == "20230601"
        assert url_stage.url_generator.wikidumps_index_prefix == "https://custom.dumps.org"

        # Check download stage
        download_stage = stages[1]
        assert isinstance(download_stage, DocumentDownloadStage)
        assert isinstance(download_stage.downloader, WikipediaDownloader)
        assert download_stage.downloader._download_dir == download_dir
        assert download_stage.downloader._verbose is True

        # Check iterate stage
        iterate_stage = stages[2]
        assert isinstance(iterate_stage, DocumentIterateStage)
        assert isinstance(iterate_stage.iterator, WikipediaIterator)
        assert iterate_stage.iterator._language == "de"
        assert iterate_stage.iterator._log_frequency == 250
        assert iterate_stage.record_limit == 200
        assert iterate_stage.filename_col == "source_file"

        # Check extract stage
        extract_stage = stages[3]
        assert isinstance(extract_stage, DocumentExtractStage)
        assert isinstance(extract_stage.extractor, WikipediaExtractor)
        assert extract_stage.extractor._language == "de"
        assert extract_stage.filename_col == "source_file"

    def test_wikipedia_stage_inputs_outputs(self, tmp_path: Path):
        """Test stage inputs and outputs specification."""
        download_dir = str(tmp_path / "downloads")

        stage = WikipediaDownloadExtractStage(
            language="en",
            download_dir=download_dir,
        )

        # The composite stage should have inputs/outputs from first and last stages
        inputs = stage.inputs()
        outputs = stage.outputs()

        # Should expect empty input (from URL generation stage)
        assert inputs == ([], [])

        # Should produce DocumentBatch with extracted text (from extract stage) + filename column
        assert outputs == (["data"], ["text", "title", "id", "url", "language", "source_id", "file_name"])

    def test_wikipedia_stage_with_custom_filename_column(self, tmp_path: Path):
        """Test stage with custom filename column."""
        download_dir = str(tmp_path / "downloads")

        stage = WikipediaDownloadExtractStage(
            language="en",
            download_dir=download_dir,
            add_filename_column="custom_source",
        )

        outputs = stage.outputs()
        assert outputs == (["data"], ["text", "title", "id", "url", "language", "source_id", "custom_source"])

    def test_wikipedia_stage_with_no_filename_column(self, tmp_path: Path):
        """Test stage without filename column."""
        download_dir = str(tmp_path / "downloads")

        stage = WikipediaDownloadExtractStage(
            language="en",
            download_dir=download_dir,
            add_filename_column=False,
        )

        outputs = stage.outputs()
        assert outputs == (["data"], ["text", "title", "id", "url", "language", "source_id"])

    def test_wikipedia_stage_component_creation(self, tmp_path: Path):
        """Test that all components are created correctly."""
        download_dir = str(tmp_path / "downloads")

        stage = WikipediaDownloadExtractStage(
            language="it",
            download_dir=download_dir,
            dump_date="20230701",
            wikidumps_index_prefix="https://test.dumps.org",
            verbose=True,
            log_frequency=100,
        )

        # Check that all components are created
        assert isinstance(stage.url_generator, WikipediaUrlGenerator)
        assert isinstance(stage.downloader, WikipediaDownloader)
        assert isinstance(stage.iterator, WikipediaIterator)
        assert isinstance(stage.extractor, WikipediaExtractor)

        # Check URL generator configuration
        assert stage.url_generator.language == "it"
        assert stage.url_generator.dump_date == "20230701"
        assert stage.url_generator.wikidumps_index_prefix == "https://test.dumps.org"

        # Check downloader configuration
        assert stage.downloader._download_dir == download_dir
        assert stage.downloader._verbose is True

        # Check iterator configuration
        assert stage.iterator._language == "it"
        assert stage.iterator._log_frequency == 100

        # Check extractor configuration
        assert stage.extractor._language == "it"

    def test_wikipedia_stage_different_languages(self, tmp_path: Path):
        """Test stage creation for different languages."""
        download_dir = str(tmp_path / "downloads")

        languages = ["en", "es", "fr", "de", "it", "ja", "zh"]

        for language in languages:
            stage = WikipediaDownloadExtractStage(
                language=language,
                download_dir=download_dir,
            )

            assert stage.language == language
            assert stage.name == f"wikipedia_{language}_pipeline"

            # Check that components are configured with correct language
            assert stage.url_generator.language == language
            assert stage.iterator._language == language
            assert stage.extractor._language == language

    def test_wikipedia_stage_url_limit_propagation(self, tmp_path: Path):
        """Test that URL limit is properly propagated."""
        download_dir = str(tmp_path / "downloads")

        stage = WikipediaDownloadExtractStage(
            language="en",
            download_dir=download_dir,
            url_limit=3,
        )

        stages = stage.decompose()
        url_stage = stages[0]
        assert url_stage.limit == 3

    def test_wikipedia_stage_record_limit_propagation(self, tmp_path: Path):
        """Test that record limit is properly propagated."""
        download_dir = str(tmp_path / "downloads")

        stage = WikipediaDownloadExtractStage(
            language="en",
            download_dir=download_dir,
            record_limit=50,
        )

        stages = stage.decompose()
        iterate_stage = stages[2]
        assert iterate_stage.record_limit == 50

    def test_wikipedia_stage_verbose_propagation(self, tmp_path: Path):
        """Test that verbose setting is properly propagated."""
        download_dir = str(tmp_path / "downloads")

        stage = WikipediaDownloadExtractStage(
            language="en",
            download_dir=download_dir,
            verbose=True,
        )

        stages = stage.decompose()
        download_stage = stages[1]
        assert download_stage.downloader._verbose is True

    def test_wikipedia_stage_log_frequency_propagation(self, tmp_path: Path):
        """Test that log frequency is properly propagated."""
        download_dir = str(tmp_path / "downloads")

        stage = WikipediaDownloadExtractStage(
            language="en",
            download_dir=download_dir,
            log_frequency=50,
        )

        stages = stage.decompose()
        iterate_stage = stages[2]
        assert iterate_stage.iterator._log_frequency == 50

    def test_wikipedia_stage_inheritance(self, tmp_path: Path):
        """Test that WikipediaDownloadExtractStage properly inherits from DocumentDownloadExtractStage."""
        download_dir = str(tmp_path / "downloads")

        stage = WikipediaDownloadExtractStage(
            language="en",
            download_dir=download_dir,
        )

        # Should have parent class attributes and methods
        assert hasattr(stage, "stages")
        assert hasattr(stage, "url_generator")
        assert hasattr(stage, "downloader")
        assert hasattr(stage, "iterator")
        assert hasattr(stage, "extractor")
        assert hasattr(stage, "decompose")
        assert hasattr(stage, "inputs")
        assert hasattr(stage, "outputs")

    def test_wikipedia_stage_stages_property(self, tmp_path: Path):
        """Test that stages property returns the correct stages."""
        download_dir = str(tmp_path / "downloads")

        stage = WikipediaDownloadExtractStage(
            language="en",
            download_dir=download_dir,
        )

        # Should have access to stages property from parent
        stages = stage.stages
        assert len(stages) == 4
        assert isinstance(stages[0], URLGenerationStage)
        assert isinstance(stages[1], DocumentDownloadStage)
        assert isinstance(stages[2], DocumentIterateStage)
        assert isinstance(stages[3], DocumentExtractStage)


class TestWikipediaDownloadExtractStageIntegration:
    """Integration tests for WikipediaDownloadExtractStage."""

    def test_wikipedia_stage_end_to_end_configuration(self, tmp_path: Path):
        """Test end-to-end configuration of Wikipedia stage."""
        download_dir = str(tmp_path / "downloads")

        # Create a comprehensive stage configuration
        stage = WikipediaDownloadExtractStage(
            language="es",
            download_dir=download_dir,
            dump_date="20230515",
            wikidumps_index_prefix="https://dumps.wikimedia.org",
            verbose=True,
            url_limit=2,
            record_limit=10,
            add_filename_column="wiki_source",
            log_frequency=5,
        )

        # Verify all configurations are correct
        assert stage.language == "es"
        assert stage.download_dir == download_dir
        assert stage.dump_date == "20230515"
        assert stage.verbose is True
        assert stage.log_frequency == 5

        # Verify decomposition works correctly
        stages = stage.decompose()
        assert len(stages) == 4

        # Verify that each stage has correct configuration
        url_gen_stage = stages[0]
        assert url_gen_stage.url_generator.language == "es"
        assert url_gen_stage.url_generator.dump_date == "20230515"
        assert url_gen_stage.limit == 2

        download_stage = stages[1]
        assert download_stage.downloader._download_dir == download_dir
        assert download_stage.downloader._verbose is True

        iterate_stage = stages[2]
        assert iterate_stage.iterator._language == "es"
        assert iterate_stage.iterator._log_frequency == 5
        assert iterate_stage.record_limit == 10
        assert iterate_stage.filename_col == "wiki_source"

        extract_stage = stages[3]
        assert extract_stage.extractor._language == "es"
        assert extract_stage.filename_col == "wiki_source"

        # Verify inputs/outputs
        inputs = stage.inputs()
        outputs = stage.outputs()
        assert inputs == ([], [])
        assert outputs == (["data"], ["text", "title", "id", "url", "language", "source_id", "wiki_source"])

    def test_wikipedia_stage_multiple_languages_configuration(self, tmp_path: Path):
        """Test configuration of multiple Wikipedia stages for different languages."""
        download_dir = str(tmp_path / "downloads")

        # Create stages for different languages
        languages = ["en", "es", "fr"]
        stages = []

        for lang in languages:
            stage = WikipediaDownloadExtractStage(
                language=lang,
                download_dir=download_dir,
                dump_date="20230501",
                url_limit=1,
                record_limit=5,
            )
            stages.append(stage)

        # Verify each stage is configured correctly
        for i, stage in enumerate(stages):
            lang = languages[i]

            assert stage.language == lang
            assert stage.name == f"wikipedia_{lang}_pipeline"
            assert stage.get_description() == f"Wikipedia {lang} pipeline for dump 20230501"

            # Check component configuration
            assert stage.url_generator.language == lang
            assert stage.iterator._language == lang
            assert stage.extractor._language == lang

    def test_wikipedia_stage_realistic_configuration(self, tmp_path: Path):
        """Test a realistic Wikipedia stage configuration."""
        download_dir = str(tmp_path / "downloads")

        # Create a stage with realistic parameters
        stage = WikipediaDownloadExtractStage(
            language="en",
            download_dir=download_dir,
            dump_date=None,  # Use latest dump
            verbose=False,
            url_limit=5,  # Process 5 dump files
            record_limit=1000,  # Process 1000 articles per file
            add_filename_column=True,  # Add filename column
            log_frequency=100,  # Log every 100 articles
        )

        # Verify configuration
        assert stage.language == "en"
        assert stage.dump_date is None
        assert stage.verbose is False
        assert stage.log_frequency == 100

        # Verify decomposition
        stages = stage.decompose()
        assert len(stages) == 4

        # Verify realistic limits
        url_gen_stage = stages[0]
        assert url_gen_stage.limit == 5

        iterate_stage = stages[2]
        assert iterate_stage.record_limit == 1000
        assert iterate_stage.iterator._log_frequency == 100

        # Verify outputs include filename
        outputs = stage.outputs()
        assert "file_name" in outputs[1]
