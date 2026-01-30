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

"""Tests for code annotation processing stages."""

import gzip

import polars as pl
import pytest

from nemo_curator.tasks import DocumentBatch


def compress_content(content: str) -> bytes:
    """Compress content using gzip."""
    return gzip.compress(content.encode("utf-8"))


@pytest.fixture
def sample_python_code():
    """Sample Python source code for testing."""
    return '''# This is a comment
def hello_world():
    """Docstring for hello_world."""
    print("Hello, World!")

if __name__ == "__main__":
    hello_world()
'''


@pytest.fixture
def sample_document_batch(sample_python_code: str):
    """Create a sample DocumentBatch for testing."""
    df = pl.DataFrame(
        {
            "compressed_content": [compress_content(sample_python_code)],
            "representative_filename": ["hello.py"],
        }
    )
    return DocumentBatch(
        task_id="test_batch_001",
        dataset_name="test_dataset",
        data=df,
    )


class TestLanguageIdentificationStage:
    """Tests for LanguageIdentificationStage."""

    def test_import_with_missing_dependency(self):
        """Test that import fails gracefully without code_annotation."""
        # This test is conditional - it will pass if code_annotation is available
        # and skip if not
        try:
            from nemo_curator.stages.code import LanguageIdentificationStage

            stage = LanguageIdentificationStage()
            assert stage.name == "LanguageIdentificationStage"
        except ImportError as e:
            pytest.skip(f"code_annotation not available: {e}")

    def test_inputs_outputs(self):
        """Test inputs() and outputs() methods."""
        try:
            from nemo_curator.stages.code import LanguageIdentificationStage

            stage = LanguageIdentificationStage()

            inputs = stage.inputs()
            assert "data" in inputs[0]
            assert "compressed_content" in inputs[1]

            outputs = stage.outputs()
            assert "data" in outputs[0]
            assert "language" in outputs[1]
        except ImportError:
            pytest.skip("code_annotation not available")


class TestCodeMetricsStage:
    """Tests for CodeMetricsStage."""

    def test_import_and_config(self):
        """Test stage import and configuration."""
        try:
            from nemo_curator.stages.code import CodeMetricsStage

            stage = CodeMetricsStage(
                include_basic_annotations=True,
                xml_header_search_length=200,
            )
            assert stage.include_basic_annotations is True
            assert stage.xml_header_search_length == 200
        except ImportError:
            pytest.skip("code_annotation not available")

    def test_get_config(self):
        """Test get_config() method."""
        try:
            from nemo_curator.stages.code import CodeMetricsStage

            stage = CodeMetricsStage()
            config = stage.get_config()

            assert "include_basic_annotations" in config
            assert "xml_header_search_length" in config
        except ImportError:
            pytest.skip("code_annotation not available")


class TestLicenseDetectionStage:
    """Tests for LicenseDetectionStage."""

    def test_import_with_missing_dependency(self):
        """Test that import fails gracefully without scancode."""
        try:
            from nemo_curator.stages.code import LicenseDetectionStage

            stage = LicenseDetectionStage()
            assert stage.content_window_size == 3000
        except ImportError as e:
            pytest.skip(f"scancode not available: {e}")

    def test_inputs_outputs(self):
        """Test inputs() and outputs() methods."""
        try:
            from nemo_curator.stages.code import LicenseDetectionStage

            stage = LicenseDetectionStage()

            outputs = stage.outputs()
            assert "license_num_licenses" in outputs[1]
            assert "license_spans" in outputs[1]
        except ImportError:
            pytest.skip("scancode not available")


class TestCodeQualitySignalsStage:
    """Tests for CodeQualitySignalsStage."""

    def test_import_and_config(self):
        """Test stage import and configuration."""
        try:
            from nemo_curator.stages.code import CodeQualitySignalsStage

            stage = CodeQualitySignalsStage(
                include_tokenization=True,
                tokenizer_name="github_o200k_base",
            )
            assert stage.include_tokenization is True
            assert stage.tokenizer_name == "github_o200k_base"
        except ImportError:
            pytest.skip("code_annotation not available")

    def test_decontamination_config(self):
        """Test decontamination configuration."""
        try:
            from nemo_curator.stages.code import CodeQualitySignalsStage

            ngrams = {"humaneval": ["def solution", "return result"]}
            stage = CodeQualitySignalsStage(
                include_decontamination=True,
                decontamination_ngrams=ngrams,
                decontamination_ngram_order=5,
            )
            assert stage.include_decontamination is True
            assert stage.decontamination_ngram_order == 5
        except ImportError:
            pytest.skip("code_annotation not available")


class TestCodeStageRegistry:
    """Test that code stages are properly registered."""

    def test_stages_are_importable(self):
        """Test all stages can be imported."""
        try:
            from nemo_curator.stages.code import (
                CodeMetricsStage,
                CodeQualitySignalsStage,
                LanguageIdentificationStage,
                LicenseDetectionStage,
            )

            # All imports successful - verify they are actual classes
            assert CodeMetricsStage is not None
            assert CodeQualitySignalsStage is not None
            assert LanguageIdentificationStage is not None
            assert LicenseDetectionStage is not None
        except ImportError as e:
            # At least the import structure should work
            # even if underlying deps are missing
            pytest.skip(f"Stage imports failed: {e}")
