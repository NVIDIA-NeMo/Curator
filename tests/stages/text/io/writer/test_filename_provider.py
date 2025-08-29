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

import os
import uuid
from unittest import mock

import pandas as pd
import pytest

from nemo_curator.stages.text.io.writer import (
    DatasetNameFilenameProvider,
    DefaultFilenameProvider,
    JsonlWriter,
    ParquetWriter,
    TemplateFilenameProvider,
)
from nemo_curator.stages.text.io.writer import utils as writer_utils
from nemo_curator.tasks import DocumentBatch


class TestFilenameProviders:
    """Test suite for FilenameProvider implementations."""

    def test_default_filename_provider_with_source_files(self, pandas_document_batch: DocumentBatch):
        """Test DefaultFilenameProvider with source files in metadata."""
        provider = DefaultFilenameProvider()
        
        # Add source files to metadata
        source_files = ["file1.jsonl", "file2.jsonl"]
        pandas_document_batch._metadata["source_files"] = source_files
        
        with mock.patch.object(
            writer_utils, "get_deterministic_hash", return_value="test_hash"
        ) as mock_hash:
            filename = provider.get_filename(pandas_document_batch, "jsonl")
            
            assert filename == "test_hash.jsonl"
            mock_hash.assert_called_once_with(source_files, pandas_document_batch.task_id)

    def test_default_filename_provider_without_source_files(self, pandas_document_batch: DocumentBatch):
        """Test DefaultFilenameProvider without source files in metadata."""
        provider = DefaultFilenameProvider()
        
        # Ensure no source files in metadata
        pandas_document_batch._metadata.pop("source_files", None)
        
        with mock.patch.object(uuid, "uuid4", return_value=mock.Mock(hex="mock_uuid")) as mock_uuid:
            filename = provider.get_filename(pandas_document_batch, "jsonl")
            
            assert filename == "mock_uuid.jsonl"
            mock_uuid.assert_called_once()

    def test_dataset_name_filename_provider_with_dataset_name(self, pandas_document_batch: DocumentBatch):
        """Test DatasetNameFilenameProvider with dataset name."""
        provider = DatasetNameFilenameProvider()
        
        # Set dataset name
        pandas_document_batch.dataset_name = "my_dataset"
        pandas_document_batch._metadata["source_files"] = ["file1.jsonl"]
        
        with mock.patch.object(
            writer_utils, "get_deterministic_hash", return_value="test_hash"
        ):
            filename = provider.get_filename(pandas_document_batch, "jsonl")
            
            assert filename == "my_dataset_test_hash.jsonl"

    def test_dataset_name_filename_provider_without_dataset_name(self, pandas_document_batch: DocumentBatch):
        """Test DatasetNameFilenameProvider without dataset name."""
        provider = DatasetNameFilenameProvider()
        
        # Set dataset name to None
        pandas_document_batch.dataset_name = None
        pandas_document_batch._metadata["source_files"] = ["file1.jsonl"]
        
        with mock.patch.object(
            writer_utils, "get_deterministic_hash", return_value="test_hash"
        ):
            filename = provider.get_filename(pandas_document_batch, "jsonl")
            
            assert filename == "test_hash.jsonl"

    def test_template_filename_provider_basic(self, pandas_document_batch: DocumentBatch):
        """Test TemplateFilenameProvider with basic template."""
        provider = TemplateFilenameProvider(template="{dataset_name}_{hash}")
        
        pandas_document_batch.dataset_name = "my_dataset"
        pandas_document_batch._metadata["source_files"] = ["file1.jsonl"]
        
        with mock.patch.object(
            writer_utils, "get_deterministic_hash", return_value="test_hash"
        ):
            filename = provider.get_filename(pandas_document_batch, "jsonl")
            
            assert filename == "my_dataset_test_hash.jsonl"

    def test_template_filename_provider_all_variables(self, pandas_document_batch: DocumentBatch):
        """Test TemplateFilenameProvider with all template variables."""
        provider = TemplateFilenameProvider(
            template="{dataset_name}_{task_id}_{hash}.{extension}"
        )
        
        pandas_document_batch.dataset_name = "my_dataset"
        pandas_document_batch._metadata["source_files"] = ["file1.jsonl"]
        
        with mock.patch.object(
            writer_utils, "get_deterministic_hash", return_value="test_hash"
        ):
            filename = provider.get_filename(pandas_document_batch, "jsonl")
            
            expected = f"my_dataset_{pandas_document_batch.task_id}_test_hash.jsonl"
            assert filename == expected

    def test_template_filename_provider_invalid_variable(self, pandas_document_batch: DocumentBatch):
        """Test TemplateFilenameProvider with invalid template variable."""
        provider = TemplateFilenameProvider(template="{invalid_variable}")
        
        with pytest.raises(ValueError, match="Invalid template variable"):
            provider.get_filename(pandas_document_batch, "jsonl")

    def test_template_filename_provider_no_dataset_name(self, pandas_document_batch: DocumentBatch):
        """Test TemplateFilenameProvider when dataset name is None."""
        provider = TemplateFilenameProvider(template="{dataset_name}_{hash}")
        
        pandas_document_batch.dataset_name = None
        pandas_document_batch._metadata["source_files"] = ["file1.jsonl"]
        
        with mock.patch.object(
            writer_utils, "get_deterministic_hash", return_value="test_hash"
        ):
            filename = provider.get_filename(pandas_document_batch, "jsonl")
            
            assert filename == "unknown_test_hash.jsonl"


class TestWritersWithFilenameProviders:
    """Test writers using custom filename providers."""

    def test_jsonl_writer_with_dataset_name_provider(
        self, pandas_document_batch: DocumentBatch, tmpdir: str
    ):
        """Test JsonlWriter with DatasetNameFilenameProvider."""
        output_dir = os.path.join(tmpdir, "jsonl_dataset_name")
        provider = DatasetNameFilenameProvider()
        writer = JsonlWriter(path=output_dir, filename_provider=provider)
        
        # Set dataset name
        pandas_document_batch.dataset_name = "test_dataset"
        
        writer.setup()
        result = writer.process(pandas_document_batch)
        
        # Verify the filename contains the dataset name
        file_path = result.data[0]
        filename = os.path.basename(file_path)
        assert filename.startswith("test_dataset_")
        assert filename.endswith(".jsonl")
        
        # Verify file exists and has correct content
        assert os.path.exists(file_path)
        df = pd.read_json(file_path, lines=True)
        pd.testing.assert_frame_equal(df, pandas_document_batch.to_pandas())

    def test_parquet_writer_with_template_provider(
        self, pandas_document_batch: DocumentBatch, tmpdir: str
    ):
        """Test ParquetWriter with TemplateFilenameProvider."""
        output_dir = os.path.join(tmpdir, "parquet_template")
        provider = TemplateFilenameProvider(template="data_{dataset_name}_{task_id}")
        writer = ParquetWriter(path=output_dir, filename_provider=provider)
        
        # Set dataset name
        pandas_document_batch.dataset_name = "my_data"
        
        writer.setup()
        result = writer.process(pandas_document_batch)
        
        # Verify the filename follows the template
        file_path = result.data[0]
        filename = os.path.basename(file_path)
        expected_start = f"data_my_data_{pandas_document_batch.task_id}"
        assert filename.startswith(expected_start)
        assert filename.endswith(".parquet")
        
        # Verify file exists and has correct content
        assert os.path.exists(file_path)
        df = pd.read_parquet(file_path)
        pd.testing.assert_frame_equal(df, pandas_document_batch.to_pandas())

    def test_writer_with_custom_extension_and_template(
        self, pandas_document_batch: DocumentBatch, tmpdir: str
    ):
        """Test writer with custom extension and template provider."""
        output_dir = os.path.join(tmpdir, "custom_ext_template")
        provider = TemplateFilenameProvider(template="{dataset_name}_custom.{extension}")
        writer = JsonlWriter(
            path=output_dir, 
            filename_provider=provider,
            file_extension="ndjson"
        )
        
        # Set dataset name
        pandas_document_batch.dataset_name = "test_data"
        
        writer.setup()
        result = writer.process(pandas_document_batch)
        
        # Verify the filename follows the template with custom extension
        file_path = result.data[0]
        filename = os.path.basename(file_path)
        assert filename == "test_data_custom.ndjson"
        
        # Verify file exists and has correct content
        assert os.path.exists(file_path)
        df = pd.read_json(file_path, lines=True)
        pd.testing.assert_frame_equal(df, pandas_document_batch.to_pandas())

    def test_backward_compatibility_no_filename_provider(
        self, pandas_document_batch: DocumentBatch, tmpdir: str
    ):
        """Test that writers work as before when no filename provider is specified."""
        output_dir = os.path.join(tmpdir, "backward_compat")
        writer = JsonlWriter(path=output_dir)  # No filename provider
        
        writer.setup()
        
        # Should use DefaultFilenameProvider automatically
        assert isinstance(writer.filename_provider, DefaultFilenameProvider)
        
        result = writer.process(pandas_document_batch)
        
        # Verify file was created and works correctly
        file_path = result.data[0]
        assert os.path.exists(file_path)
        assert file_path.endswith(".jsonl")
        
        df = pd.read_json(file_path, lines=True)
        pd.testing.assert_frame_equal(df, pandas_document_batch.to_pandas())