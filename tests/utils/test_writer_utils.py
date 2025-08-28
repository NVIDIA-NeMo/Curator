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

import csv
import json
import pathlib
import tempfile
import uuid
from unittest.mock import patch

import pandas as pd
import pytest

from nemo_curator.utils.writer_utils import (
    JsonEncoderCustom,
    write_bytes,
    write_csv,
    write_json,
    write_parquet,
)


class TestJsonEncoderCustom:
    """Test suite for JsonEncoderCustom class."""

    def test_uuid_serialization(self) -> None:
        """Test UUID serialization to string."""
        test_uuid = uuid.uuid4()
        encoder = JsonEncoderCustom()

        result = encoder.default(test_uuid)

        assert result == str(test_uuid)
        assert isinstance(result, str)

    def test_regular_object_serialization(self) -> None:
        """Test that regular objects fall back to parent implementation."""
        encoder = JsonEncoderCustom()

        # This should raise TypeError for non-serializable objects
        with pytest.raises(TypeError):
            encoder.default(object())

    def test_json_dumps_with_uuid(self) -> None:
        """Test json.dumps with UUID using custom encoder."""
        test_uuid = uuid.uuid4()
        data = {"id": test_uuid, "name": "test"}

        result = json.dumps(data, cls=JsonEncoderCustom)
        parsed = json.loads(result)

        assert parsed["id"] == str(test_uuid)
        assert parsed["name"] == "test"


class TestWriteBytes:
    """Test suite for write_bytes function."""

    def test_write_bytes_to_local_path_new_file(self) -> None:
        """Test writing bytes to a new local file."""
        test_data = b"test data"

        with tempfile.TemporaryDirectory() as tmp_dir:
            dest_path = pathlib.Path(tmp_dir) / "test.txt"

            write_bytes(
                test_data,
                dest_path,
                "test file",
                "test_video.mp4",
                verbose=False,
            )

            assert dest_path.exists()
            assert dest_path.read_bytes() == test_data

    def test_write_bytes_to_local_path_existing_file_skip(self) -> None:
        """Test writing bytes to existing local file without overwrite."""
        test_data = b"test data"
        existing_data = b"existing data"

        with tempfile.TemporaryDirectory() as tmp_dir:
            dest_path = pathlib.Path(tmp_dir) / "test.txt"
            dest_path.write_bytes(existing_data)

            with patch("nemo_curator.utils.writer_utils.logger") as mock_logger:
                write_bytes(
                    test_data,
                    dest_path,
                    "test file",
                    "test_video.mp4",
                    verbose=False,
                )

                mock_logger.warning.assert_called_once()
                assert "already exists, skipping" in mock_logger.warning.call_args[0][0]

            # File should remain unchanged
            assert dest_path.read_bytes() == existing_data

    def test_write_bytes_to_local_path_existing_file_overwrite(self) -> None:
        """Test writing bytes to existing local file with overwrite."""
        test_data = b"test data"
        existing_data = b"existing data"

        with tempfile.TemporaryDirectory() as tmp_dir:
            dest_path = pathlib.Path(tmp_dir) / "test.txt"
            dest_path.write_bytes(existing_data)

            with patch("nemo_curator.utils.writer_utils.logger") as mock_logger:
                write_bytes(
                    test_data,
                    dest_path,
                    "test file",
                    "test_video.mp4",
                    verbose=False,
                    overwrite=True,
                )

                mock_logger.warning.assert_called_once()
                assert "already exists, overwriting" in mock_logger.warning.call_args[0][0]

            # File should be overwritten
            assert dest_path.read_bytes() == test_data

    def test_write_bytes_to_local_path_backup_and_overwrite_not_implemented(self) -> None:
        """Test that backup_and_overwrite raises NotImplementedError for local paths."""
        test_data = b"test data"

        with tempfile.TemporaryDirectory() as tmp_dir:
            dest_path = pathlib.Path(tmp_dir) / "test.txt"
            dest_path.write_bytes(b"existing")

            with pytest.raises(NotImplementedError, match="Backup and overwrite is not implemented"):
                write_bytes(
                    test_data,
                    dest_path,
                    "test file",
                    "test_video.mp4",
                    verbose=False,
                    backup_and_overwrite=True,
                )

    def test_write_bytes_to_local_path_creates_directories(self) -> None:
        """Test that write_bytes creates parent directories."""
        test_data = b"test data"

        with tempfile.TemporaryDirectory() as tmp_dir:
            dest_path = pathlib.Path(tmp_dir) / "subdir" / "nested" / "test.txt"

            write_bytes(
                test_data,
                dest_path,
                "test file",
                "test_video.mp4",
                verbose=False,
            )

            assert dest_path.exists()
            assert dest_path.read_bytes() == test_data

    def test_write_bytes_to_local_path_verbose(self) -> None:
        """Test verbose logging for local path writes."""
        test_data = b"test data"

        with tempfile.TemporaryDirectory() as tmp_dir:
            dest_path = pathlib.Path(tmp_dir) / "test.txt"

            with patch("nemo_curator.utils.writer_utils.logger") as mock_logger:
                write_bytes(
                    test_data,
                    dest_path,
                    "test file",
                    "test_video.mp4",
                    verbose=True,
                )

                mock_logger.info.assert_called_once()
                assert "Writing test file for test_video.mp4" in mock_logger.info.call_args[0][0]


class TestWriteParquet:
    """Test suite for write_parquet function."""

    def test_write_parquet_to_local_path(self) -> None:
        """Test writing parquet data to local path."""
        test_data = [
            {"name": "Alice", "age": "30"},
            {"name": "Bob", "age": "25"},
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            dest_path = pathlib.Path(tmp_dir) / "test.parquet"

            write_parquet(
                test_data,
                dest_path,
                "test parquet",
                "test_video.mp4",
                verbose=False,
            )

            assert dest_path.exists()
            # Read back and verify
            df = pd.read_parquet(dest_path)
            assert len(df) == 2
            assert df.iloc[0]["name"] == "Alice"
            assert df.iloc[1]["name"] == "Bob"

    def test_write_parquet_empty_data(self) -> None:
        """Test writing empty parquet data."""
        test_data = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            dest_path = pathlib.Path(tmp_dir) / "empty.parquet"

            write_parquet(
                test_data,
                dest_path,
                "empty parquet",
                "test_video.mp4",
                verbose=False,
            )

            assert dest_path.exists()
            df = pd.read_parquet(dest_path)
            assert len(df) == 0


class TestWriteJson:
    """Test suite for write_json function."""

    def test_write_json_to_local_path(self) -> None:
        """Test writing JSON data to local path."""
        test_data = {"name": "Alice", "age": 30, "active": True}

        with tempfile.TemporaryDirectory() as tmp_dir:
            dest_path = pathlib.Path(tmp_dir) / "test.json"

            write_json(
                test_data,
                dest_path,
                "test json",
                "test_video.mp4",
                verbose=False,
            )

            assert dest_path.exists()
            # Read back and verify
            with dest_path.open("r") as f:
                loaded_data = json.load(f)
            assert loaded_data == test_data

    def test_write_json_with_uuid(self) -> None:
        """Test writing JSON data with UUID."""
        test_uuid = uuid.uuid4()
        test_data = {"id": test_uuid, "name": "test"}

        with tempfile.TemporaryDirectory() as tmp_dir:
            dest_path = pathlib.Path(tmp_dir) / "test.json"

            write_json(
                test_data,
                dest_path,
                "test json",
                "test_video.mp4",
                verbose=False,
            )

            assert dest_path.exists()
            # Read back and verify UUID was serialized as string
            with dest_path.open("r") as f:
                loaded_data = json.load(f)
            assert loaded_data["id"] == str(test_uuid)
            assert loaded_data["name"] == "test"

    def test_write_json_formatting(self) -> None:
        """Test that JSON is properly formatted with indentation."""
        test_data = {"name": "Alice", "nested": {"key": "value"}}

        with tempfile.TemporaryDirectory() as tmp_dir:
            dest_path = pathlib.Path(tmp_dir) / "test.json"

            write_json(
                test_data,
                dest_path,
                "test json",
                "test_video.mp4",
                verbose=False,
            )

            content = dest_path.read_text()
            # Check that it's properly indented
            assert "    " in content  # 4-space indentation
            assert "{\n" in content


class TestWriteCsv:
    """Test suite for write_csv function."""

    def test_write_csv_to_local_path(self) -> None:
        """Test writing CSV data to local path."""
        test_data = [
            ["name", "age"],
            ["Alice", "30"],
            ["Bob", "25"],
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            dest_path = pathlib.Path(tmp_dir) / "test.csv"

            write_csv(
                dest_path,
                "test csv",
                "test_video.mp4",
                test_data,
                verbose=False,
            )

            assert dest_path.exists()
            # Read back and verify
            with dest_path.open("r") as f:
                reader = csv.reader(f)
                rows = list(reader)
            assert rows == test_data

    def test_write_csv_empty_data(self) -> None:
        """Test writing empty CSV data."""
        test_data = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            dest_path = pathlib.Path(tmp_dir) / "empty.csv"

            write_csv(
                dest_path,
                "empty csv",
                "test_video.mp4",
                test_data,
                verbose=False,
            )

            assert dest_path.exists()
            with dest_path.open("r") as f:
                reader = csv.reader(f)
                rows = list(reader)
            assert rows == []

    def test_write_csv_special_characters(self) -> None:
        """Test writing CSV data with special characters."""
        test_data = [
            ["field1", "field2"],
            ["value,with,commas", "value\nwith\nnewlines"],
            ['"quoted"', "normal"],
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            dest_path = pathlib.Path(tmp_dir) / "special.csv"

            write_csv(
                dest_path,
                "special csv",
                "test_video.mp4",
                test_data,
                verbose=False,
            )

            assert dest_path.exists()
            # Read back and verify CSV handling of special characters
            with dest_path.open("r") as f:
                reader = csv.reader(f)
                rows = list(reader)
            assert rows == test_data
