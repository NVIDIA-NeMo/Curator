"""Test suite for writer_utils module."""

import csv
import io
import json
import pathlib
import tempfile
import uuid
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from ray_curator.utils import storage_client
from ray_curator.utils.writer_utils import (
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
                client=None,
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

            with patch("ray_curator.utils.writer_utils.logger") as mock_logger:
                write_bytes(
                    test_data,
                    dest_path,
                    "test file",
                    "test_video.mp4",
                    verbose=False,
                    client=None,
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

            with patch("ray_curator.utils.writer_utils.logger") as mock_logger:
                write_bytes(
                    test_data,
                    dest_path,
                    "test file",
                    "test_video.mp4",
                    verbose=False,
                    client=None,
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
                    client=None,
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
                client=None,
            )

            assert dest_path.exists()
            assert dest_path.read_bytes() == test_data

    def test_write_bytes_to_local_path_verbose(self) -> None:
        """Test verbose logging for local path writes."""
        test_data = b"test data"

        with tempfile.TemporaryDirectory() as tmp_dir:
            dest_path = pathlib.Path(tmp_dir) / "test.txt"

            with patch("ray_curator.utils.writer_utils.logger") as mock_logger:
                write_bytes(
                    test_data,
                    dest_path,
                    "test file",
                    "test_video.mp4",
                    verbose=True,
                    client=None,
                )

                mock_logger.info.assert_called_once()
                assert "Writing test file for test_video.mp4" in mock_logger.info.call_args[0][0]

    def test_write_bytes_to_storage_prefix_new_file(self) -> None:
        """Test writing bytes to storage prefix for new file."""
        test_data = b"test data"
        mock_client = Mock()
        mock_client.object_exists.return_value = False

        mock_prefix = Mock(spec=storage_client.StoragePrefix)
        mock_prefix.path = "s3://bucket/test.txt"

        with patch("ray_curator.utils.writer_utils.do_with_retries") as mock_retry:
            write_bytes(
                test_data,
                mock_prefix,
                "test file",
                "test_video.mp4",
                verbose=False,
                client=mock_client,
            )

            mock_client.object_exists.assert_called_once_with(mock_prefix)
            mock_retry.assert_called_once()
            # Check that the function passed to do_with_retries calls upload_bytes
            retry_func = mock_retry.call_args[0][0]
            retry_func()
            mock_client.upload_bytes.assert_called_once_with(mock_prefix, test_data)

    def test_write_bytes_to_storage_prefix_existing_file_skip(self) -> None:
        """Test writing bytes to existing storage prefix without overwrite."""
        test_data = b"test data"
        mock_client = Mock()
        mock_client.object_exists.return_value = True

        mock_prefix = Mock(spec=storage_client.StoragePrefix)
        mock_prefix.path = "s3://bucket/test.txt"

        with patch("ray_curator.utils.writer_utils.logger") as mock_logger:
            write_bytes(
                test_data,
                mock_prefix,
                "test file",
                "test_video.mp4",
                verbose=False,
                client=mock_client,
            )

            mock_client.object_exists.assert_called_once_with(mock_prefix)
            mock_client.upload_bytes.assert_not_called()
            mock_logger.warning.assert_called_once()
            assert "already exists, skipping" in mock_logger.warning.call_args[0][0]

    def test_write_bytes_to_storage_prefix_existing_file_overwrite(self) -> None:
        """Test writing bytes to existing storage prefix with overwrite."""
        test_data = b"test data"
        mock_client = Mock()
        mock_client.object_exists.return_value = True

        mock_prefix = Mock(spec=storage_client.StoragePrefix)
        mock_prefix.path = "s3://bucket/test.txt"

        with (
            patch("ray_curator.utils.writer_utils.do_with_retries") as mock_retry,
            patch("ray_curator.utils.writer_utils.logger") as mock_logger,
        ):
            write_bytes(
                test_data,
                mock_prefix,
                "test file",
                "test_video.mp4",
                verbose=False,
                client=mock_client,
                overwrite=True,
            )

            mock_client.object_exists.assert_called_once_with(mock_prefix)
            mock_retry.assert_called_once()
            mock_logger.warning.assert_called_once()
            assert "already exists, overwriting" in mock_logger.warning.call_args[0][0]

    def test_write_bytes_to_storage_prefix_backup_and_overwrite_not_implemented(self) -> None:
        """Test that backup_and_overwrite raises NotImplementedError for storage prefix."""
        test_data = b"test data"
        mock_client = Mock()
        mock_client.object_exists.return_value = True

        mock_prefix = Mock(spec=storage_client.StoragePrefix)
        mock_prefix.path = "s3://bucket/test.txt"

        with pytest.raises(NotImplementedError, match="Backup and overwrite is not implemented"):
            write_bytes(
                test_data,
                mock_prefix,
                "test file",
                "test_video.mp4",
                verbose=False,
                client=mock_client,
                backup_and_overwrite=True,
            )

    def test_write_bytes_to_storage_prefix_no_client(self) -> None:
        """Test that storage prefix without client raises ValueError."""
        test_data = b"test data"
        mock_prefix = Mock(spec=storage_client.StoragePrefix)

        with pytest.raises(ValueError, match="S3 client is required for S3 destination"):
            write_bytes(
                test_data,
                mock_prefix,
                "test file",
                "test_video.mp4",
                verbose=False,
                client=None,
            )

    def test_write_bytes_to_storage_prefix_verbose(self) -> None:
        """Test verbose logging for storage prefix writes."""
        test_data = b"test data"
        mock_client = Mock()
        mock_client.object_exists.return_value = False

        mock_prefix = Mock(spec=storage_client.StoragePrefix)
        mock_prefix.path = "s3://bucket/test.txt"

        with (
            patch("ray_curator.utils.writer_utils.do_with_retries"),
            patch("ray_curator.utils.writer_utils.logger") as mock_logger,
        ):
            write_bytes(
                test_data,
                mock_prefix,
                "test file",
                "test_video.mp4",
                verbose=True,
                client=mock_client,
            )

            mock_logger.info.assert_called_once()
            assert "Uploading test file for test_video.mp4" in mock_logger.info.call_args[0][0]

    def test_write_bytes_unexpected_destination_type(self) -> None:
        """Test that unexpected destination type raises TypeError."""
        test_data = b"test data"

        with pytest.raises(TypeError, match="Unexpected destination type"):
            write_bytes(
                test_data,
                "invalid_destination",  # type: ignore[arg-type]
                "test file",
                "test_video.mp4",
                verbose=False,
                client=None,
            )


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
                client=None,
            )

            assert dest_path.exists()
            # Read back and verify
            df = pd.read_parquet(dest_path)
            assert len(df) == 2
            assert df.iloc[0]["name"] == "Alice"
            assert df.iloc[1]["name"] == "Bob"

    def test_write_parquet_to_storage_prefix(self) -> None:
        """Test writing parquet data to storage prefix."""
        test_data = [{"name": "Alice", "age": "30"}]
        mock_client = Mock()
        mock_client.object_exists.return_value = False

        mock_prefix = Mock(spec=storage_client.StoragePrefix)

        with patch("ray_curator.utils.writer_utils.write_bytes") as mock_write_bytes:
            write_parquet(
                test_data,
                mock_prefix,
                "test parquet",
                "test_video.mp4",
                verbose=False,
                client=mock_client,
            )

            mock_write_bytes.assert_called_once()
            args, kwargs = mock_write_bytes.call_args

            # Check that the first argument is bytes (parquet data)
            assert isinstance(args[0], bytes)
            assert args[1] == mock_prefix
            assert args[2] == "test parquet"
            assert args[3] == "test_video.mp4"

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
                client=None,
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
                client=None,
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
                client=None,
            )

            assert dest_path.exists()
            # Read back and verify UUID was serialized as string
            with dest_path.open("r") as f:
                loaded_data = json.load(f)
            assert loaded_data["id"] == str(test_uuid)
            assert loaded_data["name"] == "test"

    def test_write_json_to_storage_prefix(self) -> None:
        """Test writing JSON data to storage prefix."""
        test_data = {"name": "Alice", "age": 30}
        mock_client = Mock()
        mock_client.object_exists.return_value = False

        mock_prefix = Mock(spec=storage_client.StoragePrefix)

        with patch("ray_curator.utils.writer_utils.write_bytes") as mock_write_bytes:
            write_json(
                test_data,
                mock_prefix,
                "test json",
                "test_video.mp4",
                verbose=False,
                client=mock_client,
            )

            mock_write_bytes.assert_called_once()
            args, kwargs = mock_write_bytes.call_args

            # Check that the first argument is bytes (JSON data)
            assert isinstance(args[0], bytes)
            json_content = args[0].decode("utf-8")
            loaded_data = json.loads(json_content)
            assert loaded_data == test_data

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
                client=None,
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
                client=None,
            )

            assert dest_path.exists()
            # Read back and verify
            with dest_path.open("r") as f:
                reader = csv.reader(f)
                rows = list(reader)
            assert rows == test_data

    def test_write_csv_to_storage_prefix(self) -> None:
        """Test writing CSV data to storage prefix."""
        test_data = [["name", "age"], ["Alice", "30"]]
        mock_client = Mock()
        mock_client.object_exists.return_value = False

        mock_prefix = Mock(spec=storage_client.StoragePrefix)

        with patch("ray_curator.utils.writer_utils.write_bytes") as mock_write_bytes:
            write_csv(
                mock_prefix,
                "test csv",
                "test_video.mp4",
                test_data,
                verbose=False,
                client=mock_client,
            )

            mock_write_bytes.assert_called_once()
            args, kwargs = mock_write_bytes.call_args

            # Check that the first argument is bytes (CSV data)
            assert isinstance(args[0], bytes)
            csv_content = args[0].decode("utf-8")
            reader = csv.reader(io.StringIO(csv_content))
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
                client=None,
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
                client=None,
            )

            assert dest_path.exists()
            # Read back and verify CSV handling of special characters
            with dest_path.open("r") as f:
                reader = csv.reader(f)
                rows = list(reader)
            assert rows == test_data
