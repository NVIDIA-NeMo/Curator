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

from loguru import logger

from nemo_curator.stages.text.download.common_crawl.warc_iterator import CommonCrawlWarcIterator


class TestCommonCrawlWarcIterator:
    """Test suite for CommonCrawlWarcIterator - focused on core logic correctness."""

    def test_stop_iteration_handling(self, tmp_path: Path) -> None:
        """Test that StopIteration is handled correctly when archive iterator is exhausted."""
        raw_warc_path = tmp_path / "test.warc"

        # Create a minimal valid WARC file with one response record
        http_response = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n<html><body>Test</body></html>\r\n"
        http_response_bytes = http_response.encode("utf-8")
        content_length = len(http_response_bytes)
        warc_record = (
            (
                f"WARC/1.0\r\n"
                f"WARC-Type: response\r\n"
                f"WARC-Record-ID: <urn:uuid:test123>\r\n"
                f"WARC-Date: 2022-01-01T00:00:00Z\r\n"
                f"WARC-Target-URI: http://example.com\r\n"
                f"Content-Length: {content_length}\r\n"
                f"\r\n"
            ).encode()
            + http_response_bytes
            + b"\r\n\r\n"
        )
        raw_warc_path.write_bytes(warc_record)

        iterator = CommonCrawlWarcIterator()

        # The iterate method should handle StopIteration internally and complete gracefully
        records = list(iterator.iterate(str(raw_warc_path)))

        # Should successfully extract the one record without raising StopIteration
        assert len(records) == 1
        assert records[0]["warc_id"] == "test123"
        assert records[0]["url"] == "http://example.com"

    def test_error_processing_record_continues(self, tmp_path: Path) -> None:
        """Test that exceptions during record processing are logged and processing continues."""
        raw_warc_path = tmp_path / "test.warc"

        # Create a WARC file with a response record that has no WARC-Record-ID header
        # This will cause the get_header to return None, leading to the subscriptable error
        http_response = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n<html><body>Test</body></html>\r\n"
        http_response_bytes = http_response.encode("utf-8")
        content_length = len(http_response_bytes)
        # Intentionally missing WARC-Record-ID to cause the error
        warc_record = (
            (
                f"WARC/1.0\r\n"
                f"WARC-Type: response\r\n"
                f"WARC-Date: 2022-01-01T00:00:00Z\r\n"
                f"WARC-Target-URI: http://example.com\r\n"
                f"Content-Length: {content_length}\r\n"
                f"\r\n"
            ).encode()
            + http_response_bytes
            + b"\r\n\r\n"
        )
        raw_warc_path.write_bytes(warc_record)

        iterator = CommonCrawlWarcIterator()

        with mock.patch.object(logger, "error") as mock_logger:
            extracted_records = list(iterator.iterate(str(raw_warc_path)))
            # Should have logged an error about NoneType not being subscriptable
            mock_logger.assert_called_once_with(
                "Error processing record 0 in test.warc: 'NoneType' object is not subscriptable"
            )
            # Should not have extracted any records due to the error
            assert len(extracted_records) == 0

    def test_mixed_record_types_response_only_with_correct_values(self, tmp_path: Path) -> None:
        """Test with mixed record types - only response records should be output with correct values."""
        raw_warc_path = tmp_path / "mixed_types.warc"

        # Create WARC file with multiple record types in sequence
        records = []

        # Define record configurations
        record_configs = [
            {
                "type": "warcinfo",
                "content": "software: test-crawler/1.0\r\n",
                "id": "warcinfo123",
                "target_uri": None,  # warcinfo doesn't have target URI
            },
            {
                "type": "request",
                "content": "GET /page HTTP/1.1\r\nHost: example.com\r\n\r\n",
                "id": "request123",
                "target_uri": "http://example.com/page",
            },
            {
                "type": "response",
                "content": "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n<html><body><h1>Test Page</h1></body></html>\r\n",
                "id": "response123",
                "target_uri": "http://example.com/page",
            },
            {
                "type": "metadata",
                "content": '{"url": "http://example.com", "status": "crawled"}',
                "id": "metadata123",
                "target_uri": "http://example.com/page",
            },
        ]

        # The HTML content that will actually be extracted (just the body, not HTTP headers)
        html_content = b"<html><body><h1>Test Page</h1></body></html>\r\n"

        # Generate all records in a loop
        for config in record_configs:
            content_bytes = config["content"].encode("utf-8")
            content_length = len(content_bytes)

            # Build WARC header
            header_parts = [
                "WARC/1.0\r\n",
                f"WARC-Type: {config['type']}\r\n",
                f"WARC-Record-ID: <urn:uuid:{config['id']}>\r\n",
                "WARC-Date: 2022-01-01T00:00:00Z\r\n",
            ]

            if config["target_uri"]:
                header_parts.append(f"WARC-Target-URI: {config['target_uri']}\r\n")

            header_parts.append(f"Content-Length: {content_length}\r\n\r\n")

            warc_record = "".join(header_parts).encode() + content_bytes + b"\r\n\r\n"
            records.append(warc_record)

        # Write all records to file
        raw_warc_path.write_bytes(b"".join(records))

        iterator = CommonCrawlWarcIterator()
        extracted_records = list(iterator.iterate(str(raw_warc_path)))

        # Should only extract the one response record
        assert len(extracted_records) == 1

        record = extracted_records[0]
        # Verify all values are correctly extracted
        assert record["url"] == "http://example.com/page"
        assert record["warc_id"] == "response123"  # Stripped <urn:uuid: and >
        assert record["source_id"] == "mixed_types.warc"
        # The content should be just the HTML body (warcio extracts body from HTTP response)
        assert record["content"] == html_content

        # Verify the content contains expected HTML
        assert b"<h1>Test Page</h1>" in record["content"]

    def test_output_columns(self) -> None:
        """Test that output_columns returns the expected column names."""
        iterator = CommonCrawlWarcIterator()
        columns = iterator.output_columns()

        expected_columns = ["url", "warc_id", "source_id", "content"]
        assert columns == expected_columns
