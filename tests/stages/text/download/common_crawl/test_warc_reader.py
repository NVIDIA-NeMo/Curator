# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import gzip
import sys
from unittest import mock

import pandas as pd
import pytest

from nemo_curator.stages.text.download.common_crawl.download import CommonCrawlWARCReader


class TestTransportSelection:
    """Test transport selection defaults and env-var behavior."""

    @pytest.mark.parametrize("env_value", ["1", "true", "yes", "TRUE"])
    def test_cc_use_s3_env_enables_s3(self, monkeypatch: pytest.MonkeyPatch, env_value: str) -> None:
        monkeypatch.setenv("CC_USE_S3", env_value)

        reader = CommonCrawlWARCReader()

        assert reader.use_s3 is True

    def test_use_s3_without_boto3_raises_clear_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "boto3", None)
        reader = CommonCrawlWARCReader(use_s3=True)
        reader._s3_client = None

        with pytest.raises(RuntimeError, match="boto3 is not installed"):
            reader._get_s3_client()


class TestS3KeyFromFilename:
    """Test _s3_key_from_filename prefix-stripping logic."""

    @pytest.mark.parametrize(
        ("prefix", "filename", "expected"),
        [
            ("crawl-data/", "crawl-data/CC-MAIN-2024-10/seg/warc.gz", "CC-MAIN-2024-10/seg/warc.gz"),
            ("crawl-data/", "other-path/file.gz", "other-path/file.gz"),
            ("", "crawl-data/CC-MAIN-2024-10/seg/warc.gz", "crawl-data/CC-MAIN-2024-10/seg/warc.gz"),
        ],
        ids=["strip-prefix", "prefix-no-match", "empty-prefix-passthrough"],
    )
    def test_s3_key_from_filename(self, prefix: str, filename: str, expected: str) -> None:
        reader = CommonCrawlWARCReader(use_s3=False, s3_key_prefix=prefix)
        assert reader._s3_key_from_filename(filename) == expected


class TestReadWarcRecordS3:
    """Test S3-based WARC record fetching."""

    @staticmethod
    def _make_gzipped_warc() -> bytes:
        """Build minimal gzip-compressed WARC response record."""
        warc_record = "WARC/1.0\r\nWARC-Type: response\r\nContent-Length: 11\r\n\r\nHello World"
        return gzip.compress(warc_record.encode("utf-8"))

    def test_read_warc_record_s3_default_bucket(self) -> None:
        """S3 fetch uses the default 'commoncrawl' bucket when none is specified."""
        reader = CommonCrawlWARCReader(use_s3=True)
        assert reader.s3_bucket == "commoncrawl"

        raw_gz = self._make_gzipped_warc()
        mock_body = mock.Mock()
        mock_body.read.return_value = raw_gz
        mock_client = mock.Mock()
        mock_client.get_object.return_value = {"Body": mock_body}
        reader._s3_client = mock_client

        warc_filename = "crawl-data/CC-MAIN-2024-10/seg/warc.gz"
        row = pd.Series(
            {
                "warc_filename": warc_filename,
                "warc_record_offset": 100,
                "warc_record_length": len(raw_gz),
            }
        )

        result = reader._read_warc_record_s3(row)

        assert result is not None
        mock_client.get_object.assert_called_once_with(
            Bucket="commoncrawl",
            Key=warc_filename,
            Range=f"bytes=100-{100 + len(raw_gz) - 1}",
        )

    def test_read_warc_record_s3_custom_bucket(self) -> None:
        """S3 fetch uses the specified custom bucket."""
        reader = CommonCrawlWARCReader(use_s3=True, s3_bucket="my-bucket", s3_key_prefix="crawl-data/")
        raw_gz = self._make_gzipped_warc()

        mock_body = mock.Mock()
        mock_body.read.return_value = raw_gz
        mock_client = mock.Mock()
        mock_client.get_object.return_value = {"Body": mock_body}
        reader._s3_client = mock_client

        row = pd.Series(
            {
                "warc_filename": "crawl-data/CC-MAIN-2024-10/seg/warc.gz",
                "warc_record_offset": 0,
                "warc_record_length": len(raw_gz),
            }
        )

        result = reader._read_warc_record_s3(row)

        assert result is not None
        mock_client.get_object.assert_called_once_with(
            Bucket="my-bucket",
            Key="CC-MAIN-2024-10/seg/warc.gz",
            Range=f"bytes=0-{len(raw_gz) - 1}",
        )

    def test_read_warc_record_s3_failure(self) -> None:
        """S3 exception returns None and does not raise."""
        reader = CommonCrawlWARCReader(use_s3=True)

        mock_client = mock.Mock()
        mock_client.get_object.side_effect = Exception("AccessDenied")
        reader._s3_client = mock_client

        row = pd.Series(
            {
                "warc_filename": "CC-MAIN-2024-10/seg/warc.gz",
                "warc_record_offset": 0,
                "warc_record_length": 100,
            }
        )

        result = reader._read_warc_record_s3(row)
        assert result is None


class TestReadWarcRecordHTTPS:
    """Test HTTPS-based WARC record fetching."""

    def test_read_warc_record_https(self) -> None:
        """Successful HTTPS range-request fetch returns content."""
        reader = CommonCrawlWARCReader(use_s3=False)

        warc_payload = "WARC/1.0\r\nWARC-Type: response\r\nContent-Length: 11\r\n\r\nHello World"
        raw_gz = gzip.compress(warc_payload.encode("utf-8"))

        mock_response = mock.Mock()
        mock_response.status_code = 206
        mock_response.content = raw_gz

        mock_session = mock.Mock()
        mock_session.get.return_value = mock_response
        reader._session = mock_session

        row = pd.Series(
            {
                "warc_filename": "crawl-data/CC-MAIN-2024-10/seg/warc.gz",
                "warc_record_offset": 0,
                "warc_record_length": len(raw_gz),
            }
        )

        result = reader._read_warc_record(row)

        assert result is not None
        mock_session.get.assert_called_once_with(
            "https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-10/seg/warc.gz",
            headers={"Range": f"bytes=0-{len(raw_gz) - 1}"},
            timeout=30,
        )
