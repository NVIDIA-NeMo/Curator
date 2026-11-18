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

"""Unit tests for the image download Curator stages.

Tests the core logic without requiring network access or a running Ray cluster.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nemo_curator.tasks import EmptyTask, FileGroupTask, _EmptyTask

from nemo_curator_stages.image.download.compliance.cache import InMemoryCache
from nemo_curator_stages.image.download.compliance.robots import can_fetch
from nemo_curator_stages.image.download.compliance.tdm import tdm_allowed
from nemo_curator_stages.image.download.manifest import ManifestReaderStage, validate_url


# --- URL Validation ---


class TestValidateUrl:
    def test_valid_https(self):
        assert validate_url("https://example.com/image.jpg") is True

    def test_valid_http(self):
        assert validate_url("http://example.com/image.jpg") is True

    def test_rejects_ftp(self):
        assert validate_url("ftp://example.com/file") is False

    def test_rejects_javascript(self):
        assert validate_url("javascript:alert(1)") is False

    def test_rejects_private_ip(self):
        assert validate_url("http://192.168.1.1/img.jpg") is False
        assert validate_url("http://10.0.0.1/img.jpg") is False
        assert validate_url("http://172.16.0.1/img.jpg") is False

    def test_rejects_loopback(self):
        assert validate_url("http://127.0.0.1/img.jpg") is False

    def test_allows_domain(self):
        assert validate_url("https://cdn.example.com/path/to/img.jpg") is True

    def test_rejects_no_host(self):
        assert validate_url("https:///path") is False

    def test_rejects_empty(self):
        assert validate_url("") is False


# --- Robots.txt ---


class TestRobotsCanFetch:
    def test_allowed_when_no_rules(self):
        robots = "User-agent: *\nAllow: /\n"
        assert can_fetch(robots, "my-bot", "https://example.com/image.jpg") is True

    def test_disallowed_by_rule(self):
        robots = "User-agent: *\nDisallow: /images/\n"
        assert can_fetch(robots, "my-bot", "https://example.com/images/photo.jpg") is False

    def test_none_conservative(self):
        assert can_fetch(None, "my-bot", "https://example.com/img.jpg", "conservative") is False

    def test_none_permissive(self):
        assert can_fetch(None, "my-bot", "https://example.com/img.jpg", "permissive") is True


# --- TDMRep ---


class TestTdmAllowed:
    def test_no_tdmrep_conservative(self):
        assert tdm_allowed(None, "https://example.com/img.jpg", "conservative") is False

    def test_no_tdmrep_permissive(self):
        assert tdm_allowed(None, "https://example.com/img.jpg", "permissive") is True

    def test_empty_policy(self):
        assert tdm_allowed({"policy": []}, "https://example.com/img.jpg") is True

    def test_allowed_rule(self):
        tdmrep = {"policy": [{"location": "/images/*", "permission": "allowed"}]}
        assert tdm_allowed(tdmrep, "https://example.com/images/photo.jpg") is True

    def test_not_allowed_rule(self):
        tdmrep = {"policy": [{"location": "/images/*", "permission": "not-allowed"}]}
        assert tdm_allowed(tdmrep, "https://example.com/images/photo.jpg") is False

    def test_first_match_wins(self):
        tdmrep = {
            "policy": [
                {"location": "/images/public/*", "permission": "allowed"},
                {"location": "/images/*", "permission": "not-allowed"},
            ]
        }
        assert tdm_allowed(tdmrep, "https://example.com/images/public/pic.jpg") is True
        assert tdm_allowed(tdmrep, "https://example.com/images/private/pic.jpg") is False


# --- InMemoryCache ---


class TestInMemoryCache:
    def test_set_and_get(self):
        cache = InMemoryCache()
        cache.set("key1", "value1", ttl=3600)
        assert cache.get("key1") == "value1"

    def test_missing_key(self):
        cache = InMemoryCache()
        assert cache.get("nonexistent") is None

    def test_expiry(self):
        cache = InMemoryCache()
        cache.set("key1", "value1", ttl=0)
        import time
        time.sleep(0.01)
        assert cache.get("key1") is None


# --- ManifestReaderStage ---


class TestManifestReaderStage:
    @pytest.fixture
    def manifest_file(self, tmp_path):
        """Create a test Parquet manifest."""
        df = pd.DataFrame({
            "url_id": [f"id_{i}" for i in range(10)],
            "asset_url": [f"https://example.com/img_{i}.jpg" for i in range(10)],
            "asset_mime": ["image/jpeg"] * 10,
            "target_path": [f"images/img_{i}.jpg" for i in range(10)],
            "source_ref": ["test"] * 10,
        })
        path = tmp_path / "manifest.parquet"
        table = pa.Table.from_pandas(df)
        pq.write_table(table, str(path))
        return str(path)

    def test_reads_all_urls(self, manifest_file):
        stage = ManifestReaderStage(manifest_path=manifest_file, urls_per_task=5)
        tasks = stage.process(EmptyTask)
        assert len(tasks) == 2  # 10 URLs / 5 per task
        assert all(isinstance(t, FileGroupTask) for t in tasks)
        assert len(tasks[0].data) == 5
        assert len(tasks[1].data) == 5

    def test_respects_limit(self, manifest_file):
        stage = ManifestReaderStage(manifest_path=manifest_file, urls_per_task=10, limit=3)
        tasks = stage.process(EmptyTask)
        assert len(tasks) == 1
        assert len(tasks[0].data) == 3

    def test_filters_invalid_urls(self, tmp_path):
        df = pd.DataFrame({
            "url_id": ["good", "bad_ftp", "bad_private"],
            "asset_url": [
                "https://example.com/ok.jpg",
                "ftp://example.com/bad.jpg",
                "http://192.168.1.1/internal.jpg",
            ],
            "asset_mime": ["image/jpeg"] * 3,
            "target_path": ["a.jpg", "b.jpg", "c.jpg"],
            "source_ref": ["test"] * 3,
        })
        path = tmp_path / "manifest.parquet"
        pq.write_table(pa.Table.from_pandas(df), str(path))

        stage = ManifestReaderStage(manifest_path=str(path), urls_per_task=10)
        tasks = stage.process(EmptyTask)
        assert len(tasks) == 1
        assert len(tasks[0].data) == 1
        assert "example.com/ok.jpg" in tasks[0].data[0]

    def test_preserves_metadata(self, manifest_file):
        stage = ManifestReaderStage(manifest_path=manifest_file, urls_per_task=5)
        tasks = stage.process(EmptyTask)
        assert "url_records" in tasks[0]._metadata
        assert len(tasks[0]._metadata["url_records"]) == 5
        assert tasks[0]._metadata["url_records"][0]["url_id"] == "id_0"


# --- ImageDownloaderStage (mocked HTTP) ---


class TestImageDownloaderStage:
    def test_downloads_and_returns_paths(self, tmp_path):
        from nemo_curator_stages.image.download.downloader import ImageDownloaderStage

        stage = ImageDownloaderStage(
            output_dir=str(tmp_path / "images"),
            max_workers=2,
            max_retries=1,
        )
        stage.setup()

        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.headers = {"Content-Type": "image/jpeg"}
        fake_response.content = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        fake_response.url = "https://example.com/img.jpg"
        fake_response.history = []

        task = FileGroupTask(
            task_id="test_task",
            dataset_name="test",
            data=["https://example.com/img.jpg"],
        )

        with patch.object(stage._downloader, "_get_session") as mock_session:
            mock_session.return_value.get.return_value = fake_response
            result = stage.process(task)

        assert result is not None
        assert len(result.data) == 1
        assert result._metadata["downloaded_count"] == 1

    def test_returns_none_when_all_fail(self, tmp_path):
        from nemo_curator_stages.image.download.downloader import ImageDownloaderStage

        stage = ImageDownloaderStage(
            output_dir=str(tmp_path / "images"),
            max_workers=1,
            max_retries=1,
        )
        stage.setup()

        task = FileGroupTask(
            task_id="test_task",
            dataset_name="test",
            data=["https://example.com/404.jpg"],
        )

        fake_response = MagicMock()
        fake_response.status_code = 404
        fake_response.headers = {}

        with patch.object(stage._downloader, "_get_session") as mock_session:
            mock_session.return_value.get.return_value = fake_response
            result = stage.process(task)

        assert result is None


# --- CompositeStage ---


class TestImageDownloadCompositeStage:
    def test_decomposition_with_compliance(self):
        from nemo_curator_stages.image.download.stage import ImageDownloadCompositeStage

        stage = ImageDownloadCompositeStage(
            manifest_path="dummy.parquet",
            output_dir="/tmp/images",
            enable_compliance=True,
        )
        stages = stage.decompose()
        assert len(stages) == 3
        assert stages[0].name == "ManifestReaderStage"
        assert stages[1].name == "ComplianceFilterStage"
        assert stages[2].name == "ImageDownloaderStage"

    def test_decomposition_without_compliance(self):
        from nemo_curator_stages.image.download.stage import ImageDownloadCompositeStage

        stage = ImageDownloadCompositeStage(
            manifest_path="dummy.parquet",
            output_dir="/tmp/images",
            enable_compliance=False,
        )
        stages = stage.decompose()
        assert len(stages) == 2
        assert stages[0].name == "ManifestReaderStage"
        assert stages[1].name == "ImageDownloaderStage"
