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

import sys
import types
import urllib.error
from pathlib import Path
from types import ModuleType

import pytest


def _import_module() -> ModuleType:
    # Inject a stub for optional dependency 'wget' to avoid import errors.
    if "wget" not in sys.modules:
        sys.modules["wget"] = types.SimpleNamespace(download=lambda *_args, **_kwargs: None)
    from nemo_curator.stages.audio.datasets import file_utils

    return file_utils


def _http_error(code: int, retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    return urllib.error.HTTPError(url="http://example/file", code=code, msg="boom", hdrs=headers, fp=None)


def test_skips_download_when_file_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    file_utils = _import_module()
    target = tmp_path / "train.tsv"
    target.write_text("already here", encoding="utf-8")

    called = False

    def _should_not_run(*_args: object, **_kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(file_utils, "_download_with_retries", _should_not_run)
    result = file_utils.download_file("http://example/train.tsv", str(tmp_path))

    assert result == str(target)
    assert called is False


def test_retries_then_succeeds_on_429(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    file_utils = _import_module()
    monkeypatch.setattr(file_utils.time, "sleep", lambda _seconds: None)

    attempts = {"n": 0}

    def _flaky_download(*_args: object, **_kwargs: object) -> None:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise _http_error(429)

    monkeypatch.setattr(file_utils.wget, "download", _flaky_download)
    file_utils.download_file("http://example/train.tsv", str(tmp_path))

    assert attempts["n"] == 3


def test_raises_after_exhausting_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    file_utils = _import_module()
    monkeypatch.setattr(file_utils.time, "sleep", lambda _seconds: None)

    def _always_429(*_args: object, **_kwargs: object) -> None:
        raise _http_error(429)

    monkeypatch.setattr(file_utils.wget, "download", _always_429)
    with pytest.raises(urllib.error.HTTPError):
        file_utils.download_file("http://example/train.tsv", str(tmp_path))


def test_non_retryable_http_error_raises_immediately(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    file_utils = _import_module()
    monkeypatch.setattr(file_utils.time, "sleep", lambda _seconds: None)

    attempts = {"n": 0}

    def _not_found(*_args: object, **_kwargs: object) -> None:
        attempts["n"] += 1
        raise _http_error(404)

    monkeypatch.setattr(file_utils.wget, "download", _not_found)
    with pytest.raises(urllib.error.HTTPError):
        file_utils.download_file("http://example/train.tsv", str(tmp_path))

    assert attempts["n"] == 1


def test_backoff_honors_retry_after_header() -> None:
    file_utils = _import_module()
    assert file_utils._backoff_seconds(1, "7") == pytest.approx(7.0)
    # Retry-After is capped at the max backoff.
    assert file_utils._backoff_seconds(1, "9999") == pytest.approx(file_utils._MAX_BACKOFF_SECONDS)
    # Invalid header falls back to exponential backoff (>= 2**attempt).
    assert file_utils._backoff_seconds(2, "not-a-number") >= 4.0
