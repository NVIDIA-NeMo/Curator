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

"""Resolution of model artifacts: a local path, a download, or a clear error when neither is configured."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Self

import pytest

from nemo_curator.stages.audio.tts import model_download

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    """Serve named byte payloads over the stubbed ``requests.get`` the downloader uses."""
    files: dict[str, bytes] = {}

    class _Response:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_content(self, chunk_size: int = 1) -> list[bytes]:
            return [self._payload[i : i + chunk_size] for i in range(0, len(self._payload), chunk_size)]

    def fake_get(url: str, **_: object) -> _Response:
        if url not in files:
            msg = f"unexpected url {url}"
            raise AssertionError(msg)
        return _Response(files[url])

    import requests

    monkeypatch.setattr(requests, "get", fake_get)
    return files


def test_local_path_wins_and_is_not_downloaded(tmp_path: Path, served: dict[str, bytes]) -> None:
    local = tmp_path / "model.joblib"
    local.write_bytes(b"local")
    resolved = model_download.resolve_artifact(
        str(local), "https://example.invalid/model.joblib", "sed", "GBM", "classifier_path", root=tmp_path / "cache"
    )
    assert resolved == local
    assert served == {}  # nothing was served, so nothing was fetched


def test_missing_local_path_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="classifier_path"):
        model_download.resolve_artifact(
            str(tmp_path / "absent.joblib"), None, "sed", "GBM", "classifier_path", root=tmp_path
        )


def test_no_path_and_no_url_names_the_option(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="classifier_path"):
        model_download.resolve_artifact(None, None, "sed", "GBM classifier", "classifier_path", root=tmp_path)


def test_download_caches_and_reuses(tmp_path: Path, served: dict[str, bytes]) -> None:
    url = "https://example.invalid/model.joblib"
    served[url] = b"payload"
    root = tmp_path / "cache"
    first = model_download.resolve_artifact(None, url, "sed", "GBM", "classifier_path", root=root)
    assert first.read_bytes() == b"payload"
    assert first == root / "sed" / "model.joblib"

    served.clear()  # a second call must not reach the network
    second = model_download.resolve_artifact(None, url, "sed", "GBM", "classifier_path", root=root)
    assert second == first


def test_sha256_mismatch_leaves_no_file(tmp_path: Path, served: dict[str, bytes]) -> None:
    url = "https://example.invalid/model.joblib"
    served[url] = b"payload"
    target = tmp_path / "model.joblib"
    with pytest.raises(RuntimeError, match="expected"):
        model_download.download_file(url, target, sha256="00" * 32)
    assert not target.exists()
    assert not list(tmp_path.glob(".*.part"))


def test_sha256_match_is_accepted(tmp_path: Path, served: dict[str, bytes]) -> None:
    url = "https://example.invalid/model.joblib"
    served[url] = b"payload"
    digest = hashlib.sha256(b"payload").hexdigest()
    target = model_download.download_file(url, tmp_path / "model.joblib", sha256=digest)
    assert target.read_bytes() == b"payload"


def test_stale_cache_is_refetched(tmp_path: Path, served: dict[str, bytes]) -> None:
    url = "https://example.invalid/model.joblib"
    served[url] = b"new"
    target = tmp_path / "model.joblib"
    target.write_bytes(b"old")
    resolved = model_download.download_file(url, target, sha256=hashlib.sha256(b"new").hexdigest())
    assert resolved.read_bytes() == b"new"


def test_hf_links_carry_the_token_and_others_do_not(monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "hf_test")
    hf = "https://huggingface.co/nvidia/repo/resolve/main/model.joblib"
    assert model_download._auth_headers(hf) == {"Authorization": "Bearer hf_test"}
    assert model_download._auth_headers("https://example.invalid/model.joblib") == {}
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    assert model_download._auth_headers(hf) == {}


def test_bundle_downloads_checkpoints_and_uses_shipped_calibration(
    tmp_path: Path, served: dict[str, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    from nemo_curator.stages.audio.tts.early_cutoff import bundle

    files = {name: (f"remote_{name}", hashlib.sha256(name.encode()).hexdigest()) for name in bundle.CHECKPOINT_FILES}
    monkeypatch.setattr(bundle, "CHECKPOINT_FILES", files)
    monkeypatch.setattr(model_download, "DEFAULT_CACHE_DIR", tmp_path)
    for name, (remote, _) in files.items():
        served[f"https://example.invalid/assets/{remote}"] = name.encode()

    resolved = bundle.resolve_bundle(None, "https://example.invalid/assets")  # trailing slash added
    assert resolved is not None
    for name in files:
        assert resolved[name] == tmp_path / "early_cutoff" / name
        assert resolved[name].read_bytes() == name.encode()
    for name in bundle.CALIBRATION_FILES:
        assert resolved[name].parent == bundle.CALIBRATION_DIR
    assert bundle.BUNDLE_MANIFEST not in resolved
    local = bundle.resolve_bundle(str(tmp_path / "mine"), "https://example.invalid/assets")
    assert local is not None
    assert local[bundle.ENGLISH_CALIBRATION] == tmp_path / "mine" / bundle.ENGLISH_CALIBRATION
    assert bundle.resolve_bundle(None, None) is None
