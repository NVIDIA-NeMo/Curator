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

"""Resolve a model artifact to a local path: an explicit path wins, otherwise download it once and cache it.

Every GranaryHiFi stage that needs a model file takes the same pair of options: ``<name>_path`` (use this file,
no download) and ``<name>_url`` (fetch it into ``~/.cache/nemo_curator/granary_hifi/<subdir>/`` on first use).
Downloads are written to a temporary file in the target directory and then renamed, so concurrent workers never
observe a half-written artifact, and an interrupted download leaves no usable file behind.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

from loguru import logger

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "nemo_curator" / "granary_hifi"
_CHUNK_BYTES = 1 << 20
_TIMEOUT_SEC = 300


def cache_dir(subdir: str, root: str | os.PathLike[str] | None = None) -> Path:
    """Cache directory for one artifact (``~/.cache/nemo_curator/granary_hifi/<subdir>`` by default)."""
    return (Path(root).expanduser() if root is not None else DEFAULT_CACHE_DIR) / subdir


def sha256_of(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def filename_from_url(url: str) -> str:
    name = Path(unquote(urlparse(url).path)).name
    if not name:
        msg = f"Cannot derive a file name from {url!r}; pass an explicit filename"
        raise ValueError(msg)
    return name


def _auth_headers(url: str) -> dict[str, str]:
    """Hugging Face token for huggingface.co links, so gated repos work after ``hf auth login`` or ``HF_TOKEN``."""
    host = urlparse(url).hostname or ""
    if host != "huggingface.co" and not host.endswith(".huggingface.co"):
        return {}
    from huggingface_hub import get_token

    token = get_token()
    return {"Authorization": f"Bearer {token}"} if token else {}


def download_file(url: str, target: str | os.PathLike[str], sha256: str | None = None) -> Path:
    """Download ``url`` to ``target`` unless it is already there; verify ``sha256`` when given."""
    target = Path(target).expanduser()
    if target.is_file() and target.stat().st_size > 0:
        if sha256 and sha256_of(target) != sha256:
            logger.warning("Cached {} failed its sha256 check; downloading it again", target)
            target.unlink()
        else:
            return target

    import requests

    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading {} -> {}", url, target)
    handle, temporary = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".part")
    os.close(handle)
    try:
        with requests.get(url, headers=_auth_headers(url), stream=True, timeout=_TIMEOUT_SEC) as response:
            response.raise_for_status()
            with open(temporary, "wb") as out:
                out.writelines(response.iter_content(chunk_size=_CHUNK_BYTES))
        if sha256:
            actual = sha256_of(temporary)
            if actual != sha256:
                msg = f"{url} has sha256 {actual}, expected {sha256}"
                raise RuntimeError(msg)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def resolve_artifact(  # noqa: PLR0913
    path: str | None,
    url: str | None,
    subdir: str,
    what: str,
    option: str,
    filename: str | None = None,
    sha256: str | None = None,
    root: str | os.PathLike[str] | None = None,
) -> Path:
    """Return a local file for ``what``: ``path`` if set, else download ``url`` into the cache.

    Args:
        path: Explicit local file; used as-is and never downloaded.
        url: Download location used when ``path`` is unset.
        subdir: Cache sub-directory name.
        what: Human name of the artifact, for log and error messages.
        option: The ``<name>_path`` option to name in the error when nothing is configured.
        filename: Cached file name; defaults to the last path element of ``url``.
        sha256: Expected digest of the downloaded file.
        root: Cache root overriding ``~/.cache/nemo_curator/granary_hifi``.
    """
    if path:
        resolved = Path(path).expanduser()
        if not resolved.is_file():
            msg = f"{what}: {option}={path} does not exist"
            raise FileNotFoundError(msg)
        return resolved
    if not url:
        msg = f"{what} is not available: pass {option}=<local file> (no download URL is configured)"
        raise ValueError(msg)
    return download_file(url, cache_dir(subdir, root) / (filename or filename_from_url(url)), sha256=sha256)
