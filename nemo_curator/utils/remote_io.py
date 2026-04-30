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

from __future__ import annotations

import io
import posixpath
import re
import shutil
import signal
import subprocess
import tarfile
from contextlib import contextmanager
from typing import IO, TYPE_CHECKING, Any, Literal

import fsspec
from fsspec.core import url_to_fs

if TYPE_CHECKING:
    from collections.abc import Iterator

_OP_CL_PATTERN = re.compile(r"_OP_(\d+)\.\.(\d+)_CL_")
_BRACE_RANGE_PATTERN = re.compile(r"\{(\d+)\.\.(\d+)\}")


class PipeStream:
    def __init__(self, command: str, *, allow_sigpipe: bool = False):
        self.command = command
        self.allow_sigpipe = allow_sigpipe
        self.process: subprocess.Popen[bytes] | None = None

    def __enter__(self) -> IO[bytes]:
        self.process = subprocess.Popen(  # noqa: S602
            self.command,
            shell=True,
            executable="/bin/bash",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if self.process.stdout is None:
            msg = f"Failed to open pipe command stdout: {self.command}"
            raise RuntimeError(msg)
        return self.process.stdout

    def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        if self.process is None:
            return False
        if self.process.stdout is not None and not self.process.stdout.closed:
            self.process.stdout.close()
        stderr = b""
        if self.process.stderr is not None:
            stderr = self.process.stderr.read()
            self.process.stderr.close()
        return_code = self.process.wait()
        if exc_type is None and return_code != 0 and not self._is_allowed_sigpipe_return_code(return_code):
            detail = stderr.decode("utf-8", errors="replace").strip()
            msg = f"Pipe command failed with exit code {return_code}: {self.command}"
            if detail:
                msg += f"\n{detail}"
            raise RuntimeError(msg)
        return False

    def _is_allowed_sigpipe_return_code(self, return_code: int) -> bool:
        if not self.allow_sigpipe or not hasattr(signal, "SIGPIPE"):
            return False
        sigpipe_value = signal.Signals.SIGPIPE.value
        return return_code in {128 + sigpipe_value, -sigpipe_value}


def _expand_spec_string(spec: str) -> list[str]:
    for pattern in (_OP_CL_PATTERN, _BRACE_RANGE_PATTERN):
        match = pattern.search(spec)
        if match is None:
            continue
        start_str, end_str = match.groups()
        start = int(start_str)
        end = int(end_str)
        if end < start:
            msg = f"Invalid shard range: start={start}, end={end}, spec={spec}"
            raise ValueError(msg)
        width = max(len(start_str), len(end_str))
        prefix = spec[: match.start()]
        suffix = spec[match.end() :]
        expanded = [f"{prefix}{value:0{width}d}{suffix}" for value in range(start, end + 1)]
        results: list[str] = []
        for item in expanded:
            results.extend(_expand_spec_string(item))
        return results
    return [spec]


def expand_sharded_paths(paths: str | list[str]) -> list[str]:
    if isinstance(paths, list):
        results: list[str] = []
        for path in paths:
            results.extend(_expand_spec_string(path))
        return results
    return _expand_spec_string(paths)


def resolve_transport(path: str, transport: Literal["auto", "fsspec", "pipe"]) -> Literal["fsspec", "pipe"]:
    if transport == "auto":
        return "pipe" if path.startswith("pipe:") else "fsspec"
    return transport


@contextmanager
def open_binary_stream(
    path: str,
    *,
    storage_options: dict[str, Any] | None = None,
    transport: Literal["auto", "fsspec", "pipe"] = "auto",
    allow_sigpipe: bool = False,
) -> Iterator[IO[bytes]]:
    resolved_transport = resolve_transport(path, transport)
    if resolved_transport == "pipe":
        command = path[len("pipe:") :].strip() if path.startswith("pipe:") else path
        with PipeStream(command, allow_sigpipe=allow_sigpipe) as stream:
            yield stream
    else:
        with fsspec.open(path, mode="rb", **(storage_options or {})) as stream:
            yield stream


@contextmanager
def open_text_stream(
    path: str,
    *,
    storage_options: dict[str, Any] | None = None,
    transport: Literal["auto", "fsspec", "pipe"] = "auto",
    encoding: str = "utf-8",
) -> Iterator[IO[str]]:
    resolved_transport = resolve_transport(path, transport)
    if resolved_transport == "pipe":
        with open_binary_stream(path, storage_options=storage_options, transport=transport) as stream:
            text_stream = io.TextIOWrapper(stream, encoding=encoding)
            try:
                yield text_stream
            finally:
                text_stream.detach()
    else:
        with fsspec.open(path, mode="rt", encoding=encoding, **(storage_options or {})) as stream:
            yield stream


def read_binary(
    path: str,
    *,
    storage_options: dict[str, Any] | None = None,
    transport: Literal["auto", "fsspec", "pipe"] = "auto",
    allow_sigpipe: bool = False,
) -> bytes:
    with open_binary_stream(
        path,
        storage_options=storage_options,
        transport=transport,
        allow_sigpipe=allow_sigpipe,
    ) as stream:
        return stream.read()


def iter_tar_member_names(
    tar_path: str,
    *,
    storage_options: dict[str, Any] | None = None,
    transport: Literal["auto", "fsspec", "pipe"] = "auto",
) -> Iterator[str]:
    with (
        open_binary_stream(tar_path, storage_options=storage_options, transport=transport) as stream,
        tarfile.open(fileobj=stream, mode="r|*") as tar,
    ):
        for member in tar:
            if member.isfile():
                yield member.name


def copy_path(
    source_path: str,
    destination_path: str,
    *,
    source_storage_options: dict[str, Any] | None = None,
    destination_storage_options: dict[str, Any] | None = None,
    source_transport: Literal["auto", "fsspec", "pipe"] = "auto",
) -> None:
    if destination_path.startswith("pipe:"):
        msg = f"Writing to pipe destinations is not supported: {destination_path}"
        raise ValueError(msg)

    fs, resolved_destination = url_to_fs(destination_path, **(destination_storage_options or {}))
    parent_dir = posixpath.dirname(resolved_destination)
    if parent_dir:
        fs.makedirs(parent_dir, exist_ok=True)

    with (
        open_binary_stream(
            source_path,
            storage_options=source_storage_options,
            transport=source_transport,
            allow_sigpipe=True,
        ) as source_stream,
        fs.open(resolved_destination, "wb") as destination_stream,
    ):
        shutil.copyfileobj(source_stream, destination_stream, length=1024 * 1024)


def remove_path(
    path: str,
    *,
    storage_options: dict[str, Any] | None = None,
    recursive: bool = False,
    ignore_missing: bool = False,
) -> None:
    if path.startswith("pipe:"):
        msg = f"Deleting pipe destinations is not supported: {path}"
        raise ValueError(msg)

    fs, resolved_path = url_to_fs(path, **(storage_options or {}))
    try:
        fs.rm(resolved_path, recursive=recursive)
    except FileNotFoundError:
        if not ignore_missing:
            raise


def basename_from_path(path: str) -> str:
    stripped = path[len("pipe:") :].strip() if path.startswith("pipe:") else path
    stripped = stripped.rstrip("/")
    if not stripped:
        return ""
    return stripped.rsplit("/", maxsplit=1)[-1]


def build_remote_uri(*, protocol: str, bucket: str, key: str = "") -> str:
    normalized_bucket = bucket.strip("/")
    if not normalized_bucket:
        msg = "bucket is required to build a remote URI"
        raise ValueError(msg)
    normalized_key = key.strip("/")
    if normalized_key:
        return f"{protocol}://{normalized_bucket}/{normalized_key}"
    return f"{protocol}://{normalized_bucket}"
