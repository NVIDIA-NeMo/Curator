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

"""Transactional promotion of fixed-file pipeline outputs."""

from __future__ import annotations

import contextlib
import os
import uuid
from dataclasses import dataclass
from typing import Any

from nemo_curator.stages.audio.audio_agent import _verb_pretrain


def _atomic_output_params(stage: Any) -> tuple[str, ...]:  # noqa: ANN401
    return tuple(getattr(stage, "AGENT_ATOMIC_OUTPUT_PARAMS", ())) + _verb_pretrain._atomic_output_params(stage)


@dataclass
class _StagedFileOutput:
    """One fixed-file sink redirected until the full pipeline succeeds."""

    stage: Any
    param: str
    final_path: str
    staged_path: str


@dataclass
class _ResolvedStagedFileOutput:
    """Filesystem paths and prior-state facts needed for transactional promotion."""

    index: int
    fs: Any
    staged_path: str
    final_path: str
    backup_path: str
    local: bool
    had_final: bool
    destination_key: tuple[tuple[str, ...], str]


def _atomic_source_resume_sinks(stages: list[Any]) -> list[str]:
    """Sinks whose unpublished output cannot survive source-completion resume."""
    from nemo_curator.stages.audio._agent._composite import expand_composites

    return [
        type(item.stage).__name__ for item in expand_composites(stages).stages if _atomic_output_params(item.stage)
    ]


def _prepare_staged_file_outputs(stages: list[Any], *, checkpoint_path: str | None = None) -> list[_StagedFileOutput]:
    """Redirect opted-in sinks to unique sibling files owned by this run."""
    if checkpoint_path is not None and _atomic_source_resume_sinks(stages):
        # Source completion is durable before file promotion; discarding staging
        # would make a resumed source skip rows that no longer exist.
        msg = "source checkpoint resume is unsupported with atomically staged outputs"
        raise ValueError(msg)

    outputs: list[_StagedFileOutput] = []
    token = uuid.uuid4().hex
    try:
        for stage_index, stage in enumerate(stages):
            params = _atomic_output_params(stage)
            for param in params:
                final_path = getattr(stage, param, None)
                if not isinstance(final_path, str) or not final_path:
                    continue
                staged_path = f"{final_path}.audio-agent-{token}-{stage_index}.tmp"
                setattr(stage, param, staged_path)
                outputs.append(
                    _StagedFileOutput(
                        stage=stage,
                        param=param,
                        final_path=final_path,
                        staged_path=staged_path,
                    )
                )
    except Exception:
        _restore_staged_file_outputs(outputs)
        _discard_staged_file_outputs(outputs)
        raise
    return outputs


def _restore_staged_file_outputs(outputs: list[_StagedFileOutput]) -> None:
    for output in outputs:
        setattr(output.stage, output.param, output.final_path)


def _discard_staged_file_outputs(outputs: list[_StagedFileOutput]) -> None:
    """Delete only temporary files named and owned by this run."""
    from fsspec.core import url_to_fs

    for output in outputs:
        with contextlib.suppress(Exception):
            fs, path = url_to_fs(output.staged_path)
            if fs.exists(path):
                fs.rm(path)


def _local_protocol(fs: Any) -> bool:  # noqa: ANN401
    protocol = fs.protocol
    protocols = {protocol} if isinstance(protocol, str) else set(protocol)
    return bool(protocols & {"file", "local"})


def _filesystem_protocols(fs: Any) -> tuple[str, ...]:  # noqa: ANN401
    protocol = fs.protocol
    return (protocol,) if isinstance(protocol, str) else tuple(protocol)


def _move_staged_output(output: _ResolvedStagedFileOutput, source: str, destination: str) -> None:
    if output.local:
        os.replace(source, destination)
    else:
        output.fs.mv(source, destination)


def _remove_staged_output(output: _ResolvedStagedFileOutput, path: str) -> None:
    if not output.fs.exists(path):
        return
    if output.local:
        os.unlink(path)
    else:
        output.fs.rm(path)


def _sync_staged_output_parent(output: _ResolvedStagedFileOutput) -> None:
    if not output.local:
        return
    from nemo_curator.utils.atomic_io import fsync_directory

    final_parent = os.path.abspath(os.path.dirname(output.final_path) or ".")
    with contextlib.suppress(OSError):
        fsync_directory(final_parent)


def _rollback_staged_file_outputs(
    outputs: list[_ResolvedStagedFileOutput],
    *,
    backed_up: set[int],
    promoted: set[int],
) -> list[str]:
    """Restore every prior destination after an incomplete group promotion."""
    errors: list[str] = []
    for output in reversed(outputs):
        try:
            if output.index in backed_up:
                if not output.fs.exists(output.backup_path):
                    errors.append(f"{output.final_path!r}: backup {output.backup_path!r} is missing")
                    continue
                if not output.local:
                    _remove_staged_output(output, output.final_path)
                _move_staged_output(output, output.backup_path, output.final_path)
                _sync_staged_output_parent(output)
            elif output.index in promoted:
                _remove_staged_output(output, output.final_path)
                _sync_staged_output_parent(output)
        except Exception as exc:  # noqa: BLE001 - attempt every restore before reporting all failures
            errors.append(f"{output.final_path!r}: {type(exc).__name__}: {exc}")
    return errors


def _resolve_staged_file_output(index: int, output: _StagedFileOutput) -> _ResolvedStagedFileOutput:
    """Resolve and validate one staged/final sibling pair without mutating either."""
    from fsspec.core import url_to_fs

    staged_fs, staged_path = url_to_fs(output.staged_path)
    final_fs, final_path = url_to_fs(output.final_path)
    staged_protocols = _filesystem_protocols(staged_fs)
    if set(staged_protocols) != set(_filesystem_protocols(final_fs)):
        msg = f"staged and final output use different filesystems: {output.staged_path!r}, {output.final_path!r}"
        raise ValueError(msg)
    if not staged_fs.exists(staged_path):
        msg = f"pipeline completed without creating staged output {output.staged_path!r}"
        raise FileNotFoundError(msg)
    if staged_fs.isdir(staged_path):
        msg = f"pipeline created a directory where a staged file was required: {output.staged_path!r}"
        raise IsADirectoryError(msg)
    had_final = staged_fs.exists(final_path)
    if had_final and staged_fs.isdir(final_path):
        msg = f"final file output is an existing directory: {output.final_path!r}"
        raise IsADirectoryError(msg)
    backup_path = f"{staged_path}.previous"
    if staged_fs.exists(backup_path):
        msg = f"run-owned output backup already exists: {backup_path!r}"
        raise FileExistsError(msg)
    local = _local_protocol(staged_fs)
    if local:
        final_parent = os.path.dirname(final_path) or "."
        os.makedirs(final_parent, exist_ok=True)
        with open(staged_path, "rb") as stream:
            os.fsync(stream.fileno())
    return _ResolvedStagedFileOutput(
        index=index,
        fs=staged_fs,
        staged_path=staged_path,
        final_path=final_path,
        backup_path=backup_path,
        local=local,
        had_final=had_final,
        destination_key=(tuple(sorted(staged_protocols)), final_path),
    )


def _resolve_staged_file_outputs(outputs: list[_StagedFileOutput]) -> list[_ResolvedStagedFileOutput]:
    """Preflight every destination before publication changes any of them."""
    resolved: list[_ResolvedStagedFileOutput] = []
    destinations: set[tuple[tuple[str, ...], str]] = set()
    for index, output in enumerate(outputs):
        candidate = _resolve_staged_file_output(index, output)
        if candidate.destination_key in destinations:
            msg = f"multiple stages publish to the same final output {output.final_path!r}"
            raise ValueError(msg)
        destinations.add(candidate.destination_key)
        resolved.append(candidate)
    return resolved


def _promote_staged_file_outputs(outputs: list[_StagedFileOutput]) -> None:
    """Replace all final files as one rollback-protected publication group."""
    resolved = _resolve_staged_file_outputs(outputs)

    backed_up: set[int] = set()
    promoted: set[int] = set()
    try:
        for output in resolved:
            if output.had_final:
                _move_staged_output(output, output.final_path, output.backup_path)
                backed_up.add(output.index)
            _move_staged_output(output, output.staged_path, output.final_path)
            promoted.add(output.index)
            _sync_staged_output_parent(output)
    except BaseException as exc:
        for output in resolved:
            if (
                output.had_final
                and output.index not in backed_up
                and output.fs.exists(output.backup_path)
                and not output.fs.exists(output.final_path)
            ):
                backed_up.add(output.index)
            elif not output.had_final and output.fs.exists(output.final_path):
                promoted.add(output.index)
        rollback_errors = _rollback_staged_file_outputs(
            resolved,
            backed_up=backed_up,
            promoted=promoted,
        )
        if rollback_errors:
            details = "; ".join(rollback_errors)
            msg = f"output publication failed and rollback was incomplete: {details}"
            raise RuntimeError(msg) from exc
        raise

    for output in resolved:
        if output.index in backed_up:
            # Finals are committed; stale-backup cleanup must not turn success into failure.
            with contextlib.suppress(Exception):
                _remove_staged_output(output, output.backup_path)
                _sync_staged_output_parent(output)
