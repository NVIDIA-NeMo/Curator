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

"""Bounded smoke sources and isolated output paths."""

from __future__ import annotations

import contextlib
import copy
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from nemo_curator.stages.audio.audio_agent.recipe import (
    EXECUTION_KNOB_PARAMS,
    OUTPUT_LOCATION_PARAMS,
    Recipe,
)

if TYPE_CHECKING:
    from nemo_curator.stages.audio.audio_agent.contracts import SmokeReport

# Smoke runs exercise real write behavior, but only inside an ephemeral tree.
# The ``raw_data_dir`` subtraction is defensive rather than active: that param is not in
# ``OUTPUT_LOCATION_PARAMS`` today, so it removes nothing. It stays because if the set ever
# grows to cover it, redirecting it would be wrong -- for a dataset source it is execution
# INPUT as well as an acquisition destination, and smoke refuses generated sources outright
# instead (``_bound_recipe``), keeping pre-staged roots read-only.
_SMOKE_FILE_OUTPUT_PARAMS = frozenset({"output_path", "output_manifest", "output_audio_tar_path"})


_SMOKE_DIR_OUTPUT_PARAMS = frozenset(OUTPUT_LOCATION_PARAMS - _SMOKE_FILE_OUTPUT_PARAMS - {"raw_data_dir"})


@dataclass(frozen=True)
class _SmokeBound:
    """A proved smoke-input cap, or a refusal before stage construction."""

    recipe: Recipe | None
    tmp_paths: tuple[str, ...] = ()
    input_count: int | None = None
    error: str = ""
    output_root: str | None = None


def _bound_recipe(  # noqa: C901, PLR0911, PLR0912
    recipe: Recipe,
    sample: int,
    rpt: SmokeReport,
    binding: Any,  # noqa: ANN401 - DatasetBinding without an eager import
) -> _SmokeBound:
    """Prove that the source can emit at most ``sample`` rows.

    Bounding is a closed adapter table, just like source identity. A source that
    cannot be capped without downloading or reading an unknown remote selector is
    refused before stage construction. The original recipe and every stage default
    remain untouched.
    """
    bounded = copy.deepcopy(recipe)
    if not bounded.stages:
        return _SmokeBound(None, error="recipe has no source stage")
    src = bounded.stages[0]
    if src.ref == "ManifestReader":
        manifests = tuple(getattr(binding, "selected_manifest_files", ()) or ())
        if not manifests:
            return _SmokeBound(
                None,
                error=(
                    "ManifestReader selectors are not a complete ordered set of "
                    "local manifest files; remote or mixed selectors are unsupported"
                ),
            )
        tmp, count, error = _write_bounded_manifest(
            manifests,
            sample,
            include_files=src.params.get("include_files"),
            include_files_key=src.params.get("include_files_key", "audio_filepath"),
        )
        if error:
            return _SmokeBound(None, error=error)
        execution_knobs = {key: value for key, value in src.params.items() if key in EXECUTION_KNOB_PARAMS}
        src.params = {
            "manifest_path": tmp,
            "files_per_partition": 1,
            "blocksize": None,
            "file_extensions": [".jsonl"],
            "storage_options": None,
            "include_files": src.params.get("include_files"),
            "include_files_key": src.params.get("include_files_key", "audio_filepath"),
            **execution_knobs,
        }
        rpt.notes.append(f"bounded via {len(manifests)} resolved local manifest file(s) ({count}/{sample} rows)")
        return _SmokeBound(bounded, (tmp,), count)

    if src.ref == "ReadLongFormManifestStage":
        manifest = getattr(binding, "primary_path", None)
        if not isinstance(manifest, str):
            return _SmokeBound(None, error="long-form manifest path was not resolved")
        tmp, count, error = _write_bounded_manifest((manifest,), sample)
        if error:
            return _SmokeBound(None, error=error)
        src.params["input_manifest"] = tmp
        rpt.notes.append(f"bounded via truncated long-form manifest ({count}/{sample} rows)")
        return _SmokeBound(bounded, (tmp,), count)

    if src.ref == "CreateInitialManifestFleursStage":
        if getattr(binding, "generated", False) or not getattr(binding, "profile_source", None):
            return _SmokeBound(
                None,
                error=(
                    "unstaged FLEURS would download and materialize the complete split; "
                    "pre-stage the dataset before smoke"
                ),
            )
        tmp, count, error = _write_bounded_fleurs_manifest(src, binding, sample)
        if error:
            return _SmokeBound(None, error=error)
        execution_knobs = {key: value for key, value in src.params.items() if key in EXECUTION_KNOB_PARAMS}
        src.ref = "ManifestReader"
        src.params = {"manifest_path": tmp, **execution_knobs}
        rpt.notes.append(f"bounded pre-staged FLEURS via temporary manifest ({count}/{sample} rows)")
        return _SmokeBound(bounded, (tmp,), count)

    if src.ref == "CreateInitialManifestReadSpeechStage" and getattr(binding, "generated", False):
        return _SmokeBound(
            None,
            error=(
                "unstaged ReadSpeech would download and extract the complete archive; "
                "pre-stage the dataset before smoke"
            ),
        )

    if src.ref in {
        "CreateInitialManifestAudioFolderStage",
        "CreateInitialManifestReadSpeechStage",
    }:
        configured = src.params.get("max_samples")
        if "max_samples" not in src.params:
            from nemo_curator.stages.audio.audio_agent._resolve import static_contract_for

            configured = next(p.default for p in static_contract_for(src.ref).params if p.name == "max_samples")
        limited = configured is not None and (
            configured >= 0 if src.ref == "CreateInitialManifestAudioFolderStage" else configured > 0
        )
        limit = min(sample, configured) if limited else sample
        src.params["max_samples"] = limit
        rpt.notes.append(f"bounded via max_samples={limit}")
        return _SmokeBound(bounded)

    return _SmokeBound(
        None,
        error=f"source stage {src.ref!r} has no proven smoke-bounding adapter",
    )


def _isolate_smoke_outputs(bound: _SmokeBound, rpt: SmokeReport) -> _SmokeBound:  # noqa: C901
    """Redirect every reviewed output mutation into one disposable local tree."""
    if bound.recipe is None:
        return bound
    try:
        root = tempfile.mkdtemp(prefix="audio_agent_smoke_outputs_")
    except OSError as exc:
        return _SmokeBound(
            None,
            tmp_paths=bound.tmp_paths,
            input_count=bound.input_count,
            error=f"temporary output sandbox could not be created: {type(exc).__name__}: {exc}",
        )

    aliases: dict[str, str] = {}
    try:
        for index, stage in enumerate(bound.recipe.stages):
            for key in sorted(_SMOKE_FILE_OUTPUT_PARAMS | _SMOKE_DIR_OUTPUT_PARAMS):
                value = stage.params.get(key)
                if not isinstance(value, str) or not value:
                    continue
                stage.params[key] = _smoke_output_path(
                    root,
                    index,
                    stage.ref,
                    key,
                    value,
                    aliases,
                )

            if (
                stage.ref == "MonoConversionStage"
                and stage.params.get("write_to_disk", False)
                and not stage.params.get("output_dir")
            ):
                stage.params["output_dir"] = _smoke_output_path(
                    root,
                    index,
                    stage.ref,
                    "output_dir",
                    f"<implicit:{index}:output_dir>",
                    aliases,
                )

            if stage.ref == "PyAnnoteDiarizationStage" and stage.params.get(
                "write_rttm",
                True,
            ):
                stage.params["write_rttm"] = False
                rpt.notes.append("disabled source-adjacent PyAnnote RTTM side output for smoke")

            if stage.ref in {"SplitLongAudioStage", "SplitASRAlignJoinStage"}:
                stage.params["output_dir"] = _smoke_output_path(
                    root,
                    index,
                    stage.ref,
                    "output_dir",
                    f"<implicit:{index}:split_output_dir>",
                    aliases,
                )

            if stage.ref == "CreateInitialManifestReadSpeechStage":
                # Binding already proved this source is staged. Prevent a
                # disappearing/racing directory from turning smoke into a 4.88 GB
                # acquisition between the read-only check and stage execution.
                stage.params["auto_download"] = False
    except (OSError, TypeError, ValueError) as exc:
        _cleanup([root])
        return _SmokeBound(
            None,
            tmp_paths=bound.tmp_paths,
            input_count=bound.input_count,
            error=f"output sandbox could not be configured: {type(exc).__name__}: {exc}",
        )

    rpt.notes.append("all pipeline outputs isolated in temporary smoke storage")
    return _SmokeBound(
        bound.recipe,
        (*bound.tmp_paths, root),
        bound.input_count,
        output_root=root,
    )


def _smoke_output_path(  # noqa: PLR0913
    root: str,
    stage_index: int,
    stage_ref: str,
    key: str,
    original: str,
    aliases: dict[str, str],
) -> str:
    """Map a production location to a stable path inside ``root``."""
    if original in aliases:
        return aliases[original]
    stage_dir = os.path.join(root, f"{stage_index:03d}_{stage_ref}")
    if key in _SMOKE_FILE_OUTPUT_PARAMS:
        from urllib.parse import urlsplit

        basename = os.path.basename(urlsplit(original).path.rstrip("/"))
        filename = f"{key}_{basename or 'output'}"
        target = os.path.join(stage_dir, filename)
    else:
        target = os.path.join(stage_dir, key)
    aliases[original] = target
    return target


def _smoke_write_issues(  # noqa: C901
    stages: list[Any],
    output_root: str | None,
) -> list[str]:
    """Find a disk-writing stage whose output is not proved to be isolated."""
    if not output_root:
        return ["temporary output root is missing"]

    from nemo_curator.stages.audio import agent as foundation

    issues: list[str] = []
    for stage in _walk_smoke_stages(stages):
        name = type(stage).__name__
        if name == "FilePartitioningStage":
            # ManifestReader's framework partitioner only discovers input paths
            # and is not part of the audio AgentReady contract surface.
            continue
        try:
            contract = foundation.build_contract(stage)
        except Exception as exc:  # noqa: BLE001 - inability to inspect must fail closed
            issues.append(f"{name}: disk-write contract could not be inspected ({type(exc).__name__}: {exc})")
            continue
        if not contract.gates.writes_to_disk:
            continue
        # The stage declares its own output params, so a new writer needs no edit here.
        # ``None`` still means NOT DECLARED and still fails closed: a stage that says it
        # writes to disk without naming where cannot be proven isolated, and guessing from
        # parameter names would risk a smoke writing into the caller's real output tree.
        adapter = contract.gates.output_path_params
        if adapter is None:
            issues.append(
                f"{name}: declares writes_to_disk=True but does not declare "
                "output_path_params, so its output cannot be redirected into smoke storage"
            )
            continue
        if name == "CreateInitialManifestReadSpeechStage":
            if getattr(stage, "auto_download", True):
                issues.append(f"{name}: auto_download must be disabled for a pre-staged smoke")
            continue
        if not adapter:
            # ``[]`` positively claims "I write through no redirectable parameter", true and
            # safe for the one stage handled by name just above. Accepted from ANY stage it
            # becomes the easiest way past this check: the loop below has nothing to iterate and
            # the smoke is called isolated while the stage writes into the caller's real tree.
            # Unprovable isolation fails closed, so the exemption is a name, not a shape.
            issues.append(
                f"{name}: declares writes_to_disk=True with an empty output_path_params, so "
                "there is no parameter to redirect and its isolation cannot be proven"
            )
            continue
        for param in adapter:
            value = getattr(stage, param, None)
            if not isinstance(value, str) or not _inside_smoke_root(
                value,
                output_root,
            ):
                issues.append(f"{name}.{param}: output is not inside temporary smoke storage")
    return issues


def _walk_smoke_stages(stages: list[Any]):  # noqa: ANN202 - private generator
    """Yield configured stages and the concrete children of composites."""
    from nemo_curator.stages.base import CompositeStage

    for stage in stages:
        yield stage
        if isinstance(stage, CompositeStage):
            yield from _walk_smoke_stages(list(stage.decompose()))


def _inside_smoke_root(path: str, root: str) -> bool:
    resolved_path = os.path.realpath(path)
    resolved_root = os.path.realpath(root)
    return resolved_path == resolved_root or resolved_path.startswith(resolved_root + os.sep)


def _write_bounded_manifest(
    paths: tuple[str, ...],
    n: int,
    *,
    include_files: list[str] | None = None,
    include_files_key: str = "audio_filepath",
) -> tuple[str, int, str]:
    """Sample the source's selected rows in execution order, before applying the cap."""
    from nemo_curator.stages.audio.common.manifest_reader import _row_names_file

    wanted = None if include_files is None else {os.path.abspath(os.path.expanduser(p)) for p in include_files}
    fd, tmp = tempfile.mkstemp(suffix=".jsonl", prefix="audio_agent_smoke_")
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            for path in paths:
                try:
                    with open(path, encoding="utf-8") as src:
                        for lineno, line in enumerate(src, 1):
                            if not line.strip():
                                continue
                            try:
                                row = json.loads(line)
                            except json.JSONDecodeError as exc:
                                return _bounded_write_error(
                                    tmp,
                                    count,
                                    f"{path}:{lineno} contains invalid JSON: {exc}",
                                )
                            if not isinstance(row, dict):
                                return _bounded_write_error(
                                    tmp,
                                    count,
                                    f"{path}:{lineno} must contain a JSON object",
                                )
                            if wanted is not None and not _row_names_file(row, include_files_key, wanted):
                                continue
                            out.write(line if line.endswith("\n") else line + "\n")
                            count += 1
                            if count >= n:
                                return tmp, count, ""
                except (OSError, UnicodeError) as exc:
                    return _bounded_write_error(
                        tmp,
                        count,
                        f"{path} could not be read: {type(exc).__name__}: {exc}",
                    )
    except OSError as exc:
        return _bounded_write_error(
            tmp,
            count,
            f"bounded manifest could not be written: {type(exc).__name__}: {exc}",
        )
    return tmp, count, ""


def _write_bounded_fleurs_manifest(
    source: Any,  # noqa: ANN401 - copied StageRef
    binding: Any,  # noqa: ANN401 - DatasetBinding
    n: int,
) -> tuple[str, int, str]:
    """Materialize the first ``n`` rows exactly as a pre-staged FLEURS source does."""
    identity_files = list(getattr(binding, "profile_kwargs", {}).get("identity_files") or [])
    if len(identity_files) != 1:
        return "", 0, "pre-staged FLEURS transcript was not uniquely resolved"
    transcript = identity_files[0]
    audio_root = str(binding.profile_source)
    filepath_key = source.params.get("filepath_key", "audio_filepath")
    text_key = source.params.get("text_key", "text")
    fd, tmp = tempfile.mkstemp(suffix=".jsonl", prefix="audio_agent_smoke_fleurs_")
    count = 0
    try:
        with (
            os.fdopen(fd, "w", encoding="utf-8") as out,
            open(
                transcript,
                encoding="utf-8",
            ) as src,
        ):
            for line in src:
                parts = line.strip().split("\t")
                if len(parts) < 3:  # noqa: PLR2004
                    continue
                row = {
                    filepath_key: os.path.abspath(os.path.join(audio_root, parts[1])),
                    text_key: parts[2],
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
                if count >= n:
                    break
    except (OSError, UnicodeError) as exc:
        return _bounded_write_error(
            tmp,
            count,
            f"{transcript} could not be read: {type(exc).__name__}: {exc}",
        )
    return tmp, count, ""


def _bounded_write_error(tmp: str, count: int, error: str) -> tuple[str, int, str]:
    with contextlib.suppress(OSError):
        os.remove(tmp)
    return "", count, error


def _cleanup(paths: list[str]) -> None:
    for p in paths:
        with contextlib.suppress(OSError):
            if os.path.isdir(p) and not os.path.islink(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
