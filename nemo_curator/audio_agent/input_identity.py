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

"""Resolve the dataset a recipe will actually read.

The first stage's configured parameters are the execution truth.  ``Recipe.inputs``
and the CLI ``--data`` value are assertions about that truth; this module never
injects either value into a stage.  Resolution is deliberately read-only: local
filesystem state may be inspected, but no source is downloaded or generated.

This is a closed adapter table rather than a guess based on parameter names.  A
new source stage must opt in here with its exact path semantics before its input
may participate in profiling or reuse identity.
"""

from __future__ import annotations

import glob
import os
import re
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import unquote, urlsplit

if TYPE_CHECKING:
    from nemo_curator.audio_agent.recipe import Recipe, StageRef

BindingStatus = Literal["resolved", "missing", "mismatch", "ambiguous", "unsupported"]

__all__ = ["SUPPORTED_SOURCE_REFS", "DatasetBinding", "canonical_source", "resolve_dataset_binding"]

SUPPORTED_SOURCE_REFS = (
    "ManifestReader",
    "CreateInitialManifestAudioFolderStage",
    "CreateInitialManifestFleursStage",
    "CreateInitialManifestReadSpeechStage",
    "ReadLongFormManifestStage",
)
_SUPPORTED_SOURCE_SET = frozenset(SUPPORTED_SOURCE_REFS)
_LOCAL_URI_SCHEMES = frozenset({"file", "local"})
_LONG_FORM_RESOLUTIONS = frozenset({"basename", "relative", "as_is"})
_REQUIRED_RE = re.compile(r"^REQUIRED(?:[_:\s]|$)", re.IGNORECASE)
_TEMPLATE_RE = re.compile(r"(?:\{\{.*?\}\}|\$\{.*?\}|\{[^{}]+\}|<[^<>]+>)")


@dataclass(frozen=True)
class DatasetBinding:
    """Deterministic account of the source encoded by a recipe.

    ``configured_paths`` preserves the source stage's path order.  It can contain
    more than one path only for ``ManifestReader``.  ``primary_path`` is the
    effective dataset root or manifest when there is exactly one useful identity.
    ``profile_source`` is intentionally absent for an unstaged generated dataset
    and for a multi-manifest source that cannot be represented by one profiler
    invocation. ``selected_manifest_files`` exposes the complete ordered local
    ManifestReader expansion only after every configured selector resolves.
    """

    status: BindingStatus
    source_ref: str | None = None
    configured_paths: tuple[str, ...] = ()
    primary_path: str | None = None
    profile_source: str | None = None
    selected_manifest_files: tuple[str, ...] = ()
    profile_kwargs: dict[str, Any] = field(default_factory=dict)
    issues: tuple[str, ...] = ()
    reason: str = ""
    generated: bool = False
    source_index: int | None = None

    @property
    def configured_path(self) -> str | None:
        """Compatibility shorthand when exactly one path is configured."""
        return self.configured_paths[0] if len(self.configured_paths) == 1 else None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation without sharing mutable state."""
        return {
            "status": self.status,
            "source_ref": self.source_ref,
            "configured_paths": list(self.configured_paths),
            "primary_path": self.primary_path,
            "profile_source": self.profile_source,
            "selected_manifest_files": list(self.selected_manifest_files),
            "profile_kwargs": deepcopy(self.profile_kwargs),
            "issues": list(self.issues),
            "reason": self.reason,
            "generated": self.generated,
            "source_index": self.source_index,
        }


def canonical_source(value: str) -> str:
    """Canonicalize a source path without consulting a remote filesystem.

    Plain local paths and ``file://``/``local://`` aliases become absolute,
    symlink-normalized local paths.  Other URI schemes are returned lexically
    unchanged, because normalization rules belong to their backing filesystem.

    Raises:
        TypeError: If ``value`` is not a string.
        ValueError: If it is empty or still contains a recipe placeholder.
    """
    if not isinstance(value, str):
        msg = f"source path must be a string, got {type(value).__name__}"
        raise TypeError(msg)
    source = value.strip()
    if not source:
        msg = "source path must not be empty"
        raise ValueError(msg)
    if _is_unresolved(source):
        msg = f"source path is unresolved: {value!r}"
        raise ValueError(msg)

    parsed = urlsplit(source)
    scheme = parsed.scheme.lower()
    if scheme and scheme not in _LOCAL_URI_SCHEMES:
        return source
    if scheme in _LOCAL_URI_SCHEMES:
        path = unquote(parsed.path)
        if parsed.netloc and parsed.netloc.lower() != "localhost":
            path = f"//{parsed.netloc}{path}"
    else:
        path = source
    return os.path.realpath(os.path.abspath(os.path.expanduser(path)))


def resolve_dataset_binding(recipe: Recipe, data: str | None = None) -> DatasetBinding:
    """Resolve the dataset identity encoded by ``recipe``.

    Supported source stages are intentionally limited to
    :data:`SUPPORTED_SOURCE_REFS`.  A supported source anywhere except index zero,
    or more than one supported source, is ambiguous rather than silently guessed.
    """
    if not recipe.stages:
        return _binding("missing", reason="recipe has no source stage")

    source_positions = [
        (index, stage) for index, stage in enumerate(recipe.stages) if stage.ref in _SUPPORTED_SOURCE_SET
    ]
    if len(source_positions) > 1:
        refs = ", ".join(f"{stage.ref}@{index}" for index, stage in source_positions)
        return _binding(
            "ambiguous",
            source_ref=source_positions[0][1].ref,
            source_index=source_positions[0][0],
            reason=f"recipe contains multiple supported source stages: {refs}",
        )
    if source_positions and source_positions[0][0] != 0:
        index, stage = source_positions[0]
        return _binding(
            "ambiguous",
            source_ref=stage.ref,
            source_index=index,
            reason=f"supported source {stage.ref!r} appears at index {index}; a source must be the first stage",
        )
    if not source_positions:
        first = recipe.stages[0]
        return _binding(
            "unsupported",
            source_ref=first.ref,
            source_index=0,
            reason=f"first stage {first.ref!r} is not a supported dataset source",
        )

    stage = source_positions[0][1]
    handlers = {
        "ManifestReader": _resolve_manifest_reader,
        "CreateInitialManifestAudioFolderStage": _resolve_audio_folder,
        "CreateInitialManifestFleursStage": _resolve_fleurs,
        "CreateInitialManifestReadSpeechStage": _resolve_readspeech,
        "ReadLongFormManifestStage": _resolve_long_form,
    }
    return handlers[stage.ref](recipe, stage, data)


def _is_unresolved(value: str) -> bool:
    stripped = value.strip()
    return not stripped or bool(_REQUIRED_RE.match(stripped)) or bool(_TEMPLATE_RE.search(stripped))


def _binding(  # noqa: PLR0913 - mirrors the result record at one construction boundary
    status: BindingStatus,
    *,
    source_ref: str | None = None,
    configured_paths: tuple[str, ...] = (),
    primary_path: str | None = None,
    profile_source: str | None = None,
    profile_kwargs: dict[str, Any] | None = None,
    issues: tuple[str, ...] = (),
    reason: str,
    generated: bool = False,
    source_index: int | None = None,
) -> DatasetBinding:
    all_issues = issues or (reason,)
    return DatasetBinding(
        status=status,
        source_ref=source_ref,
        configured_paths=configured_paths,
        primary_path=primary_path,
        profile_source=profile_source,
        profile_kwargs=dict(profile_kwargs or {}),
        issues=all_issues,
        reason=reason,
        generated=generated,
        source_index=source_index,
    )


def _base(  # noqa: PLR0913 - common resolved fields are clearer when named at call sites
    source_ref: str,
    *,
    configured_paths: tuple[str, ...],
    primary_path: str | None,
    profile_source: str | None,
    profile_kwargs: dict[str, Any] | None = None,
    generated: bool = False,
) -> DatasetBinding:
    reason = "dataset source resolved"
    return DatasetBinding(
        status="resolved",
        source_ref=source_ref,
        configured_paths=configured_paths,
        primary_path=primary_path,
        profile_source=profile_source,
        profile_kwargs=dict(profile_kwargs or {}),
        reason=reason,
        generated=generated,
        source_index=0,
    )


def _canonical_scalar(value: object, label: str) -> tuple[str | None, str | None]:
    try:
        return canonical_source(value), None
    except (TypeError, ValueError) as exc:
        return None, f"{label}: {exc}"


def _canonical_os_scalar(
    value: object,
    label: str,
) -> tuple[str | None, str | None, bool]:
    """Resolve a path using the literal semantics of ``os.path`` source stages.

    These stages do not expand ``~`` and do not parse ``file://``/``local://``.
    Assertion values may use those aliases, but declaring one in a stage would
    make the resolver inspect different bytes from execution. ``is_uri`` lets
    the handler report that case as unsupported rather than merely missing.
    """
    if not isinstance(value, str):
        return None, f"{label}: source path must be a string, got {type(value).__name__}", False
    if not value.strip():
        return None, f"{label}: source path must not be empty", False
    if _is_unresolved(value):
        return None, f"{label}: source path is unresolved: {value!r}", False
    if urlsplit(value).scheme:
        return None, f"{label} does not support URI syntax: {value!r}", True
    return os.path.realpath(os.path.abspath(value)), None, False


def _canonical_many(value: object, label: str) -> tuple[tuple[str, ...], str | None]:
    if isinstance(value, str):
        canonical, issue = _canonical_scalar(value, label)
        return ((canonical,) if canonical is not None else ()), issue
    if not isinstance(value, list):
        return (), f"{label} must be a string or list of strings, got {type(value).__name__}"
    if not value:
        return (), f"{label} must not be an empty list"
    paths: list[str] = []
    for index, item in enumerate(value):
        canonical, issue = _canonical_scalar(item, f"{label}[{index}]")
        if issue:
            return (), issue
        if canonical is None:  # defensive: _canonical_scalar returns a path or an issue
            return (), f"{label}[{index}] did not resolve to a path"
        paths.append(canonical)
    return tuple(paths), None


def _is_remote(path: str) -> bool:
    scheme = urlsplit(path).scheme.lower()
    return bool(scheme and scheme not in _LOCAL_URI_SCHEMES)


def _local_source_exists(path: str) -> bool:
    if glob.has_magic(path):
        return bool(glob.glob(path, recursive=True))
    return os.path.exists(path)


def _local_manifest_records(  # noqa: PLR0913
    path: str,
    *,
    label: str,
    recurse_subdirectories: bool,
    file_extensions: str | list[str] | None,
    storage_options: dict[str, Any] | None,
    sort_by_size: bool,
) -> tuple[list[tuple[str, int]], str | None]:
    """Resolve one local ManifestReader selector exactly as its partitioner does.

    ManifestReader delegates discovery to FilePartitioningStage. Reusing its
    filesystem utility here keeps extension filtering, glob expansion, and
    directory depth aligned with the bytes execution will actually select.
    """
    from nemo_curator.utils.file_utils import get_all_file_paths_and_size_under

    no_selection_reason = f"{label} selects no local files with configured file_extensions={file_extensions!r}: {path}"
    try:
        selected = get_all_file_paths_and_size_under(
            path,
            recurse_subdirectories=recurse_subdirectories,
            keep_extensions=file_extensions,
            storage_options=storage_options,
            sort_by_size=sort_by_size,
        )
    except FileNotFoundError:
        return [], no_selection_reason
    except (OSError, TypeError, ValueError) as exc:
        return [], f"{label} could not be inspected: {exc}"
    if selected:
        return selected, None
    return [], no_selection_reason


def _assert_recipe_inputs(
    binding: DatasetBinding,
    recipe: Recipe,
    expected: dict[str, tuple[tuple[str, ...], ...]],
) -> DatasetBinding:
    issues: list[str] = []
    for key, accepted in expected.items():
        if key not in recipe.inputs:
            continue
        actual, issue = _canonical_many(recipe.inputs[key], f"inputs.{key}")
        if issue:
            issues.append(issue)
        elif actual not in accepted:
            choices = " or ".join(repr(list(choice)) for choice in accepted)
            issues.append(f"inputs.{key} asserts {list(actual)!r}, but the source stage configures {choices}")
    if not issues:
        return binding
    reason = "; ".join(issues)
    return replace(binding, status="mismatch", issues=tuple(issues), reason=reason)


def _assert_cli_data(
    binding: DatasetBinding,
    data: str | None,
    accepted: tuple[str, ...],
) -> DatasetBinding:
    if data is None:
        return binding
    canonical, issue = _canonical_scalar(data, "--data")
    if issue:
        return replace(binding, status="mismatch", issues=(issue,), reason=issue)
    if canonical is None:  # defensive: _canonical_scalar returns a path or an issue
        reason = "--data did not resolve to a path"
        return replace(binding, status="mismatch", issues=(reason,), reason=reason)
    if canonical in accepted:
        return binding
    reason = f"--data resolves to {canonical!r}, but the source stage configures {list(accepted)!r}"
    return replace(binding, status="mismatch", issues=(reason,), reason=reason)


def _resolve_manifest_reader(recipe: Recipe, stage: StageRef, data: str | None) -> DatasetBinding:  # noqa: C901
    configured_manifest_path = stage.params.get("manifest_path")
    paths, issue = _canonical_many(configured_manifest_path, "ManifestReader.manifest_path")
    if issue:
        return _binding("missing", source_ref=stage.ref, source_index=0, reason=issue)

    primary = paths[0] if len(paths) == 1 else None
    binding = _base(
        stage.ref,
        configured_paths=paths,
        primary_path=primary,
        # A URI, glob, or directory is a selector, not one manifest.  Passing a
        # directory to profile_data would incorrectly identify its audio files as
        # the input rather than the manifests selected by ManifestReader.
        profile_source=(
            primary
            if primary is not None
            and not _is_remote(primary)
            and not glob.has_magic(primary)
            and os.path.isfile(primary)
            and primary.lower().endswith((".jsonl", ".json"))
            else None
        ),
    )
    binding = _assert_recipe_inputs(binding, recipe, {"manifest_path": (paths,)})
    if binding.status == "mismatch":
        return binding

    if data is not None:
        binding = _assert_cli_data(binding, data, paths)
        if binding.status == "mismatch":
            return binding

    # ManifestReader supplies this two-extension default. An explicitly-authored
    # None reaches FilePartitioningStage, whose own default additionally accepts
    # parquet, so preserve that runtime distinction.
    file_extensions = stage.params.get("file_extensions", [".jsonl", ".json"])
    if file_extensions is None:
        file_extensions = [".jsonl", ".json", ".parquet"]
    storage_options = stage.params.get("storage_options")
    sort_by_size = stage.params.get("blocksize") is not None
    selected_records: list[tuple[str, int]] = []
    selector_issues_list: list[str] = []
    for index, path in enumerate(paths):
        if _is_remote(path):
            continue
        records, selection_issue = _local_manifest_records(
            path,
            label=(
                f"ManifestReader.manifest_path[{index}]"
                if isinstance(configured_manifest_path, list)
                else "ManifestReader.manifest_path"
            ),
            # FilePartitioningStage recurses for a scalar selector, but
            # inspects each explicitly listed selector only one level deep.
            recurse_subdirectories=not isinstance(configured_manifest_path, list),
            file_extensions=file_extensions,
            storage_options=storage_options,
            sort_by_size=sort_by_size,
        )
        if selection_issue:
            selector_issues_list.append(selection_issue)
        selected_records.extend(records)
    selector_issues = tuple(selector_issues_list)
    if selector_issues:
        reason = "; ".join(selector_issues)
        return replace(binding, status="missing", profile_source=None, issues=selector_issues, reason=reason)

    if all(not _is_remote(path) for path in paths):
        selected_records.sort(key=lambda item: item[1] if sort_by_size else item[0])
        binding = replace(binding, selected_manifest_files=tuple(path for path, _size in selected_records))

    if len(paths) > 1:
        reason = (
            "ManifestReader configures multiple manifest paths; a singular profile source "
            "cannot represent the complete dataset"
        )
        return replace(binding, status="ambiguous", profile_source=None, issues=(reason,), reason=reason)
    return binding


def _resolve_audio_folder(recipe: Recipe, stage: StageRef, data: str | None) -> DatasetBinding:
    path, issue, is_uri = _canonical_os_scalar(
        stage.params.get("data_dir"),
        f"{stage.ref}.data_dir",
    )
    if issue or path is None:
        reason = issue or f"{stage.ref}.data_dir did not resolve to a path"
        return _binding(
            "unsupported" if is_uri else "missing",
            source_ref=stage.ref,
            source_index=0,
            reason=reason,
        )

    binding = _base(stage.ref, configured_paths=(path,), primary_path=path, profile_source=path)
    binding = replace(
        binding,
        profile_kwargs={
            "folder_extensions": stage.params.get(
                "extensions",
                [".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a"],
            ),
            "recursive": stage.params.get("recursive", True),
            "max_files": stage.params.get("max_samples", -1),
            # The stage emits every matching file, including names that resemble
            # split-stage intermediates. Identity must cover those executable
            # bytes even though generic profiling excludes them by default.
            "exclude_stage_intermediates": False,
        },
    )
    binding = _assert_recipe_inputs(binding, recipe, {"data_dir": ((path,),)})
    binding = _assert_cli_data(binding, data, (path,))
    if binding.status == "mismatch":
        return binding
    if not os.path.isdir(path):
        reason = f"configured audio folder does not exist: {path}"
        return replace(binding, status="missing", profile_source=None, issues=(reason,), reason=reason)
    return binding


def _resolve_fleurs(  # noqa: PLR0911 - each status is an intentional fail-closed outcome
    recipe: Recipe,
    stage: StageRef,
    data: str | None,
) -> DatasetBinding:
    raw_dir, issue, is_uri = _canonical_os_scalar(
        stage.params.get("raw_data_dir"),
        f"{stage.ref}.raw_data_dir",
    )
    if issue or raw_dir is None:
        reason = issue or f"{stage.ref}.raw_data_dir did not resolve to a path"
        return _binding(
            "unsupported" if is_uri else "missing",
            source_ref=stage.ref,
            source_index=0,
            reason=reason,
        )
    lang = stage.params.get("lang")
    split = stage.params.get("split")
    if not isinstance(lang, str) or _is_unresolved(lang):
        reason = f"{stage.ref}.lang is missing or unresolved"
        return _binding("missing", source_ref=stage.ref, source_index=0, reason=reason)
    if not isinstance(split, str) or _is_unresolved(split):
        reason = f"{stage.ref}.split is missing or unresolved"
        return _binding("missing", source_ref=stage.ref, source_index=0, reason=reason)
    effective = canonical_source(os.path.join(raw_dir, lang))
    transcript = os.path.join(effective, f"{split}.tsv")
    audio_root = os.path.join(effective, split)
    staged = os.path.isfile(transcript) and os.path.isdir(audio_root)
    generated = not staged and bool(stage.params.get("auto_download", True))
    kwargs = (
        {
            "identity_files": [transcript],
            # FLEURS emits rows from its TSV verbatim; no filename heuristic may
            # omit an audio file that the source can hand to downstream stages.
            "exclude_stage_intermediates": False,
        }
        if staged
        else {}
    )
    binding = _base(
        stage.ref,
        configured_paths=(raw_dir,),
        primary_path=effective,
        profile_source=audio_root if staged else None,
        profile_kwargs=kwargs,
        generated=generated,
    )
    binding = _assert_recipe_inputs(binding, recipe, {"raw_data_dir": ((raw_dir,),)})
    binding = _assert_cli_data(binding, data, (raw_dir, effective))
    if binding.status == "mismatch":
        return binding
    if staged:
        return replace(binding, reason="pre-staged FLEURS dataset resolved")
    if generated:
        reason = f"FLEURS dataset is not staged at {effective}, but the source can generate it with auto_download=True"
        return replace(binding, issues=(reason,), reason=reason)
    reason = f"pre-staged FLEURS dataset is incomplete at {effective} and auto_download=False"
    return replace(binding, status="missing", issues=(reason,), reason=reason)


def _find_readspeech_wavs(search_dir: str) -> str | None:
    """Mirror the source stage's read-only extracted-directory discovery."""
    if not os.path.exists(search_dir):
        return None
    if glob.glob(os.path.join(search_dir, "*.wav")):
        return search_dir
    for subdir in ("read_speech", "mnt/dnsv5/clean/read_speech", "data/mnt/dnsv5/clean/read_speech"):
        candidate = os.path.join(search_dir, subdir)
        if os.path.exists(candidate) and glob.glob(os.path.join(candidate, "*.wav")):
            return candidate
    for root, _dirs, files in os.walk(search_dir):
        if any(name.endswith(".wav") for name in files):
            return root
    return None


def _resolve_readspeech(recipe: Recipe, stage: StageRef, data: str | None) -> DatasetBinding:
    raw_dir, issue, is_uri = _canonical_os_scalar(
        stage.params.get("raw_data_dir"),
        f"{stage.ref}.raw_data_dir",
    )
    if issue or raw_dir is None:
        reason = issue or f"{stage.ref}.raw_data_dir did not resolve to a path"
        return _binding(
            "unsupported" if is_uri else "missing",
            source_ref=stage.ref,
            source_index=0,
            reason=reason,
        )

    effective = _find_readspeech_wavs(raw_dir)
    generated = effective is None and bool(stage.params.get("auto_download", True))
    primary = effective or raw_dir
    max_samples = stage.params.get("max_samples", 5000)
    max_files = max_samples if isinstance(max_samples, int) and max_samples > 0 else None
    binding = _base(
        stage.ref,
        configured_paths=(raw_dir,),
        primary_path=primary,
        profile_source=effective,
        profile_kwargs=(
            {
                "folder_extensions": [".wav"],
                "recursive": True,
                "max_files": max_files,
                "case_sensitive_extensions": True,
                "exclude_stage_intermediates": False,
            }
            if effective
            else {}
        ),
        generated=generated,
    )
    binding = _assert_recipe_inputs(binding, recipe, {"raw_data_dir": ((raw_dir,),)})
    binding = _assert_cli_data(binding, data, tuple(dict.fromkeys((raw_dir, primary))))
    if binding.status == "mismatch":
        return binding
    if effective is not None:
        return replace(binding, reason="pre-staged ReadSpeech dataset resolved")
    if generated:
        reason = (
            f"ReadSpeech dataset is not staged at {raw_dir}, but the source can generate it with auto_download=True"
        )
        return replace(binding, issues=(reason,), reason=reason)
    reason = f"no WAV files were found under {raw_dir} and auto_download=False"
    return replace(binding, status="missing", issues=(reason,), reason=reason)


def _resolve_long_form(
    recipe: Recipe,
    stage: StageRef,
    data: str | None,
) -> DatasetBinding:
    manifest, manifest_issue, manifest_is_uri = _canonical_os_scalar(
        stage.params.get("input_manifest"),
        f"{stage.ref}.input_manifest",
    )
    audio_dir, audio_issue, audio_is_uri = _canonical_os_scalar(
        stage.params.get("audio_dir"),
        f"{stage.ref}.audio_dir",
    )
    if manifest_issue or audio_issue or manifest is None or audio_dir is None:
        reason = "; ".join(issue for issue in (manifest_issue, audio_issue) if issue) or (
            f"{stage.ref} paths did not resolve"
        )
        return _binding(
            "unsupported" if manifest_is_uri or audio_is_uri else "missing",
            source_ref=stage.ref,
            source_index=0,
            reason=reason,
        )
    resolution = stage.params.get("audio_path_resolution", "basename")
    if resolution not in _LONG_FORM_RESOLUTIONS:
        reason = (
            f"{stage.ref}.audio_path_resolution must be one of {sorted(_LONG_FORM_RESOLUTIONS)!r}, got {resolution!r}"
        )
        return _binding(
            "unsupported",
            source_ref=stage.ref,
            configured_paths=(manifest, audio_dir),
            primary_path=manifest,
            source_index=0,
            reason=reason,
        )

    kwargs = {
        "audio_dir": audio_dir,
        "audio_path_resolution": resolution,
        "audio_filepath_key": stage.params.get("audio_filepath_key", "audio_filepath"),
    }
    binding = _base(
        stage.ref,
        configured_paths=(manifest, audio_dir),
        primary_path=manifest,
        profile_source=manifest,
        profile_kwargs=kwargs,
    )
    binding = _assert_recipe_inputs(
        binding,
        recipe,
        {
            "input_manifest": ((manifest,),),
            "manifest_path": ((manifest,),),
            "audio_dir": ((audio_dir,),),
        },
    )
    binding = _assert_cli_data(binding, data, (manifest,))
    if binding.status == "mismatch":
        return binding
    if not os.path.isfile(manifest):
        reason = f"long-form input manifest does not exist: {manifest}"
        return replace(binding, status="missing", profile_source=None, issues=(reason,), reason=reason)
    if resolution != "as_is" and not os.path.isdir(audio_dir):
        reason = f"long-form audio_dir does not exist for {resolution!r} path resolution: {audio_dir}"
        return replace(binding, status="missing", profile_source=None, issues=(reason,), reason=reason)
    return binding
