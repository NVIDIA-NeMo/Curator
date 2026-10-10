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

"""Read NeMo YAML speech configurations and tarred/non-tarred manifests.

The public ``NeMoSpeechAudioReader`` decomposes into a source discovery stage
and a shard reader.  A physical manifest (and its matching tar, when present)
is one ``FileGroupTask``.  This preserves a stable source boundary for
``Pipeline.run(checkpoint_path=...)`` and for per-shard output markers.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
from loguru import logger

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.audio.io._nemo_speech_paths import contained_output_path, validate_local_output_dir
from nemo_curator.stages.audio.io._nemo_speech_state import register_nemo_speech_shard
from nemo_curator.stages.audio.io.shard_key import derive_manifest_shard_key
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.tasks import AudioTask, EmptyTask, FileGroupTask
from nemo_curator.utils.hash_utils import get_deterministic_hash

_SUPPORTED_TYPES = frozenset({"nemo", "nemo_tarred"})
_ROW_INDEX_FIELD = "_nemo_curator_row_index"
_WAVEFORM_2D = 2


def _as_plain_container(value: Any) -> Any:  # noqa: ANN401
    """Resolve an OmegaConf value when OmegaConf is available."""

    try:
        from omegaconf import OmegaConf
    except ImportError:
        return value
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _load_input_cfg(yaml_path: str | None, input_cfg: Any = None) -> list[dict[str, Any]]:  # noqa: ANN401
    """Load an ``input_cfg`` list; an inline value takes precedence."""

    if input_cfg is not None:
        loaded = _as_plain_container(input_cfg)
    elif yaml_path:
        from omegaconf import OmegaConf

        with fsspec.open(yaml_path, mode="rt", encoding="utf-8") as stream:
            loaded = OmegaConf.to_container(OmegaConf.create(stream.read()), resolve=True)
    else:
        msg = "Either input_cfg or yaml_path must be provided"
        raise ValueError(msg)

    if isinstance(loaded, dict) and "input_cfg" in loaded:
        loaded = [loaded]
    if not isinstance(loaded, list) or not loaded:
        msg = f"Expected a non-empty NeMo input_cfg list, got {type(loaded).__name__}"
        raise ValueError(msg)
    if not all(isinstance(group, dict) for group in loaded):
        msg = "Every NeMo input_cfg group must be a mapping"
        raise ValueError(msg)
    return loaded


def _expand_sharded_path(path: str | list[str]) -> list[str]:
    """Expand NeMo brace/range syntax without importing NeMo at module import."""

    from nemo.collections.common.data.lhotse.nemo_adapters import expand_sharded_filepaths

    expanded = expand_sharded_filepaths(path)
    return [str(item) for item in expanded]


def _descriptors_from_entry(entry: dict[str, Any]) -> list[dict[str, Any]]:
    manifest_value = entry.get("manifest_filepath")
    if not manifest_value:
        msg = f"NeMo input_cfg entry is missing manifest_filepath: {entry!r}"
        raise ValueError(msg)

    entry_type = entry.get("type") or ("nemo_tarred" if entry.get("tarred_audio_filepaths") else "nemo")
    if entry_type not in _SUPPORTED_TYPES:
        msg = f"Unsupported NeMo speech input type {entry_type!r}; expected one of {sorted(_SUPPORTED_TYPES)}"
        raise ValueError(msg)

    manifests = _expand_sharded_path(manifest_value)
    tar_value = entry.get("tarred_audio_filepaths")
    if entry_type == "nemo_tarred" and not tar_value:
        msg = "A nemo_tarred input_cfg entry requires tarred_audio_filepaths"
        raise ValueError(msg)
    if entry_type == "nemo" and tar_value:
        msg = "tarred_audio_filepaths requires type: nemo_tarred"
        raise ValueError(msg)

    tars = _expand_sharded_path(tar_value) if tar_value else [None] * len(manifests)
    if len(manifests) != len(tars):
        msg = f"Manifest/tar shard count mismatch: {len(manifests)} manifests and {len(tars)} tar files"
        raise ValueError(msg)

    return [
        {
            "corpus": str(entry.get("corpus") or "unknown"),
            "language": str(entry.get("language") or ""),
            "manifest_path": manifest_path,
            "tar_path": tar_path,
            "shard_key_prefix": entry.get("shard_key_prefix"),
        }
        for manifest_path, tar_path in zip(manifests, tars, strict=True)
    ]


def _parse_input_cfg(
    config: list[dict[str, Any]],
    corpus_filter: list[str] | None,
    language_filter: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Flatten and filter NeMo ``input_cfg`` groups into physical shards."""

    corpus_set = set(corpus_filter or [])
    language_set = set(language_filter or [])
    descriptors: list[dict[str, Any]] = []
    for group in config:
        entries = group.get("input_cfg", [group])
        if not isinstance(entries, list):
            msg = "The input_cfg field must be a list"
            raise TypeError(msg)
        for entry in entries:
            if not isinstance(entry, dict):
                msg = "Every input_cfg entry must be a mapping"
                raise TypeError(msg)
            corpus = str(entry.get("corpus") or "unknown")
            language = str(entry.get("language") or "")
            if corpus_set and corpus not in corpus_set:
                continue
            if language_set and language not in language_set:
                continue
            descriptors.extend(_descriptors_from_entry(entry))
    return descriptors


@dataclass
class _NeMoSpeechShardTask(FileGroupTask):
    """File group whose source ID also includes its logical output shard."""

    def get_deterministic_id(self) -> str:
        shard_key = str(self.reader_config.get("shard_key") or "")
        return get_deterministic_hash(self.data, seed=f"nemo-speech:{shard_key}")


def _receipt_dir(output_dir: Path, shard_key: str) -> Path:
    return contained_output_path(output_dir, f".nemo_curator/nemo_speech_rows/{shard_key}")


def _manifest_stats(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    rows = 0
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            rows += bool(line.strip())
    return rows, digest.hexdigest()


@dataclass
class NeMoSpeechDiscoveryStage(ProcessingStage[EmptyTask, FileGroupTask]):
    """Expand NeMo YAML entries into one stable source task per shard.

    ``output_dir`` enables artifact-level restart semantics.  Shards with a
    ``<shard>.jsonl.done`` marker are skipped.  For an incomplete shard, its
    partial JSONL and progress record are removed before replay; atomically
    written Opus files are retained and reused by the writer.

    Set ``resume_mode="checkpoint"`` when passing ``checkpoint_path`` to
    ``Pipeline.run()``. In that mode, Curator's source checkpoint is the sole
    scheduling authority and output markers are integrity records only. Do not
    combine checkpoint scheduling with ``resume_mode="done_markers"``.
    """

    yaml_path: str = ""
    input_cfg: Any = None
    corpus_filter: list[str] | None = None
    language_filter: list[str] | None = None
    output_dir: str | None = None
    cleanup_partial: bool = True
    resume_mode: str = "done_markers"
    name: str = "nemo_speech_discovery"
    batch_size: int = 1

    def __post_init__(self) -> None:
        if not self.yaml_path and self.input_cfg is None:
            msg = "Either input_cfg or yaml_path is required for NeMo speech discovery"
            raise ValueError(msg)
        if self.resume_mode not in {"done_markers", "checkpoint"}:
            msg = "resume_mode must be 'done_markers' or 'checkpoint'"
            raise ValueError(msg)
        if self.output_dir is not None:
            validate_local_output_dir(self.output_dir)
        elif self.resume_mode == "done_markers":
            msg = "output_dir is required when resume_mode='done_markers'"
            raise ValueError(msg)
        # Pipeline.run(checkpoint_path=...) checks this before discovery can
        # clean artifact state. This makes the two authorities code-enforced.
        self.is_resumable = self.resume_mode == "checkpoint"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data", "reader_config"], []

    def num_workers_per_node(self) -> float | None:
        return 1

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}

    def _completed_shards(self) -> set[str]:
        if self.resume_mode != "done_markers" or not self.output_dir:
            return set()
        root = Path(self.output_dir)
        if not root.is_dir():
            return set()
        completed: set[str] = set()
        for discovered_marker in root.rglob("*.jsonl.done"):
            marker_relative = discovered_marker.relative_to(root).as_posix()
            marker = contained_output_path(root, marker_relative)
            shard_key = marker_relative[: -len(".jsonl.done")]
            manifest = contained_output_path(root, f"{shard_key}.jsonl")
            if not manifest.is_file():
                logger.warning(f"Ignoring completion marker without a manifest: {marker}")
                continue
            raw = marker.read_text(encoding="utf-8").strip()
            if not raw:
                # Compatibility with the reference implementation's empty marker.
                logger.warning(f"Accepting legacy empty NeMo speech completion marker: {marker}")
                completed.add(shard_key)
                continue
            try:
                payload = json.loads(raw)
                manifest_rows, manifest_sha256 = _manifest_stats(manifest)
                valid = isinstance(payload, dict) and (
                    payload.get("version") == 1
                    and payload.get("shard_key") == shard_key
                    and int(payload.get("expected_inputs") or 0) > 0
                    and payload.get("completed_inputs") == payload.get("expected_inputs")
                    and payload.get("manifest_rows") == manifest_rows
                    and payload.get("manifest_sha256") == manifest_sha256
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                valid = False
            if valid:
                completed.add(shard_key)
            else:
                logger.warning(f"Ignoring invalid NeMo speech completion marker: {marker}")
        return completed

    def _clean_partial_state(self, shard_key: str) -> None:
        if self.resume_mode != "done_markers" or not self.output_dir or not self.cleanup_partial:
            return
        root = Path(self.output_dir)
        partial = contained_output_path(root, f"{shard_key}.jsonl")
        legacy_progress = contained_output_path(root, f".{shard_key.replace('/', '_')}.shard_progress.json")
        for path in (partial, legacy_progress):
            if path.is_file():
                path.unlink()
                logger.info(f"Removed incomplete NeMo speech state: {path}")
        receipts = _receipt_dir(root, shard_key)
        if receipts.is_dir():
            removed = False
            for receipt in receipts.glob("*.json"):
                receipt.unlink()
                removed = True
            with contextlib.suppress(OSError):
                receipts.rmdir()
            if removed:
                logger.info(f"Removed incomplete NeMo speech row receipts: {receipts}")

    def process(self, _task: EmptyTask) -> list[FileGroupTask]:
        config = _load_input_cfg(self.yaml_path or None, self.input_cfg)
        descriptors = _parse_input_cfg(config, self.corpus_filter, self.language_filter)
        completed = self._completed_shards()
        keyed_descriptors: list[tuple[str, dict[str, Any]]] = []
        manifests_by_key: dict[str, str] = {}
        for descriptor in descriptors:
            shard_key = derive_manifest_shard_key(
                descriptor["manifest_path"],
                descriptor["corpus"],
                shard_key_prefix=descriptor["shard_key_prefix"],
            )
            previous_manifest = manifests_by_key.get(shard_key)
            if previous_manifest is not None:
                msg = (
                    f"NeMo input_cfg entries produce duplicate shard key {shard_key!r}: "
                    f"{previous_manifest!r} and {descriptor['manifest_path']!r}. "
                    "Set distinct shard_key_prefix values."
                )
                raise ValueError(msg)
            manifests_by_key[shard_key] = descriptor["manifest_path"]
            keyed_descriptors.append((shard_key, descriptor))

        tasks: list[FileGroupTask] = []
        for shard_key, descriptor in keyed_descriptors:
            if shard_key in completed:
                logger.info(f"Skipping completed NeMo speech shard: {shard_key}")
                continue
            self._clean_partial_state(shard_key)
            paths = [descriptor["manifest_path"]]
            if descriptor["tar_path"]:
                paths.append(descriptor["tar_path"])
            tasks.append(
                _NeMoSpeechShardTask(
                    dataset_name=descriptor["corpus"],
                    data=paths,
                    reader_config={
                        "corpus": descriptor["corpus"],
                        "language": descriptor["language"],
                        "manifest_path": descriptor["manifest_path"],
                        "tar_path": descriptor["tar_path"],
                        "shard_key": shard_key,
                    },
                )
            )
        logger.info(f"Discovered {len(tasks)} NeMo speech shards; skipped {len(descriptors) - len(tasks)} completed")
        return tasks


def _read_manifest_rows(manifest_path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with fsspec.open(manifest_path, mode="rt", encoding="utf-8", compression="infer") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                msg = f"Expected an object at {manifest_path}:{line_number}"
                raise TypeError(msg)
            rows.append(value)
    return rows


@dataclass
class NeMoSpeechReaderStage(ProcessingStage[FileGroupTask, AudioTask]):
    """Decode one NeMo manifest/tar shard into mono ``AudioTask`` objects."""

    output_dir: str | None = None
    max_audio_duration_sec: float | None = 12 * 60 * 60
    keep_waveform: bool = True
    reader_workers: int | None = None
    name: str = "nemo_speech_reader"
    batch_size: int = 1

    def __post_init__(self) -> None:
        if self.output_dir is not None:
            validate_local_output_dir(self.output_dir)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data", "reader_config"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        columns = [
            "audio_filepath",
            "original_file",
            "sampling_rate",
            "sample_rate",
            "duration",
            "num_channels",
            "corpus",
        ]
        if self.keep_waveform:
            columns.append("waveform")
        return ["data"], columns

    def num_workers(self) -> int | None:
        return self.reader_workers

    @staticmethod
    def _make_tar_cutset(manifest_path: str, tar_path: str) -> Any:  # noqa: ANN401
        from lhotse import CutSet
        from nemo.collections.common.data.lhotse.nemo_adapters import LazyNeMoTarredIterator

        return CutSet(
            LazyNeMoTarredIterator(
                manifest_path=manifest_path,
                tar_paths=tar_path,
                skip_missing_manifest_entries=True,
            )
        )

    @staticmethod
    def _make_non_tar_adapter(manifest_path: str) -> Any:  # noqa: ANN401
        from nemo.collections.common.data.lhotse.nemo_adapters import LazyNeMoIterator

        return LazyNeMoIterator(manifest_path)

    def _over_duration_limit(self, duration: Any) -> bool:  # noqa: ANN401
        if self.max_audio_duration_sec is None or self.max_audio_duration_sec <= 0:
            return False
        try:
            return float(duration) > self.max_audio_duration_sec
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _source_path(cut: Any) -> str:  # noqa: ANN401
        if cut.recording is not None and cut.recording.sources:
            source = cut.recording.sources[0].source
            if isinstance(source, str):
                return source
        return str(cut.id)

    def _placeholder(
        self,
        *,
        task: FileGroupTask,
        row: dict[str, Any],
        row_index: int,
        shard_total: int,
        audio_too_long: bool = False,
    ) -> AudioTask:
        shard_key = task.reader_config["shard_key"]
        audio_path = str(row.get("audio_filepath") or row.get("audio_filename") or f"missing_entry_{row_index}")
        data = {key: value for key, value in row.items() if key != "audio_filepath"}
        data.update(
            {
                "audio_filepath": audio_path,
                "original_file": audio_path,
                "corpus": task.reader_config["corpus"],
                "read_error": True,
            }
        )
        if audio_too_long:
            data["audio_too_long"] = True
        language = task.reader_config.get("language")
        if language:
            data.setdefault("source_lang", language)
        return AudioTask(
            dataset_name=task.dataset_name,
            data=data,
            _metadata={
                **task._metadata,
                "_shard_key": shard_key,
                "_shard_total": shard_total,
                "_shard_input_id": f"{row_index}",
            },
            _stage_perf=list(task._stage_perf),
        )

    def _decode_cut(
        self,
        *,
        cut: Any,  # noqa: ANN401
        task: FileGroupTask,
        row: dict[str, Any],
        row_index: int,
        shard_total: int,
    ) -> AudioTask:
        if self._over_duration_limit(cut.duration):
            return self._placeholder(
                task=task,
                row=row,
                row_index=row_index,
                shard_total=shard_total,
                audio_too_long=True,
            )

        waveform = np.asarray(cut.load_audio(), dtype=np.float32)
        if waveform.ndim == _WAVEFORM_2D:
            waveform = waveform.mean(axis=0)
        elif waveform.ndim != 1:
            waveform = waveform.squeeze()
        if waveform.ndim != 1:
            msg = f"Decoded cut {cut.id!r} has unsupported waveform shape {waveform.shape}"
            raise ValueError(msg)

        sample_rate = int(cut.recording.sampling_rate)
        source_path = self._source_path(cut)
        custom = dict(cut.custom or {})
        data = {**row, **custom}
        public_audio_path = str(row.get("audio_filepath") or source_path)
        original_file = public_audio_path if task.reader_config.get("tar_path") else source_path
        data.update(
            {
                "audio_filepath": public_audio_path,
                "original_file": original_file,
                "sampling_rate": sample_rate,
                "sample_rate": sample_rate,
                "duration": float(cut.duration),
                "num_channels": 1,
                "corpus": task.reader_config["corpus"],
            }
        )
        if self.keep_waveform:
            data["waveform"] = waveform
        language = task.reader_config.get("language")
        if language:
            data.setdefault("source_lang", language)
        return AudioTask(
            dataset_name=task.dataset_name,
            data=data,
            _metadata={
                **task._metadata,
                "_shard_key": task.reader_config["shard_key"],
                "_shard_total": shard_total,
                "_shard_input_id": f"{row_index}",
            },
            _stage_perf=list(task._stage_perf),
        )

    def _decode_or_placeholder(
        self,
        *,
        cut: Any,  # noqa: ANN401
        task: FileGroupTask,
        row: dict[str, Any],
        row_index: int,
        shard_total: int,
    ) -> AudioTask:
        shard_key = task.reader_config["shard_key"]
        try:
            return self._decode_cut(
                cut=cut,
                task=task,
                row=row,
                row_index=row_index,
                shard_total=shard_total,
            )
        except Exception as error:  # noqa: BLE001
            logger.warning(f"Could not decode {shard_key} row {row_index}: {error}")
            return self._placeholder(
                task=task,
                row=row,
                row_index=row_index,
                shard_total=shard_total,
            )

    def _process_non_tar(
        self,
        task: FileGroupTask,
        rows: list[dict[str, Any]],
    ) -> list[AudioTask]:
        manifest_path = task.reader_config["manifest_path"]
        shard_total = len(rows)
        adapter = self._make_non_tar_adapter(manifest_path)
        results: list[AudioTask] = []
        for row_index, row in enumerate(rows):
            if self._over_duration_limit(row.get("duration")):
                results.append(
                    self._placeholder(
                        task=task,
                        row=row,
                        row_index=row_index,
                        shard_total=shard_total,
                        audio_too_long=True,
                    )
                )
                continue
            adapter_row = dict(row)
            if row.get("sampling_rate") is not None:
                try:
                    offset = float(row.get("offset", 0.0))
                    duration = float(row["duration"])
                except (KeyError, TypeError, ValueError):
                    pass
                else:
                    if offset > 0.0 and duration > 0.0:
                        # LazyNeMoIterator treats ``duration`` as both the
                        # recording duration and requested segment duration
                        # when ``sampling_rate`` avoids probing the file. Give
                        # its synthetic recording enough extent for the
                        # positive offset. Lhotse clamps the resulting cut to
                        # the remaining ``duration`` samples, while ``row``
                        # below retains the original manifest semantics.
                        adapter_row["duration"] = offset + duration
            adapter.source = [adapter_row]
            try:
                cuts = iter(adapter)
                cut = next(cuts)
            except Exception as error:  # noqa: BLE001
                logger.warning(f"Could not construct {task.reader_config['shard_key']} row {row_index}: {error}")
                results.append(
                    self._placeholder(
                        task=task,
                        row=row,
                        row_index=row_index,
                        shard_total=shard_total,
                    )
                )
                continue
            results.append(
                self._decode_or_placeholder(
                    cut=cut,
                    task=task,
                    row=row,
                    row_index=row_index,
                    shard_total=shard_total,
                )
            )
        return results

    def _process_tarred(
        self,
        task: FileGroupTask,
        rows: list[dict[str, Any]],
        tar_path: str,
    ) -> list[AudioTask]:
        manifest_path = task.reader_config["manifest_path"]
        shard_total = len(rows)
        if any(_ROW_INDEX_FIELD in row for row in rows):
            msg = f"Manifest {manifest_path!r} uses reserved field {_ROW_INDEX_FIELD!r}"
            raise ValueError(msg)

        results: list[AudioTask | None] = [None] * shard_total
        manifest_name = Path(manifest_path.split("?", 1)[0]).name or "manifest.json"
        manifest_name = manifest_name.removesuffix(".gz")
        if not manifest_name.endswith((".json", ".jsonl")):
            manifest_name = "manifest.json"
        with tempfile.TemporaryDirectory(prefix="nemo-curator-tar-manifest-") as temp_dir:
            indexed_manifest = Path(temp_dir) / manifest_name
            with indexed_manifest.open("w", encoding="utf-8") as stream:
                for row_index, row in enumerate(rows):
                    indexed_row = {**row, _ROW_INDEX_FIELD: row_index}
                    stream.write(json.dumps(indexed_row, ensure_ascii=False) + "\n")

            cutset = self._make_tar_cutset(str(indexed_manifest), tar_path)
            for cut in cutset:
                custom = dict(cut.custom or {})
                row_index = custom.pop(_ROW_INDEX_FIELD, None)
                if isinstance(row_index, bool) or not isinstance(row_index, int):
                    msg = f"Tarred NeMo cut {cut.id!r} is missing a valid {_ROW_INDEX_FIELD}"
                    raise TypeError(msg)
                if not 0 <= row_index < shard_total:
                    msg = f"Tarred NeMo cut {cut.id!r} has out-of-range row index {row_index}"
                    raise ValueError(msg)
                if results[row_index] is not None:
                    msg = f"Tarred NeMo manifest row {row_index} was emitted more than once"
                    raise ValueError(msg)
                # The adapter saw our temporary indexed manifest. Restore the
                # user-facing provenance before it is merged into the output.
                custom["manifest_origin"] = manifest_path
                cut.custom = custom
                results[row_index] = self._decode_or_placeholder(
                    cut=cut,
                    task=task,
                    row=rows[row_index],
                    row_index=row_index,
                    shard_total=shard_total,
                )

        for row_index, result in enumerate(results):
            if result is None:
                results[row_index] = self._placeholder(
                    task=task,
                    row=rows[row_index],
                    row_index=row_index,
                    shard_total=shard_total,
                )
        return [result for result in results if result is not None]

    def process(self, task: FileGroupTask) -> list[AudioTask]:
        manifest_path = task.reader_config["manifest_path"]
        tar_path = task.reader_config.get("tar_path")
        rows = _read_manifest_rows(manifest_path)
        shard_total = len(rows)
        shard_key = task.reader_config["shard_key"]
        if shard_total == 0:
            msg = f"NeMo speech manifest {manifest_path!r} contains no rows; empty shards are unsupported"
            raise ValueError(msg)
        if self.output_dir is not None:
            register_nemo_speech_shard(Path(self.output_dir), shard_key, shard_total)
        results = self._process_tarred(task, rows, tar_path) if tar_path else self._process_non_tar(task, rows)
        logger.info(f"Read {len(results)}/{shard_total} entries from NeMo speech shard {shard_key}")
        return results


@dataclass
class NeMoSpeechAudioReader(CompositeStage[EmptyTask, AudioTask]):
    """User-facing reader for NeMo ``nemo`` and ``nemo_tarred`` input_cfg entries.

    ``yaml_path`` and inline ``input_cfg`` are mutually substitutable; inline
    configuration takes precedence and supports resolved OmegaConf values.
    Corpus and language filters are applied before NeMo brace expansion is
    emitted as one stable source task per physical manifest/tar pair.

    ``resume_mode="done_markers"`` reproduces artifact-driven restart behavior
    and requires ``output_dir`` to find finalized shards. Use
    ``resume_mode="checkpoint"`` instead when the pipeline runs with Curator's
    ``checkpoint_path`` argument. The two scheduling mechanisms must not be
    enabled together.
    """

    yaml_path: str = ""
    input_cfg: Any = None
    corpus_filter: list[str] | None = None
    language_filter: list[str] | None = None
    output_dir: str | None = None
    cleanup_partial: bool = True
    resume_mode: str = "done_markers"
    max_audio_duration_sec: float | None = 12 * 60 * 60
    keep_waveform: bool = True
    reader_workers: int | None = None
    name: str = "nemo_speech_audio_reader"
    _stages: list[ProcessingStage] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__init__()
        self._stages = [
            NeMoSpeechDiscoveryStage(
                yaml_path=self.yaml_path,
                input_cfg=self.input_cfg,
                corpus_filter=self.corpus_filter,
                language_filter=self.language_filter,
                output_dir=self.output_dir,
                cleanup_partial=self.cleanup_partial,
                resume_mode=self.resume_mode,
            ),
            NeMoSpeechReaderStage(
                output_dir=self.output_dir,
                max_audio_duration_sec=self.max_audio_duration_sec,
                keep_waveform=self.keep_waveform,
                reader_workers=self.reader_workers,
            ),
        ]

    def decompose(self) -> list[ProcessingStage]:
        return self._stages


__all__ = [
    "NeMoSpeechAudioReader",
    "NeMoSpeechDiscoveryStage",
    "NeMoSpeechReaderStage",
]
