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

import itertools
import json
import pickle
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor
from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.utils.file_utils import get_all_file_paths_and_size_under

_executor_map = {"ray_data": RayDataExecutor, "xenna": XennaExecutor, "ray_actors": RayActorPoolExecutor}


def setup_executor(executor_name: str) -> RayDataExecutor | XennaExecutor | RayActorPoolExecutor:
    """Setup the executor for the given name."""
    try:
        executor = _executor_map[executor_name]()
    except KeyError:
        msg = f"Executor {executor_name} not supported"
        raise ValueError(msg) from None
    return executor


def load_dataset_files(
    dataset_path: Path,
    dataset_size_gb: float | None = None,
    dataset_ratio: float | None = None,
    keep_extensions: str = "parquet",
) -> list[str]:
    """Load the dataset files at the given path and return a subset of the files whose combined size is approximately the given size in GB."""
    input_files = get_all_file_paths_and_size_under(
        dataset_path, recurse_subdirectories=True, keep_extensions=keep_extensions
    )
    if (not dataset_size_gb and not dataset_ratio) or (dataset_size_gb and dataset_ratio):
        msg = "Either dataset_size_gb or dataset_ratio must be provided, but not both"
        raise ValueError(msg)
    if dataset_size_gb:
        desired_size_bytes = (1024**3) * dataset_size_gb
    else:
        total_file_size_bytes = sum(size for _, size in input_files)
        desired_size_bytes = total_file_size_bytes * dataset_ratio

    total_size = 0
    subset_files = []
    for file, size in input_files:
        if size + total_size > desired_size_bytes:
            break
        else:
            subset_files.append(file)
            total_size += size

    return subset_files


def write_benchmark_results(results: dict, output_path: str | Path) -> None:
    """Write benchmark results (params, metrics, tasks) to the appropriate files in the output directory.

    - Writes 'params.json' and 'metrics.json' (merging with existing file contents if present and updating values).
    - Writes 'tasks.pkl' as a pickle file if present in results.
    - The output directory is created if it does not exist.

    Typically used by benchmark scripts to persist results in the format expected by the benchmarking framework.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    if "params" in results:
        params_path = output_path / "params.json"
        params_data = {}
        if params_path.exists():
            params_data = json.loads(params_path.read_text())
        params_data.update(results["params"])
        params_path.write_text(json.dumps(params_data, default=convert_paths_to_strings, indent=2))
    if "metrics" in results:
        metrics_path = output_path / "metrics.json"
        metrics_data = {}
        if metrics_path.exists():
            metrics_data = json.loads(metrics_path.read_text())
        metrics_data.update(results["metrics"])
        metrics_path.write_text(json.dumps(metrics_data, default=convert_paths_to_strings, indent=2))
    if "tasks" in results:
        (output_path / "tasks.pkl").write_bytes(pickle.dumps(results["tasks"]))


def _collect_file_size_metrics(output_path: Path, extensions: list[str]) -> tuple[list[str], int, int]:
    """Return (file_paths, num_files, total_size_bytes) for files matching extensions under output_path."""
    output_files_with_size = get_all_file_paths_and_size_under(
        str(output_path),
        recurse_subdirectories=True,
        keep_extensions=extensions,
    )
    file_paths = [path for path, _ in output_files_with_size]
    total_size_bytes = int(sum(size for _, size in output_files_with_size))
    return file_paths, len(file_paths), total_size_bytes


def _resolve_paths(path: Path, extension: str) -> tuple[list[str], int, int]:
    """Return (file_paths, num_files, total_size_bytes) for a single file or a directory."""
    if path.is_file():
        return [str(path)], 1, path.stat().st_size
    return _collect_file_size_metrics(path, [extension])


def _accumulate_modality_counts(column: pa.ChunkedArray, into: dict[str, int]) -> None:
    """Accumulate value_counts from a modality column into `into`."""
    for row in column.value_counts().to_pylist():
        key = str(row["values"]) if row["values"] is not None else "None"
        into[key] = into.get(key, 0) + int(row["counts"])


def collect_interleaved_parquet_metrics(path: Path | str) -> dict[str, Any]:
    """Collect metrics for interleaved parquet files — neutral keys, caller adds input_/output_ prefix."""
    parquet_files, num_files, total_size_bytes = _resolve_paths(Path(path), ".parquet")
    num_rows = 0
    num_samples = 0
    modality_counts: dict[str, int] = {}
    materialize_error_count = 0
    for pq_path in parquet_files:
        pf = pq.ParquetFile(pq_path)
        num_rows += pf.metadata.num_rows
        schema_names = set(pf.schema_arrow.names)
        cols = [c for c in ("sample_id", "modality", "materialize_error") if c in schema_names]
        if not cols:
            continue
        cols_set = set(cols)
        table = pf.read(columns=cols)
        if "sample_id" in cols_set:
            num_samples += pc.count_distinct(table.column("sample_id")).as_py()
        if "modality" in cols_set:
            _accumulate_modality_counts(table.column("modality"), modality_counts)
        if "materialize_error" in cols_set:
            col = table.column("materialize_error")
            materialize_error_count += col.length() - col.null_count
    return {
        "num_files": num_files,
        "total_bytes": total_size_bytes,
        "total_mb": total_size_bytes / 1e6,
        "num_rows": num_rows,
        "num_samples": num_samples,
        "num_metadata": modality_counts.get("metadata", 0),
        "num_texts": modality_counts.get("text", 0),
        "num_images": modality_counts.get("image", 0),
        "modality_counts": modality_counts,
        "materialize_error_count": materialize_error_count,
    }


def _collect_wds_modality_counts(tar_paths: list[str]) -> tuple[int, dict[str, int]]:
    """Return (num_samples, modality_counts) by reading JSON metadata from WDS tars.

    Counts metadata (one per sample), text (non-null texts entries), and image
    (non-null images entries) rows.
    """
    counts: dict[str, int] = {}
    for path in tar_paths:
        with tarfile.open(path) as tf:
            for m in tf.getmembers():
                if not m.name.endswith(".json"):
                    continue
                raw = tf.extractfile(m)
                if raw is None:
                    continue
                payload = json.loads(raw.read())
                counts["metadata"] = counts.get("metadata", 0) + 1
                text_count = sum(1 for t in payload.get("texts", []) if t is not None)
                image_count = sum(1 for img in payload.get("images", []) if img is not None)
                if text_count:
                    counts["text"] = counts.get("text", 0) + text_count
                if image_count:
                    counts["image"] = counts.get("image", 0) + image_count
    return counts.get("metadata", 0), counts


def collect_interleaved_wds_metrics(path: Path | str) -> dict[str, Any]:
    """Collect metrics for interleaved WebDataset tar archives — neutral keys, caller adds input_/output_ prefix."""
    tar_paths, num_files, total_size_bytes = _resolve_paths(Path(path), ".tar")
    num_samples, modality_counts = _collect_wds_modality_counts(tar_paths)
    total_rows = sum(modality_counts.values())
    return {
        "num_files": num_files,
        "total_bytes": total_size_bytes,
        "total_mb": total_size_bytes / 1e6,
        "num_rows": total_rows,
        "num_samples": num_samples,
        "num_texts": modality_counts.get("text", 0),
        "num_images": modality_counts.get("image", 0),
        "modality_counts": modality_counts,
    }


_BUNCHING_RUN_THRESHOLD = 0.9  # per-sample: suspicious if longest run of one type > 90% of its count
_BUNCHING_SAMPLE_THRESHOLD = 0.7  # aggregate: ordering_valid=False if >70% of samples are suspicious
_MIN_COUNT_FOR_BUNCHING = 2  # need at least 2 elements of a type to check for bunching


@dataclass
class _WdsValidationAcc:
    """Accumulates errors and ordering errors while scanning WDS tars."""

    errors: list[str] = field(default_factory=list)
    ordering_errors: list[str] = field(default_factory=list)


def _is_sample_suspicious(texts: list, images: list) -> bool:
    """Return True if one type's elements are overly bunched (longest run > 90% of its total count)."""
    sequence = [
        "T" if t is not None else "I"
        for t, img in zip(texts, images, strict=False)
        if t is not None or img is not None
    ]
    text_total = sequence.count("T")
    image_total = len(sequence) - text_total
    for type_char, total in (("T", text_total), ("I", image_total)):
        if total < _MIN_COUNT_FOR_BUNCHING:
            continue
        max_run = max(
            (sum(1 for _ in g) for ch, g in itertools.groupby(sequence) if ch == type_char),
            default=0,
        )
        if max_run / total > _BUNCHING_RUN_THRESHOLD:
            return True
    return False


def _check_interleaving(
    sample_id: str,
    texts: list,
    images: list,
    member_names: set[str],
    acc: _WdsValidationAcc,
) -> bool:
    """Check per-sample hard ordering errors. Returns True if sample is suspicious (bunching)."""
    if len(texts) != len(images):
        acc.ordering_errors.append(f"{sample_id}: texts length {len(texts)} != images length {len(images)}")
        return False
    for pos, (t, img) in enumerate(zip(texts, images, strict=False)):
        if t is not None and img is not None:
            acc.ordering_errors.append(f"{sample_id}: position {pos} has both text and image")
        elif img is not None and img not in member_names:
            acc.ordering_errors.append(f"{sample_id}: image '{img}' at position {pos} not found in tar")
    return _is_sample_suspicious(texts, images)


def _check_wds_sample(
    tf: tarfile.TarFile,
    sample_id: str,
    infos: list[tarfile.TarInfo],
    acc: _WdsValidationAcc,
    image_exts: set[str],
) -> tuple[int, bool] | None:
    """Validate one WDS sample. Returns (image_count, is_suspicious), or None on hard error."""
    json_infos = [m for m in infos if m.name.endswith(".json")]
    if not json_infos:
        acc.errors.append(f"{sample_id}: no .json metadata file")
        return None
    try:
        raw = tf.extractfile(json_infos[0])
        if raw is None:
            acc.errors.append(f"{sample_id}: .json is not a regular file")
            return None
        payload = json.loads(raw.read())
    except Exception as e:
        acc.errors.append(f"{sample_id}: .json parse error: {e}")
        return None

    texts = payload.get("texts", [])
    images = payload.get("images", [])
    member_names = {m.name for m in infos}
    suspicious = _check_interleaving(sample_id, texts, images, member_names, acc)
    image_count = sum(1 for m in infos if Path(m.name).suffix.lower() in image_exts)
    return image_count, suspicious


def validate_wds_ordering(tar_path: Path | str) -> dict[str, Any]:
    """Validate ordering within a single WebDataset tar file.

    Checks per-sample hard errors (array length mismatch, position collision, missing image)
    and aggregate bunching. ``ordering_valid`` is False if >70% of samples are suspicious.

    Returns validation fields only: 'valid', 'ordering_valid', 'errors',
    'num_samples', 'num_suspicious_samples', 'num_images'.
    """
    from nemo_curator.stages.interleaved.utils.constants import DEFAULT_IMAGE_EXTENSIONS

    tar_path = Path(tar_path)
    acc = _WdsValidationAcc()
    num_samples = 0
    num_images = 0
    num_suspicious = 0
    image_exts = set(DEFAULT_IMAGE_EXTENSIONS)

    try:
        with tarfile.open(tar_path) as tf:
            members = tf.getmembers()
            # Build sample→member map in a single pass while the tar is open.
            # Members follow {sample_id}.json and {sample_id}.{position}.{ext}
            # naming, so the sample key is the portion before the first dot.
            samples: dict[str, list[tarfile.TarInfo]] = {}
            for m in members:
                samples.setdefault(m.name.split(".")[0], []).append(m)

            for key, infos in samples.items():
                sample_id = f"{tar_path.name}: sample '{key}'"
                result = _check_wds_sample(tf, sample_id, infos, acc, image_exts)
                if result is not None:
                    image_count, suspicious = result
                    num_samples += 1
                    num_images += image_count
                    if suspicious:
                        num_suspicious += 1
    except Exception as e:
        acc.errors.append(f"{tar_path}: failed to open tar: {e}")

    suspicious_ratio = num_suspicious / num_samples if num_samples > 0 else 0.0
    ordering_valid = len(acc.ordering_errors) == 0 and suspicious_ratio <= _BUNCHING_SAMPLE_THRESHOLD
    return {
        "valid": len(acc.errors) == 0 and ordering_valid,
        "ordering_valid": ordering_valid,
        "errors": acc.errors + acc.ordering_errors,
        "num_samples": num_samples,
        "num_suspicious_samples": num_suspicious,
        "num_images": num_images,
    }


def validate_parquet_ordering(parquet_path: str | Path) -> dict[str, Any]:
    """Read a single parquet file and validate interleaved position ordering.

    Returns a dict with 'valid' (bool) and 'errors' (list of issue descriptions).
    """

    df = pd.read_parquet(parquet_path, columns=["sample_id", "position", "modality"])
    errors: list[str] = []
    for sample_id, group in df.groupby("sample_id", sort=False):
        meta = group[group["modality"] == "metadata"]
        content = group[group["modality"] != "metadata"]
        for _, row in meta.iterrows():
            if row["position"] != -1:
                errors.append(f"sample={sample_id}: metadata row has position={row['position']}, expected -1")
        if content.empty:
            continue
        positions = content["position"].tolist()
        expected = list(range(len(positions)))
        if sorted(positions) != expected:
            errors.append(f"sample={sample_id}: content positions {sorted(positions)} != expected {expected}")
    return {"valid": len(errors) == 0, "errors": errors}


def convert_paths_to_strings(obj: object) -> object:
    """
    Convert Path objects to strings, support conversions in container types in a recursive manner.
    """
    if isinstance(obj, dict):
        retval = {convert_paths_to_strings(k): convert_paths_to_strings(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        retval = [convert_paths_to_strings(item) for item in obj]
    elif isinstance(obj, Path):
        retval = str(obj)
    else:
        retval = obj
    return retval
