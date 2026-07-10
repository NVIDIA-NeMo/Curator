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

"""One-time data preparation for the audio-tagging benchmark.

Downloads three AMI SDM test meetings and the local PyAnnote diarization
snapshot once so nightly runs can use ``--raw-data-dir`` and
``--no-auto-download`` with no Hugging Face token or network dependency.

The default sources are token-free Hugging Face repos:

    diarizers-community/ami, config sdm, split test
    pyannote-community/speaker-diarization-community-1

Example usage::

    python prepare_audio_tagging_data.py \\
        --output-path /path/to/datasets/audio_tagging_ami_sdm \\
        --model-output-path /path/to/model_weights/audio_tagging/pyannote-speaker-diarization-community-1

If the model snapshot already exists locally, copy it without any HF model
download by passing ``--model-source-path``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf
from datasets import load_dataset
from huggingface_hub import snapshot_download
from loguru import logger

DEFAULT_AUDIO_TAGGING_CACHE_DIR = "/tmp/curator/audio_tagging_cache"  # noqa: S108
DEFAULT_CONTAINER_DATA_PATH = "/datasets/audio_tagging_ami_sdm"
DEFAULT_AMI_HF_REPO_ID = "diarizers-community/ami"
DEFAULT_AMI_CONFIG = "sdm"
DEFAULT_AMI_SPLIT = "test"
DEFAULT_MODEL_HF_REPO_ID = "pyannote-community/speaker-diarization-community-1"
MODEL_DIR_NAME = "pyannote-speaker-diarization-community-1"
MODEL_MARKERS = ("config.yaml", "segmentation", "embedding", "plda")
AUDIO_WAVEFORM_DIMENSIONS = 2
AUDIO_FILENAMES = (
    "audio/EN2002b.Array1-01.wav",
    "audio/ES2004c.Array1-01.wav",
    "audio/TS3003a.Array1-01.wav",
)
EXPECTED_AUDIO_BASENAMES = {Path(filename).name for filename in AUDIO_FILENAMES}
AMI_TEST_MEETINGS = (
    (1, "EN2002b.Array1-01"),
    (6, "ES2004c.Array1-01"),
    (12, "TS3003a.Array1-01"),
)
MODEL_ALLOW_PATTERNS = (
    "config.yaml",
    "embedding/**",
    "plda/**",
    "segmentation/**",
)


@dataclass(frozen=True)
class DatasetStagingConfig:
    output_path: Path
    ami_hf_repo_id: str
    ami_config: str
    ami_split: str
    cache_dir: str
    container_data_path: str


def _expected_audio_paths(output_path: Path) -> list[Path]:
    return [output_path / "audio" / Path(filename).name for filename in AUDIO_FILENAMES]


def _load_manifest_rows(manifest_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with manifest_path.open(encoding="utf-8") as manifest_file:
        for line_number, line in enumerate(manifest_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                msg = f"Manifest has invalid JSON on line {line_number}: {e}"
                raise RuntimeError(msg) from e
            if not isinstance(row, dict):
                msg = f"Manifest line {line_number} is not a JSON object"
                raise TypeError(msg)
            rows.append(row)
    if not rows:
        msg = f"Manifest contains no data rows: {manifest_path}"
        raise RuntimeError(msg)
    return rows


def _validate_manifest_contract(rows: list[dict[str, Any]], label: str) -> None:
    if len(rows) != len(AUDIO_FILENAMES):
        msg = f"{label} must contain exactly {len(AUDIO_FILENAMES)} rows, found {len(rows)}"
        raise RuntimeError(msg)

    seen_audio_basenames: set[str] = set()
    seen_audio_item_ids: set[str] = set()
    for line_number, row in enumerate(rows, start=1):
        audio_filepath = row.get("audio_filepath")
        if not isinstance(audio_filepath, str) or not audio_filepath:
            msg = f"{label} line {line_number} must contain audio_filepath"
            raise RuntimeError(msg)

        audio_basename = Path(audio_filepath).name
        if audio_basename not in EXPECTED_AUDIO_BASENAMES:
            msg = (
                f"{label} line {line_number} references unexpected audio file {audio_basename!r}; "
                f"expected one of {sorted(EXPECTED_AUDIO_BASENAMES)}"
            )
            raise RuntimeError(msg)
        if audio_basename in seen_audio_basenames:
            msg = f"{label} contains duplicate audio file {audio_basename!r}"
            raise RuntimeError(msg)
        seen_audio_basenames.add(audio_basename)

        audio_item_id = row.get("audio_item_id")
        if not isinstance(audio_item_id, str) or not audio_item_id:
            msg = f"{label} line {line_number} must contain a nonempty audio_item_id"
            raise RuntimeError(msg)
        if audio_item_id in seen_audio_item_ids:
            msg = f"{label} contains duplicate audio_item_id {audio_item_id!r}"
            raise RuntimeError(msg)
        seen_audio_item_ids.add(audio_item_id)

    if seen_audio_basenames != EXPECTED_AUDIO_BASENAMES:
        missing = sorted(EXPECTED_AUDIO_BASENAMES - seen_audio_basenames)
        msg = f"{label} is missing expected audio files: {missing}"
        raise RuntimeError(msg)


def _rewrite_manifest(source_manifest: Path, target_manifest: Path, container_data_path: str) -> int:
    rows = _load_manifest_rows(source_manifest)
    _validate_manifest_contract(rows, str(source_manifest))
    audio_container_dir = Path(container_data_path) / "audio"
    target_manifest.parent.mkdir(parents=True, exist_ok=True)
    with target_manifest.open("w", encoding="utf-8") as target_file:
        for line_number, row in enumerate(rows, start=1):
            audio_filepath = row.get("audio_filepath")
            if not isinstance(audio_filepath, str) or not audio_filepath:
                msg = f"Manifest line {line_number} must contain audio_filepath"
                raise RuntimeError(msg)
            row["audio_filepath"] = str(audio_container_dir / Path(audio_filepath).name)
            target_file.write(json.dumps(row) + "\n")
    return len(rows)


def _write_manifest(rows: list[dict[str, str]], target_manifest: Path) -> None:
    target_manifest.parent.mkdir(parents=True, exist_ok=True)
    with target_manifest.open("w", encoding="utf-8") as target_file:
        for row in rows:
            target_file.write(json.dumps(row) + "\n")


def _copy_tree_contents(source_dir: Path, target_dir: Path) -> None:
    if not source_dir.is_dir():
        msg = f"Model source directory not found: {source_dir}"
        raise FileNotFoundError(msg)
    target_dir.mkdir(parents=True, exist_ok=True)
    for item in source_dir.iterdir():
        target = target_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def verify_dataset(output_path: Path) -> bool:
    manifest_path = output_path / "manifest.jsonl"
    audio_dir = output_path / "audio"
    if not manifest_path.is_file():
        logger.error(f"Manifest not found: {manifest_path}")
        return False
    if not audio_dir.is_dir():
        logger.error(f"Audio directory not found: {audio_dir}")
        return False

    missing_audio = [path for path in _expected_audio_paths(output_path) if not path.is_file()]
    if missing_audio:
        logger.error(f"Missing expected audio files: {', '.join(str(path) for path in missing_audio)}")
        return False

    try:
        rows = _load_manifest_rows(manifest_path)
        _validate_manifest_contract(rows, str(manifest_path))
        num_rows = len(rows)
    except Exception as e:
        logger.error(f"Manifest validation failed: {e}")
        return False

    logger.info("=" * 60)
    logger.info("Audio Tagging Dataset Verification")
    logger.info("=" * 60)
    logger.info(f"  Dataset dir: {output_path}")
    logger.info(f"  Manifest:    {manifest_path} ({num_rows} rows)")
    logger.info(f"  Audio files: {len(AUDIO_FILENAMES)}")
    logger.info("=" * 60)
    return True


def verify_model(model_output_path: Path) -> bool:
    missing = [model_output_path / marker for marker in MODEL_MARKERS if not (model_output_path / marker).exists()]
    if missing:
        logger.error(f"Missing PyAnnote snapshot files/directories: {', '.join(str(path) for path in missing)}")
        return False
    logger.info("=" * 60)
    logger.info("Audio Tagging Model Verification")
    logger.info("=" * 60)
    logger.info(f"  Model dir: {model_output_path}")
    logger.info(f"  Markers:   {', '.join(MODEL_MARKERS)}")
    logger.info("=" * 60)
    return True


def _write_audio_row(audio: object, target_path: Path) -> None:
    if not hasattr(audio, "get_all_samples"):
        msg = f"Expected a datasets AudioDecoder, got {type(audio).__name__}"
        raise TypeError(msg)

    samples = audio.get_all_samples()
    waveform = samples.data
    if waveform.ndim == AUDIO_WAVEFORM_DIMENSIONS and waveform.shape[0] <= waveform.shape[1]:
        waveform = waveform.transpose(0, 1)
    audio_array = waveform.cpu().numpy()
    sf.write(target_path, audio_array, samples.sample_rate, subtype="PCM_16")


def stage_dataset(config: DatasetStagingConfig) -> None:
    config.output_path.mkdir(parents=True, exist_ok=True)
    audio_dir = config.output_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Audio Tagging AMI Dataset Download")
    logger.info(f"Repo:       {config.ami_hf_repo_id}")
    logger.info(f"Config:     {config.ami_config}")
    logger.info(f"Split:      {config.ami_split}")
    logger.info(f"Staging to: {config.output_path}")
    logger.info("=" * 60)

    dataset = load_dataset(
        config.ami_hf_repo_id,
        config.ami_config,
        split=config.ami_split,
        cache_dir=config.cache_dir,
        streaming=True,
    )
    meeting_by_index = dict(AMI_TEST_MEETINGS)
    manifest_rows: list[dict[str, str]] = []
    audio_container_dir = Path(config.container_data_path) / "audio"

    for row_index, row in enumerate(dataset):
        audio_item_id = meeting_by_index.get(row_index)
        if audio_item_id is None:
            continue

        target_audio_path = audio_dir / f"{audio_item_id}.wav"
        logger.info(
            f"Staging {audio_item_id} from "
            f"{config.ami_hf_repo_id}/{config.ami_config}/{config.ami_split} row {row_index}"
        )
        _write_audio_row(row["audio"], target_audio_path)
        manifest_rows.append(
            {
                "audio_filepath": str(audio_container_dir / target_audio_path.name),
                "audio_item_id": audio_item_id,
            }
        )

        if len(manifest_rows) == len(AMI_TEST_MEETINGS):
            break

    if len(manifest_rows) != len(AMI_TEST_MEETINGS):
        msg = (
            f"Expected to stage {len(AMI_TEST_MEETINGS)} AMI meetings from "
            f"{config.ami_hf_repo_id}/{config.ami_config}/{config.ami_split}, "
            f"but staged {len(manifest_rows)}"
        )
        raise RuntimeError(msg)

    _write_manifest(manifest_rows, config.output_path / "manifest.jsonl")
    logger.success(f"Dataset ready: {len(manifest_rows)} manifest rows and {len(AUDIO_FILENAMES)} WAV files")


def stage_model(
    model_output_path: Path,
    model_hf_repo_id: str | None,
    cache_dir: str,
    model_source_path: Path | None,
) -> None:
    logger.info("=" * 60)
    logger.info("Audio Tagging PyAnnote Snapshot Staging")
    logger.info(f"Staging to: {model_output_path}")
    logger.info("=" * 60)

    if model_source_path is not None:
        logger.info(f"Copying local model snapshot from {model_source_path}")
        _copy_tree_contents(model_source_path, model_output_path)
        return

    if not model_hf_repo_id:
        msg = "Model staging requires --model-hf-repo-id/CURATOR_AUDIO_TAGGING_MODEL_HF_REPO_ID or --model-source-path"
        raise ValueError(msg)

    snapshot_path = Path(
        snapshot_download(
            repo_id=model_hf_repo_id,
            repo_type="model",
            allow_patterns=list(MODEL_ALLOW_PATTERNS),
            cache_dir=cache_dir,
        )
    )
    logger.info(f"Copying model snapshot from {snapshot_path}")
    _copy_tree_contents(snapshot_path, model_output_path)


def _resolve_ami_hf_repo_id(value: str | None) -> str | None:
    return value or os.environ.get("CURATOR_AUDIO_TAGGING_HF_REPO_ID") or DEFAULT_AMI_HF_REPO_ID


def _resolve_model_hf_repo_id(value: str | None) -> str | None:
    return value or os.environ.get("CURATOR_AUDIO_TAGGING_MODEL_HF_REPO_ID") or DEFAULT_MODEL_HF_REPO_ID


def _has_dataset_artifacts(output_path: Path) -> bool:
    return (output_path / "manifest.jsonl").exists() or (output_path / "audio").exists()


def _has_model_artifacts(model_output_path: Path) -> bool:
    return model_output_path.exists() and any(model_output_path.iterdir())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage AMI audio-tagging benchmark data and local PyAnnote model snapshot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Directory to stage the AMI manifest.jsonl and audio/ files into.",
    )
    parser.add_argument(
        "--model-output-path",
        type=Path,
        required=True,
        help="Directory to stage pyannote-speaker-diarization-community-1 into.",
    )
    parser.add_argument(
        "--hf-repo-id",
        default=None,
        help="Token-free HF dataset repo containing AMI SDM audio. "
        "Defaults to $CURATOR_AUDIO_TAGGING_HF_REPO_ID or diarizers-community/ami.",
    )
    parser.add_argument(
        "--ami-config",
        default=DEFAULT_AMI_CONFIG,
        help="HF dataset config for AMI audio staging. Default: sdm.",
    )
    parser.add_argument(
        "--ami-split",
        default=DEFAULT_AMI_SPLIT,
        help="HF dataset split containing the benchmark meetings. Default: test.",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("CURATOR_AUDIO_TAGGING_CACHE_DIR", DEFAULT_AUDIO_TAGGING_CACHE_DIR),
        help="HF cache directory used during one-time staging.",
    )
    parser.add_argument(
        "--container-data-path",
        default=DEFAULT_CONTAINER_DATA_PATH,
        help="Container-visible dataset path written into manifest audio_filepath values.",
    )
    parser.add_argument(
        "--model-repo-subdir",
        default=MODEL_DIR_NAME,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--model-hf-repo-id",
        default=None,
        help="Token-free HF model repo containing the PyAnnote snapshot. "
        "Defaults to $CURATOR_AUDIO_TAGGING_MODEL_HF_REPO_ID or pyannote-community/speaker-diarization-community-1.",
    )
    parser.add_argument(
        "--model-source-path",
        type=Path,
        default=None,
        help="Optional local PyAnnote snapshot to copy instead of downloading it from the HF dataset repo.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing staged dataset and model paths without downloading or copying.",
    )

    args = parser.parse_args()
    output_path = args.output_path.resolve()
    model_output_path = args.model_output_path.resolve()
    model_source_path = args.model_source_path.resolve() if args.model_source_path else None
    ami_hf_repo_id = _resolve_ami_hf_repo_id(args.hf_repo_id)
    model_hf_repo_id = _resolve_model_hf_repo_id(args.model_hf_repo_id)

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    if args.verify_only:
        logger.info(f"Verifying staged audio-tagging dataset at: {output_path}")
        logger.info(f"Verifying staged audio-tagging model at: {model_output_path}")
        return 0 if verify_dataset(output_path) and verify_model(model_output_path) else 1

    dataset_ready = _has_dataset_artifacts(output_path) and verify_dataset(output_path)
    if not dataset_ready:
        if not ami_hf_repo_id:
            logger.error("Dataset staging requires --hf-repo-id or CURATOR_AUDIO_TAGGING_HF_REPO_ID")
            return 1
        stage_dataset(
            DatasetStagingConfig(
                output_path=output_path,
                ami_hf_repo_id=ami_hf_repo_id,
                ami_config=args.ami_config,
                ami_split=args.ami_split,
                cache_dir=args.cache_dir,
                container_data_path=args.container_data_path,
            )
        )
        dataset_ready = verify_dataset(output_path)

    model_ready = _has_model_artifacts(model_output_path) and verify_model(model_output_path)
    if not model_ready:
        try:
            stage_model(
                model_output_path=model_output_path,
                model_hf_repo_id=model_hf_repo_id,
                cache_dir=args.cache_dir,
                model_source_path=model_source_path,
            )
        except Exception as e:
            logger.error(f"Model staging failed: {e}")
            return 1
        model_ready = verify_model(model_output_path)

    return 0 if dataset_ready and model_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
