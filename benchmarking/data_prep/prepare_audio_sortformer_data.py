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

"""Stage the pinned public AMI SDM workload and Sortformer checkpoint."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

import soundfile as sf
from datasets import Audio, load_dataset
from huggingface_hub import hf_hub_download
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audio_sortformer_contract as contract

DEFAULT_CACHE_DIR = "/tmp/curator/audio_sortformer_cache"  # noqa: S108


def _write_audio(audio: object, target_path: Path) -> float:
    if not isinstance(audio, dict) or not isinstance(audio.get("bytes"), bytes):
        msg = "Expected embedded WAV bytes in the pinned AMI Parquet row"
        raise TypeError(msg)
    audio_bytes = audio["bytes"]
    info = sf.info(BytesIO(audio_bytes))
    if (
        info.samplerate != contract.AUDIO_SAMPLE_RATE
        or info.channels != contract.MONO_CHANNELS
        or info.subtype != "PCM_16"
    ):
        msg = f"Expected mono 16 kHz PCM-16 AMI audio, found {info.channels} channels at {info.samplerate} Hz"
        raise RuntimeError(msg)
    target_path.write_bytes(audio_bytes)
    return float(info.frames) / info.samplerate


def verify_dataset(output_path: Path) -> bool:
    try:
        rows, total_duration_s = contract.validate_staged_dataset(output_path)
    except Exception as e:
        logger.error(f"Sortformer dataset validation failed: {e}")
        return False
    logger.info(
        f"Verified {len(rows)} public AMI SDM files ({total_duration_s / 3600:.3f} hours, "
        f"{contract.DATASET_SOURCE_DOWNLOAD_BYTES:,} source bytes)"
    )
    return True


def verify_model(model_path: Path) -> bool:
    try:
        contract.validate_model(model_path)
    except Exception as e:
        logger.error(f"Sortformer model validation failed: {e}")
        return False
    logger.info(f"Verified public Sortformer checkpoint: {model_path}")
    return True


def stage_dataset(output_path: Path, cache_dir: str) -> None:
    """Build and verify in a temporary sibling before replacing prior data."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_path.name}-staging-", dir=output_path.parent) as staging_dir:
        staging_path = Path(staging_dir)
        _stage_dataset(staging_path, cache_dir)
        contract.validate_staged_dataset(staging_path)
        if output_path.exists():
            shutil.rmtree(output_path)
        staging_path.replace(output_path)


def _stage_dataset(output_path: Path, cache_dir: str) -> None:
    audio_dir = output_path / "audio"
    audio_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []

    logger.info(
        f"Staging {contract.DATASET_NUM_ROWS} public AMI SDM meetings from "
        f"{contract.DATASET_HF_REPO_ID}@{contract.DATASET_REVISION}"
    )
    for split in contract.DATASET_SPLITS:
        expected_rows = contract.DATASET_SPLIT_NUM_ROWS[split]
        dataset = load_dataset(
            contract.DATASET_HF_REPO_ID,
            contract.DATASET_CONFIG,
            split=split,
            revision=contract.DATASET_REVISION,
            cache_dir=cache_dir,
            streaming=True,
            token=False,
        ).cast_column("audio", Audio(sampling_rate=contract.AUDIO_SAMPLE_RATE, decode=False))

        split_rows = 0
        for row_index, row in enumerate(dataset):
            if row_index >= expected_rows:
                msg = f"Pinned AMI SDM {split} split contains more than {expected_rows} rows"
                raise RuntimeError(msg)
            audio_item_id = f"ami_sdm_{split}_{row_index:03d}"
            filename = f"{audio_item_id}.wav"
            duration_s = _write_audio(row["audio"], audio_dir / filename)
            rows.append(
                {
                    "audio_filepath": f"audio/{filename}",
                    "audio_item_id": audio_item_id,
                    "session_name": audio_item_id,
                    "duration": duration_s,
                    "timestamps_start": [float(value) for value in row["timestamps_start"]],
                    "timestamps_end": [float(value) for value in row["timestamps_end"]],
                    "speakers": [str(value) for value in row["speakers"]],
                }
            )
            split_rows += 1
        if split_rows != expected_rows:
            msg = f"Pinned AMI SDM {split} split must contain {expected_rows} rows, found {split_rows}"
            raise RuntimeError(msg)

    contract.validate_manifest(rows, f"{contract.DATASET_HF_REPO_ID}/{contract.DATASET_CONFIG}")
    contract.write_manifest(rows, output_path / contract.MANIFEST_FILENAME)
    logger.success(f"Dataset ready: {len(rows)} unique public AMI SDM meetings")


def stage_model(model_path: Path, cache_dir: str) -> None:
    downloaded_path = hf_hub_download(
        repo_id=contract.MODEL_HF_REPO_ID,
        filename=contract.MODEL_FILENAME,
        revision=contract.MODEL_REVISION,
        cache_dir=cache_dir,
        token=False,
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(downloaded_path, model_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--model-output-path", type=Path, required=True)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    output_path = args.output_path.resolve()
    model_path = args.model_output_path.resolve()
    if args.verify_only:
        return 0 if verify_dataset(output_path) and verify_model(model_path) else 1

    dataset_valid = verify_dataset(output_path)
    if not dataset_valid:
        stage_dataset(output_path, args.cache_dir)
        dataset_valid = True  # stage_dataset already verified the transactional output

    model_valid = verify_model(model_path)
    if not model_valid:
        stage_model(model_path, args.cache_dir)
        model_valid = verify_model(model_path)
    return 0 if dataset_valid and model_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
