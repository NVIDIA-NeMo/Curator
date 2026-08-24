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

"""Extract embedded AMI WAVs and stage the Sortformer model for nightly runs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset
from huggingface_hub import hf_hub_download
from loguru import logger

DEFAULT_CACHE_DIR = "/tmp/curator/audio_sortformer_cache"  # noqa: S108
AMI_HF_REPO_ID = "diarizers-community/ami"
AMI_CONFIG = "sdm"
AMI_HF_REVISION = "8cdaae2eaf968f3b000b6eb1204ab9b8db006ed0"  # pragma: allowlist secret
AMI_SPLIT_NUM_ROWS = {"validation": 18, "test": 16, "train": 10}
EXPECTED_AUDIO_FILENAMES = frozenset(
    f"ami_sdm_{split}_{index:03d}.wav" for split, num_rows in AMI_SPLIT_NUM_ROWS.items() for index in range(num_rows)
)
MODEL_HF_REPO_ID = "nvidia/diar_streaming_sortformer_4spk-v2.1"
MODEL_FILENAME = "diar_streaming_sortformer_4spk-v2.1.nemo"


def verify_dataset(output_path: Path) -> bool:
    manifest_path = output_path / "manifest.jsonl"
    audio_dir = output_path / "audio"
    try:
        rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
        filenames = [Path(row["audio_filepath"]).name for row in rows]
        is_ready = (
            len(rows) == len(EXPECTED_AUDIO_FILENAMES)
            and set(filenames) == EXPECTED_AUDIO_FILENAMES
            and all(
                (audio_dir / filename).is_file() and (audio_dir / filename).stat().st_size for filename in filenames
            )
            and all(row["duration"] > 0 for row in rows)
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        is_ready = False

    if not is_ready:
        logger.error(f"Incomplete Sortformer dataset under {output_path}")
        return False
    total_duration_hours = sum(row["duration"] for row in rows) / 3600
    logger.info(f"Verified {len(rows)} AMI SDM files ({total_duration_hours:.3f} audio hours)")
    return True


def verify_model(model_path: Path) -> bool:
    is_ready = model_path.is_file() and model_path.stat().st_size > 0
    if not is_ready:
        logger.error(f"Sortformer model is missing or empty: {model_path}")
    return is_ready


def stage_dataset(output_path: Path, cache_dir: str) -> None:
    audio_dir = output_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    num_rows = 0

    with (output_path / "manifest.jsonl").open("w", encoding="utf-8") as manifest_file:
        for split, expected_rows in AMI_SPLIT_NUM_ROWS.items():
            dataset = load_dataset(
                AMI_HF_REPO_ID,
                AMI_CONFIG,
                split=split,
                revision=AMI_HF_REVISION,
                cache_dir=cache_dir,
                streaming=True,
            ).cast_column("audio", Audio(decode=False))
            for row_index, row in enumerate(dataset):
                if row_index >= expected_rows:
                    break
                audio_item_id = f"ami_sdm_{split}_{row_index:03d}"
                filename = f"{audio_item_id}.wav"
                audio_path = audio_dir / filename
                audio_path.write_bytes(row["audio"]["bytes"])
                audio_info = sf.info(audio_path)
                manifest_file.write(
                    json.dumps(
                        {
                            "audio_filepath": f"audio/{filename}",
                            "audio_item_id": audio_item_id,
                            "duration": audio_info.frames / audio_info.samplerate,
                        }
                    )
                    + "\n"
                )
                num_rows += 1
    logger.success(f"Dataset ready: {num_rows} AMI SDM meetings")


def stage_model(model_path: Path, cache_dir: str) -> None:
    downloaded_path = hf_hub_download(
        repo_id=MODEL_HF_REPO_ID,
        filename=MODEL_FILENAME,
        cache_dir=cache_dir,
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

    output_path = args.output_path.resolve()
    model_path = args.model_output_path.resolve()
    if args.verify_only:
        return 0 if verify_dataset(output_path) and verify_model(model_path) else 1

    if not verify_dataset(output_path):
        stage_dataset(output_path, args.cache_dir)
    if not verify_model(model_path):
        stage_model(model_path, args.cache_dir)
    return 0 if verify_dataset(output_path) and verify_model(model_path) else 1


if __name__ == "__main__":
    raise SystemExit(main())
