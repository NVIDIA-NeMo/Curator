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

"""Stage a fixed-duration LibriSpeech prefix for nightly benchmarks."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset
from loguru import logger

DEFAULT_HF_REPO_ID = "openslr/librispeech_asr"
DEFAULT_HF_REVISION = "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"  # pragma: allowlist secret
DEFAULT_HF_CONFIG = "all"
DEFAULT_HF_SPLIT = "train.clean.100+train.clean.360+train.other.500"
DEFAULT_CACHE_DIR = "/tmp/curator/librispeech_cache"  # noqa: S108
DEFAULT_TARGET_AUDIO_HOURS = 750.0


def _copy_audio(audio: dict, target_path: Path) -> float:
    if isinstance(audio.get("bytes"), bytes):
        target_path.write_bytes(audio["bytes"])
    else:
        shutil.copyfile(audio["path"], target_path)
    return float(sf.info(target_path).duration)


def stage_dataset(  # noqa: PLR0913
    output_path: Path,
    cache_dir: str = DEFAULT_CACHE_DIR,
    hf_repo_id: str = DEFAULT_HF_REPO_ID,
    hf_revision: str = DEFAULT_HF_REVISION,
    hf_config: str = DEFAULT_HF_CONFIG,
    hf_split: str = DEFAULT_HF_SPLIT,
    target_audio_hours: float = DEFAULT_TARGET_AUDIO_HOURS,
) -> None:
    output_path = output_path.resolve()
    manifest_path = output_path / "manifest.jsonl"
    if manifest_path.is_file():
        logger.info(f"Reusing staged LibriSpeech manifest at {manifest_path}")
        return

    audio_dir = output_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_suffix(".jsonl.tmp")
    target_duration_s = target_audio_hours * 3600
    selected_duration_s = 0.0
    clips = 0

    datasets = load_dataset(
        hf_repo_id,
        hf_config,
        revision=hf_revision,
        cache_dir=cache_dir,
        streaming=True,
    ).cast_column("audio", Audio(decode=False))
    try:
        with temporary_manifest.open("w", encoding="utf-8") as manifest_file:
            for split in hf_split.split("+"):
                dataset_iterator = iter(datasets[split])
                try:
                    for row in dataset_iterator:
                        audio_item_id = str(row["id"]).replace("/", "-")
                        audio_path = audio_dir / f"{audio_item_id}.flac"
                        duration_s = _copy_audio(row["audio"], audio_path)
                        manifest_file.write(
                            json.dumps({"audio_filepath": str(audio_path), "text": row["text"]}, separators=(",", ":"))
                            + "\n"
                        )
                        clips += 1
                        selected_duration_s += duration_s
                        if selected_duration_s >= target_duration_s:
                            break
                finally:
                    dataset_iterator.close()
                    # Work around apache/arrow#45214 on PyArrow <=24.
                    gc.collect()
                    time.sleep(5)
                if selected_duration_s >= target_duration_s:
                    break
        if selected_duration_s < target_duration_s:
            msg = f"{hf_repo_id}/{hf_config}/{hf_split} contains only {selected_duration_s / 3600:.3f} hours"
            raise RuntimeError(msg)  # noqa: TRY301
        temporary_manifest.replace(manifest_path)
    except Exception:
        temporary_manifest.unlink(missing_ok=True)
        raise

    logger.success(f"Staged {clips} clips / {selected_duration_s / 3600:.4f} hours at {manifest_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--hf-repo-id", default=DEFAULT_HF_REPO_ID)
    parser.add_argument("--hf-revision", default=DEFAULT_HF_REVISION)
    parser.add_argument("--hf-config", default=DEFAULT_HF_CONFIG)
    parser.add_argument("--hf-split", default=DEFAULT_HF_SPLIT)
    parser.add_argument("--target-audio-hours", type=float, default=DEFAULT_TARGET_AUDIO_HOURS)
    args = parser.parse_args()
    stage_dataset(**vars(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
