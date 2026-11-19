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

"""Stage open English LibriSpeech recordings (CC-BY-4.0) as a Granary v2 English pipeline input of a chosen duration."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset
from loguru import logger

DEFAULT_HF_REPO_ID = "openslr/librispeech_asr"  # CC-BY-4.0, not gated
DEFAULT_HF_REVISION = "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"  # pragma: allowlist secret
DEFAULT_HF_CONFIG = "all"
DEFAULT_HF_SPLIT = "train.other.500"  # noisier speech than train.clean, closer to web audio
DEFAULT_CACHE_DIR = "/tmp/curator/librispeech_cache"  # noqa: S108


def stage_dataset(  # noqa: PLR0913
    output_path: Path,
    target_audio_hours: float,
    cache_dir: str = DEFAULT_CACHE_DIR,
    hf_repo_id: str = DEFAULT_HF_REPO_ID,
    hf_revision: str = DEFAULT_HF_REVISION,
    hf_config: str = DEFAULT_HF_CONFIG,
    hf_split: str = DEFAULT_HF_SPLIT,
) -> None:
    output_path = output_path.resolve()
    manifest_path = output_path / "manifest.jsonl"
    if manifest_path.is_file():
        logger.info(f"Reusing staged manifest at {manifest_path}")
        return
    audio_dir = output_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        hf_repo_id, hf_config, split=hf_split, revision=hf_revision, cache_dir=cache_dir, streaming=True
    ).cast_column("audio", Audio(decode=False))
    temporary_manifest = manifest_path.with_suffix(".jsonl.tmp")
    clips = 0
    seconds = 0.0
    try:
        with temporary_manifest.open("w", encoding="utf-8") as manifest_file:
            for row in dataset:
                item_id = str(row["id"]).replace("/", "-")
                audio_path = audio_dir / f"{item_id}.flac"
                audio = row["audio"]
                if isinstance(audio.get("bytes"), bytes):
                    audio_path.write_bytes(audio["bytes"])
                else:
                    shutil.copyfile(audio["path"], audio_path)
                duration = float(sf.info(audio_path).duration)
                record = {
                    "audio_item_id": item_id,
                    "audio_filepath": str(audio_path),
                    "duration": duration,
                    "text": row["text"],
                }
                manifest_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                clips += 1
                seconds += duration
                if seconds >= target_audio_hours * 3600:
                    break
        if seconds < target_audio_hours * 3600:
            msg = f"{hf_repo_id}/{hf_config}/{hf_split} holds only {seconds / 3600:.2f} hours"
            raise RuntimeError(msg)  # noqa: TRY301
        temporary_manifest.replace(manifest_path)
    except Exception:
        temporary_manifest.unlink(missing_ok=True)
        raise
    logger.success(f"Staged {clips} recordings / {seconds / 3600:.3f} hours at {manifest_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--target-audio-hours", type=float, required=True)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--hf-repo-id", default=DEFAULT_HF_REPO_ID)
    parser.add_argument("--hf-revision", default=DEFAULT_HF_REVISION)
    parser.add_argument("--hf-config", default=DEFAULT_HF_CONFIG)
    parser.add_argument("--hf-split", default=DEFAULT_HF_SPLIT)
    stage_dataset(**vars(parser.parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
