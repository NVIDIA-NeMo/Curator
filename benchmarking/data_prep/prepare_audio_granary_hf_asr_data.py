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

"""Stage the first hours of an open Hugging Face ASR dataset as a Granary v2 pipeline input.

Used for English (``openslr/librispeech_asr``) and Italian (``facebook/multilingual_librispeech``). The manifest rows are
``audio_item_id``, ``audio_filepath`` and ``duration``. The recording transcript is not written: every VAD clip would
inherit it as if it were a clip label.
"""

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

DEFAULT_CACHE_DIR = "/tmp/curator/hf_asr_cache"  # noqa: S108


def stage_dataset(  # noqa: PLR0913
    output_path: Path,
    target_audio_hours: float,
    hf_repo_id: str,
    hf_config: str,
    hf_split: str,
    hf_revision: str | None = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
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
    iterator = iter(dataset)
    try:
        with temporary_manifest.open("w", encoding="utf-8") as manifest_file:
            for row in iterator:
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
    finally:
        iterator.close()
        gc.collect()  # apache/arrow#45214: a half-read streaming dataset can hang the interpreter at exit
        time.sleep(5)
    logger.success(f"Staged {clips} recordings / {seconds / 3600:.3f} hours at {manifest_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--target-audio-hours", type=float, required=True)
    parser.add_argument("--hf-repo-id", required=True)
    parser.add_argument("--hf-config", required=True)
    parser.add_argument("--hf-split", required=True)
    parser.add_argument(
        "--hf-revision", help="dataset revision to pin; the commands in benchmarking/README.md set one"
    )
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    stage_dataset(**vars(parser.parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
