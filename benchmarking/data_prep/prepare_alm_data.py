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

"""Stage AMI timestamps for the CPU-only ALM benchmark without audio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from loguru import logger

DEFAULT_HF_REPO_ID = "diarizers-community/ami"
DEFAULT_HF_CONFIG = "sdm"
DEFAULT_HF_REVISION = "8cdaae2eaf968f3b000b6eb1204ab9b8db006ed0"  # pragma: allowlist secret
DEFAULT_CACHE_DIR = "/tmp/curator/alm_ami_cache"  # noqa: S108
AMI_SPLITS = ("train", "validation", "test")
AMI_COLUMNS = ("timestamps_start", "timestamps_end", "speakers")
AMI_SAMPLE_RATE = 16000


def stage_manifest(
    output_path: Path,
    hf_repo_id: str = DEFAULT_HF_REPO_ID,
    hf_config: str = DEFAULT_HF_CONFIG,
    hf_revision: str = DEFAULT_HF_REVISION,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> None:
    output_path = output_path.resolve()
    manifest_path = output_path / "manifest.jsonl"
    if manifest_path.is_file():
        logger.info(f"Reusing staged ALM manifest at {manifest_path}")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_suffix(".jsonl.tmp")
    meetings = segments_count = 0
    duration_s = 0.0

    try:
        with temporary_manifest.open("w", encoding="utf-8") as manifest_file:
            for split in AMI_SPLITS:
                dataset = load_dataset(
                    hf_repo_id,
                    hf_config,
                    split=split,
                    revision=hf_revision,
                    cache_dir=cache_dir,
                    streaming=True,
                    columns=list(AMI_COLUMNS),
                )
                for row_index, row in enumerate(dataset):
                    segments = [
                        {"start": float(start), "end": float(end), "speaker": speaker}
                        for start, end, speaker in zip(
                            row["timestamps_start"], row["timestamps_end"], row["speakers"], strict=True
                        )
                    ]
                    segments.sort(key=lambda segment: (segment["start"], segment["end"], segment["speaker"]))
                    duration = max(segment["end"] for segment in segments)
                    manifest_file.write(
                        json.dumps(
                            {
                                "audio_filepath": (
                                    f"hf://datasets/{hf_repo_id}@{hf_revision}/{hf_config}/{split}/{row_index:06d}"
                                ),
                                "audio_sample_rate": AMI_SAMPLE_RATE,
                                "duration": duration,
                                "segments": segments,
                            },
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    meetings += 1
                    segments_count += len(segments)
                    duration_s += duration
        temporary_manifest.replace(manifest_path)
    except Exception:
        temporary_manifest.unlink(missing_ok=True)
        raise

    logger.success(
        f"Staged {meetings} AMI meetings / {segments_count} segments / {duration_s / 3600:.2f} hours at {manifest_path}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--hf-repo-id", default=DEFAULT_HF_REPO_ID)
    parser.add_argument("--hf-config", default=DEFAULT_HF_CONFIG)
    parser.add_argument("--hf-revision", default=DEFAULT_HF_REVISION)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    args = parser.parse_args()
    stage_manifest(**vars(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
