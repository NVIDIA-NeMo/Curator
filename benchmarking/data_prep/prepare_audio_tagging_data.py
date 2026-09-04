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

"""Stage AMI audio and the PyAnnote model for the audio-tagging benchmark."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset
from huggingface_hub import snapshot_download
from loguru import logger

DEFAULT_AUDIO_TAGGING_CACHE_DIR = "/tmp/curator/audio_tagging_cache"  # noqa: S108
DEFAULT_AMI_HF_REPO_ID = "diarizers-community/ami"
DEFAULT_AMI_CONFIG = "sdm"
DEFAULT_AMI_HF_REVISION = "8cdaae2eaf968f3b000b6eb1204ab9b8db006ed0"  # pragma: allowlist secret
DEFAULT_AMI_SPLITS = ("test", "validation", "train")
DEFAULT_MIN_AUDIO_HOURS = 30.0
DEFAULT_MAX_MEETING_DURATION_MINUTES = 60.0
DEFAULT_MODEL_HF_REPO_ID = "pyannote-community/speaker-diarization-community-1"
DEFAULT_MODEL_HF_REVISION = "8a527374977391da736e0daaef26855d949d9685"  # pragma: allowlist secret
MODEL_READY_MARKER = ".curator-staging-complete"
MODEL_ALLOW_PATTERNS = ("config.yaml", "embedding/**", "plda/**", "segmentation/**")


def _copy_audio(audio: dict, target_path: Path) -> float:
    if isinstance(audio.get("bytes"), bytes):
        target_path.write_bytes(audio["bytes"])
    else:
        shutil.copyfile(audio["path"], target_path)
    return float(sf.info(target_path).duration)


def _audio_item_id(audio: dict, split: str, row_index: int) -> str:
    path = audio.get("path")
    if isinstance(path, str) and path:
        return Path(path.split("::")[-1].split("?", maxsplit=1)[0]).stem
    return f"{split}-{row_index:06d}"


def stage_dataset(  # noqa: PLR0913
    output_path: Path,
    ami_hf_repo_id: str,
    ami_config: str,
    ami_hf_revision: str,
    ami_splits: tuple[str, ...],
    cache_dir: str,
    min_audio_hours: float,
    max_meeting_duration_minutes: float,
) -> None:
    output_path = output_path.resolve()
    manifest_path = output_path / "manifest.jsonl"
    if manifest_path.is_file():
        logger.info(f"Reusing staged audio-tagging manifest at {manifest_path}")
        return

    audio_dir = output_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_suffix(".jsonl.tmp")
    target_duration_s = min_audio_hours * 3600
    max_meeting_duration_s = max_meeting_duration_minutes * 60
    selected_duration_s = 0.0
    meetings = 0
    seen_ids: set[str] = set()

    try:
        with temporary_manifest.open("w", encoding="utf-8") as manifest_file:
            for split in ami_splits:
                dataset = load_dataset(
                    ami_hf_repo_id,
                    ami_config,
                    split=split,
                    revision=ami_hf_revision,
                    cache_dir=cache_dir,
                    streaming=True,
                    columns=["audio"],
                ).cast_column("audio", Audio(decode=False))
                for row_index, row in enumerate(dataset):
                    audio = row["audio"]
                    audio_item_id = _audio_item_id(audio, split, row_index)
                    if audio_item_id in seen_ids:
                        continue
                    audio_path = audio_dir / f"{audio_item_id}.wav"
                    duration_s = _copy_audio(audio, audio_path)
                    if duration_s > max_meeting_duration_s:
                        audio_path.unlink()
                        continue
                    manifest_file.write(
                        json.dumps(
                            {
                                "audio_filepath": str(audio_path),
                                "audio_item_id": audio_item_id,
                            },
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    seen_ids.add(audio_item_id)
                    meetings += 1
                    selected_duration_s += duration_s
                    if selected_duration_s >= target_duration_s:
                        break
                if selected_duration_s >= target_duration_s:
                    break
        if selected_duration_s < target_duration_s:
            msg = f"AMI supplied only {selected_duration_s / 3600:.3f} hours below the meeting-duration limit"
            raise RuntimeError(msg)  # noqa: TRY301
        temporary_manifest.replace(manifest_path)
    except Exception:
        temporary_manifest.unlink(missing_ok=True)
        raise

    logger.success(f"Staged {meetings} AMI meetings / {selected_duration_s / 3600:.3f} hours at {manifest_path}")


def stage_model(
    output_path: Path,
    hf_repo_id: str,
    hf_revision: str,
) -> None:
    marker = output_path / MODEL_READY_MARKER
    if marker.is_file():
        logger.info(f"Reusing staged PyAnnote model at {output_path}")
        return
    snapshot_download(
        repo_id=hf_repo_id,
        repo_type="model",
        revision=hf_revision,
        allow_patterns=list(MODEL_ALLOW_PATTERNS),
        local_dir=output_path,
    )
    marker.touch()
    logger.success(f"Staged PyAnnote model at {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--model-output-path", type=Path, required=True)
    parser.add_argument(
        "--hf-repo-id",
        default=os.environ.get("CURATOR_AUDIO_TAGGING_HF_REPO_ID", DEFAULT_AMI_HF_REPO_ID),
    )
    parser.add_argument("--ami-config", default=DEFAULT_AMI_CONFIG)
    parser.add_argument("--hf-revision", default=DEFAULT_AMI_HF_REVISION)
    parser.add_argument("--ami-split", default=None)
    parser.add_argument("--min-audio-hours", type=float, default=DEFAULT_MIN_AUDIO_HOURS)
    parser.add_argument(
        "--max-meeting-duration-minutes",
        type=float,
        default=DEFAULT_MAX_MEETING_DURATION_MINUTES,
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("CURATOR_AUDIO_TAGGING_CACHE_DIR", DEFAULT_AUDIO_TAGGING_CACHE_DIR),
    )
    parser.add_argument(
        "--model-hf-repo-id",
        default=os.environ.get("CURATOR_AUDIO_TAGGING_MODEL_HF_REPO_ID", DEFAULT_MODEL_HF_REPO_ID),
    )
    parser.add_argument("--model-hf-revision", default=DEFAULT_MODEL_HF_REVISION)
    args = parser.parse_args()

    stage_dataset(
        output_path=args.output_path,
        ami_hf_repo_id=args.hf_repo_id,
        ami_config=args.ami_config,
        ami_hf_revision=args.hf_revision,
        ami_splits=(args.ami_split,) if args.ami_split else DEFAULT_AMI_SPLITS,
        cache_dir=args.cache_dir,
        min_audio_hours=args.min_audio_hours,
        max_meeting_duration_minutes=args.max_meeting_duration_minutes,
    )
    stage_model(
        output_path=args.model_output_path.resolve(),
        hf_repo_id=args.model_hf_repo_id,
        hf_revision=args.model_hf_revision,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
