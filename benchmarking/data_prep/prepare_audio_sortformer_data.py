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

"""Stage fixed-duration public LibriSpeech bundles and the Sortformer model."""

from __future__ import annotations

import argparse
import io
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import soundfile as sf
from datasets import Audio, load_dataset
from huggingface_hub import hf_hub_download
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Iterator

DEFAULT_CACHE_DIR = "/tmp/curator/audio_sortformer_cache"  # noqa: S108
LIBRISPEECH_HF_REPO_ID = "openslr/librispeech_asr"  # CC-BY-4.0
LIBRISPEECH_CONFIG = "all"
LIBRISPEECH_HF_REVISION = "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"  # pragma: allowlist secret
LIBRISPEECH_SPLITS = ("train.clean.100", "train.clean.360")
NUM_BUNDLES = 1800
BUNDLE_DURATION_S = 900
SAMPLE_RATE = 16000
MODEL_HF_REPO_ID = "nvidia/diar_streaming_sortformer_4spk-v2.1"
MODEL_FILENAME = "diar_streaming_sortformer_4spk-v2.1.nemo"


def _bundle_filename(index: int) -> str:
    return f"librispeech_bundle_{index:04d}.wav"


def verify_dataset(
    output_path: Path, num_bundles: int = NUM_BUNDLES, bundle_duration_s: int = BUNDLE_DURATION_S
) -> bool:
    manifest_path = output_path / "manifest.jsonl"
    audio_dir = output_path / "audio"
    expected_frames = bundle_duration_s * SAMPLE_RATE
    try:
        rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
        source_ids = [source_id for row in rows for source_id in row["source_clip_ids"]]
        is_ready = (
            len(rows) == num_bundles
            and len(source_ids) == len(set(source_ids))
            and all(
                row["audio_filepath"] == f"audio/{_bundle_filename(index)}"
                and row["audio_item_id"] == Path(_bundle_filename(index)).stem
                and row["duration"] == bundle_duration_s
                and row["source_clip_ids"]
                and (audio_path := audio_dir / _bundle_filename(index)).is_file()
                and (audio_info := sf.info(audio_path)).samplerate == SAMPLE_RATE
                and audio_info.channels == 1
                and audio_info.frames == expected_frames
                for index, row in enumerate(rows)
            )
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError, KeyError, TypeError):
        is_ready = False

    if not is_ready:
        logger.error(f"Incomplete Sortformer dataset under {output_path}")
        return False
    logger.info(
        f"Verified {num_bundles} unique LibriSpeech bundles ({num_bundles * bundle_duration_s / 3600:.3f} audio hours)"
    )
    return True


def verify_model(model_path: Path) -> bool:
    is_ready = model_path.is_file() and model_path.stat().st_size > 0
    if not is_ready:
        logger.error(f"Sortformer model is missing or empty: {model_path}")
    return is_ready


def _manifest_audio(source_manifest: Path) -> Iterator[tuple[str, Path | bytes]]:
    with source_manifest.open(encoding="utf-8") as manifest_file:
        for line in manifest_file:
            if not line.strip():
                continue
            row = json.loads(line)
            audio_path = Path(row["audio_filepath"])
            if not audio_path.is_absolute():
                audio_path = source_manifest.parent / audio_path
            yield str(row.get("audio_item_id", audio_path.stem)), audio_path


def _public_audio(cache_dir: str) -> Iterator[tuple[str, Path | bytes]]:
    datasets = load_dataset(
        LIBRISPEECH_HF_REPO_ID,
        LIBRISPEECH_CONFIG,
        revision=LIBRISPEECH_HF_REVISION,
        cache_dir=cache_dir,
        streaming=True,
    ).cast_column("audio", Audio(decode=False))
    for split in LIBRISPEECH_SPLITS:
        for row in datasets[split]:
            audio = row["audio"]
            yield str(row["id"]), audio["bytes"] if audio.get("bytes") is not None else Path(audio["path"])


def _write_source_audio(source: Path | bytes, target: sf.SoundFile, max_frames: int) -> int:
    source_input = io.BytesIO(source) if isinstance(source, bytes) else source
    with sf.SoundFile(source_input) as source_file:
        if source_file.samplerate != SAMPLE_RATE or source_file.channels != 1:
            msg = f"Expected mono {SAMPLE_RATE} Hz audio, got {source_file.channels} channels at {source_file.samplerate} Hz"
            raise RuntimeError(msg)
        frames_written = 0
        for block in source_file.blocks(blocksize=65536, dtype="int16", always_2d=True):
            frames = block[: max_frames - frames_written]
            target.write(frames)
            frames_written += len(frames)
            if frames_written == max_frames:
                break
    return frames_written


def stage_dataset(
    output_path: Path,
    cache_dir: str,
    source_manifest: Path | None = None,
    num_bundles: int = NUM_BUNDLES,
    bundle_duration_s: int = BUNDLE_DURATION_S,
) -> None:
    audio_dir = output_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / "manifest.jsonl"
    temporary_manifest = manifest_path.with_suffix(".jsonl.tmp")
    source_rows = iter(_manifest_audio(source_manifest) if source_manifest else _public_audio(cache_dir))
    target_frames = bundle_duration_s * SAMPLE_RATE
    used_source_ids: set[str] = set()

    try:
        with temporary_manifest.open("w", encoding="utf-8") as manifest_file:
            for bundle_index in range(num_bundles):
                filename = _bundle_filename(bundle_index)
                audio_item_id = Path(filename).stem
                source_clip_ids: list[str] = []
                frames_written = 0
                with sf.SoundFile(
                    audio_dir / filename,
                    mode="w",
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    subtype="PCM_16",
                ) as target_file:
                    while frames_written < target_frames:
                        try:
                            source_id, source_audio = next(source_rows)
                        except StopIteration as e:
                            msg = f"LibriSpeech source ended after {bundle_index} complete bundles"
                            raise RuntimeError(msg) from e
                        if source_id in used_source_ids:
                            msg = f"Duplicate source clip ID: {source_id}"
                            raise RuntimeError(msg)  # noqa: TRY301
                        used_source_ids.add(source_id)
                        source_clip_ids.append(source_id)
                        # Discard the unused tail so a source clip can belong to only one bundle.
                        frames_written += _write_source_audio(
                            source_audio, target_file, target_frames - frames_written
                        )
                manifest_file.write(
                    json.dumps(
                        {
                            "audio_filepath": f"audio/{filename}",
                            "audio_item_id": audio_item_id,
                            "duration": bundle_duration_s,
                            "source_clip_ids": source_clip_ids,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                if (bundle_index + 1) % 100 == 0 or bundle_index + 1 == num_bundles:
                    logger.info(f"Wrote bundle {bundle_index + 1}/{num_bundles}")
        temporary_manifest.replace(manifest_path)
    except Exception:
        temporary_manifest.unlink(missing_ok=True)
        raise
    logger.success(f"Dataset ready: {num_bundles} bundles / {num_bundles * bundle_duration_s / 3600:.3f} hours")


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
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--num-bundles", type=int, default=NUM_BUNDLES)
    parser.add_argument("--bundle-duration-s", type=int, default=BUNDLE_DURATION_S)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    output_path = args.output_path.resolve()
    model_path = args.model_output_path.resolve()
    source_manifest = args.source_manifest.resolve() if args.source_manifest else None
    if args.verify_only:
        return (
            0
            if verify_dataset(output_path, args.num_bundles, args.bundle_duration_s) and verify_model(model_path)
            else 1
        )

    if not verify_dataset(output_path, args.num_bundles, args.bundle_duration_s):
        stage_dataset(output_path, args.cache_dir, source_manifest, args.num_bundles, args.bundle_duration_s)
    if not verify_model(model_path):
        stage_model(model_path, args.cache_dir)
    return (
        0 if verify_dataset(output_path, args.num_bundles, args.bundle_duration_s) and verify_model(model_path) else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
