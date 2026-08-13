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

"""Build a fixed, unique JPEG WebDataset fixture from MINT-1T.

The image-curation reader currently consumes individual JPEG members, while
the staged MINT-1T source stores many images as frames in TIFF documents. This
one-time preparation script selects unique images by their source SHA-256,
extracts the corresponding TIFF frames, and writes deterministic WebDataset
shards. It deliberately over-selects a small number of candidates per shard so
that corrupt source frames can be skipped without changing the requested
output cardinality.

Selection is deterministic: the lexicographically smallest unique source
SHA-256 values that satisfy the dimension bounds are retained. The resulting
fixture and its manifests can be reused by every benchmark run.

Example::

    python prepare_image_curation_data.py \
        --input-path /datasets/multimodal/mint1t/CC-MAIN-2024-18-shard-0 \
        --output-path /datasets/image_curation/mint1t_unique_jpeg_1m_v1
"""

from __future__ import annotations

import argparse
import heapq
import io
import json
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from loguru import logger
from PIL import Image

SELECTION_ALGORITHM = "smallest-source-sha256-v1"
SELECTION_CONFIG_NAME = "selection.json"
CANDIDATE_MANIFEST_NAME = "candidates.jsonl"
SUCCESS_MARKER_NAME = "_SUCCESS.json"
SHARD_MANIFEST_DIR_NAME = "shard_manifests"
TIFF_EXTENSIONS = (".tiff", ".tif")
SHA256_HEX_LENGTH = 64
MAX_JPEG_QUALITY = 100


@dataclass(frozen=True)
class ImageReference:
    """Location and provenance for one unique source image."""

    source_sha256: str
    source_tar: str
    json_member: str
    image_member: str
    frame_index: int
    width: int
    height: int


@dataclass(frozen=True)
class BuildShardTask:
    """Serializable work item for one output shard."""

    input_path: str
    output_path: str
    shard_index: int
    expected_images: int
    jpeg_quality: int
    candidates: tuple[ImageReference, ...]


def _valid_sha256(value: object) -> str | None:
    if not isinstance(value, str) or len(value) != SHA256_HEX_LENGTH:
        return None
    normalized = value.lower()
    try:
        int(normalized, 16)
    except ValueError:
        return None
    return normalized


def _resolve_default_image_member(json_member: str, member_names: set[str]) -> str | None:
    stem = str(Path(json_member).with_suffix(""))
    return next((f"{stem}{extension}" for extension in TIFF_EXTENSIONS if f"{stem}{extension}" in member_names), None)


def _references_from_payload(  # noqa: PLR0913
    payload: dict[str, Any],
    *,
    source_tar: str,
    json_member: str,
    member_names: set[str],
    min_side: int,
    max_pixels: int,
) -> tuple[list[ImageReference], dict[str, int]]:
    """Return eligible image references from one MINT-1T JSON sample."""
    images = payload.get("images")
    metadata = payload.get("image_metadata")
    stats = {"image_tokens": 0, "eligible": 0, "missing_member": 0, "invalid_metadata": 0}
    if not isinstance(images, list) or not isinstance(metadata, list):
        stats["invalid_metadata"] = 1
        return [], stats

    default_member = _resolve_default_image_member(json_member, member_names)
    frame_counters: dict[str, int] = {}
    references: list[ImageReference] = []
    metadata_index = 0
    for image_token in images:
        if image_token is None:
            continue
        stats["image_tokens"] += 1
        image_metadata = metadata[metadata_index] if metadata_index < len(metadata) else None
        metadata_index += 1
        image_member = image_token if isinstance(image_token, str) and image_token in member_names else default_member
        if image_member is None:
            stats["missing_member"] += 1
            continue

        frame_index = frame_counters.get(image_member, 0)
        frame_counters[image_member] = frame_index + 1
        if not isinstance(image_metadata, dict):
            stats["invalid_metadata"] += 1
            continue
        source_sha256 = _valid_sha256(image_metadata.get("sha256"))
        width = image_metadata.get("width")
        height = image_metadata.get("height")
        if (
            source_sha256 is None
            or not isinstance(width, int)
            or isinstance(width, bool)
            or not isinstance(height, int)
            or isinstance(height, bool)
            or min(width, height) < min_side
            or width * height > max_pixels
        ):
            stats["invalid_metadata"] += 1
            continue
        references.append(
            ImageReference(
                source_sha256=source_sha256,
                source_tar=source_tar,
                json_member=json_member,
                image_member=image_member,
                frame_index=frame_index,
                width=width,
                height=height,
            )
        )
        stats["eligible"] += 1
    return references, stats


def select_candidates(  # noqa: C901, PLR0912
    input_path: Path,
    *,
    num_candidates: int,
    min_side: int,
    max_pixels: int,
) -> tuple[list[ImageReference], dict[str, int]]:
    """Select the smallest unique source hashes without retaining the full corpus in memory."""
    input_tars = sorted(input_path.glob("*.tar"))
    if not input_tars:
        msg = f"No .tar files found under {input_path}"
        raise FileNotFoundError(msg)

    # Negative hash integers make heap[0] the largest currently retained SHA.
    selected_heap: list[tuple[int, int, ImageReference]] = []
    selected_hashes: set[str] = set()
    stats = {
        "input_tars": len(input_tars),
        "json_samples": 0,
        "image_tokens": 0,
        "eligible": 0,
        "duplicates_seen_while_selected": 0,
        "missing_member": 0,
        "invalid_metadata": 0,
    }
    serial = 0
    for tar_index, tar_path in enumerate(input_tars, start=1):
        relative_tar = tar_path.relative_to(input_path).as_posix()
        with tarfile.open(tar_path, "r") as source_tar:
            members = [member for member in source_tar.getmembers() if member.isfile()]
            member_names = {member.name for member in members}
            json_members = sorted(
                (member for member in members if member.name.endswith(".json")), key=lambda m: m.name
            )
            for member in json_members:
                extracted = source_tar.extractfile(member)
                if extracted is None:
                    stats["invalid_metadata"] += 1
                    continue
                try:
                    payload = json.load(extracted)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    stats["invalid_metadata"] += 1
                    continue
                if not isinstance(payload, dict):
                    stats["invalid_metadata"] += 1
                    continue
                stats["json_samples"] += 1
                references, sample_stats = _references_from_payload(
                    payload,
                    source_tar=relative_tar,
                    json_member=member.name,
                    member_names=member_names,
                    min_side=min_side,
                    max_pixels=max_pixels,
                )
                for key in ("image_tokens", "eligible", "missing_member", "invalid_metadata"):
                    stats[key] += sample_stats[key]
                for reference in references:
                    if reference.source_sha256 in selected_hashes:
                        stats["duplicates_seen_while_selected"] += 1
                        continue
                    hash_value = int(reference.source_sha256, 16)
                    if len(selected_heap) < num_candidates:
                        heapq.heappush(selected_heap, (-hash_value, serial, reference))
                        selected_hashes.add(reference.source_sha256)
                        serial += 1
                    elif hash_value < -selected_heap[0][0]:
                        _, _, removed = heapq.heapreplace(selected_heap, (-hash_value, serial, reference))
                        selected_hashes.remove(removed.source_sha256)
                        selected_hashes.add(reference.source_sha256)
                        serial += 1
        if tar_index % 100 == 0 or tar_index == len(input_tars):
            logger.info(
                f"Scanned {tar_index}/{len(input_tars)} source tars; retained "
                f"{len(selected_heap)}/{num_candidates} unique candidates"
            )

    if len(selected_heap) != num_candidates:
        msg = f"Found only {len(selected_heap)} eligible unique images; {num_candidates} candidates are required"
        raise RuntimeError(msg)
    candidates = [entry[2] for entry in selected_heap]
    # Source ordering minimizes source-tar opens during extraction. Selection
    # remains defined exclusively by the source hash.
    candidates.sort(key=lambda ref: (ref.source_tar, ref.image_member, ref.frame_index, ref.source_sha256))
    stats["selected_candidates"] = len(candidates)
    return candidates, stats


def _selection_config(  # noqa: PLR0913
    input_path: Path,
    *,
    num_images: int,
    images_per_tar: int,
    candidate_buffer_per_shard: int,
    min_side: int,
    max_pixels: int,
) -> dict[str, Any]:
    num_shards = num_images // images_per_tar
    return {
        "algorithm": SELECTION_ALGORITHM,
        "input_path": str(input_path.resolve()),
        "num_images": num_images,
        "images_per_tar": images_per_tar,
        "num_shards": num_shards,
        "candidate_buffer_per_shard": candidate_buffer_per_shard,
        "num_candidates": num_shards * (images_per_tar + candidate_buffer_per_shard),
        "min_side": min_side,
        "max_pixels": max_pixels,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_path, path)


def _write_candidates(
    output_path: Path, candidates: list[ImageReference], config: dict[str, Any], stats: dict[str, int]
) -> None:
    manifest_path = output_path / CANDIDATE_MANIFEST_NAME
    temporary_path = manifest_path.with_name(f".{manifest_path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as manifest_file:
        for reference in candidates:
            manifest_file.write(json.dumps(asdict(reference), sort_keys=True) + "\n")
    os.replace(temporary_path, manifest_path)
    _write_json_atomic(output_path / SELECTION_CONFIG_NAME, {**config, "scan_stats": stats})


def _load_candidates(output_path: Path, expected_config: dict[str, Any]) -> list[ImageReference] | None:
    config_path = output_path / SELECTION_CONFIG_NAME
    manifest_path = output_path / CANDIDATE_MANIFEST_NAME
    if not config_path.is_file() and not manifest_path.is_file():
        return None
    if not config_path.is_file() or not manifest_path.is_file():
        msg = f"Incomplete cached selection under {output_path}; remove the selection files and retry"
        raise RuntimeError(msg)
    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    mismatches = {
        key: (saved_config.get(key), value) for key, value in expected_config.items() if saved_config.get(key) != value
    }
    if mismatches:
        msg = f"Cached selection configuration does not match requested configuration: {mismatches}"
        raise RuntimeError(msg)
    candidates: list[ImageReference] = []
    with manifest_path.open(encoding="utf-8") as manifest_file:
        for line in manifest_file:
            if line.strip():
                candidates.append(ImageReference(**json.loads(line)))
    expected_count = int(expected_config["num_candidates"])
    if len(candidates) != expected_count:
        msg = f"Cached candidate manifest has {len(candidates)} rows; expected {expected_count}"
        raise RuntimeError(msg)
    logger.info(f"Reusing {len(candidates)} cached candidates from {manifest_path}")
    return candidates


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.size = size
    info.mtime = 0
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def _open_source_image(source_tar: tarfile.TarFile, member_name: str) -> Image.Image:
    extracted = source_tar.extractfile(member_name)
    if extracted is None:
        msg = f"Unable to extract source image member {member_name}"
        raise FileNotFoundError(msg)
    return Image.open(io.BytesIO(extracted.read()))


def _require_exact_shard(num_images: int, expected_images: int, decode_errors: int, shard_index: int) -> None:
    if num_images != expected_images:
        msg = f"Shard {shard_index} produced {num_images}/{expected_images} images after {decode_errors} decode errors"
        raise RuntimeError(msg)


def _build_shard(task: BuildShardTask) -> dict[str, Any]:  # noqa: C901, PLR0912, PLR0915
    output_path = Path(task.output_path)
    shard_name = f"{task.shard_index:06d}.tar"
    shard_path = output_path / shard_name
    shard_manifest_path = output_path / SHARD_MANIFEST_DIR_NAME / f"{task.shard_index:06d}.jsonl"
    if shard_path.is_file() and shard_manifest_path.is_file():
        existing_rows = sum(1 for line in shard_manifest_path.open(encoding="utf-8") if line.strip())
        if existing_rows == task.expected_images:
            return {"shard": shard_name, "images": existing_rows, "decode_errors": 0, "reused": True}

    temporary_tar_path = shard_path.with_name(f".{shard_path.name}.partial")
    temporary_manifest_path = shard_manifest_path.with_name(f".{shard_manifest_path.name}.partial")
    temporary_tar_path.unlink(missing_ok=True)
    temporary_manifest_path.unlink(missing_ok=True)

    input_path = Path(task.input_path)
    current_tar_name: str | None = None
    current_image_member: str | None = None
    current_tar: tarfile.TarFile | None = None
    current_image: Image.Image | None = None
    manifest_rows: list[dict[str, Any]] = []
    decode_errors = 0
    try:
        with tarfile.open(temporary_tar_path, "w") as output_tar:
            for reference in task.candidates:
                if len(manifest_rows) >= task.expected_images:
                    break
                try:
                    if reference.source_tar != current_tar_name:
                        if current_image is not None:
                            current_image.close()
                        if current_tar is not None:
                            current_tar.close()
                        current_tar = tarfile.open(input_path / reference.source_tar, "r")  # noqa: SIM115
                        current_tar_name = reference.source_tar
                        current_image_member = None
                        current_image = None
                    if reference.image_member != current_image_member:
                        if current_image is not None:
                            current_image.close()
                        current_image = _open_source_image(current_tar, reference.image_member)
                        current_image_member = reference.image_member
                    current_image.seek(reference.frame_index)
                    jpeg_buffer = io.BytesIO()
                    with current_image.copy() as frame:
                        if frame.mode == "RGB":
                            frame.save(
                                jpeg_buffer,
                                format="JPEG",
                                quality=task.jpeg_quality,
                                optimize=False,
                                progressive=False,
                            )
                        else:
                            with frame.convert("RGB") as rgb_frame:
                                rgb_frame.save(
                                    jpeg_buffer,
                                    format="JPEG",
                                    quality=task.jpeg_quality,
                                    optimize=False,
                                    progressive=False,
                                )
                    jpeg_bytes = jpeg_buffer.getvalue()
                except (FileNotFoundError, OSError, RuntimeError, SyntaxError, ValueError, EOFError):
                    decode_errors += 1
                    current_image_member = None
                    if current_image is not None:
                        current_image.close()
                        current_image = None
                    continue

                member_name = f"{reference.source_sha256}.jpg"
                output_tar.addfile(_tar_info(member_name, len(jpeg_bytes)), io.BytesIO(jpeg_bytes))
                manifest_rows.append(
                    {
                        **asdict(reference),
                        "output_shard": shard_name,
                        "output_member": member_name,
                        "output_index": len(manifest_rows),
                    }
                )
        _require_exact_shard(len(manifest_rows), task.expected_images, decode_errors, task.shard_index)
        with temporary_manifest_path.open("w", encoding="utf-8") as manifest_file:
            for row in manifest_rows:
                manifest_file.write(json.dumps(row, sort_keys=True) + "\n")
        os.replace(temporary_tar_path, shard_path)
        os.replace(temporary_manifest_path, shard_manifest_path)
    except Exception:
        temporary_tar_path.unlink(missing_ok=True)
        temporary_manifest_path.unlink(missing_ok=True)
        raise
    finally:
        if current_image is not None:
            current_image.close()
        if current_tar is not None:
            current_tar.close()
    return {"shard": shard_name, "images": len(manifest_rows), "decode_errors": decode_errors, "reused": False}


def build_dataset(  # noqa: PLR0913
    input_path: Path,
    output_path: Path,
    candidates: list[ImageReference],
    *,
    num_images: int,
    images_per_tar: int,
    candidate_buffer_per_shard: int,
    jpeg_quality: int,
    workers: int,
) -> dict[str, int]:
    """Materialize candidates into exact-cardinality JPEG WebDataset shards."""
    num_shards = num_images // images_per_tar
    candidates_per_shard = images_per_tar + candidate_buffer_per_shard
    expected_candidates = num_shards * candidates_per_shard
    if len(candidates) != expected_candidates:
        msg = f"Received {len(candidates)} candidates; expected {expected_candidates}"
        raise ValueError(msg)
    manifest_dir = output_path / SHARD_MANIFEST_DIR_NAME
    manifest_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        BuildShardTask(
            input_path=str(input_path),
            output_path=str(output_path),
            shard_index=shard_index,
            expected_images=images_per_tar,
            jpeg_quality=jpeg_quality,
            candidates=tuple(
                candidates[shard_index * candidates_per_shard : (shard_index + 1) * candidates_per_shard]
            ),
        )
        for shard_index in range(num_shards)
    ]
    totals = {"images": 0, "decode_errors": 0, "reused_shards": 0}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for completed, result in enumerate(executor.map(_build_shard, tasks), start=1):
            totals["images"] += int(result["images"])
            totals["decode_errors"] += int(result["decode_errors"])
            totals["reused_shards"] += int(bool(result["reused"]))
            if completed % 10 == 0 or completed == num_shards:
                logger.info(f"Built or reused {completed}/{num_shards} output shards")
    return totals


def verify_dataset(output_path: Path, *, num_images: int, images_per_tar: int) -> dict[str, int]:
    """Fully verify shard cardinality, extensions, and global source uniqueness."""
    num_shards = num_images // images_per_tar
    tar_paths = sorted(output_path.glob("*.tar"))
    if len(tar_paths) != num_shards:
        msg = f"Found {len(tar_paths)} output tars; expected {num_shards}"
        raise RuntimeError(msg)

    member_names: set[str] = set()
    for expected_index, tar_path in enumerate(tar_paths):
        if tar_path.name != f"{expected_index:06d}.tar":
            msg = f"Unexpected shard name {tar_path.name}; expected {expected_index:06d}.tar"
            raise RuntimeError(msg)
        with tarfile.open(tar_path, "r") as dataset_tar:
            shard_members = [member for member in dataset_tar.getmembers() if member.isfile()]
        if len(shard_members) != images_per_tar:
            msg = f"Shard {tar_path.name} contains {len(shard_members)} files; expected {images_per_tar}"
            raise RuntimeError(msg)
        for member in shard_members:
            if not member.name.endswith(".jpg"):
                msg = f"Unexpected non-JPEG member {member.name} in {tar_path.name}"
                raise RuntimeError(msg)
            if member.name in member_names:
                msg = f"Duplicate output image member {member.name}"
                raise RuntimeError(msg)
            member_names.add(member.name)
    if len(member_names) != num_images:
        msg = f"Found {len(member_names)} unique images; expected {num_images}"
        raise RuntimeError(msg)
    return {"num_shards": len(tar_paths), "num_images": len(member_names), "unique_images": len(member_names)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True, help="Directory containing MINT-1T tar shards")
    parser.add_argument("--output-path", type=Path, required=True, help="Directory for the prepared JPEG WebDataset")
    parser.add_argument("--num-images", type=int, default=1_000_000, help="Exact number of unique output images")
    parser.add_argument("--images-per-tar", type=int, default=1_000, help="Exact images per output tar")
    parser.add_argument(
        "--candidate-buffer-per-shard",
        type=int,
        default=50,
        help="Extra deterministic candidates per output shard to tolerate corrupt source frames",
    )
    parser.add_argument("--min-side", type=int, default=32, help="Minimum source width and height")
    parser.add_argument("--max-pixels", type=int, default=40_000_000, help="Maximum source width times height")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality for extracted frames")
    parser.add_argument("--workers", type=int, default=16, help="Parallel output-shard workers")
    parser.add_argument("--selection-only", action="store_true", help="Write candidate selection without decoding")
    parser.add_argument("--verify-only", action="store_true", help="Verify an existing prepared dataset")
    args = parser.parse_args()

    if args.num_images <= 0 or args.images_per_tar <= 0 or args.num_images % args.images_per_tar != 0:
        parser.error("--num-images must be positive and evenly divisible by --images-per-tar")
    if args.candidate_buffer_per_shard < 0:
        parser.error("--candidate-buffer-per-shard must be nonnegative")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if not 1 <= args.jpeg_quality <= MAX_JPEG_QUALITY:
        parser.error("--jpeg-quality must be between 1 and 100")
    if args.selection_only and args.verify_only:
        parser.error("--selection-only and --verify-only cannot be combined")

    input_path = args.input_path.resolve()
    output_path = args.output_path.resolve()
    if args.verify_only:
        verification = verify_dataset(output_path, num_images=args.num_images, images_per_tar=args.images_per_tar)
        logger.success(f"Verified image-curation fixture: {verification}")
        return 0
    if not input_path.is_dir():
        parser.error(f"Input directory does not exist: {input_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    config = _selection_config(
        input_path,
        num_images=args.num_images,
        images_per_tar=args.images_per_tar,
        candidate_buffer_per_shard=args.candidate_buffer_per_shard,
        min_side=args.min_side,
        max_pixels=args.max_pixels,
    )
    candidates = _load_candidates(output_path, config)
    if candidates is None:
        candidates, scan_stats = select_candidates(
            input_path,
            num_candidates=int(config["num_candidates"]),
            min_side=args.min_side,
            max_pixels=args.max_pixels,
        )
        _write_candidates(output_path, candidates, config, scan_stats)
        logger.success(f"Selected {len(candidates)} deterministic unique candidates")
    if args.selection_only:
        return 0

    build_stats = build_dataset(
        input_path,
        output_path,
        candidates,
        num_images=args.num_images,
        images_per_tar=args.images_per_tar,
        candidate_buffer_per_shard=args.candidate_buffer_per_shard,
        jpeg_quality=args.jpeg_quality,
        workers=args.workers,
    )
    verification = verify_dataset(output_path, num_images=args.num_images, images_per_tar=args.images_per_tar)
    success_payload = {
        **config,
        "jpeg_quality": args.jpeg_quality,
        "build_stats": build_stats,
        "verification": verification,
    }
    _write_json_atomic(output_path / SUCCESS_MARKER_NAME, success_payload)
    logger.success(f"Prepared and verified image-curation fixture: {success_payload}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
