# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Montreal Forced Aligner -> portable audio shards -> VAD refinement."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tarfile
import tempfile
from itertools import batched
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from tutorials.audio.vad_refinement.rectify_vad_boundaries import (
    SileroSortformerVADDetector,
    SortformerVADConfig,
    VADConfig,
    _check_alignment_timing,
    _marked_words,
    _validate_projection,
    _write_json,
    _write_jsonl,
    process_manifest,
)
from tutorials.audio.vad_refinement.sound_expert_intervals import SoundExpertConfig

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from nemo_curator.tasks import AudioTask
    from tutorials.audio.vad_refinement.rectify_vad_boundaries import VADDetector


class BatchAligner(Protocol):
    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]: ...


def write_alignment(row: dict[str, Any], result: dict[str, Any], output_root: Path) -> None:
    """Keep the supplied transcript and the official MFA TextGrid unchanged."""
    source = result.get("textgrid_filepath")
    if result.get("mfa_skipped") or not source:
        msg = f"MFA returned no word alignment for {row['segment_id']}"
        raise ValueError(msg)
    target = output_root / "alignments" / f"{row['segment_id']}.TextGrid"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    row.update(
        source_text=row.get("text"),
        alignment_source="montreal_forced_aligner",
        alignment_textgrid=str(target.resolve()),
        mfa_textgrid=str(target.resolve()),
    )


def read_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    msg = f"{path}:{line_number}: expected an object"
                    raise TypeError(msg)
                yield row


def _key(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def _validate_alignment(row: dict[str, Any]) -> None:
    duration = float(row["duration"])
    words = _marked_words(Path(row["mfa_textgrid"]))
    if not math.isfinite(duration) or duration <= 0 or not words:
        msg = f"No usable word alignment for {row['segment_id']}"
        raise ValueError(msg)
    _validate_projection(words, words)
    if any(not all(math.isfinite(float(value)) for value in word[:2]) for word in words):
        msg = f"Nonfinite word boundary for {row['segment_id']}"
        raise ValueError(msg)
    if row["offset"] != 0.0:
        msg = "Sharded alignment must use clip-local timestamps (offset=0)"
        raise ValueError(msg)
    _check_alignment_timing(row, Path(row["mfa_textgrid"]))


def input_tasks(
    input_config: Path | None, input_manifest: Path | None, *, batch_size: int = 64
) -> Iterator[tuple[str, list[AudioTask]]]:
    """Use Curator's physical-shard discovery and NeMo/Lhotse audio decoding."""
    import fsspec

    from nemo_curator.stages.audio.io.nemo_speech_reader import (
        NeMoSpeechDiscoveryStage,
        NeMoSpeechReaderStage,
    )
    from nemo_curator.tasks import EmptyTask, FileGroupTask

    config = None
    if input_manifest is not None:
        config = [
            {
                "type": "nemo",
                "corpus": "input",
                "shard_key_prefix": "input",
                "manifest_filepath": str(input_manifest.resolve()),
            }
        ]
    discovery = NeMoSpeechDiscoveryStage(
        yaml_path=str(input_config.resolve()) if input_config else "",
        input_cfg=config,
        resume_mode="checkpoint",
    )
    reader = NeMoSpeechReaderStage()
    for source in discovery.process(EmptyTask()):
        source_key = str(source.reader_config["shard_key"])
        if source.reader_config.get("tar_path"):
            yield source_key, reader.process(source)
            continue
        manifest_path = source.reader_config["manifest_path"]
        with fsspec.open(manifest_path, "rt", encoding="utf-8", compression="infer") as stream:
            rows = (json.loads(line) for line in stream if line.strip())
            for chunk_index, chunk in enumerate(batched(rows, batch_size)):
                with tempfile.TemporaryDirectory(prefix="curator-alignment-input-") as temporary:
                    chunk_manifest = Path(temporary) / "manifest.jsonl"
                    for row in chunk:
                        audio = row.get("audio_filepath")
                        if audio and "://" not in audio and not Path(audio).is_absolute():
                            row["audio_filepath"] = str((Path(manifest_path).parent / audio).resolve())
                    _write_jsonl(chunk_manifest, list(chunk))
                    chunk_task = FileGroupTask(
                        dataset_name=source.dataset_name,
                        data=[str(chunk_manifest)],
                        reader_config={**source.reader_config, "manifest_path": str(chunk_manifest)},
                    )
                    tasks = reader.process(chunk_task)
                    for task in tasks:
                        task._metadata["_shard_input_id"] = str(
                            chunk_index * batch_size + int(task._metadata["_shard_input_id"])
                        )
                        task.data["manifest_origin"] = manifest_path
                yield source_key, tasks


def align_tasks(  # noqa: C901, PLR0913
    sources: Iterable[tuple[str, list[AudioTask]]],
    output_root: Path,
    aligner: BatchAligner,
    *,
    batch_size: int,
    worker_count: int = 1,
    worker_index: int = 0,
) -> list[Path]:
    """Force-align supplied transcripts with MFA; empty or invalid output fails."""
    import soundfile as sf

    if batch_size < 1 or worker_count < 1 or not 0 <= worker_index < worker_count:
        msg = "Invalid batch size or worker assignment"
        raise ValueError(msg)
    manifests: list[Path] = []
    for source_key, tasks in sources:
        for batch in batched(tasks, batch_size):
            batch_key = _key(f"{source_key}:{batch[0]._metadata['_shard_input_id']}")
            if int(_key(batch_key), 16) % worker_count != worker_index:
                continue
            target = output_root / "aligned" / f"{batch_key}.jsonl"
            if target.exists():
                msg = f"Alignment output already exists: {target}; use a fresh output root"
                raise FileExistsError(msg)
            rows = []
            for task in batch:
                data = task.data
                if data.get("read_error") or "waveform" not in data:
                    msg = f"Input audio could not be decoded: {source_key}/{task._metadata.get('_shard_input_id')}"
                    raise ValueError(msg)
                row_id = _key(f"{source_key}:{task._metadata['_shard_input_id']}")
                clip_path = output_root / "clips" / f"{row_id}.wav"
                clip_path.parent.mkdir(parents=True, exist_ok=True)
                if clip_path.exists():
                    msg = f"Clip output already exists: {clip_path}; use a fresh output root"
                    raise FileExistsError(msg)
                # The reader has already applied offset/duration. Both MFA and VAD
                # consume this same lossless clip with a local time origin of zero.
                sf.write(clip_path, data["waveform"], int(data["sampling_rate"]), subtype="PCM_16")
                row = {key: value for key, value in data.items() if key != "waveform"}
                row.update(
                    source_audio_filepath=data["audio_filepath"],
                    source_offset=data.get("offset", 0.0),
                    source_session_id=data.get("session_id"),
                    source_shard=source_key,
                    source_row=task._metadata["_shard_input_id"],
                    segment_id=row_id,
                    session_id=f"clip-{row_id}",
                    speaker_id=str(data.get("speaker_id") or data.get("speaker") or f"speaker-{row_id}"),
                    audio_filepath=str(clip_path.resolve()),
                    offset=0.0,
                    duration=len(data["waveform"]) / int(data["sampling_rate"]),
                    audio_num_frames=len(data["waveform"]),
                    audio_sample_rate=int(data["sampling_rate"]),
                    audio_duration=len(data["waveform"]) / int(data["sampling_rate"]),
                )
                task.data.clear()
                task.data.update(row)
                rows.append(row)
            aligned = aligner.process_batch(list(batch))
            by_id = {task.data["segment_id"]: task.data for task in aligned}
            if len(aligned) != len(rows) or len(by_id) != len(rows):
                msg = f"MFA changed input row count in batch {batch_key}"
                raise RuntimeError(msg)
            for row in rows:
                result = by_id[row["segment_id"]]
                write_alignment(row, result, output_root)
                _validate_alignment(row)
            _write_jsonl(target, rows)
            manifests.append(target)
    return manifests


def pack_shards(aligned_manifests: list[Path], shard_root: Path, *, records_per_shard: int) -> list[Path]:
    """Write portable NeMo manifest/tar pairs with lossless alignment sidecars."""
    if records_per_shard < 1:
        msg = "records_per_shard must be positive"
        raise ValueError(msg)
    shard_root.mkdir(parents=True, exist_ok=True)
    rows = (row for manifest in sorted(aligned_manifests) for row in read_rows(manifest))
    manifests: list[Path] = []
    seen: set[str] = set()
    for shard_index, batch in enumerate(batched(rows, records_per_shard)):
        target = shard_root / f"shard-{shard_index:06d}"
        if target.exists():
            msg = f"Shard output already exists: {target}; use a fresh shard root"
            raise FileExistsError(msg)
        with tempfile.TemporaryDirectory(prefix=".packing-", dir=shard_root) as temporary:
            staging = Path(temporary) / target.name
            (staging / "alignments").mkdir(parents=True)
            packed_rows = []
            with tarfile.open(staging / f"audio_{shard_index}.tar", "w", dereference=True) as archive:
                for source in batch:
                    row_id = str(source["segment_id"])
                    if row_id in seen:
                        msg = f"Duplicate aligned clip ID: {row_id}"
                        raise ValueError(msg)
                    seen.add(row_id)
                    _validate_alignment(source)
                    member_id = _key(row_id)
                    audio_name = f"{member_id}.wav"
                    archive.add(source["audio_filepath"], arcname=audio_name, recursive=False)
                    grid_name = f"alignments/{member_id}.TextGrid"
                    shutil.copyfile(source["mfa_textgrid"], staging / grid_name)
                    row = {
                        **source,
                        "audio_filepath": audio_name,
                        "mfa_textgrid": grid_name,
                        "alignment_textgrid": grid_name,
                        "offset": 0.0,
                        "shard_id": shard_index,
                    }
                    packed_rows.append(row)
            _write_jsonl(staging / f"manifest_{shard_index}.jsonl", packed_rows)
            _write_json(staging / "shard.json", {"rows": len(packed_rows), "format": "nemo_tarred"})
            staging.rename(target)
        manifests.append(target / f"manifest_{shard_index}.jsonl")
    if not manifests:
        msg = "No aligned rows to shard"
        raise ValueError(msg)
    # Relative paths make the complete directory relocatable before inference.
    entries = [
        {
            "type": "nemo_tarred",
            "corpus": "aligned",
            "shard_key_prefix": "aligned",
            "manifest_filepath": str(path.relative_to(shard_root)),
            "tarred_audio_filepaths": str(
                (path.parent / f"audio_{int(path.parent.name.removeprefix('shard-'))}.tar").relative_to(shard_root)
            ),
        }
        for path in manifests
    ]
    _write_json(shard_root / "input.json", [{"input_cfg": entries}])
    return manifests


def refine_shards(  # noqa: PLR0915
    shard_root: Path,
    output_root: Path,
    detector: VADDetector,
    *,
    worker_count: int = 1,
    worker_index: int = 0,
) -> list[Path]:
    """Read physical tar shards through Curator and preserve per-shard outputs."""
    import soundfile as sf

    from nemo_curator.stages.audio.io.nemo_speech_reader import NeMoSpeechReaderStage
    from nemo_curator.tasks import FileGroupTask

    if worker_count < 1 or not 0 <= worker_index < worker_count:
        msg = "Invalid worker assignment"
        raise ValueError(msg)
    manifests = sorted(shard_root.glob("shard-*/manifest_*.jsonl"))
    if not manifests:
        msg = f"No physical shards under {shard_root}"
        raise ValueError(msg)
    reader = NeMoSpeechReaderStage()
    outputs = []
    for manifest in manifests:
        shard_name = manifest.parent.name
        if int(shard_name.removeprefix("shard-")) % worker_count != worker_index:
            continue
        target = output_root / shard_name
        target.mkdir(parents=True, exist_ok=False)
        source_rows = list(read_rows(manifest))
        source = FileGroupTask(
            dataset_name="aligned",
            data=[str(manifest), str(manifest.parent / f"audio_{int(shard_name.removeprefix('shard-'))}.tar")],
            reader_config={
                "manifest_path": str(manifest),
                "tar_path": str(manifest.parent / f"audio_{int(shard_name.removeprefix('shard-'))}.tar"),
                "shard_key": shard_name,
                "corpus": "aligned",
            },
        )
        tasks = reader.process(source)
        if len(tasks) != len(source_rows):
            msg = f"Shard row count changed: {shard_name}"
            raise RuntimeError(msg)
        refined_rows = []
        seen_rows: set[int] = set()
        for task in tasks:
            index = int(task._metadata["_shard_input_id"])
            if index in seen_rows or not 0 <= index < len(source_rows):
                msg = f"Invalid/duplicate decoded row {index} in {shard_name}"
                raise ValueError(msg)
            seen_rows.add(index)
            row = dict(source_rows[index])
            if task.data.get("read_error") or "waveform" not in task.data:
                msg = f"Cannot decode {shard_name} row {index}; no completion marker written"
                raise ValueError(msg)
            row_dir = target / "clips" / f"{index:06d}"
            row_dir.mkdir(parents=True)
            audio_path = row_dir / "audio.wav"
            sf.write(audio_path, task.data["waveform"], int(task.data["sampling_rate"]), subtype="FLOAT")
            grid_path = (manifest.parent / row["mfa_textgrid"]).resolve()
            if not grid_path.is_relative_to(manifest.parent.resolve()):
                msg = f"Alignment path escapes physical shard: {grid_path}"
                raise ValueError(msg)
            local_grid = row_dir / "alignment.TextGrid"
            shutil.copyfile(grid_path, local_grid)
            row.update(
                audio_filepath=str(audio_path.resolve()),
                mfa_textgrid=str(local_grid.resolve()),
                alignment_textgrid=str(local_grid.resolve()),
                offset=0.0,
                duration=len(task.data["waveform"]) / int(task.data["sampling_rate"]),
            )
            _validate_alignment(row)
            row_manifest = row_dir / "input.jsonl"
            _write_jsonl(row_manifest, [row])
            summary = process_manifest(row_manifest, row_dir / "refined", detector)
            summary["anchor_source"] = row["alignment_source"]
            summary["refinement_algorithm"]["hard_constraint"] = "preserve_alignment_words"
            summary["refinement_algorithm"]["word_boundary_policy"] = "midpoint_of_alignment_gap_for_shared_component"
            summary["refinement_algorithm"]["weights"]["alignment"] = summary["refinement_algorithm"]["weights"].pop(
                "mfa"
            )
            _write_json(Path(summary["output_manifest"]).parent / "audit.json", summary)
            (refined_row,) = read_rows(Path(summary["output_manifest"]))
            refined_row.update(
                alignment_shard=shard_name,
                alignment_shard_row=index,
                refinement_audit=str(Path(summary["output_manifest"]).parent / "audit.json"),
            )
            refined_rows.append(refined_row)
        output = target / "manifest_refined.jsonl"
        _write_jsonl(output, refined_rows)
        _write_json(target / "complete.json", {"rows": len(refined_rows), "source_manifest": str(manifest.resolve())})
        outputs.append(output)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("align", "shard", "refine", "all"), default="all")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input-manifest", type=Path)
    source.add_argument("--input-config", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mfa-command", default="mfa")
    parser.add_argument("--mfa-dictionary")
    parser.add_argument("--mfa-acoustic-model")
    parser.add_argument("--mfa-model-root", type=Path)
    parser.add_argument("--mfa-jobs", type=int, default=4)
    parser.add_argument("--mfa-timeout", type=float, default=3600.0)
    parser.add_argument("--alignment-batch-size", type=int, default=32)
    parser.add_argument("--records-per-shard", type=int, default=64)
    parser.add_argument("--sortformer-model")
    parser.add_argument("--sortformer-device", default="auto")
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=0)
    args = parser.parse_args()
    if min(args.mfa_jobs, args.alignment_batch_size, args.records_per_shard, args.worker_count) < 1:
        parser.error("Batch, shard and worker counts must be positive")
    if not 0 <= args.worker_index < args.worker_count:
        parser.error("worker-index must be in [0, worker-count)")
    if not math.isfinite(args.mfa_timeout) or args.mfa_timeout <= 0:
        parser.error("mfa-timeout must be finite and positive")
    if args.stage in {"align", "all"} and (
        not (args.input_manifest or args.input_config) or not args.mfa_dictionary or not args.mfa_acoustic_model
    ):
        parser.error("Alignment requires input, --mfa-dictionary and --mfa-acoustic-model")
    if args.stage in {"refine", "all"} and not args.sortformer_model:
        parser.error("Refinement requires --sortformer-model")
    if args.stage in {"shard", "all"} and args.worker_count != 1:
        parser.error("Shard assembly is single-writer; parallelize align and refine separately")
    return args


def main() -> int:
    args = parse_args()
    root = args.output_root.resolve()
    if args.stage in {"align", "all"}:
        from nemo_curator.stages.audio.alignment import MFAAlignmentStage

        aligner = MFAAlignmentStage(
            output_dir=str(root / "mfa"),
            mfa_command=args.mfa_command,
            dictionary=args.mfa_dictionary,
            acoustic_model=args.mfa_acoustic_model,
            num_jobs=args.mfa_jobs,
            batch_size=args.alignment_batch_size,
            mfa_root_dir=str(args.mfa_model_root) if args.mfa_model_root else None,
            g2p_model="",
            create_rttm=False,
            create_ctm=False,
            speaker_key="speaker_id",
            align_timeout_seconds=args.mfa_timeout,
            local_mfa_base_dir=str(
                Path(tempfile.gettempdir()) / "curator-mfa" / _key(str(root)) / f"worker-{args.worker_index:06d}"
            ),
        )
        try:
            aligner.setup_on_node()
            aligner.setup()
            align_tasks(
                input_tasks(args.input_config, args.input_manifest, batch_size=args.alignment_batch_size),
                root,
                aligner,
                batch_size=args.alignment_batch_size,
                worker_count=args.worker_count,
                worker_index=args.worker_index,
            )
        finally:
            aligner.teardown()
    if args.stage in {"shard", "all"}:
        pack_shards(
            list((root / "aligned").glob("*.jsonl")), root / "shards", records_per_shard=args.records_per_shard
        )
    if args.stage in {"refine", "all"}:
        detector = SileroSortformerVADDetector(
            VADConfig(),
            SortformerVADConfig(model=args.sortformer_model, device=args.sortformer_device),
            SoundExpertConfig(),
        )
        refine_shards(
            root / "shards", root / "refined", detector, worker_count=args.worker_count, worker_index=args.worker_index
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
