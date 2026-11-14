# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""English audio + text -> existing Curator MFA -> Lhotse -> native FastMSS.

The three subcommands permit MFA and FastMSS to use their own existing
environments. They exchange ordinary JSONL/CTM and Lhotse manifests, without
changing the agent, MFA stage, or FastMSS generation implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter
from importlib.metadata import distribution
from pathlib import Path

_MIN_SPEAKERS = 2
_CTM_FIELDS = 5
_SAMPLE_RATE = 16000
_MAX_CLIP_DURATION = 20
_MAX_MEETINGS = 32
_MAX_DURATION_TARGET = 300


def _source_row(row: object) -> dict:
    if not isinstance(row, dict):
        msg = "Each manifest row must be an object"
        raise TypeError(msg)
    for key in ("audio_filepath", "text", "speaker"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            msg = f"Every clip requires a nonempty {key}"
            raise ValueError(msg)
    if any(character.isspace() for character in row["speaker"]):
        msg = "Use speaker IDs without whitespace, such as Speaker_0, consistently across the corpus"
        raise ValueError(msg)
    audio = Path(row["audio_filepath"]).expanduser()
    if not audio.is_absolute() or not audio.is_file() or audio.suffix.lower() != ".wav":
        msg = "Use existing absolute WAV paths; prepare mono 16-kHz whole clips first"
        raise ValueError(msg)
    if row.get("offset", 0) != 0:
        msg = "This tutorial takes whole clips: materialize nonzero-offset excerpts first"
        raise ValueError(msg)
    return {**row, "audio_filepath": str(audio.resolve())}


def read_rows(manifest: Path, limit: int | None = None) -> list[dict]:
    """Read whole mono WAV clips with supplied text and one known speaker per clip."""
    if limit is not None and limit < _MIN_SPEAKERS:
        msg = "Select at least two clips from different speakers"
        raise ValueError(msg)
    rows = []
    seen = set()
    with manifest.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = _source_row(json.loads(line))
            canonical = row["audio_filepath"]
            if canonical in seen:
                msg = "An audio clip may occur only once in the source manifest"
                raise ValueError(msg)
            seen.add(canonical)
            rows.append({**row, "audio_filepath": canonical})
            if limit is not None and len(rows) >= limit:
                break
    if len({row["speaker"] for row in rows}) < _MIN_SPEAKERS:
        msg = "FastMSS needs prepared speech from at least two distinct speakers"
        raise ValueError(msg)
    return rows


def _fresh_directory(path: Path) -> Path:
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def align(args: argparse.Namespace) -> Path:
    """Run Curator's existing MFA stage; never treat its fallback intervals as alignment."""
    from nemo_curator.backends.base import BaseStageAdapter
    from nemo_curator.stages.audio.alignment import MFAAlignmentStage
    from nemo_curator.stages.audio.common import ManifestWriterStage
    from nemo_curator.tasks import AudioTask

    rows = read_rows(args.input_manifest, args.limit)
    if args.batch_size < 1 or args.num_jobs < 1:
        msg = "batch_size and num_jobs must be positive"
        raise ValueError(msg)
    if not args.dictionary.is_file() or not args.acoustic_model.is_file():
        msg = "Pass existing local English dictionary and acoustic model files"
        raise FileNotFoundError(msg)
    output = _fresh_directory(args.output_dir)
    stage = MFAAlignmentStage(
        output_dir=str(output / "alignment"),
        mfa_command=args.mfa_command,
        dictionary=str(args.dictionary.resolve()),
        acoustic_model=str(args.acoustic_model.resolve()),
        g2p_model="",
        mfa_root_dir=str(output / "mfa_work"),
        copy_models_to_local=False,
        num_jobs=args.num_jobs,
        align_timeout_seconds=args.timeout,
        create_ctm=True,
        create_rttm=True,
    )
    aligned = []
    stage.setup_on_node()
    stage.setup()
    try:
        adapter = BaseStageAdapter(stage)
        for start in range(0, len(rows), args.batch_size):
            tasks = [
                AudioTask(dataset_name="english_source", data=row) for row in rows[start : start + args.batch_size]
            ]
            aligned.extend(adapter.process_batch(tasks))
    finally:
        stage.teardown()
    failures = [row.data["audio_filepath"] for row in aligned if row.data.get("mfa_skipped", True)]
    if failures:
        (output / "failed_clips.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
        msg = "MFA skipped clips; inspect failed_clips.json. No aligned manifest was accepted"
        raise RuntimeError(msg)
    manifest = output / "aligned.jsonl"
    writer = ManifestWriterStage(output_path=str(manifest))
    writer.setup()
    try:
        writer.process_batch(aligned)
    finally:
        writer.teardown()
    return manifest


def read_word_alignment(path: Path, duration: float) -> list:
    """Convert the existing stage's word CTM, in clip-relative seconds, to Lhotse items."""
    from lhotse.supervision import AlignmentItem

    items = []
    recording_ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != _CTM_FIELDS:
            msg = "Expected word CTM: recording_id channel start duration word"
            raise ValueError(msg)
        recording_id, channel, start, length, word = fields
        start, length = float(start), float(length)
        if (
            channel != "1"
            or not math.isfinite(start + length)
            or start < 0
            or length <= 0
            or start + length > duration + 0.002
        ):
            msg = "Word alignment lies outside its mono recording or has invalid timing"
            raise ValueError(msg)
        if items and start < items[-1].start + items[-1].duration - 0.002:
            msg = "Word intervals must be ordered and non-overlapping within a single-speaker clip"
            raise ValueError(msg)
        recording_ids.add(recording_id)
        items.append(AlignmentItem(symbol=word, start=start, duration=length))
    if not items or len(recording_ids) != 1:
        msg = "Each clip requires a nonempty CTM for exactly one recording"
        raise ValueError(msg)
    return items


def prepare(args: argparse.Namespace) -> Path:
    """Create genuine aligned MonoCuts with read-only input copies below one dependency root."""
    import yaml
    from lhotse import CutSet, MonoCut, Recording
    from lhotse.supervision import SupervisionSegment

    if (
        isinstance(args.meetings, bool)
        or not 1 <= args.meetings <= _MAX_MEETINGS
        or not math.isfinite(args.duration)
        or not 0 < args.duration <= _MAX_DURATION_TARGET
    ):
        msg = "This small example requires 1..32 meetings and a finite duration target in (0, 300]"
        raise ValueError(msg)
    rows = read_rows(args.aligned_manifest)
    prepared = []
    for row in rows:
        if row.get("mfa_skipped", True) or not row.get("ctm_filepath"):
            msg = "Only successful MFA output is accepted; duration-based fallback CTM is not alignment"
            raise ValueError(msg)
        recording = Recording.from_file(row["audio_filepath"])
        if (
            recording.sampling_rate != _SAMPLE_RATE
            or recording.channel_ids != [0]
            or not 0 < recording.duration <= _MAX_CLIP_DURATION
        ):
            msg = "This small tutorial requires mono 16-kHz clips no longer than 20 seconds"
            raise ValueError(msg)
        items = read_word_alignment(Path(row["ctm_filepath"]), recording.duration)
        identity = hashlib.sha256(
            json.dumps([row["audio_filepath"], row["speaker"], row["text"]]).encode()
        ).hexdigest()[:20]
        prepared.append((row, items, identity))
    output = _fresh_directory(args.output_dir)
    (output / "audio").mkdir()
    cuts = []
    for row, items, identity in prepared:
        destination = output / "audio" / f"{identity}.wav"
        shutil.copyfile(row["audio_filepath"], destination)
        recording = Recording.from_file(destination, recording_id=identity)
        supervision = SupervisionSegment(
            id=identity,
            recording_id=recording.id,
            start=0,
            duration=recording.duration,
            channel=0,
            text=row["text"],
            speaker=row["speaker"],
            language="English",
            alignment={"word": items},
        )
        cuts.append(
            MonoCut(
                id=identity,
                start=0,
                duration=recording.duration,
                channel=0,
                recording=recording,
                supervisions=[supervision],
            )
        )
    CutSet.from_cuts(cuts).to_file(output / "speech_train.jsonl.gz")
    (output / "provenance.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    counts = dict(Counter(row["speaker"] for row in rows))
    (output / "summary.json").write_text(
        json.dumps({"clips": len(rows), "speakers": counts}, indent=2), encoding="utf-8"
    )
    config = {
        "defaults": ["default", "_self_"],
        "hydra": {"searchpath": ["file://${oc.env:FASTMSS_ROOT}/recipes"]},
        "manifest_dir": str(output),
        "manifest_prefix": "speech",
        "dset_splits": ["train"],
        "n_meetings": args.meetings,
        "duration": args.duration,
        "min_max_spk": [2, 2],
        "max_concurrent_speakers": 2,
        "min_spk_utt": 1,
        "split_fa_factor": None,
        "add_noise": False,
        "reverberate": False,
        "seed": 42,
    }
    config_path = output / "meeting.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def simulate(args: argparse.Namespace) -> Path:
    """Use the existing manifest adapter and consumers, with one-request smoke first."""
    from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor
    from nemo_curator.stages.audio.audio_agent import smoke, validate
    from nemo_curator.stages.audio.audio_agent.recipe import Recipe, StageRef, build_stages
    from nemo_curator.stages.audio.simulation import write_fastmss_requests

    native_inputs = args.prepared_inputs.expanduser().resolve()
    package = distribution("fastmss")
    if not (native_inputs / "meeting.yaml").is_file():
        msg = "Run the prepare subcommand first"
        raise FileNotFoundError(msg)
    output = _fresh_directory(args.output_dir)
    roots = {
        "native_inputs_path": str(native_inputs),
        "engine_path": str(package.locate_file("fastmss")),
        "recipe_path": str(package.locate_file("recipes")),
        "worker_lock_path": str(output / "worker-lock.json"),
    }
    requests = output / "requests.jsonl"
    count = write_fastmss_requests([str(native_inputs / "meeting.yaml")], str(requests), **roots)
    destination = output / "meetings.jsonl"
    recipe = Recipe(
        stages=[
            StageRef("FastMSSRequests", {"manifest_path": str(requests), **roots}),
            StageRef("FastMSSRequestStage", {"output_dir": str(output / "generated"), **roots}),
            StageRef("GetAudioDurationStage", {}),
            StageRef(
                "PreserveByValueStage",
                {"input_value_key": "duration", "target_value": args.min_duration, "operator": "ge"},
            ),
            StageRef("ManifestWriterStage", {"output_path": str(destination)}),
        ]
    )
    import yaml

    recipe.freeze()
    (output / "recipe.yaml").write_text(yaml.safe_dump(recipe.to_dict(), sort_keys=False), encoding="utf-8")
    verdict = validate(recipe)
    (output / "validation.json").write_text(json.dumps(verdict, default=str, indent=2), encoding="utf-8")
    if not verdict["runnable"]:
        msg = "Recipe validation failed; inspect validation.json"
        raise RuntimeError(msg)
    result = smoke(recipe, sample=1, executor=RayActorPoolExecutor(show_progress=False))
    (output / "smoke.json").write_text(json.dumps(result, default=str, indent=2), encoding="utf-8")
    if result["status"] != "completed" or result["errors"]:
        msg = "Bounded smoke failed; inspect smoke.json"
        raise RuntimeError(msg)
    if not args.execute:
        print(
            json.dumps(
                {
                    "planned_meetings": count,
                    "smoke": result["status"],
                    "next": "Choose a new output directory and add --execute to generate the full small example.",
                }
            )
        )
        return output / "smoke.json"
    from nemo_curator.pipeline import Pipeline

    stages, errors = build_stages(recipe)
    if errors:
        raise RuntimeError(str(errors))
    rows = Pipeline(name="english_mfa_fastmss", stages=stages).run(RayActorPoolExecutor(show_progress=False))
    (output / "result.json").write_text(
        json.dumps({"requested": count, "retained": len(rows), "manifest": str(destination)}, indent=2),
        encoding="utf-8",
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    align_parser = commands.add_parser("align", help="audio + supplied text -> existing Curator MFA stage")
    align_parser.add_argument("--input-manifest", type=Path, required=True)
    align_parser.add_argument("--output-dir", type=Path, required=True)
    align_parser.add_argument("--dictionary", type=Path, required=True)
    align_parser.add_argument("--acoustic-model", type=Path, required=True)
    align_parser.add_argument("--mfa-command", default="mfa")
    align_parser.add_argument("--num-jobs", type=int, default=1)
    align_parser.add_argument("--batch-size", type=int, default=32)
    align_parser.add_argument("--limit", type=int, default=32)
    align_parser.add_argument("--timeout", type=float, default=3600)
    align_parser.set_defaults(function=align)
    prepare_parser = commands.add_parser("prepare", help="successful word CTM -> native aligned Lhotse CutSet")
    prepare_parser.add_argument("--aligned-manifest", type=Path, required=True)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument("--meetings", type=int, default=4)
    prepare_parser.add_argument("--duration", type=float, default=10)
    prepare_parser.set_defaults(function=prepare)
    simulate_parser = commands.add_parser(
        "simulate", help="request manifest -> native FastMSS -> standard meeting JSONL"
    )
    simulate_parser.add_argument("--prepared-inputs", type=Path, required=True)
    simulate_parser.add_argument("--output-dir", type=Path, required=True)
    simulate_parser.add_argument("--min-duration", type=float, default=1)
    simulate_parser.add_argument("--execute", action="store_true")
    simulate_parser.set_defaults(function=simulate)
    args = parser.parse_args()
    print(args.function(args))


if __name__ == "__main__":
    main()
