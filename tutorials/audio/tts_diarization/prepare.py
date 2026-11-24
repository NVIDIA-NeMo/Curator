# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare utterance jobs, aligned Lhotse inputs and paper simulation shards."""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import soundfile as sf
from omegaconf import OmegaConf

MIN_REFERENCE_SECONDS = 5
MIN_CTM_FIELDS = 5

if TYPE_CHECKING:
    from lhotse import MonoCut


def read_rows(path: Path) -> list[dict]:
    """Read JSONL and resolve audio/annotation paths relative to its directory."""
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("audio_filepath", "reference_audio", "ctm_filepath", "rttm_filepath"):
                if row.get(key):
                    p = Path(row[key]).expanduser()
                    row[key] = str((path.parent / p).resolve())
            rows.append(row)
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    """Write a new manifest without overwriting a previous experiment."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def reference_pools(
    references: Path,
    strategy: str,
    languages: list[str] | None,
    include_test_references: bool,
) -> dict[str, list[dict]]:
    """Select and validate the reference catalog before assigning any jobs."""
    pools = defaultdict(list)
    for ref in read_rows(references):
        if any(not isinstance(ref.get(key), str) or not ref[key].strip() for key in ("speaker", "reference_audio")):
            msg = "Reference rows require speaker and reference_audio"
            raise ValueError(msg)
        if strategy == "target-domain" and ref.get("split") not in {"train", "dev", "test"}:
            msg = "Target-domain references require an explicit train/dev/test split"
            raise ValueError(msg)
        if ref.get("split") == "test" and not include_test_references:
            continue
        if languages and ref.get("language") not in languages:
            continue
        if sf.info(ref["reference_audio"]).duration < MIN_REFERENCE_SECONDS:
            msg = f"Reference shorter than 5 seconds: {ref['reference_audio']}"
            raise ValueError(msg)
        pools[ref["speaker"]].append(ref)
    if not pools:
        msg = "No eligible reference speakers"
        raise ValueError(msg)
    return pools


def read_texts(texts: Path) -> list[dict]:
    """Read a nonempty text pool with stable, unique utterance identities."""
    rows = read_rows(texts)
    if not rows or any(
        not isinstance(row.get(key), str) or not row[key].strip() for row in rows for key in ("text", "utterance_id")
    ):
        msg = "Texts require unique utterance_id values and nonempty text"
        raise ValueError(msg)
    if len({row["utterance_id"] for row in rows}) != len(rows):
        msg = "Duplicate source utterance_id"
        raise ValueError(msg)
    return rows


def prepare_jobs(  # noqa: PLR0913
    texts: Path,
    references: Path,
    output: Path,
    strategy: str,
    seed: int = 42,
    utterances_per_speaker: int | None = None,
    languages: list[str] | None = None,
    include_test_references: bool = False,
) -> int:
    """Pair text and voice independently, keeping split/channel provenance."""
    if strategy not in {"direct", "cross-lingual", "target-domain"}:
        msg = f"Unknown generation strategy: {strategy}"
        raise ValueError(msg)
    rows = read_texts(texts)
    pools = reference_pools(references, strategy, languages, include_test_references)
    rng = random.Random(seed)  # noqa: S311
    pairs = []
    if strategy == "direct":
        for row in rows:
            if row.get("speaker") not in pools:
                msg = f"No same-speaker reference for {row.get('speaker')}"
                raise ValueError(msg)
            pairs.append((row, rng.choice(pools[row["speaker"]])))
    else:
        count = (
            utterances_per_speaker
            if utterances_per_speaker is not None
            else (125 if strategy == "cross-lingual" else 500)
        )
        if not 0 < count <= len(rows):
            msg = "utterances_per_speaker must be positive and no larger than the text pool"
            raise ValueError(msg)
        for speaker in sorted(pools):
            for row in rng.sample(rows, count):
                pairs.append((row, rng.choice(pools[speaker])))
    jobs = []
    for index, (row, ref) in enumerate(pairs):
        identity = json.dumps([strategy, seed, index, row["utterance_id"], ref["speaker"]])
        job_id = "tts_" + hashlib.sha256(identity.encode()).hexdigest()[:24]
        jobs.append(
            {
                "utterance_id": job_id,
                "source_utterance_id": row["utterance_id"],
                "text": row["text"],
                "source_audio_filepath": row.get("audio_filepath", ""),
                "source_duration": row.get("duration"),
                "source_speaker": row.get("speaker"),
                "speaker": ref["speaker"],
                "reference_audio": ref["reference_audio"],
                "reference_split": ref.get("split"),
                "reference_channel": ref.get("channel"),
                "reference_language": ref.get("language"),
                "split": row.get("split", "train"),
                "chapter_id": row.get("chapter_id"),
                "strategy": strategy,
            }
        )
    write_rows(output, jobs)
    return len(jobs)


def source_cut(row: dict, kind: str) -> "MonoCut":
    """Convert one successful alignment into a checked, mono Lhotse cut."""
    from lhotse import MonoCut, Recording, SupervisionSegment
    from lhotse.supervision import AlignmentItem

    if row.get("mfa_skipped") or not row.get("ctm_filepath"):
        msg = "Every FastMSS source requires successful MFA and CTM word alignment"
        raise ValueError(msg)
    speaker = row.get("speaker")
    if not isinstance(speaker, str) or not speaker or any(c.isspace() for c in speaker):
        msg = "Each source requires a single-token speaker ID"
        raise ValueError(msg)
    key = hashlib.sha256(json.dumps([kind, row.get("utterance_id"), row["audio_filepath"]]).encode()).hexdigest()
    recording = Recording.from_file(row["audio_filepath"], recording_id=key)
    if recording.num_channels != 1:
        msg = "FastMSS source waveforms must be mono"
        raise ValueError(msg)
    words = []
    for line in Path(row["ctm_filepath"]).read_text().splitlines():
        fields = line.split()
        if len(fields) < MIN_CTM_FIELDS:
            msg = "Malformed CTM word alignment"
            raise ValueError(msg)
        start, duration = float(fields[2]), float(fields[3])
        if not 0 <= start < start + duration <= recording.duration + 0.001:
            msg = "CTM timing is outside its source waveform"
            raise ValueError(msg)
        if words and start < words[-1].end - 0.001:
            msg = "CTM words must be ordered and non-overlapping"
            raise ValueError(msg)
        words.append(AlignmentItem(fields[4], start, duration))
    if not words:
        msg = "Empty CTM word alignment"
        raise ValueError(msg)
    supervision = SupervisionSegment(
        id=key,
        recording_id=key,
        start=0,
        duration=recording.duration,
        channel=0,
        text=row["text"],
        speaker=speaker,
        alignment={"word": words},
    )
    return MonoCut(
        id=key,
        start=0,
        duration=recording.duration,
        channel=0,
        recording=recording,
        supervisions=[supervision],
        custom={"source_kind": kind},
    )


def export_cutset(synthetic: Path, output_dir: Path, real: Path | None = None) -> dict:
    """Export aligned sources with word timings, preserving speaker identities."""
    from lhotse import CutSet

    destination = output_dir / "paper_train.jsonl.gz"
    if destination.exists():
        msg = f"CutSet already exists: {destination}"
        raise FileExistsError(msg)
    cuts, counts = [], Counter()
    ids = set()
    for kind, manifest in (("tts", synthetic), ("real", real)):
        if manifest is None:
            continue
        for row in read_rows(manifest):
            cut = source_cut(row, kind)
            if cut.id in ids:
                msg = "Duplicate source utterance"
                raise ValueError(msg)
            ids.add(cut.id)
            cuts.append(cut)
            counts[kind] += 1
    if not cuts:
        msg = "No aligned sources"
        raise ValueError(msg)
    output_dir.mkdir(parents=True, exist_ok=True)
    CutSet.from_cuts(cuts).to_file(destination)
    summary = {
        "utterances": dict(counts),
        "speakers": len({c.supervisions[0].speaker for c in cuts}),
        "cutset": str(destination.resolve()),
    }
    (output_dir / "sources.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def prepare_recipes(  # noqa: PLR0913
    corpus_dir: Path,
    output_dir: Path,
    musan_root: Path | None,
    count_unit: int = 4000,
    duration: float = 90,
    seed: int = 42,
    augment: bool = True,
    min_utterances: int = 5,
) -> list[Path]:
    """Create 1/3/6/10 shards using the published diarization settings."""
    if count_unit <= 0 or duration <= 0 or min_utterances <= 0:
        msg = "Counts, duration and min_utterances must be positive"
        raise ValueError(msg)
    if augment and (musan_root is None or any(not (musan_root / part).is_dir() for part in ("noise", "music"))):
        msg = "Paper augmentation requires MUSAN noise/ and music/ directories"
        raise ValueError(msg)
    paths = [output_dir / f"{n}spk.yaml" for n in range(1, 5)]
    if any(path.exists() for path in paths):
        msg = "Simulation recipes already exist; choose a fresh output directory"
        raise FileExistsError(msg)
    output_dir.mkdir(parents=True, exist_ok=True)
    for n, ratio, path in zip(range(1, 5), (1, 3, 6, 10), paths, strict=True):
        config = {
            "defaults": ["default", "_self_"],
            "hydra": {"searchpath": ["file://${oc.env:FASTMSS_ROOT}/recipes"]},
            "manifest_dir": str(corpus_dir.resolve()),
            "manifest_prefix": "paper",
            "dset_splits": ["train"],
            "seed": seed + n,
            "duration": duration,
            "n_meetings": count_unit * ratio,
            "min_max_spk": [n, n],
            "max_concurrent_speakers": n,
            "min_spk_utt": min_utterances,
            "split_fa_factor": 0.2,
            "hmm_params": {"p_ind": [1, 0, 0, 0] if n == 1 else [0.15, 0.21, 0.44, 0.20]},
            "use_markov": False,
            "boost_overlap_factor": None,
            "add_noise": augment,
            "noise_probability_global": 0.5,
            "noise_folders": [str((musan_root / part).resolve()) for part in ("noise", "music")] if augment else None,
            "reverberate": augment,
            "reverb_prob": 0.5,
            "save_anechoic": False,
            "save_rttm": True,
            "save_nemo_manifest": True,
        }
        OmegaConf.save(OmegaConf.create(config), path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    jobs = sub.add_parser("jobs")
    for flag in ("texts", "references", "output"):
        jobs.add_argument(f"--{flag}", type=Path, required=True)
    jobs.add_argument("--strategy", choices=["direct", "cross-lingual", "target-domain"], required=True)
    jobs.add_argument("--seed", type=int, default=42)
    jobs.add_argument("--utterances-per-speaker", type=int)
    jobs.add_argument("--languages", nargs="+")
    jobs.add_argument("--include-test-references", action="store_true")
    export = sub.add_parser("export")
    export.add_argument("--synthetic", type=Path, required=True)
    export.add_argument("--real", type=Path)
    export.add_argument("--output-dir", type=Path, required=True)
    recipes = sub.add_parser("recipes")
    recipes.add_argument("--corpus-dir", type=Path, required=True)
    recipes.add_argument("--output-dir", type=Path, required=True)
    recipes.add_argument("--musan-root", type=Path)
    recipes.add_argument("--count-unit", type=int, default=4000)
    recipes.add_argument("--duration", type=float, default=90)
    recipes.add_argument("--seed", type=int, default=42)
    recipes.add_argument("--no-augmentation", action="store_true")
    recipes.add_argument("--min-utterances", type=int, default=5)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "jobs":
        print(prepare_jobs(**args), "jobs")
    elif command == "export":
        print(json.dumps(export_cutset(**args), indent=2))
    else:
        args["augment"] = not args.pop("no_augmentation")
        print("\n".join(map(str, prepare_recipes(**args))))


if __name__ == "__main__":
    main()
