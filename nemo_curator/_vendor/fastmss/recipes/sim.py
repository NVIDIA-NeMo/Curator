import glob
import io
import json
import logging
import os
import random
import re
import sys
from collections import defaultdict
from functools import partial
import multiprocessing as mp
from pathlib import Path

import hydra
import lhotse
import numpy as np
import soundfile as sf
from lhotse import CutSet, Recording, RecordingSet, SupervisionSet
from lhotse.manipulation import combine as combine_manifests
from lhotse.parallel import parallel_map
from lhotse.supervision import SupervisionSegment
from omegaconf import DictConfig
from tqdm import tqdm

from fastmss.rirsimulator import RIRSimulator
from fastmss.simulator import ConversationalMeetingSimulator
from fastmss.utils import split_monocuts_batch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_simulator: ConversationalMeetingSimulator = None

# Set in main() before the worker Pool is forked. Worker processes inherit
# this read-only via copy-on-write so we don't have to pickle a large dict
# through Pool initargs (which was the cause of the hang on many CPUs).
# REQUIRES the 'fork' start method — see the guard in stage 4.
_global_spk2cuts = None

def _worker_init(cfg, output_dir, rirs, noise_files, base_seed):
    worker_seed = base_seed + os.getpid()
    np.random.seed(worker_seed % (2**32))
    random.seed(worker_seed)

    global _simulator
    _simulator = ConversationalMeetingSimulator(
        cfg,
        Path(output_dir).absolute() / Path("audio"),
        _global_spk2cuts,
        rirs=rirs,
        noise_files=noise_files,
    )

def _build_spk2cuts(cfg, all_cuts):
    spk2cuts = {}
    n_total = 0
    n_dropped_duration = 0
    n_dropped_alignment = 0
    for cut in all_cuts:
        n_total += 1
        if (
            cut.duration > cfg.max_utt_duration
            or cut.duration < cfg.min_utt_duration
        ):
            n_dropped_duration += 1
            continue
        if not (
            hasattr(cut.supervisions[0], "alignment")
            and cut.supervisions[0].alignment
            and "word" in cut.supervisions[0].alignment
            and len(cut.supervisions[0].alignment["word"]) > 0
        ):
            n_dropped_alignment += 1
            continue
        c_spk = cut.supervisions[0].speaker
        if len({x.speaker for x in cut.supervisions}) != 1:
            logger.warning(
                "Skipping cut %s: expected exactly one speaker per cut, got %d.",
                cut.id, len({x.speaker for x in cut.supervisions}),
            )
            continue
        if c_spk not in spk2cuts:
            spk2cuts[c_spk] = []
        spk2cuts[c_spk].append(cut)

    n_before_min_utt = sum(len(v) for v in spk2cuts.values())
    n_spk_before = len(spk2cuts)
    for spk in list(spk2cuts.keys()):
        if len(spk2cuts[spk]) < cfg.min_spk_utt:
            del spk2cuts[spk]
    logger.info(
        f"_build_spk2cuts: {n_total} input cuts -> "
        f"dropped {n_dropped_duration} by duration, "
        f"{n_dropped_alignment} for missing alignment; "
        f"{n_spk_before} speakers -> {len(spk2cuts)} after min_spk_utt filter."
    )
    return spk2cuts

def _worker_gen_audio(uuid: str):
    """Called per task. Uses the process-local simulator."""
    return _simulator.gen_audio(uuid)


def merge_rttm_entries(entries, gap_threshold=0.2):
    """Merge same-speaker entries separated by gaps <= gap_threshold."""
    by_speaker = defaultdict(list)
    for start, duration, speaker, channel in entries:
        by_speaker[(speaker, channel)].append((start, start + duration))

    merged = []
    for (speaker, channel), segs in by_speaker.items():
        segs.sort()
        cur_start, cur_end = segs[0]
        for start, end in segs[1:]:
            if start - cur_end <= gap_threshold:
                cur_end = max(cur_end, end)
            else:
                merged.append((cur_start, cur_end - cur_start, speaker, channel))
                cur_start, cur_end = start, end
        merged.append((cur_start, cur_end - cur_start, speaker, channel))

    merged.sort(key=lambda x: x[0])
    return merged


def discard(x, duration):
    info = sf.SoundFile(x)
    if (len(info) / info.samplerate) > duration:
        return x
    else:
        return None


def nemo_manifest_to_cutset(nemo_manifest_path):
    """Convert a NeMo diarization manifest (JSONL with rttm_filepath) to a Lhotse CutSet."""
    recs = []
    sups = []
    seg_idx = 0

    with open(nemo_manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            audio_filepath = entry["audio_filepath"]
            recording_id = Path(audio_filepath).stem

            rec = Recording.from_file(audio_filepath, recording_id=recording_id)
            recs.append(rec)

            rttm_path = entry.get("rttm_filepath")
            if not rttm_path or not Path(rttm_path).exists():
                continue

            with open(rttm_path) as rf:
                for rttm_line in rf:
                    rttm_line = rttm_line.strip()
                    if not rttm_line or rttm_line.startswith(";"):
                        continue
                    parts = rttm_line.split()
                    if parts[0] != "SPEAKER":
                        continue
                    channel = int(parts[2])
                    start = float(parts[3])
                    duration = float(parts[4])
                    speaker = parts[7]

                    sups.append(
                        SupervisionSegment(
                            id=f"{recording_id}_{seg_idx:06d}",
                            recording_id=recording_id,
                            start=round(start, 6),
                            duration=round(duration, 6),
                            channel=channel,
                            speaker=speaker,
                        )
                    )
                    seg_idx += 1

    return CutSet.from_manifests(
        recordings=RecordingSet.from_recordings(recs),
        supervisions=SupervisionSet.from_segments(sups),
    )


@hydra.main(version_base=None, config_path=".", config_name="default")
def main(cfg: DictConfig) -> None:

    if hasattr(cfg, "seed") and cfg.seed is not None:
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

    data_dir = cfg.data_dir if cfg.get("data_dir") else cfg.output_dir
    rir_dir = cfg.rir_dir if cfg.get("rir_dir") else cfg.output_dir

    def _mark_done(path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def _is_done(path):
        return Path(path).exists()

    logger.info(f"data_dir={data_dir}  rir_dir={rir_dir}  output_dir={cfg.output_dir}")

    # ------------------------------------------------------------------ #
    # Stage 1: Load source CutSet  (-> data_dir)
    # ------------------------------------------------------------------ #
    if cfg.stage <= 1 and _is_done(Path(data_dir) / "manifests" / ".done_stage1"):
        logger.info(
            "Stage 1 already done (found .done_stage1 in data_dir/manifests), skipping."
        )
    elif cfg.stage <= 1:
        Path(data_dir).mkdir(exist_ok=True, parents=True)
        all_cuts = []
        for split in cfg.dset_splits:
            c_cut = CutSet.from_file(
                Path(cfg.manifest_dir) / f"{cfg.manifest_prefix}_{split}.jsonl.gz"
            )
            all_cuts.append(c_cut)
        all_cuts = combine_manifests(all_cuts)

        all_cuts = all_cuts.to_eager()
        sample = all_cuts.subset(first=min(200, len(all_cuts)))
        n_with_align = sum(
            1
            for cut in sample
            for sup in cut.supervisions
            if sup.alignment and "word" in sup.alignment
        )
        if n_with_align == 0:
            logger.warning(
                "=" * 72 + "\n"
                "  NO WORD-LEVEL ALIGNMENTS FOUND in source CutSet.\n"
                "  FastMSS relies on word alignments to split utterances at natural\n"
                "  pauses (split_fa_factor) and to place speaker turns precisely.\n"
                "  Without alignments, simulated conversations will have less\n"
                "  realistic timing and overlap -- separation/diarization/TS-ASR\n"
                "  performance trained on this data will be noticeably worse.\n"
                "  For LibriSpeech: run `lhotse prepare librispeech --alignments-dir\n"
                "  <CorentinJ-style dir with *.alignment.txt files>` (see\n"
                "  https://github.com/CorentinJ/librispeech-alignments).\n" + "=" * 72
            )

        exclude = set(str(s) for s in (cfg.get("exclude_speakers") or []))
        if exclude:
            all_cuts = all_cuts.to_eager()
            before = len(all_cuts)
            all_cuts = all_cuts.filter(
                lambda cut: all(str(s.speaker) not in exclude for s in cut.supervisions)
            ).to_eager()
            logger.info(
                f"Excluded speakers {exclude}: {before} -> {len(all_cuts)} cuts"
            )

        if (
            hasattr(cfg, "split_fa_factor")
            and cfg.split_fa_factor is not None
            and cfg.split_fa_factor > 0
        ):
            all_cuts = all_cuts.to_eager()
            before = len(all_cuts)
            all_cuts = split_monocuts_batch(
                all_cuts,
                cfg.split_fa_factor,
                num_jobs=cfg.n_jobs,
            )
            logger.info(
                f"Split cuts at pauses > {cfg.split_fa_factor}s: "
                f"{before} -> {len(all_cuts)} cuts"
            )

        logger.info("Saving source CutSet to disk")
        (Path(data_dir) / "manifests").mkdir(exist_ok=True, parents=True)
        all_cuts.to_file(os.path.join(data_dir, "manifests", "all_cuts.jsonl.gz"))
        _mark_done(Path(data_dir) / "manifests" / ".done_stage1")

    # ------------------------------------------------------------------ #
    # Stage 2: Noise preparation
    # ------------------------------------------------------------------ #
    if cfg.stage <= 2 and cfg.add_noise:
        if _is_done(Path(cfg.output_dir) / "manifests" / ".done_stage2"):
            logger.info("Stage 2 already done, skipping.")
        elif cfg.noise_folders is not None:
            logger.info("Parsing background noise files.")
            noise_files = []
            for c_folder in cfg.noise_folders:
                for c_ext in [".wav", ".flac", ".mp3"]:
                    tmp = glob.glob(
                        os.path.join(c_folder, "**/*" + c_ext), recursive=True
                    )
                    noise_files.extend(tmp)

            # Apply regex filter if specified
            if (
                hasattr(cfg, "noise_filename_pattern")
                and cfg.noise_filename_pattern is not None
            ):
                pattern = re.compile(cfg.noise_filename_pattern)
                original_count = len(noise_files)
                noise_files = [
                    f for f in noise_files if pattern.search(os.path.basename(f))
                ]
                filtered_count = original_count - len(noise_files)
                logger.info(
                    f"Applied filename pattern '{cfg.noise_filename_pattern}': "
                    f"kept {len(noise_files)} files, filtered out {filtered_count} files."
                )

            worker = partial(discard, duration=cfg.filter_noise_len)
            # filter noise that are too short
            filtered = []
            for n in tqdm(
                parallel_map(worker, noise_files, num_jobs=cfg.n_jobs),
                total=len(noise_files),
                desc="Parsing noise files.",
            ):
                filtered.append(n)

            filtered = [x for x in filtered if x is not None]
            diff = len(noise_files) - len(filtered)
            logger.info(
                f"Discarded {diff} noise files as they were shorter than {cfg.filter_noise_len}. Now {len(filtered)}, before {len(noise_files)}."
            )
            noise_files = filtered

            assert len(noise_files) > 0, "No noise files found, wrong path?"
            Path(cfg.output_dir, "manifests").mkdir(parents=True, exist_ok=True)
            out_file = os.path.join(cfg.output_dir, "manifests", "noise_files.txt")
            with open(out_file, "w") as f:
                f.writelines([str(x) + "\n" for x in noise_files])
            logger.info(f"Noise files paths saved in {out_file}")
            _mark_done(Path(cfg.output_dir) / "manifests" / ".done_stage2")

    # ------------------------------------------------------------------ #
    # Stage 3: RIR simulation  (-> rir_dir)
    # ------------------------------------------------------------------ #
    if cfg.stage <= 3 and cfg.reverberate:
        if _is_done(Path(rir_dir) / ".done_stage3"):
            logger.info("Stage 3 already done (found .done_stage3 in rir_dir), skipping.")
        else:
            logger.info("Simulating RIRs using Pyroomacoustics")

            from copy import deepcopy

            from omegaconf import OmegaConf

            rir_cfg = deepcopy(cfg)
            OmegaConf.update(rir_cfg, "output_dir", rir_dir)
            simulator = RIRSimulator(rir_cfg)
            worker = partial(simulator.gen_rirs)

            meeting_ids = iter([f"room_{x}" for x in range(cfg.n_rirs)])
            all_rooms = []
            for c_rirs in tqdm(
                parallel_map(worker, meeting_ids, num_jobs=cfg.n_jobs),
                total=cfg.n_rirs,
                desc="Simulating room impulse responses (RIRs)",
            ):
                all_rooms.append(c_rirs)

            Path(rir_dir).mkdir(parents=True, exist_ok=True)
            out_file = os.path.join(rir_dir, "all_rooms.json")
            with open(out_file, "w") as f:
                json.dump(all_rooms, f, indent=4)
            _mark_done(Path(rir_dir) / ".done_stage3")

    # ------------------------------------------------------------------ #
    # Stage 4: Meeting simulation  (-> output_dir)
    # ------------------------------------------------------------------ #
    if cfg.stage <= 4 and _is_done(Path(cfg.output_dir) / "manifests" / ".done_stage4"):
        logger.info("Stage 4 already done, skipping.")
    elif cfg.stage <= 4:
        try:
            all_cuts
        except NameError:
            logger.info("Loading source CutSet")
            all_cuts = lhotse.load_manifest(
                os.path.join(data_dir, "manifests", "all_cuts.jsonl.gz")
            )
            logger.info("Source CutSet loaded")

        if cfg.add_noise and cfg.noise_folders is not None:
            out_file = os.path.join(cfg.output_dir, "manifests", "noise_files.txt")
            with open(out_file, "r") as f:
                noise_files = f.readlines()
            noise_files = [x.strip("\n") for x in noise_files]
        else:
            noise_files = None

        if cfg.reverberate:
            out_file = os.path.join(rir_dir, "all_rooms.json")
            with open(out_file, "r") as f:
                rirs = json.load(f)
        else:
            rirs = None

        output_dir = Path(cfg.output_dir).absolute() / Path("manifests")
        output_dir.mkdir(exist_ok=True, parents=True)

        uuids = [f"simulation_{x}" for x in range(cfg.n_meetings)]

        base_seed = cfg.seed if (hasattr(cfg, "seed") and cfg.seed is not None) else 42

        logger.info("Building spk2cuts in main process...")
        spk2cuts = _build_spk2cuts(cfg, all_cuts)
        logger.info(f"Done: {len(spk2cuts)} speakers across {sum(len(v) for v in spk2cuts.values())} cuts.")

        if mp.get_start_method(allow_none=True) not in (None, "fork"):
            raise RuntimeError("Stage 4 requires the 'fork' multiprocessing start method so workers inherit _global_spk2cuts without pickling.")

        global _global_spk2cuts
        _global_spk2cuts = spk2cuts  # workers inherit this via fork, read-only is safe

        init_args = (
            cfg,
            str(cfg.output_dir),
            rirs,
            noise_files,
            base_seed,
        )

        recordings = []
        supervisions = []

        ctx = mp.get_context("fork")  # fork is required so workers inherit _global_spk2cuts
        with ctx.Pool(processes=cfg.n_jobs, initializer=_worker_init, initargs=init_args) as pool:
             for c_rec, c_sup in tqdm(pool.imap_unordered(_worker_gen_audio, uuids), total=cfg.n_meetings, desc="Simulating meetings"):
                 recordings.append(c_rec)
                 supervisions.extend(c_sup)
                 
        supervisions = SupervisionSet(supervisions)
        recordings = RecordingSet(recordings)
        lhotse.validate_recordings_and_supervisions(recordings=recordings, supervisions=supervisions)
        logger.info(f"Saving simulated manifests to {output_dir}.")
        supervisions.to_file(
            output_dir / f"synth-{cfg.manifest_prefix}-train-supervisions.jsonl.gz"
        )
        recordings.to_file(
            output_dir / f"synth-{cfg.manifest_prefix}-train-recordings.jsonl.gz"
        )

        if cfg.save_cutset:
            cutset = lhotse.CutSet.from_manifests(
                recordings=recordings, supervisions=supervisions
            )
            cutset.to_file(
                output_dir / f"synth-{cfg.manifest_prefix}-train-cuts.jsonl.gz"
            )

        _mark_done(Path(cfg.output_dir) / "manifests" / ".done_stage4")

    # ------------------------------------------------------------------ #
    # Stage 5: RTTM + NeMo manifest generation (optional)
    # ------------------------------------------------------------------ #
    save_rttm = cfg.get("save_rttm", False)
    save_nemo_manifest = cfg.get("save_nemo_manifest", False)

    if cfg.stage <= 5 and (save_rttm or save_nemo_manifest):
        if _is_done(Path(cfg.output_dir) / ".done_stage5"):
            logger.info("Stage 5 already done, skipping.")
        else:
            manifest_dir = Path(cfg.output_dir).absolute() / "manifests"

            try:
                recordings
                supervisions
            except NameError:
                logger.info("Loading simulated manifests from disk")
                recordings = lhotse.load_manifest(
                    manifest_dir
                    / f"synth-{cfg.manifest_prefix}-train-recordings.jsonl.gz"
                )
                supervisions = lhotse.load_manifest(
                    manifest_dir
                    / f"synth-{cfg.manifest_prefix}-train-supervisions.jsonl.gz"
                )

            cutset = CutSet.from_manifests(
                recordings=recordings, supervisions=supervisions
            )

            if save_rttm:
                # --- RTTM generation ---
                logger.info("Generating RTTM files from simulated manifests")
                rttm_out_dir = Path(cfg.output_dir).absolute() / "rttm"
                rttm_out_dir.mkdir(parents=True, exist_ok=True)

                rttm_entries = defaultdict(list)
                total_words = 0
                total_segments = 0

                for cut in cutset:
                    rec_id = cut.recording.id if cut.recording else cut.id
                    for sup in cut.supervisions:
                        total_segments += 1
                        speaker = (
                            str(sup.speaker) if sup.speaker is not None else "unknown"
                        )
                        channel = sup.channel

                        if sup.alignment and "word" in sup.alignment:
                            for item in sup.alignment["word"]:
                                if not item.symbol or item.symbol.strip() == "":
                                    continue
                                rttm_entries[rec_id].append(
                                    (item.start, item.duration, speaker, channel)
                                )
                                total_words += 1
                        else:
                            rttm_entries[rec_id].append(
                                (sup.start, sup.duration, speaker, channel)
                            )
                            total_words += 1

                total_merged = 0
                for rec_id, entries in rttm_entries.items():
                    entries = merge_rttm_entries(entries, gap_threshold=0.2)
                    total_merged += len(entries)
                    out_path = rttm_out_dir / f"{rec_id}.rttm"
                    with open(out_path, "w") as f:
                        for start, duration, speaker, channel in entries:
                            f.write(
                                f"SPEAKER {rec_id} {channel} "
                                f"{start:.4f} {duration:.4f} "
                                f"<NA> <NA> {speaker} <NA> <NA>\n"
                            )

                logger.info(
                    f"RTTM generation complete: {total_words} word-level entries from "
                    f"{total_segments} segments merged into {total_merged} entries "
                    f"(gap_threshold=0.2s) across {len(rttm_entries)} recording(s) "
                    f"saved to {rttm_out_dir}"
                )

            if save_nemo_manifest:
                # --- NeMo manifest ---
                logger.info("Generating NeMo-style diarization manifest")

                if not save_rttm:
                    logger.warning(
                        "save_nemo_manifest=true requires save_rttm=true. Generating RTTMs first."
                    )
                    # Ensure RTTMs exist
                    rttm_out_dir = Path(cfg.output_dir).absolute() / "rttm"
                    if not rttm_out_dir.exists():
                        raise RuntimeError(
                            "RTTM directory not found. Set save_rttm=true to generate RTTMs."
                        )
                else:
                    rttm_out_dir = Path(cfg.output_dir).absolute() / "rttm"

                rec2sups = defaultdict(list)
                for sup in supervisions:
                    rec2sups[sup.recording_id].append(sup)

                nemo_manifest_path = (
                    Path(cfg.output_dir).absolute() / "nemo_manifest.json"
                )
                n_written = 0
                with open(nemo_manifest_path, "w") as f_out:
                    for rec in recordings:
                        sups = rec2sups[rec.id]
                        if not sups:
                            logger.warning(
                                f"Recording {rec.id} has no supervisions, skipping"
                            )
                            continue

                        audio_path = rec.sources[0].source
                        num_speakers = len(set(s.speaker for s in sups))
                        rttm_path = rttm_out_dir / f"{rec.id}.rttm"

                        entry = {
                            "audio_filepath": audio_path,
                            "offset": 0.0,
                            "duration": round(rec.duration, 6),
                            "num_speakers": num_speakers,
                            "rttm_filepath": str(rttm_path.absolute()),
                        }
                        f_out.write(json.dumps(entry) + "\n")
                        n_written += 1

                logger.info(
                    f"NeMo manifest saved to {nemo_manifest_path} ({n_written} entries)"
                )

                # --- NeMo manifest statistics (round-trip verification) ---
                logger.info(
                    "Computing statistics from NeMo manifest (round-trip verification)"
                )
                nemo_cutset = nemo_manifest_to_cutset(str(nemo_manifest_path))
                buf = io.StringIO()
                old_stdout = sys.stdout
                sys.stdout = buf
                try:
                    nemo_cutset.describe(full=True)
                finally:
                    sys.stdout = old_stdout
                stats_text = buf.getvalue()
                logger.info(
                    "NeMo manifest statistics (from RTTM, post-merge):\n%s", stats_text
                )

                stats_path = Path(cfg.output_dir).absolute() / "overlap_stats.txt"
                stats_path.write_text(stats_text)
                logger.info(f"Overlap statistics saved to {stats_path}")

            # --- Dataset statistics ---
            logger.info("Computing dataset statistics")
            buf = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = buf
            try:
                cutset.describe(full=True)
            finally:
                sys.stdout = old_stdout
            logger.info("Dataset statistics:\n%s", buf.getvalue())

            _mark_done(Path(cfg.output_dir) / ".done_stage5")


if __name__ == "__main__":
    main()
