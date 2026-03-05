# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Speaker diarization on CallHome English using Streaming Sortformer via NeMo Curator.

Runs InferenceSortformerStage through a Pipeline + RayActorPoolExecutor for
parallel GPU inference, then evaluates Diarization Error Rate (DER) against
the CHA ground-truth annotations.

Usage:
    python tutorials/audio/callhome_diar/run.py --data-dir /path/to/callhome_eng0
    python tutorials/audio/callhome_diar/run.py --data-dir /path/to/callhome_eng0 --clean
"""

import argparse
import json
import re
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

import ray
import torch

from nemo_curator.backends.experimental.ray_actor_pool import RayActorPoolExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.inference.sortformer import InferenceSortformerStage
from nemo_curator.tasks import AudioBatch

COLLAR = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Streaming Sortformer diarization on CallHome English and evaluate DER.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to CallHome-eng0 dataset root (contains *.wav files and eng/ subdir with .cha files).",
    )
    parser.add_argument(
        "--rttm-out-dir",
        type=Path,
        default=Path("rttm_callhome_sortformer"),
        help="Directory to write RTTM output files (default: rttm_callhome_sortformer).",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=Path("callhome_sortformer_results.json"),
        help="Path for detailed per-file DER results JSON (default: callhome_sortformer_results.json).",
    )
    parser.add_argument(
        "--model",
        default="nvidia/diar_streaming_sortformer_4spk-v2",
        help="Hugging Face Sortformer model id.",
    )
    parser.add_argument(
        "--collar",
        type=float,
        default=COLLAR,
        help=f"Collar tolerance in seconds for DER scoring (default: {COLLAR}).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing RTTM output directory before running inference.",
    )
    parser.add_argument(
        "--chunk-len",
        type=int,
        default=340,
        help="Streaming chunk size in 80ms frames (default: 340).",
    )
    parser.add_argument(
        "--chunk-right-context",
        type=int,
        default=40,
        help="Right context frames (default: 40).",
    )
    parser.add_argument(
        "--fifo-len",
        type=int,
        default=40,
        help="FIFO queue size in frames (default: 40).",
    )
    parser.add_argument(
        "--spkcache-update-period",
        type=int,
        default=300,
        help="Speaker cache update period in frames (default: 300).",
    )
    parser.add_argument(
        "--spkcache-len",
        type=int,
        default=188,
        help="Speaker cache size in frames (default: 188).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Audio pre-processing
# ---------------------------------------------------------------------------


def ensure_mono(wav_path: Path, mono_dir: Path) -> Path:
    """Return a mono 16 kHz WAV, downmixing stereo via sox if needed."""
    mono_path = mono_dir / wav_path.name
    if mono_path.exists():
        return mono_path
    mono_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(  # noqa: S603
        ["sox", str(wav_path), "-c", "1", "-r", "16000", str(mono_path)],  # noqa: S607
        check=True,
        capture_output=True,
    )
    return mono_path


# ---------------------------------------------------------------------------
# CHA ground-truth parsing
# ---------------------------------------------------------------------------


def parse_cha(path: Path) -> tuple[list[dict], float, float]:
    """Parse a CHA file into segments and derive a UEM scoring region.

    Returns:
        (segments, uem_start, uem_end) where segments is a list of
        {"speaker", "start", "end"} dicts and the UEM region spans
        the earliest to latest annotated timestamp.
    """
    segs: list[dict] = []
    with open(path) as f:
        for line in f:
            m = re.match(r"^\*([A-Z]):\t", line)
            ts = re.search(r"\x15(\d+)_(\d+)\x15", line)
            if m and ts:
                segs.append(
                    {
                        "speaker": m.group(1),
                        "start": int(ts.group(1)) / 1000,
                        "end": int(ts.group(2)) / 1000,
                    }
                )
    if not segs:
        return segs, 0.0, 0.0
    uem_start = min(s["start"] for s in segs)
    uem_end = max(s["end"] for s in segs)
    return segs, uem_start, uem_end


# ---------------------------------------------------------------------------
# DER evaluation
# ---------------------------------------------------------------------------


def compute_der(  # noqa: C901, PLR0912
    gt: list[dict],
    pred: list[dict],
    uem_start: float,
    uem_end: float,
    collar: float = 0.0,
) -> dict | None:
    """Frame-level DER restricted to UEM region with collar tolerance."""
    if not gt or not pred:
        return None

    pred = [
        {"speaker": s["speaker"], "start": max(s["start"], uem_start), "end": min(s["end"], uem_end)}
        for s in pred
        if s["end"] > uem_start and s["start"] < uem_end
    ]
    pred = [s for s in pred if s["end"] > s["start"]]
    if not pred:
        return None

    collar_zones: list[tuple[float, float]] = []
    if collar > 0:
        for s in gt:
            collar_zones.append((s["start"] - collar, s["start"] + collar))
            collar_zones.append((s["end"] - collar, s["end"] + collar))

    def in_collar(t: float) -> bool:
        return any(lo <= t <= hi for lo, hi in collar_zones)

    mv: Counter = Counter()
    for g in gt:
        for p in pred:
            ov = min(g["end"], p["end"]) - max(g["start"], p["start"])
            if ov > 0:
                mv[(g["speaker"], p["speaker"])] += ov
    used, mapping = set(), {}
    for _v, gs, ps in sorted([(v, g, p) for (g, p), v in mv.items()], reverse=True):
        if gs not in mapping and ps not in used:
            mapping[gs] = ps
            used.add(ps)
    inv = {v: k for k, v in mapping.items()}

    step = 0.01
    nf = int((uem_end - uem_start) / step) + 1

    gf: dict[int, set] = {}
    pf: dict[int, set] = {}
    for s in gt:
        for i in range(max(0, int((s["start"] - uem_start) / step)), min(nf, int((s["end"] - uem_start) / step))):
            gf.setdefault(i, set()).add(s["speaker"])
    for s in pred:
        mm = inv.get(s["speaker"], f"x_{s['speaker']}")
        for i in range(max(0, int((s["start"] - uem_start) / step)), min(nf, int((s["end"] - uem_start) / step))):
            pf.setdefault(i, set()).add(mm)

    miss = fa = conf = correct = total = 0
    for i in range(nf):
        t = uem_start + i * step
        if in_collar(t):
            continue
        gs = gf.get(i, set())
        ps = pf.get(i, set())
        if gs:
            total += len(gs)
            for s in gs:
                if s in ps:
                    correct += 1
                elif ps:
                    conf += 1
                else:
                    miss += 1
        fa += len(ps - gs if gs else ps)

    ts = total * step
    if ts == 0:
        return None
    return {
        "der": (miss + fa + conf) * step / ts * 100,
        "miss": miss * step / ts * 100,
        "fa": fa * step / ts * 100,
        "conf": conf * step / ts * 100,
        "correct": correct * step / ts * 100,
        "gt_speech_s": ts,
        "pred_speech_s": sum(s["end"] - s["start"] for s in pred),
        "gt_speakers": len({s["speaker"] for s in gt}),
        "pred_speakers": len({s["speaker"] for s in pred}),
        "uem_start": uem_start,
        "uem_end": uem_end,
    }


def read_rttm(path: Path) -> list[dict]:
    """Read an RTTM file into a list of {"speaker", "start", "end"} dicts."""
    segments = []
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if p and p[0] == "SPEAKER":
                segments.append({"speaker": p[7], "start": float(p[3]), "end": float(p[3]) + float(p[4])})
    return segments


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_summary(results: list[dict], collar: float) -> None:
    """Print aggregate DER statistics."""
    print(f"\n{'=' * 75}")
    print(f"COMPLETED: {len(results)} files evaluated (collar={collar}s)")
    print(f"{'=' * 75}", flush=True)

    if not results:
        return

    avg_der = sum(r["der"] for r in results) / len(results)
    avg_miss = sum(r["miss"] for r in results) / len(results)
    avg_fa = sum(r["fa"] for r in results) / len(results)
    avg_conf = sum(r["conf"] for r in results) / len(results)
    avg_correct = sum(r["correct"] for r in results) / len(results)

    total_gt = sum(r["gt_speech_s"] for r in results)
    w_der = sum(r["der"] * r["gt_speech_s"] for r in results) / total_gt
    w_miss = sum(r["miss"] * r["gt_speech_s"] for r in results) / total_gt
    w_fa = sum(r["fa"] * r["gt_speech_s"] for r in results) / total_gt
    w_conf = sum(r["conf"] * r["gt_speech_s"] for r in results) / total_gt

    spk_match = sum(1 for r in results if r["gt_speakers"] == r["pred_speakers"])

    print("\n--- Macro-Average (equal weight per file) ---")
    print(f"  DER:     {avg_der:.1f}%")
    print(f"  Miss:    {avg_miss:.1f}%")
    print(f"  FA:      {avg_fa:.1f}%")
    print(f"  Confuse: {avg_conf:.1f}%")
    print(f"  Correct: {avg_correct:.1f}%")

    print("\n--- Weighted Average (by GT speech duration) ---")
    print(f"  DER:     {w_der:.1f}%")
    print(f"  Miss:    {w_miss:.1f}%")
    print(f"  FA:      {w_fa:.1f}%")
    print(f"  Confuse: {w_conf:.1f}%")

    print("\n--- Speaker Count ---")
    print(f"  Exact match: {spk_match}/{len(results)} ({spk_match / len(results) * 100:.0f}%)")
    gt_counts = Counter(r["gt_speakers"] for r in results)
    pred_counts = Counter(r["pred_speakers"] for r in results)
    print(f"  GT distribution:   {dict(sorted(gt_counts.items()))}")
    print(f"  Pred distribution: {dict(sorted(pred_counts.items()))}")

    by_der = sorted(results, key=lambda r: r["der"])
    print("\n--- Best 5 files ---")
    for r in by_der[:5]:
        print(f"  {r['file_id']}: DER={r['der']:.1f}% (spk={r['gt_speakers']}gt/{r['pred_speakers']}pred)")
    print("\n--- Worst 5 files ---")
    for r in by_der[-5:]:
        print(f"  {r['file_id']}: DER={r['der']:.1f}% (spk={r['gt_speakers']}gt/{r['pred_speakers']}pred)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: C901, PLR0915
    args = parse_args()

    data_dir: Path = args.data_dir
    cha_dir = data_dir / "eng"
    mono_dir = data_dir / "mono"
    rttm_out: Path = args.rttm_out_dir
    results_json: Path = args.results_json
    collar: float = args.collar

    wav_files = sorted(data_dir.glob("*.wav"))
    print(f"Found {len(wav_files)} WAV files in {data_dir}", flush=True)

    if args.clean and rttm_out.exists():
        shutil.rmtree(rttm_out)
        print(f"Cleaned {rttm_out}", flush=True)
    rttm_out.mkdir(exist_ok=True)

    # --- Pre-process: ensure mono 16 kHz WAVs ---
    print("Ensuring mono 16 kHz WAVs (sox downmix)...", flush=True)
    mono_map: dict[str, Path] = {}
    for wav in wav_files:
        mono_map[wav.stem] = ensure_mono(wav, mono_dir)
    print(f"Mono files ready in {mono_dir}", flush=True)

    # --- Build initial tasks (skip already-processed files) ---
    done = {p.stem for p in rttm_out.glob("*.rttm")}
    initial_tasks = []
    for wav in wav_files:
        fid = wav.stem
        if fid in done:
            continue
        cha_path = cha_dir / f"{fid}.cha"
        if not cha_path.exists():
            continue
        initial_tasks.append(
            AudioBatch(
                data=[{"audio_filepath": str(mono_map[fid]), "session_name": fid}],
                task_id=f"callhome_{fid}",
                dataset_name="callhome_eng0",
            )
        )

    print(f"Already done: {len(done)}, remaining: {len(initial_tasks)}", flush=True)

    # --- Run inference via Pipeline + RayActorPoolExecutor ---
    if initial_tasks:
        stage = InferenceSortformerStage(
            model_name=args.model,
            rttm_out_dir=str(rttm_out),
            chunk_len=args.chunk_len,
            chunk_right_context=args.chunk_right_context,
            fifo_len=args.fifo_len,
            spkcache_update_period=args.spkcache_update_period,
            spkcache_len=args.spkcache_len,
            inference_batch_size=1,
        )

        pipeline = Pipeline(
            name="callhome_sortformer_diarization",
            stages=[stage],
        )

        num_gpus = torch.cuda.device_count()
        print(f"Detected {num_gpus} GPU(s)", flush=True)
        ray.init(num_gpus=num_gpus, ignore_reinit_error=True)

        executor = RayActorPoolExecutor()

        print("Starting parallel inference via RayActorPoolExecutor...", flush=True)
        t0 = time.time()
        pipeline.run(executor=executor, initial_tasks=initial_tasks)
        print(f"\nInference done in {(time.time() - t0) / 60:.1f} min", flush=True)

    # --- Compute DER for ALL processed files ---
    print(f"\nComputing DER (collar={collar}s, UEM from CHA)...", flush=True)
    results = []
    for rttm_path in sorted(rttm_out.glob("*.rttm")):
        fid = rttm_path.stem
        cha_path = cha_dir / f"{fid}.cha"
        if not cha_path.exists():
            continue
        gt, uem_start, uem_end = parse_cha(cha_path)
        if not gt:
            continue
        pred = read_rttm(rttm_path)
        if not pred:
            continue
        metrics = compute_der(gt, pred, uem_start, uem_end, collar=collar)
        if metrics is None:
            continue
        metrics["file_id"] = fid
        results.append(metrics)

    print_summary(results, collar)

    with open(results_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {results_json}")


if __name__ == "__main__":
    main()
