"""Multi-GPU orchestration: build manifest, shard, launch workers, show
progress, merge results.

All Python — no bash wrapper needed.  Each GPU runs as a subprocess for
CUDA context isolation.  The parent process polls per-GPU log files to
drive a unified tqdm progress bar showing total utterances across all GPUs.
"""

import logging
import os
import re
import subprocess
import sys
import time
from typing import List, Optional

from nemo_curator.stages.audio.speaker_id.data.manifest import build_manifest
from nemo_curator.stages.audio.speaker_id.utils.io import merge_embedding_shards

logger = logging.getLogger(__name__)

_DONE_RE = re.compile(r"done=(\d+)")


def _count_done_in_log(log_path: str) -> int:
    """Parse the last occurrence of ``done=N`` from a GPU log file."""
    if not os.path.isfile(log_path):
        return 0
    try:
        with open(log_path, "rb") as f:
            # Read last 4 KB — the tqdm line is always near the end
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 4096))
            tail = f.read().decode("utf-8", errors="replace")
        matches = _DONE_RE.findall(tail)
        return int(matches[-1]) if matches else 0
    except Exception:
        return 0


def launch_multigpu(
    base_dir: str,
    language: str,
    subsets: List[str],
    model_name: str,
    num_gpus: int,
    batch_dur: float = 600.0,
    sample_rate: int = 16000,
    num_mel_bins: int = 80,
    model_cache_dir: Optional[str] = None,
    skip_extract: bool = False,
) -> str:
    """Full pipeline: build manifest, launch per-GPU workers, show progress,
    merge shards.

    Returns path to the merged embeddings file.
    """
    lang_dir = os.path.join(base_dir, language)
    output_dir = os.path.join(lang_dir, "wespeaker_embeddings")
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: build manifest so we know total utterance count ──
    logger.info("Building manifest for %s ...", language)
    all_entries, _ = build_manifest(lang_dir, subsets)
    total_utts = len(all_entries)
    total_hours = sum(e.get("duration", 0) for e in all_entries) / 3600

    durs = sorted([e.get("duration", 0) for e in all_entries], reverse=True)
    longest = durs[0] if durs else 0
    median = durs[len(durs) // 2] if durs else 0

    per_gpu = [(total_utts + num_gpus - 1) // num_gpus] * num_gpus
    remainder = total_utts % num_gpus
    if remainder:
        for i in range(remainder, num_gpus):
            per_gpu[i] = total_utts // num_gpus

    print(flush=True)
    print("=" * 60, flush=True)
    print(" Multi-GPU Speaker Embedding Extraction", flush=True)
    print("=" * 60, flush=True)
    print(f"  Language:    {language}", flush=True)
    print(f"  Model:       {model_name}", flush=True)
    print(f"  GPUs:        {num_gpus}", flush=True)
    print(f"  Batch dur:   {batch_dur}s", flush=True)
    print(f"  Total utts:  {total_utts:,}", flush=True)
    print(f"  Total hours: {total_hours:.2f}h", flush=True)
    print(f"  Longest utt: {longest:.1f}s   Median: {median:.1f}s", flush=True)
    for i in range(num_gpus):
        print(f"  GPU {i}:       {per_gpu[i]:,} utterances", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)

    # ── Step 2: launch per-GPU subprocesses ──
    script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "scripts",
        "extract_embeddings.py",
    )

    procs = []
    log_files = []

    for gpu_id in range(num_gpus):
        log_path = os.path.join(output_dir, f"log_gpu{gpu_id}.txt")
        log_files.append(log_path)

        cmd = [
            sys.executable, script,
            "--base-dir", base_dir,
            "--language", language,
            "--subsets", *subsets,
            "--model", model_name,
            "--device", f"cuda:{gpu_id}",
            "--gpu-id", str(gpu_id),
            "--num-gpus", str(num_gpus),
            "--batch-dur", str(batch_dur),
            "--sample-rate", str(sample_rate),
            "--num-mel-bins", str(num_mel_bins),
            "--skip-extract",
        ]
        if model_cache_dir:
            cmd.extend(["--model-cache-dir", model_cache_dir])

        logger.info("Launching GPU %d ...", gpu_id)

        with open(log_path, "w", encoding="utf-8") as lf:
            proc = subprocess.Popen(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
            )
        procs.append((gpu_id, proc))

    # ── Step 3: poll progress with tqdm ──
    from tqdm import tqdm

    pbar = tqdm(
        total=total_utts,
        desc="All GPUs",
        unit="utt",
        dynamic_ncols=True,
    )

    prev_total = 0
    while True:
        # Check if all processes finished
        all_done = all(proc.poll() is not None for _, proc in procs)

        current_total = sum(_count_done_in_log(lp) for lp in log_files)
        delta = current_total - prev_total
        if delta > 0:
            pbar.update(delta)
            prev_total = current_total

        per_gpu_done = [_count_done_in_log(lp) for lp in log_files]
        pbar.set_postfix_str(
            "  ".join(f"gpu{i}={d}" for i, d in enumerate(per_gpu_done))
        )

        if all_done:
            # Final read to catch any remaining progress
            current_total = sum(_count_done_in_log(lp) for lp in log_files)
            delta = current_total - prev_total
            if delta > 0:
                pbar.update(delta)
            break

        time.sleep(2)

    pbar.close()

    # ── Step 4: check for failures ──
    failed = 0
    for gpu_id, proc in procs:
        if proc.returncode != 0:
            failed += 1
            logger.error(
                "GPU %d FAILED (exit %d). See %s",
                gpu_id, proc.returncode, log_files[gpu_id],
            )
        else:
            logger.info("GPU %d finished successfully", gpu_id)

    if failed:
        raise RuntimeError(
            f"{failed}/{num_gpus} GPU processes failed. Check logs in {output_dir}"
        )

    # ── Step 5: merge shards ──
    logger.info("All GPUs done. Merging %d shards...", num_gpus)
    merged_emb_path, _ = merge_embedding_shards(output_dir, num_gpus)
    logger.info("Merged embeddings: %s", merged_emb_path)
    return merged_emb_path
