#!/usr/bin/env python3
"""Multi-GPU WeSpeaker embedding extraction.

Builds the manifest, shows per-GPU utterance counts, launches one process
per GPU with a unified tqdm progress bar, then merges all shards.

Usage:
    python run_multigpu.py --base-dir /disk_f_nvd/datasets/Yodas/ --language da --num-gpus 2
    python run_multigpu.py --base-dir /disk_f_nvd/datasets/Yodas/ --language hr --num-gpus 4 --batch-dur 500
    python run_multigpu.py --base-dir /disk_f_nvd/datasets/Yodas/ --language hr --model /path/to/model
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def main():
    p = argparse.ArgumentParser(
        description="Multi-GPU speaker embedding extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--base-dir", required=True,
                    help="Root data directory (e.g. /disk_f_nvd/datasets/Yodas/)")
    p.add_argument("--language", required=True,
                    help="Language code (e.g. hr, da, de)")
    p.add_argument("--subsets", nargs="+", default=["0_by_whisper", "0_from_captions"],
                    help="Subset directories to process")
    p.add_argument("--model", default="voxblink2_samresnet100_ft",
                    help="WeSpeaker model name or local path")
    p.add_argument("--model-cache-dir", default=None,
                    help="Directory to cache downloaded models")
    p.add_argument("--num-gpus", type=int, default=None,
                    help="Number of GPUs (default: auto-detect)")
    p.add_argument("--batch-dur", type=float, default=600.0,
                    help="Max audio seconds per dynamic batch")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--num-mel-bins", type=int, default=80)
    args = p.parse_args()

    if args.num_gpus is None:
        import torch
        args.num_gpus = torch.cuda.device_count()
        if args.num_gpus == 0:
            print("ERROR: No GPUs detected. Use --num-gpus to override.", file=sys.stderr)
            sys.exit(1)

    from nemo_curator.stages.audio.speaker_id.multigpu.launcher import launch_multigpu

    merged_path = launch_multigpu(
        base_dir=args.base_dir,
        language=args.language,
        subsets=args.subsets,
        model_name=args.model,
        num_gpus=args.num_gpus,
        batch_dur=args.batch_dur,
        sample_rate=args.sample_rate,
        num_mel_bins=args.num_mel_bins,
        model_cache_dir=args.model_cache_dir,
    )

    print(f"\nDone! Embeddings: {merged_path}")


if __name__ == "__main__":
    main()
