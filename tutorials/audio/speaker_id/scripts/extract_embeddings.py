#!/usr/bin/env python3
"""Extract speaker embeddings from a downloaded YODAS dataset.

Single-GPU mode (default):
    python scripts/extract_embeddings.py --base-dir /disk_f_nvd/datasets/Yodas/ --language hr

Multi-GPU (preferred — use run_multigpu.py):
    python run_multigpu.py --base-dir /disk_f_nvd/datasets/Yodas/ --language hr --num-gpus 4

Per-GPU worker (launched by run_multigpu.py, not typically called directly):
    python scripts/extract_embeddings.py --base-dir ... --language hr --num-gpus 4 --gpu-id 0 --skip-extract
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(
        description="Extract WeSpeaker speaker embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--base-dir", required=True,
                    help="Root data directory (e.g. /disk_f_nvd/datasets/Yodas/)")
    p.add_argument("--language", required=True,
                    help="Language code (e.g. hr)")
    p.add_argument("--subsets", nargs="+", default=["0_by_whisper", "0_from_captions"],
                    help="Subset directories to process")
    p.add_argument("--model", default="voxblink2_samresnet100_ft",
                    help="WeSpeaker model name or local path")
    p.add_argument("--model-cache-dir", default=None,
                    help="Directory to cache downloaded models")
    p.add_argument("--device", default=None,
                    help="Device string (default: cuda:<gpu-id>)")
    p.add_argument("--gpu-id", type=int, default=0,
                    help="GPU index for this worker (for manual multi-GPU)")
    p.add_argument("--num-gpus", type=int, default=1,
                    help="Total GPUs for sharding")
    p.add_argument("--batch-dur", type=float, default=600.0,
                    help="Max audio seconds per dynamic batch")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--num-mel-bins", type=int, default=80)
    p.add_argument("--output-dir", default=None,
                    help="Optional output directory for embeddings/wav.scp")
    p.add_argument("--skip-extract", action="store_true",
                    help="Skip tar extraction (wavs already extracted)")
    p.add_argument("--skip-embed", action="store_true",
                    help="Skip embedding extraction (only extract + manifest)")
    p.add_argument("--merge-only", action="store_true",
                    help="Only merge per-GPU shards (no extraction)")
    args = p.parse_args()

    if args.device is None:
        args.device = f"cuda:{args.gpu_id}"

    lang_dir = os.path.join(args.base_dir, args.language)
    output_dir = args.output_dir or os.path.join(lang_dir, "wespeaker_embeddings")
    os.makedirs(output_dir, exist_ok=True)

    # --- merge-only mode ---
    if args.merge_only:
        from nemo_curator.stages.audio.speaker_id.utils.io import merge_embedding_shards
        logger.info("Merging %d GPU shards...", args.num_gpus)
        emb_path, utt_path = merge_embedding_shards(output_dir, args.num_gpus)
        logger.info("Merged: %s", emb_path)
        return

    # --- multi-GPU launcher mode (num_gpus > 1 and no gpu-id override) ---
    if args.num_gpus > 1 and args.gpu_id == 0 and not args.skip_extract:
        from nemo_curator.stages.audio.speaker_id.multigpu.launcher import launch_multigpu
        launch_multigpu(
            base_dir=args.base_dir,
            language=args.language,
            subsets=args.subsets,
            model_name=args.model,
            num_gpus=args.num_gpus,
            batch_dur=args.batch_dur,
            sample_rate=args.sample_rate,
            num_mel_bins=args.num_mel_bins,
            model_cache_dir=args.model_cache_dir,
            skip_extract=args.skip_extract,
        )
        return

    # --- single-GPU / per-GPU worker mode ---

    # Step 1: Extract tars
    if not args.skip_extract and not args.skip_embed:
        from nemo_curator.stages.audio.speaker_id.data.tar_extractor import extract_tars
        logger.info("Step 1: Extracting wav files from tar archives...")
        extract_tars(lang_dir, args.subsets)

    # Step 2: Build manifest
    from nemo_curator.stages.audio.speaker_id.data.manifest import build_manifest
    logger.info("Step 2: Building manifest and wav.scp...")
    all_entries, wav_scp_path = build_manifest(lang_dir, args.subsets)

    if args.skip_embed:
        logger.info("Skipping embedding extraction (--skip-embed)")
        return

    # Step 3: Extract embeddings
    from nemo_curator.stages.audio.speaker_id.embedding.model_loader import load_wespeaker_model
    from nemo_curator.stages.audio.speaker_id.embedding.extractor import extract_embeddings
    from nemo_curator.stages.audio.speaker_id.utils.io import save_embeddings

    logger.info("Step 3: Extracting embeddings...")
    loaded = load_wespeaker_model(
        args.model, device=args.device, model_cache_dir=args.model_cache_dir,
    )

    # Shard for multi-GPU (interleaved so each GPU gets a mix of durations)
    shard = all_entries[args.gpu_id :: args.num_gpus]

    logger.info(
        "GPU %d/%d: %d/%d utterances",
        args.gpu_id, args.num_gpus, len(shard), len(all_entries),
    )

    embeddings, utt_ids = extract_embeddings(
        entries=shard,
        model=loaded.model,
        frontend_type=loaded.frontend_type,
        device=loaded.device,
        batch_dur=args.batch_dur,
        sample_rate=args.sample_rate,
        num_mel_bins=args.num_mel_bins,
    )

    suffix = f"_gpu{args.gpu_id}" if args.num_gpus > 1 else ""
    emb_path, utt_path = save_embeddings(embeddings, utt_ids, output_dir, suffix=suffix)
    logger.info("Saved %d embeddings -> %s", len(utt_ids), emb_path)

    if args.num_gpus == 1:
        logger.info("Done!")
    else:
        logger.info(
            "GPU %d done. After all GPUs finish, run with --merge-only --num-gpus %d",
            args.gpu_id, args.num_gpus,
        )


if __name__ == "__main__":
    main()
