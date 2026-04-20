#!/usr/bin/env python3
"""Download YODAS / Granary dataset (audio tars + manifests) from S3.

Usage:
    python scripts/download_data.py --language hr
    python scripts/download_data.py --language hr --base-dir /data/Yodas --s3cfg ~/.s3cfg[pdx]
    python scripts/download_data.py --language de --subsets 0_by_whisper --workers 32
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

from nemo_curator.stages.audio.speaker_id.data.downloader import download_dataset


def main():
    p = argparse.ArgumentParser(description="Download YODAS dataset from S3")
    p.add_argument("--language", required=True, help="Language code (e.g. hr, de, cs)")
    p.add_argument("--base-dir", default="/disk_f_nvd/datasets/Yodas/",
                    help="Local root directory for downloaded data")
    p.add_argument("--subsets", nargs="+", default=["0_by_whisper", "0_from_captions"],
                    help="Subset names to download")
    p.add_argument("--s3cfg", default="~/.s3cfg[default]",
                    help="Path to .s3cfg file with section, e.g. ~/.s3cfg[default]")
    p.add_argument("--audio-pattern",
                    default="s3://yodas2/{language}/{subset}/audio__OP_0..63_CL_.tar",
                    help="S3 pattern template for audio tars")
    p.add_argument("--manifest-pattern",
                    default=("s3://granary/version_1_0/manifests/manifests_all_pnc/"
                             "ASR_updated/YODAS2/{language}/{subset}/"
                             "sharded_manifests_updated/manifest__OP_0..63_CL_.json"),
                    help="S3 pattern template for manifest JSONs")
    p.add_argument("--workers", type=int, default=16, help="Parallel download threads")
    p.add_argument("--force", action="store_true", help="Re-download even if file exists")
    args = p.parse_args()

    download_dataset(
        language=args.language,
        subsets=args.subsets,
        base_dir=args.base_dir,
        s3cfg=args.s3cfg,
        audio_pattern_template=args.audio_pattern,
        manifest_pattern_template=args.manifest_pattern,
        workers=args.workers,
        force=args.force,
    )


if __name__ == "__main__":
    main()
