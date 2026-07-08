#!/usr/bin/env python3
"""Stage 1: encode each postprocessed WAV as 16 kHz mono Opus under audio_16k/."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from david_ai_common import PipelineError, load_jsonl, log_exception, resample_opus, run_main, run_thread_pool

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifests-dir", type=Path, required=True)
    ap.add_argument("--target-sr", type=int, default=16000)
    ap.add_argument("--opus-bitrate", default="32k", help="Opus encoder bitrate (default: 32k)")
    ap.add_argument("--workers", type=int, default=1, help="Parallel encode workers")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    manifests_dir = args.manifests_dir.resolve()
    pairs: dict[str, tuple[Path, Path]] = {}
    manifest_errors = 0

    for path in sorted(manifests_dir.glob("*_norm.jsonl")):
        if path.name == "all_norm.jsonl":
            continue
        try:
            rows = load_jsonl(path)
        except Exception as exc:
            log_exception(f"cannot load manifest {path}", exc)
            manifest_errors += 1
            continue
        for row in rows:
            try:
                src = Path(row["audio_filepath"])
                dst = Path(row["audio_filepath_16k"])
            except KeyError as exc:
                log_exception(f"missing audio path in {path}", exc)
                manifest_errors += 1
                continue
            pairs[str(dst)] = (src, dst)

    if not pairs:
        raise PipelineError(f"No audio paths found in {manifests_dir}/*_norm.jsonl")

    converted = skipped = failed = 0

    def _encode_one(item: tuple[str, tuple[Path, Path]]) -> str:
        _dst_str, (src, dst) = item
        if dst.exists() and not args.force:
            return "skipped"
        if args.force and dst.exists():
            try:
                dst.unlink()
            except OSError as exc:
                log_exception(f"cannot remove existing file {dst}", exc)
                return "failed"
        if not src.is_file():
            logger.warning("Missing source audio: %s", src)
            return "failed"
        return (
            "converted"
            if resample_opus(src, dst, target_sr=args.target_sr, bitrate=args.opus_bitrate)
            else "failed"
        )

    workers = max(1, args.workers)
    for status in run_thread_pool(sorted(pairs.items()), _encode_one, workers=workers):
        if status == "converted":
            converted += 1
        elif status == "skipped":
            skipped += 1
        else:
            failed += 1

    logger.info(
        "Done: converted=%d skipped=%d failed=%d (unique files=%d, workers=%d)",
        converted,
        skipped,
        failed,
        len(pairs),
        workers,
    )
    return 1 if (failed or manifest_errors) else 0


if __name__ == "__main__":
    run_main(main)
