# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""
Run fuzzy-dedup *identification only* (no removal) on a JSONL/Parquet corpus,
as step 2 of the fuzzy-dedup-eval example.

Example:
    python tutorials/eval/dedup/2_run_fuzzy_dedup.py \
        --input-path output/dedup_eval/raw_corpus \
        --input-filetype jsonl \
        --cache-dir output/dedup_eval/fuzzy_cache \
        --output-dir output/dedup_eval/fuzzy_ids
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from nemo_curator.core.client import RayClient
from nemo_curator.stages.deduplication.fuzzy.identify_duplicates import DUPLICATE_IDS_SUBDIR
from nemo_curator.stages.deduplication.fuzzy.workflow import ID_GENERATOR_OUTPUT_FILENAME, FuzzyDeduplicationWorkflow


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-path", type=str, required=True, help="Input directory of Parquet/JSONL files.")
    parser.add_argument(
        "--input-filetype", type=str, choices=["parquet", "jsonl"], default="jsonl", help="Input file type."
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="Cache directory for dedup intermediates (must be empty between runs). "
        "build_pair_dataset.py reads group labels from '<cache-dir>/ConnectedComponentsStage/'.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for duplicate IDs and the id generator mapping. "
        "build_pair_dataset.py reads '<output-dir>/FuzzyDuplicateIds/' and "
        f"'<output-dir>/{ID_GENERATOR_OUTPUT_FILENAME}'.",
    )
    parser.add_argument("--text-field", type=str, default="text", help="Field containing the text to deduplicate.")
    parser.add_argument("--input-blocksize", type=str, default="1GiB", help="Size of input blocks to read.")
    parser.add_argument(
        "--bands-per-iteration", type=int, default=5, help="Number of bands to shuffle concurrently (reduce if OOM)."
    )
    # MinHash + LSH parameters
    parser.add_argument("--seed", type=int, default=42, help="Seed for minhash permutations.")
    parser.add_argument(
        "--char-ngrams", type=int, default=24, help="Size of character n-grams for MinHash (recommended: >= 20)."
    )
    parser.add_argument("--num-bands", type=int, default=20, help="Number of bands/buckets for LSH.")
    parser.add_argument("--minhashes-per-band", type=int, default=13, help="Number of hashes per band.")
    parser.add_argument(
        "--use-64-bit-hash",
        action="store_true",
        default=False,
        help="Use 64-bit hash function (default: 32-bit).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    cache_dir = Path(args.cache_dir)
    if cache_dir.exists() and list(cache_dir.glob("*")):
        logger.warning(f"Cache directory {cache_dir} is not empty. Clear it between runs to avoid conflicts.")

    input_file_extensions = [".parquet"] if args.input_filetype == "parquet" else [".jsonl", ".json"]

    ray_client = RayClient()
    ray_client.start()
    try:
        logger.info("Running fuzzy deduplication identification (no removal)...")
        fuzzy_workflow = FuzzyDeduplicationWorkflow(
            input_path=args.input_path,
            cache_path=args.cache_dir,
            output_path=args.output_dir,
            input_filetype=args.input_filetype,
            input_file_extensions=input_file_extensions,
            input_blocksize=args.input_blocksize,
            text_field=args.text_field,
            perform_removal=False,
            char_ngrams=args.char_ngrams,
            num_bands=args.num_bands,
            minhashes_per_band=args.minhashes_per_band,
            use_64_bit_hash=args.use_64_bit_hash,
            bands_per_iteration=args.bands_per_iteration,
            seed=args.seed,
        )
        fuzzy_workflow.run()
    finally:
        ray_client.stop()

    connected_components_dir = cache_dir / "ConnectedComponentsStage"
    duplicate_ids_dir = Path(args.output_dir) / DUPLICATE_IDS_SUBDIR
    id_generator_path = Path(args.output_dir) / ID_GENERATOR_OUTPUT_FILENAME

    logger.info("Fuzzy dedup identification complete. Artifacts for build_pair_dataset.py:")
    logger.info(f"  group labels (_curator_dedup_id, _duplicate_group_id): {connected_components_dir}")
    logger.info(f"  removal ids  (_curator_dedup_id):                       {duplicate_ids_dir}")
    logger.info(f"  id generator state:                                     {id_generator_path}")
    if not connected_components_dir.exists():
        logger.warning(
            "No ConnectedComponentsStage output found -- this usually means no fuzzy duplicates were "
            "detected in the input corpus, so there are no duplicate groups to build pairs from."
        )


if __name__ == "__main__":
    main()
