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

import argparse
import os
from pathlib import Path

from loguru import logger

from nemo_curator.core.client import RayClient
from nemo_curator.stages.deduplication.fuzzy.workflow import (
    FuzzyDeduplicationWorkflow,
    ID_GENERATOR_OUTPUT_FILENAME,
)
from nemo_curator.stages.deduplication.fuzzy.identify_duplicates import (
    DUPLICATE_IDS_SUBDIR,
)
from nemo_curator.stages.text.deduplication.removal_workflow import TextDuplicatesRemovalWorkflow


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fuzzy deduplication on Parquet or JSONL files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Input directory or glob pattern for Parquet/JSONL files",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        required=True,
        help="Cache directory for deduplication intermediates (must be empty between runs)",
    )
    parser.add_argument(
        "--duplicate_ids_output_path",
        type=str,
        required=True,
        help="Output directory for duplicate IDs and id generator mapping",
    )
    parser.add_argument(
        "--deduplicated_output_path",
        type=str,
        default=None,
        help="Output directory for deduplicated data (default: {output_path}/deduplicated)",
    )
    parser.add_argument(
        "--text_field",
        type=str,
        default="text",
        help="Field containing the text to deduplicate",
    )
    parser.add_argument(
        "--input_filetype",
        type=str,
        choices=["parquet", "jsonl"],
        default="jsonl",
        help="Input file type (auto-detected if not specified)",
    )
    parser.add_argument(
        "--input_blocksize",
        type=str,
        default="1GiB",
        help="Size of input blocks to read",
    )
    parser.add_argument(
        "--bands_per_iteration",
        type=int,
        default=5,
        help="Number of bands to shuffle concurrently (reduce if OOM)",
    )
    # MinHash + LSH parameters
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for minhash permutations",
    )
    parser.add_argument(
        "--char_ngrams",
        type=int,
        default=24,
        help="Size of character n-grams for MinHash (recommended: >= 20)",
    )
    parser.add_argument(
        "--num_bands",
        type=int,
        default=20,
        help="Number of bands/buckets for LSH",
    )
    parser.add_argument(
        "--minhashes_per_band",
        type=int,
        default=13,
        help="Number of hashes per band",
    )
    parser.add_argument(
        "--use_64_bit_hash",
        action="store_true",
        default=False,
        help="Use 64-bit hash function (default: 32-bit)",
    )

    args = parser.parse_args()

    # Check if cache directory is empty (recommended)
    cache_files = list(Path(args.cache_dir).glob("*"))
    if cache_files:
        logger.warning(
            f"Cache directory {args.cache_dir} is not empty. "
            "It's recommended to clear it between runs to avoid conflicts."
        )

    # Get the input file extensions based on the input filetype
    if args.input_filetype == "parquet":
        input_file_extensions = [".parquet"]
    elif args.input_filetype == "jsonl":
        input_file_extensions = [".jsonl", ".json"]

    ray_client = RayClient()
    ray_client.start()

    try:
        # Step 1: Run fuzzy deduplication to identify duplicates
        logger.info("Running fuzzy deduplication workflow to identify duplicate IDs...")
        fuzzy_workflow = FuzzyDeduplicationWorkflow(
            input_path=args.input_path,
            cache_path=args.cache_dir,
            output_path=args.duplicate_ids_output_path,
            input_filetype=args.input_filetype,
            input_file_extensions=input_file_extensions,
            input_blocksize=args.input_blocksize,
            text_field=args.text_field,
            perform_removal=False,  # Only identification, not removal
            char_ngrams=args.char_ngrams,
            num_bands=args.num_bands,
            minhashes_per_band=args.minhashes_per_band,
            use_64_bit_hash=args.use_64_bit_hash,
            bands_per_iteration=args.bands_per_iteration,
            seed=args.seed,
        )
        fuzzy_workflow.run()

        # Step 2: Remove duplicates using the identified duplicate IDs
        # Fuzzy deduplication outputs:
        # - Duplicate IDs: {output_path}/FuzzyDuplicateIds/ (parquet files with "id" column)
        # - ID generator: {output_path}/fuzzy_id_generator.json
        duplicate_ids_path = os.path.join(args.duplicate_ids_output_path, "FuzzyDuplicateIds")
        id_generator_path = os.path.join(args.duplicate_ids_output_path, "fuzzy_id_generator.json")

        logger.info("Running text duplicates removal workflow to remove duplicates...")
        removal_workflow = TextDuplicatesRemovalWorkflow(
            input_path=args.input_path,
            ids_to_remove_path=duplicate_ids_path,
            output_path=args.deduplicated_output_path,
            input_filetype=args.input_filetype,
            input_file_extensions=input_file_extensions,
            input_id_field="_curator_dedup_id",
            ids_to_remove_duplicate_id_field="_curator_dedup_id",
            input_blocksize=args.input_blocksize,
            id_generator_path=id_generator_path,
        )
        removal_workflow.run()

        logger.info("Pipeline completed successfully.")
        logger.info(f"Deduplication complete! Deduplicated output: {args.deduplicated_output_path}")

    finally:
        ray_client.stop()

if __name__ == "__main__":
    main()
