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
Turn the outputs of `2_run_fuzzy_dedup.py` into a labeled JSONL dataset
of document pairs suitable for `LLMJudgeWorkflow`.

Writes raw pairs only -- 4_span_alignment.py adds `semantic_diff`/`truncated`
as a separate step.

Example:
    python tutorials/eval/dedup/3_build_pair_dataset.py \
        --input-path output/dedup_eval/raw_corpus \
        --input-filetype jsonl \
        --cache-dir output/dedup_eval/fuzzy_cache \
        --fuzzy-output-dir output/dedup_eval/fuzzy_ids \
        --output-path output/dedup_eval/keeper_removed_pairs_raw \
        --pair-strategy keeper_removed
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import ray
from build_pairs.all_pairwise import build_all_pairwise_pairs, build_oversized_group_pairs
from build_pairs.cross_group_sample import build_cross_group_sample_pairs
from build_pairs.keeper_removed import build_keeper_removed_pairs
from loguru import logger

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.deduplication.fuzzy.identify_duplicates import DUPLICATE_IDS_SUBDIR
from nemo_curator.stages.deduplication.fuzzy.utils import CURATOR_FUZZY_DUPLICATE_GROUP_FIELD
from nemo_curator.stages.deduplication.fuzzy.workflow import ID_GENERATOR_OUTPUT_FILENAME
from nemo_curator.stages.deduplication.id_generator import (
    CURATOR_DEDUP_ID_STR,
    create_id_generator_actor,
    kill_id_generator_actor,
)
from nemo_curator.stages.text.io.reader import JsonlReader, ParquetReader
from nemo_curator.stages.text.io.writer import ParquetWriter

_VALID_STRATEGIES = ("keeper_removed", "all_pairwise", "cross_group_sample")
CORPUS_WITH_IDS_SUBDIR = "CorpusWithIds"


def _read_parquet_glob(directory: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(Path(directory) / "*.parquet")))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


@ray.remote
class _RowCounter:
    def __init__(self) -> None:
        self._total = 0

    def add(self, count: int) -> None:
        self._total += count

    def total(self) -> int:
        return self._total


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--ray-temp-dir", default="/tmp/ray", help="Ray temporary directory.")  # noqa: S108

    parser.add_argument("--input-path", required=True, help="Original corpus path. MUST match step 2.")
    parser.add_argument("--input-filetype", choices=["parquet", "jsonl"], default="jsonl", help="MUST match step 2.")
    parser.add_argument("--input-blocksize", default="1GiB", help="MUST match step 2.")
    parser.add_argument(
        "--cache-dir", required=True, help="Step 2's --cache-dir. Also where this script writes CorpusWithIds/."
    )
    parser.add_argument("--fuzzy-output-dir", required=True, help="Step 2's --output-dir.")
    parser.add_argument("--output-path", required=True, help="Directory for the raw pairs JSONL output.")

    parser.add_argument(
        "--pair-strategy",
        choices=_VALID_STRATEGIES,
        required=True,
        help="Run once per strategy, with a different --output-path each time.",
    )
    parser.add_argument(
        "--cross-group-samples", type=int, default=200, help="Pairs to sample for 'cross_group_sample'."
    )
    parser.add_argument(
        "--max-group-size",
        type=int,
        default=200,
        help="Largest 'all_pairwise' group expanded to every pair (C(n,2) pairs); larger groups are sampled instead.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--pairs-per-file",
        type=int,
        default=2000,
        help="Max pairs per output file. Only applies to 'cross_group_sample'; "
        "the other strategies are sharded by Ray Data instead.",
    )

    parser.add_argument(
        "--oversized-group-pairs",
        type=int,
        default=50,
        help="Random pairs sampled from each 'all_pairwise' group larger than --max-group-size (0 disables).",
    )

    return parser.parse_args()


def main() -> None:  # noqa: PLR0912, PLR0915
    args = _parse_args()
    cache_dir = Path(args.cache_dir)

    id_generator_path = str(Path(args.fuzzy_output_dir) / ID_GENERATOR_OUTPUT_FILENAME)
    logger.info(f"Re-reading corpus from {args.input_path} with ids replayed from {id_generator_path}...")
    corpus_dir = cache_dir / CORPUS_WITH_IDS_SUBDIR
    metadata_file = corpus_dir / "_reassignment_metadata.json"
    metadata = {
        "input_path": args.input_path,
        "input_filetype": args.input_filetype,
        "input_blocksize": args.input_blocksize,
    }

    cached_metadata = json.loads(metadata_file.read_text(encoding="utf-8")) if metadata_file.exists() else None
    if any(corpus_dir.glob("*.parquet")):
        if cached_metadata != metadata:
            msg = (
                f"{corpus_dir} was built from different or unknown inputs ({cached_metadata}) than this run "
                f"requested ({metadata}). Delete {corpus_dir} to rebuild it, or point --cache-dir at "
                "a fresh directory."
            )
            raise RuntimeError(msg)
        logger.info(f"Reusing already-written {corpus_dir} (delete it to force a re-read).")
    else:
        reader_class = ParquetReader if args.input_filetype == "parquet" else JsonlReader
        reassign_pipeline = Pipeline(
            name="dedup_eval_id_reassignment",
            description="Re-read the original corpus with the same _curator_dedup_id values fuzzy dedup assigned.",
            stages=[
                reader_class(
                    file_paths=args.input_path,
                    blocksize=args.input_blocksize,
                    _assign_ids=True,
                    fields=["url", "text"],
                ),
                ParquetWriter(path=str(corpus_dir)),
            ],
        )

        reassign_client = RayClient(ray_temp_dir=args.ray_temp_dir)
        reassign_client.start()
        try:
            create_id_generator_actor(id_generator_path)
            try:
                reassign_pipeline.run()
            finally:
                kill_id_generator_actor()
        finally:
            reassign_client.stop()
            ray.shutdown()

        if not any(corpus_dir.glob("*.parquet")):
            msg = (
                f"No documents were read back from {args.input_path!r}; "
                "check --input-path/--input-filetype/--input-blocksize."
            )
            raise RuntimeError(msg)

        metadata_file.write_text(json.dumps(metadata), encoding="utf-8")

    connected_components_dir = cache_dir / "ConnectedComponentsStage"
    group_labels_df = _read_parquet_glob(str(connected_components_dir))
    if group_labels_df.empty:
        msg = (
            f"No group labels found under {connected_components_dir}. Either no fuzzy duplicates were found "
            "by 2_run_fuzzy_dedup.py, or --cache-dir does not match the value passed to it."
        )
        raise RuntimeError(msg)
    logger.info(f"Loaded group labels for {len(group_labels_df)} documents across duplicate groups.")

    removal_df = _read_parquet_glob(str(Path(args.fuzzy_output_dir) / DUPLICATE_IDS_SUBDIR))
    removed_ids = (
        np.unique(removal_df[CURATOR_DEDUP_ID_STR].to_numpy(dtype=np.int64))
        if not removal_df.empty
        else np.empty(0, dtype=np.int64)
    )
    logger.info(f"Loaded {len(removed_ids)} ids fuzzy dedup marked for removal.")

    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.pair_strategy == "cross_group_sample":
        build_cross_group_sample_pairs(
            args.cross_group_samples, corpus_dir, args.seed, group_labels_df, args.pairs_per_file, output_dir
        )
        return

    label_ids = group_labels_df[CURATOR_DEDUP_ID_STR].to_numpy(dtype=np.int64)
    label_groups = group_labels_df[CURATOR_FUZZY_DUPLICATE_GROUP_FIELD].to_numpy(dtype=np.int64)
    sort_order = np.argsort(label_ids)
    label_ids, label_groups = label_ids[sort_order], label_groups[sort_order]

    ray_client = RayClient(ray_temp_dir=args.ray_temp_dir)
    ray_client.start()
    try:
        counter = _RowCounter.remote()

        if args.pair_strategy == "keeper_removed":
            paired_ds = build_keeper_removed_pairs(label_ids, removed_ids, label_groups, corpus_dir, counter)
        else:
            paired_ds = build_all_pairwise_pairs(
                label_groups, args.max_group_size, args.oversized_group_pairs, label_ids, counter, corpus_dir
            )

        paired_ds.write_json(str(output_dir))
        num_pairs = ray.get(counter.total.remote())

        if args.pair_strategy == "all_pairwise" and args.oversized_group_pairs > 0:
            num_pairs += build_oversized_group_pairs(
                label_ids,
                label_groups,
                args.max_group_size,
                args.seed,
                args.oversized_group_pairs,
                corpus_dir,
                output_dir,
            )
    finally:
        ray_client.stop()
        ray.shutdown()

    if not num_pairs:
        logger.warning("No pairs were constructed. Check --pair-strategy and that duplicate groups exist.")
    logger.info(f"Wrote {num_pairs} '{args.pair_strategy}' raw pairs to {output_dir}. Run 4_span_alignment.py next.")


if __name__ == "__main__":
    main()
