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
import functools
import glob
import json
import random
from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow.parquet as pq
import ray.data
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

PairStrategy = Literal["keeper_removed", "all_pairwise", "cross_group_sample"]
_VALID_STRATEGIES = ("keeper_removed", "all_pairwise", "cross_group_sample")
CORPUS_WITH_IDS_SUBDIR = "CorpusWithIds"
_DOCUMENT_COLUMNS = [CURATOR_DEDUP_ID_STR, "url", "text"]
_PAIR_ROW_COLUMNS = [
    "pair_type",
    "expected_duplicate",
    "group_id_a",
    "group_id_b",
    "id_a",
    "id_b",
    "doc_id_a",
    "doc_id_b",
    "text_a",
    "text_b",
    "pair_id",
]


def _reassign_ids_to_parquet(  # noqa: PLR0913
    *,
    input_path: str,
    input_filetype: str,
    input_blocksize: str,
    id_generator_path: str,
    cache_dir: Path,
    ray_temp_dir: str,
) -> Path:
    corpus_dir = cache_dir / CORPUS_WITH_IDS_SUBDIR
    metadata_file = corpus_dir / "_reassignment_metadata.json"
    metadata = {"input_path": input_path, "input_filetype": input_filetype, "input_blocksize": input_blocksize}

    if corpus_dir.exists() and any(corpus_dir.glob("*.parquet")):
        if not metadata_file.exists():
            msg = (
                f"{corpus_dir} has Parquet files but no {metadata_file.name} to verify they match "
                "--input-path/--input-filetype/--input-blocksize. Delete it and re-run."
            )
            raise RuntimeError(msg)

        cached_metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        if cached_metadata != metadata:
            msg = (
                f"{corpus_dir} was built from different inputs ({cached_metadata}) than this run "
                f"requested ({metadata}). Delete {corpus_dir} to rebuild it, or point --cache-dir at "
                "a fresh directory."
            )
            raise RuntimeError(msg)

        logger.info(f"Reusing already-written {corpus_dir} (delete it to force a re-read).")
        return corpus_dir

    reader_class = ParquetReader if input_filetype == "parquet" else JsonlReader
    pipeline = Pipeline(
        name="dedup_eval_id_reassignment",
        description="Re-read the original corpus with the same _curator_dedup_id values fuzzy dedup assigned.",
        stages=[
            reader_class(file_paths=input_path, blocksize=input_blocksize, _assign_ids=True, fields=["url", "text"]),
            ParquetWriter(path=str(corpus_dir)),
        ],
    )

    ray_client = RayClient(ray_temp_dir=ray_temp_dir)
    ray_client.start()
    try:
        create_id_generator_actor(id_generator_path)
        try:
            pipeline.run()
        finally:
            kill_id_generator_actor()
    finally:
        ray_client.stop()

    if not any(corpus_dir.glob("*.parquet")):
        msg = (
            f"No documents were read back from {input_path!r}; check --input-path/--input-filetype/--input-blocksize."
        )
        raise RuntimeError(msg)

    metadata_file.write_text(json.dumps(metadata), encoding="utf-8")
    return corpus_dir


def _read_parquet_glob(directory: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(Path(directory) / "*.parquet")))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _make_pair_row(doc_a: dict, doc_b: dict, *, pair_type: str, expected_duplicate: bool) -> dict:
    group_id_a = doc_a.get(CURATOR_FUZZY_DUPLICATE_GROUP_FIELD)
    group_id_b = doc_b.get(CURATOR_FUZZY_DUPLICATE_GROUP_FIELD)

    return {
        "pair_type": pair_type,
        "expected_duplicate": expected_duplicate,
        "group_id_a": int(group_id_a) if pd.notna(group_id_a) else None,
        "group_id_b": int(group_id_b) if pd.notna(group_id_b) else None,
        "id_a": doc_a[CURATOR_DEDUP_ID_STR],
        "id_b": doc_b[CURATOR_DEDUP_ID_STR],
        "doc_id_a": doc_a.get("url"),
        "doc_id_b": doc_b.get("url"),
        # 4_span_alignment.py adds semantic_diff/truncated from these before the judge sees them.
        "text_a": doc_a.get("text"),
        "text_b": doc_b.get("text"),
    }


def run_grouped_strategy(  # noqa: C901
    args: argparse.Namespace,
    *,
    corpus_dir: Path,
    group_labels_df: pd.DataFrame,
    removed_ids: set[int],
    output_dir: Path,
) -> None:
    """CLI entry point for --pair-strategy keeper_removed/all_pairwise."""

    ray_client = RayClient(ray_temp_dir=args.ray_temp_dir)
    ray_client.start()
    try:
        group_by_id = dict(
            zip(
                group_labels_df[CURATOR_DEDUP_ID_STR].tolist(),
                group_labels_df[CURATOR_FUZZY_DUPLICATE_GROUP_FIELD].tolist(),
                strict=True,
            )
        )

        def _attach_group_id(batch: pd.DataFrame, *, group_by_id: dict[int, int]) -> pd.DataFrame:
            batch = batch.copy()
            batch[CURATOR_FUZZY_DUPLICATE_GROUP_FIELD] = batch[CURATOR_DEDUP_ID_STR].map(group_by_id)
            return batch

        documents_ds = ray.data.read_parquet(str(corpus_dir), columns=_DOCUMENT_COLUMNS)
        tagged_ds = documents_ds.map_batches(
            functools.partial(_attach_group_id, group_by_id=group_by_id), batch_format="pandas"
        )
        grouped_ds = tagged_ds.filter(lambda row: pd.notna(row[CURATOR_FUZZY_DUPLICATE_GROUP_FIELD]))

        def _build_pairs_for_group(
            group_df: pd.DataFrame, *, strategy: PairStrategy, removed_ids: set[int], max_group_size: int
        ) -> pd.DataFrame:
            group_id = int(group_df[CURATOR_FUZZY_DUPLICATE_GROUP_FIELD].iloc[0])
            if strategy == "keeper_removed":
                keepers = group_df[~group_df[CURATOR_DEDUP_ID_STR].isin(removed_ids)]
                removed = group_df[group_df[CURATOR_DEDUP_ID_STR].isin(removed_ids)]
                pairs = []
                for _, keeper_row in keepers.iterrows():
                    for _, removed_row in removed.iterrows():
                        pairs.append(
                            _make_pair_row(
                                keeper_row, removed_row, pair_type="keeper_removed", expected_duplicate=True
                            )
                        )
            elif len(group_df) > max_group_size:
                logger.warning(
                    f"Skipping duplicate group {group_id} ({len(group_df)} documents) for all_pairwise -- "
                    f"larger than --max-group-size={max_group_size}."
                )
                pairs = []
            else:
                rows = group_df.to_dict("records")
                pairs = []
                for i in range(len(rows)):
                    for j in range(i + 1, len(rows)):
                        pairs.append(
                            _make_pair_row(rows[i], rows[j], pair_type="all_pairwise", expected_duplicate=True)
                        )
            for index, pair in enumerate(pairs):
                pair["pair_id"] = f"{strategy}-{group_id}-{index:04d}"
            return pd.DataFrame(pairs, columns=_PAIR_ROW_COLUMNS)

        paired_ds = grouped_ds.groupby(CURATOR_FUZZY_DUPLICATE_GROUP_FIELD).map_groups(
            functools.partial(
                _build_pairs_for_group,
                strategy=args.pair_strategy,
                removed_ids=removed_ids,
                max_group_size=args.max_group_size,
            ),
            batch_format="pandas",
        )
        # materialize() runs the pipeline once; without it, write_json()/count() would each redo it.
        paired_ds = paired_ds.materialize()
        paired_ds.write_json(str(output_dir))
        num_pairs = paired_ds.count()
    finally:
        ray_client.stop()

    if not num_pairs:
        logger.warning("No pairs were constructed. Check --pair-strategy and that duplicate groups exist.")
    logger.info(f"Wrote {num_pairs} '{args.pair_strategy}' raw pairs to {output_dir}. Run 4_span_alignment.py next.")


def run_cross_group_strategy(  # noqa: C901, PLR0912, PLR0915
    args: argparse.Namespace, *, corpus_dir: Path, group_labels_df: pd.DataFrame, output_dir: Path
) -> None:
    """CLI entry point for --pair-strategy cross_group_sample."""

    sample_size = max(args.cross_group_samples * 10, 2000)
    files = sorted(corpus_dir.glob("*.parquet"))

    row_counts = {file: pq.ParquetFile(file).metadata.num_rows for file in files}
    total_rows = sum(row_counts.values())
    if total_rows == 0:
        documents_df = pd.DataFrame(columns=_DOCUMENT_COLUMNS)
    else:
        frames = []
        for file_index, (file, count) in enumerate(row_counts.items()):
            if count == 0:
                continue
            file_target = max(1, round(sample_size * count / total_rows))
            chunk = pd.read_parquet(file, columns=_DOCUMENT_COLUMNS)
            # Vary the seed per file; otherwise every file would sample the same row positions.
            frames.append(chunk.sample(n=min(file_target, len(chunk)), random_state=args.seed + file_index))
        documents_df = pd.concat(frames, ignore_index=True)

    logger.info(f"Sampled {len(documents_df)} documents for cross-group negative sampling.")

    group_labels_df = group_labels_df.astype({CURATOR_DEDUP_ID_STR: documents_df[CURATOR_DEDUP_ID_STR].dtype})
    merged = documents_df.merge(group_labels_df, on=CURATOR_DEDUP_ID_STR, how="left")
    rng = random.Random(args.seed)  # noqa: S311

    if len(merged) < 2:  # noqa: PLR2004
        pairs = []
    else:
        records = merged.to_dict("records")
        pairs = []
        attempts = 0
        max_attempts = args.cross_group_samples * 20 + 100
        seen: set[tuple[int, int]] = set()
        while len(pairs) < args.cross_group_samples and attempts < max_attempts:
            attempts += 1
            doc_a, doc_b = rng.sample(records, k=2)
            group_a = doc_a.get(CURATOR_FUZZY_DUPLICATE_GROUP_FIELD)
            group_b = doc_b.get(CURATOR_FUZZY_DUPLICATE_GROUP_FIELD)
            same_group = pd.notna(group_a) and pd.notna(group_b) and group_a == group_b
            if same_group:
                continue
            id_a, id_b = doc_a[CURATOR_DEDUP_ID_STR], doc_b[CURATOR_DEDUP_ID_STR]
            key = (min(id_a, id_b), max(id_a, id_b))
            if key in seen:
                continue
            seen.add(key)
            pairs.append(_make_pair_row(doc_a, doc_b, pair_type="cross_group_sample", expected_duplicate=False))
        if len(pairs) < args.cross_group_samples:
            logger.warning(
                f"Could only sample {len(pairs)}/{args.cross_group_samples} cross-group pairs from {len(records)} docs."
            )

    for idx, pair in enumerate(pairs):
        pair["pair_id"] = f"pair-{idx:07d}"

    if not pairs:
        logger.warning("No pairs were constructed. Check --pair-strategy and that duplicate groups exist.")

    num_files = 0
    for shard_start in range(0, len(pairs), args.pairs_per_file):
        shard = pairs[shard_start : shard_start + args.pairs_per_file]
        part_file = output_dir / f"part_{num_files:05d}.jsonl"
        with part_file.open("w", encoding="utf-8") as f:
            for pair in shard:
                f.write(json.dumps(pair, ensure_ascii=False, default=str) + "\n")
        num_files += 1

    logger.info(
        f"Wrote {len(pairs)} '{args.pair_strategy}' raw pairs across {num_files} part file(s) to {output_dir}. "
        "Run 4_span_alignment.py next."
    )


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
        "--max-group-size", type=int, default=200, help="Skip 'all_pairwise' groups larger than this (C(n,2) pairs)."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--pairs-per-file",
        type=int,
        default=2000,
        help="Max pairs per output file. Only applies to 'cross_group_sample'; "
        "the other strategies are sharded by Ray Data instead.",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cache_dir = Path(args.cache_dir)

    id_generator_path = str(Path(args.fuzzy_output_dir) / ID_GENERATOR_OUTPUT_FILENAME)
    logger.info(f"Re-reading corpus from {args.input_path} with ids replayed from {id_generator_path}...")
    corpus_dir = _reassign_ids_to_parquet(
        input_path=args.input_path,
        input_filetype=args.input_filetype,
        input_blocksize=args.input_blocksize,
        id_generator_path=id_generator_path,
        cache_dir=cache_dir,
        ray_temp_dir=args.ray_temp_dir,
    )

    connected_components_dir = cache_dir / "ConnectedComponentsStage"
    group_labels_df = _read_parquet_glob(str(connected_components_dir))
    if group_labels_df.empty:
        msg = (
            f"No group labels found under {connected_components_dir}. Either no fuzzy duplicates were found "
            "by 2_run_fuzzy_dedup.py, or --cache-dir does not match the value passed to it."
        )
        raise RuntimeError(msg)
    logger.info(f"Loaded group labels for {len(group_labels_df)} documents across duplicate groups.")

    duplicate_ids_dir = Path(args.fuzzy_output_dir) / DUPLICATE_IDS_SUBDIR
    removal_df = _read_parquet_glob(str(duplicate_ids_dir))
    removed_ids = set(removal_df[CURATOR_DEDUP_ID_STR].tolist()) if not removal_df.empty else set()
    logger.info(f"Loaded {len(removed_ids)} ids fuzzy dedup marked for removal.")

    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.pair_strategy in ("keeper_removed", "all_pairwise"):
        run_grouped_strategy(
            args,
            corpus_dir=corpus_dir,
            group_labels_df=group_labels_df,
            removed_ids=removed_ids,
            output_dir=output_dir,
        )
    else:
        run_cross_group_strategy(args, corpus_dir=corpus_dir, group_labels_df=group_labels_df, output_dir=output_dir)


if __name__ == "__main__":
    main()
