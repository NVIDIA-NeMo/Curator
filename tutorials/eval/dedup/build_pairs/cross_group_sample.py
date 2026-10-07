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
Helper script for --pair-strategy "cross_group_sample" option of 3_build_pair_dataset.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import json
import random

import pandas as pd
import pyarrow.parquet as pq
from loguru import logger

from build_pairs.utils import _DOCUMENT_COLUMNS
from nemo_curator.stages.deduplication.fuzzy.utils import CURATOR_FUZZY_DUPLICATE_GROUP_FIELD
from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR


def build_cross_group_sample_pairs(  # noqa: C901, PLR0913, PLR0917
    cross_group_samples: int,
    corpus_dir: Path,
    seed: int,
    group_labels_df: pd.DataFrame,
    pairs_per_file: int,
    output_dir: Path,
) -> None:
    sample_size = max(cross_group_samples * 10, 2000)
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
            frames.append(chunk.sample(n=min(file_target, len(chunk)), random_state=seed + file_index))
        documents_df = pd.concat(frames, ignore_index=True)

    logger.info(f"Sampled {len(documents_df)} documents for cross-group negative sampling.")

    group_labels_df = group_labels_df.astype({CURATOR_DEDUP_ID_STR: documents_df[CURATOR_DEDUP_ID_STR].dtype})
    merged = documents_df.merge(group_labels_df, on=CURATOR_DEDUP_ID_STR, how="left")
    rng = random.Random(seed)  # noqa: S311

    pairs = []
    if len(merged) >= 2:  # noqa: PLR2004
        records = merged.to_dict("records")
        attempts = 0
        max_attempts = cross_group_samples * 20 + 100
        seen: set[tuple[int, int]] = set()
        while len(pairs) < cross_group_samples and attempts < max_attempts:
            attempts += 1
            doc_a, doc_b = rng.sample(records, k=2)
            group_a = doc_a.get(CURATOR_FUZZY_DUPLICATE_GROUP_FIELD)
            group_b = doc_b.get(CURATOR_FUZZY_DUPLICATE_GROUP_FIELD)
            if pd.notna(group_a) and pd.notna(group_b) and group_a == group_b:
                continue
            id_a, id_b = doc_a[CURATOR_DEDUP_ID_STR], doc_b[CURATOR_DEDUP_ID_STR]
            key = (min(id_a, id_b), max(id_a, id_b))
            if key in seen:
                continue
            seen.add(key)
            pairs.append(
                {
                    "pair_type": "cross_group_sample",
                    "expected_duplicate": False,
                    "group_id_a": int(group_a) if pd.notna(group_a) else None,
                    "group_id_b": int(group_b) if pd.notna(group_b) else None,
                    "id_a": id_a,
                    "id_b": id_b,
                    "doc_id_a": doc_a.get("url"),
                    "doc_id_b": doc_b.get("url"),
                    # 4_span_alignment.py adds semantic_diff/truncated from these before the judge sees them.
                    "text_a": doc_a.get("text"),
                    "text_b": doc_b.get("text"),
                    "pair_id": f"pair-{len(pairs):07d}",
                }
            )
        if len(pairs) < cross_group_samples:
            logger.warning(
                f"Could only sample {len(pairs)}/{cross_group_samples} cross-group pairs from {len(records)} docs."
            )

    if not pairs:
        logger.warning("No pairs were constructed. Check --pair-strategy and that duplicate groups exist.")

    num_files = 0
    for shard_start in range(0, len(pairs), pairs_per_file):
        shard = pairs[shard_start : shard_start + pairs_per_file]
        part_file = output_dir / f"part_{num_files:05d}.jsonl"
        with part_file.open("w", encoding="utf-8") as f:
            for pair in shard:
                f.write(json.dumps(pair, ensure_ascii=False, default=str) + "\n")
        num_files += 1

    logger.info(
        f"Wrote {len(pairs)} 'cross_group_sample' raw pairs across {num_files} part file(s) to {output_dir}. "
        "Run 4_span_alignment.py next."
    )
