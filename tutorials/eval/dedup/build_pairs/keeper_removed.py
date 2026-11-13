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
Helper script for --pair-strategy "keeper_removed" option of 3_build_pair_dataset.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import ray
import ray.data
from loguru import logger

from build_pairs.utils import _DOCUMENT_COLUMNS, _fetch_documents, _lookup, _pairs_table
from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR

_PAIR_BATCH_ROWS = 20_000


def build_keeper_removed_pairs(
    label_ids: np.ndarray,
    removed_ids: np.ndarray,
    label_groups: np.ndarray,
    corpus_dir: Path,
    counter: ray.actor.ActorHandle,
) -> ray.data.Dataset:
    # Keepers are the grouped docs not marked for removal, sorted by group id.
    is_removed, _ = _lookup(label_ids, removed_ids, removed_ids)
    keeper_ids = label_ids[~is_removed]
    logger.info(f"{len(keeper_ids)} keeper documents across {len(np.unique(label_groups))} duplicate groups.")
    keepers = _fetch_documents(corpus_dir, keeper_ids)
    _, keeper_groups = _lookup(keepers[CURATOR_DEDUP_ID_STR].to_numpy(), label_ids, label_groups)
    keeper_order = np.argsort(keeper_groups, kind="stable")
    keepers, keeper_groups = keepers.take(pa.array(keeper_order)), keeper_groups[keeper_order]

    label_ids_ref, label_groups_ref = ray.put(label_ids), ray.put(label_groups)
    removed_ids_ref, keepers_ref, keeper_groups_ref = (
        ray.put(removed_ids),
        ray.put(keepers),
        ray.put(keeper_groups),
    )

    def _build_keeper_removed_pairs(batch: pa.Table) -> pa.Table:
        batch_label_ids, batch_label_groups = ray.get(label_ids_ref), ray.get(label_groups_ref)
        batch_removed_ids = ray.get(removed_ids_ref)
        batch_keepers, batch_keeper_groups = ray.get(keepers_ref), ray.get(keeper_groups_ref)

        ids = batch[CURATOR_DEDUP_ID_STR].to_numpy()
        in_group, groups = _lookup(ids, batch_label_ids, batch_label_groups)
        batch_is_removed, _ = _lookup(ids, batch_removed_ids, batch_removed_ids)
        mask = in_group & batch_is_removed

        removed_docs = batch.filter(pa.array(mask))
        removed_groups = groups[mask]

        # A group normally has one keeper; if it has several, every keeper pairs with every removed doc.
        first = np.searchsorted(batch_keeper_groups, removed_groups, side="left")
        counts = np.searchsorted(batch_keeper_groups, removed_groups, side="right") - first
        removed_rows = np.repeat(np.arange(len(removed_groups)), counts)
        within_group = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        keeper_rows = np.repeat(first, counts) + within_group

        doc_b = removed_docs.take(pa.array(removed_rows))
        doc_a = batch_keepers.take(pa.array(keeper_rows))
        group_ids = removed_groups[removed_rows]
        pair_ids = pc.binary_join_element_wise(
            pa.scalar("keeper_removed"),
            pc.cast(pa.array(group_ids), pa.string()),
            pc.cast(doc_b[CURATOR_DEDUP_ID_STR], pa.string()),
            pa.scalar("-"),
        )
        pairs = _pairs_table(doc_a, doc_b, group_ids, pair_ids, pair_type="keeper_removed")
        counter.add.remote(pairs.num_rows)
        return pairs

    # Without batch_size each task receives a whole input block and expands it into pairs in one go.
    return ray.data.read_parquet(str(corpus_dir), columns=_DOCUMENT_COLUMNS).map_batches(
        _build_keeper_removed_pairs, batch_format="pyarrow", batch_size=_PAIR_BATCH_ROWS
    )
