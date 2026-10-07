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
Helper script for --pair-strategy "all_pairwise" option of 3_build_pair_dataset.py
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
from nemo_curator.stages.deduplication.fuzzy.utils import CURATOR_FUZZY_DUPLICATE_GROUP_FIELD
from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR


def build_all_pairwise_pairs(  # noqa: PLR0913, PLR0917
    label_groups: np.ndarray,
    max_group_size: int,
    oversized_group_pairs: int,
    label_ids: np.ndarray,
    counter: ray.actor.ActorHandle,
    corpus_dir: Path,
) -> ray.data.Dataset:
    # Drop groups that cannot yield pairs before the shuffle so their text is never moved.
    unique_groups, group_sizes = np.unique(label_groups, return_counts=True)
    too_large = group_sizes > max_group_size
    logger.info(
        f"{int(too_large.sum())} duplicate groups are larger than --max-group-size={max_group_size}; "
        f"each contributes at most {oversized_group_pairs} sampled pairs."
    )
    keep_groups = unique_groups[~too_large & (group_sizes >= 2)]  # noqa: PLR2004
    keep = np.isin(label_groups, keep_groups)
    tag_label_ids_ref, tag_label_groups_ref = ray.put(label_ids[keep]), ray.put(label_groups[keep])

    def tag_group_id(batch: pa.Table) -> pa.Table:
        tag_ids, tag_groups = ray.get(tag_label_ids_ref), ray.get(tag_label_groups_ref)
        found, groups = _lookup(batch[CURATOR_DEDUP_ID_STR].to_numpy(), tag_ids, tag_groups)
        return batch.filter(pa.array(found)).append_column(
            CURATOR_FUZZY_DUPLICATE_GROUP_FIELD, pa.array(groups[found], type=pa.int64())
        )

    def _build_all_pairwise_pairs(group: pa.Table) -> pa.Table:
        group_id = group[CURATOR_FUZZY_DUPLICATE_GROUP_FIELD][0].as_py()
        first, second = np.triu_indices(group.num_rows, k=1)
        index = pc.cast(pa.array(np.arange(len(first))), pa.string())
        pair_ids = pc.binary_join_element_wise(
            pa.scalar(f"all_pairwise-{group_id}"), pc.utf8_lpad(index, 4, "0"), pa.scalar("-")
        )
        pairs = _pairs_table(
            group.take(pa.array(first)),
            group.take(pa.array(second)),
            np.full(len(first), group_id, dtype=np.int64),
            pair_ids,
            pair_type="all_pairwise",
        )
        counter.add.remote(pairs.num_rows)
        return pairs

    tagged_ds = ray.data.read_parquet(str(corpus_dir), columns=_DOCUMENT_COLUMNS).map_batches(
        tag_group_id, batch_format="pyarrow"
    )
    return tagged_ds.groupby(CURATOR_FUZZY_DUPLICATE_GROUP_FIELD).map_groups(
        _build_all_pairwise_pairs, batch_format="pyarrow"
    )


def build_oversized_group_pairs(  # noqa: PLR0913, PLR0917
    label_ids: np.ndarray,
    label_groups: np.ndarray,
    max_group_size: int,
    seed: int,
    oversized_group_pairs: int,
    corpus_dir: Path,
    output_dir: Path,
) -> int:
    # Draw up to --oversized-group-pairs random distinct pairs from every group larger than --max-group-size.
    if label_ids.max() >= 1 << 31:
        msg = "Document ids must be below 2**31 to deduplicate sampled pairs."
        raise ValueError(msg)

    group_order = np.argsort(label_groups, kind="stable")
    members, member_groups = label_ids[group_order], label_groups[group_order]
    sampled_groups, starts, sizes = np.unique(member_groups, return_index=True, return_counts=True)
    oversized = sizes > max_group_size
    sampled_groups, starts, sizes = sampled_groups[oversized], starts[oversized], sizes[oversized]

    rng = np.random.default_rng(seed)
    size = np.repeat(sizes, oversized_group_pairs)
    start = np.repeat(starts, oversized_group_pairs)
    group = np.repeat(sampled_groups, oversized_group_pairs)
    first = (rng.random(len(size)) * size).astype(np.int64)
    second = (rng.random(len(size)) * (size - 1)).astype(np.int64)
    second += second >= first  # Shift past `first` so the two members are always distinct.

    member_a, member_b = members[start + first], members[start + second]
    low, high = np.minimum(member_a, member_b), np.maximum(member_a, member_b)
    _, unique_rows = np.unique((low << 32) | high, return_index=True)
    unique_rows.sort()
    id_a, id_b, groups = low[unique_rows], high[unique_rows], group[unique_rows]

    if len(id_a) > 0:
        documents = _fetch_documents(corpus_dir, np.unique(np.concatenate([id_a, id_b])))
        document_ids = documents[CURATOR_DEDUP_ID_STR].to_numpy()
        document_order = np.argsort(document_ids)
        documents, document_ids = documents.take(pa.array(document_order)), document_ids[document_order]

        # `groups` is non-decreasing, so a pair's index in its group is its offset from the group's first row.
        index_in_group = np.arange(len(groups)) - np.searchsorted(groups, groups, side="left")
        pair_ids = pc.binary_join_element_wise(
            pa.scalar("all_pairwise"),
            pc.cast(pa.array(groups), pa.string()),
            pc.utf8_lpad(pc.cast(pa.array(index_in_group), pa.string()), 4, "0"),
            pa.scalar("-"),
        )
        oversized_pairs = _pairs_table(
            documents.take(pa.array(np.searchsorted(document_ids, id_a))),
            documents.take(pa.array(np.searchsorted(document_ids, id_b))),
            groups,
            pair_ids,
            pair_type="all_pairwise",
        )
        ray.data.from_arrow(oversized_pairs).write_json(str(output_dir))
        return oversized_pairs.num_rows

    return 0
