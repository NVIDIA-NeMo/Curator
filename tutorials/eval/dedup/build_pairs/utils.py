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
Utility functions for building pairs.
"""

from pathlib import Path

import numpy as np
import pyarrow as pa
import ray
import ray.data

from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR

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


def _lookup(values: np.ndarray, sorted_keys: np.ndarray, sorted_payload: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(sorted_keys) == 0:
        return np.zeros(len(values), dtype=bool), np.zeros(len(values), dtype=sorted_payload.dtype)
    positions = np.minimum(np.searchsorted(sorted_keys, values), len(sorted_keys) - 1)
    return sorted_keys[positions] == values, sorted_payload[positions]


def _pairs_table(
    doc_a: pa.Table,
    doc_b: pa.Table,
    group_ids: np.ndarray,
    pair_ids: pa.Array,
    *,
    pair_type: str,
) -> pa.Table:
    n = doc_a.num_rows
    groups = pa.array(group_ids, type=pa.int64())
    columns = {
        "pair_type": pa.repeat(pair_type, n),
        "expected_duplicate": pa.repeat(True, n),
        "group_id_a": groups,
        "group_id_b": groups,
        "id_a": doc_a[CURATOR_DEDUP_ID_STR],
        "id_b": doc_b[CURATOR_DEDUP_ID_STR],
        "doc_id_a": doc_a["url"],
        "doc_id_b": doc_b["url"],
        # 4_span_alignment.py adds semantic_diff/truncated from these before the judge sees them.
        "text_a": doc_a["text"],
        "text_b": doc_b["text"],
        "pair_id": pair_ids,
    }
    return pa.table(columns).select(_PAIR_ROW_COLUMNS)


def _fetch_documents(corpus_dir: Path, sorted_ids: np.ndarray) -> pa.Table:
    ids_ref = ray.put(sorted_ids)

    def select_ids(batch: pa.Table) -> pa.Table:
        ids = ray.get(ids_ref)
        found, _ = _lookup(batch[CURATOR_DEDUP_ID_STR].to_numpy(), ids, ids)
        return batch.filter(pa.array(found))

    documents_ds = ray.data.read_parquet(str(corpus_dir), columns=_DOCUMENT_COLUMNS).map_batches(
        select_ids, batch_format="pyarrow"
    )
    blocks = ray.get(documents_ds.to_arrow_refs())
    return pa.concat_tables(blocks) if blocks else pa.table({name: [] for name in _DOCUMENT_COLUMNS})
