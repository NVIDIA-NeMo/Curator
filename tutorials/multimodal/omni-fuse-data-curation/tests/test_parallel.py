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

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from omnifuse_tutorial.data import expert_file, make_batch, read_jsonl, write_json, write_jsonl, write_npy
from omnifuse_tutorial.parallel import _merge


def test_merge_keeps_records_manifest_and_embeddings_aligned(tmp_path: Path) -> None:
    config = SimpleNamespace(
        run_dir=tmp_path / "run",
        eee=SimpleNamespace(experts=["fusion"], embedding_dim=2),
        parallelism=SimpleNamespace(records_per_shard=2, sns_workers=1, eee_workers=1, required_gpus=1.0),
    )
    number, tasks = {"a": 1, "b": 2, "c": 3}, []
    for index, ids in enumerate([["a", "b"], ["c"]]):
        shard, records = tmp_path / str(index), [{"pair_id": pair_id, "modality": "text"} for pair_id in ids]
        rows = [row for pair_id in ids for row in ([number[pair_id], 0], [0, number[pair_id]])]
        write_npy(expert_file(shard, "fusion", "interleaved"), rows)
        metadata = {
            "shard_index": index,
            "shard_count": 2,
            "sns_manifest_path": str(write_jsonl(shard / "manifest.jsonl", records)),
            "embedding_errors_path": str(write_json(shard / "errors.json", [])),
            "embeddings_shard_dir": str(shard),
        }
        tasks.append(make_batch("test", records, metadata))

    merged = _merge(config, tasks[::-1])

    for key in ("sns_records_path", "sns_manifest_path", "embedding_records_path"):
        np.testing.assert_array_equal([row["pair_id"] for row in read_jsonl(merged[key])], ["a", "b", "c"])
    embeddings = config.run_dir / "embeddings"
    np.testing.assert_array_equal(np.load(expert_file(embeddings, "fusion", "raw")), [[1, 0], [2, 0], [3, 0]])
    np.testing.assert_array_equal(np.load(expert_file(embeddings, "fusion", "annotation")), [[0, 1], [0, 2], [0, 3]])
    for broken in ([tasks[0], tasks[0]], tasks[1:]):
        with pytest.raises(ValueError, match="unique and contiguous"):
            _merge(config, broken)
