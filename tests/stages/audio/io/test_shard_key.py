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

from __future__ import annotations

import pytest

from nemo_curator.stages.audio.io.shard_key import derive_manifest_shard_key, validate_shard_key


def test_derives_corpus_relative_key_and_strips_longest_manifest_suffix() -> None:
    manifest = "s3://speech/datasets/yodas/en/manifests/manifest_0007.jsonl.gz"

    assert derive_manifest_shard_key(manifest, "YODAS") == "yodas/en/manifests/manifest_0007"


def test_prefix_keeps_distinguishing_tail_after_last_anchor() -> None:
    manifest = "s3://speech/archive/dataset-id/old/dataset-id/bucket_3/manifest_0042.json"
    prefix = "catalog/en/dataset-id"

    assert derive_manifest_shard_key(manifest, "logical-corpus", shard_key_prefix=prefix) == (
        "catalog/en/dataset-id/bucket_3/manifest_0042"
    )


def test_missing_prefix_anchor_falls_back_to_manifest_basename() -> None:
    assert (
        derive_manifest_shard_key(
            "s3://speech/en/manifest_0042.jsonl",
            "logical-corpus",
            shard_key_prefix="catalog/en/dataset-id",
        )
        == "catalog/en/dataset-id/manifest_0042"
    )


@pytest.mark.parametrize(
    "shard_key",
    [
        "",
        ".",
        "..",
        "../outside",
        "catalog/../../outside",
        "/absolute/path",
        ".nemo_curator/receipts",
    ],
)
def test_rejects_empty_absolute_and_traversing_shard_keys(shard_key: str) -> None:
    with pytest.raises(ValueError, match="Unsafe NeMo shard key"):
        validate_shard_key(shard_key)


def test_normalizes_windows_separators() -> None:
    assert validate_shard_key(r"catalog\en\manifest_0") == "catalog/en/manifest_0"


def test_rejects_traversal_from_prefix_or_manifest_tail() -> None:
    with pytest.raises(ValueError, match="Unsafe NeMo shard key"):
        derive_manifest_shard_key(
            "s3://speech/dataset/manifest.jsonl",
            "dataset",
            shard_key_prefix="catalog/../outside",
        )

    with pytest.raises(ValueError, match="Unsafe NeMo shard key"):
        derive_manifest_shard_key("/data/dataset/../outside.jsonl", "dataset")


@pytest.mark.parametrize(
    ("manifest", "corpus", "message"),
    [
        ("/data/other/manifest.jsonl", "dataset", "not found"),
        ("/data/dataset/copy/dataset/manifest.jsonl", "dataset", "found 2 times"),
        ("/data/dataset/manifest.jsonl", "", "corpus must be non-empty"),
    ],
)
def test_corpus_anchor_must_be_unambiguous(manifest: str, corpus: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        derive_manifest_shard_key(manifest, corpus)
