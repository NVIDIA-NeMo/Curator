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

import pytest

from nemo_curator.stages.audio.io.shard_key import derive_manifest_shard_key


def test_corpus_path_produces_stable_relative_key() -> None:
    manifest = "/data/yodas/en/sharded_manifests/manifest_42.jsonl"

    assert derive_manifest_shard_key(manifest, "yodas") == "yodas/en/sharded_manifests/manifest_42"


def test_prefix_keeps_language_and_bucket_identity() -> None:
    manifest = "s3://bucket/es/dataset/bucket_5/manifests/manifest_42.json"
    prefix = "catalog/es/dataset"

    assert (
        derive_manifest_shard_key(
            manifest,
            "catalog",
            shard_key_prefix=prefix,
        )
        == "catalog/es/dataset/bucket_5/manifests/manifest_42"
    )


def test_language_prefixes_do_not_collide() -> None:
    manifest = "s3://bucket/es/dataset/manifests/manifest_42.json"

    es = derive_manifest_shard_key(manifest, "catalog", shard_key_prefix="catalog/es/dataset")
    de = derive_manifest_shard_key(manifest, "catalog", shard_key_prefix="catalog/de/dataset")

    assert es != de


def test_missing_corpus_requires_explicit_prefix() -> None:
    with pytest.raises(ValueError, match="not found"):
        derive_manifest_shard_key("s3://bucket/manifests/one.json", "catalog")
