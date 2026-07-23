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

"""Sharded SQLite URL sidecars for Lance image address resolution."""

from nemo_curator.stages.interleaved.lance.sidecar.builder import build_sharded_sqlite_url_lance_sidecar
from nemo_curator.stages.interleaved.lance.sidecar.format import (
    decode_rowaddr,
    decode_uint64,
    encode_uint64,
    hash_url,
    read_lance_url_sidecar_manifest,
    shard_for_digest,
)
from nemo_curator.stages.interleaved.lance.sidecar.resolver import (
    LanceUrlSidecarCoordinate,
    ShardedSqliteUrlLanceAddressResolutionStage,
)

__all__ = [
    "LanceUrlSidecarCoordinate",
    "ShardedSqliteUrlLanceAddressResolutionStage",
    "build_sharded_sqlite_url_lance_sidecar",
    "decode_rowaddr",
    "decode_uint64",
    "encode_uint64",
    "hash_url",
    "read_lance_url_sidecar_manifest",
    "shard_for_digest",
]
