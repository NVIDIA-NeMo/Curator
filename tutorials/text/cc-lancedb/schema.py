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

"""Canonical schema for the unified CC → Lance dataset.

All 121 CC-MAIN snapshots (2013-2026) write to a single Lance URI.
snapshot_id carries a BITMAP index for instant per-snapshot filtering.
content is stored as a lance blob column (large_binary).
"""

import pyarrow as pa

CC_LANCE_SCHEMA = pa.schema(
    [
        pa.field("snapshot_id", pa.string(), nullable=True),  # nullable: None when iterator has no snapshot_id
        pa.field("url", pa.string()),  # https://example.com/page
        pa.field("warc_id", pa.string()),  # <urn:uuid:...>
        pa.field("source_id", pa.string()),  # CC-MAIN-...-00000.warc.gz
        pa.field("content", pa.large_binary()),  # raw HTML bytes (lance blob)
        pa.field("cc_extracted_text_trafilatura", pa.large_string()),
        pa.field("cc_extracted_text_justext", pa.large_string()),
        pa.field("cc_extracted_text_resiliparse", pa.large_string()),
    ]
)
