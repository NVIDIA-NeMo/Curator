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

"""Build distributed scalar indices on a Lance CC dataset.

Uses ``lance_ray.index.create_scalar_index`` which handles fragment distribution,
parallel index segment construction, merging, and the final atomic
``LanceDataset.commit()`` — no custom Ray remote functions needed.

Run after ``build_cc_lancedb.py`` has written the dataset.

Usage:
  python build_lance_index.py \\
      --lance-uri s3://vjawa-cc-lance/cc_url_index \\
      --num-workers 64
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from lance_ray.index import create_scalar_index  # noqa: E402
from loguru import logger  # noqa: E402

from nemo_curator.stages.text.io.writer.utils import s3_credentials_from_env  # noqa: E402

# BTREE: high-cardinality string lookups; BITMAP: low-cardinality filters
_INDEXES: list[tuple[str, str]] = [
    ("cc_url", "BTREE"),
    ("cc_snapshot_id", "BITMAP"),
    ("url_host_name", "BTREE"),
]


def main(args: argparse.Namespace) -> None:
    for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        if not os.environ.get(var):
            logger.error(f"Missing required env var: {var}")
            sys.exit(1)

    storage_options = s3_credentials_from_env()

    for column, index_type in _INDEXES:
        logger.info(f"Building {index_type} index on '{column}' ({args.num_workers} workers)…")
        create_scalar_index(
            uri=args.lance_uri,
            column=column,
            index_type=index_type,
            num_workers=args.num_workers,
            storage_options=storage_options,
        )
        logger.info(f"  ✓ {column} ({index_type})")

    logger.info("All indexes built.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build distributed scalar indices on a Lance CC dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lance-uri", required=True, help="Lance dataset URI (e.g. s3://bucket/dataset).")
    parser.add_argument("--num-workers", type=int, default=64, help="Ray workers for parallel index building.")
    main(parser.parse_args())
