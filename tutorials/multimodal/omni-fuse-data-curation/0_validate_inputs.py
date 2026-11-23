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

"""Step 0: Validate the tutorial config, data manifests, API key, and local model assets."""

from __future__ import annotations

from omnifuse_tutorial.config import languagebind_root
from omnifuse_tutorial.data import load_records
from utils import parse_config, print_result


def main() -> None:
    config = parse_config(__doc__)

    # Load and validate every configured data pool manifest.
    records = load_records(config.data_pools)

    # Check required API credentials and local model assets before launching GPU jobs.
    missing: list[str] = []
    if config.eee.needs_api_key:
        missing.append("NV_BUILD_API_KEY")
    if "fusion" in config.eee.experts and not languagebind_root().exists():
        missing.append("LANGUAGEBIND_ROOT")

    needs_video_forward = config.sns.enabled and config.sns.direction in {"forward", "bidirectional"}
    has_video_pool = any(pool.modality == "video" for pool in config.data_pools)
    if needs_video_forward and has_video_pool and not config.sns.cg_detr_checkpoint.exists():
        missing.append(str(config.sns.cg_detr_checkpoint))

    if missing:
        raise RuntimeError("Missing required API keys or local assets: " + ", ".join(missing))

    parallel = config.parallelism
    print_result(
        {
            "experiment_id": config.experiment_id,
            "run_dir": str(config.run_dir),
            "records": len(records),
            "modalities": sorted({record["modality"] for record in records}),
            "sns_embedding_batch_size": config.sns.embedding_batch_size,
            "experts": config.eee.experts,
            "parallelism": {
                "enabled": parallel.enabled,
                "records_per_shard": parallel.records_per_shard,
                "sns_workers": parallel.sns_workers,
                "eee_workers": parallel.eee_workers,
            },
        }
    )


if __name__ == "__main__":
    main()
