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

"""Step 4: Rank projected records by query similarity and export the datablend."""

from __future__ import annotations

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.pipeline import Pipeline

from omnifuse_tutorial.stages import DatablendRankingStage, PairManifestReaderStage
from utils import parse_config, print_result


def main() -> None:
    config = parse_config(__doc__)

    projection_records = config.run_dir / "projection" / "records.jsonl"
    if not projection_records.exists():
        raise FileNotFoundError(f"Missing {projection_records}. Run 3_project.py first.")

    # Embed the datablend query with the anchor expert and rank Step 3's projected records.
    pipeline = Pipeline(
        name=f"{config.experiment_id}-4-datablend",
        description="Omni-Fuse step 4: Datablend ranking",
        stages=[
            PairManifestReaderStage(config, records_path=projection_records),
            DatablendRankingStage(config),
        ],
    )
    tasks = pipeline.run(executor=RayDataExecutor())
    if not tasks:
        raise RuntimeError(f"Curator pipeline {pipeline.name} produced no output tasks")

    final_task = tasks[-1]
    print_result(
        {
            "run_dir": str(config.run_dir),
            "datablend_ranked_path": final_task._metadata.get("datablend_ranked_path"),
            "datablend_size": final_task._metadata.get("datablend_size"),
            "datablend_topk_path": final_task._metadata.get("datablend_topk_path"),
        }
    )


if __name__ == "__main__":
    main()
