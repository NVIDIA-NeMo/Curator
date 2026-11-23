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

"""Step 1: Run Symmetric Nucleus Subsampling (SNS) over paired multimodal records."""

from __future__ import annotations

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.pipeline import Pipeline

from omnifuse_tutorial.data import task_records, write_jsonl
from omnifuse_tutorial.stages import PairManifestReaderStage, SNSStage
from utils import parse_config, print_result


def main() -> None:
    config = parse_config(__doc__)

    # Read paired records from the configured data pools and refine each pair with SNS.
    pipeline = Pipeline(
        name=f"{config.experiment_id}-1-sns",
        description="Omni-Fuse step 1: Symmetric Nucleus Subsampling",
        stages=[
            PairManifestReaderStage(config),
            SNSStage(config),
        ],
    )
    tasks = pipeline.run(executor=RayDataExecutor())
    if not tasks:
        raise RuntimeError(f"Curator pipeline {pipeline.name} produced no output tasks")

    # Save the SNS-refined records for Step 2 (2_embed.py).
    final_task = tasks[-1]
    records_path = write_jsonl(config.run_dir / "sns" / "records.jsonl", task_records(final_task))
    print_result(
        {
            "run_dir": str(config.run_dir),
            "sns_manifest_path": final_task._metadata.get("sns_manifest_path"),
            "sns_records_path": str(records_path),
        }
    )


if __name__ == "__main__":
    main()
