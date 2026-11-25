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

"""Step 2: Run the Expert Embedding Engine (EEE) on SNS-refined records."""

from __future__ import annotations

from omnifuse_tutorial.data import task_records, write_jsonl
from omnifuse_tutorial.stages import EEEEmbeddingStage, PairManifestReaderStage
from utils import parse_config, print_result

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.pipeline import Pipeline


def main() -> None:
    config = parse_config(__doc__)

    sns_records = config.run_dir / "sns" / "records.jsonl"
    if not sns_records.exists():
        msg = f"Missing {sns_records}. Run 1_sns.py first."
        raise FileNotFoundError(msg)

    # Read Step 1's SNS-refined records and compute raw + annotation embeddings for each expert.
    pipeline = Pipeline(
        name=f"{config.experiment_id}-2-embed",
        description="Omni-Fuse step 2: Expert Embedding Engine",
        stages=[
            PairManifestReaderStage(config, records_path=sns_records),
            EEEEmbeddingStage(config),
        ],
    )
    tasks = pipeline.run(executor=RayDataExecutor())
    if not tasks:
        msg = f"Curator pipeline {pipeline.name} produced no output tasks"
        raise RuntimeError(msg)

    # Save the embedded record list for Step 3 (3_project.py).
    final_task = tasks[-1]
    records_path = write_jsonl(config.run_dir / "embeddings" / "records.jsonl", task_records(final_task))
    print_result(
        {
            "run_dir": str(config.run_dir),
            "embedding_metadata_path": final_task._metadata.get("embedding_metadata_path"),
            "embedding_records_path": str(records_path),
        }
    )


if __name__ == "__main__":
    main()
