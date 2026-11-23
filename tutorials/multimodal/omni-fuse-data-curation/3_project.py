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

"""Step 3: Train the Omni-Fuse projection MLP and project concatenated expert embeddings."""

from __future__ import annotations

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.pipeline import Pipeline

from omnifuse_tutorial.data import task_records, write_jsonl
from omnifuse_tutorial.stages import PairManifestReaderStage, ProjectionTrainingStage
from utils import parse_config, print_result


def main() -> None:
    config = parse_config(__doc__)

    embedding_records = config.run_dir / "embeddings" / "records.jsonl"
    if not embedding_records.exists():
        raise FileNotFoundError(f"Missing {embedding_records}. Run 2_embed.py first.")

    # Train the projection MLP over Step 2's expert embeddings and save projected vectors.
    pipeline = Pipeline(
        name=f"{config.experiment_id}-3-project",
        description="Omni-Fuse step 3: Projection training",
        stages=[
            PairManifestReaderStage(config, records_path=embedding_records),
            ProjectionTrainingStage(config),
        ],
    )
    tasks = pipeline.run(executor=RayDataExecutor())
    if not tasks:
        raise RuntimeError(f"Curator pipeline {pipeline.name} produced no output tasks")

    # Save the record list for Step 4 (4_datablend.py).
    final_task = tasks[-1]
    records_path = write_jsonl(config.run_dir / "projection" / "records.jsonl", task_records(final_task))
    print_result(
        {
            "run_dir": str(config.run_dir),
            "projected_embeddings_path": final_task._metadata.get("projected_embeddings_path"),
            "projection_metrics_path": final_task._metadata.get("projection_metrics_path"),
            "projection_model_path": final_task._metadata.get("projection_model_path"),
            "projection_records_path": str(records_path),
        }
    )


if __name__ == "__main__":
    main()
