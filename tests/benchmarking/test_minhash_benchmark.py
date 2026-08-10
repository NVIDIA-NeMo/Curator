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

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parents[2] / "benchmarking" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from minhash_benchmark import _build_pipeline  # noqa: E402

from nemo_curator.backends.utils import RayStageSpecKeys  # noqa: E402


def test_reader_worker_bounds_are_applied_to_reader_stage(tmp_path: Path) -> None:
    pipeline = _build_pipeline(
        input_files=["input.parquet"],
        output_path=tmp_path,
        input_task_type="DocumentBatch",
        input_filetype="parquet",
        files_per_partition=None,
        blocksize="1GiB",
        text_field="raw_content",
        minhash_field="_minhash_signature",
        char_ngrams=24,
        num_hashes=260,
        seed=42,
        use_64bit_hash=False,
        read_kwargs={},
        write_kwargs={},
        pool=True,
        reader_ray_data_initial_workers=12,
        reader_ray_data_max_workers=12,
        reader_ray_data_num_cpus=2,
        minhash_num_workers=8,
        minhash_ray_data_initial_workers=None,
        minhash_ray_data_num_cpus=2,
        minhash_ray_data_max_concurrency=2,
        minhash_ray_data_max_tasks_in_flight_per_actor=None,
    )

    pipeline.build()

    reader_stage = next(stage for stage in pipeline.stages if stage.name == "parquet_reader")
    minhash_stage = next(stage for stage in pipeline.stages if stage.name == "MinHashStage")
    assert reader_stage.ray_stage_spec()[RayStageSpecKeys.INITIAL_WORKERS] == 12
    assert reader_stage.ray_stage_spec()[RayStageSpecKeys.MAX_WORKERS] == 12
    assert reader_stage.ray_stage_spec()[RayStageSpecKeys.RAY_NUM_CPUS] == 2
    assert minhash_stage.num_workers() == 8
    assert minhash_stage.ray_stage_spec()[RayStageSpecKeys.RAY_NUM_CPUS] == 2
    assert minhash_stage.ray_stage_spec()[RayStageSpecKeys.MAX_CONCURRENCY] == 2
