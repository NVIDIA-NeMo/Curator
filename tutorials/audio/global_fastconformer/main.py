# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run payload-neutral global bucketing with FastConformer."""

import argparse

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.common import ManifestWriterStage
from nemo_curator.stages.audio.global_bucketing import (
    GlobalAudioManifestPlannerStage,
    GlobalAudioParentAssemblerStage,
)
from nemo_curator.stages.audio.inference.asr.global_fastconformer import FastConformerDispatchStage
from nemo_curator.stages.audio.inference.batch_policy import BatchPolicy
from nemo_curator.stages.dispatch_batch import DispatchBatchUnpackStage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    args = parser.parse_args()

    policy = BatchPolicy(
        buckets_sec=[0.0, 30.0, 120.0, 600.0],
        max_items_per_batch_by_bucket=[32, 16, 8, 1],
        max_audio_sec_per_batch=600.0,
    )
    pipeline = Pipeline(
        name="global_fastconformer",
        stages=[
            GlobalAudioManifestPlannerStage(
                manifest_path=args.input_manifest,
                max_model_input_duration_s=600.0,
                batch_policy=policy,
            ),
            FastConformerDispatchStage(batch_policy=policy),
            DispatchBatchUnpackStage(),
            GlobalAudioParentAssemblerStage(),
            ManifestWriterStage(output_path=args.output_manifest),
        ],
    )
    pipeline.run()


if __name__ == "__main__":
    main()
