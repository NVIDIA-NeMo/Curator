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

import argparse

from nemo_curator.backends.experimental.ray_actor_pool import RayActorPoolExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.deduplication.semantic.kmeans import KMeansStage


def main(args: argparse.Namespace) -> None:
    ray_client = RayClient(num_cpus=args.num_cpus, num_gpus=args.num_gpus)
    ray_client.start()

    kmeans_executor = RayActorPoolExecutor()
    pipeline = Pipeline(name="2_clustering")

    kmeans_stage = KMeansStage(
        n_clusters=args.n_clusters,
        id_field=args.id_field,
        embedding_field=args.embedding_field,
        input_path=args.input_path,
        output_path=args.output_path,
        metadata_fields=[args.text_field],
        embedding_dim=args.embedding_dim,
        input_filetype=args.input_filetype,
        verbose=args.verbose,
        max_iter=args.max_iter,
        tol=args.tol,
        random_state=args.random_state,
        init=args.init,
        n_init=args.n_init,
        oversampling_factor=args.oversampling_factor,
        max_samples_per_batch=args.max_samples_per_batch,
    )

    pipeline.add_stage(kmeans_stage)
    pipeline.run(kmeans_executor)

    ray_client.stop()


def attach_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Ray cluster args
    parser.add_argument("--num-cpus", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=None)

    # I/O args
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--input-filetype", type=str, required=True, choices=["parquet", "jsonl"])

    # K-means args
    parser.add_argument("--n-clusters", type=int, default=1000)
    parser.add_argument("--id-field", type=str, default="id")
    parser.add_argument("--embedding-field", type=str, default="embeddings")
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--init", type=str, default="k-means||", choices=["k-means||", "random"])
    parser.add_argument("--n-init", type=int, default=1)
    parser.add_argument("--oversampling-factor", type=float, default=2.0)
    parser.add_argument("--max-samples-per-batch", type=int, default=1 << 15)

    return parser


if __name__ == "__main__":
    main(attach_args().parse_args())
