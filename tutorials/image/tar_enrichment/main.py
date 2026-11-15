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

"""Annotate images that live inside TARs, located by a Parquet catalog.

Each catalog row says where one image is: TAR path, byte offset and byte size.
The pipeline fetches images with ranged reads, decodes and measures them on CPU,
and writes one row per input image with CLIP embedding, aesthetic and NSFW scores.
No image is dropped: failures keep their row with null scores and a status.
"""

from __future__ import annotations

import argparse
import time

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.image.embedders.clip_embedder import ImageEmbeddingStage
from nemo_curator.stages.image.filters.aesthetic_filter import ImageAestheticFilterStage
from nemo_curator.stages.image.filters.nsfw_filter import ImageNSFWFilterStage
from nemo_curator.stages.image.io.convert import ConvertImageBatchToDocumentBatchStage
from nemo_curator.stages.image.io.tar_range_reader import ImageTarRangeReader
from nemo_curator.stages.image.measurements import ImageMeasurementStage
from nemo_curator.stages.image.preprocess import ImageDecodeStage, ImageResizeStage
from nemo_curator.stages.text.io.reader.parquet import ParquetReader
from nemo_curator.stages.text.io.writer.parquet import ParquetWriter


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    columns = {
        "tar_column": args.tar_column,
        "offset_column": args.offset_column,
        "size_column": args.size_column,
        "id_column": args.id_column,
    }
    storage_options = {"client_kwargs": {"endpoint_url": args.endpoint_url}} if args.endpoint_url else {}
    cpu = {"num_workers": args.cpu_workers}
    return Pipeline(
        name="image_tar_enrichment",
        stages=[
            ParquetReader(file_paths=args.input, fields=list(columns.values())),
            ImageTarRangeReader(**columns, storage_options=storage_options, fetch_workers=args.fetch_workers),
            # Decode, measure and shrink on CPU so GPUs only receive 224x224 uint8 pixels.
            ImageDecodeStage().with_(**cpu),
            ImageMeasurementStage().with_(**cpu),
            ImageResizeStage().with_(**cpu),
            ImageEmbeddingStage(
                model_dir=args.model_dir,
                num_gpus_per_worker=0.5,
                model_inference_batch_size=128,
                remove_image_data=True,
            ).with_(num_workers=args.clip_workers),
            # score_threshold=None annotates every image instead of filtering.
            ImageAestheticFilterStage(
                model_dir=args.model_dir, num_gpus_per_worker=0.5, model_inference_batch_size=128, score_threshold=None
            ),
            ImageNSFWFilterStage(
                model_dir=args.model_dir, num_gpus_per_worker=0.5, model_inference_batch_size=128, score_threshold=None
            ),
            ConvertImageBatchToDocumentBatchStage(
                fields=["image_id", "metadata", "embedding", "aesthetic_score", "nsfw_score"]
            ),
            ParquetWriter(path=args.output, write_kwargs={"compression": "zstd"}),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Catalog Parquet file or directory")
    parser.add_argument("--output", required=True, help="Output directory for annotated Parquet")
    parser.add_argument("--model-dir", required=True, help="Directory holding the CLIP, aesthetic and NSFW weights")
    parser.add_argument("--endpoint-url", help="S3-compatible endpoint; credentials come from the AWS environment")
    parser.add_argument("--tar-column", default="tar_path")
    parser.add_argument("--offset-column", default="byte_offset")
    parser.add_argument("--size-column", default="byte_size")
    parser.add_argument("--id-column", default="image_id")
    # Defaults fit one node with 4 GPUs and ~110 CPUs; scale with the cluster.
    parser.add_argument("--fetch-workers", type=int, default=64, help="Concurrent ranged reads")
    parser.add_argument("--cpu-workers", type=int, default=76, help="Decode/measurement/resize workers")
    parser.add_argument("--clip-workers", type=int, default=6, help="CLIP actors, half a GPU each")
    args = parser.parse_args()

    client = RayClient()
    client.start()
    try:
        start = time.perf_counter()
        build_pipeline(args).run(executor=RayDataExecutor())
        print(f"Finished in {time.perf_counter() - start:.1f}s; output in {args.output}")
    finally:
        client.stop()


if __name__ == "__main__":
    main()
