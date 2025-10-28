
# ArXiv document batch processing

Pipeline:

1. Download arXiv papers
2. Convert MBART-50 to TensorRT-LLM format
3. Prepare arXiv documents
4. Partition documents
5. Translate documents
6. Augment documents

## Download arXiv papers

ArXiv provides bulk data access through Amazon S3, see [arXiv documentation](https://info.arxiv.org/help/bulk_data_s3.html) for more information.

> **Note:** please review and comply with the terms of use for arXiv bulk data.

## Set up Triton Server

Batch translation uses the [Triton Inference Server](https://github.com/triton-inference-server/server) for optimized inference. Use a Docker image based on [tritonserver](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags) and convert the MBART-50 Hugging Face model to Triton/TensorRT-LLM format (edit paths in `scripts/setup_tritonserver.sh` if needed). This writes the converted model into `/workspace/tritonserver_data`.

```sh
docker run --gpus all --rm \
  -v $PWD:/workspace \
  -w /workspace \
  nvcr.io/nvidia/tritonserver:24.12-trtllm-python-py3 \
  bash scripts/setup_tritonserver.sh
```

## Batch prepare

Next, edit input and output paths within `batch_prepare.py` and run a batch job to prepare documents for translation. This filters out documents that fail to parse so GPU compute time is not wasted later.

```sh
docker run --rm \
  -v $PWD:/workspace \
  -v path/to/data:/data \
  -e PYTHONPATH=/workspace/src \
  latex-augment python3 scripts/arxiv/batch_prepare.py
```

## Batch partition

Partition the prepared documents into size-balanced shards for translation and later compilation. Edit input and output paths within `batch_partition.py`, then run:

```sh
docker run --rm \
  -v $PWD:/workspace \
  -v path/to/data:/data \
  -e PYTHONPATH=/workspace/src \
  latex-augment python3 scripts/arxiv/batch_partition.py
```

## Batch translation

Edit input and output paths within `batch_translate.py` and run multiple jobs on GPU nodes. Each process expects `<language>` and a unique `<job-id>` from 0..N-1 such that each job processes 8 shards in parallel on 8 GPUs.

```sh
docker run --gpus all --rm \
  -v $PWD:/workspace \
  -v path/to/data:/data \
  -e PYTHONPATH=/workspace/src \
  latex-augment python3 -m torch.distributed.run --standalone --nproc-per-node=8 \
  scripts/arxiv/batch_translate.py <language> <job-id>
```

## Batch augmentation

Edit input and output paths within `batch_augment.py` and run it on a CPU node using the above Docker image.
```sh
docker run --rm \
  -v $PWD:/workspace \
  -v path/to/data:/data \
  -e PYTHONPATH=/workspace/src \
  latex-augment python3 scripts/arxiv/batch_augment.py <language>
```
