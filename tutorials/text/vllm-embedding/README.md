# High-Throughput Text Embedding with vLLM

This tutorial embeds a text column of Parquet files with `VLLMEmbeddingModelStage`, which runs [vLLM](https://github.com/vllm-project/vllm) inside each GPU worker. It defaults to [`nvidia/Nemotron-3.5-Embed-1B-NVFP4`](https://huggingface.co/nvidia/Nemotron-3.5-Embed-1B-NVFP4) and focuses on the settings that decide throughput for bulk embedding.

## Install

```
uv venv --python 3.12 && source .venv/bin/activate
uv pip install --no-sources "nemo_curator[vllm] @ git+https://github.com/NVIDIA-NeMo/Curator"
```

Tested with vLLM 0.34.0, PyTorch 2.14.0 (CUDA 13.0) and Transformers 5.22.2 on Linux arm64. `--no-sources` makes `uv` take vLLM and PyTorch from PyPI; without it, Curator's CUDA 12.9 index settings apply and the install does not resolve. NVFP4 checkpoints need a GPU with native FP4 support (NVIDIA Blackwell) to run at full speed.

## Usage

```
python tutorials/text/vllm-embedding/main.py \
    --input-path /path/to/parquet \
    --output-path /path/to/embeddings \
    --text-field text \
    --keep-fields id \
    --replicas-per-gpu 3
```

The output has the `--keep-fields` columns and an `embeddings` column, in input row order. The input text is not copied to the output.

For the default model the script sets what it needs: the `passage: ` prefix (see [Text prefixes](#text-prefixes)) and the vLLM arguments `dtype=bfloat16` and `trust_remote_code=True`. To embed queries, pass the query prefix from the model card with `--text-prefix`. For another model, pass `--model`; no prefix or extra vLLM arguments are applied unless you give them.

Embed one corpus with one vLLM version. Vectors from different versions are close but not identical (mean cosine 0.999 between 0.30 and 0.34 on the default model), which is enough to disturb exact comparisons.

## Settings that matter

vLLM's defaults are sized for interactive chat. For bulk embedding, three settings decide whether the GPU stays busy.

- **`--max-num-seqs` and `--max-num-batched-tokens`** (defaults here: 256 and 262144): how much work one engine step may hold. vLLM's engine loop is a single Python process, so small steps spend most of their time on per-step overhead while the GPU waits. Raise `--max-num-batched-tokens` with `--max-num-seqs`; a step is limited by whichever is reached first.
- **`--replicas-per-gpu`**: how many model replicas share each GPU. Between steps, a replica tokenizes the next chunk and converts results on the CPU, and its GPU is idle. A second or third replica fills those gaps. This is worth it when the model is small next to the GPU's memory; each replica gets an equal share of 80% of it. Watch GPU utilization: add replicas while it is clearly below 100%, and stop when it no longer rises.
- **`--max-model-len`**: texts longer than this are truncated. Attention cost grows faster than length, so a few very long documents can take a large share of the run. If the first N tokens represent your documents well enough, a lower limit is the cheapest speedup available.

`--num-workers` caps the total number of replicas. Left unset, Curator starts as many as the GPUs allow, which is the number of GPUs times `--replicas-per-gpu`.

## Text prefixes

Many embedding models are trained with a prefix that marks the input's role, such as `passage: ` for documents and `query: ` for queries. Embedding a document without its prefix produces a valid-looking vector that does not match vectors made with it. `--text-prefix` sets `text_prefix` on the stage, which prepends the string before tokenization. Check the model card for the exact string, including trailing whitespace.

## Other options

- `--embedding-dtype`: `float32` by default. `float16` halves the output size; keep `float32` if the embeddings feed semantic deduplication.
- `--cache-dir`: where the model is downloaded and loaded from.
- `--vllm-init-kwargs`: arguments for `vllm.LLM` as JSON, for example `'{"dtype": "bfloat16", "trust_remote_code": true}'`. Giving this replaces the default model's arguments.
