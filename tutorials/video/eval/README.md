# Video Caption Quality Evaluation

Use this alignment score as a fast regression signal to capture large semantic
drift. Run scoring after model updates or pipeline refactors.

## Method

CosmosEmbed1 is a video-text embedding model (ViT + QFormer) with understanding
of temporal motion and physical actions. It projects both video and text into a
shared continuous space, where cosine similarity directly measures semantic
alignment.

CosmosEmbed1's text encoder is limited to 128 tokens. Since VLM captions sometimes
exceed this limit, we use a summarizer LLM to extract visual elements and
compress each caption to a single ≤128-token chunk before encoding.

**Dataset construction**: Randomly sample 3,000 videos from the source dataset,
run the pipeline with fixed-stride splitting (10s) to produce clips, filter by
aesthetic score (>=3.5), and compute CosmosEmbed1-224p embeddings (256-dim) per
clip. Apply K-means (K=200) on the clip embeddings and select one representative
clip per cluster (closest to centroid, one per source video). The final benchmark
is a set of 200 clips.

## Regression Workflow

1. Run scoring on the fixed benchmark dataset.
2. Flag for human review if:
   - Score drops >5% from baseline, or
   - A new model expected to be better scores lower than the baseline.
3. Sample the 10 worst-scoring clips for human spot-check. Human review
   determines if the drop is a real quality regression or an alignment score
   artifact (e.g., the new model is just more verbose).

## Quick Start

### 1. Build benchmark dataset (one-time)

Samples videos from a large dataset, generates CosmosEmbed1 embeddings, and
selects diverse representatives via K-means clustering.

```bash
python build_benchmark_dataset.py \
    --video-dir /path/to/video_dataset \
    --output-dir /path/to/benchmark_200 \
    --model-dir /path/to/models \
    --sample-size 3000 \
    --num-clusters 200
```

### 2. Generate captions (per model)

Run the video pipeline on the benchmark clips with each captioning model.
Include ``--embedding-algorithm cosmos-embed1-224p`` to generate video embeddings
alongside captions (needed for scoring in Step 3).

```bash
python tutorials/video/getting-started/video_split_clip_example.py \
    --video-dir /path/to/benchmark_200/input \
    --output-path /path/to/captions_qwen25 \
    --model-dir /path/to/models \
    --splitting-algorithm fixed_stride \
    --fixed-stride-split-duration 10.0 \
    --embedding-algorithm cosmos-embed1-224p \
    --generate-captions \
    --captioning-algorithm qwen2.5 \
    --aesthetic-threshold 3.5
```

Repeat with ``--captioning-algorithm nemotron-bf16``, ``qwen3``, etc.
For Nemotron, add ``--captioning-model-does-preprocess``.

### 3. Score

Use the ``ce1_embd/`` directory from any of the captioning runs as
``--embedding-dir`` (all runs produce identical video embeddings).

```bash
# First run: summarize and save
python caption_clipscore.py \
    --embedding-dir /path/to/captions_qwen25/ce1_embd \
    --cosmos-model-dir /path/to/models \
    --summarizer-model /path/to/Llama-3.1-8B-Instruct \
    --caption-dirs \
        qwen25=/path/to/captions_qwen25 \
        qwen3=/path/to/captions_qwen3 \
    --uid-list /path/to/benchmark_200/selected_uids.txt \
    --save-summaries /path/to/benchmark_200/summaries.json \
    --output-csv results.csv

# Subsequent runs: load cached summaries (no LLM needed)
python caption_clipscore.py \
    --embedding-dir /path/to/captions_qwen25/ce1_embd \
    --cosmos-model-dir /path/to/models \
    --load-summaries /path/to/benchmark_200/summaries.json \
    --caption-dirs \
        qwen25=/path/to/captions_qwen25 \
        qwen3=/path/to/captions_qwen3 \
    --uid-list /path/to/benchmark_200/selected_uids.txt \
    --output-csv results.csv
```

The ``--uid-list`` flag filters scoring to the K-means selected clips only
(the pipeline may produce more clips than were selected for the benchmark).
UIDs are resolved by (source\_video, duration\_span) across pipeline runs.

### 4. Evaluate a new model

When adding a new captioning model, only generate captions and score the new
model -- no need to re-run or re-summarize baseline models.

```bash
# Generate captions for the new model
python tutorials/video/getting-started/video_split_clip_example.py \
    --video-dir /path/to/benchmark_200/input \
    --output-path /path/to/captions_new_model \
    --model-dir /path/to/models \
    --splitting-algorithm fixed_stride \
    --fixed-stride-split-duration 10.0 \
    --embedding-algorithm cosmos-embed1-224p \
    --generate-captions \
    --captioning-algorithm <new-model> \
    --aesthetic-threshold 3.5

# Score the new model
python caption_clipscore.py \
    --embedding-dir /path/to/captions_new_model/ce1_embd \
    --cosmos-model-dir /path/to/models \
    --summarizer-model /path/to/Llama-3.1-8B-Instruct \
    --caption-dirs new_model=/path/to/captions_new_model \
    --uid-list /path/to/benchmark_200/selected_uids.txt \
    --save-summaries /path/to/captions_new_model/summaries.json \
    --output-csv new_model_results.csv
```

Compare the new model score against the cached baseline scores directly.
