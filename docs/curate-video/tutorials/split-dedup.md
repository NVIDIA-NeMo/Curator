---
description: "End-to-end workflow tutorial covering the video curation process from splitting through semantic deduplication (Ray/Python)"
categories: ["video-curation"]
tags: ["workflow", "pipeline", "video-splitting", "deduplication", "semantic", "ray"]
personas: ["mle-focused", "data-scientist-focused"]
difficulty: "intermediate"
content_type: "tutorial"
modality: "video-only"

---

(video-tutorials-split-dedup)=

# Split and Remove Duplicates Workflow

Learn how to run the splitting pipeline to generate clips and embeddings, then remove near-duplicate clips using semantic duplicate removal.

```{contents} Tutorial Steps:
:local:
:depth: 2
```

## Before You Start

- Complete the [Get Started guide](gs-video).

---

## 1. Generate Clips and Embeddings

Run the splitting example. Set `VIDEO_DIR`, `OUT_DIR`, and `MODEL_DIR` first.

```bash
python -m nemo_curator.examples.video.video_split_clip_example \
  --video-dir "$VIDEO_DIR" \
  --model-dir "$MODEL_DIR" \
  --output-clip-path "$OUT_DIR" \
  --splitting-algorithm fixed_stride \
  --fixed-stride-split-duration 10.0 \
  --embedding-algorithm internvideo2 \
  --transcode-encoder libopenh264 \
  --verbose
```

Writer-related flags you can add:

```bash
  --no-upload-clips          # Do not write MP4 files
  --dry-run                   # Write nothing; validate only
  --no-generate-embeddings   # Disable embedding outputs (enabled by default)
  --generate-captions        # Include caption fields in JSON metadata
  --generate-previews        # Write .webp previews for caption windows
```

The pipeline writes embeddings under `$OUT_DIR/iv2_embd_parquet/` (or `ce1_embd_parquet/` if you use Cosmos-Embed1).

### Embedding Format Example

The pipeline writes embeddings to Parquet with two columns:

- **id**: String UUID for the clip
- **embedding**: List of float values with length equal to the model’s embedding dimension (for InternVideo2, 512)

::::{tab-set}

:::{tab-item} Directory layout

```text
$OUT_DIR/
  iv2_embd_parquet/
    1a2b3c4d-....parquet
    5e6f7g8h-....parquet
```

:::

:::{tab-item} Schema

```text
id: string
embedding: list<float32>  # length = 512 for InternVideo2
```

:::

:::{tab-item} Sample row

```json
{"id": "a3f2b0c1-7d64-4a1e-9f2b-8b0f6d1e2c33", "embedding": [0.0123, -0.0456, 0.0031, 0.1279]}
```

:::

:::{tab-item} Read example

```python
import pyarrow.parquet as pq

table = pq.read_table(f"{OUT_DIR}/iv2_embd_parquet")
df = table.to_pandas()
print(df.head())  # columns: id, embedding (list[float])
```

:::

::::

---

## 2. Run Semantic Duplicate Removal

Use K-means clustering followed by pairwise similarity on the Parquet embeddings.

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.stages.deduplication.semantic.kmeans import KMeansStage
from nemo_curator.stages.deduplication.semantic.pairwise import PairwiseStage

INPUT_PARQUET = f"{OUT_DIR}/iv2_embd_parquet"  # or s3://...
OUTPUT_DIR = f"{OUT_DIR}/semantic_dedup"

pipe = Pipeline(name="video_semantic_dedup", description="K-means + pairwise duplicate removal")

pipe.add_stage(
    KMeansStage(
        n_clusters=1000,
        id_field="id",
        embedding_field="embedding",
        input_path=INPUT_PARQUET,
        output_path=f"{OUTPUT_DIR}/kmeans",
        input_filetype="parquet",
        embedding_dim=512,
        read_kwargs={"storage_options": None},
        write_kwargs={"storage_options": None},
    )
)

pipe.add_stage(
    PairwiseStage(
        id_field="id",
        embedding_field="embedding",
        input_path=f"{OUTPUT_DIR}/kmeans",
        output_path=f"{OUTPUT_DIR}/pairwise",
        which_to_keep="hard",   # or "easy" or "random"
        sim_metric="cosine",    # or "l2"
        pairwise_batch_size=1024,
        read_kwargs={"storage_options": None},
        write_kwargs={"storage_options": None},
    )
)

pipe.run()
```

---

## 3. Inspect Results

- K-means outputs per-cluster partitions under `${OUTPUT_DIR}/kmeans/`.
- Pairwise outputs per-cluster similarity files under `${OUTPUT_DIR}/pairwise/` with columns including `id`, `max_id`, and `cosine_sim_score`.
- Use these to decide keep/remove policies or downstream sampling.

## 4. Export for Training

After duplicate removal, export curated clips and metadata for training. Common video exports:

- Parquet index + media files (mp4/webp) under `${OUT_DIR}`
- Tar archives (WebDataset-style) containing per-clip payloads and JSON/Parquet metadata

Video-specific pointers:

- Use `ClipWriterStage` path helpers to locate outputs: `nemo_curator/stages/video/io/clip_writer.py`.
  - Processed videos: `get_output_path_processed_videos(OUT_DIR)`
  - Clip chunks and previews: `get_output_path_processed_clip_chunks(OUT_DIR)`, `get_output_path_previews(OUT_DIR)`
  - Embeddings parquet: `${OUT_DIR}/iv2_embd_parquet` (or `${OUT_DIR}/ce1_embd_parquet`)

### Example Export

The following example packages clips and minimal JSON metadata into tar files:

```python
import json, math, os, tarfile
import io
from glob import glob

OUT_DIR = os.environ["OUT_DIR"]
clips_dir = os.path.join(OUT_DIR, "clips")  # adjust if filtering path used
meta_parquet = os.path.join(OUT_DIR, "iv2_embd_parquet")

def iter_clips(path):
    for p in glob(os.path.join(path, "**", "*.mp4"), recursive=True):
        clip_id = os.path.splitext(os.path.basename(p))[0]
        yield clip_id, p

def write_shards(items, out_dir, samples_per_shard=10000, max_shards=5):
    os.makedirs(out_dir, exist_ok=True)
    shard, buf = 0, []
    for i, (clip_id, mp4_path) in enumerate(items, 1):
        buf.append((clip_id, mp4_path))
        if i % samples_per_shard == 0:
            _write_tar(shard, buf, out_dir, max_shards)
            shard, buf = shard + 1, []
    if buf:
        _write_tar(shard, buf, out_dir, max_shards)

def _write_tar(shard, records, out_dir, max_shards):
    tar_name = f"{shard:0{max_shards}d}.tar"
    tar_path = os.path.join(out_dir, tar_name)
    with tarfile.open(tar_path, "w") as tf:
        for clip_id, mp4_path in records:
            tf.add(mp4_path, arcname=f"{clip_id}.mp4")
            info = tarfile.TarInfo(name=f"{clip_id}.json")
            payload = json.dumps({"id": clip_id}).encode("utf-8")
            info.size = len(payload)
            tf.addfile(info, fileobj=io.BytesIO(payload))

write_shards(iter_clips(clips_dir), os.path.join(OUT_DIR, "wds"))
```
