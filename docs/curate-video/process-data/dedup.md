---
description: "Remove near-duplicate clips using semantic clustering and pairwise similarity"
categories: ["video-curation"]
tags: ["deduplication", "semantic", "pairwise", "kmeans", "video"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "howto"
modality: "video-only"
---

(video-process-dedup)=

# Duplicate Identification

Use clip-level embeddings to identify near-duplicate video clips so your dataset remains compact, diverse, and efficient to train on.

## Before You Start
- Make sure you have embeddings which are written by the [`ClipWriterStage`](video-save-export) under `ce1_embd_parquet/`. For a runnable workflow, refer to the [Split and Remove Duplicates Workflow](video-tutorials-split-dedup). The embeddings must be in parquet files containing the columns `id` and `embedding`.
- Verify local paths or configure S3-compatible credentials. Provide `storage_options` in read/write keyword arguments when reading or writing cloud paths.


## How it Works

Duplicate identification operates on clip-level embeddings produced during processing:

1. **Inputs**
   - Parquet batches from `ClipWriterStage` under `ce1_embd_parquet/`
   - Columns: `id`, `embedding`

2. **Outputs**
   - Cluster: `KMeansStage` partitions embeddings and writes centroid distances (for example, `cosine_dist_to_cent`).
   - Pairwise: `PairwiseStage` computes within-cluster similarity on GPU and, for each clip, emits `max_id` and `cosine_sim_score`. Ranking controls whether to prefer outliers ("hard") or representatives ("easy").
   - Identify: `IdentifyDuplicatesStage` filters pairs with `cosine_sim_score >= 1.0 - eps` and writes Parquet files of duplicate `id`s for removal during export.

---

## Quickstart

Use the semantic duplicate workflow with clip embeddings written to Parquet.

:::::{tab-set}

::::{tab-item} Single Step Workflow

The `SemanticDeduplicationWorkflow` provides an end-to-end interface that orchestrates K-means clustering, pairwise similarity computation, and duplicate identification:

```python
from nemo_curator.stages.deduplication.semantic.workflow import SemanticDeduplicationWorkflow
from nemo_curator.stages.deduplication.semantic.ranking import RankingStrategy
from nemo_curator.backends.xenna import XennaExecutor

workflow = SemanticDeduplicationWorkflow(
    input_path="/path/to/embeddings/",  # e.g., ce1_embd_parquet/
    output_path="/path/to/duplicates/",
    cache_path="/path/to/cache/",  # Optional: defaults to output_path
    n_clusters=1000,
    id_field="id",
    embedding_field="embedding",
    embedding_dim=768,  # Embedding dimension (768 for Cosmos-Embed1, varies by model)
    input_filetype="parquet",
    eps=0.1,  # Similarity threshold: cosine_sim >= 1.0 - eps identifies duplicates
    ranking_strategy=RankingStrategy.metadata_based(
        metadata_cols=["cosine_dist_to_cent", "id"],
        ascending=[True, True],
    ),
    pairwise_batch_size=1024,
    read_kwargs={"storage_options": None},  # Add S3 credentials here if needed
    write_kwargs={"storage_options": None},
    verbose=True,
)

# Run with XennaExecutor (GPU-accelerated)
executor = XennaExecutor()
results = workflow.run(executor)
```

:::{note}
**Determine `eps` first**: Before running the full workflow, we recommend first running K-means and pairwise steps (set `eps=None`) to inspect similarity distributions and determine an appropriate `eps` threshold. See the tip below for details.
:::

The workflow automatically:
1. Runs K-means clustering to partition embeddings into clusters
2. Computes pairwise similarity within each cluster
3. Identifies duplicates based on the `eps` threshold
4. Writes duplicate IDs to `output_path/duplicates/`

```{seealso}
For detailed information about how semantic deduplication works, see [Semantic Deduplication](text-process-data-format-sem-dedup). The algorithm and concepts are the same for video clips as for text documents.
```

::::

::::{tab-item} Individual Stages

For advanced users who need fine-grained control, you can run the stages individually:

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.deduplication.semantic.kmeans import KMeansStage
from nemo_curator.stages.deduplication.semantic.pairwise import PairwiseStage
from nemo_curator.stages.deduplication.semantic.ranking import RankingStrategy
from nemo_curator.stages.deduplication.semantic.identify_duplicates import IdentifyDuplicatesStage

pipe = Pipeline(name="semantic_dedup")

pipe.add_stage(
    KMeansStage(
        n_clusters=1000,
        id_field="id",
        embedding_field="embedding",
        input_path="/path/to/embeddings/",
        output_path="/path/to/kmeans_out/",
        input_filetype="parquet",
        embedding_dim=512,
    )
)

pipe.add_stage(
    PairwiseStage(
        id_field="id",
        embedding_field="embedding",
        input_path="/path/to/kmeans_out/",
        output_path="/path/to/pairwise_out/",
        ranking_strategy=RankingStrategy.metadata_based(
            metadata_cols=["cosine_dist_to_cent", "id"],
            ascending=[True, True],
        ),
    )
)

pipe.add_stage(
    IdentifyDuplicatesStage(
        output_path="/path/to/duplicates/",
        eps=0.1,
    )
)

pipe.run()
```

::::

::::{tab-item} Script Flags

No example script flags are available for duplicate identification in the split pipeline. Run these stages as a separate job against Parquet embeddings written by the example pipeline's writer.

::::
:::::

:::{tip}
**Recommended Workflow: Determine `eps` First**

The `eps` parameter is highly data-dependent and affects how many duplicates are identified. We recommend a two-step approach:

1. **Step 1: Run K-means and pairwise without duplicate identification**
   - Use `SemanticDeduplicationWorkflow` with `eps=None` (or run K-means and pairwise stages individually)
   - This generates pairwise similarity scores without identifying duplicates

2. **Step 2: Inspect the similarity distribution**
   - Analyze the `cosine_sim_score` values in the pairwise results
   - Determine an appropriate `eps` threshold based on your data characteristics
   - For example, if 20% of pairs have similarity ≥ 0.9, you might use `eps=0.1` (since `cosine_sim >= 1.0 - eps`)

3. **Step 3: Run the full workflow with your chosen `eps`**
   - Use `SemanticDeduplicationWorkflow` with the determined `eps` value
   - Or run `IdentifyDuplicatesStage` separately on the pairwise results

For a detailed example of this workflow with similarity analysis, see the [Step-by-Step Semantic Deduplication tutorial](https://github.com/NVIDIA-NeMo/Curator/blob/main/tutorials/text/deduplication/semantic/semantic_step_by_step.ipynb) (demonstrated on text data, but the approach applies to video clips as well).
:::

:::{tip}
**Custom Ranking with Metadata Columns**

If your embedding Parquet files contain additional metadata columns (such as video quality scores, duration, resolution, or other clip attributes), you can use `RankingStrategy.metadata_based()` to create custom ranking methods. This allows you to prioritize which clips to keep within duplicate groups based on your specific criteria.

For example, to prefer higher quality or longer duration clips:

```python
from nemo_curator.stages.deduplication.semantic.ranking import RankingStrategy

# Prefer clips with higher quality scores, then longer duration
ranking_strategy = RankingStrategy.metadata_based(
    metadata_cols=["quality_score", "duration"],
    ascending=[False, False],  # False = descending (higher is better)
)

# Or prefer clips closer to cluster centroid, then by quality
ranking_strategy = RankingStrategy.metadata_based(
    metadata_cols=["cosine_dist_to_cent", "quality_score"],
    ascending=[True, False],  # Closer to centroid first, then higher quality
)
```

The metadata columns must be present in your embedding Parquet files and will be preserved through the K-means stage. Specify these columns using the `metadata_fields` parameter in `KMeansStage` or `SemanticDeduplicationWorkflow`.
:::

## Parameters

:::::{tab-set}

::::{tab-item} KMeansStage

```{list-table} KMeansStage (semantic clustering)
:header-rows: 1

* - Parameter
  - Description
* - `n_clusters`
  - Number of clusters for K‑means (for example, 1,000+ for multi‑million clip sets).
* - `id_field`
  - Column name containing clip IDs (for example, `"id"`).
* - `embedding_field`
  - Column with vector data (for example, `"embedding"`).
* - `input_path`
  - Path to Parquet embeddings directory from the writer.
* - `output_path`
  - Directory for K‑means outputs (sharded by cluster).
* - `input_filetype`
  - Use `"parquet"` for video embeddings.
* - `embedding_dim`
  - Embedding dimension (Cosmos‑Embed1 varies by variant: 768 for most).
```

::::

::::{tab-item} PairwiseStage

```{list-table} PairwiseStage (within‑cluster similarity)
:header-rows: 1

* - Parameter
  - Description
* - `ranking_strategy`
  - Ranking strategy for selecting which clips to keep within clusters. Use `RankingStrategy.metadata_based(metadata_cols=[...], ascending=[...])` to sort by metadata columns (for example, `metadata_cols=["cosine_dist_to_cent", "id"]`). Use `RankingStrategy.random()` for random selection.
* - `pairwise_batch_size`
  - Batch size for GPU pairwise computation (default `1024`). Increase with available memory.
* - `embedding_dim`
  - Embedding dimension for memory estimates and batching.
* - `id_field`
  - Column name containing clip IDs (for example, `"id"`).
* - `embedding_field`
  - Column with vector data (for example, `"embedding"`).
* - `input_path`
  - Path to K-means output directory (sharded by cluster).
* - `output_path`
  - Directory for pairwise similarity outputs.
```

::::

::::{tab-item} IdentifyDuplicatesStage

```{list-table} IdentifyDuplicatesStage (duplicate identification)
:header-rows: 1

* - Parameter
  - Description
* - `output_path`
  - Directory to write Parquet files containing duplicate `id`s.
* - `eps`
  - Similarity threshold: pairs with `cosine_sim_score >= 1.0 - eps` are identified as duplicates (for example, `0.1` means similarity >= `0.9`).
* - `read_kwargs`
  - Optional keyword arguments for reading files (including `storage_options` for cloud storage).
* - `write_kwargs`
  - Optional keyword arguments for writing files (including `storage_options` for cloud storage).
* - `verbose`
  - Enable verbose logging (default `False`).
```

::::

::::{tab-item} SemanticDeduplicationWorkflow

The `SemanticDeduplicationWorkflow` accepts parameters from all three stages (KMeansStage, PairwiseStage, and IdentifyDuplicatesStage). See the tabs above for parameter descriptions.

```{list-table} SemanticDeduplicationWorkflow (workflow-specific parameters)
:header-rows: 1

* - Parameter
  - Description
* - `cache_path`
  - Directory for intermediate results (K-means and pairwise outputs). Defaults to `output_path` if not specified.
* - `cache_kwargs`
  - Optional keyword arguments for writing cache files (including `storage_options` for cloud storage). Defaults to `write_kwargs` if not specified.
* - `clear_output`
  - Clear output directory before running (default `True`).
* - `metadata_fields`
  - List of metadata field names to preserve in output (optional).
```

For parameters shared with individual stages, refer to:
- **KMeansStage** tab: `input_path`, `output_path`, `n_clusters`, `id_field`, `embedding_field`, `embedding_dim`
- **PairwiseStage** tab: `ranking_strategy`, `pairwise_batch_size`
- **IdentifyDuplicatesStage** tab: `eps`
- Common parameters: `read_kwargs`, `write_kwargs`, `verbose`

::::
:::::

---

## Removing Duplicates

The duplicate identification stages (`IdentifyDuplicatesStage` or `SemanticDeduplicationWorkflow` with `eps` specified) write Parquet files containing duplicate clip IDs to the output directory (typically `output_path/duplicates/`). These files contain a single column `id` with the IDs of clips that should be removed.

**It is your responsibility to exclude these duplicate IDs when exporting or persisting your final dataset.** The removal process depends on how you want to persist and shard your data:
