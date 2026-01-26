# Resource Optimization Guide

Guidelines for configuring CPU, GPU, and memory resources in NeMo Curator pipelines.

## Resource Configuration

NeMo Curator uses the `Resources` dataclass to configure stage requirements:

```python
from nemo_curator.stages.resources import Resources

# CPU-only stage
Resources(cpus=4.0)

# GPU stage with memory fraction (shares GPU with other stages)
Resources(gpu_memory_gb=16.0)

# GPU stage with dedicated GPU(s)
Resources(gpus=1.0)  # One full GPU
Resources(gpus=2.0)  # Two full GPUs

# Mixed CPU + GPU
Resources(cpus=4.0, gpu_memory_gb=16.0)
```

**Important**: Cannot specify both `gpus` and `gpu_memory_gb` simultaneously.

---

## Resource Patterns by Stage Type

### CPU-Only Stages

| Stage Category | Typical CPUs | Notes |
|----------------|--------------|-------|
| IO (Reader/Writer) | 1-2 | I/O bound |
| Heuristic Filters | 1-2 | Lightweight |
| Modifiers | 1-2 | Text transforms |
| Motion Vector Decode | 2-4 | FFmpeg-based |

### GPU Memory Fraction Stages

For stages that can share a GPU:

| Stage | gpu_memory_gb | Notes |
|-------|---------------|-------|
| `TransNetV2ClipExtractionStage` | 16.0 | Scene detection |
| `CosmosEmbed1EmbeddingStage` | 16.0 | Video embeddings |
| `ImageEmbeddingStage` | 0.25 GPU | CLIP (very small) |

### Full GPU Stages

For stages that need dedicated GPU(s):

| Stage | gpus | Notes |
|-------|------|-------|
| `CaptionGenerationStage` | 1.0 | Qwen VL model |
| Large classifiers | 1.0 | Full model load |

---

## Dataset Size Recommendations

### Text Curation

| Dataset Size | Nodes | GPUs | Memory | Notes |
|--------------|-------|------|--------|-------|
| < 10GB | 1 | 1 | 32GB | Single machine |
| 10-100GB | 1 | 4 | 64GB | Multi-GPU node |
| 100GB-500GB | 2-4 | 8-16 | 128GB+ | Small cluster |
| 500GB-1TB | 4-8 | 16-32 | 256GB+ | Medium cluster |
| > 1TB | 8+ | 32+ | 512GB+ | Large cluster |

### Video Curation

| Dataset Size | Nodes | GPUs | Notes |
|--------------|-------|------|-------|
| < 1000 videos | 1 | 1-2 | Single machine |
| 1K-10K videos | 1-2 | 4-8 | Captioning bottleneck |
| 10K-100K videos | 4-8 | 16-32 | Distributed |
| > 100K videos | 8+ | 32+ | Large cluster |

---

## Fuzzy Deduplication Resources

Fuzzy dedup is memory-intensive. Use these guidelines:

### Memory Estimation Formula

```
memory_per_doc ≈ 8 bytes * num_hashes + overhead
total_memory ≈ num_docs * memory_per_doc * bands_per_iteration / num_bands
```

### Parameter Tuning for Memory

| Dataset Size | bands_per_iteration | Notes |
|--------------|---------------------|-------|
| < 100GB | 5-10 (default) | Fast |
| 100-500GB | 2-3 | Moderate memory |
| > 500GB | 1 | Streaming mode |

### OOM Prevention

If encountering Out-of-Memory errors:

1. **Reduce `bands_per_iteration`** - Most effective
2. **Increase cluster size** - More nodes = more memory
3. **Use 64-bit hash** - Only for >1B documents
4. **Enable streaming mode** - For very large datasets

---

## Ray Cluster Configuration

### Worker Configuration

```yaml
ray_client:
  _target_: nemo_curator.core.client.RayClient
  num_cpus: null  # Use all available
  num_gpus: 4     # Number of GPUs per node
```

### Memory Settings

For memory-intensive workloads:

```python
import ray
ray.init(
    object_store_memory=100_000_000_000,  # 100GB object store
    _memory=200_000_000_000,              # 200GB heap
)
```

---

## Stage-Specific Recommendations

### Classifiers

| Classifier | Batch Size | GPU Memory | Throughput |
|------------|------------|------------|------------|
| QualityClassifier | 64 | ~8GB | ~1000 docs/s |
| FineWebEduClassifier | 32 | ~12GB | ~500 docs/s |
| AegisClassifier | 16 | ~16GB | ~200 docs/s |

### Video Stages

| Stage | Batch Size | GPU Memory | Throughput |
|-------|------------|------------|------------|
| TransNetV2 | 1 | ~8GB | ~2 clips/s |
| CaptionGeneration | 1 | ~24GB | ~0.5 clips/s |
| CosmosEmbed1 | 4 | ~12GB | ~4 clips/s |

---

## Executor Configuration

### XennaExecutor (Default)

```python
from nemo_curator.backends.xenna import XennaExecutor

executor = XennaExecutor(config={
    "logging_interval": 60,
    "ignore_failures": False,
    "execution_mode": "streaming",  # For large datasets
    "cpu_allocation_percentage": 0.95,
    "autoscale_interval_s": 180,
})
```

### Execution Modes

| Mode | Use Case | Memory |
|------|----------|--------|
| `batch` | Small datasets, full reprocessing | Higher |
| `streaming` | Large datasets, incremental | Lower |

---

## Monitoring

### Key Metrics to Watch

1. **GPU Utilization** - Should be >80% for GPU stages
2. **Memory Usage** - Watch for approaching limits
3. **Shuffle Stage** - Bottleneck in deduplication
4. **Task Throughput** - Documents/clips per second

### Common Bottlenecks

| Symptom | Cause | Solution |
|---------|-------|----------|
| Low GPU util | Small batches | Increase batch_size |
| High memory | Large bands_per_iteration | Reduce parameter |
| Slow IO | Network bottleneck | Use local storage |
| Task failures | OOM | Add more workers |
