# Audio Stage-Wise Profiling

Per-stage performance profiling for the two audio pipelines in NeMo Curator: **FLEURS** (ASR inference, GPU) and **ALM** (audio language model data curation, CPU). Results produced using the benchmark scripts in `benchmarking/scripts/` and analyzed via `TaskPerfUtils.collect_stage_metrics()`.

## Machine Specs

- CPU: Intel Core i9-9900KF @ 3.60GHz (8 cores / 16 threads)
- RAM: 32 GB
- GPU: NVIDIA GeForce RTX 3080 Ti 12 GB
- OS: Ubuntu 20.04, Linux 5.15

## FLEURS Pipeline (GPU)

**Configuration:** `nvidia/parakeet-tdt-0.6b-v2`, `en_us`, `dev` split, WER threshold 75, 1 GPU.

| Metric | Xenna | Ray Data |
|--------|-------|----------|
| Wall clock | 60.71s | 45.78s |
| Tasks processed | 394 | 394 |
| Throughput (tasks/sec) | 6.49 | 8.61 |

### Per-Stage Timing (Xenna)

| Stage | Total Time | Mean/Task | Items | % of Sum |
|-------|-----------|-----------|-------|----------|
| CreateInitialManifestFleurs | 2,147.28s | 5.4499s | 0* | 91.4% |
| ASR_inference | 144.69s | 0.3672s | 6,244 | 6.2% |
| GetPairwiseWerStage | 0.01s | 0.00003s | 394 | <0.1% |
| GetAudioDurationStage | 0.23s | 0.0006s | 394 | <0.1% |
| PreserveByValueStage | 0.01s | 0.00003s | 394 | <0.1% |
| AudioToDocumentStage | 0.12s | 0.0003s | 394 | <0.1% |
| JsonlWriter | 0.20s | 0.0005s | 394 | <0.1% |

\* `items_processed=0` for CreateInitialManifestFleurs because it is a generator stage (produces tasks from nothing).

### Per-Stage Timing (Ray Data)

| Stage | Total Time | Mean/Task | Items |
|-------|-----------|-----------|-------|
| CreateInitialManifestFleurs | 2,181.12s | 5.5358s | 0* |
| ASR_inference | 133.94s | 0.3400s | 6,244 |
| GetPairwiseWerStage | 0.01s | 0.00003s | 394 |
| GetAudioDurationStage | 0.19s | 0.0005s | 394 |
| PreserveByValueStage | 0.00s | 0.00001s | 394 |
| AudioToDocumentStage | 0.09s | 0.0002s | 394 |
| JsonlWriter | 0.13s | 0.0003s | 394 |

### FLEURS Bottlenecks

1. **CreateInitialManifestFleurs (91.4% of cumulative stage time).** This stage downloads audio files from the HuggingFace FLEURS dataset and creates the initial manifest. The high per-task time (~5.5s) is dominated by network I/O and audio file extraction. In a nightly benchmark context this is a one-time cost per run and does not reflect production throughput, but it is the dominant contributor to wall clock time.

2. **ASR_inference (6.2%).** NeMo ASR model inference on GPU. This is the actual compute bottleneck for production workloads where data is already downloaded. Mean inference time is ~0.35s per batch of ~16 audio files. GPU utilization is the limiting factor here.

3. **All other stages (<0.1% combined).** WER computation, duration extraction, filtering, format conversion, and writing are negligible.

### FLEURS Proposed Optimizations

- **Pre-download datasets** for nightly benchmarks to isolate inference throughput from download time. The benchmark currently re-downloads on every run.
- **Increase ASR batch size** beyond 16 if GPU memory allows, to improve GPU utilization.
- **Pipeline parallelism** between download and inference stages is already handled by the executor.

## ALM Pipeline (CPU)

**Configuration:** `sample_input.jsonl` (5 entries), repeat-factor=2000 (10,000 effective entries), 120s windows, 50% overlap.

| Metric | Xenna | Ray Data |
|--------|-------|----------|
| Wall clock | 92.63s | 37.10s |
| Entries processed | 10,000 | 10,000 |
| Builder windows | 362,000 | 362,000 |
| Filtered windows | 50,000 | 50,000 |
| Throughput (entries/sec) | 107.96 | 269.55 |
| Throughput (windows/sec) | 3,908.10 | 9,757.86 |

### Per-Stage Timing (Xenna)

| Stage | Total Time | Mean/Task | Items | % of Sum |
|-------|-----------|-----------|-------|----------|
| file_partitioning | 74.76s | 0.0075s | 0* | 33.3% |
| alm_manifest_reader_stage | 30.70s | 0.0031s | 10,000 | 13.7% |
| repeat_entries | 99.28s | 0.0099s | 10,000 | 44.2% |
| alm_data_builder | 17.69s | 0.0018s | 10,000 | 7.9% |
| alm_data_overlap | 2.51s | 0.0003s | 10,000 | 1.1% |

### Per-Stage Timing (Ray Data)

| Stage | Total Time | Mean/Task | Items |
|-------|-----------|-----------|-------|
| file_partitioning | 13.30s | 0.0013s | 0* |
| alm_manifest_reader_stage | 5.61s | 0.0006s | 10,000 |
| repeat_entries | 87.40s | 0.0087s | 10,000 |
| alm_data_builder | 8.93s | 0.0009s | 10,000 |
| alm_data_overlap | 1.26s | 0.0001s | 10,000 |

### ALM Bottlenecks

1. **repeat_entries (44.2% on Xenna, highest on Ray Data too).** This is the scale-testing stage that duplicates entries in-memory. At repeat_factor=2000 it creates 10,000 entries from 5 originals. Each duplication involves copying the task data dict and metadata. This is a benchmark artifact, not a production bottleneck.

2. **file_partitioning (33.3% on Xenna).** File discovery and partitioning overhead. Significantly faster on Ray Data (13.30s vs 74.76s) suggesting Xenna's file partitioning has overhead at small file counts.

3. **alm_data_builder (7.9%).** The core windowing algorithm. Processes ~39.8 segments per entry and creates ~36.2 windows per entry. At 0.0018s/entry this is well-optimized.

4. **alm_data_overlap (1.1%).** Overlap filtering is the fastest stage. Reduces 362,000 windows to 50,000 (86% reduction).

### ALM Proposed Optimizations

- **Xenna file_partitioning overhead** is disproportionately high compared to Ray Data (74.76s vs 13.30s). Investigate whether Xenna's CompositeStage decomposition adds scheduling overhead for small manifest files.
- **repeat_entries** dominates but is a benchmark-only stage. Production workloads with real manifests of 10k+ entries would skip this entirely.
- **alm_data_builder** is already fast. Further optimization would require algorithmic changes to the windowing logic.

## Summary of Top Bottlenecks

| Pipeline | #1 Bottleneck | #2 Bottleneck | Actionable? |
|----------|--------------|--------------|-------------|
| FLEURS | Data download (91%) | ASR inference (6%) | Pre-download for benchmarks; tune batch size |
| ALM | repeat_entries (44%) | file_partitioning (33%) | Benchmark artifact; investigate Xenna overhead |

## DGX A100 Baseline (Official Benchmark Machine)

### Machine Specs

- GPU: 8× NVIDIA A100-SXM4-80GB
- CPU: 64 cores
- OS: Ubuntu, Linux 5.15

### FLEURS Pipeline (GPU)

**Configuration:** `nvidia/stt_hy_fastconformer_hybrid_large_pc`, `hy_am`, `train` split, WER threshold 5.5, 1 GPU.

| Metric | Xenna | Ray Data |
|--------|-------|----------|
| Wall clock | 100.79s | 123.53s |
| Tasks processed | 404 | 404 |
| Throughput (tasks/sec) | 4.01 | 3.27 |

### ALM Pipeline (CPU)

**Configuration:** `sample_input.jsonl` (5 entries), repeat-factor=2000 (10,000 effective entries), 120s windows, 50% overlap.

| Metric | Xenna | Ray Data |
|--------|-------|----------|
| Wall clock | 38.07s | 26.39s |
| Entries processed | 10,000 | 10,000 |
| Builder windows | 362,000 | 362,000 |
| Filtered windows | 50,000 | 50,000 |
| Throughput (entries/sec) | 262.70 | 378.91 |
| Throughput (windows/sec) | 9,509.63 | 13,716.57 |

### Comparison with Local Machine

**FLEURS:** The DGX and local FLEURS results are **not directly comparable** because the configurations differ:

| Parameter | Local | DGX |
|-----------|-------|-----|
| Model | `parakeet-tdt-0.6b-v2` (0.6B) | `stt_hy_fastconformer_hybrid_large_pc` (larger) |
| Language | `en_us` | `hy_am` |
| Split | `dev` (394 tasks) | `train` (404 tasks) |
| GPU | RTX 3080 Ti 12GB | A100-SXM4-80GB |

The DGX shows lower FLEURS throughput primarily because the nightly config uses a larger model with higher per-batch inference cost. The executor ranking also flips: Xenna is faster on DGX (4.01 vs 3.27 t/s) while Ray Data was faster locally (8.61 vs 6.49 t/s), consistent with Xenna's streaming executor better utilizing the GPU under heavier model loads. Both runs include dataset download time in wall-clock, which varies with network conditions.

**ALM:** The DGX ALM results use the same configuration as local and are directly comparable:

| Metric | Local Xenna | DGX Xenna | Local Ray Data | DGX Ray Data |
|--------|------------|-----------|---------------|-------------|
| Wall clock | 92.63s | 38.07s | 37.10s | 26.39s |
| Throughput (entries/sec) | 107.96 | 262.70 | 269.55 | 378.91 |
| Throughput (windows/sec) | 3,908.10 | 9,509.63 | 9,757.86 | 13,716.57 |

The DGX is **2.4× faster on Xenna** and **1.4× faster on Ray Data** for ALM, directly reflecting the higher core count (64 vs 16 threads). Xenna benefits more from the additional cores since its streaming executor parallelizes stages more aggressively.

## Nightly Benchmark Requirements

Based on these baselines, the following regression thresholds are set in `nightly-benchmark.yaml` (observed - 5% buffer):

| Entry | Metric | Threshold |
|-------|--------|-----------|
| audio_fleurs_xenna | is_success | true |
| audio_fleurs_raydata | is_success | true |
| alm_pipeline_xenna | is_success, total_builder_windows >= 1, total_filtered_windows >= 1 | (existing) |
| alm_pipeline_ray_data | is_success, total_builder_windows >= 1, total_filtered_windows >= 1 | (existing) |

> **Note:** Throughput `min_value` requirements are not yet set for audio entries because the FLEURS benchmark includes variable download time (network-dependent). Once a pre-downloaded dataset path is available in the nightly environment, throughput thresholds should be added matching the image/video pattern (observed - 5% buffer).
