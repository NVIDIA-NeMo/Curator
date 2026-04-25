# Audio Stage-Wise Profiling

Per-stage performance profiling for the two audio pipelines in NeMo Curator: **FLEURS** (ASR inference, GPU) and **ALM** (audio language model data curation, CPU). Results produced using the benchmark scripts in `benchmarking/scripts/` and analyzed via `TaskPerfUtils.collect_stage_metrics()`.

## Machine Specs

- CPU: Intel Core i9-9900KF @ 3.60GHz (8 cores / 16 threads)
- RAM: 32 GB
- GPU: NVIDIA GeForce RTX 3080 Ti 12 GB
- OS: Ubuntu 20.04, Linux 5.15

## FLEURS Pipeline (GPU)

**Configuration:** `nvidia/stt_hy_fastconformer_hybrid_large_pc`, `hy_am`, `dev` split, WER threshold 75.0, 1 GPU.

**Data scale:** 391 output tasks from the dev split. The pipeline downloads audio files from HuggingFace FLEURS, runs ASR inference, computes WER, filters, and writes JSONL output.

| Metric | Xenna | Ray Data |
|--------|-------|----------|
| Wall clock | 42.78s | 27.62s |
| Tasks processed | 391 | 391 |
| Throughput (tasks/sec) | 9.14 | 14.16 |

### FLEURS Per-Stage Breakdown

| Stage | Xenna Total (s) | Xenna % | Ray Data Total (s) | Ray Data % |
|-------|-----------------|---------|-------------------|------------|
| CreateInitialManifestFleurs | 2,437.05 | 96.2% | 1,311.64 | 93.4% |
| ASR_inference | 96.95 | 3.8% | 91.64 | 6.5% |
| GetPairwiseWerStage | 0.01 | <0.1% | 0.01 | <0.1% |
| GetAudioDurationStage | 0.25 | <0.1% | 0.20 | <0.1% |
| PreserveByValueStage | 0.01 | <0.1% | 0.00 | <0.1% |
| AudioToDocumentStage | 0.12 | <0.1% | 0.09 | <0.1% |
| jsonl_writer | 0.21 | <0.1% | 0.13 | <0.1% |

### FLEURS Bottlenecks

1. **CreateInitialManifestFleurs (96.2% on Xenna, 93.4% on Ray Data).** This stage downloads audio files from the HuggingFace FLEURS dataset and creates the initial manifest. The high per-task time (~6.2s on Xenna, ~3.4s on Ray Data) is dominated by network I/O and audio file extraction. In a benchmark context this is a one-time cost per run and does not reflect production throughput, but it is the dominant contributor to wall clock time.

2. **ASR_inference (3.8% on Xenna, 6.5% on Ray Data).** NeMo ASR model inference on GPU. This is the actual compute bottleneck for production workloads where data is already downloaded. Mean inference time is ~0.25s per batch. GPU utilization is the limiting factor here.

3. **All other stages (<0.1% combined).** WER computation, duration extraction, filtering, format conversion, and writing are negligible.

### FLEURS Proposed Optimizations

- **Pre-download datasets** for nightly benchmarks to isolate inference throughput from download time. The benchmark currently re-downloads on every run.
- **Increase ASR batch size** if GPU memory allows, to improve GPU utilization.
- **Pipeline parallelism** between download and inference stages is already handled by the executor.

## ALM Pipeline (CPU)

**Configuration:** `sample_input.jsonl` (5 entries), repeat-factor=2000 (10,000 effective entries), 120s windows, 50% overlap.

**Data scale:** 5 base entries totalling 3,162.5s (0.88 hours), 199 segments. After 2000x repeat: 10,000 entries (1,757 effective hours). Pipeline produces 362,000 builder windows and 50,000 filtered windows (6,071,000s total filtered duration).

| Metric | Xenna (streaming) | Xenna (batch) | Ray Data |
|--------|-------------------|---------------|----------|
| Wall clock | 94.79s | 63.65s | 27.65s |
| Entries processed | 10,000 | 10,000 | 10,000 |
| Builder windows | 362,000 | 362,000 | 362,000 |
| Filtered windows | 50,000 | 50,000 | 50,000 |
| Throughput (entries/sec) | 105.50 | 157.10 | 361.70 |
| Throughput (windows/sec) | 3,819.09 | 5,687.17 | 13,093.50 |

### ALM Per-Stage Breakdown

| Stage | Xenna Streaming | % | Xenna Batch | % | Ray Data | % |
|-------|----------------|---|-------------|---|----------|---|
| file_partitioning | 42.70s | 19.0% | 18.25s | 6.4% | 15.31s | 12.7% |
| alm_manifest_reader | 34.78s | 15.4% | 30.29s | 10.6% | 6.03s | 5.0% |
| repeat_entries | 127.09s | 56.4% | 221.70s | 77.6% | 89.01s | 73.9% |
| alm_data_builder | 18.06s | 8.0% | 13.70s | 4.8% | 8.92s | 7.4% |
| alm_data_overlap | 2.54s | 1.1% | 1.61s | 0.6% | 1.25s | 1.0% |

### ALM Bottlenecks

1. **repeat_entries (56.4% streaming, 77.6% batch, 73.9% Ray Data).** This is the scale-testing stage that duplicates entries in-memory. At repeat_factor=2000 it creates 10,000 entries from 5 originals. Each duplication involves copying the task data dict and metadata. This is a benchmark artifact, not a production bottleneck.

2. **file_partitioning (19.0% streaming, 6.4% batch, 12.7% Ray Data).** File discovery and partitioning overhead. Xenna streaming is the slowest (42.70s) due to autoscaler overhead for a single-file manifest. Batch mode reduces this to 18.25s by avoiding autoscaler polling. Ray Data is comparable at 15.31s.

3. **alm_manifest_reader (15.4% streaming, 10.6% batch, 5.0% Ray Data).** Reading the JSONL manifest and creating AudioTasks. Xenna streaming is 5-6x slower than Ray Data here (34.78s vs 6.03s) due to per-stage actor startup overhead.

4. **alm_data_builder (8.0%).** The core windowing algorithm. Processes ~39.8 segments per entry and creates ~36.2 windows per entry. At 0.0009-0.0018s/entry this is well-optimized.

5. **alm_data_overlap (1.1%).** Overlap filtering is the fastest stage. Reduces 362,000 windows to 50,000 (86% reduction).

### ALM Execution Mode Analysis

For CPU-only pipelines like ALM, the choice of executor and execution mode has significant impact:

- **Xenna streaming** is the slowest due to autoscaler polling (180s interval) and per-stage actor lifecycle overhead. The autoscaler is designed for GPU-heavy pipelines where dynamic worker adjustment is valuable.
- **Xenna batch** is ~1.5x faster than streaming. Fixed allocation eliminates autoscaler overhead, and file_partitioning drops from 42.70s to 18.25s.
- **Ray Data** is ~3.4x faster than streaming and ~2.3x faster than batch. It chains all stages into a single streaming execution plan with operator fusion, avoiding per-stage actor creation/teardown entirely.

For GPU-heavy pipelines (like FLEURS with ASR inference), Xenna streaming mode is recommended as its autoscaler dynamically adjusts GPU worker counts.

### ALM Proposed Optimizations

- **Xenna file_partitioning + manifest_reader overhead** is disproportionately high compared to Ray Data. The combined overhead is 77.48s (streaming) vs 21.34s (Ray Data). Batch mode helps (48.54s) but Ray Data's operator fusion remains superior.
- **repeat_entries** dominates but is a benchmark-only stage. Production workloads with real manifests of 10k+ entries would skip this entirely.
- **alm_data_builder** is already fast. Further optimization would require algorithmic changes to the windowing logic.

## Summary of Top Bottlenecks

| Pipeline | #1 Bottleneck | #2 Bottleneck | Actionable? |
|----------|--------------|--------------|-------------|
| FLEURS | Data download (96%) | ASR inference (4%) | Pre-download for benchmarks; tune batch size |
| ALM | repeat_entries (56-78%) | file_partitioning (6-19%) | Benchmark artifact; use batch mode or Ray Data for CPU-only pipelines |

## Nightly Benchmark Requirements

Based on these baselines, the following regression thresholds are set in `nightly-benchmark.yaml` (observed - 5% buffer):

| Entry | Metric | Threshold |
|-------|--------|-----------|
| audio_fleurs_xenna | is_success | true |
| audio_fleurs_raydata | is_success | true |
| alm_pipeline_xenna | is_success, total_builder_windows >= 1, total_filtered_windows >= 1 | (existing) |
| alm_pipeline_ray_data | is_success, total_builder_windows >= 1, total_filtered_windows >= 1 | (existing) |

> **Note:** Throughput `min_value` requirements are not yet set for audio entries because the FLEURS benchmark includes variable download time (network-dependent). Once a pre-downloaded dataset path is available in the nightly environment, throughput thresholds should be added matching the image/video pattern (observed - 5% buffer).
