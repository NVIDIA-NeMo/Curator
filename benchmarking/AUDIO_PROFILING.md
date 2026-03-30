# Audio Stage-Wise Profiling

Per-stage performance profiling for the two audio pipelines in NeMo Curator: **FLEURS** (ASR inference, GPU) and **ALM** (audio language model data curation, CPU). Results produced using the benchmark scripts in `benchmarking/scripts/` and analyzed via `TaskPerfUtils.collect_stage_metrics()`.

## Machine Specs

- GPU: 8× NVIDIA A100-SXM4-80GB
- CPU: 64 cores
- OS: Ubuntu, Linux 5.15

## FLEURS Pipeline (GPU)

**Configuration:** `nvidia/stt_hy_fastconformer_hybrid_large_pc`, `hy_am`, `train` split, WER threshold 5.5, 1 GPU.

| Metric | Xenna | Ray Data |
|--------|-------|----------|
| Wall clock | 100.79s | 123.53s |
| Tasks processed | 404 | 404 |
| Throughput (tasks/sec) | 4.01 | 3.27 |

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
| Wall clock | 38.07s | 26.39s |
| Entries processed | 10,000 | 10,000 |
| Builder windows | 362,000 | 362,000 |
| Filtered windows | 50,000 | 50,000 |
| Throughput (entries/sec) | 262.70 | 378.91 |
| Throughput (windows/sec) | 9,509.63 | 13,716.57 |

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

## Nightly Benchmark Requirements

Based on these baselines, the following regression thresholds are set in `nightly-benchmark.yaml` (observed - 5% buffer):

| Entry | Metric | Threshold |
|-------|--------|-----------|
| audio_fleurs_xenna | is_success | true |
| audio_fleurs_raydata | is_success | true |
| alm_pipeline_xenna | is_success, total_builder_windows >= 1, total_filtered_windows >= 1 | (existing) |
| alm_pipeline_ray_data | is_success, total_builder_windows >= 1, total_filtered_windows >= 1 | (existing) |

> **Note:** Throughput `min_value` requirements are not yet set for audio entries because the FLEURS benchmark includes variable download time (network-dependent). Once a pre-downloaded dataset path is available in the nightly environment, throughput thresholds should be added matching the image/video pattern (observed - 5% buffer).
