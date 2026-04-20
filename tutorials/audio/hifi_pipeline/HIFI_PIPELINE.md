## Summary

End-to-end HIFI data curation pipeline for speech audio: SED inference, SED postprocessing, segment extraction, diarization, three-pass transcription cascade (Qwen3-Omni ASR + verification + LLM PnC), speaker embedding + clustering, and UTMOS quality scoring.

## Pipeline stages

```
SED -> SED postprocess -> segment extract -> diarize ->
transcribe (3-pass cascade) -> speaker embed -> group by video ->
speaker cluster -> utmos
```

| # | Stage | Compute | Description |
|---|-------|---------|-------------|
| 1 | **SEDInferenceStage** | GPU, batch | PANNs CNN14 sound event detection, outputs per-frame probability matrix (T x 527 AudioSet classes) as NPZ sidecar files |
| 2 | **SEDPostprocessingStage** | CPU, 1:1 | Aggregates speech-class probabilities, applies threshold/hysteresis/smoothing/merging to detect clean-speech events |
| 3 | **SegmentExtractorStage** | CPU, 1:N | Fan-out: SED events -> individual AudioTasks with start/end timestamps |
| 4 | **InferenceSortformerStage** | GPU | Sortformer diarization: speaker segmentation and labeling |
| 5 | **TranscriptionCascadeStage** | GPU, 3-pass | Three sequential LLM passes via YAML prompt configs (see below) |
| 6 | **SpeakerEmbeddingLhotseStage** | GPU, batch | TitaNet-Large 192-dim speaker embeddings via NeMo |
| 7a | **GroupByVideoStage** | CPU, 1:1 | Resolves and annotates each row with a `video_id` extracted from manifest fields (`id`, `youtube_id`, `audio_item_id`) or regex on `audio_filepath`. Enables per-video clustering downstream |
| 7b | **SpeakerClusteringStage** | CPU | Agglomerative Hierarchical Clustering with cosine similarity. Supports global, shard-level, and grouped (`batch_size`) modes. Uses `video_id` for per-video clustering when available |
| 8 | **GetUtmosv2ScoreStage** | GPU, 1:1 | UTMOSv2 Mean Opinion Score per utterance. Supports local WAV, remote audio (s3://, ais://), and any ffmpeg-supported format (opus, m4a, flac, mp3, etc.) |

### Transcription cascade passes

| Pass | Model | Input | Output |
|------|-------|-------|--------|
| 1 | Qwen3-Omni | Audio-only ASR + number normalization | `qwen3_omni_pred_text` |
| 2 | Qwen3-Omni | Audio + draft text verification | `qwen3_omni_verified_text` |
| 3 | Qwen3-LLM | Text-only punctuation & capitalization | `qwen3_llm_corrected_text` |

## Docker containers

Two container images, each targeting a subset of pipeline stages:

### `curator-hifi-pipeline` (full pipeline with vLLM)

Base: `qwenllm/qwen3-omni` (~35 GB). Includes Qwen3-Omni vLLM for transcription cascade.

```bash
# Build
docker build -t curator-hifi-pipeline -f tutorials/audio/hifi_pipeline/Dockerfile .

# Run (all stages, server-based transcription)
docker run --gpus all --ipc=host --shm-size=8g \
  -v /path/to/data:/data -v /path/to/models:/models \
  curator-hifi-pipeline \
  --input_manifest /data/manifest.jsonl \
  --stages sed,sed_post,segment,diarize,transcribe,embed,cluster,utmos \
  --sed_checkpoint /models/Cnn14_DecisionLevelMax.pth \
  --language Ru

# Run (in-process vLLM via XennaExecutor)
docker run --gpus all --ipc=host --shm-size=8g \
  -v /path/to/data:/data \
  curator-hifi-pipeline \
  --data_config /data/granary.yaml \
  --stages transcribe \
  --language Ru
```

### `curator-nemo-stages` (SED + diarize + speaker ID + UTMOS)

Base: `pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime` (~8.5 GB). No vLLM -- for stages that don't need LLM inference.

```bash
# Build
docker build -t curator-nemo-stages -f tutorials/audio/hifi_pipeline/Dockerfile.nemo .

# Run
docker run --gpus all --ipc=host --shm-size=8g \
  -v /path/to/data:/data -v /path/to/models:/models \
  curator-nemo-stages \
  --input_manifest /data/manifest.jsonl \
  --stages sed,sed_post,diarize,embed,cluster,utmos

# UTMOS only (with utmosv2 installed at runtime)
docker run --gpus all --ipc=host --shm-size=8g \
  --entrypoint bash \
  -v /path/to/Curator:/opt/Curator \
  -v /path/to/data:/data \
  curator-nemo-stages \
  -c "apt-get update -qq && apt-get install -y -qq wget >/dev/null 2>&1 && \
      pip install -q 'utmosv2 @ git+https://github.com/sarulab-speech/UTMOSv2.git' && \
      PYTHONPATH=/opt/Curator python /opt/Curator/tutorials/audio/hifi_pipeline/run_pipeline.py \
        --input_manifest /data/manifest.jsonl \
        --output_dir /data/output \
        --stages utmos"
```

### Stage-to-container mapping

| Stage | `curator-hifi-pipeline` | `curator-nemo-stages` |
|-------|:-----------------------:|:---------------------:|
| sed, sed_post, segment | yes | yes |
| diarize | yes | yes |
| transcribe (server) | yes | -- |
| transcribe (in-process) | yes | -- |
| embed, cluster | yes | yes |
| utmos | yes* | yes* |

\* Requires `utmosv2` package installed at runtime (not baked into either image).

## UTMOS scoring details

`GetUtmosv2ScoreStage` computes UTMOSv2 MOS quality scores (1-5 scale) per audio entry.

### Audio input modes

**Mode 1 — In-memory waveform (AIS streaming, preferred for large datasets):**

When upstream is `NemoTarredAudioReader`, each `AudioTask` carries a `waveform` (numpy array) and `sample_rate` decoded in memory from NeMo tar archives streamed via AIS. No files are downloaded to disk. The stage resamples to 16kHz, writes a temp WAV for UTMOSv2, scores, and discards. Use `--data_config` to activate this mode.

```bash
python run_pipeline.py \
    --data_config /path/to/granary.yaml \
    --stages utmos \
    --output_dir /output
```

**Mode 2 — File path (local or remote):**

When reading from JSONL manifests, the stage resolves `audio_filepath` to local or remote files. Supports `s3://`, `ais://`, `http(s)://` via CLI download, and any ffmpeg-supported format (opus, m4a, flac, mp3, etc.).

```bash
python run_pipeline.py \
    --input_manifest /data/manifest.jsonl \
    --stages utmos \
    --output_dir /output
```

**Mode 3 — Mixed:** A `DocumentBatch` may contain entries with waveforms and entries with file paths; each is handled appropriately.

### GPU-accelerated spectrograms (Slurm scoring)

The standalone `score_utmos_shard.py` script patches UTMOSv2 to compute mel spectrograms on GPU via `torchaudio` instead of the default CPU-based `librosa`. This avoids the main bottleneck in UTMOSv2's fusion model (16 mel spectrograms per sample computed on CPU). The spectrogram computation is moved from the dataset `__getitem__` (CPU DataLoader) to the model `forward` pass (GPU), yielding ~35x speedup with negligible score deviation (correlation > 0.999 vs unpatched).

| Approach | Time per shard (~1400 files) |
|----------|----------------------------|
| Default librosa (CPU) | ~35 min |
| torchaudio (CPU replacement) | ~13 min |
| **torchaudio (GPU, used in script)** | **~2 min** |

### Dedicated UTMOS container

`Dockerfile.utmos` provides a lightweight container with UTMOSv2 + torchaudio + aistore SDK:

```bash
docker build -t curator-utmos -f tutorials/audio/hifi_pipeline/Dockerfile.utmos .
```

## CLI reference

```bash
python run_pipeline.py \
    --input_manifest /data/manifest.jsonl \
    --output_dir /data/output \
    --stages sed,sed_post,segment,diarize,transcribe,embed,group_video,cluster,utmos \
    --language Ru \
    --sed_checkpoint /models/Cnn14_DecisionLevelMax.pth \
    --vllm_host localhost --vllm_port 8200 \
    --omni_model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --llm_model Qwen/Qwen3-30B-A3B-Instruct \
    --cluster_threshold 0.292 \
    --cluster_batch_size 64 \
    --utmos_batch_size 16
```

Stages can be run selectively: `--stages utmos` runs only UTMOS scoring.

## Prompt configs

YAML-based prompt templates in `stages/audio/request/prompts/{lang}/`:

| Language | 1st pass (ASR) | 2nd pass (verify) | 3rd pass (PnC) |
|----------|---------------|-------------------|----------------|
| En | `En/1st_pass.yaml` | `En/2nd_pass.yaml` | `En/3_llm_pnc.yaml` |
| Fi | `Fi/1st_pass.yaml` | `Fi/2nd_pass.yaml` | `Fi/3_llm_pnc.yaml` |
| Ru | `Ru/1st_pass.yaml` | `Ru/2nd_pass.yaml` | `Ru/3_llm_pnc.yaml` |
| Uk | `Uk/1st_pass.yaml` | `Uk/2nd_pass.yaml` | `Uk/3_llm_pnc.yaml` |

## Slurm speaker ID tooling (`slurm_speaker_id/`)

| Script | Purpose |
|--------|---------|
| `extract_shard_embeddings.py` | Per-shard TitaNet embedding extraction from AIS-backed NeMo tars |
| `submit_embeddings.sh` | Slurm array job submission with chunking for >1000 shards |
| `cluster_global.py` | Dense AHC global clustering (grouped by 64 shards) |
| `cluster_global_v2.py` | Memory-efficient FAISS-based clustering: O(N*k) instead of O(N^2) |
| `cluster_by_video.py` | Per-video-ID clustering with global-unique speaker labels |
| `submit_clustering_grouped.sh` | Grouped clustering Slurm submission |
| `submit_clustering.sh` / `submit_clustering_by_video.sh` | Other clustering submission variants |
| `run_clustering.py` | Clustering orchestrator using Curator's `SpeakerClusteringStage` |

## Library modules

| Module | Purpose |
|--------|---------|
| `stages/audio/inference/sed.py` | SED inference stage wrapping PANNs CNN14 |
| `stages/audio/inference/sed_models/cnn14.py` | CNN14 / CNN14-DecisionLevelMax model definitions |
| `stages/audio/inference/transcription_cascade_inprocess.py` | In-process vLLM 3-pass transcription cascade |
| `stages/audio/postprocessing/sed_postprocessing.py` | SED postprocessing: threshold, hysteresis, merging |
| `stages/audio/postprocessing/sed_utils.py` | SED utilities: smoothing, AudioSet class mapping |
| `stages/audio/request/prompt_template.py` | YAML prompt config loader and renderer |
| `stages/audio/request/text_only_llm_request.py` | Text-only LLM request stage (PnC / standalone) |
| `stages/audio/request/transcription_cascade.py` | Three-pass transcription cascade orchestrator |
| `stages/audio/segmentation/segment_extractor.py` | Fan-out stage: SED events -> individual AudioTasks |
| `stages/audio/preprocessing/group_by_video.py` | Resolve and annotate video ID from manifest fields or audio filepath |
| `stages/audio/metrics/utmosv2_score.py` | UTMOSv2 MOS scoring (local + remote + any format) |
| `stages/audio/filtering/utmos.py` | Quality filtering by MOS threshold |

## Tests

| Test file | Covers |
|-----------|--------|
| `test_sed.py` | SED inference stage |
| `test_sed_postprocessing.py` | SED postprocessing: thresholds, merging, edge cases |
| `test_prepare_omni_request.py` | Omni request preparation |
| `test_prompt_template.py` | YAML prompt loading and rendering |
| `test_transcription_cascade.py` | Three-pass cascade orchestration |
| `test_segment_extractor.py` | Segment fan-out from SED events |
