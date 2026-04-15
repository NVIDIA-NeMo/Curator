## Summary

- Add end-to-end TTS data curation pipeline stages: SED inference, SED postprocessing, segment extraction, three-pass transcription cascade (Qwen3-Omni ASR + verification + LLM PnC), text-only LLM request, and speaker clustering improvements.
- Add YAML-based prompt configs for four languages (En, Fi, Ru, Uk) with three passes each (ASR, verification, punctuation & capitalization).
- Add `batch_size` parameter to `SpeakerClusteringStage` for grouped clustering -- clusters shards in groups of N with globally unique speaker labels, controlling memory vs. clustering scope trade-off.
- Integrate existing `GetUtmosv2ScoreStage` (MOS quality scoring) and `UTMOSFilterStage` (quality filtering) from dev branch into the pipeline.
- Add Slurm-based speaker ID tooling for large-scale embedding extraction and global clustering on AIS-backed NeMo tarred audio datasets.

## Pipeline stages

1. **SEDInferenceStage** (GPU, batch) -- runs PANNs CNN14 on audio, outputs per-frame class probability matrix (T x 527 AudioSet classes) as compressed NPZ sidecar files.
2. **SEDPostprocessingStage** (CPU, 1:1) -- reads NPZ probability matrices, aggregates speech-class probabilities (noisy-or), applies threshold/hysteresis/smoothing/merging to detect clean-speech events. Optional GBT classifier for event filtering.
3. **SegmentExtractorStage** (CPU, 1:N fan-out) -- takes `predicted_events` from SED postprocessing and produces one `AudioTask` per speech event with start/end timestamps. Same fan-out pattern as `VADSegmentationStage`.
4. **TranscriptionCascadeStage** (GPU, 3-pass) -- orchestrates three sequential LLM passes using YAML prompt configs:
   - **Pass 1** (Qwen3-Omni): Audio-only ASR + number normalization → `qwen3_omni_pred_text`
   - **Pass 2** (Qwen3-Omni): Audio + draft text verification → `qwen3_omni_verified_text`
   - **Pass 3** (Qwen3-LLM): Text-only punctuation & capitalization → `qwen3_llm_corrected_text`
5. **TextOnlyLLMRequestStage** (GPU, 1:1) -- sends text-only messages to an LLM via OpenAI API for punctuation, capitalization, and text refinement. Used standalone or as Pass 3 of the cascade.
6. **GetUtmosv2ScoreStage** (GPU, 1:1) -- compute UTMOSv2 MOS quality score per utterance.
7. **UTMOSFilterStage** (CPU, 1:1) -- filter utterances by MOS quality threshold.
8. **SpeakerEmbeddingLhotseStage** (GPU, batch) -- TitaNet-Large 192-dim speaker embeddings via NeMo `EncDecSpeakerLabelModel`, saved as per-shard NPZ files.
9. **SpeakerClusteringStage** (CPU) -- Agglomerative Hierarchical Clustering (AHC) with cosine similarity threshold. Supports three modes: global (all shards), shard-level (per shard), and grouped via `batch_size` (N shards per group with globally unique labels). Writes manifest with `speaker_label` and `confidence_score` per utterance.

## New library modules

| Module | Purpose |
|--------|---------|
| `stages/audio/inference/sed.py` | SED inference stage wrapping PANNs CNN14 |
| `stages/audio/inference/sed_models/cnn14.py` | CNN14 / CNN14-DecisionLevelMax model definitions |
| `stages/audio/postprocessing/sed_postprocessing.py` | SED postprocessing: threshold, hysteresis, merging |
| `stages/audio/postprocessing/sed_utils.py` | SED utilities: smoothing, AudioSet class mapping |
| `stages/audio/request/prompt_template.py` | YAML prompt config loader and renderer |
| `stages/audio/request/text_only_llm_request.py` | Text-only LLM request stage (PnC / standalone) |
| `stages/audio/request/transcription_cascade.py` | Three-pass transcription cascade orchestrator |
| `stages/audio/segmentation/segment_extractor.py` | Fan-out stage: SED events → individual AudioTasks |

## Modified modules

| Module | Change |
|--------|--------|
| `stages/audio/speaker_id/speaker_clustering_and_scoring.py` | Add `batch_size` parameter for grouped clustering; graceful handling of missing embeddings (`speaker_label=-1, confidence_score=0.0`) |
| `stages/audio/__init__.py` | Export new stages |
| `stages/audio/request/__init__.py` | Export new request stages |
| `stages/audio/segmentation/__init__.py` | Export `SegmentExtractorStage` |

## Prompt configs

YAML-based prompt templates in `stages/audio/request/prompts/{lang}/`:

| Language | 1st pass (ASR) | 2nd pass (verify) | 3rd pass (PnC) |
|----------|---------------|-------------------|----------------|
| En | `En/1st_pass.yaml` | `En/2nd_pass.yaml` | `En/3_llm_pnc.yaml` |
| Fi | `Fi/1st_pass.yaml` | `Fi/2nd_pass.yaml` | `Fi/3_llm_pnc.yaml` |
| Ru | `Ru/1st_pass.yaml` | `Ru/2nd_pass.yaml` | `Ru/3_llm_pnc.yaml` |
| Uk | `Uk/1st_pass.yaml` | `Uk/2nd_pass.yaml` | `Uk/3_llm_pnc.yaml` |

## Slurm speaker ID tooling (`tutorials/audio/tts_pipeline/slurm_speaker_id/`)

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
| `check_status.sh` | Monitor embedding/clustering progress per corpus |
| `auto_submit_1bw.sh` | Auto-retry submission for large corpora |

## Tutorial pipeline runner

`tutorials/audio/tts_pipeline/run_pipeline.py` -- end-to-end CLI that chains all stages:

```
SED → SED postprocess → segment extract → diarize →
transcribe (3-pass cascade) → speaker embed → speaker cluster
```

Supports selective stage execution via `--stages sed,sed_post,segment,...`.

## Tests

| Test file | Covers |
|-----------|--------|
| `test_sed.py` | SED inference stage |
| `test_sed_postprocessing.py` | SED postprocessing: thresholds, merging, edge cases |
| `test_prepare_omni_request.py` | Omni request preparation |
| `test_prompt_template.py` | YAML prompt loading and rendering |
| `test_transcription_cascade.py` | Three-pass cascade orchestration |
| `test_segment_extractor.py` | Segment fan-out from SED events |

## Usage

```python
from nemo_curator.stages.audio.inference.sed import SEDInferenceStage, SEDConfig
from nemo_curator.stages.audio.postprocessing.sed_postprocessing import SEDPostprocessingStage
from nemo_curator.stages.audio.segmentation.segment_extractor import SegmentExtractorStage
from nemo_curator.stages.audio.request.transcription_cascade import TranscriptionCascadeStage
from nemo_curator.stages.audio.metrics.utmosv2_score import GetUtmosv2ScoreStage
from nemo_curator.stages.audio.filtering.utmos import UTMOSFilterStage
from nemo_curator.stages.audio.speaker_id.speaker_clustering_and_scoring import SpeakerClusteringStage

# SED + postprocessing + segment extraction
sed = SEDInferenceStage(config=SEDConfig(checkpoint_path="/models/Cnn14_DecisionLevelMax.pth"))
sed_post = SEDPostprocessingStage()
segmenter = SegmentExtractorStage()

# Three-pass transcription cascade
cascade = TranscriptionCascadeStage(
    language="Ru",
    omni_model="Qwen/Qwen3-Omni",
    llm_model="Qwen/Qwen3-235B-A22B",
    vllm_host="localhost",
    vllm_port=8200,
)

# UTMOS quality scoring + filtering
utmos_scorer = GetUtmosv2ScoreStage()
utmos_filter = UTMOSFilterStage(mos_threshold=3.5)

# Speaker clustering -- grouped mode (64 shards per group)
clustering = SpeakerClusteringStage(
    input_manifest="/data/manifests/manifest_{0..255}.jsonl",
    embedding_dir="/data/embeddings",
    output_manifest_dir="/data/output_manifests",
    batch_size=64,
    threshold=0.292,
)
```
