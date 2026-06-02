# DNS Challenge Read Speech Pipeline

Process the DNS Challenge Read Speech dataset using NeMo Curator's audio pipeline with automatic download support.

The pipeline downloads the dataset (4.88 GB compressed, 14,279 WAV files at 48kHz, 19.3 hours total audio) and applies quality filtering.

## Prerequisites

- **Python**: 3.10+
- **GPU**: Optional but recommended (see [GPU Memory Requirements](#gpu-memory-requirements))

### System Dependencies

No system-level packages are required beyond Python. All audio I/O is handled by `soundfile` and `torchaudio` (bundled via pip).

### Python Dependencies

Install NeMo Curator with audio dependencies using [uv](https://docs.astral.sh/uv/):

```bash
# GPU (recommended)
uv sync --extra audio_cuda12

# CPU only
uv sync --extra audio_cpu
```

The full pipeline requires: `soundfile`, `torchaudio`, `librosa`, `scipy`, `pydub`, `onnxruntime`/`onnxruntime-gpu`, `silero-vad`, and `nemo_toolkit[asr]`. These are all included in the `audio_cuda12` / `audio_cpu` extras.

## Quick Start

### Validate Your Setup (~2–3 minutes)

Run a minimal end-to-end test with 10 samples to confirm everything is installed correctly:

```bash
# Quick check with UTMOS + VAD only (~1–2 minutes)
python pipeline.py \
    --raw_data_dir ./dns_data \
    --max-samples 10 \
    --enable-utmos \
    --enable-vad

# Full validation with all filters (~2–3 minutes)
python pipeline.py \
    --raw_data_dir ./dns_data \
    --max-samples 10 \
    --enable-utmos \
    --enable-vad \
    --enable-sigmos \
    --enable-band-filter \
    --enable-speaker-separation
```

This downloads the dataset on first run, processes only 10 files, and writes results to `./dns_data/result/`. Expected wall-clock time with all filters: **~2–3 minutes** (model downloads add negligible overhead; runtime is dominated by model loading and initialization).

### Standard Usage

```bash
# Auto-download dataset and process (default: 5000 samples)
python pipeline.py \
    --raw_data_dir ./dns_data \
    --enable-utmos \
    --enable-vad

# Process all 14,279 files
python pipeline.py \
    --raw_data_dir ./dns_data \
    --max-samples -1 \
    --enable-utmos \
    --enable-vad

# Use pre-downloaded data
python pipeline.py \
    --raw_data_dir /path/to/existing/read_speech \
    --no-auto-download \
    --enable-utmos
```

## Dataset Overview

**DNS Challenge 5 - Read Speech (Track 1 Headset)**

- **Source**: [Microsoft DNS Challenge](https://github.com/microsoft/DNS-Challenge)
- **Format**: WAV files (mono or stereo), 48,000 Hz
- **License**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Download size**: 4.88 GB (compressed)
- **Extracted size**: 6.3 GB
- **Files**: 14,279 WAV files
- **Total duration**: 19.3 hours (~69,578 seconds)
- **Avg duration per file**: 4.9 seconds
- **Unique readers**: 318
- **Unique books**: 262

### Dataset Structure

```
raw_data_dir/
└── read_speech/          # auto-extracted from archive
    ├── book_00000_chp_0009_reader_06709_0_seg_1_seg1.wav
    ├── book_00000_chp_0009_reader_06709_0_seg_2_seg1.wav
    └── ... (14,279 WAV files)
```

## Pipeline Architecture

The pipeline supports four topologies based on which features are enabled:

| Topology | Flags | Output (per input file) |
|----------|-------|------------------------|
| Combo 1 | *(none)* | 1 row with whole-file scores |
| Combo 2 | `--enable-vad` | N rows, one per speech segment |
| Combo 3 | `--enable-speaker-separation` | K rows, one per speaker with diarization timestamps |
| Combo 4 | `--enable-vad --enable-speaker-separation` | K*M rows, one per speaker-segment |

```
CreateInitialManifestReadSpeechStage
  Downloads and scans read_speech directory, parses filenames
      |
      v
AudioDataFilterStage (auto-selects topology)
  Combo 1: MonoConversion -> Filters -> TimestampMapper
  Combo 2: MonoConversion -> VAD(fan-out) -> Filters -> TimestampMapper
  Combo 3: MonoConversion -> Filters -> SpeakerSep(fan-out) -> Filters -> TimestampMapper
  Combo 4: MonoConversion -> VAD(nested) -> Filters -> SegmentConcat
            -> SpeakerSep -> VAD_Speaker(fan-out) -> Filters -> TimestampMapper
      |
      v
AudioToDocumentStage -> JsonlWriter
  Output: manifest.jsonl
```

## Running the Pipeline

### Option 1: Python Script (`pipeline.py`)

```bash
# With all filters
python pipeline.py \
    --raw_data_dir ./dns_data \
    --enable-vad \
    --enable-utmos \
    --enable-sigmos \
    --enable-band-filter \
    --enable-speaker-separation

# Process all 14,279 files
python pipeline.py \
    --raw_data_dir ./dns_data \
    --max-samples -1 \
    --enable-utmos \
    --enable-sigmos
```

### Option 2: YAML Config (`run.py`)

```bash
# Default (all 14,279 files as configured in pipeline.yaml)
python run.py \
    --config-path . \
    --config-name pipeline.yaml \
    raw_data_dir=./dns_data

# Limit to 5000 samples
python run.py \
    --config-path . \
    --config-name pipeline.yaml \
    raw_data_dir=./dns_data \
    max_samples=5000
```

## Command Line Options

### Required

| Option          | Description                                          |
|-----------------|------------------------------------------------------|
| `--raw_data_dir` | Directory for data download or path to existing data |

### Download Settings

| Option               | Default | Description                        |
|----------------------|---------|------------------------------------|
| `--auto-download`    | `true`  | Auto-download dataset (~4.88 GB)   |
| `--no-auto-download` |         | Disable auto-download              |

### Processing

| Option           | Default              | Description                                    |
|------------------|----------------------|------------------------------------------------|
| `--output_dir`   | `{raw_data_dir}/result` | Output directory                            |
| `--max-samples`  | `5000`               | Max samples (-1 for all 14,279 files)          |
| `--batch_size`   | `1`                  | Batch size                                     |
| `--sample_rate`  | `48000`              | Audio sample rate                              |
| `--clean`        | `false`              | Clean output dir                               |
| `--backend`      | `xenna`              | Execution backend: `xenna` or `ray_data`       |
| `--verbose`      | `false`              | DEBUG logging                                  |

### Filter Toggles and Thresholds

| Option                          | Default     | Description                          |
|---------------------------------|-------------|--------------------------------------|
| `--enable-vad`                  | `false`     | Enable VAD segmentation              |
| `--vad-min-duration`            | `2.0`       | Min segment (sec)                    |
| `--vad-max-duration`            | `60.0`      | Max segment (sec)                    |
| `--vad-threshold`               | `0.5`       | VAD threshold (0–1)                  |
| `--vad-min-interval-ms`         | `500`       | Min silence to split segments (ms)   |
| `--vad-speech-pad-ms`           | `300`       | Padding before/after speech (ms)     |
| `--enable-utmos`                | `false`     | Enable UTMOS filter                  |
| `--utmos-mos-threshold`         | `3.4`       | Min UTMOS MOS (0–5)                  |
| `--enable-sigmos`               | `false`     | Enable SIGMOS filter                 |
| `--sigmos-noise-threshold`      | `4.0`       | Min SIGMOS noise (0–5)               |
| `--sigmos-ovrl-threshold`       | `3.5`       | Min SIGMOS overall (0–5)             |
| `--enable-band-filter`          | `false`     | Enable band filter                   |
| `--band-value`                  | `full_band` | Band type to pass                    |
| `--enable-speaker-separation`   | `false`     | Enable speaker diarization           |
| `--speaker-exclude-overlaps`    | `true`      | Exclude overlapping speech           |
| `--no-speaker-exclude-overlaps` |             | Allow overlapping speaker segments   |
| `--speaker-min-duration`        | `0.8`       | Min speaker segment (sec)            |

## Parameter Tuning

All filter thresholds and resource allocations (CPU/GPU per stage) can also be customized directly in the [default config file](https://github.com/NVIDIA-NeMo/Curator/blob/main/nemo_curator/stages/audio/advanced_pipelines/audio_data_filter/default_config.yaml) without changing code:

```
nemo_curator/stages/audio/advanced_pipelines/audio_data_filter/default_config.yaml
```

This YAML controls thresholds, enable/disable flags, and `cpus`/`gpus` resource allocation for each stage. Any values you pass via CLI or `pipeline.yaml` override these defaults.

### UTMOS MOS Threshold (`--utmos-mos-threshold`)

UTMOS predicts a [Mean Opinion Score (MOS)](https://en.wikipedia.org/wiki/Mean_opinion_score) on a 0–5 scale:

| MOS Range   | Quality   | Description                               |
|-------------|-----------|-------------------------------------------|
| 4.0 – 5.0  | Excellent | Broadcast quality, minimal artifacts      |
| 3.5 – 4.0  | Good      | Clear speech, minor imperfections         |
| 3.0 – 3.5  | Fair      | Noticeable distortion but intelligible    |
| 2.0 – 3.0  | Poor      | Significant quality issues                |
| 0.0 – 2.0  | Bad       | Heavily degraded, difficult to understand |

The default threshold of **3.4** ("fair-to-good") balances data retention against quality. Recommended thresholds by use case:

| Use Case              | Suggested Threshold | Rationale                                     |
|-----------------------|---------------------|-----------------------------------------------|
| TTS training data     | ≥ 4.0               | TTS requires high-fidelity reference audio    |
| ASR fine-tuning       | ≥ 3.4               | ASR benefits from diverse, intelligible audio |
| General data curation | ≥ 3.4 (default)     | Good quality/quantity trade-off               |
| Maximum data retention| ≥ 2.5               | Keeps most data, filters only worst clips     |

### SIGMOS Thresholds

[SIGMOS](https://github.com/microsoft/SIG-Challenge/tree/main/ICASSP2024/sigmos) is a neural MOS estimator based on the [ITU-T P.804](https://arxiv.org/pdf/2309.07385.pdf) standard. It predicts **7 quality dimensions** (all on a 0–5 scale, higher = better):

| Dimension   | Output Field     | Description                                              | Use Case                                                        |
|-------------|------------------|----------------------------------------------------------|-----------------------------------------------------------------|
| **NOISE**   | `sigmos_noise`   | Background noise level and residual noise artifacts      | Filter noisy recordings; critical for TTS/ASR training data     |
| **OVRL**    | `sigmos_ovrl`    | Overall perceptual speech quality                        | General-purpose quality gate; single best dimension for broad filtering |
| **SIG**     | `sigmos_sig`     | Speech signal quality and distortion (P.835)             | Detect clipping, codec artifacts, or speech degradation         |
| **COL**     | `sigmos_col`     | Spectral coloration — unnatural tonal shifts             | Catch bandwidth mismatch, over-processed audio, or codec coloring |
| **DISC**    | `sigmos_disc`    | Discontinuity — gaps, clicks, or dropouts in speech      | Filter recordings with packet loss, glitches, or editing artifacts |
| **LOUD**    | `sigmos_loud`    | Loudness appropriateness of the signal                   | Detect clipped/over-compressed or too-quiet recordings          |
| **REVERB**  | `sigmos_reverb`  | Reverberation — room echo and reflections                | Filter reverberant recordings; important for close-talk TTS data |

The pipeline currently exposes two CLI thresholds:

- **`--sigmos-noise-threshold`** (default: 4.0): Minimum noise score. 4.0 passes only clips with very low background noise.
- **`--sigmos-ovrl-threshold`** (default: 3.5): Minimum overall quality. 3.5 passes clips with good overall quality.

All 7 scores are written to the output manifest regardless of which thresholds are active, so you can post-filter on any dimension. The `SIGMOSFilterStage` also supports thresholds for all 7 dimensions programmatically (`sig_threshold`, `col_threshold`, `disc_threshold`, `loud_threshold`, `reverb_threshold`).

### VAD Parameters

| Parameter                | Default | Effect of Raising                           | Effect of Lowering                        |
|--------------------------|---------|---------------------------------------------|-------------------------------------------|
| `--vad-threshold`        | 0.5     | Stricter: fewer, higher-confidence segments | More aggressive: captures quieter speech  |
| `--vad-min-duration`     | 2.0s    | Drops short utterances                      | Keeps very short speech fragments         |
| `--vad-max-duration`     | 60.0s   | No effect (already generous)                | Forces splits on long segments            |
| `--vad-min-interval-ms`  | 500ms   | Merges segments across longer pauses        | Splits on shorter silences                |
| `--vad-speech-pad-ms`    | 300ms   | Wider padding around speech                 | Tighter cuts (risk clipping speech)       |

### Band Filter (`--band-value`)

The [NeMo Curator Speech Bandwidth Filter](https://huggingface.co/nvidia/nemocurator-speech-bandwidth-filter) is a scikit-learn Random Forest Classifier that classifies audio as `full_band` (high fidelity) or `narrow_band` (low fidelity) based on spectral characteristics. It runs entirely on CPU — no GPU required.

- **Input**: PCM F32 audio at 16 kHz or 48 kHz
- **Output**: Integer label (1 = full\_band, 0 = narrow\_band)
- **Default**: `full_band` — keeps only high-fidelity wideband recordings
- **Use `narrow_band`** for telephony-style or low-bandwidth data curation

## Output Format

Results saved to `{output_dir}/*.jsonl`. The output schema depends on the topology:

### Core fields (always present)

| Field | Description |
|-------|-------------|
| `original_file` | Path to the source audio file |
| `original_start_ms` | Start position in original file (ms) |
| `original_end_ms` | End position in original file (ms) |
| `duration_ms` | Duration in milliseconds |
| `duration` | Duration in seconds |

### Combo 3 additional fields (speaker-only)

| Field | Description |
|-------|-------------|
| `diar_segments` | List of `[start_sec, end_sec]` pairs for when the speaker talks |
| `speaking_duration` | Total speaking time in seconds (sum of diar_segments) |

### Passthrough fields (controlled by `passthrough_keys`)

These fields are copied from the pipeline stages to the output.
By default, all built-in filter scores are included:

| Field | Source | Default |
|-------|--------|---------|
| `speaker_id` | SpeakerSeparation | included |
| `num_speakers` | SpeakerSeparation | included |
| `sample_rate` | MonoConversion | included |
| `utmos_mos` | UTMOSFilter | included |
| `sigmos_noise`, `sigmos_ovrl`, ... | SIGMOSFilter | included |
| `band_prediction` | BandFilter | included |

To customize which fields appear in output, set `passthrough_keys` in the config:

```python
AudioDataFilterStage(config={
    "timestamp_mapper": {
        "passthrough_keys": ["utmos_mos", "sigmos_ovrl"],  # only these
    },
})
```

**Safety**: Non-serializable fields (`waveform`, `audio`, `segments`, etc.)
are always blocked, even if added to `passthrough_keys`.
A warning is logged if blocked keys are detected in the configuration.

**Speaker separation note**: When speaker separation is enabled, the parent
task's `duration` and `num_samples` fields are dropped before building
per-speaker child tasks, since each speaker segment has its own duration
computed from the diarization result. Only `audio`/`waveform` (non-serializable)
and `duration`/`num_samples` (parent-specific) are dropped; all other fields
are inherited by child tasks.

### Example outputs

**Combo 1** (no VAD, no speaker):
```json
{"original_file": "/path/to/file.wav", "original_start_ms": 0, "original_end_ms": 10500, "duration_ms": 10500, "duration": 10.5, "utmos_mos": 3.9, "sigmos_ovrl": 3.5}
```

**Combo 2** (VAD only):
```json
{"original_file": "/path/to/file.wav", "original_start_ms": 5200, "original_end_ms": 13200, "duration_ms": 8000, "duration": 8.0, "utmos_mos": 4.1, "sigmos_ovrl": 3.7}
```

**Combo 3** (speaker only):
```json
{"original_file": "/path/to/file.wav", "original_start_ms": 5200, "original_end_ms": 120500, "duration_ms": 115300, "duration": 115.3, "speaking_duration": 43.4, "diar_segments": [[5.2, 15.4], [30.1, 42.8], [100.0, 120.5]], "speaker_id": "speaker_0", "num_speakers": 3}
```

**Combo 4** (VAD + speaker):
```json
{"original_file": "/path/to/file.wav", "original_start_ms": 7200, "original_end_ms": 11200, "duration_ms": 4000, "duration": 4.0, "speaker_id": "speaker_0", "num_speakers": 3, "utmos_mos": 4.2}
```

## Extracting Audio Segments

After the pipeline produces a `manifest.jsonl`, use `extract_segments.py` to extract the actual audio segments from the original files. The script auto-detects the pipeline topology from the manifest schema.

### Basic Usage

```bash
# Extract from a single manifest file
python extract_segments.py -m ./dns_data/result/manifest.jsonl -o ./extracted/

# Extract from a directory of jsonl files (auto-combines them)
python extract_segments.py -m ./dns_data/result/ -o ./extracted/

# Output as FLAC
python extract_segments.py -m ./dns_data/result/ -o ./extracted/ -f flac
```

### Extraction per topology

| Topology | What it extracts | File naming |
|----------|-----------------|-------------|
| Combo 1 | Full file (single segment) | `{name}_segment_000.wav` |
| Combo 2 | Each VAD segment | `{name}_segment_000.wav` |
| Combo 3 | Each speaking interval per speaker | `{name}_speaker_0_segment_000.wav` |
| Combo 4 | Each speaker-segment | `{name}_speaker_0_segment_000.wav` |

### Output files

```
extracted/
├── {name}_speaker_0_segment_000.wav  # Audio segments
├── {name}_speaker_0_segment_001.wav
├── metadata.csv                      # Per-segment metadata with quality scores
├── manifest.jsonl                    # Combined manifest (when input is a directory)
└── extraction_summary.json           # Statistics summary
```

The `metadata.csv` contains one row per extracted segment with columns:
`filename`, `original_file`, `start_sec`, `end_sec`, `duration`, and all quality scores from the manifest.

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--manifest, -m` | required | Path to manifest.jsonl or directory of .jsonl files |
| `--output-dir, -o` | required | Directory for extracted audio segments |
| `--output-format, -f` | `wav` | Output format: `wav`, `flac`, or `ogg` |
| `--verbose, -v` | `false` | Enable verbose (DEBUG) logging |

> **Note**: Supported output formats are `wav`, `flac`, and `ogg` via `soundfile`.

**Storage**: ~11 GB (4.88 GB download + 6.3 GB extracted WAV files; archive is deleted after extraction).

## GPU Memory Requirements

| Stage              | GPU Required? | Default Resources    | Peak VRAM (measured) | Notes                                                         |
|--------------------|---------------|----------------------|----------------------|---------------------------------------------------------------|
| Mono Conversion    | No            | CPU-only             | —                    | scipy/soundfile                                               |
| VAD (Silero)       | No            | `cpus=1.0, gpus=0.1` | negligible          | Silero VAD is lightweight; runs well on CPU                   |
| Band Filter        | No            | `cpus=1.0, gpus=0.0` | —                   | scikit-learn Random Forest, always CPU                        |
| UTMOS              | No            | `cpus=1.0, gpus=0.1` | ~1.4 GB             | PyTorch; uses GPU if available, falls back to CPU             |
| SIGMOS             | No            | `cpus=1.0, gpus=0.1` | ~1.4 GB             | ONNX model (P.804); uses GPU if available, falls back to CPU |
| Speaker Separation | No            | `cpus=1.0, gpus=0.3` | ~1.4 GB             | Sortformer NeMo model; CPU supported but significantly slower |

> **Measured on NVIDIA A100-SXM4-80GB** with 5,000 samples, all filters enabled. Sustained VRAM: **~6.3 GB** (all models loaded concurrently in streaming mode). GPU utilization: **23–92%** sustained during processing. In batch/sequential mode, VRAM is not cumulative — each stage loads its model, processes, then releases GPU memory.
>
> Default GPU resource allocations total **0.9 GPU** for the main branch (VAD `0.1` + UTMOS `0.1` + SIGMOS `0.1` + Speaker Separation `0.3`). When `--enable-speaker-separation` is set, `AudioDataFilterStage.decompose()` instantiates a **duplicated post-speaker filter branch** (VAD `0.1` + Band `0.0` + UTMOS `0.1` + SIGMOS `0.1` = **+0.3 GPU**), so the peak fractional allocation across all stages is **0.9 GPU** — safely within a single GPU. All fractions can be raised in [`default_config.yaml`](https://github.com/NVIDIA-NeMo/Curator/blob/main/nemo_curator/stages/audio/advanced_pipelines/audio_data_filter/default_config.yaml) for faster scheduling on multi-GPU nodes.

**No stage strictly requires a GPU** — all stages fall back to CPU if CUDA is unavailable. However, GPU is strongly recommended for UTMOS, SIGMOS, and Speaker Separation as CPU inference is significantly slower. Band Filter and VAD run efficiently on CPU.

## Performance

### Timing Estimates

Measured on a single **NVIDIA A100-SXM4-80GB** (1 GPU, 20 CPUs), DNS Challenge Read Speech dataset, all filters enabled (UTMOS + VAD + SIGMOS + Band + Speaker Separation), `--backend ray_data`:

| Configuration                                        | Samples | Wall-Clock Time      | Ray Execution Time |
|------------------------------------------------------|---------|----------------------|--------------------|
| Validate setup (`--max-samples 10`, all filters)     | 10      | **~1 min 42 sec**    | ~42 sec            |
| Default (`--max-samples 5000`)                       | 5,000   | **~14 min 34 sec**   | ~13 min 24 sec     |
| Full dataset (`--max-samples -1`)                    | 14,279  | **~40 min 5 sec**    | ~36 min 57 sec     |

> **Hardware setup**: 1× A100-SXM4-80GB GPU, 20 CPUs. Wall-clock time includes Ray startup, model loading, and worker initialization. Ray Execution Time is the actual data processing time reported by `streaming_executor`.

Model download overhead is negligible. Runtime is dominated by model loading/initialization (Sortformer NeMo model, SIGMOS ONNX session, UTMOS PyTorch model). Band Filter is a lightweight scikit-learn model. Silero VAD is bundled with the `silero-vad` pip package and requires no runtime download.

Models downloaded on first run and cached for subsequent runs:

| Model                  | Cache Location                                                                   | Size    |
|------------------------|----------------------------------------------------------------------------------|---------|
| UTMOS (SpeechMOS)      | `~/.cache/torch/hub/`                                                            | ~393 MB |
| SIGMOS (ONNX)          | `~/.cache/nemo_curator/sigmos_model/`                                            | ~26 MB  |
| Band filter (sklearn)  | `~/.cache/huggingface/hub/models--nvidia--nemocurator-speech-bandwidth-filter/`   | ~7 MB   |
| Sortformer (Speaker Sep) | `~/.cache/huggingface/hub/models--nvidia--diar_sortformer_4spk-v1/`            | ~471 MB |
| Silero VAD             | Bundled with `silero-vad` pip package (no download)                              | —       |

### Expected Filtering Ratios

With default thresholds on the DNS Challenge Read Speech dataset:

| Filter Combination                                          | Approx. Pass Rate | Notes                                        |
|-------------------------------------------------------------|--------------------|----------------------------------------------|
| UTMOS ≥ 3.4 only                                           | ~75–85%            | Most DNS clips are clean read speech         |
| VAD defaults only                                          | ~90–95%            | Few clips are pure silence                   |
| Band Filter (`full_band`) only                             | ~90%               | Most DNS clips are full-band                 |
| UTMOS ≥ 3.4 + VAD + Band Filter (`full_band`)              | ~65–75%            | Combined filtering                           |
| All filters (UTMOS + SIGMOS + VAD + Band + Speaker Sep)    | **~23%**           | Measured: 3,306 / 14,279 passed (full dataset) |

> **Measured on A100-SXM4-80GB** with the full 14,279-sample dataset, all filters at default thresholds. SIGMOS (OVRL ≥ 3.5, NOISE ≥ 4.0) is the most aggressive filter — it drops the majority of samples. To retain more data, consider lowering `--sigmos-ovrl-threshold`.

These ratios depend on the dataset. Noisier or lower-quality audio will see higher drop rates. If your pass rate deviates significantly from these ranges, verify your threshold settings.

## Troubleshooting

| Issue                                    | Solution                                                                     |
|------------------------------------------|------------------------------------------------------------------------------|
| No audio files found                     | Check `--auto-download` is enabled or verify path to existing data           |
| `AF_UNIX path length` error             | `export RAY_TMPDIR=/tmp`                                                     |
| CUDA out of memory                       | Disable some filters, use `--max-samples`, or reduce `--batch_size`          |
| Download interrupted                     | Re-run pipeline; it skips already-downloaded files                           |
| Pipeline appears hung                    | See "Is my pipeline hung?" below                                             |
| UTMOS/SIGMOS model download fails        | Check internet connectivity; models are fetched from HuggingFace/ONNX Hub   |
| SIGSEGV / actor crash during model load  | See [Known Issues](../README.md#known-issues) — set `OTEL_SDK_DISABLED=true` |

### Is My Pipeline Hung?

If the pipeline seems stuck:

1. **Check logs**: Run with `--verbose` for DEBUG-level output showing per-file progress.
2. **Check GPU utilization**: Run `watch -n1 nvidia-smi` — active inference shows >0% GPU utilization.
3. **Common causes of apparent hangs**:
   - **First-run model download**: UTMOS, SIGMOS, Band filter, and Sortformer models are downloaded on first execution. This can take a few minutes on slow connections.
   - **Large batch startup**: The executor may take time to initialize workers, especially with Ray.
   - **OOM recovery**: If a stage hits GPU memory limits, it may retry or fall back to CPU silently.
4. **Expected stage durations** (5000 samples, A100-SXM4-80GB):
   - Manifest creation: ~10–30 seconds
   - Mono conversion: ~1–2 minutes
   - VAD segmentation: ~3–5 minutes
   - UTMOS scoring: ~5–10 minutes
   - SIGMOS scoring: ~3–5 minutes
   - Speaker separation: ~5–15 minutes

## Composability

The readspeech pipeline output (JSONL manifest with quality scores) can be chained with other NeMo Curator audio tutorials:

- **`single_speaker_filter/`** — Further filter the manifest to retain only single-speaker segments.
- **`alm/`** — Use the filtered manifest as input for Audio Language Model training window construction.
- **`fleurs/`** — Compare quality-filtered DNS data against FLEURS ASR evaluation data.

The output manifest follows the standard NeMo audio manifest format (`audio_filepath`, `duration`, metadata fields), making it directly usable as training data for NeMo ASR or TTS models.

## Citation

```bibtex
@inproceedings{dubey2023icassp,
  title={ICASSP 2023 Deep Noise Suppression Challenge},
  author={Dubey, Harishchandra and Aazami, Ashkan and Gopal, Vishak and
          Naderi, Babak and Braun, Sebastian and Cutler, Ross and
          Gamper, Hannes and Golestaneh, Mehrsa and Aichner, Robert},
  booktitle={ICASSP},
  year={2023}
}
```

## License

- **DNS Challenge Dataset**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **NeMo Curator**: Apache License 2.0
