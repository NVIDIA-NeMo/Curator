# DNS Challenge Read Speech Pipeline

This tutorial demonstrates how to process the DNS Challenge Read Speech dataset using NeMo Curator's audio processing pipeline with **automatic download support**.

> **✨ New**: The pipeline now **automatically downloads and extracts** the DNS Challenge Read Speech dataset! Just run the pipeline and it will handle the download for you.

> **📦 Partial Download (Default)**: By default, the pipeline downloads a **partial dataset** (~30GB, ~2,500 samples) which is perfect for learning and testing. This is **sufficient for most tutorial purposes** and much faster to download. You can optionally configure it to download the full dataset (~182GB, ~14,000 samples).

**Comparison with Other Datasets**:

| Dataset | Download Behavior |
|---------|-------------------|
| **Read Speech** | ✅ **Automatic download by pipeline stage** (configurable) |
| **FLEURS** | ✅ Automatic download by pipeline stage |

## Workflow Overview

The pipeline automatically handles everything:

```
┌─────────────────────────────────────────────────────┐
│  NeMo Curator Pipeline (Single Command)            │
│                                                     │
│  1. Auto-Download Dataset (~30GB partial)          │
│     • Downloads from Azure Blob Storage            │
│     • Extracts archives                            │
│     • Skips if already downloaded                  │
│                                                     │
│  2. CreateInitialManifestReadSpeechStage           │
│     • Scans audio files                            │
│     • Extracts metadata                            │
│                                                     │
│  3. AudioDataFilterStage (Quality Filters)         │
│     • VAD segmentation                             │
│     • Quality assessment (NISQA, SIGMOS)           │
│     • Speaker separation                           │
│                                                     │
│  4. Output manifest.jsonl                          │
└─────────────────────────────────────────────────────┘
```

> **Key Point**: The pipeline **automatically downloads and processes** data in one go. You can disable auto-download if you've already downloaded the dataset manually.

---

## Dataset Overview

**DNS Challenge 5 - Read Speech (Track 1 Headset)**
- **Source**: [Microsoft DNS Challenge](https://github.com/microsoft/DNS-Challenge)
- **Format**: WAV files (mono or stereo)
- **Sample Rate**: 48,000 Hz
- **License**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

**Dataset Options**:

| Version | Download Size | Samples | Configuration | Use Case |
|---------|--------------|---------|---------------|----------|
| **Partial (Default)** | ~30GB | ~2,500 | `download_parts: 1` | Tutorials, testing, learning the pipeline |
| **Complete** | ~182GB | ~14,000 | `download_parts: 6` | Production, research, complete dataset |

**File Naming Convention**:
```
book_XXXXX_chp_XXXX_reader_XXXXX_X_seg_X_segX.wav
```

Example: `book_00000_chp_0009_reader_06709_0_seg_1_seg1.wav`

The filenames encode:
- **book_XXXXX**: Book ID
- **chp_XXXX**: Chapter number
- **reader_XXXXX**: Reader ID
- **seg_X**: Segment information

---

## Quick Start (Automatic Download)

The simplest way to get started is to let the pipeline automatically download the dataset:

```bash
# Clone DNS-Challenge repo (if needed for download scripts)
cd /path/to/workspace

# Run pipeline - it will automatically download and process data!
python pipeline.py \
    --raw_data_dir ./dns_data \
    --enable-nisqa \
    --enable-vad

# That's it! The pipeline will:
# 1. Download ~30GB partial dataset (first run only)
# 2. Extract audio files
# 3. Process with quality filters
# 4. Generate manifest.jsonl
```

**Download Configuration**:
- `--auto-download` (default: `true`) - Automatically download dataset
- `--download-parts 1` (default) - Download partial dataset (~30GB, ~2,500 samples)
- `--download-parts 6` - Download full dataset (~182GB, ~14,000 samples)

**Disable Auto-Download**:
If you've already downloaded the dataset manually, you can disable auto-download:

```bash
python pipeline.py \
    --raw_data_dir /path/to/existing/read_speech \
    --no-auto-download \
    --enable-nisqa
```

---

## Manual Download (Optional - Advanced)

> **Note**: Manual download is **optional**. The pipeline automatically downloads the dataset by default. Use manual download only if you need fine-grained control or want to pre-download the dataset.

If you've manually downloaded the dataset, you can disable auto-download:

```bash
python pipeline.py \
    --raw_data_dir /path/to/your/read_speech \
    --no-auto-download \
    --enable-nisqa
```

For manual download instructions, see the [DNS Challenge repository](https://github.com/microsoft/DNS-Challenge).

---

## Dataset Structure

After automatic download (or manual extraction), the dataset structure will be:

```
raw_data_dir/
└── read_speech/                        # Automatically created
    ├── book_00000_chp_0009_reader_06709_0_seg_1_seg1.wav
    ├── book_00000_chp_0009_reader_06709_0_seg_2_seg1.wav
    ├── book_00000_chp_0009_reader_06709_0_seg_3_seg1.wav
    └── ... (~2,500 files for partial, ~14,000+ for full)
```

---

## Prerequisites

### System Requirements

**For Partial Download (Default)**:
- **Storage**: ~35GB free space (30GB download + extraction)
- **RAM**: 16GB+ recommended
- **GPU**: Optional but recommended for quality filters (NISQA, SIGMOS)

**For Full Download (Optional)**:
- **Storage**: ~200GB free space (182GB for data + extraction space)
- **RAM**: 32GB+ recommended
- **GPU**: Recommended for faster processing

### Software Requirements

```bash
# Install NeMo Curator with audio dependencies
pip install nemo-curator[audio_cuda12]

# Or install required packages
pip install soundfile librosa pydub loguru hydra-core wget
```

---

## Running the Pipeline

The pipeline automatically handles download, extraction, and processing:

### Option 1: Python Script (pipeline.py)

**Basic usage with auto-download** (partial dataset, ~30GB):
```bash
python pipeline.py \
    --raw_data_dir ./dns_data \
    --enable-nisqa

# The pipeline will:
# 1. Automatically download ~30GB partial dataset (first run only)
# 2. Extract to ./dns_data/read_speech/
# 3. Process ~2,500 samples with NISQA filter
```

**Download full dataset** (~182GB, ~14,000 samples):
```bash
python pipeline.py \
    --raw_data_dir ./dns_data \
    --download-parts 6 \
    --max-samples -1 \
    --enable-nisqa \
    --enable-sigmos
```

**Full pipeline with all filters**:
```bash
python pipeline.py \
    --raw_data_dir ./dns_data \
    --enable-vad \
    --enable-nisqa \
    --enable-sigmos \
    --enable-band-filter \
    --enable-speaker-separation \
    --gpus 1.0 \
    --cpus 4.0 \
    --verbose
```

**Disable auto-download** (use pre-downloaded data):
```bash
python pipeline.py \
    --raw_data_dir /path/to/existing/read_speech \
    --no-auto-download \
    --enable-nisqa
```

**Custom output directory**:
```bash
python pipeline.py \
    --raw_data_dir ./dns_data \
    --output_dir ./custom_output \
    --enable-nisqa \
    --max-samples 1000
```

### Option 2: YAML Config (run.py - Recommended)

Using Hydra configuration for reproducibility:

```bash
# Basic usage with auto-download (default)
python run.py \
    --config-path . \
    --config-name pipeline.yaml \
    raw_data_dir=./dns_data

# Download full dataset
python run.py \
    --config-path . \
    --config-name pipeline.yaml \
    raw_data_dir=./dns_data \
    download_parts=6 \
    max_samples=-1

# Override multiple settings
python run.py \
    --config-path . \
    --config-name pipeline.yaml \
    raw_data_dir=./dns_data \
    max_samples=10000 \
    gpus=2.0 \
    processors.1.config.enable_vad=true \
    processors.1.config.enable_speaker_separation=true

# Disable auto-download (use existing data)
python run.py \
    --config-path . \
    --config-name pipeline.yaml \
    raw_data_dir=/path/to/existing/read_speech \
    auto_download=false
```

**Edit pipeline.yaml** for persistent configuration:
```yaml
# pipeline.yaml
raw_data_dir: ./dns_data
auto_download: true        # Enable/disable auto-download
download_parts: 1         # 1 = ~30GB partial, 6 = ~182GB full
max_samples: 10000  # -1 for all
gpus: 1.0
cpus: 4.0

processors:
  # ... stage configurations ...
```

---

## Pipeline Architecture

The pipeline processes the DNS Challenge Read Speech dataset through four main stages:

```
┌─────────────────────────────────────────────────────────┐
│ Stage 1: CreateInitialManifestReadSpeechStage          │
│ • Scans EXISTING read_speech directory for WAV files    │
│ • Parses filenames to extract metadata                  │
│ • Creates AudioBatch with file paths and metadata       │
│ • NOTE: Does NOT download data (expects pre-downloaded) │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 2: AudioDataFilterStage (Composite)               │
│ • Mono conversion (if needed)                           │
│ • VAD segmentation (optional)                           │
│ • Band filter: full_band/narrow_band (optional)         │
│ • NISQA: speech quality assessment (optional)           │
│ • SIGMOS: signal quality assessment (optional)          │
│ • Speaker separation (optional)                         │
│ • Timestamp tracking throughout                         │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 3: AudioToDocumentStage                           │
│ • Converts AudioBatch → DocumentBatch                   │
│ • Preserves all metadata and quality scores             │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 4: JsonlWriter                                    │
│ • Writes manifest.jsonl to output directory             │
│ • One line per segment (JSONL format)                   │
└─────────────────────────────────────────────────────────┘
```

### Stage Details

#### 1. CreateInitialManifestReadSpeechStage

- **Purpose**: Scans and creates manifest from **already-downloaded** WAV files
- **Input**: Directory path to existing audio files (via `raw_data_dir`)
- **Output**: AudioBatch with file metadata
- **Important**: This stage **does NOT download** data. It expects audio files to already exist in `raw_data_dir`
- **Metadata Extracted**:
  - `book_id`: Extracted from filename
  - `reader_id`: Extracted from filename
  - `chapter`: Chapter number
- **Sample Selection**: 
  - Configurable via `max_samples` (default: 5000)
  - Partial download has ~2,500 samples total
  - Full download has ~14,000 samples total

#### 2. AudioDataFilterStage

A composite stage that combines multiple audio processing sub-stages:

- **MonoConversion**: Converts stereo to mono (48kHz)
- **VADSegmentation**: Extracts speech segments (optional)
- **BandFilter**: Classifies full_band vs narrow_band (optional)
- **NISQAFilter**: Measures speech quality (MOS, noisiness, etc.) (optional)
- **SIGMOSFilter**: Measures signal quality (noise, overall, etc.) (optional)
- **SpeakerSeparation**: Separates by speaker (optional)
- **TimestampTracker**: Maintains accurate timestamps throughout

#### 3. AudioToDocumentStage

- Converts AudioBatch format to DocumentBatch for compatibility with text pipeline writers
- Preserves all metadata and quality scores

#### 4. JsonlWriter

- Writes final manifest to JSONL format
- One JSON object per line (newline-delimited JSON)

---

## Output Format

Results are saved to `${output_dir}/manifest.jsonl` (default: `${raw_data_dir}/result/manifest.jsonl`)

### Output Fields

Each line in the manifest contains:

#### Core Fields (Always Present)

| Field | Type | Description |
|-------|------|-------------|
| `audio_filepath` | `str` | Absolute path to the original WAV file |
| `sample_rate` | `int` | Audio sample rate (48000 Hz) |
| `book_id` | `str` | Book identifier extracted from filename |
| `reader_id` | `str` | Reader identifier extracted from filename |

#### Timestamp Fields (When VAD or Speaker Separation Enabled)

| Field | Type | Description |
|-------|------|-------------|
| `original_start_ms` | `int` | Start position in original file (milliseconds) |
| `original_end_ms` | `int` | End position in original file (milliseconds) |
| `duration_ms` | `int` | Segment duration (milliseconds) |
| `duration_sec` | `float` | Segment duration (seconds) |

#### Speaker Fields (When Speaker Separation Enabled)

| Field | Type | Description |
|-------|------|-------------|
| `speaker_id` | `str` | Speaker identifier (e.g., "speaker_0", "speaker_1") |
| `num_speakers` | `int` | Total number of speakers detected in the audio |

#### Quality Scores (When Filters Enabled)

**NISQA Scores** (when `--enable-nisqa`):
| Field | Range | Description |
|-------|-------|-------------|
| `nisqa_mos` | 1–5 | Mean Opinion Score (overall quality) |
| `nisqa_noi` | 1–5 | Noisiness (higher = less noisy) |
| `nisqa_col` | 1–5 | Coloration/distortion |
| `nisqa_dis` | 1–5 | Discontinuity |
| `nisqa_loud` | 1–5 | Loudness appropriateness |

**SIGMOS Scores** (when `--enable-sigmos`):
| Field | Range | Description |
|-------|-------|-------------|
| `sigmos_noise` | 1–5 | Background noise quality |
| `sigmos_ovrl` | 1–5 | Overall signal quality |
| `sigmos_sig` | 1–5 | Signal quality |
| `sigmos_col` | 1–5 | Coloration |
| `sigmos_disc` | 1–5 | Discontinuity |
| `sigmos_loud` | 1–5 | Loudness |
| `sigmos_reverb` | 1–5 | Reverberation (higher = less reverb) |

**Band Classification** (when `--enable-band-filter`):
| Field | Values | Description |
|-------|--------|-------------|
| `band_prediction` | `"full_band"` or `"narrow_band"` | Bandwidth classification |

### Sample Output

**Basic output** (no filters):
```json
{
  "audio_filepath": "/path/to/read_speech/book_00000_chp_0009_reader_06709_0_seg_1_seg1.wav",
  "sample_rate": 48000,
  "book_id": "00000",
  "reader_id": "06709"
}
```

**With all filters enabled**:
```json
{
  "audio_filepath": "/path/to/read_speech/book_00000_chp_0009_reader_06709_0_seg_1_seg1.wav",
  "sample_rate": 48000,
  "book_id": "00000",
  "reader_id": "06709",
  "original_start_ms": 1500,
  "original_end_ms": 5200,
  "duration_ms": 3700,
  "duration_sec": 3.7,
  "speaker_id": "speaker_0",
  "num_speakers": 1,
  "nisqa_mos": 4.7,
  "nisqa_noi": 4.5,
  "sigmos_noise": 4.2,
  "sigmos_ovrl": 3.9,
  "band_prediction": "full_band"
}
```

---

## Command Line Options

### Required Arguments

| Option | Description |
|--------|-------------|
| `--raw_data_dir` | Directory where data will be downloaded to (if auto-download) or path to existing read_speech directory |

### Download Settings

| Option | Default | Description |
|--------|---------|-------------|
| `--auto-download` | `true` | Automatically download and extract dataset |
| `--no-auto-download` | `false` | Disable auto-download (use existing data) |
| `--download-parts` | `1` | Number of parts to download (1-6). 1 = ~30GB (~2,500 samples), 6 = ~182GB (~14,000 samples) |

### Optional Arguments

| Option | Default | Description |
|--------|---------|-------------|
| `--output_dir` | `{raw_data_dir}/result` | Output directory for manifest.jsonl |
| `--max-samples` | `5000` | Maximum samples to process. Set to `-1` for all |
| `--batch_size` | `1` | Batch size for processing |
| `--clean` | `false` | Clean output directory before processing |
| `--verbose` | `false` | Enable DEBUG level logging |

### Resource Allocation

| Option | Default | Description |
|--------|---------|-------------|
| `--gpus` | `1.0` | GPU allocation (0.0-1.0 for fractional, >1 for multiple) |
| `--cpus` | `1.0` | CPU cores for parallel processing |

### Audio Processing

| Option | Default | Description |
|--------|---------|-------------|
| `--sample_rate` | `48000` | Expected audio sample rate (Hz) |

### Filter Toggles

| Option | Default | Description |
|--------|---------|-------------|
| `--enable-vad` | `false` | Enable VAD segmentation |
| `--enable-nisqa` | `false` | Enable NISQA quality filter |
| `--enable-sigmos` | `false` | Enable SIGMOS quality filter |
| `--enable-band-filter` | `false` | Enable bandwidth classification |
| `--enable-speaker-separation` | `false` | Enable speaker diarization |

### VAD Settings

| Option | Default | Description |
|--------|---------|-------------|
| `--vad-min-duration` | `2.0` | Minimum segment duration (seconds) |
| `--vad-max-duration` | `60.0` | Maximum segment duration (seconds) |
| `--vad-threshold` | `0.5` | VAD detection threshold (0.0-1.0) |

### Quality Thresholds

| Option | Default | Description |
|--------|---------|-------------|
| `--nisqa-mos-threshold` | `4.5` | Minimum NISQA MOS score (1-5) |
| `--nisqa-noi-threshold` | `4.3` | Minimum NISQA noisiness score (1-5) |
| `--sigmos-noise-threshold` | `4.0` | Minimum SIGMOS noise score (1-5) |
| `--sigmos-ovrl-threshold` | `3.5` | Minimum SIGMOS overall score (1-5) |
| `--band-value` | `"full_band"` | Band type to pass (`"full_band"` or `"narrow_band"`) |

---

## Troubleshooting

### Issue: No Audio Files Found

**Error**:
```
No audio files found in dataset
Found 0 WAV files in /path/to/read_speech
```

**Cause**: Download may have failed, or incorrect path

**Solution**:
1. **Check if auto-download is enabled**: Default is `--auto-download` (enabled)
2. **Verify download completed**: Check logs for download progress and extraction
3. **Check the path**: If auto-download is enabled, files will be in `{raw_data_dir}/read_speech/`
4. **Try manual download disable**: If you have existing data, use `--no-auto-download`
5. **Verify files exist**:
   ```bash
   ls {raw_data_dir}/read_speech/*.wav | head
   ```

### Issue: Ray Socket Path Too Long

**Error**:
```
AF_UNIX path length cannot exceed 107 bytes
```

**Solution**:
```bash
export RAY_TMPDIR=/tmp
```

### Issue: Out of Memory (GPU)

**Symptoms**: CUDA out of memory errors

**Solutions**:
1. Reduce GPU allocation: `--gpus 0.5`
2. Disable some filters
3. Process in smaller batches: `--max-samples 1000`

### Issue: Download Interrupted

**Symptoms**: Download stops midway, partial files exist

**Solution**: Re-run the pipeline. The download stage will:
1. Check if files already exist
2. Skip download if complete
3. Resume if files are missing

If issues persist, delete partial files and re-run:
```bash
rm -rf {raw_data_dir}/read_speech
python pipeline.py --raw_data_dir {raw_data_dir} --enable-nisqa
```

### Issue: Path Not Found During Pipeline

**Symptoms**: Pipeline can't find the audio directory

**Check**:
1. **If using auto-download**: Files will be in `{raw_data_dir}/read_speech/`, not just `{raw_data_dir}/`
2. **If using existing data**: Use `--no-auto-download` and point to the directory containing WAV files
3. Verify files exist: `ls -lh {raw_data_dir}/read_speech/*.wav`
4. Check permissions: `ls -ld {raw_data_dir}`
5. Use absolute paths if needed: `--raw_data_dir /full/path`

### Issue: Slow Processing

**Solutions**:
1. Increase GPU allocation: `--gpus 2.0`
2. Increase CPU allocation: `--cpus 8.0`
3. Disable expensive filters (NISQA/SIGMOS)
4. Process subset first: `--max-samples 1000`

---

## Performance Tips

1. **GPU Usage**: Enable GPU for quality filters (NISQA, SIGMOS) significantly speeds up processing
2. **Parallel Processing**: Increase `--cpus` for CPU-bound stages
3. **Batch Processing**: Default batch_size=1 is optimal for this dataset
4. **Filter Selection**: Enable only needed filters - each filter adds processing time
5. **Storage**: Use SSD for faster I/O with large datasets

---

## Example Workflows

### Workflow 1: Quick Test (Auto-Download + 1000 samples)

Perfect for testing the pipeline setup with automatic download:

```bash
# Pipeline automatically downloads ~30GB partial dataset
python pipeline.py \
    --raw_data_dir ./dns_data \
    --max-samples 1000 \
    --enable-nisqa \
    --enable-band-filter \
    --verbose
```

### Workflow 2: Process All Partial Dataset (~2,500 samples)

Process all samples from the partial download:

```bash
python pipeline.py \
    --raw_data_dir ./dns_data \
    --max-samples -1 \
    --enable-vad \
    --enable-nisqa \
    --enable-sigmos \
    --enable-band-filter \
    --gpus 1.0 \
    --cpus 4.0 \
    --verbose
```

Expected time: Download ~1-2 hours + Processing ~30-60 minutes (with GPU)

### Workflow 3: Full Dataset Processing (All Filters)

Automatically download full dataset (~182GB, ~14,000 samples):

```bash
python pipeline.py \
    --raw_data_dir ./dns_data_full \
    --download-parts 6 \
    --max-samples -1 \
    --enable-vad \
    --enable-nisqa \
    --enable-sigmos \
    --enable-band-filter \
    --enable-speaker-separation \
    --gpus 2.0 \
    --cpus 8.0 \
    --verbose
```

Expected time: Download ~4-8 hours + Processing ~3-5 hours (with 2 GPUs)

### Workflow 4: High-Quality Segments Only

Extract only the highest quality segments:

```bash
python pipeline.py \
    --raw_data_dir ./dns_data \
    --max-samples -1 \
    --enable-nisqa \
    --nisqa-mos-threshold 4.8 \
    --nisqa-noi-threshold 4.5 \
    --enable-band-filter \
    --band-value full_band \
    --verbose
```

---

## Summary: Key Points

### Download vs Pipeline Processing

This pipeline **automatically handles everything** in a single command:

1. **Auto-Download & Extraction** (Automatic):
   - Downloads from Azure Blob Storage
   - Extracts archives
   - Organizes files
   - Skips if already downloaded

2. **Pipeline Processing** (NeMo Curator):
   - Scans audio files
   - Extracts metadata from filenames
   - Applies quality filters
   - Generates manifest.jsonl

### Why Auto-Download?

- **One-command simplicity**: No manual download steps required
- **Smart caching**: Automatically skips if data already exists
- **Configurable**: Choose partial (~30GB) or full (~182GB) dataset
- **Resumable**: Can resume if interrupted

### Quick Start

Simply run:
```bash
python pipeline.py --raw_data_dir ./dns_data --enable-nisqa
```

That's it! The pipeline will:
- ✅ Download ~30GB partial dataset (first run only)
- ✅ Extract files automatically
- ✅ Process with quality filters
- ✅ Generate manifest.jsonl

### Disable Auto-Download

If you've pre-downloaded the dataset:
```bash
python pipeline.py --raw_data_dir /path/to/read_speech --no-auto-download --enable-nisqa
```

---

## Citation

If you use the DNS Challenge dataset in your research, please cite:

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

---

## License

- **DNS Challenge Dataset**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **NeMo Curator**: Apache License 2.0

