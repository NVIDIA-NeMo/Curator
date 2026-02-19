# DNS Challenge Read Speech Pipeline

Process the DNS Challenge Read Speech dataset using NeMo Curator's audio pipeline with **automatic download support**.

By default, the pipeline downloads a **partial dataset** (~30GB, ~2,500 samples) which is sufficient for tutorials and testing. Optionally configure it to download the full dataset (~182GB, ~14,000 samples).

## Quick Start

```bash
# Auto-download partial dataset and process
python pipeline.py \
    --raw_data_dir ./dns_data \
    --enable-nisqa \
    --enable-vad

# Use pre-downloaded data
python pipeline.py \
    --raw_data_dir /path/to/existing/read_speech \
    --no-auto-download \
    --enable-nisqa
```

## Dataset Overview

**DNS Challenge 5 - Read Speech (Track 1 Headset)**
- **Source**: [Microsoft DNS Challenge](https://github.com/microsoft/DNS-Challenge)
- **Format**: WAV files (mono or stereo), 48,000 Hz
- **License**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

| Version | Size | Samples | Configuration |
|---------|------|---------|---------------|
| **Partial (Default)** | ~30GB | ~2,500 | `download_parts: 1` |
| **Complete** | ~182GB | ~14,000 | `download_parts: 6` |

### Dataset Structure

```
raw_data_dir/
└── read_speech/
    ├── book_00000_chp_0009_reader_06709_0_seg_1_seg1.wav
    ├── book_00000_chp_0009_reader_06709_0_seg_2_seg1.wav
    └── ... (~2,500 files for partial, ~14,000+ for full)
```

## Pipeline Architecture

```
CreateInitialManifestReadSpeechStage
  Scans read_speech directory, parses filenames, creates AudioBatch
      |
      v
AudioDataFilterStage
  Mono conversion -> VAD -> Band Filter -> NISQA -> SIGMOS
  -> Speaker Separation -> Timestamp Tracking
      |
      v
AudioToDocumentStage -> JsonlWriter
  Output: manifest.jsonl
```

## Running the Pipeline

### Option 1: Python Script (pipeline.py)

```bash
# Partial dataset with all filters
python pipeline.py \
    --raw_data_dir ./dns_data \
    --enable-vad \
    --enable-nisqa \
    --enable-sigmos \
    --enable-band-filter \
    --enable-speaker-separation \
    --gpus 1.0 \
    --cpus 4.0

# Full dataset
python pipeline.py \
    --raw_data_dir ./dns_data \
    --download-parts 6 \
    --max-samples -1 \
    --enable-nisqa \
    --enable-sigmos
```

### Option 2: YAML Config (run.py)

```bash
python run.py \
    --config-path . \
    --config-name pipeline.yaml \
    raw_data_dir=./dns_data

# Override settings
python run.py \
    --config-path . \
    --config-name pipeline.yaml \
    raw_data_dir=./dns_data \
    download_parts=6 \
    max_samples=-1
```

## Command Line Options

### Required

| Option | Description |
|--------|-------------|
| `--raw_data_dir` | Directory for data download or path to existing data |

### Download Settings

| Option | Default | Description |
|--------|---------|-------------|
| `--auto-download` | `true` | Auto-download dataset |
| `--no-auto-download` | | Disable auto-download |
| `--download-parts` | `1` | Parts to download (1-6). 1=~30GB, 6=~182GB |

### Processing

| Option | Default | Description |
|--------|---------|-------------|
| `--output_dir` | `{raw_data_dir}/result` | Output directory |
| `--max-samples` | `5000` | Max samples (-1 for all) |
| `--batch_size` | `1` | Batch size |
| `--gpus` | `1.0` | GPU allocation |
| `--cpus` | `1.0` | CPU cores |
| `--sample_rate` | `48000` | Audio sample rate |
| `--clean` | `false` | Clean output dir |
| `--verbose` | `false` | DEBUG logging |

### Filter Toggles and Thresholds

| Option | Default | Description |
|--------|---------|-------------|
| `--enable-vad` | `false` | Enable VAD segmentation |
| `--vad-min-duration` | `2.0` | Min segment (sec) |
| `--vad-max-duration` | `60.0` | Max segment (sec) |
| `--vad-threshold` | `0.5` | VAD threshold (0-1) |
| `--enable-nisqa` | `false` | Enable NISQA filter |
| `--nisqa-mos-threshold` | `4.5` | Min NISQA MOS (1-5) |
| `--nisqa-noi-threshold` | `4.3` | Min NISQA noisiness (1-5) |
| `--enable-sigmos` | `false` | Enable SIGMOS filter |
| `--sigmos-noise-threshold` | `4.0` | Min SIGMOS noise (1-5) |
| `--sigmos-ovrl-threshold` | `3.5` | Min SIGMOS overall (1-5) |
| `--enable-band-filter` | `false` | Enable band filter |
| `--band-value` | `full_band` | Band type to pass |
| `--enable-speaker-separation` | `false` | Enable speaker diarization |

## Output Format

Results saved to `{output_dir}/manifest.jsonl`:

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
  "nisqa_mos": 4.7,
  "sigmos_noise": 4.2,
  "band_prediction": "full_band"
}
```

## Prerequisites

```bash
pip install nemo-curator[audio_cuda12]

# Or install packages individually
pip install soundfile librosa pydub loguru hydra-core wget
```

**Storage**: ~35GB for partial, ~200GB for full dataset.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No audio files found | Check `--auto-download` is enabled or verify path to existing data |
| `AF_UNIX path length` error | `export RAY_TMPDIR=/tmp` |
| CUDA out of memory | Reduce `--gpus`, disable some filters, or use `--max-samples` |
| Download interrupted | Re-run pipeline; it skips already-downloaded files |

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
