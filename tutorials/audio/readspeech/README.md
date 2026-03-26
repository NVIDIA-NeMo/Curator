# DNS Challenge Read Speech Pipeline

Process the DNS Challenge Read Speech dataset using NeMo Curator's audio pipeline with **automatic download support**.

The pipeline downloads the dataset (4.88 GB compressed, 14,279 WAV files at 48kHz, 19.3 hours total audio) and applies quality filtering.

## Quick Start

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

```
CreateInitialManifestReadSpeechStage
  Downloads and scans read_speech directory, parses filenames, creates AudioTask
      |
      v
AudioDataFilterStage
  Mono conversion -> VAD -> Band Filter -> UTMOS -> SIGMOS
  -> Speaker Separation -> Timestamp Tracking
      |
      v
AudioToDocumentStage -> JsonlWriter
  Output: manifest.jsonl
```

## Running the Pipeline

### Option 1: Python Script (pipeline.py)

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

### Option 2: YAML Config (run.py)

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

| Option | Description |
|--------|-------------|
| `--raw_data_dir` | Directory for data download or path to existing data |

### Download Settings

| Option | Default | Description |
|--------|---------|-------------|
| `--auto-download` | `true` | Auto-download dataset (~4.88 GB) |
| `--no-auto-download` | | Disable auto-download |

### Processing

| Option | Default | Description |
|--------|---------|-------------|
| `--output_dir` | `{raw_data_dir}/result` | Output directory |
| `--max-samples` | `5000` | Max samples (-1 for all 14,279 files) |
| `--batch_size` | `1` | Batch size |
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
| `--vad-min-interval-ms` | `500` | Min silence to split segments (ms) |
| `--vad-speech-pad-ms` | `300` | Padding before/after speech (ms) |
| `--enable-utmos` | `false` | Enable UTMOS filter |
| `--utmos-mos-threshold` | `3.4` | Min UTMOS MOS (1-5) |
| `--enable-sigmos` | `false` | Enable SIGMOS filter |
| `--sigmos-noise-threshold` | `4.0` | Min SIGMOS noise (1-5) |
| `--sigmos-ovrl-threshold` | `3.5` | Min SIGMOS overall (1-5) |
| `--enable-band-filter` | `false` | Enable band filter |
| `--band-value` | `full_band` | Band type to pass |
| `--enable-speaker-separation` | `false` | Enable speaker diarization |
| `--speaker-exclude-overlaps` | `true` | Exclude overlapping speech |
| `--no-speaker-exclude-overlaps` | | Allow overlapping speaker segments |
| `--speaker-min-duration` | `0.8` | Min speaker segment (sec) |

## Output Format

Results saved to `{output_dir}/*.jsonl`:

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
  "utmos_mos": 3.9,
  "sigmos_noise": 4.2,
  "band_prediction": "full_band"
}
```

## Extracting Audio Segments

After the pipeline produces a `manifest.jsonl`, use `extract_segments.py` to extract the actual audio segments from the original files based on the timestamps in the manifest.

### Basic Usage

```bash
# Extract segments from a single manifest file
python extract_segments.py \
    --manifest ./dns_data/result/manifest.jsonl \
    --output-dir ./extracted_segments

# Extract from a directory of jsonl files (auto-combines them)
python extract_segments.py \
    --manifest ./dns_data/result/ \
    --output-dir ./extracted_segments

# Output as FLAC instead of WAV
python extract_segments.py \
    --manifest ./dns_data/result/manifest.jsonl \
    --output-dir ./extracted_segments \
    --output-format flac
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--manifest, -m` | required | Path to manifest.jsonl or directory of .jsonl files |
| `--output-dir, -o` | required | Directory for extracted audio segments |
| `--output-format, -f` | `wav` | Output format: wav, flac, ogg (no extra deps), mp3, m4a (need pydub + ffmpeg) |
| `--verbose, -v` | `false` | Enable verbose (DEBUG) logging |

### Output

Extracted files are named based on the original filename with speaker and segment info:

```
extracted_segments/
├── book_00025_chp_0019_reader_04069_speaker_0_segment_000.wav
├── book_00025_chp_0019_reader_04069_speaker_0_segment_001.wav
├── book_00025_chp_0019_reader_04069_speaker_1_segment_000.wav
├── manifest.jsonl              # Combined manifest (when input is a directory)
└── extraction_summary.json     # Statistics summary
```

Without speaker separation, files are named `{original_name}_segment_{num}.wav`.

The script also generates an `extraction_summary.json` with statistics including total segments extracted, total duration, and per-speaker segment counts.

> **Note**: wav/flac/ogg output works out of the box via `soundfile`. For mp3/m4a output,
> install `pip install pydub` and `sudo apt install ffmpeg` (or `conda install ffmpeg`).


**Storage**: ~11 GB (4.88 GB download + 6.3 GB extracted WAV files; archive is deleted after extraction).

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No audio files found | Check `--auto-download` is enabled or verify path to existing data |
| `AF_UNIX path length` error | `export RAY_TMPDIR=/tmp` |
| CUDA out of memory | Disable some filters or use `--max-samples` |
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
