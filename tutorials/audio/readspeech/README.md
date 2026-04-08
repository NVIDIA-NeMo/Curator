# DNS Challenge Read Speech Pipeline

Process the DNS Challenge Read Speech dataset using NeMo Curator's audio pipeline with **automatic download support**.

The pipeline downloads the dataset (4.88 GB compressed, 14,279 WAV files at 48kHz, 19.3 hours total audio) and applies quality filtering.

## Prerequisites

Install NeMo Curator with audio dependencies using [uv](https://docs.astral.sh/uv/):

```bash
# GPU (recommended)
uv sync --extra audio_cuda12

# CPU only
uv sync --extra audio_cpu
```

The full pipeline requires: `soundfile`, `torchaudio`, `librosa`, `scipy`, `pydub`, `onnxruntime`/`onnxruntime-gpu`, `silero-vad`, and `nemo_toolkit[asr]`. These are all included in the `audio_cuda12` / `audio_cpu` extras.

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
| `--backend` | `xenna` | Execution backend: `xenna` or `ray_data` |
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
| `--utmos-mos-threshold` | `3.4` | Min UTMOS MOS (0-5) |
| `--enable-sigmos` | `false` | Enable SIGMOS filter |
| `--sigmos-noise-threshold` | `4.0` | Min SIGMOS noise (0-5) |
| `--sigmos-ovrl-threshold` | `3.5` | Min SIGMOS overall (0-5) |
| `--enable-band-filter` | `false` | Enable band filter |
| `--band-value` | `full_band` | Band type to pass |
| `--enable-speaker-separation` | `false` | Enable speaker diarization |
| `--speaker-exclude-overlaps` | `true` | Exclude overlapping speech |
| `--no-speaker-exclude-overlaps` | | Allow overlapping speaker segments |
| `--speaker-min-duration` | `0.8` | Min speaker segment (sec) |

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
| Combo 1 | Full file copy | `{name}.wav` |
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
