# ASR Data Processing Pipeline

This tutorial shows how to use NeMo Curator to turn already-downloaded ASR datasets into normalized, split-aware training manifests.

The pipeline is designed for public ASR corpora where the user handles download access and storage, while Curator handles:

- dataset-specific ingestion and split assignment
- audio conversion to WAV, 16 kHz, mono, PCM16
- canonical ASR metadata creation
- language-resource-driven transcript normalization
- transcript quality statistics by language and source
- train/dev/test JSONL manifest writing

## When to Use This Pipeline

Use this pipeline when you already have a dataset on disk and want to convert it into a consistent ASR training format. For example, a downloaded HuggingFace Arrow dataset, a tarred audio corpus with transcripts, or a custom internal dataset can each be wrapped by an ingestion handler that emits the same `AudioTask` shape.

This is different from the audio tagging tutorial. The tagging pipeline starts from raw unlabelled audio and creates labels using diarization and ASR models. This ASR data pipeline starts from labelled ASR data and standardizes it for training.

## Pipeline Overview

```text
Downloaded dataset
      |
      v
Dataset ingestion handler
  - reads the source-specific layout
  - decodes or extracts audio
  - writes WAV/16 kHz/mono/PCM16
  - assigns train/dev/test split_type
  - emits AudioTask with ASRMetadata fields
      |
      v
TranscriptNormalizationStage
  - loads language resources
  - applies pretok replacements
  - removes configured characters
  - records unknown characters
      |
      v
TranscriptStatsStage
  - counts valid/invalid transcripts
  - tracks hours and split distribution
  - groups summary by language and source
  - optionally drops invalid tasks
      |
      v
SplitAwareManifestWriter
  - writes per-language, per-split JSONL manifests
```

The canonical task data comes from `ASRMetadata` and includes fields such as:

```json
{
  "audio_filepath": "/data/curated/gu/dev/audio/gu_valid_0.wav",
  "text": "ગુજરાતી વાક્ય",
  "duration": 3.21,
  "lang": "gu",
  "split_type": "dev",
  "source": "IndicVoices",
  "sample_rate": 16000,
  "num_channels": 1,
  "orig_sample_rate": 48000,
  "orig_num_channels": 1
}
```

## Quick Start: IndicVoices

The repository includes an IndicVoices ingestion handler and a YAML config at `configs/indicvoices.yaml`.

Install the audio dependencies from the Curator repository root:

```bash
uv sync --extra audio_cuda12
source .venv/bin/activate
```

Run the YAML pipeline with the ASR data pipeline runner:

```bash
python tutorials/audio/asr_data_pipeline/main.py \
  --config-path ../../../configs \
  --config-name indicvoices \
  raw_data_dir=/data/asr/IndicVoices/raw \
  output_dir=/data/asr/IndicVoices/curated \
  'langs=[gu]' \
  'stages.0.native_splits=[valid]' \
  'stages.0.split_dir_pattern={lang}/{split}' \
  stages.0.extraction_workers=32
```

For a single-language sample whose downloaded directory looks like `/data/asr/gu/indic_voices/valid`, use:

```bash
python tutorials/audio/asr_data_pipeline/main.py \
  --config-path ../../../configs \
  --config-name indicvoices \
  raw_data_dir=/data/asr/gu/indic_voices \
  output_dir=/data/asr/gu/indic_voices_curated \
  'langs=[gu]' \
  'stages.0.native_splits=[valid]' \
  'stages.0.split_dir_pattern={split}' \
  stages.0.extraction_workers=16
```

`HuggingFaceASRDatasetHandler` passes native `train` through as `train`. For IndicVoices, set `valid_split_strategy: dev_test` to deterministically split native `valid` into `dev` and `test` using `dev_fraction`, which defaults to `0.6`.

## Example YAML

The key parts of `configs/indicvoices.yaml` are:

```yaml
raw_data_dir: /data/asr/IndicVoices/raw
output_dir: /data/asr/IndicVoices/curated
langs:
  - gu

stages:
  - _target_: nemo_curator.stages.audio.asr.datasets.huggingface.HuggingFaceASRDatasetHandler
    raw_data_dir: ${raw_data_dir}
    output_dir: ${output_dir}/uncleaned
    langs: ${langs}
    source_name: IndicVoices
    native_splits:
      - valid
    split_dir_pattern: "{lang}/{split}"
    valid_split_strategy: dev_test
    extraction_workers: 70
    dev_fraction: 0.6
    write_manifest: true
    manifest_splits:
      - train
      - dev
      - test

  - _target_: nemo_curator.stages.audio.asr.normalization.TranscriptNormalizationStage
    text_key: text
    lang_key: lang
    output_text_key: text
    output_original_text_key: text_original
    remove_pnc_chars: false
    lowercase_text: false
    code_switch_langs: []

  - _target_: nemo_curator.stages.audio.asr.normalization.TranscriptStatsStage
    text_key: text
    duration_key: duration
    split_key: split_type
    unknown_chars_key: unknown_chars
    transcript_error_key: transcript_error
    drop_invalid: true
    code_switch_langs: []
    output_summary_path: ${output_dir}/transcript_stats_summary.json

  - _target_: nemo_curator.stages.audio.asr.io.split_manifest_writer.SplitAwareManifestWriter
    output_dir: ${output_dir}
    output_filename_pattern: "{split}_normalized.jsonl"
    langs: ${langs}
    splits:
      - train
      - dev
      - test
```

Set `write_manifest: true` on the ingestion handler only when you also want unnormalized manifests from the source stage. The downstream `SplitAwareManifestWriter` writes the normalized manifests after transcript cleanup and filtering. For known sources such as `IndicVoices`, `Kathbath`, and `Shrutilipi`, source-specific metadata keys are preserved by default based on `source_name`; use `extra_keys` only for a new or custom source.

## Output Layout

For Gujarati IndicVoices with native `valid`, the output structure looks like:

```text
output_dir/
├── transcript_stats_summary.json
├── gu/
│   ├── train_normalized.jsonl
│   ├── dev_normalized.jsonl
│   └── test_normalized.jsonl
└── uncleaned/
    └── gu/
        ├── dev.jsonl
        ├── test.jsonl
        ├── dev/
        │   └── audio/
        │       └── gu_valid_0.wav
        └── test/
            └── audio/
                └── gu_valid_3.wav
```

If `write_manifest` is disabled on the handler, the `uncleaned` JSONL files are not written, but converted audio is still produced.

## Transcript Normalization Resources

Normalization resources live under:

```text
nemo_curator/stages/audio/asr/normalization/langs/
├── remove_chars.txt
└── gu/
    ├── alphabet.txt
    └── pretok.jsonl
```

Each supported language has its own folder:

- `alphabet.txt`: characters considered valid after normalization
- `pretok.jsonl`: regex replacement rules applied before character removal

The root-level `langs/remove_chars.txt` contains characters removed for all languages during cleanup.
Standard punctuation characters are maintained in `_PUNCTUATION_CHARS_BY_LANG` in `nemo_curator/stages/audio/asr/normalization/transcript.py`.

If `remove_pnc_chars: true`, punctuation listed for the language in `_PUNCTUATION_CHARS_BY_LANG` is removed. If `remove_pnc_chars: false`, those punctuation characters are retained during the removal step.

Set `lowercase_text: true` when you want the final normalized transcript lowercased. For code-switched data, set `code_switch_langs` to combine additional language alphabets, pretok rules, and punctuation with the task's primary `lang`; for example, `code_switch_langs: ["en"]` allows English text inside a Gujarati transcript. Use the same `code_switch_langs` on `TranscriptStatsStage` so `alpha_minus_known_chars` is computed against the combined vocabulary.

Rows with characters outside `alphabet.txt` are marked with:

```json
{
  "unknown_chars": {"x": 1},
  "transcript_error": true
}
```

`TranscriptNormalizationStage` always returns the task. Filtering is handled by `TranscriptStatsStage` using `drop_invalid`.

## Statistics Summary

`TranscriptStatsStage` writes an atomic JSON summary to `output_summary_path` and logs a readable summary when the pipeline finishes. The summary is grouped three ways:

- `by_language`: full stats for every language and dataset source
- `by_language_overall`: full stats for each language across all sources
- top-level fields: global stats across every language and source

Each block includes `unknown_chars`, which reports the most frequent unknown characters with both count and dataset-level character rate. Use this to decide whether a character should be added to `alphabet.txt`, normalized in `pretok.jsonl`, or removed with `remove_chars.txt`.

Example display:

```text
[transcript_stats] Transcript normalization summary
  per_language_source:
    lang=gu source=IndicVoices
      transcripts: total=100 valid=95 (95.00%) invalid=5 (5.00%)
      hours: total=2.40 valid_after_filter=2.31 invalid_removed=0.09
      split_hours: {'train': {'total': 1.6, 'valid': 1.55, 'invalid': 0.05}, 'dev': {'total': 0.5, 'valid': 0.48, 'invalid': 0.02}}
      chars: total=8400 unique_known=52 unique_known_rate=0.62% unique_unknown=3 unique_unknown_rate=0.04%
      unknown_chars: @=count=12 rate=0.14%, #=count=4 rate=0.05%, x=count=1 rate=0.01%
      alpha_minus_known_chars: ['ઁ', 'ઋ']
      split_counts: {'train': {'total': 70, 'valid': 67, 'invalid': 3}, 'dev': {'total': 30, 'valid': 28, 'invalid': 2}}

    lang=hi source=IndicVoices
      transcripts: total=80 valid=78 (97.50%) invalid=2 (2.50%)
      hours: total=1.90 valid_after_filter=1.86 invalid_removed=0.04
      split_hours: {'train': {'total': 1.2, 'valid': 1.18, 'invalid': 0.02}, 'test': {'total': 0.7, 'valid': 0.68, 'invalid': 0.02}}
      chars: total=6900 unique_known=58 unique_known_rate=0.84% unique_unknown=1 unique_unknown_rate=0.01%
      unknown_chars: ॐ=count=1 rate=0.01%
      alpha_minus_known_chars: ['ॠ']
      split_counts: {'train': {'total': 55, 'valid': 54, 'invalid': 1}, 'test': {'total': 25, 'valid': 24, 'invalid': 1}}

  per_language_overall:
    lang=gu overall
      transcripts: total=100 valid=95 (95.00%) invalid=5 (5.00%)
      hours: total=2.40 valid_after_filter=2.31 invalid_removed=0.09
      split_hours: {'train': {'total': 1.6, 'valid': 1.55, 'invalid': 0.05}, 'dev': {'total': 0.5, 'valid': 0.48, 'invalid': 0.02}}
      chars: total=8400 unique_known=52 unique_known_rate=0.62% unique_unknown=3 unique_unknown_rate=0.04%
      unknown_chars: @=count=12 rate=0.14%, #=count=4 rate=0.05%, x=count=1 rate=0.01%
      alpha_minus_known_chars: ['ઁ', 'ઋ']
      split_counts: {'train': {'total': 70, 'valid': 67, 'invalid': 3}, 'dev': {'total': 30, 'valid': 28, 'invalid': 2}}

  global:
    all languages/sources
      transcripts: total=180 valid=173 (96.11%) invalid=7 (3.89%)
      hours: total=4.30 valid_after_filter=4.17 invalid_removed=0.13
      unknown_chars: @=count=12 rate=0.08%, #=count=4 rate=0.03%, x=count=1 rate=0.01%, ॐ=count=1 rate=0.01%
```

For multi-source data, make sure every emitted task has a meaningful `source` value. The statistics stage uses the `source` field by default, configurable with `source_key`.

## Multiple Sources and Languages

The normalizer and stats stages can process multiple languages in one stream as long as each task has:

- `lang`: language code with a matching resource folder
- `source`: dataset/source name
- `split_type`: output split name
- `duration`: audio duration in seconds
- `text`: transcript text

A single dataset handler can emit data from many languages and sources. For example, a combined ingestion handler may read:

```text
raw_data_dir/
├── IndicVoices/
│   ├── gu/
│   │   └── valid/
│   └── hi/
│       └── valid/
└── InternalCorpus/
    ├── gu/
    │   └── train.tsv
    └── hi/
        └── train.tsv
```

and emit `source="IndicVoices"` or `source="InternalCorpus"` per row. Avoid chaining multiple fan-out ingestion handlers in a single linear pipeline; instead, create a combined ingestion handler for one streaming run, or run source-specific pipelines and merge manifests afterwards.

## Adding a New Ingestion Handler

New datasets should subclass `BaseASRDatasetHandlerStage`. The handler owns source-specific details: how to find files, read transcripts, decode audio, preserve metadata, and map native splits into Curator `split_type` values.

### Handler Responsibilities

1. Discover raw files or dataset shards under `raw_data_dir`.
2. Decode or extract the source audio.
3. Call `convert_audio()` to write WAV, 16 kHz, mono, PCM16 output.
4. Assign `split_type`, such as `train`, `dev`, or `test`.
5. Create an `ASRMetadata` object for each utterance.
6. Call `write_manifest_entry(meta)` if handler-owned manifests are enabled.
7. Return `AudioTask` objects using `build_audio_task(meta)`.

### Minimal Skeleton

```python
from __future__ import annotations

import os
from dataclasses import dataclass, field

from nemo_curator.stages.audio.asr.datasets.base import BaseASRDatasetHandlerStage
from nemo_curator.stages.audio.asr.metadata import ASRMetadata


@dataclass
class MyDatasetHandler(BaseASRDatasetHandlerStage):
    name: str = "my_dataset_handler"
    source_name: str = "MyDataset"
    native_splits: list[str] = field(default_factory=lambda: ["train", "valid"])

    def _output_splits(self) -> list[str]:
        return ["train", "dev", "test"]

    def assign_split(self, native_split: str, utterance_id: str) -> str:
        if native_split == "valid":
            return "dev"  # or deterministic dev/test logic
        return native_split

    def process(self, _) -> list:
        tasks = []
        for lang in self.langs:
            for native_split in self.native_splits:
                for row in self._iter_rows(lang, native_split):
                    utterance_id = row["id"]
                    split_type = self.assign_split(native_split, utterance_id)
                    audio_dir = self.audio_output_dir(lang, split_type)
                    dst_path = os.path.join(audio_dir, f"{utterance_id}.wav")

                    audio_info = self.convert_audio(
                        row["audio_array"],
                        row["sample_rate"],
                        row.get("num_channels", 1),
                        dst_path,
                    )

                    meta = ASRMetadata(
                        audio_filepath=dst_path,
                        text=row["text"],
                        duration=audio_info["duration"],
                        lang=lang,
                        split_type=split_type,
                        source=self.source_name,
                        sample_rate=self.target_sample_rate,
                        num_channels=self.target_channels,
                        orig_sample_rate=audio_info["orig_sample_rate"],
                        orig_num_channels=audio_info["orig_num_channels"],
                        extra={"speaker_id": row.get("speaker_id")},
                    )
                    self.write_manifest_entry(meta)
                    tasks.append(self.build_audio_task(meta))
        return tasks
```

Use `setup()` for heavy imports such as `datasets`, `soundfile`, or dataset-specific SDKs. This keeps driver-side imports light and matches Curator stage patterns.

### YAML Entry

After adding the class, point the YAML `_target_` at the new handler:

```yaml
stages:
  - _target_: nemo_curator.stages.audio.asr.datasets.my_dataset.MyDatasetHandler
    raw_data_dir: /data/asr/MyDataset/raw
    output_dir: /data/asr/MyDataset/curated/uncleaned
    langs:
      - gu
      - hi
    native_splits:
      - train
      - valid
    extraction_workers: 32
    skip_untar: false
    write_manifest: false

  - _target_: nemo_curator.stages.audio.asr.normalization.TranscriptNormalizationStage
    remove_pnc_chars: true

  - _target_: nemo_curator.stages.audio.asr.normalization.TranscriptStatsStage
    drop_invalid: true
    output_summary_path: /data/asr/MyDataset/curated/transcript_stats_summary.json

  - _target_: nemo_curator.stages.audio.asr.io.split_manifest_writer.SplitAwareManifestWriter
    output_dir: /data/asr/MyDataset/curated
    output_filename_pattern: "{split}_normalized.jsonl"
    langs:
      - gu
      - hi
    splits:
      - train
      - dev
      - test
```

## Adding a New Language

To normalize a new language, add a folder under `nemo_curator/stages/audio/asr/normalization/langs/`:

```text
langs/
├── remove_chars.txt
└── xx/
    ├── alphabet.txt
    └── pretok.jsonl
```

Then set `lang: "xx"` in the emitted `ASRMetadata` and add the language's punctuation characters to `_PUNCTUATION_CHARS_BY_LANG` in `nemo_curator/stages/audio/asr/normalization/transcript.py`.

Start with a conservative `alphabet.txt`. Unknown characters are surfaced in the stats summary, so you can inspect the first run and decide whether the character is valid, should be normalized by `pretok.jsonl`, or should be removed by `remove_chars.txt`.

## Performance Notes

- Dataset handlers run with `xenna_workers=1` by default and parallelize extraction internally using `extraction_workers`.
- Increase `extraction_workers` for CPU-heavy decoding and resampling, but keep it within available CPU and I/O limits.
- Use `skip_untar: true` to reuse converted audio files from a previous run.
- `TranscriptStatsStage` runs as one worker so the dataset-level summary is exact.
- `SplitAwareManifestWriter` runs as one worker to avoid concurrent writes to the same manifest file.

## Troubleshooting

### `ModuleNotFoundError: nemo_curator.stages.audio.asr`

Make sure you are running from the Curator checkout and importing the local package:

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```

or reinstall Curator in editable mode:

```bash
pip install -e .
```

### No Output Rows

- Check that `raw_data_dir` and `split_dir_pattern` match the downloaded dataset layout.
- Check handler logs for `skipped_missing_text`, `skipped_missing_audio`, and `skipped_audio_load`.
- If `TranscriptStatsStage(drop_invalid=true)` is enabled, inspect `transcript_stats_summary.json` for invalid transcript counts and unknown characters.

### Missing Normalization Resources

If normalization fails with `Missing ASR normalization resource`, add the language folder and the required files under `normalization/langs/`.

### Unexpected Punctuation Removal

Set `remove_pnc_chars: false` to retain punctuation listed in `_PUNCTUATION_CHARS_BY_LANG`. Characters in `remove_chars.txt` that are not punctuation will still be removed.

## Related Files

- `configs/indicvoices.yaml`
- `tutorials/audio/asr_data_pipeline/main.py`
- `tutorials/audio/indicvoices/pipeline.py`
- `nemo_curator/stages/audio/asr/datasets/base.py`
- `nemo_curator/stages/audio/asr/datasets/indicvoices.py`
- `nemo_curator/stages/audio/asr/normalization/transcript.py`
- `nemo_curator/stages/audio/asr/normalization/stats.py`
- `nemo_curator/stages/audio/asr/io/split_manifest_writer.py`
