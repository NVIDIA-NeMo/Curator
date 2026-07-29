# Clinical Report Curation

This tutorial demonstrates a curation pipeline for clinical/medical text
reports (e.g. radiology, pathology, or general practice notes), combining
PII anonymization with domain-specific quality filtering.

## What this pipeline does

1. **PII anonymization** ([`GlinerPiiRedactor`](../gliner-pii-redaction/gliner_pii_redactor.py)) —
   runs first, following a privacy-by-design principle: patient identifiers
   are stripped before any other stage touches the data. The GLiNER PII
   label set includes healthcare-specific identifiers out of the box
   (`medical_record_number`, `health_plan_beneficiary_number`), making it
   directly applicable to clinical text without modification.

2. **Generic heuristic pre-filters** (`WordCountFilter`, `NonAlphaNumericFilter`) —
   cheap, applied early to discard obviously low-quality documents before
   more expensive stages run. Thresholds are re-tuned for medical text:
   the default `NonAlphaNumericFilter` cutoff is often too strict for
   reports full of dosages, lab values, and units (e.g. `120/80 mmHg`,
   `0.4mg`).

3. **`ClinicalSectionFilter` (score-only pass)** — records how many
   distinct clinical sections (e.g. *History*, *Diagnosis*, *Plan*) each
   document contains, without discarding anything yet. This is useful for
   inspecting the score distribution on your real corpus before committing
   to a `min_sections` cutoff.

4. **`ClinicalSectionFilter` (filtering pass)** — discards documents that
   don't meet the minimum number of distinct clinical sections, which
   typically indicates a fragmented, truncated, or non-clinical document.

5. **Write** — the curated, anonymized dataset is written back to JSONL,
   in a subfolder named after the report language (e.g. `curated/en/`).

## Why this filter combination

Generic text-quality filters (word count, symbol ratio, repetition) catch
boilerplate and garbage text, but they say nothing about whether a document
is *structurally* a valid clinical report. `ClinicalSectionFilter` closes
that gap by checking for the presence of standard report sections
(anamnesis/history, examination, diagnosis, treatment/plan), in five
languages (`en`, `it`, `es`, `fr`, `de`).

## Requirements

```bash
uv pip install gliner
```

By default, this tutorial runs `GlinerPiiRedactor` on **CPU** (`--use_gpu`
is `False` unless passed explicitly). To enable GPU acceleration, pass
`--use_gpu` and follow the hardware/software requirements documented in the
[`gliner-pii-redaction`](../gliner-pii-redaction/) tutorial this stage is
borrowed from: 1 NVIDIA GPU, Volta architecture or newer, CUDA 12.x
(tested with `gliner==0.2.24`).

The first run downloads the `nvidia/gliner-pii` model from Hugging Face
(cached locally afterward).

## Usage

`main.py` imports `GlinerPiiRedactor` from a local module, so run it from
this directory (not the repository root):

```bash
cd /path/to/Curator/tutorials/text/clinical-report-curation
python main.py --input_dir data/clinical_notes --language en
```

| Argument | Description | Default |
|---|---|---|
| `--input_dir` | Directory of raw clinical report `.jsonl` files (one `{"text": ...}` per line) | *(required)* |
| `--data_root` | Directory where curated output is written | `./data` |
| `--language` | Report language for section-keyword matching: `en`, `it`, `es`, `fr`, `de` | `en` |
| `--use_gpu` | Enable GPU acceleration for the GLiNER PII model | `False` |

A small synthetic dataset (`data/clinical_notes/sample_clinical_notes.jsonl`)
is included for a quick end-to-end test — it mixes valid multilingual
clinical reports, generic non-clinical filler text, and a variety of fake
PII (names, addresses, phone numbers, national IDs) to verify that both
redaction and structural filtering behave as expected.

```bash
python main.py --input_dir data/clinical_notes --language en
cat data/curated/en/*.jsonl
```

## Known limitation

`ClinicalSectionFilter` matches section keywords for a single language per
run. A multilingual corpus therefore needs either (a) separate runs per
language on pre-split data, or (b) a language-detection stage placed before
the filter to route each document to the correct keyword set. The latter is
a natural extension not implemented here, to keep this example focused.
