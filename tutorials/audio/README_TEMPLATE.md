# <Tutorial Title>

<One-line description of what this tutorial does and why it matters.>

## Overview

<2-3 sentences expanding on the one-liner. Mention the key stages and what makes this pipeline useful.>

### Pipeline flow

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Stage 1    │───▶│   Stage 2    │───▶│   Stage 3    │───▶│   Stage N    │
│  (describe)  │    │  (describe)  │    │  (describe)  │    │  (describe)  │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
     input              ...                  ...               output
```

## Prerequisites

- Python 3.11+
- NeMo Curator installed (see [installation guide](https://docs.nvidia.com/nemo/curator/latest/admin/installation.html))
- **GPU**: <required / recommended / not needed> — <minimum VRAM if applicable>
- **System packages**: <list any, e.g. `sox`, or "None">

```bash
# GPU (recommended)
uv sync --extra audio_cuda12

# CPU only
uv sync --extra audio_cpu
```

## Dataset

<Describe the dataset: source, format, size, license.>

| Property | Value |
|---|---|
| **Source** | <link> |
| **Format** | <e.g. WAV mono 16kHz, JSONL manifest> |
| **Size** | <download size and/or number of files> |
| **License** | <license name + link> |
| **Auto-download** | <Yes — via `--flag` / No — requires manual acquisition> |

## Quick start

Audio tutorials are YAML-only: put the pipeline in `pipeline.yaml` and run it
with the generic runner (`--config-path` is relative to
`nemo_curator/config/run.py`). Put reusable logic in a library stage, not in a
tutorial script.

Run with bundled or auto-downloaded sample data in under 2 minutes:

```bash
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/<tutorial> \
  --config-name pipeline \
  <minimal overrides>
```

Expected output:

```
<paste a representative snippet of terminal output>
```

## Usage

### Common overrides

| Override | Default | Description |
|---|---|---|
| `manifest_path` | *(required)* | <description> |
| `output_path` | `./<tutorial>_output.jsonl` | <description> |
| `backend` | `ray_data` | Execution backend: `xenna` or `ray_data` |
| `stages.<i>.<field>` | | Override one stage argument |
| `+stages.<i>.num_workers=N` | executor default | Cap one stage's worker count (`+` adds a key the YAML does not set) |

### Using custom data

```bash
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/<tutorial> \
  --config-name pipeline \
  manifest_path=/path/to/your/manifest.jsonl \
  output_path=./my_output.jsonl
```

### Choosing a backend

| Backend | Description | When to use |
|---|---|---|
| `xenna` | Cosmos-Xenna streaming engine with automatic worker allocation. Set `execution_mode` to `streaming` or `batch`. | Most workloads, CI/nightly benchmarks. |
| `ray_data` | Built on Ray Data `map_batches`. | Development, machines without Xenna GPU support, or Ray Data integration preferred. |

## Pipeline stages

<Describe each stage in order: what it does, key parameters, what it adds to the task data.>

### Stage 1: `<StageName>`

<What it does. 2-3 sentences.>

### Stage N: `<StageName>`

<What it does.>

## Parameters and tuning

| Parameter | Range | Effect |
|---|---|---|
| `<param>` | `<low>` – `<high>` | <What happens at each extreme> |

## Output format

Results are written to `<path>`. Each line contains:

```json
{
  "field1": "<type — description>",
  "field2": "<type — description>"
}
```

| Field | Type | Description |
|---|---|---|
| `field1` | string | <description> |
| `field2` | float | <description> |

## Performance

| Metric | Value | Hardware |
|---|---|---|
| Throughput | <X files/sec or samples/sec> | <GPU model, CPU count> |
| Total time (sample data) | <X seconds> | <hardware> |

## Composability

This tutorial's stages can be combined with other NeMo Curator audio stages:

```python
from nemo_curator.pipeline import Pipeline

pipeline = Pipeline(
    name="custom",
    stages=[
        # ... upstream stages ...
        ThisTutorialStage(...),
        # ... downstream stages ...
    ],
)
```

<Mention which upstream/downstream stages are natural pairings.>

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| <symptom> | <why it happens> | <what to do> |

## Citation / License

<Dataset citation, model card link, license terms.>
