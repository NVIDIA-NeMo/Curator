---
name: classify
description: |
  Run ML classifiers on text datasets for quality, domain, safety, and
  educational content scoring. Use when the user wants to classify text,
  score quality, detect domains, filter by safety, or assess educational value.
license: Apache-2.0
metadata:
  author: nvidia
  version: "1.0"
  modality: text
  gpu-required: true
disable-model-invocation: true
---

# Text Classification

Run ML classifiers on text datasets for quality assessment.

## When to Use

- Scoring text quality for LLM training
- Filtering by educational content
- Domain classification
- Safety/content moderation
- After heuristic filtering, before deduplication

## Available Classifiers

All classifiers below are **CompositeStages** that decompose into tokenizer and model stages.

### Quality Classifiers

| Classifier | Purpose | Output |
|------------|---------|--------|
| `QualityClassifier` | General quality scoring | "High", "Medium", "Low" |
| `FineWebEduClassifier` | Educational quality (0-5 scale) | Binary label + float (0-5) + int (0-5) scores |
| `FineWebMixtralEduClassifier` | Edu quality (Mixtral-based) | Binary label + float (0-5) + int (0-5) scores |
| `FineWebNemotronEduClassifier` | Edu quality (Nemotron-based) | Binary label + float (0-5) + int (0-5) scores |

### Domain Classifiers

| Classifier | Purpose | Output |
|------------|---------|--------|
| `DomainClassifier` | Categorize by domain | Domain category string |
| `MultilingualDomainClassifier` | Multi-language domains | Domain category string |
| `ContentTypeClassifier` | Content type detection | Content type string |

### Safety Classifiers

| Classifier | Purpose | Output |
|------------|---------|--------|
| `AegisClassifier` | Content safety classification | "safe", "O1"-"O13", or "unknown" |
| `InstructionDataGuardClassifier` | Instruction safety | Safety category |

### Complexity Classifiers

| Classifier | Purpose | Output |
|------------|---------|--------|
| `PromptTaskComplexityClassifier` | Prompt complexity | Complexity scores |

## Classifier Output Fields

Each classifier adds specific fields to the data (defaults shown):

| Classifier | Output Field(s) | Type |
|------------|-----------------|------|
| `QualityClassifier` | `quality_pred` | string ("High", "Medium", "Low") |
| `FineWebEduClassifier` | `fineweb-edu-score-label`, `fineweb-edu-score-float`, `fineweb-edu-score-int` | string ("high_quality" or "low_quality"), float (0.0-5.0), int (0-5) |
| `DomainClassifier` | `domain_pred` | string |
| `AegisClassifier` | `aegis_pred` | string ("safe", "O1"-"O13", or "unknown") |
| `ContentTypeClassifier` | `content_pred` | string |

**Note**: Field names are configurable via `label_field` parameter.

## Quick Start

### Quality Classification

```yaml
stages:
  - _target_: nemo_curator.stages.text.classifiers.QualityClassifier
    model_inference_batch_size: 256  # Default, adjust based on GPU memory
    # Output: "quality_pred" column with "High", "Medium", or "Low"
```

To keep only high-quality documents, use the built-in `filter_by` parameter:

```yaml
stages:
  - _target_: nemo_curator.stages.text.classifiers.QualityClassifier
    model_inference_batch_size: 256
    filter_by: ["High", "Medium"]  # Keep High and Medium quality
```

### Educational Content Scoring

```yaml
stages:
  - _target_: nemo_curator.stages.text.classifiers.FineWebEduClassifier
    model_inference_batch_size: 256
    # Outputs:
    #   - "fineweb-edu-score-label": string label
    #   - "fineweb-edu-score-float": float score (0.0-5.0)
    #   - "fineweb-edu-score-int": integer score (0-5)
```

To filter by educational quality, use the label field (binary: "high_quality" or "low_quality"):

```yaml
stages:
  - _target_: nemo_curator.stages.text.classifiers.FineWebEduClassifier
    model_inference_batch_size: 256
    filter_by: ["high_quality"]  # Keep only high quality (score >= 2.5)
```

**Note**: The `filter_by` parameter filters on `label_field`, which contains `"high_quality"` (score >= 2.5) or `"low_quality"` (score < 2.5). To filter by specific integer scores, add a separate filter stage after classification.

### Domain Classification

```yaml
stages:
  - _target_: nemo_curator.stages.text.classifiers.DomainClassifier
    model_inference_batch_size: 256
    # Output: "domain_pred" column with domain category
```

To keep specific domains:

```yaml
stages:
  - _target_: nemo_curator.stages.text.classifiers.DomainClassifier
    model_inference_batch_size: 256
    filter_by: ["Science", "Computers_and_Electronics", "Jobs_and_Education"]
```

### Safety Filtering with Aegis

The `AegisClassifier` outputs "safe", category codes "O1" through "O13" representing different unsafe content types, or "unknown" if the response cannot be parsed.

```yaml
stages:
  - _target_: nemo_curator.stages.text.classifiers.AegisClassifier
    aegis_variant: "nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0"
    model_inference_batch_size: 64
    filter_by: ["safe"]  # Keep only safe content
```

**Aegis Categories** (from [HuggingFace](https://huggingface.co/nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0)):
- `safe`: Content is safe
- `O1`-`O13`: Various unsafe content categories (Violence, Sexual, Criminal Planning, etc.)
- `unknown`: Response could not be parsed

**Note**: Aegis requires HuggingFace access to `meta-llama/LlamaGuard-7b`. Set `hf_token` parameter.

## Common Parameters

All classifiers share these parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_inference_batch_size` | 256 | Batch size for model inference |
| `label_field` | varies | Column name for predictions |
| `text_field` | "text" | Input text column name |
| `filter_by` | None | List of labels to keep |
| `max_chars` | varies | Max characters to process |
| `sort_by_length` | True | Sort by length for efficiency |
| `autocast` | True | Use mixed precision |

## Educational Score Guide (FineWebEdu)

The FineWebEdu classifiers produce scores from 0-5:

| Score | Meaning | Use Case |
|-------|---------|----------|
| 0-1 | Low educational value | Filter out |
| 2 | Some educational content | Include for general LLMs |
| 3 | Moderate educational value | Good for instruction-following |
| 4 | High educational value | Excellent for knowledge LLMs |
| 5 | Exceptional educational content | Premium training data |

### Recommended Thresholds

| Use Case | Threshold | Resulting Data |
|----------|-----------|----------------|
| General LLM | >= 2 | Broad coverage |
| Instruction LLM | >= 3 | Quality-focused |
| Educational LLM | >= 4 | High-quality subset |
| Premium dataset | >= 4.5 | Best-of-best |

## Domain Categories

The `DomainClassifier` produces these categories:

- `Arts_and_Entertainment`
- `Autos_and_Vehicles`
- `Beauty_and_Fitness`
- `Books_and_Literature`
- `Business_and_Industrial`
- `Computers_and_Electronics`
- `Finance`
- `Food_and_Drink`
- `Games`
- `Health`
- `Hobbies_and_Leisure`
- `Home_and_Garden`
- `Internet_and_Telecom`
- `Jobs_and_Education`
- `Law_and_Government`
- `News`
- `Online_Communities`
- `People_and_Society`
- `Pets_and_Animals`
- `Real_Estate`
- `Reference`
- `Science`
- `Shopping`
- `Sports`
- `Travel_and_Transportation`

## Pipeline Ordering

Recommended order for classifiers in a pipeline:

```
1. Heuristic Filtering (fast, removes obvious junk)
2. Quality Classification (removes low-quality)
3. Domain Classification (categorizes content)
4. Safety Classification (removes unsafe)
5. Educational Scoring (identifies high-value)
6. Deduplication (removes duplicates)
```

Running expensive classifiers after filtering reduces compute costs.

## Multi-Classifier Pipeline

```yaml
stages:
  # Quality scoring and filtering
  - _target_: nemo_curator.stages.text.classifiers.QualityClassifier
    model_inference_batch_size: 256
    filter_by: ["High", "Medium"]

  # Domain classification
  - _target_: nemo_curator.stages.text.classifiers.DomainClassifier
    model_inference_batch_size: 256

  # Educational scoring
  - _target_: nemo_curator.stages.text.classifiers.FineWebEduClassifier
    model_inference_batch_size: 256

  # Safety check - keep only safe content
  - _target_: nemo_curator.stages.text.classifiers.AegisClassifier
    aegis_variant: "nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0"
    model_inference_batch_size: 64
    filter_by: ["safe"]
```

## Related Skills

- `/filter` - Heuristic filtering (run before classification)
- `/curate` - Full curation workflow
- `/stages` - All available stages
