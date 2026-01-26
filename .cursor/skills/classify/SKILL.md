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

### Quality Classifiers

| Classifier | Purpose | GPU Memory |
|------------|---------|------------|
| `QualityClassifier` | General quality scoring | ~8 GB |
| `FineWebEduClassifier` | Educational quality (1-5 scale) | ~12 GB |
| `FineWebMixtralEduClassifier` | Edu quality (Mixtral-based) | ~16 GB |
| `FineWebNemotronEduClassifier` | Edu quality (Nemotron-based) | ~16 GB |

### Domain Classifiers

| Classifier | Purpose | GPU Memory |
|------------|---------|------------|
| `DomainClassifier` | Categorize by domain | ~8 GB |
| `MultilingualDomainClassifier` | Multi-language domains | ~12 GB |
| `ContentTypeClassifier` | Content type detection | ~8 GB |

### Safety Classifiers

| Classifier | Purpose | GPU Memory |
|------------|---------|------------|
| `AegisClassifier` | Content safety scoring | ~16 GB |
| `InstructionDataGuardClassifier` | Instruction safety | ~12 GB |

### Complexity Classifiers

| Classifier | Purpose | GPU Memory |
|------------|---------|------------|
| `PromptTaskComplexityClassifier` | Prompt complexity | ~8 GB |

## Quick Start

### Quality Classification

```yaml
stages:
  - _target_: nemo_curator.stages.text.classifiers.QualityClassifier
    batch_size: 64
```

### Educational Content Scoring

```yaml
stages:
  - _target_: nemo_curator.stages.text.classifiers.FineWebEduClassifier
    batch_size: 32

  # Filter by edu score >= 3 (out of 5)
  - _target_: nemo_curator.stages.text.modules.ScoreFilter
    score_field: "edu_score"
    threshold: 3
```

### Domain Classification

```yaml
stages:
  - _target_: nemo_curator.stages.text.classifiers.DomainClassifier
    batch_size: 64

  # Optional: Filter by specific domains
  # - _target_: nemo_curator.stages.text.modules.Filter
  #   filter_field: "domain"
  #   keep_values: ["science", "technology", "education"]
```

### Safety Filtering

```yaml
stages:
  - _target_: nemo_curator.stages.text.classifiers.AegisClassifier
    batch_size: 16

  # Remove unsafe content
  - _target_: nemo_curator.stages.text.modules.ScoreFilter
    score_field: "safety_score"
    threshold: 0.5  # Adjust based on requirements
```

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

The `DomainClassifier` produces categories including:

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
- `Travel`

## Batch Size Tuning

| GPU Memory | Recommended Batch Size |
|------------|------------------------|
| 8 GB | 16-32 |
| 16 GB | 32-64 |
| 24 GB | 64-128 |
| 40 GB | 128-256 |

Larger batch sizes improve throughput but require more memory.

## Classifier Output Fields

Each classifier adds fields to the data:

| Classifier | Output Field | Type |
|------------|--------------|------|
| QualityClassifier | `quality_score` | float |
| FineWebEduClassifier | `edu_score` | float (0-5) |
| DomainClassifier | `domain` | string |
| AegisClassifier | `safety_score` | float |
| ContentTypeClassifier | `content_type` | string |

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
  # Quality scoring
  - _target_: nemo_curator.stages.text.classifiers.QualityClassifier
    batch_size: 64

  # Filter low quality
  - _target_: nemo_curator.stages.text.modules.ScoreFilter
    score_field: "quality_score"
    threshold: 0.5

  # Domain classification
  - _target_: nemo_curator.stages.text.classifiers.DomainClassifier
    batch_size: 64

  # Educational scoring
  - _target_: nemo_curator.stages.text.classifiers.FineWebEduClassifier
    batch_size: 32

  # Safety check
  - _target_: nemo_curator.stages.text.classifiers.AegisClassifier
    batch_size: 16
```

## Related Skills

- `/filter` - Heuristic filtering (run before classification)
- `/curate` - Full curation workflow
- `/stages` - All available stages
