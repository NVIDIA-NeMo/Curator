---
description: "Perform distributed data classification using GPU-accelerated models for domain, quality, safety, and content assessment"
categories: ["how-to-guides"]
tags: ["distributed-classification", "gpu", "domain", "quality", "safety", "crossfit", "scalable"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "text-only"
---

(text-process-data-filter-dist-classifier)=
# Distributed Data Classification

NVIDIA NeMo Curator provides a module for performing distributed classification on large text datasets using GPU acceleration. This enables the categorization and filtering of text documents based on multiple dimensions such as domain, quality, safety, educational value, content type, and more. These classifications can enhance the quality of training data for large language models by identifying high-value content and removing problematic material.

## How It Works

Classification pipelines are built from `ray_curator` stages and executed with a distributed executor:

- Parallelize with `XennaExecutor` or other backends
- Use pre-trained HF models via `DistributedDataClassifier` subclasses
- Tokenize once with `TokenizerStage` and run batched model inference
- Optionally filter results in the same composite stage

---

## Usage

Use the built-in classifier stages; models must fit in a single GPU. Run with a distributed executor.

### Classifier Comparison

| Classifier | Purpose | Model Location | Key Parameters | Requirements |
|---|---|---|---|---|
| DomainClassifier | Categorize English text by domain | [nvidia/domain-classifier](https://huggingface.co/nvidia/domain-classifier) | `filter_by`, `text_field` | None |
| MultilingualDomainClassifier | Categorize text in 52 languages by domain | [nvidia/multilingual-domain-classifier](https://huggingface.co/nvidia/multilingual-domain-classifier) | `filter_by`, `text_field` | None |
| QualityClassifier | Assess document quality | [nvidia/quality-classifier-deberta](https://huggingface.co/nvidia/quality-classifier-deberta) | `filter_by`, `text_field` | None |
| AegisClassifier | Detect unsafe content | [nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0](https://huggingface.co/nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0) | `aegis_variant`, `filter_by` | HuggingFace token |
| InstructionDataGuardClassifier | Detect poisoning attacks | [nvidia/instruction-data-guard](https://huggingface.co/nvidia/instruction-data-guard) | `text_field`, `pred_column` | HuggingFace token |
| FineWebEduClassifier | Score educational value | [HuggingFaceFW/fineweb-edu-classifier](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier) | `pred_column`, `int_column` | None |
| FineWebMixtralEduClassifier | Score educational value (Mixtral annotations) | [nvidia/nemocurator-fineweb-mixtral-edu-classifier](https://huggingface.co/nvidia/nemocurator-fineweb-mixtral-edu-classifier) | `pred_column`, `int_column` | None |
| FineWebNemotronEduClassifier | Score educational value (Nemotron annotations) | [nvidia/nemocurator-fineweb-nemotron-4-edu-classifier](https://huggingface.co/nvidia/nemocurator-fineweb-nemotron-4-edu-classifier) | `pred_column`, `int_column` | None |
| ContentTypeClassifier | Categorize by speech type | [nvidia/content-type-classifier-deberta](https://huggingface.co/nvidia/content-type-classifier-deberta) | `filter_by`, `text_field` | None |
| PromptTaskComplexityClassifier | Classify prompt tasks and complexity | [nvidia/prompt-task-and-complexity-classifier](https://huggingface.co/nvidia/prompt-task-and-complexity-classifier) | `text_field` | None |

### Domain Classifier

The Domain Classifier categorizes English text documents into specific domains or subject areas.

```python
from ray_curator.pipeline.pipeline import Pipeline
from ray_curator.backends.xenna.executor import XennaExecutor
from ray_curator.stages.text.io.reader.jsonl import JsonlReader
from ray_curator.stages.text.classifiers.domain import DomainClassifier
from ray_curator.stages.text.io.writer.jsonl import JsonlWriter

pipeline = Pipeline(name="domain_classification")
pipeline.add_stage(JsonlReader(file_paths="books_dataset/*.jsonl"))
pipeline.add_stage(DomainClassifier(filter_by=["Games", "Sports"]))
pipeline.add_stage(JsonlWriter(output_dir="games_and_sports/"))

XennaExecutor().run(stages=pipeline._stages)  # or pipeline.run(XennaExecutor())
```

### Multilingual Domain Classifier

Functionally similar to the Domain Classifier, but supports 52 languages.

```python
from ray_curator.stages.text.classifiers.domain import MultilingualDomainClassifier

pipeline = Pipeline(name="multilingual_domain")
pipeline.add_stage(JsonlReader(file_paths="multilingual_dataset/*.jsonl"))
pipeline.add_stage(MultilingualDomainClassifier(filter_by=["Games", "Sports"]))
pipeline.add_stage(JsonlWriter(output_dir="multilingual_domains/"))
pipeline.run(XennaExecutor())
```

### Quality Classifier

The Quality Classifier assesses document quality on a scale from Low to High.

```python
from ray_curator.stages.text.classifiers.quality import QualityClassifier

pipeline = Pipeline(name="quality_classifier")
pipeline.add_stage(JsonlReader(file_paths="web_documents/*.jsonl"))
pipeline.add_stage(QualityClassifier(filter_by=["High", "Medium"]))
pipeline.add_stage(JsonlWriter(output_dir="quality_medium_plus/"))
pipeline.run(XennaExecutor())
```

### AEGIS Safety Model

The AEGIS classifier detects unsafe content across 13 critical risk categories. It requires a HuggingFace token for access to Llama Guard.

```python
from ray_curator.stages.text.classifiers.aegis import AegisClassifier

token = "hf_1234"  # Your HuggingFace user access token
pipeline = Pipeline(name="aegis")
pipeline.add_stage(JsonlReader(file_paths="content/*.jsonl"))
pipeline.add_stage(AegisClassifier(
    aegis_variant="nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0",
    token=token,
    filter_by=["safe", "O13"],
))
pipeline.add_stage(JsonlWriter(output_dir="safe_and_O13/"))
pipeline.run(XennaExecutor())
```

The classifier adds a column with labels: "safe," "O1" through "O13" (each representing specific safety risks), or "unknown." For raw LLM output, use:

```python
safety_classifier = AegisClassifier(
    aegis_variant="nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0",
    token=token,
    keep_raw_pred=True,
    raw_pred_column="raw_predictions"
)
```

### Instruction Data Guard

Detects LLM poisoning attacks in instruction-response datasets. Requires HuggingFace token access.

```python
from ray_curator.stages.text.classifiers.aegis import InstructionDataGuardClassifier

token = "hf_1234"  # Your HuggingFace user access token
pipeline = Pipeline(name="instruction_data_guard")
pipeline.add_stage(JsonlReader(file_paths="instruction_data/*.jsonl"))
pipeline.add_stage(InstructionDataGuardClassifier(token=token))
pipeline.add_stage(JsonlWriter(output_dir="guard_scores/"))
pipeline.run(XennaExecutor())
```

The output includes two columns: a float score `instruction_data_guard_poisoning_score` and a Boolean `is_poisoned`.

### FineWeb Educational Content Classifier

Scores documents on educational value from 0–5. This helps prioritize content for knowledge-intensive tasks.

```python
from ray_curator.stages.text.classifiers.fineweb_edu import FineWebEduClassifier

pipeline = Pipeline(name="fineweb_edu")
pipeline.add_stage(JsonlReader(file_paths="web_documents/*.jsonl"))
pipeline.add_stage(FineWebEduClassifier(
    model_inference_batch_size=256,
    pred_column="fineweb-edu-score-float",
    int_score_column="fineweb-edu-score-int",
))
pipeline.add_stage(JsonlWriter(output_dir="fineweb_scores/"))
pipeline.run(XennaExecutor())
```

### FineWeb Mixtral and Nemotron Edu Classifiers

Similar to the FineWeb Edu Classifier but trained with different annotation sources:

- **FineWebMixtralEduClassifier**: Uses annotations from Mixtral 8x22B-Instruct
- **FineWebNemotronEduClassifier**: Uses annotations from Nemotron-4-340B-Instruct

Both provide a quality label column marking scores above 2.5 as "high_quality":

```python
from ray_curator.stages.text.classifiers.fineweb_edu import (
    FineWebMixtralEduClassifier,
    FineWebNemotronEduClassifier,
)

pipeline = Pipeline(name="fineweb_mixtral")
pipeline.add_stage(JsonlReader(file_paths="web_documents/*.jsonl"))
pipeline.add_stage(FineWebMixtralEduClassifier())
pipeline.add_stage(JsonlWriter(output_dir="fineweb_mixtral/"))
pipeline.run(XennaExecutor())
```

### Content Type Classifier

Categorizes documents into 11 distinct speech types.

```python
from ray_curator.stages.text.classifiers.content_type import ContentTypeClassifier

pipeline = Pipeline(name="content_type")
pipeline.add_stage(JsonlReader(file_paths="content/*.jsonl"))
pipeline.add_stage(ContentTypeClassifier(filter_by=["Blogs", "News"]))
pipeline.add_stage(JsonlWriter(output_dir="blogs_or_news/"))
pipeline.run(XennaExecutor())
```

### Prompt Task and Complexity Classifier

Classifies prompts by task type and complexity dimensions.

```python
from ray_curator.stages.text.classifiers.prompt_task_complexity import (
    PromptTaskComplexityClassifier,
)

pipeline = Pipeline(name="prompt_task_and_complexity")
pipeline.add_stage(JsonlReader(file_paths="prompts/*.jsonl"))
pipeline.add_stage(PromptTaskComplexityClassifier())
pipeline.add_stage(JsonlWriter(output_dir="prompt_task_complexity/"))
pipeline.run(XennaExecutor())
```

## Execution Backends

Pipelines can run on different executors. The `XennaExecutor` schedules stages with declared `Resources` over available nodes/GPUs. See `ray_curator.backends.xenna` for details.