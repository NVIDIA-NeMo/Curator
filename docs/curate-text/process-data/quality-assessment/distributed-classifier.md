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

The distributed data classification in NeMo Curator works by:

1. **Parallel Processing**: Chunking datasets across multiple computing nodes and GPUs to accelerate classification
2. **Pre-trained Models**: Using specialized models for different classification tasks
3. **Batched Inference**: Optimizing throughput with intelligent batching via CrossFit integration
4. **Consistent API**: Providing a unified interface through the `DistributedDataClassifier` base class

The `DistributedDataClassifier` is designed to run on GPU clusters with minimal code changes regardless of which specific classifier you're using. All classifiers support filtering based on classification results and storing prediction scores as metadata.

:::{note}
Distributed classification requires GPU acceleration and is not supported for CPU-only processing. As long as GPU resources are available and NeMo Curator is correctly installed, GPU acceleration is handled automatically.
:::

---

## Usage

NVIDIA NeMo Curator provides a base class `DistributedDataClassifier` that can be extended to fit your specific model. The only requirement is that the model can fit on a single GPU. This module operates on the GPU and works within the pipeline framework using DocumentBatch processing.

### Classifier Comparison

| Classifier | Purpose | Model Location | Key Parameters | Requirements |
|---|---|---|---|---|
| DomainClassifier | Categorize English text by domain | [nvidia/domain-classifier](https://huggingface.co/nvidia/domain-classifier) | `filter_by`, `text_field` | None |
| MultilingualDomainClassifier | Categorize text in 52 languages by domain | [nvidia/multilingual-domain-classifier](https://huggingface.co/nvidia/multilingual-domain-classifier) | `filter_by`, `text_field` | None |
| QualityClassifier | Assess document quality | [nvidia/quality-classifier-deberta](https://huggingface.co/nvidia/quality-classifier-deberta) | `filter_by`, `text_field` | None |
| AegisClassifier | Detect unsafe content | [nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0](https://huggingface.co/nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0) | `aegis_variant`, `filter_by` | HuggingFace token |
| InstructionDataGuardClassifier | Detect poisoning attacks | [nvidia/instruction-data-guard](https://huggingface.co/nvidia/instruction-data-guard) | `text_field`, `label_field` | HuggingFace token |
| FineWebEduClassifier | Score educational value | [HuggingFaceFW/fineweb-edu-classifier](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier) | `label_field`, `int_field` | None |
| FineWebMixtralEduClassifier | Score educational value (Mixtral annotations) | [nvidia/nemocurator-fineweb-mixtral-edu-classifier](https://huggingface.co/nvidia/nemocurator-fineweb-mixtral-edu-classifier) | `label_field`, `int_field`, `model_inference_batch_size=1024` | None |
| FineWebNemotronEduClassifier | Score educational value (Nemotron annotations) | [nvidia/nemocurator-fineweb-nemotron-4-edu-classifier](https://huggingface.co/nvidia/nemocurator-fineweb-nemotron-4-edu-classifier) | `label_field`, `int_field`, `model_inference_batch_size=1024` | None |
| ContentTypeClassifier | Categorize by speech type | [nvidia/content-type-classifier-deberta](https://huggingface.co/nvidia/content-type-classifier-deberta) | `filter_by`, `text_field` | None |
| PromptTaskComplexityClassifier | Classify prompt tasks and complexity | [nvidia/prompt-task-and-complexity-classifier](https://huggingface.co/nvidia/prompt-task-and-complexity-classifier) | `text_field` | None |

### Domain Classifier

The Domain Classifier categorizes English text documents into specific domains or subject areas.

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.classifiers import DomainClassifier

# Create pipeline
pipeline = Pipeline(name="domain_classification")

# Load dataset
reader = JsonlReader(
    file_paths="books_dataset/",
    fields=["text", "id"]
)
pipeline.add_stage(reader)

# Apply the classifier, filtering for specific domains
domain_classifier = DomainClassifier(filter_by=["Games", "Sports"])
pipeline.add_stage(domain_classifier)

# Save the results
writer = JsonlWriter(path="games_and_sports/")
pipeline.add_stage(writer)

# Execute pipeline
results = pipeline.run()  # Uses XennaExecutor by default
```

### Multilingual Domain Classifier

Functionally similar to the Domain Classifier, but supports 52 languages.

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.classifiers import MultilingualDomainClassifier

pipeline = Pipeline(name="multilingual_domain_classification")
pipeline.add_stage(JsonlReader(file_paths="multilingual_dataset/", fields=["text", "id"]))
pipeline.add_stage(MultilingualDomainClassifier(filter_by=["Games", "Sports"]))
pipeline.add_stage(JsonlWriter(path="classified_output/"))

results = pipeline.run()  # Uses XennaExecutor by default
```

### Quality Classifier

The Quality Classifier assesses document quality using the NVIDIA Quality Classifier DeBERTa model.

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.classifiers import QualityClassifier

pipeline = Pipeline(name="quality_classification")
pipeline.add_stage(JsonlReader(file_paths="web_documents/", fields=["text", "id"]))
pipeline.add_stage(QualityClassifier())
pipeline.add_stage(JsonlWriter(path="quality_classified/"))

results = pipeline.run()  # Uses XennaExecutor by default
```

:::{note}
The exact label categories returned by the Quality Classifier depend on the model configuration. Check the prediction column in your results to see the available labels for filtering with the `filter_by` parameter.
:::

### AEGIS Safety Classifier

The AEGIS classifier detects unsafe content across 13 critical risk categories. It requires a HuggingFace token for access to Llama Guard.

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.classifiers import AegisClassifier

# Create pipeline
pipeline = Pipeline(name="aegis_classification")

# Load dataset
reader = JsonlReader(
    file_paths="content/",
    fields=["text", "id"]
)
pipeline.add_stage(reader)

# Apply the AEGIS classifier
token = "hf_1234"  # Your HuggingFace user access token
safety_classifier = AegisClassifier(
    aegis_variant="nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0",
    hf_token=token,
    filter_by=["safe", "O13"]  # Keep only safe content and "needs caution" category
)
pipeline.add_stage(safety_classifier)

# Save the results
writer = JsonlWriter(path="safe_content/")
pipeline.add_stage(writer)

# Execute pipeline
results = pipeline.run()  # Uses XennaExecutor by default
```

The classifier adds a column with labels: "safe," "O1" through "O13" (each representing specific safety risks), or "unknown." For raw LLM output, use:

```python
safety_classifier = AegisClassifier(
    aegis_variant="nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0",
    hf_token=token,
    keep_raw_output=True,
    raw_output_field="raw_predictions"
)
```

### Instruction Data Guard

Detects LLM poisoning attacks in instruction-response datasets. Requires HuggingFace token access.

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.classifiers import InstructionDataGuardClassifier

# Create pipeline
pipeline = Pipeline(name="instruction_data_guard")

# Load dataset
# For instruction-response data: "Instruction: {instruction}. Input: {input_}. Response: {response}."
reader = JsonlReader(
    file_paths="instruction_data/",
    fields=["text", "id"]
)
pipeline.add_stage(reader)

# Apply the classifier
token = "hf_1234"  # Your HuggingFace user access token
classifier = InstructionDataGuardClassifier(hf_token=token)
pipeline.add_stage(classifier)

# Save the results
writer = JsonlWriter(path="guard_classified/")
pipeline.add_stage(writer)

# Execute pipeline
results = pipeline.run()  # Uses XennaExecutor by default
```

The output includes two columns: a float score `instruction_data_guard_poisoning_score` and a Boolean `is_poisoned`.

### FineWeb Educational Content Classifier

Scores documents on educational value from 0–5. This helps prioritize content for knowledge-intensive tasks.

#### Score Ranges and Meanings

| Score | Label | Description | Example Content |
|-------|-------|-------------|-----------------|
| 0-1 | Very Low | No educational value | Spam, advertisements, broken content |
| 2 | Low | Minimal educational content | Simple lists, basic product descriptions |
| 3 | Moderate | Some educational value | News articles, basic how-to guides |
| 4 | High | Good educational content | Detailed tutorials, academic discussions |
| 5 | Very High | Excellent educational material | Comprehensive guides, scholarly articles |

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.classifiers import FineWebEduClassifier

# Create pipeline
pipeline = Pipeline(name="fineweb_edu_classification")

# Load dataset
reader = JsonlReader(
    file_paths="web_documents/*.jsonl",
    fields=["text", "id"]
)
pipeline.add_stage(reader)

# Apply the FineWeb Edu classifier
edu_classifier = FineWebEduClassifier(
    model_inference_batch_size=256,
    float_score_field="fineweb-edu-score-float",  # Raw float scores
    int_score_field="fineweb-edu-score-int",      # Rounded integer scores
    label_field="fineweb-edu-score-label"         # Quality labels
)
pipeline.add_stage(edu_classifier)

# Save the results
writer = JsonlWriter(path="edu_classified/")
pipeline.add_stage(writer)

# Execute pipeline
results = pipeline.run()  # Uses XennaExecutor by default
```

### FineWeb Mixtral and Nemotron Edu Classifiers

Similar to the FineWeb Edu Classifier but trained with different annotation sources:

- **FineWebMixtralEduClassifier**: Uses annotations from Mixtral 8x22B-Instruct
- **FineWebNemotronEduClassifier**: Uses annotations from Nemotron-4-340B-Instruct

Both provide a quality label column marking scores above 2.5 as "high_quality":

#### Quality Label Mapping

| Score Range | Quality Label | Description |
|-------------|---------------|-------------|
| 0.0 - 2.5 | `low_quality` | Below average educational value |
| 2.5 - 5.0 | `high_quality` | Above average educational value |

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.classifiers import FineWebMixtralEduClassifier  # or FineWebNemotronEduClassifier

# Create pipeline
pipeline = Pipeline(name="fineweb_mixtral_edu_classification")

# Load dataset
reader = JsonlReader(
    file_paths="web_documents/*.jsonl",
    fields=["text", "id"]
)
pipeline.add_stage(reader)

# Apply the FineWeb Mixtral Edu classifier
classifier = FineWebMixtralEduClassifier(
    float_score_field="fineweb-mixtral-edu-score-float",  # Raw float scores
    int_score_field="fineweb-mixtral-edu-score-int",      # Rounded integer scores
    label_field="fineweb-mixtral-edu-score-label"          # "high_quality" or "low_quality"
)
pipeline.add_stage(classifier)

# Save the results
writer = JsonlWriter(path="mixtral_edu_classified/")
pipeline.add_stage(writer)

# Execute pipeline
results = pipeline.run()  # Uses XennaExecutor by default
```

### Content Type Classifier

Categorizes documents into 11 distinct speech types.

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.classifiers import ContentTypeClassifier

# Create pipeline
pipeline = Pipeline(name="content_type_classification")

# Load dataset
reader = JsonlReader(
    file_paths="content/",
    fields=["text", "id"]
)
pipeline.add_stage(reader)

# Apply the Content Type classifier
classifier = ContentTypeClassifier(filter_by=["Blogs", "News"])
pipeline.add_stage(classifier)

# Save the results
writer = JsonlWriter(path="content_type_classified/")
pipeline.add_stage(writer)

# Execute pipeline
results = pipeline.run()  # Uses XennaExecutor by default
```

### Prompt Task and Complexity Classifier

Classifies prompts by task type and complexity dimensions.

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.classifiers import PromptTaskComplexityClassifier

# Create pipeline
pipeline = Pipeline(name="prompt_task_complexity_classification")

# Load dataset
reader = JsonlReader(
    file_paths="prompts/",
    fields=["text", "id"]
)
pipeline.add_stage(reader)

# Apply the Prompt Task Complexity classifier
classifier = PromptTaskComplexityClassifier()
pipeline.add_stage(classifier)

# Save the results
writer = JsonlWriter(path="prompt_complexity_classified/")
pipeline.add_stage(writer)

# Execute pipeline
results = pipeline.run()  # Uses XennaExecutor by default
```

## Performance Optimization

NVIDIA NeMo Curator's distributed classifiers are optimized for high-throughput processing through several key features:

### Intelligent Batching and Sequence Handling

The classifiers optimize throughput through:

- **Length-based sorting**: Input sequences are sorted by length when `sort_by_length=True` (default)
- **Efficient batching**: Similar-length sequences are grouped together to minimize padding overhead
- **GPU memory optimization**: Batches are sized to maximize GPU utilization based on available memory
