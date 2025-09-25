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

<!-- ## Hardware Requirements

Distributed data classification requires GPU acceleration for optimal performance:

### Minimum Requirements

- **GPU Memory**: 8GB+ VRAM per GPU (varies by model)
- **System Memory**: 32GB+ RAM recommended
- **Storage**: Fast SSD storage for large datasets

### Recommended Configuration

- **Multi-GPU**: 2+ GPUs for improved throughput
- **GPU Memory**: 16GB+ VRAM per GPU for larger models
- **Network**: High-bandwidth interconnect for multi-node setups

### Model-Specific Requirements

| Classifier | GPU Memory | Notes |
|------------|------------|-------|
| QualityClassifier | 4GB+ | DeBERTa-based, moderate memory usage |
| DomainClassifier | 4GB+ | DeBERTa-based, moderate memory usage |
| AegisClassifier | 8GB+ | Llama Guard-based, higher memory usage |
| FineWebEduClassifier | 6GB+ | Transformer-based, variable batch sizes |

 -->

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
| InstructionDataGuardClassifier | Detect poisoning attacks | [nvidia/instruction-data-guard](https://huggingface.co/nvidia/instruction-data-guard) | `text_field`, `pred_column` | HuggingFace token |
| FineWebEduClassifier | Score educational value | [HuggingFaceFW/fineweb-edu-classifier](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier) | `pred_column`, `int_column` | None |
| FineWebMixtralEduClassifier | Score educational value (Mixtral annotations) | [nvidia/nemocurator-fineweb-mixtral-edu-classifier](https://huggingface.co/nvidia/nemocurator-fineweb-mixtral-edu-classifier) | `pred_column`, `int_column` | None |
| FineWebNemotronEduClassifier | Score educational value (Nemotron annotations) | [nvidia/nemocurator-fineweb-nemotron-4-edu-classifier](https://huggingface.co/nvidia/nemocurator-fineweb-nemotron-4-edu-classifier) | `pred_column`, `int_column` | None |
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
    file_paths="books_dataset/*.jsonl",
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
pipeline.add_stage(JsonlReader(file_paths="multilingual_dataset/*.jsonl", fields=["text", "id"]))
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
pipeline.add_stage(JsonlReader(file_paths="web_documents/*.jsonl", fields=["text", "id"]))
pipeline.add_stage(QualityClassifier())
pipeline.add_stage(JsonlWriter(path="quality_classified/"))

results = pipeline.run()  # Uses XennaExecutor by default
```

:::{note}
The exact label categories returned by the Quality Classifier depend on the model configuration. Check the prediction column in your results to see the available labels for filtering with the `filter_by` parameter.
:::

### AEGIS Safety Model

The AEGIS classifier detects unsafe content across 13 critical risk categories. It requires a HuggingFace token for access to Llama Guard.

```python
from nemo_curator.classifiers import AegisClassifier

input_dataset = DocumentDataset.read_json("content/*.jsonl", backend="cudf")

token = "hf_1234"  # Your HuggingFace user access token
safety_classifier = AegisClassifier(
    aegis_variant="nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0",
    token=token,
    filter_by=["safe", "O13"]  # Keep only safe content and "needs caution" category
)
result_dataset = safety_classifier(dataset=input_dataset)
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
from nemo_curator.classifiers import InstructionDataGuardClassifier

# For instruction-response data: "Instruction: {instruction}. Input: {input_}. Response: {response}."
input_dataset = DocumentDataset.read_json("instruction_data/*.jsonl", backend="cudf")

token = "hf_1234"  # Your HuggingFace user access token
classifier = InstructionDataGuardClassifier(token=token)
result_dataset = classifier(dataset=input_dataset)
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
from nemo_curator.classifiers import FineWebEduClassifier

input_dataset = DocumentDataset.read_json("web_documents/*.jsonl", backend="cudf")
edu_classifier = FineWebEduClassifier(
    batch_size=256,
    pred_column="fineweb-edu-score",     # Raw float scores
    int_column="fineweb-edu-score-int"   # Rounded integer scores
)
result_dataset = edu_classifier(dataset=input_dataset)

# Extract highly educational content (scores 4-5)
high_edu_dataset = result_dataset[result_dataset["fineweb-edu-score-int"] >= 4]
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
from nemo_curator.classifiers import FineWebMixtralEduClassifier  # or FineWebNemotronEduClassifier

classifier = FineWebMixtralEduClassifier(
    pred_column="score",                 # Raw float scores
    int_column="score-int",              # Rounded integer scores
    quality_label_column="quality-label" # "high_quality" or "low_quality"
)
result_dataset = classifier(dataset=input_dataset)
```

### Content Type Classifier

Categorizes documents into 11 distinct speech types.

```python
from nemo_curator.classifiers import ContentTypeClassifier

input_dataset = DocumentDataset.read_json("content/*.jsonl", backend="cudf")
classifier = ContentTypeClassifier(filter_by=["Blogs", "News"])
result_dataset = classifier(dataset=input_dataset)
```

### Prompt Task and Complexity Classifier

Classifies prompts by task type and complexity dimensions.

```python
from nemo_curator.classifiers import PromptTaskComplexityClassifier

input_dataset = DocumentDataset.read_json("prompts/*.jsonl", backend="cudf")
classifier = PromptTaskComplexityClassifier()
result_dataset = classifier(dataset=input_dataset)
```

## Scaling with Different Executors

All NVIDIA NeMo Curator classifiers support different execution backends for enhanced scalability and performance. By default, pipelines use the `XennaExecutor`, but you can choose different backends based on your computational requirements.

```python
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.classifiers import QualityClassifier

# Create pipeline with classifier
pipeline = Pipeline(name="classifier_pipeline")
pipeline.add_stage(read_stage)
pipeline.add_stage(QualityClassifier())
pipeline.add_stage(write_stage)

# Run with default Xenna executor (recommended)
executor = XennaExecutor(config={"execution_mode": "streaming"})
results = pipeline.run(executor)
```

For large-scale distributed classification tasks, consider using Ray-based executors or other backends. Refer to the {doc}`Pipeline Execution Backends </reference/infrastructure/execution-backends>` reference for detailed information about available executors, their configurations, and when to use each backend type.

## CrossFit Integration

CrossFit is an open-source library by RAPIDS AI for fast offline inference scaled to multi-node multi-GPU environments. It accelerates NVIDIA NeMo Curator's classifiers with:

- PyTorch integration for model inference
- Efficient I/O and tokenization with cuDF
- Smart batching/chunking for optimized processing
- 1.4x-4x performance improvement over Dask + PyTorch baselines

### Sorted Sequence Data Loader

The key feature of CrossFit used in NVIDIA NeMo Curator is the sorted sequence data loader, which optimizes throughput by:

- Sorting input sequences by length
- Grouping similar-length sequences into batches
- Efficiently allocating batches to GPU memory based on estimated memory footprints

See the [rapidsai/crossfit](https://github.com/rapidsai/crossfit) repository for more information.