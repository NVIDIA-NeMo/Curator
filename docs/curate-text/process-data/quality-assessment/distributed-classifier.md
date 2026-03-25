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
3. **Batched Inference**: Optimizing throughput with intelligent batching
4. **Consistent API**: Providing a unified interface through the `DistributedDataClassifier` base class

The `DistributedDataClassifier` is designed to run on GPU clusters with minimal code changes regardless of which specific classifier you're using. All classifiers support filtering based on classification results and storing prediction scores as metadata.

:::{note}
Distributed classification requires GPU acceleration and is not supported for CPU-only processing. As long as GPU resources are available and NeMo Curator is correctly installed, GPU acceleration is handled automatically.
:::

```{tip}
**Running the tutorial notebooks**: The classification tutorial notebooks require the `text_cuda12` or `all` installation extra to include all relevant dependencies. If you encounter `ModuleNotFoundError`, reinstall with the appropriate extra:

    uv pip install "nemo-curator[text_cuda12]"

When using classifiers that download from Hugging Face (such as Aegis and InstructionDataGuard), set your `HF_TOKEN` environment variable to avoid rate limiting:

    export HF_TOKEN="your_token_here"
```

---

## Usage

NVIDIA NeMo Curator provides a base class `DistributedDataClassifier` that can be extended to fit your specific model. The only requirement is that the model can fit on a single GPU. This module operates on the GPU and works within the pipeline framework using DocumentBatch processing.

### Classifier Comparison

| Classifier | Purpose | Model Location | Key Parameters | Requirements |
|---|---|---|---|---|
| DomainClassifier | Assigns one of 26 domain labels (such as "Sports," "Science," "News") to English text | [nvidia/domain-classifier](https://huggingface.co/nvidia/domain-classifier) | `filter_by`, `text_field` | None |
| MultilingualDomainClassifier | Assigns domain labels to text in 52 languages; same labels as DomainClassifier | [nvidia/multilingual-domain-classifier](https://huggingface.co/nvidia/multilingual-domain-classifier) | `filter_by`, `text_field` | None |
| QualityClassifier | Rates document quality as "Low," "Medium," or "High" using a DeBERTa model | [nvidia/quality-classifier-deberta](https://huggingface.co/nvidia/quality-classifier-deberta) | `filter_by`, `text_field` | None |
| AegisClassifier | Detects unsafe content across 13 risk categories (violence, hate speech, and others) using LlamaGuard | [nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0](https://huggingface.co/nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0) | `aegis_variant`, `filter_by` | HuggingFace token |
| InstructionDataGuardClassifier | Identifies LLM poisoning attacks in instruction-response pairs | [nvidia/instruction-data-guard](https://huggingface.co/nvidia/instruction-data-guard) | `text_field`, `label_field` | HuggingFace token |
| FineWebEduClassifier | Scores educational value from 0 to 5 (0=spam, 5=scholarly) for training data selection | [HuggingFaceFW/fineweb-edu-classifier](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier) | `label_field`, `int_field` | None |
| FineWebMixtralEduClassifier | Scores educational value from 0 to 5 using Mixtral 8x22B annotation data | [nvidia/nemocurator-fineweb-mixtral-edu-classifier](https://huggingface.co/nvidia/nemocurator-fineweb-mixtral-edu-classifier) | `label_field`, `int_field`, `model_inference_batch_size=1024` | None |
| FineWebNemotronEduClassifier | Scores educational value from 0 to 5 using Nemotron-4-340B annotation data | [nvidia/nemocurator-fineweb-nemotron-4-edu-classifier](https://huggingface.co/nvidia/nemocurator-fineweb-nemotron-4-edu-classifier) | `label_field`, `int_field`, `model_inference_batch_size=1024` | None |
| ContentTypeClassifier | Categorizes text into 11 speech types (such as "Blogs," "News," "Academic") | [nvidia/content-type-classifier-deberta](https://huggingface.co/nvidia/content-type-classifier-deberta) | `filter_by`, `text_field` | None |
| PromptTaskComplexityClassifier | Labels prompts by task type (such as QA and summarization) and complexity dimensions | [nvidia/prompt-task-and-complexity-classifier](https://huggingface.co/nvidia/prompt-task-and-complexity-classifier) | `text_field` | None |

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

## Custom Model Integration

You can integrate your own classification models by extending `DistributedDataClassifier`. Refer to the [Text Classifiers README](https://github.com/NVIDIA-NeMo/Curator/tree/main/nemo_curator/stages/text/classifiers#text-classifiers) for implementation details and examples.

## Performance Optimization

NVIDIA NeMo Curator's distributed classifiers are optimized for high-throughput processing through several key features:

### CPU-based tokenization and GPU-based model inference

Each classifier is broken down under the hood into a tokenizer stage and a model inference stage. Tokenization is run on the CPU while model inference is run on the GPU. For example, this means that behind the scenes, the `DomainClassifier` stage is actually being broken down into 2 stages (some parameters and details omitted to avoid complexity):

```python
class TokenizerStage:
    self.resources = Resources(cpus=1)
    self.model_identifier = "nvidia/domain-classifier"
    self.text_field = "text"
    self.padding_side = "right"
    ...
class ModelStage:
    self.resources = Resources(cpus=1, gpus=1)
    self.model_identifier = "nvidia/domain-classifier"
    self.model_inference_batch_size = 256
    ...
```

Pipelines take care of resource allocation and autoscaling to achieve enhanced performance and minimize GPU idleness. This means that we are able to achieve speedups by ensuring that model inference is run in parallel across all available GPUs, while other stages such as I/O, tokenization, and filtering are run across all available CPUs. This is possible because Curator pipelines are composable, which allows each stage in a pipeline to run independently and with its own specified hardware resources.

### Intelligent Batching and Sequence Handling

The classifiers optimize throughput through:

- **Length-based sorting**: Input sequences are sorted by length when `sort_by_length=True` (default)
- **Efficient batching**: Similar-length sequences are grouped together to minimize padding overhead
- **GPU memory optimization**: Batches are sized to maximize GPU utilization based on available memory

### Avoid Unnecessary Re-Tokenization

Several of the text classifiers use the same tokenizer before running the model forward pass. To avoid unnecessary re-tokenization, the `keep_tokens` and `use_existing_tokens` parameters can be used.

**Important: Not every text classifier uses the same tokenizer, so it is important to confirm that classifiers' tokenizers are compatible with each other. Curator will not verify this for you.**

The `ContentTypeClassifier`, `QualityClassifier`, `DomainClassifier`, and `PromptTaskComplexityClassifier` all use a DeBERTa tokenizer, which means that we only need to tokenize once. To avoid unnecessary re-tokenization, you can do:

```python
# Since this is the first classifier in the pipeline, there are no existing tokens to use,
# but we can make sure to keep the computed tokens for the next classifier
content_type_classifier = ContentTypeClassifier(use_existing_tokens=False, keep_tokens=True, ...)
pipeline.add_stage(content_type_classifier)

# Use tokens from the previous classifier and keep tokens for the next classifier
quality_classifier = QualityClassifier(use_existing_tokens=True, keep_tokens=True, ...)
pipeline.add_stage(quality_classifier)

# Use tokens from the previous classifier and keep tokens for the next classifier
domain_classifier = DomainClassifier(use_existing_tokens=True, keep_tokens=True, ...)
pipeline.add_stage(domain_classifier)

# Use tokens from the previous classifier
# Since this is the final classifier in the pipeline, we drop the computed tokens
prompt_task_complexity_classifier = PromptTaskComplexityClassifier(use_existing_tokens=True, keep_tokens=False, ...)
pipeline.add_stage(prompt_task_complexity_classifier)
```

In addition to the above example, the `FineWebEduClassifier`, `FineWebMixtralEduClassifier`, and `FineWebNemotronEduClassifier` are all compatible with each other:

```python
fineweb_classifier = FineWebEduClassifier(use_existing_tokens=False, keep_tokens=True, ...)
pipeline.add_stage(fineweb_classifier)

fineweb_mixtral_classifier = FineWebMixtralEduClassifier(use_existing_tokens=True, keep_tokens=True, ...)
pipeline.add_stage(fineweb_mixtral_classifier)

fineweb_nemotron_classifier = FineWebNemotronEduClassifier(use_existing_tokens=True, keep_tokens=False, ...)
pipeline.add_stage(fineweb_nemotron_classifier)
```

The `AegisClassifier` variants ([nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0](https://huggingface.co/nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0) and [nvidia/Aegis-AI-Content-Safety-LlamaGuard-Permissive-1.0](https://huggingface.co/nvidia/Aegis-AI-Content-Safety-LlamaGuard-Permissive-1.0)) are compatible with each other as well. This example is a bit more complex because it also involves keeping the formatted Aegis prompt field. See the `AegisClassifier` implementation for more details.

```python
aegis_defensive_classifier = AegisClassifier(
    aegis_variant="nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0",
    label_field="aegis_defensive_pred",
    use_existing_tokens=False,
    keep_tokens=True,
    keep_aegis_prompt_field=True,
    ...
)
pipeline.add_stage(aegis_defensive_classifier)

aegis_permissive_classifier = AegisClassifier(
    aegis_variant="nvidia/Aegis-AI-Content-Safety-LlamaGuard-Permissive-1.0",
    label_field="aegis_permissive_pred",
    use_existing_tokens=True,
    aegis_prompt_field="_curator_hidden_text",  # created by aegis_defensive_classifier
    keep_tokens=False,
    keep_aegis_prompt_field=False,
    ...
)
pipeline.add_stage(aegis_permissive_classifier)
```
