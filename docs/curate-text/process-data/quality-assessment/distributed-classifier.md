---
description: "Perform task-based data classification using composable processing stages for domain, quality, safety, and content assessment"
categories: ["how-to-guides"]
tags: ["task-based-classification", "ray", "pipeline", "filters", "extensible", "scalable"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "text-only"
---

(text-process-data-filter-task-classifier)=

# Task-Based Data Classification

NVIDIA NeMo Curator provides a flexible, task-based classification system using Ray for distributed processing. This enables categorization and filtering of text documents through modular processing stages that you can combine into custom pipelines for domain, quality, safety, and content assessment.

## How It Works

The task-based classification system works by:

1. **Pipeline Architecture**: Composing processing stages into workflows that operate on DocumentBatch tasks
2. **Extensible Filters**: Using DocumentFilter base classes for custom scoring and filtering logic
3. **Ray Distribution**: Leveraging Ray for distributed execution across nodes and GPU devices
4. **Task Flow**: Processing data in batches that flow through interconnected stages

The system provides three core classification stages that you can combine and customize for different use cases, providing fine-grained control over classification workflows.

---

## Core Classification Stages

The task-based classification system provides three main stages that you can compose into pipelines:

### Stage Types

| Stage | Purpose | Input | Output | Use Case |
|---|---|---|---|---|
| `Score` | Add classification scores to documents | DocumentBatch | DocumentBatch with score columns | Computing metrics without filtering |
| `Filter` | Filter documents based on existing metadata | DocumentBatch | Filtered DocumentBatch | Applying filters to pre-scored data |
| `ScoreFilter` | Score and filter documents in one step | DocumentBatch | Scored and filtered DocumentBatch | Combined scoring and filtering |

### Available Document Filters

| Filter | Purpose | Module | Key Parameters |
|---|---|---|---|
| `FastTextQualityFilter` | Assess document quality using FastText | `ray_curator.stages.filters.fasttext_filter` | `alpha`, `label` |
| `FastTextLangId` | Language identification | `ray_curator.stages.filters.fasttext_filter` | `min_langid_score` |
| `DocumentFilter` | Base class for custom filters | `ray_curator.stages.filters.doc_filter` | Abstract base |
| Heuristic filters | Text quality assessment | `ray_curator.stages.filters.heuristic_filter` | Various parameters |

## Basic Usage

### Quality Classification with FastText

Assess document quality using FastText models integrated into the pipeline system.

```python
from ray_curator.pipeline import Pipeline
from ray_curator.stages.io.reader import JsonlReader
from ray_curator.stages.modules import ScoreFilter
from ray_curator.stages.filters import FastTextQualityFilter
from ray_curator.backends.xenna import XennaExecutor

# Create classification pipeline
pipeline = Pipeline([
    JsonlReader("books_dataset/*.jsonl"),
    ScoreFilter(
        filter_obj=FastTextQualityFilter(alpha=3.0, label="__label__hq"),
        text_field="text",
        score_field="quality_score"
    )
])

# Execute pipeline
executor = XennaExecutor()
results = pipeline.run(executor)
```

### Language Identification

Identify document language using FastText language identification.

```python
from ray_curator.stages.filters import FastTextLangId

pipeline = Pipeline([
    JsonlReader("multilingual_dataset/*.jsonl"),
    ScoreFilter(
        filter_obj=FastTextLangId(min_langid_score=0.3),
        text_field="text",
        score_field="language_info"
    )
])
```

### Separate Scoring and Filtering

Use separate stages for more complex workflows.

```python
from ray_curator.stages.modules import Score, Filter

pipeline = Pipeline([
    JsonlReader("web_documents/*.jsonl"),
    # First, add quality scores
    Score(
        score_fn=FastTextQualityFilter(alpha=3.0),
        score_field="quality_score",
        text_field="text"
    ),
    # Then filter based on scores
    Filter(
        filter_fn=lambda score: score > 0.7,
        filter_field="quality_score"
    )
])
```

## Creating Custom Filters

Build custom classification logic by extending the DocumentFilter base class.

### Custom Domain Filter Example

```python
from ray_curator.stages.filters import DocumentFilter
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class CustomDomainFilter(DocumentFilter):
    def __init__(self, target_domains: list[str], model_name: str = "nvidia/domain-classifier"):
        super().__init__()
        self.target_domains = target_domains
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
    
    def setup_on_node(self, node_info=None, worker_metadata=None):
        """Download model on each node"""
        # Model download logic here
        pass
    
    def setup(self, worker_metadata=None):
        """Load model on each worker"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.eval()
    
    def score_document(self, text: str) -> str:
        """Classify document domain"""
        inputs = self.tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = predictions.argmax().item()
            return self.model.config.id2label[predicted_class]
    
    def keep_document(self, domain: str) -> bool:
        """Filter based on target domains"""
        return domain in self.target_domains

# Use in pipeline
pipeline = Pipeline([
    JsonlReader("content/*.jsonl"),
    ScoreFilter(
        filter_obj=CustomDomainFilter(target_domains=["Science", "Technology"]),
        text_field="text",
        score_field="domain"
    )
])
```

### Heuristic Filter Example

```python
from ray_curator.stages.filters.heuristic_filter import WordCountFilter

# Use built-in heuristic filters
pipeline = Pipeline([
    JsonlReader("documents/*.jsonl"),
    ScoreFilter(
        filter_obj=WordCountFilter(min_words=50, max_words=1000),
        text_field="text",
        score_field="word_count"
    )
])
```

## Advanced Pipeline Patterns

### Multi-Stage Classification

Combine several classification stages for comprehensive document processing.

```python
from ray_curator.stages.io.writer import JsonlWriter

# Complex classification pipeline
pipeline = Pipeline([
    JsonlReader("raw_documents/*.jsonl"),
    
    # Language identification
    Score(
        score_fn=FastTextLangId(),
        score_field="language",
        text_field="text"
    ),
    
    # Quality assessment
    Score(
        score_fn=FastTextQualityFilter(alpha=3.0),
        score_field="quality_score",
        text_field="text"
    ),
    
    # Filter based on multiple criteria
    Filter(
        filter_fn=lambda row: row["language"] == "en" and row["quality_score"] > 0.7,
        filter_field=["language", "quality_score"]
    ),
    
    # Save results
    JsonlWriter("filtered_documents/")
])
```

### Batch Processing Configuration

Configure processing stages for optimal performance.

```python
# Configure resources for GPU-intensive tasks
from ray_curator.stages.resources import Resources

custom_filter = ScoreFilter(
    filter_obj=CustomDomainFilter(target_domains=["Science"]),
    text_field="text",
    score_field="domain"
)

# Set GPU requirements
custom_filter._resources = Resources(gpu_memory_gb=8.0, cpus=2.0)

pipeline = Pipeline([
    JsonlReader("large_dataset/*.jsonl", files_per_partition=10),
    custom_filter
])
```

### Error Handling and Monitoring

The task-based system provides built-in fault tolerance and performance monitoring through the Ray backend. Each DocumentBatch task tracks processing statistics and supports independent retry if processing fails.
