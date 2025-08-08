---
description: "Create and combine custom filters using NeMo Curator's flexible framework for specialized data quality requirements"
categories: ["how-to-guides"]
tags: ["custom-filters", "extensible", "flexible", "advanced", "framework", "task-based"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "advanced"
content_type: "how-to"
modality: "text-only"
---

(text-process-data-filter-custom)=

# Custom Filters

NVIDIA NeMo Curator provides a flexible framework for implementing and combining custom filters to meet your specific data quality requirements. Whether you need to filter documents based on domain-specific criteria or optimize your pipeline's performance, custom filters give you complete control over the filtering process.

## Creating Custom Filters

Custom filters in NeMo Curator inherit from the `DocumentFilter` abstract base class, which requires implementing two key methods:

1. `score_document`: Analyzes a document and assigns it a quality score
2. `keep_document`: Determines whether to keep a document based on its score

Here's a simple example of a custom filter:

```python
from ray_curator.stages.filters.doc_filter import DocumentFilter

class CustomWordFilter(DocumentFilter):
    def __init__(self, target_words, min_occurrences=1):
        super().__init__()  # Call the parent constructor
        self._target_words = set(target_words)
        self._min_occurrences = min_occurrences
        self._name = 'custom_word_filter'
        
    def score_document(self, text: str):
        """Count occurrences of target words in the document."""
        words = text.lower().split()
        count = sum(1 for word in words if word in self._target_words)
        return count
        
    def keep_document(self, score: int):
        """Keep documents with enough target words."""
        return score >= self._min_occurrences
```

Custom filters automatically work with both pandas and PyArrow data formats used in the task-based processing architecture.

## Using Custom Filters

Once you've defined your custom filter, you can use it with NeMo Curator's task-based processing framework:

```python
from ray_curator.pipeline.pipeline import Pipeline
from ray_curator.stages.modules.score_filter import ScoreFilter
from ray_curator.stages.io.reader.jsonl import JsonlReader
from ray_curator.stages.io.writer.jsonl import JsonlWriter

# Create and configure your custom filter
my_filter = CustomWordFilter(
    target_words=["machine", "learning", "ai", "deep", "neural"],
    min_occurrences=3
)

# Create a pipeline
pipeline = Pipeline(
    name="custom_filter_pipeline",
    description="Apply custom word filter to documents"
)

# Add stages to the pipeline
pipeline.add_stage(JsonlReader(file_paths="input_data/*.jsonl"))
pipeline.add_stage(ScoreFilter(
    filter_obj=my_filter,
    text_field="text",
    score_field="target_word_count"
))
pipeline.add_stage(JsonlWriter(output_dir="filtered_output/"))

# Build and run the pipeline
pipeline.build()

# Execute with your chosen backend (Ray Data or Xenna)
from ray_curator.backends.experimental.ray_data.executor import RayDataExecutor
executor = RayDataExecutor()
pipeline.run(executor)
```

## Optimizing Performance

The framework automatically optimizes performance through its task-based architecture. DocumentBatch objects contain documents that process together, providing natural batching without requiring special decorators.

For computationally intensive filters, you can optimize performance by:

1. **Using vectorized operations**: Use pandas or NumPy vectorized operations when possible
2. **Configuring batch sizes**: Adjust the `files_per_partition` parameter in readers
3. **Resource allocation**: Specify appropriate CPU/GPU resources for your stages

```python
from ray_curator.stages.filters.doc_filter import DocumentFilter
import pandas as pd

class OptimizedCustomFilter(DocumentFilter):
    def __init__(self, threshold=0.5):
        super().__init__()
        self._threshold = threshold
        self._name = 'optimized_custom_filter'
    
    def score_document(self, text: str):
        # Single document scoring logic for individual processing
        return len(text.split()) / max(text.count('.'), 1)  # words per sentence
    
    def keep_document(self, score: float):
        """Filter logic applied to individual document scores."""
        return score >= self._threshold
```

Processing stages automatically handle batching of DocumentBatch objects containing documents.

## Filter Composition Methods

NeMo Curator makes it easy to combine filters using pipeline composition:

### Sequential Pipeline Composition

Use the `Pipeline` class to apply a series of filters in order:

```python
from ray_curator.pipeline.pipeline import Pipeline
from ray_curator.stages.modules.score_filter import ScoreFilter
from ray_curator.stages.filters.heuristic_filter import WordCountFilter, NonAlphaNumericFilter, UrlsFilter
from ray_curator.stages.io.reader.jsonl import JsonlReader
from ray_curator.stages.io.writer.jsonl import JsonlWriter

# Create a pipeline with multiple filters
filter_pipeline = Pipeline(
    name="sequential_filter_pipeline",
    description="Apply multiple filters in sequence"
)

# Add reader
filter_pipeline.add_stage(JsonlReader(file_paths="input_data/*.jsonl"))

# Add filtering stages
filter_pipeline.add_stage(ScoreFilter(
    filter_obj=WordCountFilter(min_words=100),
    text_field="text"
))
filter_pipeline.add_stage(ScoreFilter(
    filter_obj=NonAlphaNumericFilter(max_non_alpha_numeric_to_text_ratio=0.3),
    text_field="text"
))
filter_pipeline.add_stage(ScoreFilter(
    filter_obj=UrlsFilter(max_url_to_text_ratio=0.2),
    text_field="text"
))

# Add writer
filter_pipeline.add_stage(JsonlWriter(output_dir="high_quality_output/"))

# Execute the pipeline
filter_pipeline.build()
executor = RayDataExecutor()
filter_pipeline.run(executor)
```

### Custom Composite Stages

For complex filter combinations, you can create custom composite stages that provide voting or parallel filtering logic:

```python
from dataclasses import dataclass
from ray_curator.stages.base import CompositeStage, ProcessingStage
from ray_curator.stages.modules.score_filter import ScoreFilter
from ray_curator.tasks import DocumentBatch

@dataclass
class VotingFilterStage(CompositeStage[DocumentBatch, DocumentBatch]):
    """Custom composite stage that implements voting filter logic."""
    
    filters: list[DocumentFilter]
    min_passing: int = 2
    text_field: str = "text"
    
    def decompose(self) -> list[ProcessingStage]:
        """Decompose into individual scoring stages."""
        stages = []
        
        # Add scoring stages for each filter
        for i, filter_obj in enumerate(self.filters):
            stages.append(ScoreFilter(
                filter_obj=filter_obj,
                text_field=self.text_field,
                score_field=f"filter_{i}_score"
            ))
        
        # Add final voting stage
        stages.append(VotingDecisionStage(
            filter_count=len(self.filters),
            min_passing=self.min_passing
        ))
        
        return stages
    
    def get_description(self) -> str:
        return f"Voting filter requiring {self.min_passing} of {len(self.filters)} filters to pass"
```

This approach leverages the modular nature of the pipeline architecture to provide complex filtering logic.

## Scoring Without Filtering

Sometimes you want to add quality scores to your documents without actually filtering them:

```python
from ray_curator.pipeline.pipeline import Pipeline
from ray_curator.stages.modules.score_filter import Score
from ray_curator.stages.filters.heuristic_filter import WordCountFilter, NonAlphaNumericFilter
from ray_curator.stages.io.reader.jsonl import JsonlReader
from ray_curator.stages.io.writer.jsonl import JsonlWriter

# Create a pipeline for adding scores
scoring_pipeline = Pipeline(
    name="scoring_pipeline",
    description="Add quality scores without filtering"
)

# Add reader
scoring_pipeline.add_stage(JsonlReader(file_paths="input_data/*.jsonl"))

# Add scoring stages (no filtering)
scoring_pipeline.add_stage(Score(
    score_fn=WordCountFilter(),  # Uses the filter's score_document method
    text_field="text",
    score_field="word_count"
))

scoring_pipeline.add_stage(Score(
    score_fn=NonAlphaNumericFilter(),
    text_field="text", 
    score_field="symbol_ratio"
))

# Add writer to save scored documents
scoring_pipeline.add_stage(JsonlWriter(output_dir="scored_output/"))

# Execute the pipeline
scoring_pipeline.build()
executor = RayDataExecutor()
scoring_pipeline.run(executor)
```

## Filtering on Existing Metadata

If your dataset already contains quality metrics, you can filter directly on those:

```python
from ray_curator.pipeline.pipeline import Pipeline
from ray_curator.stages.modules.score_filter import Filter
from ray_curator.stages.io.reader.jsonl import JsonlReader
from ray_curator.stages.io.writer.jsonl import JsonlWriter

# Create a pipeline to filter on existing metadata
metadata_filter_pipeline = Pipeline(
    name="metadata_filter_pipeline",
    description="Filter documents based on existing scores"
)

# Add reader
metadata_filter_pipeline.add_stage(JsonlReader(file_paths="scored_data/*.jsonl"))

# Filter based on existing metadata field
metadata_filter_pipeline.add_stage(Filter(
    filter_fn=lambda score: score < 0.3,  # Keep only documents with toxicity < 0.3
    filter_field="toxicity_score"
))

# Add writer for filtered results
metadata_filter_pipeline.add_stage(JsonlWriter(output_dir="safe_documents/"))

# Execute the pipeline
metadata_filter_pipeline.build()
executor = RayDataExecutor()
metadata_filter_pipeline.run(executor)
```

## Best Practices

When developing custom filters:

1. **Leverage task-based processing**: The framework automatically handles batching through DocumentBatch objects
2. **Add meaningful metadata**: Store scores that provide insight into which documents pass filters
3. **Start simple**: Begin with basic filters and incrementally add complexity
4. **Test on samples**: Check your filters on small samples before processing large datasets using `limit` parameters in readers
5. **Track filter impact**: Use pipeline logging and stage performance metrics to track processing
6. **Resource allocation**: Specify appropriate CPU/GPU resources for computationally intensive filters
7. **Modular design**: Create reusable filters that work across different pipelines
8. **Document behavior**: Add clear documentation about what your filter does and its parameters

### Resource Configuration Example

```python
from ray_curator.stages.resources import Resources

class ComputeIntensiveFilter(DocumentFilter):
    # Override default resources for GPU-based processing
    _resources = Resources(cpus=2.0, gpu_memory_gb=4.0)
    
    def score_document(self, text: str):
        # GPU-accelerated scoring logic
        pass
```
