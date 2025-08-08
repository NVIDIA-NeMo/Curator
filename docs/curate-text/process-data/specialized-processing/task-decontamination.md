---
description: "Remove downstream task data from training datasets to prevent evaluation contamination and ensure valid benchmarking using ray-curator's task-based processing pipeline"
categories: ["how-to-guides"]
tags: ["task-decontamination", "benchmarks", "contamination", "evaluation", "n-grams", "downstream-tasks", "ray-curator", "stages", "pipeline"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "advanced"
content_type: "how-to"
modality: "text-only"
---

# Downstream Task Decontamination

(text-process-data-filter-task-decontamination)=

## Background

After training, practitioners evaluate large language models by their performance on downstream tasks consisting of unseen test data. When dealing with large datasets there is potential for leakage of this test data into the model's training dataset. NVIDIA NeMo Curator follows the approach of [OpenAI GPT3](https://arxiv.org/pdf/2005.14165.pdf) and [Microsoft Turing NLG 530B](https://arxiv.org/abs/2201.11990) to remove sections of documents in your dataset that are present in downstream tasks.

Ray-curator implements task decontamination using its task-based processing architecture, where documents flow through the pipeline as `DocumentBatch` tasks processed by specialized `ProcessingStage` components.

## Usage

Ray-curator implements task decontamination as a `ProcessingStage` that integrates into data processing pipelines. Here's how to use it:

```python
from ray_curator.pipeline import Pipeline
from ray_curator.stages.io.reader import JsonlReader
from ray_curator.stages.io.writer import JsonlWriter
from ray_curator.stages.modules.score_filter import ScoreFilter
from ray_curator.stages.filters.downstream_tasks import TaskDecontaminationFilter
from ray_curator.backends.experimental.ray_data import RayDataExecutor

# Define downstream tasks for decontamination
downstream_tasks = [
    "squad",        # SQuAD reading comprehension
    "trivia_qa",    # TriviaQA question answering
    "winogrande",   # Winogrande commonsense reasoning
]

# Create the decontamination stage
decontamination_stage = ScoreFilter(
    filter_obj=TaskDecontaminationFilter(
        tasks=downstream_tasks,
        max_ngram_size=13,
        max_matches=10,
        min_document_length=200,
        remove_char_each_side=200,
        max_splits=10
    ),
    text_field="text",
    score_field="contamination_score"  # Optional: keep scores for analysis
)

# Build the processing pipeline
pipeline = Pipeline(
    name="task_decontamination",
    description="Remove downstream task contamination from training data"
).add_stage(
    JsonlReader(file_paths="books_dataset/")
).add_stage(
    decontamination_stage
).add_stage(
    JsonlWriter(output_dir="decontaminated_books/")
)

# Execute the pipeline
executor = RayDataExecutor()
results = pipeline.run(executor)
```

### Parameters

The `TaskDecontaminationFilter` accepts several parameters to control the decontamination process:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tasks` | Required | List of downstream task names (strings) |
| `max_ngram_size` | `13` | Max size of n-grams to check for contamination |
| `max_matches` | `10` | If an n-gram appears more than this number of times, it's considered too common and not removed |
| `min_document_length` | `200` | Min character length for split documents to keep |
| `remove_char_each_side` | `200` | Number of characters to remove on either side of matching n-gram |
| `max_splits` | `10` | Max number of splits allowed before discarding document entirely |

The `ScoreFilter` stage provides other parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `text_field` | `"text"` | Field in DocumentBatch containing document text |
| `score_field` | `None` | Optional field to store contamination scores |
| `invert` | `False` | If True, keep contaminated documents instead of removing them |

For example, to use more aggressive removal settings:

```python
decontamination_stage = ScoreFilter(
    filter_obj=TaskDecontaminationFilter(
        tasks=downstream_tasks,
        max_ngram_size=10,               # Use smaller n-grams for matching
        max_matches=5,                   # Remove n-grams that appear in fewer documents
        remove_char_each_side=300,       # Remove more context around matches
        min_document_length=500          # Keep only longer document fragments
    ),
    text_field="content",                # Use different text field
    score_field="decontam_score"         # Store contamination scores
)
```

### Available Downstream Tasks

Ray-curator provides built-in support for common benchmark tasks. You can specify these tasks by name:

| Task Category | Available Tasks |
|---------------|----------------|
| **Question Answering** | `"squad"`, `"trivia_qa"`, `"quac"`, `"web_qa"`, `"coqa"`, `"drop"` |
| **Reading Comprehension** | `"race"`, `"multi_rc"`, `"record"` |
| **Commonsense Reasoning** | `"piqa"`, `"copa"`, `"winogrande"`, `"story_cloze"` |
| **Natural Language Inference** | `"anli"`, `"rte"`, `"cb"`, `"wic"` |
| **Knowledge Tasks** | `"arc_easy"`, `"arc_challenge"`, `"open_book_qa"`, `"bool_q"`, `"lambada"` |
| **Multi-task Benchmarks** | `"mmlu"`, `"big_bench_hard"`, `"big_bench_light"` |
| **Specialized Tasks** | `"wsc"`, `"num_dasc"`, `"multilingual"` |

Specify tasks as strings when creating the filter:

```python
# Use multiple tasks for comprehensive decontamination
tasks = ["squad", "trivia_qa", "mmlu", "winogrande", "anli"]

decontamination_filter = TaskDecontaminationFilter(tasks=tasks)
```

## Advanced Processing Configurations

Ray-curator's task-based architecture provides flexible options for implementing task decontamination with different processing strategies:

### Multi-Stage Pipeline Approach

For complex decontamination workflows, you can break the process into several stages:

```python
from ray_curator.stages.modules.score_filter import Score

# Stage 1: Score documents for contamination
scoring_stage = Score(
    score_fn=TaskDecontaminationFilter(tasks=["squad", "trivia_qa"]),
    score_field="contamination_score",
    text_field="text"
)

# Stage 2: Filter based on scores
filtering_stage = Filter(
    filter_fn=lambda score: score < 0.5,  # Keep documents with low contamination
    filter_field="contamination_score"
)

# Build pipeline with separate scoring and filtering
pipeline = Pipeline(
    name="multi_stage_decontamination"
).add_stage(JsonlReader(file_paths="input/")
).add_stage(scoring_stage
).add_stage(filtering_stage
).add_stage(JsonlWriter(output_dir="output/"))
```

### Batch Processing Configuration

Configure batch processing for better performance on large datasets:

```python
# Configure larger batch sizes for better throughput
decontamination_stage = ScoreFilter(
    filter_obj=TaskDecontaminationFilter(tasks=downstream_tasks),
    text_field="text"
).with_(
    batch_size=100,  # Process 100 documents per batch
    resources=Resources(cpus=4.0, gpu_memory_gb=8.0)
)
```

### Distributed Processing

Ray-curator automatically distributes processing across available resources:

```python
# Configure executor for distributed processing
executor = RayDataExecutor(config={
    "num_workers": 16,           # Use 16 worker processes
    "resources_per_worker": {
        "CPU": 2,
        "memory": 4_000_000_000  # 4GB memory per worker
    }
})

results = pipeline.run(executor)
```

## Creating Custom Downstream Tasks

If you need to decontaminate against a custom benchmark task not included in ray-curator, you can extend the `TaskDecontaminationFilter` to support custom n-gram sources:

```python
from ray_curator.stages.filters.doc_filter import DocumentFilter
from ray_curator.stages.utils.text_utils import get_words

class CustomTaskFilter(DocumentFilter):
    def __init__(self, 
                 task_data_path: str,
                 max_ngram_size: int = 13,
                 max_matches: int = 10,
                 min_document_length: int = 200,
                 remove_char_each_side: int = 200,
                 max_splits: int = 10):
        super().__init__()
        self.task_data_path = task_data_path
        self.max_ngram_size = max_ngram_size
        self.max_matches = max_matches
        self.min_document_length = min_document_length
        self.remove_char_each_side = remove_char_each_side
        self.max_splits = max_splits
        self._task_ngrams = self._load_task_ngrams()
    
    def _load_task_ngrams(self) -> set[str]:
        """Load n-grams from custom task data."""
        import json
        ngrams = set()
        
        with open(self.task_data_path) as f:
            for line in f:
                example = json.loads(line)
                # Extract text from your custom task format
                text_fields = [example.get("question", ""), 
                              example.get("context", ""),
                              example.get("answer", "")]
                
                for text in text_fields:
                    if text:
                        words, _ = get_words(text)
                        if len(words) >= 8:  # minimum n-gram size
                            for i in range(len(words) - self.max_ngram_size + 1):
                                ngram = " ".join(words[i:i + self.max_ngram_size])
                                ngrams.add(ngram)
        
        return ngrams
    
    def score_document(self, text: str) -> float:
        """Score document based on n-gram contamination."""
        words, _ = get_words(text)
        contamination_count = 0
        
        for i in range(len(words) - self.max_ngram_size + 1):
            ngram = " ".join(words[i:i + self.max_ngram_size])
            if ngram in self._task_ngrams:
                contamination_count += 1
                
        return contamination_count / max(1, len(words) - self.max_ngram_size + 1)
    
    def keep_document(self, score: float) -> bool:
        """Keep documents with low contamination scores."""
        return score < 0.1  # Threshold for contamination
```

Use your custom filter in a pipeline:

```python
# Create custom task decontamination stage
custom_decontamination = ScoreFilter(
    filter_obj=CustomTaskFilter(
        task_data_path="my_custom_task_data.jsonl",
        max_ngram_size=13
    ),
    text_field="text",
    score_field="custom_contamination_score"
)

# Add to pipeline
pipeline.add_stage(custom_decontamination)
```

## Performance Considerations

Task decontamination can be computationally intensive for large datasets. Ray-curator's distributed architecture provides several optimization strategies:

### Resource Optimization

1. **Choose important tasks**: Start with the most critical benchmark tasks for your application
2. **Configure batch sizes**: Larger batch sizes improve throughput but require more memory
3. **Adjust n-gram size**: Smaller values of `max_ngram_size` reduce computation but may increase false positives
4. **Use distributed processing**: Ray automatically scales across available CPU/GPU resources

### Pipeline Optimization

```python
# Optimize pipeline for throughput
optimized_stage = ScoreFilter(
    filter_obj=TaskDecontaminationFilter(tasks=["squad"]),
    text_field="text"
).with_(
    batch_size=200,  # Larger batches for better GPU utilization
    resources=Resources(cpus=8.0)  # More CPU cores for parallel processing
)

# Use efficient executors
executor = RayDataExecutor(config={
    "enable_auto_log_stats": True,  # Monitor performance
    "verbose_stats_logs": True
})
```

### Memory Management

1. **File partitioning**: Use appropriate `files_per_partition` in JsonlReader
2. **Progressive processing**: Process large datasets in chunks using file limits
3. **Resource monitoring**: Use Ray's built-in resource monitoring

### Scaling Strategies

- **Horizontal scaling**: Add more worker nodes to distribute processing
- **Vertical scaling**: Use machines with more CPU cores and memory
- **Hybrid approach**: Combine CPU workers for text processing with GPU workers for model-based filters

## References

- [Language Models are Zero-Shot Learners (Brown et al., 2020)](https://arxiv.org/abs/2005.14165)
- [Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B (Smith et al., 2021)](https://arxiv.org/abs/2201.11990)
- [Ray Datasets: Distributed Data Loading and Compute](https://docs.ray.io/en/latest/data/dataset.html)