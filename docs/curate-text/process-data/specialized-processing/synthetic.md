---
description: "Create custom filters for detecting and filtering synthetic or AI-generated content in datasets"
categories: ["how-to-guides"]
tags: ["synthetic-detection", "custom-filters", "embedding", "qa-pairs", "document-processing"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "advanced"
content_type: "how-to"
modality: "text-only"
---

(text-process-data-filter-synthetic)=

# Custom Synthetic Text Detection

This guide shows how to create custom filters for detecting and filtering synthetic or AI-generated content using the ray-curator framework. You'll learn how to build document filters that can identify patterns common in machine-generated text, such as excessive lexical similarity between questions and contexts or unanswerable questions.

Ray-curator's flexible DocumentFilter system allows you to create specialized filters for synthetic text detection by building custom scoring and filtering logic. This is especially valuable when creating high-quality datasets for question-answering systems, retrieval tasks, and other applications where authentic content is crucial.

## Filter Architecture

Ray-curator uses a task-based processing pipeline where documents flow through stages as `DocumentBatch` objects. Custom filters inherit from the `DocumentFilter` base class and integrate into processing pipelines using the `ScoreFilter` stage.

The filter architecture consists of:

1. **DocumentFilter**: Base class that defines the scoring and filtering interface
2. **ScoreFilter Stage**: Processing stage that applies filters to document batches  
3. **Pipeline**: Orchestrates processing stages in sequence
4. **Executors**: Backend implementations for running pipelines

Custom synthetic detection filters typically build two key methods:

- `score_document()`: Analyzes text and returns a numeric score
- `keep_document()`: Determines whether to keep documents based on their scores

## Creating Custom Synthetic Filters

Here's how to implement custom filters for detecting synthetic content in question-answering datasets:

### Embedding-Based Easiness Filter

Create a filter that identifies questions with excessive lexical similarity to their contexts:

```python
from dataclasses import dataclass
import numpy as np
import pandas as pd
from openai import OpenAI

from ray_curator.stages.filters.doc_filter import DocumentFilter
from ray_curator.stages.modules.score_filter import ScoreFilter
from ray_curator.pipeline import Pipeline
from ray_curator.stages.io.reader.jsonl import JsonlReader
from ray_curator.stages.io.writer.jsonl import JsonlWriter
from ray_curator.backends.xenna import XennaExecutor

@dataclass
class EasinessFilter(DocumentFilter):
    """Filter that identifies questions too easily retrievable from their context."""
    
    base_url: str
    api_key: str
    model: str
    percentile: float = 0.7
    truncate: str = "NONE"
    
    def __post_init__(self):
        super().__init__()
        self._name = "easiness_filter"
        self.client = None

    def score_document(self, text: str) -> float:
        """Calculate embedding similarity between question and context."""
        if self.client is None:
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        
        # Assume text is formatted as "Context: {context}\nQuestion: {question}"
        parts = text.split("\nQuestion: ")
        if len(parts) != 2:
            return 0.0
            
        context = parts[0].replace("Context: ", "")
        question = parts[1]
        
        return self._calc_similarity(context, question)
    
    def keep_document(self, score: float) -> bool:
        """Keep documents with similarity below the percentile threshold."""
        # This would typically be implemented after collecting scores
        # For simplicity, using a fixed threshold here
        return score <= 0.8
    
    def _calc_similarity(self, context: str, question: str) -> float:
        """Calculate cosine similarity between context and question embeddings."""
        try:
            # Get embeddings
            context_response = self.client.embeddings.create(
                input=[context], 
                model=self.model,
                extra_body={"input_type": "passage", "truncate": self.truncate}
            )
            question_response = self.client.embeddings.create(
                input=[question], 
                model=self.model,
                extra_body={"input_type": "query", "truncate": self.truncate}
            )
            
            context_embed = np.array(context_response.data[0].embedding)
            question_embed = np.array(question_response.data[0].embedding)
            
            # Calculate cosine similarity
            similarity = np.dot(context_embed, question_embed) / (
                np.linalg.norm(context_embed) * np.linalg.norm(question_embed)
            )
            return float(similarity)
            
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0

# Create and configure the filter stage
easiness_filter = ScoreFilter(
    filter_obj=EasinessFilter(
        base_url="https://your-embedding-api-endpoint",
        api_key="your-api-key",
        model="embedding-model-name",
        percentile=0.7
    ),
    text_field="text",
    score_field="easiness_score"
)
```

### LLM-Based Question Quality Filter

Create a filter that uses language models to determine if questions can be answered from their contexts:

```python
import json
from dataclasses import dataclass

@dataclass
class AnswerabilityFilter(DocumentFilter):
    """Filter that identifies questions that cannot be answered from their context."""
    
    base_url: str
    api_key: str
    model: str
    system_prompt: str
    user_prompt_template: str
    num_criteria: int = 1
    
    def __post_init__(self):
        super().__init__()
        self._name = "answerability_filter"
        self.client = None

    def score_document(self, text: str) -> str:
        """Use LLM to evaluate if question is answerable from context."""
        if self.client is None:
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        
        # Parse the text to extract context and question
        parts = text.split("\nQuestion: ")
        if len(parts) != 2:
            return '{"criterion_1": "N"}'
            
        context = parts[0].replace("Context: ", "")
        question = parts[1]
        
        return self._llm_as_judge(context, question)
    
    def keep_document(self, score: str) -> bool:
        """Keep documents where all criteria are met."""
        try:
            criteria = json.loads(score)
            # All criteria must be "Y" to keep the document
            for i in range(self.num_criteria):
                if criteria.get(f"criterion_{i + 1}", "N") != "Y":
                    return False
            return True
        except (json.JSONDecodeError, KeyError):
            # If parsing fails, default to keeping the document
            return True
    
    def _llm_as_judge(self, context: str, question: str) -> str:
        """Query LLM to evaluate answerability criteria."""
        user_query = self.system_prompt + "\n\n"
        user_query += self.user_prompt_template.format(text=context, question=question)
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": user_query}],
                temperature=0.1,
                max_tokens=512
            )
            return completion.choices[0].message.content or '{"criterion_1": "N"}'
        except Exception as e:
            print(f"LLM API error: {e}")
            return '{"criterion_1": "N"}'

# Configure the answerability filter
answerability_filter = ScoreFilter(
    filter_obj=AnswerabilityFilter(
        base_url="https://your-llm-api-endpoint",
        api_key="your-api-key",
        model="gpt-4",
        system_prompt="You are an expert at determining if questions can be answered from given context.",
        user_prompt_template="Context: {text}\n\nQuestion: {question}\n\nIs this question answerable from the context? Reply with JSON: {{\"criterion_1\": \"Y\"}} or {{\"criterion_1\": \"N\"}}",
        num_criteria=1
    ),
    text_field="text",
    score_field="answerability_result"
)
```

## Complete Pipeline Example

Here's how to build a complete pipeline that reads QA data, applies synthetic content filters, and writes the results:

```python
from ray_curator.pipeline import Pipeline
from ray_curator.backends.xenna import XennaExecutor

# Create the processing pipeline
pipeline = Pipeline(
    name="synthetic_qa_filtering",
    description="Filter synthetic content from QA datasets"
)

# Add stages in sequence
pipeline.add_stage(
    # Read JSONL data files
    JsonlReader(
        file_paths="qa_dataset/*.jsonl",
        files_per_partition=10
    )
)

pipeline.add_stage(
    # Apply easiness filter first
    easiness_filter
)

pipeline.add_stage(
    # Then apply answerability filter
    answerability_filter
)

pipeline.add_stage(
    # Write filtered results
    JsonlWriter(
        output_dir="filtered_qa_dataset/"
    )
)

# Execute the pipeline
executor = XennaExecutor()
results = pipeline.run(executor)

print(f"Pipeline completed. Processed {len(results)} document batches.")
```

## Advanced Configuration

### Multi-Criteria Evaluation

You can create more sophisticated quality filters with evaluation criteria:

```python
multi_criteria_filter = AnswerabilityFilter(
    base_url="https://your-llm-api-endpoint",
    api_key="your-api-key",
    model="gpt-4",
    system_prompt="You are an expert at evaluating question-context pairs for training data quality.",
    user_prompt_template="""Context: {text}

Question: {question}

Evaluate the following criteria:
1. Is the question answerable from the context? (Y/N)
2. Is the answer clearly stated in the context? (Y/N)
3. Does the question require reasoning beyond simple extraction? (Y/N)

Format your response as JSON: {{"criterion_1": "Y", "criterion_2": "Y", "criterion_3": "Y"}}""",
    num_criteria=3
)
```

## Best Practices

### Implementation Guidelines

When implementing custom synthetic text filters, follow these best practices:

1. **Start with conservative thresholds**: Begin with less aggressive filtering to preserve dataset size
   ```python
   # Conservative easiness filter
   conservative_easiness = EasinessFilter(
       base_url="https://api-endpoint",
       api_key="your-key", 
       model="embedding-model",
       percentile=0.5,  # Filter only the most obvious cases
       truncate="NONE"
   )
   ```

2. **Handle API failures gracefully**: Implement robust error handling for external API calls
   ```python
   def score_document(self, text: str) -> float:
       try:
           # API call logic
           return self._calculate_score(text)
       except Exception as e:
           print(f"API error: {e}")
           return 0.0  # Default to keeping document
   ```

3. **Monitor filter performance**: Track filtering statistics during pipeline execution
   ```python
   # Add logging to your pipeline
   pipeline = Pipeline("synthetic_filtering")
   pipeline.add_stage(Score(score_fn=easiness_filter, score_field="easiness"))
   pipeline.add_stage(Score(score_fn=answerability_filter, score_field="answerability"))
   pipeline.add_stage(combined_filter)  # Apply actual filtering
   ```

## Example Use Cases

### High-Quality QA Dataset Creation

Create a comprehensive pipeline for producing high-quality question-answering datasets:

```python
# Complete QA dataset filtering pipeline
qa_pipeline = Pipeline("qa_quality_pipeline")

# Read raw QA data
qa_pipeline.add_stage(JsonlReader(
    file_paths="raw_qa_data/*.jsonl",
    files_per_partition=50
))

# Score questions for easiness (but don't filter yet)
qa_pipeline.add_stage(Score(
    score_fn=EasinessFilter(
        base_url="https://embedding-api",
        api_key="your-key",
        model="text-embedding-3-large",
        percentile=0.7
    ),
    score_field="easiness_score",
    text_field="text"
))

# Score questions for answerability
qa_pipeline.add_stage(Score(
    score_fn=AnswerabilityFilter(
        base_url="https://llm-api", 
        api_key="your-key",
        model="gpt-4",
        system_prompt="Evaluate QA pairs for training data quality.",
        user_prompt_template="Context: {text}\nQuestion: {question}\nAnswerable? JSON: {{\"criterion_1\": \"Y/N\"}}",
        num_criteria=1
    ),
    score_field="answerability_score",
    text_field="text"
))

# Apply combined filtering logic
qa_pipeline.add_stage(Filter(
    filter_fn=lambda row: (
        row["easiness_score"] <= 0.8 and  # Not too easy
        "Y" in row["answerability_score"]  # Answerable
    ),
    filter_field=["easiness_score", "answerability_score"]
))

# Write high-quality results
qa_pipeline.add_stage(JsonlWriter(output_dir="high_quality_qa/"))

# Execute
executor = XennaExecutor()
results = qa_pipeline.run(executor)
```

### Resource Optimization

For large-scale processing, consider resource allocation and batching:

```python
from ray_curator.stages.resources import Resources

# Configure GPU resources for LLM-based filtering
answerability_stage = ScoreFilter(
    filter_obj=AnswerabilityFilter(...),
    text_field="text"
).with_(
    resources=Resources(gpus=1.0),  # Allocate full GPU
    batch_size=32  # Process multiple documents together
)

# CPU-only for embedding similarity
easiness_stage = ScoreFilter(
    filter_obj=EasinessFilter(...),
    text_field="text"
).with_(
    resources=Resources(cpus=4.0),
    batch_size=64
)
```

By implementing custom synthetic text detection filters with ray-curator's flexible architecture, you can create robust pipelines for ensuring the authenticity and quality of your training datasets. The task-based processing model allows for efficient scaling and resource management across different filtering stages.
