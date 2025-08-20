---
description: "Generate Python coding problems for dialogue data with macro topics, subtopics, and problems for various skill levels"
categories: ["how-to-guides"]
tags: ["python", "coding", "programming", "problem-generation", "dialogue-data", "nemotron"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "text-only"
---

(text-gen-data-pipelines-python)=
# Python Generation Pipeline

This pipeline generates Python coding problems for dialogue data, as used in Nemotron-4 340B.

## Steps
1. Generate macro topics relating to Python
2. Generate subtopics for each macro topic
3. Generate a Python coding problem for each topic

## Setup

### Set up the LLM Client

Configure your LLM client (example with OpenAI):

```python
from openai import OpenAI

openai_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1", 
    api_key="<insert NVIDIA API key>"
)
```

### Create the NeMo Curator Client Wrapper

Wrap the client with NeMo Curator's client wrapper:

```python
from nemo_curator.services import OpenAIClient

llm_client = OpenAIClient(openai_client)
```

### Initialize the Generator

Create the NemotronGenerator instance:

```python
from nemo_curator.synthetic import NemotronGenerator

generator = NemotronGenerator(llm_client)
```

## Example Usage

```python
from nemo_curator.synthetic import NemotronGenerator
from nemo_curator.services import OpenAIClient
from nemo_curator.synthetic.error import YamlConversionError
from openai import OpenAI

# Set up LLM client
openai_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1", 
    api_key="<insert NVIDIA API key>"
)
llm_client = OpenAIClient(openai_client)
generator = NemotronGenerator(llm_client)

model = "mistralai/mixtral-8x7b-instruct-v0.1"

# Generate macro topics
macro_topic_responses = generator.generate_python_macro_topics(
    n_macro_topics=20,
    model=model
)

# Convert responses to list format
try:
    macro_topics_list = generator.convert_response_to_yaml_list(
        llm_response=macro_topic_responses[0],
        model=model
    )
except YamlConversionError as e:
    print(f"Error converting macro topics: {e}")
    # Handle conversion error or retry

# Generate subtopics for the first macro topic
subtopic_responses = generator.generate_python_subtopics(
    macro_topic=macro_topics_list[0],
    n_subtopics=5,
    model=model
)

# Convert subtopic responses to list format
try:
    subtopic_list = generator.convert_response_to_yaml_list(
        llm_response=subtopic_responses[0],
        model=model
    )
except YamlConversionError as e:
    print(f"Error converting subtopics: {e}")
    # Handle conversion error or retry

# Combine macro topics and subtopics
topics = macro_topics_list + subtopic_list

# Generate Python problems for the first topic
question_responses = generator.generate_python_problem(
    topic=topics[0],
    n_openlines=10,
    model=model
)

# Convert question responses to list format
try:
    questions = generator.convert_response_to_yaml_list(
        llm_response=question_responses[0],
        model=model
    )
except YamlConversionError as e:
    print(f"Error converting questions: {e}")
    # Handle conversion error or retry

print(f"Generated {len(questions)} Python problems for topic: {topics[0]}")
```

### End-to-End Pipeline

For a complete automated workflow, use the end-to-end pipeline:

```python
try:
    python_questions = generator.run_python_pipeline(
        n_macro_topics=20,
        n_subtopics=5,
        n_openlines=10,
        model=model,
    )
    print(f"Generated {len(python_questions)} Python coding problems")
    print(f"First question: {python_questions[0]}")
except YamlConversionError as e:
    print(f"Pipeline error: {e}")
    # Handle pipeline errors - you may want to retry with ignore_conversion_failure=True
```

### Error Handling

The pipeline methods may raise `YamlConversionError` when the LLM response cannot be parsed into the expected YAML list format. You can handle this by:

1. **Catching and retrying**: Retry the generation with different parameters
2. **Using pipeline options**: Set `ignore_conversion_failure=True` in `run_python_pipeline()` to skip failed conversions
3. **Manual parsing**: Parse the raw LLM responses manually if automatic conversion fails

```python
# Example with error tolerance
python_questions = generator.run_python_pipeline(
    n_macro_topics=20,
    n_subtopics=5,
    n_openlines=10,
    model=model,
    ignore_conversion_failure=True,  # Skip failed conversions
)
``` 