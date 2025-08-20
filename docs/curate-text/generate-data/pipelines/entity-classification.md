---
description: "Classify entities as math or Python-related using LLMs for filtering and labeling Wikipedia entries and other content"
categories: ["how-to-guides"]
tags: ["entity-classification", "math", "python", "filtering", "labeling", "wikipedia"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "text-only"
---

(text-gen-data-pipelines-entity-classification)=
# Entity Classification Pipeline

This pipeline uses an LLM to classify entities as math- or Python-related, following the methodology used in Nemotron-4 340B training. The classification helps identify whether Wikipedia entities or other text content relates to mathematical concepts or Python programming topics.

## Before You Start

- **LLM Client Setup**: The `NemotronGenerator` requires an `LLMClient` instance to interface with language models. Refer to the [LLM services documentation](text-generate-data-connect-service) for details on configuring your client with specific model providers.

---

## Setup Steps

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
from nemo_curator import OpenAIClient

client = OpenAIClient(openai_client)
```

### Initialize the Generator

Create the NemotronGenerator instance:

```python
from nemo_curator.synthetic import NemotronGenerator

generator = NemotronGenerator(client)
```

### Configure Generation Parameters

Set up your model and generation parameters:

```python
model = "mistralai/mixtral-8x7b-instruct-v0.1"
model_kwargs = {
    "temperature": 0.3,  # Lower temperature for consistent classification
    "top_p": 0.9,
    "max_tokens": 200,
}
```

### Classify Math Entities

Use the generator to classify math-related entities:

```python
math_classification_responses = generator.classify_math_entity(
    entity="Set theory",
    model=model,
    model_kwargs=model_kwargs
)

print(math_classification_responses[0])
# Output:
# Yes, the concept "Set theory" belongs to one of the following categories:
# - Important mathematics axioms, theorems, algorithms, equations, or inequalities.
# Set theory is a fundamental branch of mathematics...
```

### Classify Python Entities

Use the generator to classify Python-related entities:

```python
python_classification_responses = generator.classify_python_entity(
    entity="List comprehension",
    model=model,
    model_kwargs=model_kwargs
)

print(python_classification_responses[0])
# Output:
# Yes, the concept "List comprehension" belongs to one of the following categories:
# - Programming concepts like loops, functions, and data structures in python.
# List comprehensions are a concise way to create lists in Python...
```

## Advanced Configuration

Customize the classification process with custom prompts and parameters:

```python
# Import default prompt templates
from nemo_curator.synthetic.prompts import (
    DEFAULT_MATH_CLASSIFICATION_PROMPT_TEMPLATE,
    DEFAULT_PYTHON_CLASSIFICATION_PROMPT_TEMPLATE
)

# Configure advanced model parameters
advanced_model_kwargs = {
    "temperature": 0.1,  # Very low temperature for consistent classification
    "top_p": 0.95,
    "max_tokens": 300,
    "seed": 42  # For reproducible results
}

# Custom prompt parameters (optional)
custom_prompt_kwargs = {
    "additional_context": "Consider advanced topics as well."
}

# Math classification with custom parameters
math_response = generator.classify_math_entity(
    entity="Differential geometry",
    model=model,
    prompt_template=DEFAULT_MATH_CLASSIFICATION_PROMPT_TEMPLATE,
    prompt_kwargs=custom_prompt_kwargs,
    model_kwargs=advanced_model_kwargs
)

# Python classification with custom parameters
python_response = generator.classify_python_entity(
    entity="Django framework",
    model=model,
    prompt_template=DEFAULT_PYTHON_CLASSIFICATION_PROMPT_TEMPLATE,
    prompt_kwargs=custom_prompt_kwargs,
    model_kwargs=advanced_model_kwargs
)

print("Math classification:", math_response[0])
print("Python classification:", python_response[0])
```

## Batch Processing

For processing multiple entities efficiently:

```python
# Math entities to classify
math_entities = [
    "Linear algebra",
    "Cooking recipes",
    "Calculus",
    "Sports statistics",
    "Probability theory"
]

# Python entities to classify
python_entities = [
    "For loops",
    "Renaissance art",
    "NumPy arrays",
    "Italian cuisine",
    "Machine learning algorithms"
]

# Batch process math entities
math_results = []
for entity in math_entities:
    response = generator.classify_math_entity(
        entity=entity,
        model=model,
        model_kwargs=model_kwargs
    )
    classification = "Math-related" if response[0].lower().startswith("yes") else "Not math-related"
    math_results.append((entity, classification))

# Batch process Python entities
python_results = []
for entity in python_entities:
    response = generator.classify_python_entity(
        entity=entity,
        model=model,
        model_kwargs=model_kwargs
    )
    classification = "Python-related" if response[0].lower().startswith("yes") else "Not Python-related"
    python_results.append((entity, classification))

# Print results
print("Math Entity Classifications:")
for entity, classification in math_results:
    print(f"  {entity}: {classification}")

print("\nPython Entity Classifications:")
for entity, classification in python_results:
    print(f"  {entity}: {classification}")
```

## Classification Workflow

Combine both classification types for comprehensive entity analysis:

```python
def classify_entity_comprehensive(entity, generator, model, model_kwargs):
    """Classify an entity for both math and Python relevance."""
    
    # Get math classification
    math_response = generator.classify_math_entity(
        entity=entity,
        model=model,
        model_kwargs=model_kwargs
    )
    
    # Get Python classification
    python_response = generator.classify_python_entity(
        entity=entity,
        model=model,
        model_kwargs=model_kwargs
    )
    
    # Parse responses
    is_math = math_response[0].lower().startswith("yes")
    is_python = python_response[0].lower().startswith("yes")
    
    return {
        "entity": entity,
        "is_math_related": is_math,
        "is_python_related": is_python,
        "math_response": math_response[0],
        "python_response": python_response[0]
    }

# Example usage
entities_to_classify = [
    "Machine learning",
    "Quadratic equations", 
    "Web scraping",
    "Medieval history"
]

comprehensive_results = []
for entity in entities_to_classify:
    result = classify_entity_comprehensive(
        entity, generator, model, model_kwargs
    )
    comprehensive_results.append(result)

# Display results
for result in comprehensive_results:
    print(f"\nEntity: {result['entity']}")
    print(f"Math-related: {result['is_math_related']}")
    print(f"Python-related: {result['is_python_related']}") 