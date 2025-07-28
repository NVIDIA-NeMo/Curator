---
description: "Generate diverse question-answer pairs from documents using various cognitive skills following Nemotron-4 340B methodology"
categories: ["how-to-guides"]
tags: ["diverse-qa", "question-generation", "cognitive-skills", "document-processing", "nemotron"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "text-only"
---

(text-gen-data-pipelines-diverse-qa)=
# Diverse QA Generation Pipeline

This pipeline generates a list of diverse QA pairs from a document, useful for building question-answering datasets. The pipeline creates questions that require different cognitive skills and cover various aspects of the input text, following the methodology used in Nemotron-4 340B training.

## Before You Start

- **LLM Client Setup**: The `NemotronCCGenerator` requires an `LLMClient` instance to interface with language models. Refer to the [LLM services documentation](text-generate-data-connect-service) for details on configuring your client with specific model providers.

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

Create the NemotronCCGenerator instance:

```python
from nemo_curator.synthetic import NemotronCCGenerator

generator = NemotronCCGenerator(client)
```

### Configure Generation Parameters

Set up your model and generation parameters:

```python
model = "nv-mistralai/mistral-nemo-12b-instruct"
document = "The moon is bright. It shines at night."
model_kwargs = {
    "temperature": 0.5,
    "top_p": 0.9,
    "max_tokens": 600,
}
```

### Generate Diverse QA Pairs

Use the generator to create diverse question-answer pairs:

```python
responses = generator.generate_diverse_qa(
    document=document,
    model=model,
    model_kwargs=model_kwargs
)

print(responses[0])
# Output:
# Here are the questions and answers based on the provided text:
# - Question: What characteristic does the moon have? Answer: The moon is bright.
# - Question: When does the moon shine? Answer: It shines at night.
# - Question: Is the moon bright? Answer: Yes, the moon is bright.
```

## Advanced Configuration

Customize the QA generation process with additional parameters:

```python
# Use custom prompts and system messages
from nemo_curator.synthetic.prompts import (
    DIVERSE_QA_PROMPT_TEMPLATE,
    NEMOTRON_CC_SYSTEM_PROMPT
)

# Configure advanced model parameters
advanced_model_kwargs = {
    "temperature": 0.7,  # Higher temperature for more diverse questions
    "top_p": 0.9,
    "max_tokens": 800,
    "seed": 42  # For reproducible results
}

# Custom prompt parameters (optional)
custom_prompt_kwargs = {
    "max_questions": 5  # Limit number of questions if needed
}

responses = generator.generate_diverse_qa(
    document=document,
    model=model,
    prompt_template=DIVERSE_QA_PROMPT_TEMPLATE,
    system_prompt=NEMOTRON_CC_SYSTEM_PROMPT,
    prompt_kwargs=custom_prompt_kwargs,
    model_kwargs=advanced_model_kwargs
)

print(responses[0])
```

## Batch Processing

For processing multiple documents efficiently:

```python
documents = [
    "The sun provides energy for all life on Earth through photosynthesis.",
    "Water freezes at 0 degrees Celsius and boils at 100 degrees Celsius.",
    "Gravity is the force that attracts objects toward each other."
]

qa_results = []
for doc in documents:
    responses = generator.generate_diverse_qa(
        document=doc,
        model=model,
        model_kwargs=model_kwargs
    )
    qa_results.append(responses[0])

# Print results
for i, result in enumerate(qa_results):
    print(f"Document {i+1} QA pairs:\n{result}\n")
```

## Post-processing

You can use the `NemotronCCDiverseQAPostprocessor` to reformat the output for downstream use:

```python
from nemo_curator.synthetic import NemotronCCDiverseQAPostprocessor
from transformers import AutoTokenizer

# Initialize tokenizer (optional, for token-based sampling)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Configure post-processor
postprocessor = NemotronCCDiverseQAPostprocessor(
    tokenizer=tokenizer,
    text_field="text",
    response_field="response",
    max_num_pairs=3,
    prefix="Here are the questions and answers based on the provided text:"
)

# Apply post-processing to your dataset
processed_dataset = postprocessor(your_dataset)
``` 