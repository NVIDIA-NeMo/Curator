---
description: "Generate closed-ended questions about documents for evaluation and comprehension datasets using Nemotron-4 340B approach"
categories: ["how-to-guides"]
tags: ["closed-qa", "question-generation", "document-based", "evaluation", "comprehension", "nemotron"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "text-only"
---

(text-gen-data-pipelines-closed-qa)=
# Closed Q&A Generation Pipeline

This pipeline generates closed-ended questions about a given document, as used in Nemotron-4 340B. Closed-ended questions are specific questions that can be answered directly from the provided document content, as opposed to open-ended questions that require broader knowledge.

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
from nemo_curator.services import OpenAIClient

llm_client = OpenAIClient(openai_client)
```

### Initialize the Generator

Create the NemotronGenerator instance:

```python
from nemo_curator.synthetic import NemotronGenerator

generator = NemotronGenerator(llm_client)
```

### Configure Generation Parameters

Set up your model and generation parameters:

```python
model = "mistralai/mixtral-8x7b-instruct-v0.1"
document = "Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal."
n_openlines = 5
```

### Generate Questions from Document

Use the generator to create closed-ended questions:

```python
closed_qa_responses = generator.generate_closed_qa_instructions(
    document=document,
    n_openlines=n_openlines,
    model=model,
)

# Parse the responses to extract individual questions
closed_qa_questions = generator.convert_response_to_yaml_list(
    closed_qa_responses[0], 
    model=model
)

print(closed_qa_questions[0])
# Output:
# "Which President of the United States gave this speech?"
```

### Run the End-to-End Pipeline

For processing multiple documents, use the complete pipeline:

```python
documents = [
    "Four score and seven years ago our fathers brought forth on this continent...",
    "We hold these truths to be self-evident, that all men are created equal...",
    # Add more documents as needed
]

closed_qa_questions = generator.run_closed_qa_pipeline(
    documents=documents,
    n_openlines=n_openlines,
    model=model,
)

print(closed_qa_questions[0])
# Output:
# (0, "Which President of the United States gave this speech?")
``` 