---
description: "Extract key knowledge from documents summarizing important facts and concepts using the Nemotron-CC methodology"
categories: ["how-to-guides"]
tags: ["knowledge-extraction", "fact-extraction", "summarization", "nemotron-cc", "information-extraction"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "text-only"
---

(text-gen-data-pipelines-extract-knowledge)=
# Knowledge Extraction Pipeline

This pipeline extracts key knowledge from documents, summarizing important facts, concrete details, and key concepts. It follows the methodology used in the Nemotron-CC paper for distilling essential information from text sources into structured, information-dense extracts.

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
from nemo_curator.synthetic.nemotron_cc import NemotronCCGenerator

generator = NemotronCCGenerator(client)
```

### Configure Generation Parameters

Set up your model and generation parameters:

```python
model = "nv-mistralai/mistral-nemo-12b-instruct"
model_kwargs = {
    "temperature": 0.5,
    "top_p": 0.9,
    "max_tokens": 1400,
}
```

### Extract Knowledge from Documents

Use the generator to extract key knowledge from text:

```python
document = ("The moon is bright. It shines at night. I love the moon. I first saw it up"
           " close through a telescope in 1999 at a sleepover.")

responses = generator.extract_knowledge(
    document=document,
    model=model,
    model_kwargs=model_kwargs
)

print(responses[0])
# Output:
# The moon is a reflective body visible from the Earth at night.
```

## Advanced Configuration

Customize the knowledge extraction process with custom prompts and parameters:

```python
# Import default prompt templates
from nemo_curator.synthetic.prompts import (
    EXTRACT_KNOWLEDGE_PROMPT_TEMPLATE,
    NEMOTRON_CC_SYSTEM_PROMPT
)

# Configure advanced model parameters
advanced_model_kwargs = {
    "temperature": 0.3,  # Lower temperature for more focused extraction
    "top_p": 0.95,
    "max_tokens": 2000,
    "seed": 42  # For reproducible results
}

# Custom prompt parameters (optional)
custom_prompt_kwargs = {
    "focus_area": "scientific facts"
}

# Knowledge extraction with custom parameters
knowledge_response = generator.extract_knowledge(
    document=document,
    model=model,
    prompt_template=EXTRACT_KNOWLEDGE_PROMPT_TEMPLATE,
    system_prompt=NEMOTRON_CC_SYSTEM_PROMPT,
    prompt_kwargs=custom_prompt_kwargs,
    model_kwargs=advanced_model_kwargs
)

print("Extracted knowledge:", knowledge_response[0])
```

## Batch Processing

For processing multiple documents efficiently:

```python
# Documents to process
documents = [
    "Albert Einstein developed the theory of relativity in 1905. This revolutionary theory changed our understanding of space and time.",
    "Python is a high-level programming language created by Guido van Rossum in 1991. It emphasizes code readability and simplicity.",
    "The Great Wall of China spans over 13,000 miles and was built over many centuries. It remains one of the most impressive architectural achievements in history.",
    "Photosynthesis is the process by which plants convert sunlight into energy. This process produces oxygen as a byproduct, which is essential for most life on Earth."
]

# Batch process documents
extraction_results = []
for i, doc in enumerate(documents):
    response = generator.extract_knowledge(
        document=doc,
        model=model,
        model_kwargs=model_kwargs
    )
    extraction_results.append({
        "document_id": i + 1,
        "original_text": doc,
        "extracted_knowledge": response[0]
    })

# Print results
print("Knowledge Extraction Results:")
for result in extraction_results:
    print(f"\nDocument {result['document_id']}:")
    print(f"Original: {result['original_text'][:80]}...")
    print(f"Extracted: {result['extracted_knowledge']}")
```

## Knowledge Processing Workflow

Combine knowledge extraction with other NemotronCC methods for comprehensive document processing:

```python
def process_document_comprehensive(document, generator, model, model_kwargs):
    """Extract knowledge and generate diverse content from a document."""
    
    # Extract key knowledge
    knowledge_response = generator.extract_knowledge(
        document=document,
        model=model,
        model_kwargs=model_kwargs
    )
    
    # Generate diverse QA pairs
    qa_response = generator.generate_diverse_qa(
        document=document,
        model=model,
        model_kwargs=model_kwargs
    )
    
    # Distill essential content
    distill_response = generator.distill(
        document=document,
        model=model,
        model_kwargs=model_kwargs
    )
    
    return {
        "original_document": document,
        "extracted_knowledge": knowledge_response[0],
        "qa_pairs": qa_response[0],
        "distilled_content": distill_response[0]
    }

# Example usage
sample_document = """
The Amazon rainforest covers 2.1 million square miles across nine countries in South America. 
It contains approximately 390 billion trees and is home to 10% of all known species on Earth. 
The rainforest plays a crucial role in regulating global climate by absorbing carbon dioxide 
and producing oxygen. Unfortunately, deforestation threatens this vital ecosystem, with an 
area the size of a football field being cleared every minute.
"""

comprehensive_result = process_document_comprehensive(
    sample_document, generator, model, model_kwargs
)

# Display results
print("Comprehensive Document Processing:")
print(f"\nOriginal Document: {comprehensive_result['original_document'][:100]}...")
print(f"\nExtracted Knowledge: {comprehensive_result['extracted_knowledge']}")
print(f"\nQA Pairs: {comprehensive_result['qa_pairs']}")
print(f"\nDistilled Content: {comprehensive_result['distilled_content']}")
``` 