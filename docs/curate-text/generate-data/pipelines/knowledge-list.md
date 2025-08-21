---
description: "Generate structured knowledge lists from documents creating organized collections of factual information and key concepts"
categories: ["how-to-guides"]
tags: ["knowledge-lists", "structured-data", "factual-information", "bullet-points", "nemotron-cc"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "text-only"
---

(text-gen-data-pipelines-knowledge-list)=
# Knowledge List Generation Pipeline

This pipeline generates a structured list of knowledge items from documents, creating organized collections of factual information, concrete details, and key concepts. It follows the methodology used in the Nemotron-CC paper for extracting information-dense bullet points from text sources.

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
model_kwargs = {
    "temperature": 0.5,
    "top_p": 0.9,
    "max_tokens": 600,
}
```

### Generate Knowledge Lists from Documents

Use the generator to create structured knowledge lists:

```python
document = "The moon is bright. It shines at night."

responses = generator.generate_knowledge_list(
    document=document,
    model=model,
    model_kwargs=model_kwargs
)

print(responses[0])
# Output:
# - The moon is made of rock and dust.
# - The moon is the only natural satellite of the Earth.
# ...
```

## Advanced Configuration

Customize the knowledge list generation process with custom prompts and parameters:

```python
# Import default prompt templates
from nemo_curator.synthetic.prompts import (
    KNOWLEDGE_LIST_PROMPT_TEMPLATE,
    NEMOTRON_CC_SYSTEM_PROMPT
)

# Configure advanced model parameters
advanced_model_kwargs = {
    "temperature": 0.3,  # Lower temperature for more consistent lists
    "top_p": 0.95,
    "max_tokens": 800,
    "seed": 42  # For reproducible results
}

# Custom prompt parameters (optional)
custom_prompt_kwargs = {
    "focus_area": "key facts and statistics"
}

# Knowledge list generation with custom parameters
knowledge_list_response = generator.generate_knowledge_list(
    document=document,
    model=model,
    prompt_template=KNOWLEDGE_LIST_PROMPT_TEMPLATE,
    system_prompt=NEMOTRON_CC_SYSTEM_PROMPT,
    prompt_kwargs=custom_prompt_kwargs,
    model_kwargs=advanced_model_kwargs
)

print("Generated knowledge list:", knowledge_list_response[0])
```

## Batch Processing

For processing multiple documents efficiently:

```python
# Documents to process
documents = [
    "Solar panels convert sunlight into electricity through photovoltaic cells. They are becoming increasingly popular for residential and commercial use due to declining costs and environmental benefits.",
    "Machine learning algorithms can identify patterns in data that humans might miss. Deep learning, a subset of machine learning, uses neural networks with multiple layers to process complex information.",
    "The human brain contains approximately 86 billion neurons connected by trillions of synapses. These connections enable complex thought processes, memory formation, and decision-making.",
    "Cryptocurrency operates on blockchain technology, which provides decentralized and secure transaction records. Bitcoin, launched in 2009, was the first successful implementation of this technology."
]

# Batch process documents
knowledge_list_results = []
for i, doc in enumerate(documents):
    response = generator.generate_knowledge_list(
        document=doc,
        model=model,
        model_kwargs=model_kwargs
    )
    knowledge_list_results.append({
        "document_id": i + 1,
        "original_text": doc,
        "knowledge_list": response[0]
    })

# Print results
print("Knowledge List Generation Results:")
for result in knowledge_list_results:
    print(f"\nDocument {result['document_id']}:")
    print(f"Original: {result['original_text'][:80]}...")
    print(f"Knowledge List:\n{result['knowledge_list']}")
```

## Knowledge List Processing Workflow

Combine knowledge list generation with other NemotronCC methods for comprehensive document processing:

```python
def process_document_comprehensive(document, generator, model, model_kwargs):
    """Generate knowledge lists and other content formats from a document."""
    
    # Generate structured knowledge list
    knowledge_list_response = generator.generate_knowledge_list(
        document=document,
        model=model,
        model_kwargs=model_kwargs
    )
    
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
    
    return {
        "original_document": document,
        "knowledge_list": knowledge_list_response[0],
        "extracted_knowledge": knowledge_response[0],
        "qa_pairs": qa_response[0]
    }

# Example usage
sample_document = """
Renewable energy sources are becoming increasingly important in the fight against climate change. 
Solar power capacity has grown by over 20% annually for the past decade, making it one of the 
fastest-growing energy sources globally. Wind energy has also seen remarkable growth, with modern 
turbines capable of generating 15 megawatts of power. These technologies, combined with improved 
battery storage systems, are making renewable energy more reliable and cost-effective than 
traditional fossil fuels in many regions.
"""

comprehensive_result = process_document_comprehensive(
    sample_document, generator, model, model_kwargs
)

# Display results
print("Comprehensive Document Processing:")
print(f"\nOriginal Document: {comprehensive_result['original_document'][:100]}...")
print(f"\nKnowledge List:\n{comprehensive_result['knowledge_list']}")
print(f"\nExtracted Knowledge: {comprehensive_result['extracted_knowledge']}")
print(f"\nQA Pairs: {comprehensive_result['qa_pairs']}")
```

## Post-processing

You can use the `NemotronCCKnowledgeListPostprocessor` to clean and format the output:

```python
from nemo_curator.synthetic import NemotronCCKnowledgeListPostprocessor
from nemo_curator.datasets import DocumentDataset

# Create dataset from generator output
data = {"text": responses}
dataset = DocumentDataset.from_dict(data)

# Apply postprocessing to clean bullet points and formatting
postprocessor = NemotronCCKnowledgeListPostprocessor(text_field="text")
processed_dataset = postprocessor(dataset)

print(processed_dataset.df.compute().iloc[0]["text"])
# Output (cleaned):
# The moon is made of rock and dust.
# The moon is the only natural satellite of the Earth.
# ...
``` 