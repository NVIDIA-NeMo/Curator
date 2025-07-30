---
description: "Generate open-ended questions for dialogue data covering general knowledge following the Nemotron-4 340B approach"
categories: ["how-to-guides"]
tags: ["open-qa", "dialogue-data", "general-knowledge", "question-generation", "nemotron"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "text-only"
---

(text-gen-data-pipelines-open-qa)=
# Open Q&A Generation Pipeline

This pipeline generates open-ended questions ("openlines") for dialogue data, following the approach used in Nemotron-4 340B. Unlike closed-ended questions that are based on specific documents, open-ended questions cover general knowledge and can be answered using broad understanding of various topics.

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
model_kwargs = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 500,
}
```

### Generate Questions Step by Step

Use the generator to create open-ended questions through the four-step process:

```python
from nemo_curator.synthetic.error import YamlConversionError

try:
    # Step 1: Generate macro topics
    macro_topic_responses = generator.generate_macro_topics(
        n_macro_topics=20, 
        model=model,
        model_kwargs=model_kwargs
    )
    macro_topics_list = generator.convert_response_to_yaml_list(
        macro_topic_responses[0], 
        model=model
    )
    
    print(f"Generated {len(macro_topics_list)} macro topics:")
    for i, topic in enumerate(macro_topics_list[:3], 1):
        print(f"{i}. {topic}")

    # Step 2: Generate subtopics for the first macro topic
    subtopic_responses = generator.generate_subtopics(
        macro_topic=macro_topics_list[0], 
        n_subtopics=5, 
        model=model,
        model_kwargs=model_kwargs
    )
    subtopic_list = generator.convert_response_to_yaml_list(
        subtopic_responses[0], 
        model=model
    )
    
    print(f"\nGenerated {len(subtopic_list)} subtopics for '{macro_topics_list[0]}':")
    for i, subtopic in enumerate(subtopic_list, 1):
        print(f"{i}. {subtopic}")

    # Step 3: Combine topics for question generation
    topics = macro_topics_list + subtopic_list

    # Generate questions from the first topic
    question_responses = generator.generate_open_qa_from_topic(
        topic=topics[0], 
        n_openlines=10, 
        model=model,
        model_kwargs=model_kwargs
    )
    questions = generator.convert_response_to_yaml_list(
        question_responses[0], 
        model=model
    )
    
    print(f"\nGenerated {len(questions)} questions for '{topics[0]}':")
    for i, question in enumerate(questions[:3], 1):
        print(f"{i}. {question}")

    # Step 4: Revise the first question
    revised_questions_responses = generator.revise_open_qa(
        openline=questions[0], 
        n_revisions=5, 
        model=model,
        model_kwargs=model_kwargs
    )
    revised_questions = generator.convert_response_to_yaml_list(
        revised_questions_responses[0], 
        model=model
    )
    
    print(f"\nGenerated {len(revised_questions)} revised versions:")
    for i, revision in enumerate(revised_questions, 1):
        print(f"{i}. {revision}")

except YamlConversionError as e:
    print(f"Error converting LLM response to structured format: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Advanced Configuration

Customize the question generation process with different parameters and error handling strategies:

```python
# Import prompt templates for customization
from nemo_curator.synthetic.prompts import (
    DEFAULT_MACRO_TOPICS_PROMPT_TEMPLATE,
    DEFAULT_SUBTOPICS_PROMPT_TEMPLATE,
    DEFAULT_OPEN_QA_FROM_TOPICS_PROMPT_TEMPLATE,
    DEFAULT_REVISE_OPEN_QA_PROMPT_TEMPLATE,
    DEFAULT_YAML_CONVERSION_PROMPT_TEMPLATE
)

# Configure advanced model parameters
advanced_model_kwargs = {
    "temperature": 0.8,  # Higher temperature for more creative questions
    "top_p": 0.95,
    "max_tokens": 800,
    "seed": 42  # For reproducible results
}

conversion_model_kwargs = {
    "temperature": 0.2,  # Lower temperature for more consistent parsing
    "max_tokens": 1000
}

# Generate with custom prompts and error handling
try:
    questions = generator.generate_open_qa_from_topic(
        topic="Climate change and environmental sustainability",
        n_openlines=15,
        model=model,
        prompt_template=DEFAULT_OPEN_QA_FROM_TOPICS_PROMPT_TEMPLATE,
        model_kwargs=advanced_model_kwargs
    )
    
    parsed_questions = generator.convert_response_to_yaml_list(
        questions[0],
        model=model,
        prompt_template=DEFAULT_YAML_CONVERSION_PROMPT_TEMPLATE,
        model_kwargs=conversion_model_kwargs
    )
    
    print(f"Generated {len(parsed_questions)} climate-related questions:")
    for i, question in enumerate(parsed_questions[:5], 1):
        print(f"{i}. {question}")
        
except YamlConversionError as e:
    print(f"Parsing failed: {e}")
    # Implement fallback parsing or retry logic
except Exception as e:
    print(f"Generation failed: {e}")
```

## End-to-End Pipeline

For automated processing with comprehensive error handling, use the complete pipeline:

```python
try:
    # Complete pipeline execution with error resilience
    open_qa_questions = generator.run_open_qa_pipeline(
        n_macro_topics=20,
        n_subtopics=5,
        n_openlines=10,
        n_revisions=5,
        model=model,
        base_model_kwargs=advanced_model_kwargs,
        conversion_model_kwargs=conversion_model_kwargs,
        ignore_conversion_failure=True,  # Continue on conversion errors
        combine_topics=True  # Mix macro and subtopics
    )
    
    print(f"Generated {len(open_qa_questions)} total questions")
    print("\nSample questions:")
    for i, question in enumerate(open_qa_questions[:5], 1):
        print(f"{i}. {question}")
    
    # Example output:
    # Generated 2000 total questions
    # Sample questions:
    # 1. What are some effective sources of renewable energy?
    # 2. How does artificial intelligence impact modern healthcare?
    # 3. What factors contribute to sustainable urban development?
    
except Exception as e:
    print(f"Pipeline failed: {e}")
    # Implement recovery strategies or partial results handling
```

## Error Handling Strategies

The pipeline provides multiple approaches for handling generation and parsing errors:

```python
# Strategy 1: Graceful degradation with partial results
try:
    questions = generator.run_open_qa_pipeline(
        n_macro_topics=10,
        n_subtopics=3,
        n_openlines=5,
        n_revisions=3,
        model=model,
        ignore_conversion_failure=True  # Skip failed conversions
    )
    print(f"Successfully generated {len(questions)} questions (some may have been skipped)")
    
except Exception as e:
    print(f"Critical pipeline failure: {e}")

# Strategy 2: Detailed error tracking
from nemo_curator.synthetic.error import YamlConversionError

errors = []
successful_questions = []

for topic in ["Technology", "Health", "Environment"]:
    try:
        responses = generator.generate_open_qa_from_topic(
            topic=topic,
            n_openlines=5,
            model=model
        )
        
        questions = generator.convert_response_to_yaml_list(
            responses[0], 
            model=model
        )
        successful_questions.extend(questions)
        
    except YamlConversionError as e:
        errors.append(f"Parsing error for topic '{topic}': {e}")
    except Exception as e:
        errors.append(f"Generation error for topic '{topic}': {e}")

print(f"Generated {len(successful_questions)} questions successfully")
if errors:
    print(f"Encountered {len(errors)} errors:")
    for error in errors:
        print(f"  - {error}") 