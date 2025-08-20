---
description: "Generate math questions for dialogue data with macro topics, subtopics, and problems at various educational levels"
categories: ["how-to-guides"]
tags: ["math", "education", "problem-generation", "dialogue-data", "educational-levels", "nemotron"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "text-only"
---

(text-gen-data-pipelines-math)=
# Math Generation Pipeline

This pipeline generates math questions for dialogue data, as used in Nemotron-4 340B. It creates structured mathematical problems targeted at specific educational levels, following a three-step process: macro topic generation, subtopic development, and problem creation.

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
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 500,
}
```

### Generate Math Questions Step by Step

Use the generator to create math problems through the three-step process:

```python
from nemo_curator.synthetic.exceptions import YamlConversionError

try:
    # Step 1: Generate macro topics
    macro_topic_responses = generator.generate_math_macro_topics(
        n_macro_topics=5,
        school_level="university",
        model=model,
        model_kwargs=model_kwargs
    )
    macro_topics = generator.convert_response_to_yaml_list(
        macro_topic_responses[0],
        model=model
    )
    
    print(f"Generated {len(macro_topics)} macro topics:")
    for i, topic in enumerate(macro_topics, 1):
        print(f"{i}. {topic}")

    # Step 2: Generate subtopics
    subtopic_responses = generator.generate_math_subtopics(
        macro_topic=macro_topics[0],
        n_subtopics=3,
        model=model,
        model_kwargs=model_kwargs
    )
    subtopics = generator.convert_response_to_yaml_list(
        subtopic_responses[0],
        model=model
    )
    
    print(f"\nGenerated {len(subtopics)} subtopics for '{macro_topics[0]}':")
    for i, subtopic in enumerate(subtopics, 1):
        print(f"{i}. {subtopic}")

    # Step 3: Generate math problems
    combined_topics = macro_topics + subtopics
    question_responses = generator.generate_math_problem(
        topic=combined_topics[0],
        n_openlines=3,
        model=model,
        model_kwargs=model_kwargs
    )
    math_questions = generator.convert_response_to_yaml_list(
        question_responses[0],
        model=model
    )
    
    print(f"\nGenerated {len(math_questions)} math problems:")
    for i, question in enumerate(math_questions, 1):
        print(f"{i}. {question}")

except YamlConversionError as e:
    print(f"Error converting LLM response to structured format: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Advanced Configuration

Customize the math generation process with different education levels and advanced parameters:

```python
# Import prompt templates for customization
from nemo_curator.synthetic.prompts import (
    DEFAULT_MATH_MACRO_TOPICS_PROMPT_TEMPLATE,
    DEFAULT_MATH_SUBTOPICS_PROMPT_TEMPLATE,
    MATH_PROBLEM_GENERAL_PROMPT_TEMPLATE,
    MATH_PROBLEM_BEGINNER_PROMPT_TEMPLATE
)

# Configure advanced model parameters
advanced_model_kwargs = {
    "temperature": 0.5,  # Lower temperature for more consistent problems
    "top_p": 0.95,
    "max_tokens": 800,
    "seed": 42  # For reproducible results
}

# Generate problems for different education levels
education_levels = ["elementary", "middle school", "high school", "university"]

for level in education_levels:
    print(f"\n=== {level.upper()} LEVEL MATH TOPICS ===")
    
    try:
        macro_responses = generator.generate_math_macro_topics(
            n_macro_topics=3,
            school_level=level,
            model=model,
            prompt_template=DEFAULT_MATH_MACRO_TOPICS_PROMPT_TEMPLATE,
            model_kwargs=advanced_model_kwargs
        )
        
        topics = generator.convert_response_to_yaml_list(
            macro_responses[0],
            model=model
        )
        
        for i, topic in enumerate(topics, 1):
            print(f"{i}. {topic}")
            
    except Exception as e:
        print(f"Error generating topics for {level}: {e}")
```

## End-to-End Pipeline

For a complete automated workflow, use the `run_math_pipeline` method:

```python
try:
    # Complete pipeline execution
    math_questions = generator.run_math_pipeline(
        n_macro_topics=10,
        school_level="high school",
        n_subtopics=5,
        n_openlines=3,
        model=model,
        model_kwargs=model_kwargs,
        ignore_conversion_failure=True  # Continue on conversion errors
    )
    
    print(f"Generated {len(math_questions)} total math questions")
    print("\nSample questions:")
    for i, question in enumerate(math_questions[:5], 1):
        print(f"{i}. {question}")
    
    # Example output:
    # Generated 150 total math questions
    # Sample questions:
    # 1. Prove that the square root of 2 is irrational.
    # 2. Find the derivative of f(x) = x³ + 2x² - 5x + 1.
    # 3. Solve the system of equations: 2x + 3y = 7, x - y = 1.
    
except YamlConversionError as e:
    print(f"Pipeline conversion error: {e}")
except Exception as e:
    print(f"Pipeline error: {e}")
```

## Batch Processing

For processing multiple education levels or topics efficiently:

```python
# Configuration for multiple education levels
education_configs = [
    {"level": "elementary", "n_macro_topics": 5, "n_subtopics": 3, "n_openlines": 2},
    {"level": "middle school", "n_macro_topics": 8, "n_subtopics": 4, "n_openlines": 3},
    {"level": "high school", "n_macro_topics": 10, "n_subtopics": 5, "n_openlines": 4},
    {"level": "university", "n_macro_topics": 12, "n_subtopics": 6, "n_openlines": 5}
]

# Batch process different education levels
math_results = []
for config in education_configs:
    try:
        questions = generator.run_math_pipeline(
            n_macro_topics=config["n_macro_topics"],
            school_level=config["level"],
            n_subtopics=config["n_subtopics"],
            n_openlines=config["n_openlines"],
            model=model,
            model_kwargs=model_kwargs,
            ignore_conversion_failure=True
        )
        
        math_results.append({
            "education_level": config["level"],
            "total_questions": len(questions),
            "sample_questions": questions[:3],
            "config": config
        })
        
    except Exception as e:
        print(f"Error processing {config['level']}: {e}")

# Display results
print("Math Question Generation Results:")
for result in math_results:
    print(f"\n{result['education_level'].upper()} LEVEL:")
    print(f"Total questions generated: {result['total_questions']}")
    print("Sample questions:")
    for i, question in enumerate(result['sample_questions'], 1):
        print(f"  {i}. {question}")
```

## Comprehensive Math Content Workflow

Combine math generation with different problem types and difficulty levels:

```python
def generate_comprehensive_math_content(generator, topic, model, model_kwargs):
    """Generate various types of math content for a given topic."""
    
    # Generate general math problems
    general_problems = generator.generate_math_problem(
        topic=topic,
        n_openlines=3,
        model=model,
        prompt_template=MATH_PROBLEM_GENERAL_PROMPT_TEMPLATE,
        model_kwargs=model_kwargs
    )
    
    # Generate beginner-friendly problems
    beginner_problems = generator.generate_math_problem(
        topic=topic,
        n_openlines=3,
        model=model,
        prompt_template=MATH_PROBLEM_BEGINNER_PROMPT_TEMPLATE,
        model_kwargs=model_kwargs
    )
    
    try:
        general_questions = generator.convert_response_to_yaml_list(
            general_problems[0], model=model
        )
        beginner_questions = generator.convert_response_to_yaml_list(
            beginner_problems[0], model=model
        )
        
        return {
            "topic": topic,
            "general_problems": general_questions,
            "beginner_problems": beginner_questions,
            "total_problems": len(general_questions) + len(beginner_questions)
        }
        
    except YamlConversionError as e:
        print(f"Conversion error for topic '{topic}': {e}")
        return None

# Example usage
sample_topics = ["calculus", "linear algebra", "probability", "geometry"]

comprehensive_results = []
for topic in sample_topics:
    result = generate_comprehensive_math_content(
        generator, topic, model, model_kwargs
    )
    if result:
        comprehensive_results.append(result)

# Display comprehensive results
print("Comprehensive Math Content Generation:")
for result in comprehensive_results:
    print(f"\n=== {result['topic'].upper()} ===")
    print(f"Total problems: {result['total_problems']}")
    
    print("\nGeneral Problems:")
    for i, problem in enumerate(result['general_problems'], 1):
        print(f"  {i}. {problem}")
    
    print("\nBeginner Problems:")
    for i, problem in enumerate(result['beginner_problems'], 1):
        print(f"  {i}. {problem}")
```

## Error Handling

The math generation pipeline can raise `YamlConversionError` when the LLM response can't be parsed into the expected YAML list format. Always wrap pipeline calls in try-catch blocks:

```python
from nemo_curator.synthetic.exceptions import YamlConversionError

try:
    result = generator.convert_response_to_yaml_list(response[0], model=model)
except YamlConversionError:
    # Handle conversion failure - response may need manual review
    print("Failed to parse LLM response - check response format")
```

## Configuration Options

The pipeline supports several configuration parameters:

- `n_macro_topics`: Number of high-level math topics to generate
- `school_level`: Target education level ("elementary," "middle school," "high school," "university")
- `n_subtopics`: Number of subtopics per macro topic
- `n_openlines`: Number of questions to generate per topic
- `model`: LLM model identifier
- `ignore_conversion_failure`: Set to `True` to skip failed YAML conversions instead of raising errors
- `combine_topics`: Whether to mix macro topics with subtopics when generating problems
- `prompt_template`: Custom prompt template for specialized problem types 