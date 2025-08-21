---
description: "Customize prompt templates for synthetic data generation with built-in and user-defined templates to control LLM behavior"
categories: ["how-to-guides"]
tags: ["prompt-engineering", "customization", "templates", "advanced", "generation-control"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "advanced"
content_type: "how-to"
modality: "text-only"
---

(text-gen-data-pipelines-customizing-prompts)=
# Customizing Prompt Templates

This guide shows how to customize prompt templates in NeMo Curator's synthetic data generation pipelines. Prompt templates are strings with placeholders that control how the LLM generates synthetic data. You can use default templates, select from alternative templates, or create completely custom templates for more targeted data generation.

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

## Understanding Prompt Templates

### What are Prompt Templates

Prompt templates are format strings with placeholders for variables like `{topic}`, `{n_openlines}`, or `{text_material_type}`. These placeholders get populated with actual values when generating synthetic data.

For example, the default writing task template:
```python
DEFAULT_WRITING_TASK_PROMPT_TEMPLATE = (
    'Can you generate {n_openlines} tasks, each of which requires to create a "{text_material_type}" related to {topic}? '
    'Each task should be concise and include one or two sentences only. The tasks should be as diverse as possible. '
    'Your answer should be a list of tasks.'
)
```

### Available Default Templates

NeMo Curator provides default templates for all generation types:

```python
from nemo_curator.synthetic.prompts import (
    DEFAULT_WRITING_TASK_PROMPT_TEMPLATE,
    DEFAULT_OPEN_QA_FROM_TOPICS_PROMPT_TEMPLATE,
    DEFAULT_MATH_MACRO_TOPICS_PROMPT_TEMPLATE,
    PYTHON_PROBLEM_BEGINNER_PROMPT_TEMPLATE,
    PYTHON_PROBLEM_ADVANCED_PROMPT_TEMPLATE,
    # ... and many more
)
```

```{tip}
See all available templates in `nemo_curator.synthetic.prompts`.
```

### Template Parameters and Placeholders

Each generation method requires specific parameters in its template. Common parameters include:

- `{topic}` - The subject matter for generation
- `{n_openlines}` - Number of items to generate
- `{text_material_type}` - Type of writing (essay, poem, etc.)
- `{school_level}` - Educational level for math problems
- `{language}` - Programming language for coding problems
- `{document}` - Source document for closed Q&A

## Customization Methods

### Using Default Templates

Most generation methods work out-of-the-box with default templates:

```python
model = "mistralai/mixtral-8x7b-instruct-v0.1"

# Uses DEFAULT_WRITING_TASK_PROMPT_TEMPLATE automatically
writing_tasks = generator.generate_writing_tasks(
    topic="Climate Change",
    text_material_type="Essays",
    n_openlines=5,
    model=model,
)
```

### Selecting Alternative Templates

Choose from pre-built alternatives for different complexity levels:

```python
from nemo_curator.synthetic import (
    PYTHON_PROBLEM_BEGINNER_PROMPT_TEMPLATE,
    PYTHON_PROBLEM_INTERMEDIATE_PROMPT_TEMPLATE,
    PYTHON_PROBLEM_ADVANCED_PROMPT_TEMPLATE
)

# Use advanced template for expert-level problems
advanced_problems = generator.generate_python_problem(
    topic="Machine Learning Algorithms",
    n_openlines=3,
    model=model,
    prompt_template=PYTHON_PROBLEM_ADVANCED_PROMPT_TEMPLATE,
)
```

### Creating Custom Templates

Define your own templates with custom requirements and formatting:

```python
# Custom template with specific constraints
custom_writing_template = (
    "Generate {n_openlines} {text_material_type} about {topic}. "
    "Each should be exactly 2 paragraphs long, include at least one statistic, "
    "and be written for a {audience} audience. "
    "Format as a numbered list."
)

# Use custom template with additional parameters
custom_tasks = generator.generate_writing_tasks(
    topic="Renewable Energy",
    text_material_type="blog posts",
    n_openlines=3,
    model=model,
    prompt_template=custom_writing_template,
    prompt_kwargs={"audience": "high school students"},
)
```

### Advanced Template Techniques

#### Conditional Logic in Templates
```python
math_template_with_difficulty = (
    "Generate {n_openlines} mathematics problems about {topic}. "
    "{'Make them challenging for advanced students.' if difficulty == 'hard' else 'Keep them accessible for beginners.'} "
    "Include step-by-step solutions."
)

problems = generator.generate_math_problem(
    topic="Calculus",
    n_openlines=2,
    model=model,
    prompt_template=math_template_with_difficulty,
    prompt_kwargs={"difficulty": "hard"},
)
```

#### Multi-language Templates
```python
multilingual_template = (
    "Generate {n_openlines} coding problems about {topic} in {language}. "
    "Provide the problem statement in {human_language}. "
    "Include example input/output and explain the algorithm."
)

multilingual_problems = generator.generate_python_problem(
    topic="Data Structures",
    n_openlines=2,
    model=model,
    prompt_template=multilingual_template,
    prompt_kwargs={"human_language": "Spanish"},
)
```

## Practical Examples

### Customizing Writing Task Prompts

#### Creative Writing Focus
```python
creative_template = (
    "Create {n_openlines} imaginative {text_material_type} prompts about {topic}. "
    "Each should spark creativity, include sensory details, and suggest an emotional tone. "
    "Avoid clichés and encourage original thinking."
)

creative_prompts = generator.generate_writing_tasks(
    topic="Urban Exploration",
    text_material_type="short story",
    n_openlines=3,
    model=model,
    prompt_template=creative_template,
)
```

#### Academic Writing Focus
```python
academic_template = (
    "Generate {n_openlines} academic {text_material_type} topics about {topic}. "
    "Each should require research, critical analysis, and evidence-based arguments. "
    "Include specific research questions and methodology suggestions."
)

academic_prompts = generator.generate_writing_tasks(
    topic="Climate Policy",
    text_material_type="research paper",
    n_openlines=2,
    model=model,
    prompt_template=academic_template,
)
```

### Customizing Math Problem Prompts

#### Real-world Application Focus
```python
applied_math_template = (
    "Create {n_openlines} real-world problems involving {topic}. "
    "Each problem should relate to practical situations like engineering, finance, or daily life. "
    "Include realistic data and explain why the math concept is useful."
)

applied_problems = generator.generate_math_problem(
    topic="Statistics",
    n_openlines=3,
    model=model,
    prompt_template=applied_math_template,
)
```

### Customizing Q&A Prompts

#### Socratic Method Style
```python
socratic_template = (
    "Based on the topic '{topic}', generate {n_openlines} thought-provoking questions "
    "that encourage deep thinking and self-discovery. Use the Socratic method: "
    "ask questions that lead to more questions rather than simple factual answers."
)

socratic_questions = generator.generate_open_qa_from_topic(
    topic="Ethics in AI",
    n_openlines=4,
    model=model,
    prompt_template=socratic_template,
)
```

## Asynchronous Generation with Custom Templates

All customization techniques work with asynchronous generation for better performance:

```python
from openai import AsyncOpenAI
from nemo_curator import AsyncOpenAIClient
from nemo_curator.synthetic import AsyncNemotronGenerator

# Setup async client
async_openai_client = AsyncOpenAI(
    base_url="https://integrate.api.nvidia.com/v1", 
    api_key="<insert NVIDIA API key>"
)
async_client = AsyncOpenAIClient(async_openai_client)
async_generator = AsyncNemotronGenerator(async_client, max_concurrent_requests=10)

# Use custom template asynchronously
custom_template = "Your custom template here with {placeholders}..."

results = await async_generator.generate_writing_tasks(
    topic="Your Topic",
    text_material_type="Your Type",
    n_openlines=5,
    model=model,
    prompt_template=custom_template,
    prompt_kwargs={"custom_param": "value"},
)
```

## See Also

For complete pipeline examples and end-to-end workflows:

- {doc}`open-qa` - Open Q&A generation pipeline
- {doc}`writing-task` - Writing task generation pipeline  
- {doc}`closed-qa` - Closed Q&A generation pipeline
- {doc}`math` - Math problem generation pipeline
- {doc}`asynchronous` - Asynchronous generation techniques 