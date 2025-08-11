---
description: "Generate synthetic text data using large language models for pre-training, fine-tuning, and evaluation tasks with comprehensive pipeline support"
categories: ["workflows"]
tags: ["synthetic-data", "data-generation", "llm", "pipelines", "openai", "nemotron"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "workflow"
modality: "text-only"
---

(generate-data)=
# Generate Data

Generate synthetic text data using large language models (LLMs) for pre-training, fine-tuning, and evaluation tasks. Create high-quality training data for low-resource languages and domains, or perform knowledge distillation from existing models.

## How it Works

NeMo Curator's synthetic data generation capabilities are organized into several components:

1. **Model Integration**: Connect to OpenAI-compatible model endpoints or self-hosted models
2. **Generation Pipelines**: Use pre-built pipelines for common generation tasks
3. **Custom Workflows**: Combine components to create specialized generation pipelines
4. **Quality Control**: Filter and validate generated data using NeMo Curator's processing tools

---

## Service Connections

Connect your data generation workflows to powerful language models and scoring services. Choose from cloud-based APIs or deploy models in your own infrastructure.

::::{grid} 2 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`link-external;1.5em;sd-mr-1` OpenAI Integration
:link: text-generate-data-connect-service-openai
:link-type: ref
Connect to OpenAI's API endpoints for GPT models and other services
+++
{bdg-secondary}`openai`
{bdg-secondary}`gpt`
{bdg-secondary}`api`
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` NeMo Deploy Integration
:link: text-generate-data-connect-service-nemo-deploy
:link-type: ref
Deploy and connect to models using NVIDIA NeMo Deploy
+++
{bdg-secondary}`nemo-deploy`
{bdg-secondary}`self-hosted`
{bdg-secondary}`deployment`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Reward Model Integration
:link: text-generate-data-connect-service-reward-models
:link-type: ref
Integrate reward models for quality scoring and filtering
+++
{bdg-secondary}`reward-model`
{bdg-secondary}`quality`
{bdg-secondary}`scoring`
:::

::::

## Generation Pipelines

Transform your data needs into production-ready synthetic datasets using specialized generation pipelines.

### Q&A Generation Pipelines

Use these pipelines to generate question-and-answer data for training, evaluation, and comprehension tasks.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`repo-template;1.5em;sd-mr-1` Closed Q&A Generation Pipeline
:link: text-gen-data-pipelines-closed-qa
:link-type: ref
Generate closed-ended questions about a given document. Ideal for creating evaluation or comprehension datasets.
+++
{bdg-secondary}`closed-qa`
{bdg-secondary}`document`
:::

:::{grid-item-card} {octicon}`repo-template;1.5em;sd-mr-1` Open Q&A Generation Pipeline
:link: text-gen-data-pipelines-open-qa
:link-type: ref
Generate open-ended questions ("openlines") for dialogue data, including macro topics, subtopics, and detailed revisions.
+++
{bdg-secondary}`open-qa`
{bdg-secondary}`question-generation`
:::

:::{grid-item-card} {octicon}`repo-template;1.5em;sd-mr-1` Diverse QA Generation Pipeline
:link: text-gen-data-pipelines-diverse-qa
:link-type: ref
Generate diverse question-answer pairs from documents for QA datasets.
+++
{bdg-secondary}`qa-pairs`
{bdg-secondary}`diverse`
:::

::::

### Content Transformation & Summarization

Transform, rewrite, and summarize documents to create clear, concise, and structured text data.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`repo-template;1.5em;sd-mr-1` Wikipedia Style Rewrite Pipeline
:link: text-gen-data-pipelines-wikipedia
:link-type: ref
Rewrite documents into a style similar to Wikipedia, improving clarity and scholarly tone.
+++
{bdg-secondary}`wikipedia`
{bdg-secondary}`rewrite`
:::

:::{grid-item-card} {octicon}`repo-template;1.5em;sd-mr-1` Distillation Pipeline
:link: text-gen-data-pipelines-distillation
:link-type: ref
Distill documents to concise summaries, removing redundancy and focusing on key information.
+++
{bdg-secondary}`distillation`
{bdg-secondary}`summarization`
:::

:::{grid-item-card} {octicon}`repo-template;1.5em;sd-mr-1` Knowledge Extraction Pipeline
:link: text-gen-data-pipelines-extract-knowledge
:link-type: ref
Extract key knowledge and facts from documents for summarization and analysis.
+++
{bdg-secondary}`knowledge`
{bdg-secondary}`extraction`
:::

:::{grid-item-card} {octicon}`repo-template;1.5em;sd-mr-1` Knowledge List Generation Pipeline
:link: text-gen-data-pipelines-knowledge-list
:link-type: ref
Extract structured knowledge lists from documents for downstream use.
+++
{bdg-secondary}`knowledge-lists`
{bdg-secondary}`extraction`
:::

::::

### Dialogue & Writing

Create synthetic dialogues and writing tasks to support conversational and creative data generation.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`repo-template;1.5em;sd-mr-1` Dialogue Generation Pipeline
:link: text-gen-data-pipelines-dialogue
:link-type: ref
Generate multi-turn dialogues and two-turn prompts for preference data. Synthesize conversations where an LLM plays both user and assistant.
+++
{bdg-secondary}`dialogue`
{bdg-secondary}`multi-turn`
:::

:::{grid-item-card} {octicon}`repo-template;1.5em;sd-mr-1` Writing Task Generation Pipeline
:link: text-gen-data-pipelines-writing-task
:link-type: ref
Generate writing prompts (essays, poems, etc.) and revise them for detail and diversity. Useful for creative and instructional datasets.
+++
{bdg-secondary}`writing`
{bdg-secondary}`creative`
:::

::::

### STEM & Coding

Generate math and coding problems, as well as classify entities for STEM-related datasets.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`repo-template;1.5em;sd-mr-1` Math Generation Pipeline
:link: text-gen-data-pipelines-math
:link-type: ref
Generate math questions for dialogue data, including macro topics, subtopics, and problems at various school levels.
+++
{bdg-secondary}`math`
{bdg-secondary}`education`
:::

:::{grid-item-card} {octicon}`repo-template;1.5em;sd-mr-1` Python Generation Pipeline
:link: text-gen-data-pipelines-python
:link-type: ref
Generate Python coding problems for dialogue data, including macro topics, subtopics, and problems for various skill levels.
+++
{bdg-secondary}`python`
{bdg-secondary}`coding`
:::

:::{grid-item-card} {octicon}`repo-template;1.5em;sd-mr-1` Entity Classification Pipeline
:link: text-gen-data-pipelines-entity-classification
:link-type: ref
Classify entities (for example, Wikipedia entries) as math- or Python-related using an LLM. Useful for filtering or labeling data for downstream tasks.
+++
{bdg-secondary}`entity-classification`
{bdg-secondary}`math`
{bdg-secondary}`python`
:::

::::

### Infrastructure & Customization

Leverage asynchronous pipelines and customizable prompts to scale and tailor your data generation workflows.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`repo-template;1.5em;sd-mr-1` Asynchronous Generation Pipeline
:link: text-gen-data-pipelines-async
:link-type: ref
Generate synthetic data in parallel using asynchronous pipelines for maximum efficiency. Ideal for large-scale prompt generation and working with rate-limited LLM APIs. Provides async alternatives to all major text data generation pipelines in NeMo Curator.
+++
{bdg-secondary}`async`
{bdg-secondary}`parallel`
{bdg-secondary}`LLM`
:::

::::

## Integrations

Combine generation with powerful filtering and processing capabilities.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`git-merge;1.5em;sd-mr-1` Integration with NeMo Curator
:link: integration
:link-type: doc
Combine synthetic data generation with other NeMo Curator modules for filtering and processing
+++
{bdg-secondary}`filtering`
{bdg-secondary}`processing`
{bdg-secondary}`pipeline`
:::

::::

```{toctree}
:maxdepth: 2
:titlesonly:
:hidden:

Services <connect-service/index>
Pipelines <pipelines/index>
Integration with NeMo Curator <integration>
```
