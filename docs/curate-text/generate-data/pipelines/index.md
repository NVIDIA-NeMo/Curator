---
description: "Pre-built pipelines for generating high-quality synthetic text data including Q&A, content transformation, dialogue, and STEM topics"
categories: ["workflows"]
tags: ["pipelines", "qa-generation", "dialogue", "content-transformation", "math", "python", "writing-tasks"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "workflow"
modality: "text-only"
---

(text-generate-data-pipelines)=
# Text Data Generation Pipelines

NeMo Curator provides pre-built pipelines for generating high-quality synthetic text data. These pipelines implement proven approaches for creating training data across different formats and styles.

---

## Q&A Generation Pipelines

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

## Content Transformation & Summarization

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

## Dialogue & Writing

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

## STEM & Coding

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

## Infrastructure & Customization

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

## References

Find additional resources and guidance for customizing prompts and using generation pipelines effectively.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`repo-template;1.5em;sd-mr-1` Customizing Prompts
:link: text-gen-data-pipelines-customizing-prompts
:link-type: ref
Customize prompt templates for any generation step. Use built-in or user-defined templates to control LLM behavior.
+++
{bdg-secondary}`custom`
{bdg-secondary}`prompt-template`
:::

::::

```{toctree}
:maxdepth: 2
:titlesonly:
:hidden:

Asynchronous <asynchronous>
Closed Q&A  <closed-qa>
Dialogue <dialogue>
Distillation <distillation>
Diverse Q&A  <diverse-qa>
Entity Classification <entity-classification>
Knowledge Extraction <extract-knowledge>
Knowledge List  <knowledge-list>
Math <math>
Open Q&A <open-qa>
Python <python>
Wikipedia Style Rewrite  <wikipedia>
Writing Task <writing-task>
Customizing Prompts <customizing-prompts>
```
