---
description: "Release notes and version history for NeMo Curator platform updates and new features"
categories: ["reference"]
tags: ["release-notes", "changelog", "updates"]
personas: ["data-scientist-focused", "mle-focused", "admin-focused", "devops-focused"]
difficulty: "reference"
content_type: "reference"
modality: "universal"
---

(about-release-notes)=

# NeMo Curator Release Notes: {{ current_release }}

## Synthetic Data Generation

New Ray-based synthetic data generation capabilities for creating and augmenting training data using LLMs:

- **LLM Client Infrastructure**: OpenAI-compatible async/sync clients with automatic rate limiting, retry logic, and exponential backoff
- **Multilingual Q&A Generation**: Generate synthetic Q&A pairs across multiple languages using customizable prompts
- **NemotronCC Pipelines**: Advanced text transformation and knowledge extraction workflows:
  - **Wikipedia Paraphrasing**: Improve low-quality text by rewriting in Wikipedia-style prose
  - **Diverse QA**: Generate diverse question-answer pairs for reading comprehension training
  - **Distill**: Create condensed, information-dense paraphrases preserving key concepts
  - **Extract Knowledge**: Extract factual content as textbook-style passages
  - **Knowledge List**: Extract structured fact lists from documents

Learn more in the [Synthetic Data Generation documentation](../../curate-text/synthetic/index.md).


---

## What's Next

The next release will focus on ...

```{toctree}
:hidden:
:maxdepth: 4

Migration Guide <migration-guide>
Migration FAQ <migration-faq>

```
