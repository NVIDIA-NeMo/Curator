---
description: "Reference documentation for Nemotron-CC synthetic data generation tasks and stages"
categories: ["reference"]
tags: ["nemotron-cc", "stages", "api-reference"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "advanced"
content_type: "reference"
modality: "text-only"
---

(nemotron-cc-tasks)=
# Nemotron-CC Task Reference

This reference documents each Nemotron-CC synthetic data generation stage, including prompt templates, configuration options, and post-processing details.

:::{note}
**System Prompt Usage**: Not all stages use a system prompt.
- `WikipediaParaphrasingStage` and `DistillStage` include system prompts
- `DiverseQAStage`, `ExtractKnowledgeStage`, and `KnowledgeListStage` use only user prompts (no system prompt)
:::

## WikipediaParaphrasingStage

Rewrites low-quality text in Wikipedia-style prose, improving readability and structure.

### Purpose

Transform noisy or poorly-written web data into high-quality, encyclopedic text suitable for training language models.

### Configuration

```python
from nemo_curator.stages.synthetic.nemotron_cc.nemotron_cc import WikipediaParaphrasingStage

stage = WikipediaParaphrasingStage(
    client=llm_client,
    model_name="meta/llama-3.3-70b-instruct",
    generation_config=generation_config,
    input_field="text",
    output_field="rephrased",
)
```

### Prompt Template

The stage uses a system prompt establishing the assistant persona and a user prompt requesting paraphrasing:

```text
System: A chat between a curious user and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the questions.

User: For the following paragraph give me a diverse paraphrase of the same in
high quality English language as in sentences on Wikipedia. Begin your answer
on a separate line with "Here is a paraphrased version:".

Text: {document}
```

Refer to the [full prompt in source](https://github.com/NVIDIA-NeMo/Curator/blob/main/nemo_curator/stages/synthetic/nemotron_cc/prompts.py).

### Post-Processing

The Wikipedia post-processing pipeline:
1. Filters by token count (max 510 tokens)
2. Removes markdown formatting
3. Validates prefix "Here is a paraphrased version:"
4. Removes the prefix from output
5. Removes quotation marks
6. Joins document segments
7. Filters documents below 50 tokens

---

## DiverseQAStage

Generates diverse question-answer pairs from document content.

### Purpose

Create reading comprehension training data with varied question types and cognitive complexity levels.

### Configuration

```python
from nemo_curator.stages.synthetic.nemotron_cc.nemotron_cc import DiverseQAStage

stage = DiverseQAStage(
    client=llm_client,
    model_name="meta/llama-3.3-70b-instruct",
    generation_config=generation_config,
    input_field="text",
    output_field="diverse_qa",
)
```

### Prompt Template

The stage requests up to eight diverse Q&A pairs with specific formatting:

```text
Task: Read the text, ask questions and answer them.

Follow these instructions:
1. Ask diverse questions that require different cognitive skills
2. Ask questions in various forms:
   - Yes/No questions
   - Open-ended questions (what, how, when, where, why, who)
   - Multi-choice questions with options
   - Comparison questions
   - Reading comprehension questions
   - Problem-solving questions
3. Focus on factual information and key concepts
4. Use clear and concise language
5. Use plain text (no Markdown)
6. Format: Question: [question] Answer: [answer]

Text: {document}
```

### Post-Processing with DiverseQAPostProcessingStage

The `DiverseQAPostProcessingStage` performs specialized parsing:

```python
from nemo_curator.stages.synthetic.nemotron_cc.nemotron_cc import DiverseQAPostProcessingStage

post_stage = DiverseQAPostProcessingStage(
    input_field="text",
    qa_field="diverse_qa",
    tokenizer=tokenizer,  # For length-based sampling
    prefix="Here are the questions and answers based on the provided text:",
    max_num_pairs=10,
)
```

**Post-processing logic:**
1. Parse Q&A pairs from bullet-formatted output
2. Merge question and answer lines
3. Shuffle pairs randomly
4. Sample pairs based on input document length (using tokenizer)
5. Concatenate original document with selected Q&A pairs

The number of Q&A pairs sampled is proportional to input length:
```python
num_pairs = random.randint(1, max(1, int(max_num_pairs * num_tokens / 150)))
```

---

## DistillStage

Creates condensed, information-dense paraphrases while preserving key concepts.

### Purpose

Generate training data that captures essential knowledge in a more accessible format, suitable for knowledge distillation.

### Configuration

```python
from nemo_curator.stages.synthetic.nemotron_cc.nemotron_cc import DistillStage

stage = DistillStage(
    client=llm_client,
    model_name="meta/llama-3.3-70b-instruct",
    generation_config=generation_config,
    input_field="text",
    output_field="distill",
)
```

### Prompt Template

```text
System: You are an artificial intelligence assistant. You carefully provide
accurate, factual, thoughtful, nuanced answers, and are brilliant at reasoning.

User: Your task is to read and paraphrase the provided text following these instructions:
- Create a condensed but accurate and informative version
- Preserve crucial information, key concepts, important values, factual details
- Retain technical terms and specialized vocabulary
- Retain examples and explanations of reasoning
- Only include information present in the original text
- Write in plain text without formatting

Text: {document}

Task: Paraphrase in high-quality English. Begin with "Paraphrased Text:".
```

### Post-Processing

1. Filter by token count (max 1598 tokens)
2. Remove markdown formatting
3. Validate "Paraphrased Text:" prefix
4. Remove the prefix
5. Remove quotation marks
6. Filter documents below 50 tokens

---

## ExtractKnowledgeStage

Extracts and rewrites knowledge as textbook-style passages.

### Purpose

Convert raw text into educational-quality passages organized by domain, suitable for building knowledge bases.

### Configuration

```python
from nemo_curator.stages.synthetic.nemotron_cc.nemotron_cc import ExtractKnowledgeStage

stage = ExtractKnowledgeStage(
    client=llm_client,
    model_name="meta/llama-3.3-70b-instruct",
    generation_config=generation_config,
    input_field="text",
    output_field="extract_knowledge",
)
```

### Prompt Template

```text
Your task is to rewrite knowledge from the provided text following these instructions:
- Rewrite as passages using easy-to-understand, high-quality English
  like sentences in textbooks and Wikipedia
- Focus on content in disciplines: humanities, social sciences, natural sciences,
  technology, engineering, math, law, business, management, art, education,
  agricultural sciences, politics, and history
- Disregard content without useful facts or knowledge
- Retain examples and supporting evidence
- Do not add or alter details
- Write in plain text
- Do not add titles or comments

Text: {document}

Task: Rewrite facts and knowledge as passages following the instructions.
```

### Post-Processing

1. Filter by token count (max 1398 tokens)
2. Remove markdown formatting
3. Remove passage labels ("Passage:", "Passage 1:", etc.)
4. Filter documents below 50 tokens

---

## KnowledgeListStage

Extracts structured fact lists from documents.

### Purpose

Generate bullet-pointed factual content for structured knowledge extraction.

### Configuration

```python
from nemo_curator.stages.synthetic.nemotron_cc.nemotron_cc import KnowledgeListStage

stage = KnowledgeListStage(
    client=llm_client,
    model_name="meta/llama-3.3-70b-instruct",
    generation_config=generation_config,
    input_field="text",
    output_field="knowledge_list",
)
```

### Prompt Template

```text
Review the text and extract the key information. Follow these instructions:
- Provide a concise and organized list of factual information
- Include concrete details, key concepts, and important statistics
- Ensure each point is clear, specific, and supported by the original text
- Ensure extracted text is information-dense
- Do not add titles or headings

Text: {document}

Task: Extract factual information, concrete details, and key concepts.
```

### Post-Processing with KnowledgeListPostProcessingStage

```python
from nemo_curator.stages.synthetic.nemotron_cc.nemotron_cc import KnowledgeListPostProcessingStage

post_stage = KnowledgeListPostProcessingStage(
    input_field="knowledge_list",
)
```

**Post-processing logic:**
1. Skip the first line if it doesn't start with a bullet marker
2. Remove leading bullet markers ("- ") and indentation prefixes ("  ")
3. Join lines with newlines

---

## Customizing Prompts

To use custom prompts while maintaining Nemotron-CC infrastructure, subclass `BaseSyntheticStage`:

```python
from dataclasses import dataclass
from nemo_curator.stages.synthetic.nemotron_cc.base import BaseSyntheticStage


@dataclass
class CustomSyntheticStage(BaseSyntheticStage):
    system_prompt: str = "You are a helpful assistant specialized in..."
    prompt: str = """Your custom prompt template here.

Text: {document}

Instructions: ..."""
    input_field: str = "text"
    output_field: str = "custom_output"

    @property
    def name(self) -> str:
        return "CustomSyntheticStage"
```

The `{document}` placeholder is replaced with the content from `input_field`.

---

## Complete Configuration Example

The following example shows the conceptual configuration structure for Nemotron-CC tasks. For production pipelines, see the [tutorial examples](https://github.com/NVIDIA-NeMo/Curator/tree/main/tutorials/synthetic/nemotron_cc).

```python
TASK_CONFIG = {
    "diverse_qa": {
        "system_prompt": None,  # DiverseQAStage uses no system prompt
        "prompt_template": DIVERSE_QA_PROMPT_TEMPLATE,
        "min_document_tokens": 30,
        "min_segment_tokens": 30,
        "max_input_tokens": 1000,
        "max_output_tokens": 598,
    },
    "distill": {
        "system_prompt": NEMOTRON_CC_DISTILL_SYSTEM_PROMPT,
        "prompt_template": DISTILL_PROMPT_TEMPLATE,
        "min_document_tokens": 30,
        "min_segment_tokens": 10,
        "max_input_tokens": 2000,
        "max_output_tokens": 1598,
    },
    "extract_knowledge": {
        "system_prompt": None,  # ExtractKnowledgeStage uses no system prompt
        "prompt_template": EXTRACT_KNOWLEDGE_PROMPT_TEMPLATE,
        "min_document_tokens": 30,
        "min_segment_tokens": 30,
        "max_input_tokens": 1400,
        "max_output_tokens": 1400,
    },
    "knowledge_list": {
        "system_prompt": None,  # KnowledgeListStage uses no system prompt
        "prompt_template": KNOWLEDGE_LIST_PROMPT_TEMPLATE,
        "min_document_tokens": 30,
        "min_segment_tokens": 30,
        "max_input_tokens": 1000,
        "max_output_tokens": 598,
    },
    "wikipedia_paraphrasing": {
        "system_prompt": NEMOTRON_CC_SYSTEM_PROMPT,
        "prompt_template": WIKIPEDIA_REPHRASING_PROMPT_TEMPLATE,
        "min_document_tokens": 5,
        "min_segment_tokens": 5,
        "max_input_tokens": 512,
        "max_output_tokens": 510,
    },
}

GENERATION_CONFIG = {
    "MAX_INPUT_TOKENS": 2000,
    "MAX_OUTPUT_TOKENS": 1600,
    "TOP_K": 0,
    "TOP_P": 0.9,
    "TEMPERATURE": 0.5,
}
```

---

## Source Code References

- **Prompts**: [`nemo_curator/stages/synthetic/nemotron_cc/prompts.py`](https://github.com/NVIDIA-NeMo/Curator/blob/main/nemo_curator/stages/synthetic/nemotron_cc/prompts.py)
- **Stages**: [`nemo_curator/stages/synthetic/nemotron_cc/nemotron_cc.py`](https://github.com/NVIDIA-NeMo/Curator/blob/main/nemo_curator/stages/synthetic/nemotron_cc/nemotron_cc.py)
- **Base Class**: [`nemo_curator/stages/synthetic/nemotron_cc/base.py`](https://github.com/NVIDIA-NeMo/Curator/blob/main/nemo_curator/stages/synthetic/nemotron_cc/base.py)
- **Pipeline Helpers**: [`tutorials/synthetic/nemotron_cc/nemotron_cc_pipelines.py`](https://github.com/NVIDIA-NeMo/Curator/blob/main/tutorials/synthetic/nemotron_cc/nemotron_cc_pipelines.py)
- **Full Tutorial Examples**: [`tutorials/synthetic/nemotron_cc/`](https://github.com/NVIDIA-NeMo/Curator/tree/main/tutorials/synthetic/nemotron_cc)

