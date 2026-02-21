# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

from transformers import AutoTokenizer

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.synthetic.nemotron_cc.nemotron_cc import (
    DiverseQAPostProcessingStage,
    KnowledgeListPostProcessingStage,
)
from nemo_curator.stages.text.filters import Filter, ScoreFilter
from nemo_curator.stages.text.filters.heuristic import SubstringFilter
from nemo_curator.stages.text.filters.token import TokenCountFilter
from nemo_curator.stages.text.modifiers import Modify
from nemo_curator.stages.text.modifiers.string import (
    LineRemover,
    MarkdownRemover,
    QuotationRemover,
    Slicer,
)
from nemo_curator.stages.text.modules.joiner import DocumentJoiner
from nemo_curator.stages.text.modules.splitter import DocumentSplitter


def get_prefix_token_count(tokenizer: AutoTokenizer, system_prompt: str, user_prompt_template: str) -> int:
    """
    Calculate the number of tokens in the prompt prefix.

    This is used to determine how many tokens are taken up by the system prompt
    and user prompt template, so we can calculate the maximum segment size.

    Args:
        tokenizer: The tokenizer to use for counting tokens
        system_prompt: The system prompt string
        user_prompt_template: The user prompt template string (e.g., "Summarize: {text}")

    Returns:
        Number of tokens in the prefix
    """
    # Construct a sample prompt with placeholder text
    # Extract the template without the {text} placeholder
    if "{text}" in user_prompt_template:
        template_without_text = user_prompt_template.replace("{text}", "")
    else:
        template_without_text = user_prompt_template

    # Combine system prompt and template
    full_prefix = f"{system_prompt}\n{template_without_text}"

    # Count tokens
    tokens = tokenizer.encode(full_prefix)
    return len(tokens)


def add_preprocessing_pipeline(  # noqa: PLR0913
    pipeline: Pipeline,
    text_field: str,
    system_prompt: str,
    user_prompt_template: str,
    min_document_tokens: int,
    min_segment_tokens: int,
    max_input_tokens: int,
    args: argparse.Namespace,
) -> Pipeline:
    """
    Add Nemotron-CC preprocessing pipeline.

    This pipeline performs the following operations:
    1. Filter out documents that are too short
    2. Split documents into segments by newline
    3. Filter out segments that are too long for the model
    4. Join adjacent short segments to maximize input utilization
    5. Filter out segments that are still too short after joining

    Args:
        pipeline: The pipeline to add stages to
        text_field: The field containing the text to process
        system_prompt: The system prompt for the LLM
        user_prompt_template: The user prompt template (e.g., "Summarize: {text}")
        min_document_tokens: Minimum tokens for a document to be considered
        min_segment_tokens: Minimum tokens for a segment after joining
        max_input_tokens: Maximum input tokens the model can handle
        args: Command line arguments containing tokenizer info

    Returns:
        Updated pipeline with preprocessing stages added
    """
    # Calculate the maximum segment size accounting for prompt overhead
    prefix_token_count = get_prefix_token_count(args.tokenizer, system_prompt, user_prompt_template)
    max_segment_tokens = max_input_tokens - prefix_token_count - 2  # -2 for safety margin

    # Filter out documents that are too short
    pipeline.add_stage(
        ScoreFilter(
            TokenCountFilter(
                tokenizer=args.tokenizer,
                hf_token=args.hf_token,
                min_tokens=min_document_tokens,
            ),
            text_field=text_field,
            score_field="document_token_count",
        ),
    )

    # Split documents into segments by newline
    pipeline.add_stage(
        DocumentSplitter(
            separator="\n",
            text_field=text_field,
            segment_id_field="segment_id",
        ),
    )

    # Filter out segments that are too long for the model
    pipeline.add_stage(
        ScoreFilter(
            TokenCountFilter(
                tokenizer=args.tokenizer,
                hf_token=args.hf_token,
                max_tokens=max_segment_tokens,
            ),
            text_field=text_field,
            score_field="segment_token_count",
        ),
    )

    # Join adjacent short segments to maximize input utilization
    # This will combine short segments up to max_segment_tokens
    pipeline.add_stage(
        DocumentJoiner(
            separator="\n",
            text_field=text_field,
            segment_id_field="segment_id",
            document_id_field="id",
            max_length=max_segment_tokens,
            length_field="segment_token_count",
            drop_segment_id_field=False,  # Keep segment_id for potential debugging
        ),
    )

    # Filter out segments that are too short even after joining
    pipeline.add_stage(
        Filter(
            filter_fn=lambda x: x >= min_segment_tokens,
            filter_field="segment_token_count",
        ),
    )

    return pipeline


def add_wikipedia_postprocessing_pipeline(
    pipeline: Pipeline, llm_response_field: str, args: argparse.Namespace
) -> Pipeline:
    """
    Add Wikipedia postprocessing pipeline.

    This pipeline performs the following operations:
    1. Filter segments by maximum token count
    2. Remove markdown formatting
    3. Filter documents that don't start with expected prefix
    4. Remove the paraphrase prefix
    5. Remove quotation marks
    6. Join paragraphs belonging to the same document
    7. Filter out documents that are too short

    Args:
        pipeline: The pipeline to add stages to
        llm_response_field: The field containing the LLM response
        args: Command line arguments containing tokenizer info

    Returns:
        Updated pipeline with Wikipedia postprocessing stages added
    """
    max_rephrased_tokens = 510
    min_document_tokens = 50

    # Filter by token count (segment level)
    pipeline.add_stage(
        ScoreFilter(
            TokenCountFilter(
                tokenizer=args.tokenizer,
                hf_token=args.hf_token,
                max_tokens=max_rephrased_tokens,
            ),
            text_field=llm_response_field,
            score_field="rephrased_segment_token_count",
        ),
    )

    # Remove markdown formatting
    pipeline.add_stage(
        Modify(
            modifier_fn=MarkdownRemover(),
            input_fields=llm_response_field,
        ),
    )

    # Remove documents not starting with the specified prefix
    pipeline.add_stage(
        ScoreFilter(
            SubstringFilter(substring="Here is a paraphrased version:", position="prefix"),
            text_field=llm_response_field,
            score_field="substring",
        ),
    )

    # Remove the paraphrase prefix
    pipeline.add_stage(
        Modify(
            modifier_fn=Slicer(
                left="Here is a paraphrased version:",
                include_left=False,
                strip=True,
            ),
            input_fields=llm_response_field,
        ),
    )

    # Remove quotation marks
    pipeline.add_stage(
        Modify(
            modifier_fn=QuotationRemover(),
            input_fields=llm_response_field,
        ),
    )

    # Concat paragraphs belonging to the same document
    pipeline.add_stage(
        DocumentJoiner(
            separator="\n",
            text_field=llm_response_field,
            segment_id_field="segment_id",
            document_id_field="id",
            drop_segment_id_field=False,
        ),
    )

    # Filter out documents that are too short (document level)
    pipeline.add_stage(
        ScoreFilter(
            TokenCountFilter(
                tokenizer=args.tokenizer,
                hf_token=args.hf_token,
                min_tokens=min_document_tokens,
            ),
            text_field=llm_response_field,
            score_field="rephrased_document_token_count",
        ),
    )

    return pipeline


def add_diverse_qa_postprocessing_pipeline(
    pipeline: Pipeline, llm_response_field: str, args: argparse.Namespace
) -> Pipeline:
    """Add DiverseQA postprocessing pipeline."""
    max_rephrased_tokens = 598
    min_document_tokens = 100

    # Filter by token count (segment level)
    pipeline.add_stage(
        ScoreFilter(
            TokenCountFilter(
                tokenizer=args.tokenizer,
                hf_token=args.hf_token,
                max_tokens=max_rephrased_tokens,
            ),
            text_field=llm_response_field,
            score_field="rephrased_segment_token_count",
        ),
    )

    # Remove markdown formatting
    pipeline.add_stage(
        Modify(
            modifier_fn=MarkdownRemover(),
            input_fields=llm_response_field,
        ),
    )

    # Reformat QA pairs
    pipeline.add_stage(
        DiverseQAPostProcessingStage(
            input_field="text",
            qa_field=llm_response_field,
            tokenizer=args.tokenizer,
        ),
    )

    # Filter out documents that are too short (document level)
    pipeline.add_stage(
        ScoreFilter(
            TokenCountFilter(
                tokenizer=args.tokenizer,
                hf_token=args.hf_token,
                min_tokens=min_document_tokens,
            ),
            text_field=llm_response_field,
            score_field="rephrased_document_token_count",
        ),
    )

    return pipeline


def add_distill_postprocessing_pipeline(
    pipeline: Pipeline, llm_response_field: str, args: argparse.Namespace
) -> Pipeline:
    """Add Distill postprocessing pipeline."""
    max_rephrased_tokens = 1598
    min_document_tokens = 50

    # Filter by token count (segment level)
    pipeline.add_stage(
        ScoreFilter(
            TokenCountFilter(
                tokenizer=args.tokenizer,
                hf_token=args.hf_token,
                max_tokens=max_rephrased_tokens,
            ),
            text_field=llm_response_field,
            score_field="rephrased_segment_token_count",
        ),
    )

    # Remove markdown formatting
    pipeline.add_stage(
        Modify(
            modifier_fn=MarkdownRemover(),
            input_fields=llm_response_field,
        ),
    )

    # Remove documents not starting with the specified prefix
    pipeline.add_stage(
        ScoreFilter(
            SubstringFilter(substring="Paraphrased Text:", position="prefix"),
            text_field=llm_response_field,
            score_field="substring",
        ),
    )

    # Remove the paraphrase prefix
    pipeline.add_stage(
        Modify(
            modifier_fn=Slicer(
                left="Paraphrased Text:",
                include_left=False,
                strip=True,
            ),
            input_fields=llm_response_field,
        ),
    )

    # Remove quotation marks
    pipeline.add_stage(
        Modify(
            modifier_fn=QuotationRemover(),
            input_fields=llm_response_field,
        ),
    )

    # Filter out documents that are too short
    pipeline.add_stage(
        ScoreFilter(
            TokenCountFilter(
                tokenizer=args.tokenizer,
                hf_token=args.hf_token,
                min_tokens=min_document_tokens,
            ),
            text_field=llm_response_field,
            score_field="rephrased_document_token_count",
        ),
    )

    return pipeline


def add_extract_knowledge_postprocessing_pipeline(
    pipeline: Pipeline, llm_response_field: str, args: argparse.Namespace
) -> Pipeline:
    """Add ExtractKnowledge postprocessing pipeline."""
    max_rephrased_tokens = 1398
    min_document_tokens = 50

    # Filter by token count (segment level)
    pipeline.add_stage(
        ScoreFilter(
            TokenCountFilter(
                tokenizer=args.tokenizer,
                hf_token=args.hf_token,
                max_tokens=max_rephrased_tokens,
            ),
            text_field=llm_response_field,
            score_field="rephrased_segment_token_count",
        ),
    )

    # Remove markdown formatting
    pipeline.add_stage(
        Modify(
            modifier_fn=MarkdownRemover(),
            input_fields=llm_response_field,
        ),
    )

    # Remove passage lines
    pipeline.add_stage(
        Modify(
            modifier_fn=LineRemover(patterns=["Passage:", "Passage 1:", "Passage 2:", "Passage 3:"]),
            input_fields=llm_response_field,
        ),
    )

    # Filter out documents that are too short (document level)
    pipeline.add_stage(
        ScoreFilter(
            TokenCountFilter(
                tokenizer=args.tokenizer,
                hf_token=args.hf_token,
                min_tokens=min_document_tokens,
            ),
            text_field=llm_response_field,
            score_field="rephrased_document_token_count",
        ),
    )

    return pipeline


def add_knowledge_list_postprocessing_pipeline(
    pipeline: Pipeline, llm_response_field: str, args: argparse.Namespace
) -> Pipeline:
    """Add KnowledgeList postprocessing pipeline."""
    max_rephrased_tokens = 598
    min_document_tokens = 50

    # Filter by token count (segment level)
    pipeline.add_stage(
        ScoreFilter(
            TokenCountFilter(
                tokenizer=args.tokenizer,
                hf_token=args.hf_token,
                max_tokens=max_rephrased_tokens,
            ),
            text_field=llm_response_field,
            score_field="rephrased_segment_token_count",
        ),
    )

    # Remove markdown formatting
    pipeline.add_stage(
        Modify(
            modifier_fn=MarkdownRemover(),
            input_fields=llm_response_field,
        ),
    )

    # Knowledge list post-processing
    pipeline.add_stage(
        KnowledgeListPostProcessingStage(
            input_field=llm_response_field,
        ),
    )

    # Filter out documents that are too short (document level)
    pipeline.add_stage(
        ScoreFilter(
            TokenCountFilter(
                tokenizer=args.tokenizer,
                hf_token=args.hf_token,
                min_tokens=min_document_tokens,
            ),
            text_field=llm_response_field,
            score_field="rephrased_document_token_count",
        ),
    )

    return pipeline
