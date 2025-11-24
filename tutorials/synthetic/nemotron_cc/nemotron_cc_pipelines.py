# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.synthetic.nemotron_cc.nemotron_cc import (
    DiverseQAPostProcessingStage,
    KnowledgeListPostProcessingStage,
)
from nemo_curator.stages.text.filters.heuristic_filter import SubstringFilter, TokenCountFilter
from nemo_curator.stages.text.modifiers.line_remover import LineRemover
from nemo_curator.stages.text.modifiers.markdown_remover import MarkdownRemover
from nemo_curator.stages.text.modifiers.quotation_remover import QuotationRemover
from nemo_curator.stages.text.modifiers.slicer import Slicer
from nemo_curator.stages.text.modules.modifier import Modify
from nemo_curator.stages.text.modules.score_filter import ScoreFilter

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
    """Add Nemotron-CC preprocessing pipeline."""

    # TODO: add coressponding support for add_preprocessing_pipeline
    print("Unused system_prompt: ", system_prompt)
    print("Unused user_prompt_template: ", user_prompt_template)
    print("Unused min_segment_tokens: ", min_segment_tokens)
    print("Unused max_input_tokens: ", max_input_tokens)

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

    return pipeline

def add_wikipedia_postprocessing_pipeline(
    pipeline: Pipeline, _llm_response_field: str, _args: argparse.Namespace
) -> Pipeline:
    """Add Wikipedia postprocessing pipeline."""

    return pipeline

def add_diverse_qa_postprocessing_pipeline(pipeline: Pipeline, llm_response_field: str, args: argparse.Namespace) -> Pipeline:
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

def add_distill_postprocessing_pipeline(pipeline: Pipeline, llm_response_field: str, args: argparse.Namespace) -> Pipeline:
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

def add_extract_knowledge_postprocessing_pipeline(pipeline: Pipeline, llm_response_field: str, args: argparse.Namespace) -> Pipeline:
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

def add_knowledge_list_postprocessing_pipeline(pipeline: Pipeline, llm_response_field: str, args: argparse.Namespace) -> Pipeline:
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
