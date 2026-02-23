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


import random
from dataclasses import dataclass

import pandas as pd
from transformers import AutoTokenizer

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.synthetic.nemotron_cc.base import BaseSyntheticStage
from nemo_curator.stages.synthetic.nemotron_cc.prompts import (
    DISTILL_PROMPT_TEMPLATE,
    DIVERSE_QA_PROMPT_TEMPLATE,
    EXTRACT_KNOWLEDGE_PROMPT_TEMPLATE,
    KNOWLEDGE_LIST_PROMPT_TEMPLATE,
    NEMOTRON_CC_DISTILL_SYSTEM_PROMPT,
    NEMOTRON_CC_SYSTEM_PROMPT,
    WIKIPEDIA_REPHRASING_PROMPT_TEMPLATE,
)
from nemo_curator.tasks import DocumentBatch


@dataclass
class WikipediaParaphrasingStage(BaseSyntheticStage):
    system_prompt: str = NEMOTRON_CC_SYSTEM_PROMPT
    prompt: str = WIKIPEDIA_REPHRASING_PROMPT_TEMPLATE
    input_field: str = "text"
    output_field: str = "rephrased"

@dataclass
class DiverseQAStage(BaseSyntheticStage):
    system_prompt: str = None
    prompt: str = DIVERSE_QA_PROMPT_TEMPLATE
    input_field: str = "text"
    output_field: str = "diverse_qa"
    tokenizer: AutoTokenizer = None
    prefix: str = "Here are the questions and answers based on the provided text:"
    max_num_pairs: int = 10


@dataclass
class DiverseQAPostProcessingStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    Post-processing stage for DiverseQA outputs. It parses the raw generated QA list,
    normalizes bullets, optionally samples pairs based on input length/tokenizer,
    and concatenates the original document text with the selected QA pairs.
    """

    input_field: str = "text"
    qa_field: str = "diverse_qa"
    tokenizer: AutoTokenizer | None = None
    prefix: str = "Here are the questions and answers based on the provided text:"
    max_num_pairs: int = 10
    name: str = "DiverseQAPostProcessing"

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()

        def _format_row(row: pd.Series) -> str:
            text = row[self.input_field]
            generated_text = row[self.qa_field]
            lines = [line.strip() for line in generated_text.split("\n") if line.strip()]
            if not lines:
                return ""

            # Remove the "- " prefix
            lines = [line[2:].strip() if line.startswith("- ") else line for line in lines]

            if lines[0] == self.prefix:
                lines = lines[1:]

            # Merge question and answer lines
            qa_pairs = []
            for line in lines:
                if line.startswith("Question:"):
                    qa_pairs.append(line)
                elif qa_pairs:
                    qa_pairs[-1] += "\n" + line
                else:
                    return ""

            if len(qa_pairs) == 0:
                return ""

            # Shuffle the QA pairs and sample up to max_num_pairs
            random.shuffle(qa_pairs)
            if self.tokenizer is not None:
                num_tokens = len(self.tokenizer.tokenize(text))
                qa_pairs = qa_pairs[: random.randint(1, max(1, int(self.max_num_pairs * num_tokens / 150)))]  # noqa: S311
            else:
                qa_pairs = qa_pairs[: random.randint(1, self.max_num_pairs)]  # noqa: S311
            qa_pairs_str = "\n\n".join(qa_pairs)

            # Concatenate the document and the QA pairs
            return f"{text}\n\n{qa_pairs_str}"

        df[self.qa_field] = df.apply(_format_row, axis=1)

        return DocumentBatch(
            data=df,
            dataset_name=batch.dataset_name,
            task_id=f"{batch.task_id}_{self.name}",
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )

@dataclass
class DistillStage(BaseSyntheticStage):
    system_prompt: str = NEMOTRON_CC_DISTILL_SYSTEM_PROMPT
    prompt: str = DISTILL_PROMPT_TEMPLATE
    input_field: str = "text"
    output_field: str = "distill"

@dataclass
class ExtractKnowledgeStage(BaseSyntheticStage):
    system_prompt: str = None
    prompt: str = EXTRACT_KNOWLEDGE_PROMPT_TEMPLATE
    input_field: str = "text"
    output_field: str = "extract_knowledge"

@dataclass
class KnowledgeListStage(BaseSyntheticStage):
    system_prompt: str = None
    prompt: str = KNOWLEDGE_LIST_PROMPT_TEMPLATE
    input_field: str = "text"
    output_field: str = "knowledge_list"

@dataclass
class KnowledgeListPostProcessingStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    Post-processing stage that formats knowledge list outputs generated by the LLM.
    It normalizes leading bullet markers and trims indentation, producing a clean newline-separated list.
    """

    input_field: str = "knowledge_list"
    name: str = "KnowledgeListPostProcessing"

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()

        def _format_text(generated_text: str) -> str:
            lines: list[str] = []
            for idx, line in enumerate(generated_text.split("\n")):
                if idx == 0 and not line.startswith("-"):
                    continue
                if line.startswith(("  ", "- ")):
                    lines.append(line[2:].strip())
                else:
                    lines.append(line)
            return "\n".join(lines)

        # Read from knowledge_list, process, and write back to knowledge_list
        df[self.input_field] = df[self.input_field].fillna("").apply(_format_text)

        return DocumentBatch(
            data=df,
            dataset_name=batch.dataset_name,
            task_id=f"{batch.task_id}_{self.name}",
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
