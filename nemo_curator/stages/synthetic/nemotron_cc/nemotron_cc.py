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
from dataclasses import dataclass
from transformers import AutoTokenizer
import random
import pandas as pd
import asyncio


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

    def _process_llm_response(self, text: str, response: list[str]) -> str:
        generated_text = response[0] if response else ""
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

    def _process_sync(self, df: pd.DataFrame) -> list[str]:
        """Process DataFrame using synchronous sequential processing."""
        def generate_response(row: pd.Series) -> str:
            prompt = self._process_llm_prompt(row)
            if self.system_prompt:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ]
            else:
                messages = [
                    {"role": "user", "content": prompt}
                ]
            response = self.client.query_model(
                model=self.model_name, 
                messages=messages, 
                generation_config=self.generation_config,
            )
            original_text = row[self.input_field]
            return self._process_llm_response(text=original_text, response=response)

        # Sequential processing row by row
        return df.apply(generate_response, axis=1).tolist()

    async def _generate_responses_async(self, df: pd.DataFrame) -> list[str]:
        """Generate responses asynchronously using concurrent requests."""

        async def generate_response_async(row: pd.Series) -> str:
            prompt = self._process_llm_prompt(row)
            if self.system_prompt:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ]
            else:
                messages = [
                    {"role": "user", "content": prompt}
                ]
            response = await self.client.query_model(
                model=self.model_name,
                messages=messages,
                generation_config=self.generation_config,
            )
            original_text = row[self.input_field]
            return self._process_llm_response(text=original_text, response=response)

        # Create tasks for all rows and execute concurrently
        tasks = [generate_response_async(row) for _, row in df.iterrows()]
        return await asyncio.gather(*tasks)

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

    def _process_llm_response(self, response: list[str]) -> str:
        generated_text = response[0] if response else ""
        lines = []
        for idx, line in enumerate(generated_text.split("\n")):
            if idx == 0 and not line.startswith("-"):
                continue

            if line.startswith(("  ", "- ")):
                lines.append(line[2:].strip())
            else:
                lines.append(line)
        return "\n".join(lines)
