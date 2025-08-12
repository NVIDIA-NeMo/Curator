# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import asyncio
import os
import traceback

import numpy as np
import pandas as pd
from tqdm.asyncio import tqdm

from nemo_curator import AsyncLLMClient
from nemo_curator.synthetic import AsyncNemotronGenerator

PROMPT_GENERATE_QUESTIONS_FROM_ANSWER = """TEXT:
{document}

Given the above text, generate exactly {n_openlines} questions that can be answered by the text. All questions must be answerable by the text and be relevant to the text.
Do not directly reference the text in the questions.
Every question should be a complete sentence and end with a question mark. There should be no other text besides the questions.
Begin each question with `* ` and end each question with a newline character. Also, each question must be concise.
Make sure to generate exactly {n_openlines} questions.
"""

PROMPT_PARAPHRASE_TEXT = """TEXT:
{document}

Given the above text, paraphrase the text. Produce exactly {n_openlines} variants.
There should be no other text besides the paraphrased text.
The paraphrased text must be shorter than the original text. The paraphrased text must be factually correct and relevant to the original text.
Begin each variant with `* ` and end each variant with a newline character.
Make sure to generate exactly {n_openlines} variants.
"""


class SyntheticGenerator:
    def __init__(  # noqa: PLR0913
        self,
        async_llm_client: AsyncLLMClient,
        sdg_model: str,
        sdg_model_kwargs: dict,
        reward_model: str | None,
        n_variants: int = 1,
        random_seed: int = 42,
        max_concurrent_entries: int = 320,
    ):
        """
        Initializes the SyntheticGenerator object.

        Args:
            async_llm_client: The asynchronous LLM client.
            sdg_model: The path to the SDG model.
            sdg_model_kwargs: Additional keyword arguments for the SDG model.
            reward_model: The reward model for quality assignment (optional). If not provided,
                the quality will be assigned based on the human-assigned scores.
            n_variants: The number of variants to generate (default: 1).
            random_seed: The random seed for reproducibility (default: 42).
            max_concurrent_entries: The maximum number of concurrent entries (default: 320).
        """
        self.client = async_llm_client
        self.generator = AsyncNemotronGenerator(self.client)
        self.random_state = np.random.RandomState(random_seed)
        self.sdg_model = sdg_model
        self.sdg_model_kwargs = sdg_model_kwargs
        self.reward_model = reward_model
        self.n_variants = n_variants
        self.max_concurrent_entries = max_concurrent_entries

    def run(
        self,
        source_df: pd.DataFrame,
        out_dir: str,
        synth_prefix: str,
        synth_gen_ratio: float,
    ) -> str:
        """
        Runs the synthetic data generation process.

        Args:
            source_df: The source dataframe containing the original data.
            out_dir: The output directory where the synthetic data will be saved.
            synth_prefix: The prefix to be added to the names of the synthetic data files.
            synth_gen_ratio: The ratio of synthetic data to be generated compared to the original data.

        Returns:
            The path to the directory where the synthetic data is saved.
        """
        return asyncio.run(
            self._synthesize_from_source(
                source_df,
                out_dir,
                synth_prefix,
                synth_gen_ratio,
            ),
        )

    def _split_sdg_responses(self, sdg_response: str) -> list[str]:
        """
        Splits the SDG response into a list of individual entries.

        Args:
            sdg_response: The SDG response string.

        Returns:
            A list of individual SDG entries.
        """
        return [entry.strip("*").strip() for entry in sdg_response[0].split("\n") if entry]

    def _write_all_to_file(self, gen_entries, out_fp: str) -> None:  # noqa: ANN001
        """
        Write all generated synthetic data to a JSON file. If nothing was generated, skip writing to the file.

        Args:
            gen_entries: List of generated entries.
            out_fp: Output file path.
        """
        synth_titles = []
        synth_questions = []
        synth_answers = []
        synth_scores = []
        synth_filenames = []
        synth_ids = []
        synth_tags = []

        for gen_idx, (row_slice, gen_entry) in enumerate(gen_entries):
            for idx, (titles, questions, answers, scores) in enumerate(gen_entry):
                if (
                    len(questions) != self.n_variants
                    or len(answers) != self.n_variants
                    or len(titles) != self.n_variants
                ):
                    print(
                        f"    Skipping synthetic record at ({gen_idx}, {idx}) due to unexpected lengths. The LLM may have failed to generate the expected number of variants.",
                    )
                    continue

                row = row_slice.iloc[idx]
                synth_titles.extend(titles)
                synth_questions.extend(questions)
                synth_answers.extend(answers)
                synth_scores.extend(scores)
                synth_filenames.extend([row["file_name"] + ".synth"] * self.n_variants)
                synth_ids.extend(
                    [f"{row['id']}-synth-{i}" for i in range(self.n_variants)],
                )
                synth_tags.extend([row["tags"]] * self.n_variants)

        if not synth_titles:
            print("    No valid synthetic data generated. Skipping writing to file.")
            return

        # Create a Series object with the generated data.
        gen_data = {
            "answer": synth_answers,
            "answer_score": synth_scores,
            "file_name": synth_filenames,
            "id": synth_ids,
            "question": synth_questions,
            "question_score": synth_scores,
            "tags": synth_tags,
            "title": synth_titles,
        }
        # Dump the data into a file.
        gen_df = pd.DataFrame(gen_data)
        gen_df.to_json(out_fp, orient="records", lines=True)

    async def _prompt_model(
        self,
        row: pd.Series,
    ) -> tuple[list[str], list[str], list[str]]:
        """
        Generates synthetic data by prompting a language model with a given question and answer.

        Args:
            row: The input row containing the question and answer.

        Returns:
            Tuple[List[str], List[str], List[str]]: A tuple containing the generated titles, questions, answers,
            and their corresponding scores.
        """
        question = row["question"]
        answer = row["answer"]

        gen_title = self.generator.generate_closed_qa_instructions(
            document=answer,
            n_openlines=self.n_variants,
            prompt_template=PROMPT_GENERATE_QUESTIONS_FROM_ANSWER,
            model=self.sdg_model,
            model_kwargs=self.sdg_model_kwargs,
        )
        gen_question = self.generator.generate_closed_qa_instructions(
            document=question,
            n_openlines=self.n_variants,
            prompt_template=PROMPT_PARAPHRASE_TEXT,
            model=self.sdg_model,
            model_kwargs=self.sdg_model_kwargs,
        )
        gen_answer = self.generator.generate_closed_qa_instructions(
            document=answer,
            n_openlines=self.n_variants,
            prompt_template=PROMPT_PARAPHRASE_TEXT,
            model=self.sdg_model,
            model_kwargs=self.sdg_model_kwargs,
        )

        gen_title, gen_question, gen_answer = await asyncio.gather(
            gen_title,
            gen_question,
            gen_answer,
        )

        gen_title = self._split_sdg_responses(gen_title)
        gen_question = self._split_sdg_responses(gen_question)
        gen_answer = self._split_sdg_responses(gen_answer)

        scores = []

        # Use the reward model to assign scores to the generated data.
        if self.reward_model:
            for t, q, a in zip(gen_title, gen_question, gen_answer, strict=False):
                messages = [
                    {"role": "user", "content": f"{t}\n\n{q}"},
                    {"role": "assistant", "content": a},
                ]
                scores.append(
                    self.client.query_reward_model(
                        messages=messages,
                        model=self.reward_model,
                    ),
                )

            scores = await asyncio.gather(*scores)
            # Convert each score to a scale of -2 to 2.
            scores = [int(score["helpfulness"] - 2) for score in scores]
        else:
            # Assign a score of 0 to all generated data.
            scores = [0] * self.n_variants

        return gen_title, gen_question, gen_answer, scores

    async def _synthesize_from_source(
        self,
        source_df: pd.DataFrame,
        out_dir_path: str,
        synth_prefix: str,
        synth_gen_ratio: float,
    ) -> str:
        """
        Synthesizes data from a source DataFrame and saves it to a specified directory.

        Args:
            source_df: The source DataFrame containing the data to synthesize.
            out_dir_path: The path to the directory where the synthesized data will be saved.
            synth_prefix: The prefix to use for the synthesized data file.
            synth_gen_ratio: The ratio of data to synthesize from the source DataFrame.

        Returns:
            The path to the directory where the synthesized data is saved.
        """
        os.makedirs(out_dir_path, exist_ok=True)
        # Randomly select a subset of the data to synthesize.
        source_df = source_df.sample(
            frac=synth_gen_ratio,
            random_state=self.random_state,
        )
        prompt_requests = []

        # Generate prompts for each row in the source data and submit them to the LLM.
        for _, row in source_df.iterrows():
            gen_entry = self._prompt_model(row)
            prompt_requests.append(gen_entry)

        gen_entries = []

        for i in tqdm(
            range(0, len(prompt_requests), self.max_concurrent_entries),
            desc=f"Synthesizing {len(source_df)} rows",
        ):
            slice_end = min(i + self.max_concurrent_entries, len(prompt_requests))
            row_slice = source_df[i:slice_end]
            request_slice = prompt_requests[i:slice_end]

            try:
                result = await tqdm.gather(
                    *request_slice,
                    desc=f"---- Rows {i} to {slice_end}",
                )
                gen_entries.append((row_slice, result))
            except Exception as _:  # noqa: BLE001
                print(
                    f"    Generation failed for rows {i} to {slice_end} due to the following exception:",
                )
                print("---------------------------------------------------------")
                traceback.print_exc()
                print("---------------------------------------------------------")
                print("Continuing synthetic data generation...")

        # Write the generated data to a file.
        out_fp = f"{out_dir_path}/{synth_prefix}.jsonl"
        self._write_all_to_file(gen_entries, out_fp)
        return out_dir_path
