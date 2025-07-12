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

import numpy as np

from ray_curator.stages.filters.doc_filter import DocumentFilter
from ray_curator.stages.utils.reasoning_utils import LLM_attempt_generator, LLM_attempt_grader, grade_attempt_with_llm, generate_attempt_with_llm, classify_domain_with_llm

class ExactMatchCorrectnessFilter(DocumentFilter):

    def score_document(self, sample: dict) -> float:
        problem = sample["problem"]
        attempt = sample["attempt"]
        solution = sample["solution"]

        # check exact match between attempt and solution
        if attempt == solution:
            document_score = 1.0
        else:
            document_score = 0.0

        return document_score

    def keep_document(self, score: float) -> bool:
        return score == 1.0


class LLMBasedCorrectnessFilter(DocumentFilter):
    def __init__(self, model_attempt_grader: str, prompt_template: str):
        self._model_attempt_grader = LLM_attempt_grader(model_attempt_grader)
        if prompt_template is not None:
            self._prompt_template = prompt_template
        else:
            self._prompt_template = """
            You are an AI assistant for grading a science problem. The user will provide you with the question itself, an attempt made by a student and the correct answer to the problem. Your job is to judge whether the attempt is correct by comparing it with the correct answer. If the expected solution concludes with a number or choice, there should be no ambiguity. If the expected solution involves going through the entire reasoning process, you should judge the attempt based on whether the reasoning process is correct with correct answer if helpful.

            The user will provide theattemptandthe correctanswer inthefollowing format:

            # Problem
            {problem}

            ## Attempt
            {attempt}

            ## Correct answer
            {solution}

            Explain your reasoning, and end your response on a new line with only "Yes" or "No" (without quotes).
            """

    def score_document(self, sample: dict) -> float:
        document_score = grade_attempt_with_llm(self._model_attempt_grader, sample, self._prompt_template)
        
        return document_score

    def keep_document(self, score: float) -> bool:
        return score == 1.0


class LLMBasedAttemptGenerator(DocumentFilter):
    def __init__(self, model_path: str, prompt_template: str):
        self._model_path = model_path
        if prompt_template is not None:
            self._prompt_template = prompt_template
        else:
            self._prompt_template = """
            """


class LLMBasedDifficultyFilter(DocumentFilter):
    def __init__(self, model_attempt_generator_1: str, model_attempt_generator_2: str, model_attempt_grader: str, prompt_template: str):
        self._model_attempt_generator_1 = LLM_attempt_generator(model_attempt_generator_1)
        self._model_attempt_generator_2 = LLM_attempt_generator(model_attempt_generator_2)
        self._model_attempt_grader = LLM_attempt_grader(model_attempt_grader)
        if prompt_template is not None:
            self._prompt_template = prompt_template
        else:
            self._prompt_template = """
            """

    def score_document(self, sample: dict) -> float:
        problem = sample["problem"]
        # Generate attempts using both models
        model_1_prompt = f"Solve the following problem:\n{problem}"
        model_2_prompt = f"Solve the following problem:\n{problem}"
        
        # Generate response from models
        model_1_attempt = generate_attempt_with_llm(self._model_attempt_generator_1, problem, self._prompt_template)
        model_2_attempt = generate_attempt_with_llm(self._model_attempt_generator_2, problem, self._prompt_template)

        # Grade attempts with LLM
        model_1_score = grade_attempt_with_llm(self._model_attempt_grader, model_1_attempt, self._prompt_template)
        model_2_score = grade_attempt_with_llm(self._model_attempt_grader, model_2_attempt, self._prompt_template)

        return [model_1_score, model_2_score]

    def keep_document(self, scores: list[float]) -> bool:
        return scores[0] == 1.0 and scores[1] == 1.0


class ReasoningLengthDifficultyFilter(DocumentFilter):
    def __init__(self, min_length: int):
        self._min_length = min_length

    def score_document(self, sample: dict) -> float:
        attempt = sample["attempt"]
        solution = sample["solution"]

        # check if the attempt is longer than min_length
        if len(attempt) > self._min_length:
            document_score = 1.0
        else:
            document_score = 0.0
        
        return document_score
    
    def keep_document(self, score: float) -> bool:
        return score == 1.0


class LLMBasedDomainClassifier(DocumentFilter???):
    def __init__(self, model_domain_classifier: str, prompt_template: str, domain_list: list[dict]):
        self._model_domain_classifier = LLM_domain_classifier(model_domain_classifier)
        if prompt_template is not None:
            self._prompt_template = prompt_template
        else:
            self._prompt_template = """
            You are a helpful assistant that classifies questions into different subjects based on the provided classification rubrics.
            You will be given a question and a list of subjects.
            You need to classify the question into one of the subjects.
            If the question has multiple subjects, you should classify the question into the most relevant subject.
            Explain your reasoning, and end your response on a new line with two-digit code of the subject that the question belongs to.
            """

        domain_prompt = '\n\n'.join([node['prompt'] for node in domain_list])
        self._prompt_template = self._prompt_template.format(domain_prompt=domain_prompt)

    def classify_domains(self, sample: dict) -> str:
        question = sample["question"]
        result = classify_domain_with_llm(self._model_domain_classifier, question, self._prompt_template)
        return result

    def call(self, batch: DocumentBatch) -> DocumentBatch:
        classified_domains = self.classify_domains(batch)
        return classified_domains


class DiversityFilter(???):
    def __init__(self):
        return

    def diversity_sampling(self, all_domains, batch: DocumentBatch):
        all_domains = list(set([featurized_dict[question_hash(example['question'])]['domain'] for example in geminiall]))
        def uniform_sample(all_domains):
            return np.random.choice(all_domains, size=1)[0]

        questions_ordered_by_domain = {}
        for domain in tqdm(all_domains):
            questions_ordered_by_domain[domain] = {k:v for k, v in featurized_dict.items() if v['domain'] == domain}

        pbar = tqdm(initial=len(selected_qhashes), total=1000, desc="Sampling questions")
        while len(selected_qhashes) < 1000:
            # first sample 300 uniformly over all domains
            if len(selected_qhashes) < 700:
                random_domain = uniform_sample(all_domains)
            else:
                random_domain = benchmark_sample(benchmark_domains, benchmark_weights)
            # Sort by chain length and take the longest one
            domain_examples = questions_ordered_by_domain[random_domain]
            qhashes = list(domain_examples.keys())
            lengths = np.array([int(domain_examples[qhash]['gemini_length']) for qhash in qhashes])
            ranks = len(lengths) - 1 - np.argsort(np.argsort(lengths))
            length_weights = np.power(2.0, -ranks)
            length_weights = length_weights / length_weights.sum()
            selected_qhash = np.random.choice(qhashes, p=length_weights)
            selected_qhashes.add(selected_qhash)
            questions_ordered_by_domain[random_domain].pop(selected_qhash)
            if len(questions_ordered_by_domain[random_domain]) == 0:
                if random_domain in all_domains:
                    all_domains.remove(random_domain)
                if random_domain in benchmark_domains:
                    benchmark_weights = np.delete(benchmark_weights, benchmark_domains.index(random_domain))
                    benchmark_weights = benchmark_weights / benchmark_weights.sum()
                    benchmark_domains.remove(random_domain)
            pbar.update(1)
        pbar.close()
        return selected_qhashes

    def call(self, batch: DocumentBatch) -> DocumentBatch:
        sampled_batch = self.diversity_sampling(batch)
        return sampled_batch
