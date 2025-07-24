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

DEFAULT_REASONING_TRACE_PROMPT_TEMPLATE = """
You are a helpful assistant that generates reasoning traces and answers for a given problem.
Problem: {problem}
"""

DEFAULT_GRADING_PROMPT_TEMPLATE = """
You are an AI assistant for grading a science problem. The user will provide you with the question itself, an attempt made by a student and the correct answer to the problem. Your job is to judge whether the attempt is correct by comparing it with the correct answer. If the expected solution concludes with a number or choice, there should be no ambiguity. If the expected solution involves going through the entire reasoning process, you should judge the attempt based on whether the reasoning process is correct with correct answer if helpful.
The user will provide theattemptandthe correctanswer inthefollowing format:
# Problem
{problem}
## Attempt
{attempt}
## Correct answer
{solution}
Explain your reasoning, and end your response on a new line with strictly only "Yes" or "No", no punctuation marks.
(Example for last line: '\nYes', '\nNo').
If not knowing the answer, respond with "No".
"""

DEFAULT_DOMAIN_CLASSIFICATION_PROMPT_TEMPLATE = """
You are a helpful assistant that classifies questions into different subjects based on the provided classification rubrics.
You will be given a question and a list of subjects.
You need to classify the question into one of the subjects.
If the question has multiple subjects, you should classify the question into the most relevant subject.
Explain your reasoning, and end your response on a new line with two-digit code of the subject that the question belongs to
(Example for last line: '\n01', '\n02', '\n03', etc., no punctuation marks).
## Question
{question}
## Classification rubrics
{domains_prompt}
"""
