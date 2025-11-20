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

# Nemotron-CC prompts

NEMOTRON_CC_SYSTEM_PROMPT = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the questions."

NEMOTRON_CC_DISTILL_SYSTEM_PROMPT = "You are an artificial intelligence assistant. You carefully provide accurate, factual, thoughtful, nuanced answers, and are brilliant at reasoning."

WIKIPEDIA_REPHRASING_PROMPT_TEMPLATE = """For the following paragraph give me a diverse paraphrase of the same in high quality English language as in sentences on Wikipedia. Begin your answer on a separate line with "Here is a paraphrased version:".

Text: {document}"""

DIVERSE_QA_PROMPT_TEMPLATE = """Task:
Read the text, ask questions and answer them.

Follow these instructions:
1. Ask diverse questions that require different cognitive skills or cover different aspects of the text.
2. Ask questions in various forms such as:
  - Yes/No questions that require determining whether a statement is true or false.
  - Open-ended questions that begin with words like what, how, when, where, why and who.
  - Multi-choice questions that offers two or more options to choose from. Include the options in the question.
  - Comparison questions that compare two quantities or objects and determine the relationship between them.
  - Reading comprehension questions that test the ability to understand and analyze the text.
  - Problem-solving questions that test the ability to solve mathematical, physical, or logical problems.
3. Focus on asking questions about factual information, important knowledge, or concrete details in the text.
4. Write questions and answers using clear and concise language.
5. Use plain text. Do not use Markdown.
6. Each question and answer pair should be on a separate line. Tag the question with "Question:" and the answer with "Answer:".

Text:
{document}

Task:
After reading the above text, ask up to 8 questions and provide the correct answers following the instructions. Give your response in this format:

Here are the questions and answers based on the provided text:
- Question: [first question] Answer: [first answer]
- Question: [second question] Answer: [second answer]
...."""

DISTILL_PROMPT_TEMPLATE = """Your task is to read and paraphrase the provided text following these instructions:
- Aim to create a condensed but accurate and informative version of the original text, not a simplistic summary.
- Capture and preserve the crucial information, key concepts, important values, factual details in the original text, while making it more readable and accessible.
- Retain technical terms, specialized vocabulary, and complex concepts.
- Retain examples, explanations of reasoning processes, and supporting evidence to maintain the text's depth and context.
- Only include information that is present in the original text. Do not adding new or unsubstantiated claims.
- Write the text in plain text without formatting.

Here is the text:
{document}

Task:
After thoroughly reading the above text, paraphrase it in high-quality and clear English following the instructions. Begin your response with "Paraphrased Text:"."""

EXTRACT_KNOWLEDGE_PROMPT_TEMPLATE = """Your task is to rewrite knowledge from the provided text following these instructions.
- Rewrite the text as a passage or passages using easy-to-understand and high-quality English like sentences in textbooks and Wikipedia.
- Focus on content in disciplines such as humanities, social sciences, natural sciences, technology, engineering, math, law and legal, business, management, art, education, agricultural sciences, politics, and history.
- Disregard content that does not contain useful facts or knowledge.
- Retain examples, explanations of reasoning processes, and supporting evidence to maintain the text's depth and context.
- Do not add or alter details. Only restate what is already in the text.
- Write in plain text.
- Do not add titles, subtitles, note, or comment.

Text:
{document}

Task:
Rewrite facts and knowledge from the above text as a passage or passages following the instructions."""

KNOWLEDGE_LIST_PROMPT_TEMPLATE = """Review the text and extract the key information. Follow these instructions:
- Carefully read the above text and provide a concise and organized list of factual information, concrete details, key concepts, and important numbers and statistics extracted from the text.
- Ensure each point is clear, specific, and supported by the original text.
- Ensure the extract text is information-dense and easier to learn from.
- Do not add titles or headings.

Text:
{document}

Task:
Extract the factual information, concrete details, and key concepts from the above text following the instructions."""
