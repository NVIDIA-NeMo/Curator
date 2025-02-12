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
DEFAULT_YAML_CONVERSION_PROMPT_TEMPLATE = "The following document contains a list of items. Parse the list of items into a yaml list of strings. Do not parse any other part of the document. There should be no additional formatting to your response, just the yaml list of strings.\n\n {llm_response}"

DEFAULT_MACRO_TOPICS_PROMPT_TEMPLATE = "Can you generate {n_macro_topics} comprehensive topics that encompass various aspects of our daily life, the world, and science? Your answer should be a list of topics. Make the topics as diverse as possible.For example, 1. Food and drinks. \n2. Technology.\n"

DEFAULT_SUBTOPICS_PROMPT_TEMPLATE = "Can you generate {n_subtopics} comprehensive topics that encompass various aspects of {macro_topic}? Your answer should be a list of topics. Make the topics as diverse as possible."

DEFAULT_OPEN_QA_FROM_TOPICS_PROMPT_TEMPLATE = "Can you generate {n_openlines} questions or requests related to {topic}? The questions and requests should be as diverse possible. Your answer should be a list."

DEFAULT_REVISE_OPEN_QA_PROMPT_TEMPLATE = "Question: {openline}\n\nCan you revise the question above to include more contexts or details? The revised questions can be any of the follows:\n1. Adding some context to the original question. The context might state the importance of the question, explain background knowledge, or add other reasonable information.\n2. Change the questions into a different format or style, e.g., imperative statements, length requirements for the answer, etc.\n3. Elongated questions that require to elaborate on specific topic or discuss a certain point.\n4. Any other related questions or statements.\n\nThe revised question should contain two, three, or four sentences. You should generate {n_revisions} revised questions or statements in a list. Make them as diverse as possible."

DEFAULT_WRITING_TASK_PROMPT_TEMPLATE = 'Can you generate {n_openlines} tasks, each of which requires to create a "{text_material_type}" related to {topic}? Each task should be concise and include one or two sentences only. The tasks should be as diverse as possible. Your answer should be a list of tasks.'

DEFAULT_REVISE_WRITING_TASK_PROMPT_TEMPLATE = "TASK: {openline}\n\nCan you revise the task above to include more detailed requirements? These requirements can be any of the follows:\n1. Require to elaborate on a specific topic or discuss a certain point.\n2. Require to include some examples, data points, or references.\n3. Require to follow specific formats or styles, e.g., no more than 300 words, including specific words, etc.\n4. Any other reasonable requests to make the task more detailed.\n\nThe revised task should contain two, three, or four sentences. You should generate {n_revisions} revised tasks in a list. Make the tasks as diverse as possible."

DEFAULT_CLOSED_QA_PROMPT_TEMPLATE = "TEXT: {document}\n\nGiven the text above, can you come up with {n_openlines} questions or tasks? They can be any of the follows:\n1. Asking certain information in the text;\n2. Summarizing, repharsing or explaining the text;\n3. Writing something similar to the text;\n4. Any other reasonable requests related to the text.\n\nMake the questions or tasks as diverse as possible."

DEFAULT_MATH_MACRO_TOPICS_PROMPT_TEMPLATE = "Can you generate {n_macro_topics} comprehensive topics that encompass the mathematics knowledge taughted in {school_level}? Your answer should be a list of topics. Make the topics as diverse as possible."

DEFAULT_MATH_SUBTOPICS_PROMPT_TEMPLATE = 'List {n_subtopics} mathemathics topics that encompass various aspects of "{macro_topic}". Your answer should be a list of topics. Make the topics as diverse as possible.'

DEFAULT_MATH_CLASSIFICATION_PROMPT_TEMPLATE = 'Does the concept "{entity}" belong to one of the following categories?\n- Math concepts taught at elementary school, middle school, high school, and univiersity.\n- Important mathematics axioms, theorems, algorithms, equations, or inequalities.\n- Representative math problems, functions, and applications.\n\nYour answer should start with "Yes" or "No".'

MATH_PROBLEM_GENERAL_PROMPT_TEMPLATE = 'Generate {n_openlines} mathematics problems which are related to "{topic}" or can be addressed using "{topic}". Your answer should be a list of problems. Make them as diverse as possible.'

MATH_PROBLEM_BEGINNER_PROMPT_TEMPLATE = 'Generate {n_openlines} mathematics problems which are related to "{topic}" or can be addressed using "{topic}". These problems should be suitable for beginners who just learnt "{topic}". Your answer should be a list of problems. Make them as diverse as possible.'

DEFAULT_PYTHON_MACRO_TOPICS_PROMPT_TEMPLATE = (
    "List {n_macro_topics} important concepts in the python language."
)

DEFAULT_PYTHON_SUBTOPICS_PROMPT_TEMPLATE = 'List {n_subtopics} important concepts related to "{macro_topic}" in the python language.'

DEFAULT_PYTHON_CLASSIFICATION_PROMPT_TEMPLATE = 'Does the concept "{entity}" belong to one of the following categories?\n- Programming concepts like loops, functions, and data structures in python.\n- Important functions, objects, or libraries in python.\n- Mathematical concepts like linear algebra which can be implemented in python.\n- Basic algorithms or problems in computer science likes Greedy Search and Dynamics programming which can be addressed in python.\n\nYour answer should start with "Yes" or "No".'

PYTHON_PROBLEM_BEGINNER_PROMPT_TEMPLATE = 'Generate {n_openlines} {language} coding problems related to "{topic}". These problems should be suitable for beginners who just learnt "{topic}". Your answer should be a list of problems. Make them as diverse as possible.'

PYTHON_PROBLEM_INTERMEDIATE_PROMPT_TEMPLATE = 'Generate {n_openlines} {language} coding problems related to "{topic}". These problems should be suitable for medium-level programmers with some experiences of "{topic}". Your answer should be a list of problems. Make them as diverse as possible.'

PYTHON_PROBLEM_ADVANCED_PROMPT_TEMPLATE = 'Generate {n_openlines} {language} coding problems related to "{topic}". These problems should be suitable for advanced programmers with solid knowledge and experiences of "{topic}". Your answer should be a list of problems. Make them as diverse as possible.'

DIALOGUE_NORMAL_USER_TURN_PROMPT_TEMPLATE = "Here is a conversation between a user and an assistant.\n<|The Start of Assistant's Conversation with User|>\n{conversation_history}\n<|The End of Assistant's Conversation with User|>\n\nGiven the conversation above, generate a followup request or question in the tone of User. Directly give me the question without extraneous words."

DIALOGUE_COMPLEX_USER_TURN_PROMPT_TEMPLATE = "Here is a conversation between a user and an assistant.\n<|The Start of Assistant's Conversation with User|>\n{conversation_history}\n<|The End of Assistant's Conversation with User|>\n\nGiven the conversation above, generate a followup request or question in the tone of User. Make sure the question is complex and diverse enough and suitable as a followup question. Directly give me the question without extraneous words."

DIALOGUE_CONCISE_USER_TURN_PROMPT_TEMPLATE = "Here is a conversation between a user and an assistant.\n<|The Start of Assistant's Conversation with User|>\n{conversation_history}\n<|The End of Assistant's Conversation with User|>\n\nGiven the conversation above, generate a followup request or question in the toneof User. Be critical. Make sure the question is concise and has a real-life tone. Directly give me the question without extraneous words."


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
