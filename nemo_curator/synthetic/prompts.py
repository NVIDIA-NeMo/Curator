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
