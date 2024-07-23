extract_user_interest_prompt = """\
You are given a Persona and a Passage. Your task is to immitate the persona and create a list interesting topics from the given passage.

<Persona>
{persona}
</Persona>

<Passage>
The following information is from a file with the title "{file_name}".

{passage}
</Passage>

Answer format - Generate a json with the following fields
- "list_of_interest": [<fill with 1-5 word desription>]

Use Reflective Thinking: Step back from the problem, take the time for introspection and self-reflection. Examine personal biases, assumptions, and mental models that may influence problem-solving, and being open to learning from past experiences to improve future approaches. Show your thinking before giving an answer.
"""

extract_compatible_question_type_prompt = """\
You are a teacher are trying to identify which is the most types of question that will test your student's capabilities that can be asked about "{interest}" from the Passage below.
Note that the type of question should be grounded in with the information in the passage, and should have to rely on being pedantic, or general knowledge.

<Types of Questions>
{types}
</Types of Questions>

<Passage>
The following information is from a file with the title "{file_name}".

{passage}
</Passage>

Answer format - Generate a json with the following fields
- "reasoning" : show your reasoning
- "list_of_extractable_types_of_questions": [<extractive or abstractive or diagnostic or sentiment or aggregative>]
"""

extract_questions_prompt = """\
You are interviewing an expert. Generate 3 meaningful questions about {interest} from the given Passage. The questions should be {types}

These are questions for a viva, not an written examination.

<Passage>
The following information is from a file with the title "{file_name}".

{passage}
</Passage>

Answer format - Generate a json with the following fields
- "generated_questions": [questions]
"""

filter_relevance_prompt = """\
You are a juror, and are tasked with giving a judgement if there is enough evidence in the passage to answer a given question.
- Make no assumptions or use your exisiting knowledge.
- The evidence should be in the passage. The existance of pointer to the evidence doesn't qualify as sufficently useful.

Question: {question}

<Passage>
The following information is from a file with the title "{file_name}".

{passage}
</Passage>

<Judgements-Options>
- "Beyond a reasonable doubt" - There is enough evidence in the passage or the information in the passage can be used to completely answer the question beyond a reasonable doubt.
- "Somewhat relevant" - Only part of evidence required to completely answer, or to reason through get the answer is available in the passage.
- "Not useful" - The passage doesn't contain enough information to answer the question.
</Judgement-Options>

Generate your answer in a json format with the fields below
- "Reasoning": 1-10 words of reasoning
- "Your_Decision": "fill with judgement option"
"""

conversational_re_write_prompt = """\
Your task is to make minor edits to Old_Question if needed to make it sound Conversational.
- Remove phrases like "based on the given passage/information..." by making it a does or what or how or why question.
- Questions shouldn't have all the identifiers for extracting information, ie, humans are imprecise, assume context is already there.

Old_Question: {question}

<Passage>
The following information is from a file with the title "{file_name}".

{passage}
</Passage>

Answer format - Generate a json with the following fields
- "re_written_question": <fill>"""

intelligent_question_filter_prompt = """\
You are in iritated teacher. Classify a student's question in the following types.
- Type_A: A question with which student extracts valuable insights, data points, or information.
- Type_B: A pedantic or a general knowledge question.
- Type_C: It would be hard to identify the subject of the conversation without the information in the passage. These types of questions are missing proper nouns.

Question: {question}

<Passage>
The following information is from a file with the title "{file_name}".

{passage}
</Passage>

Answer Format - Generate a json with the following fields
- "Type_of_question": <Fill with Type_A or Type_B or Type_C>
"""

extract_writing_style = """\
Use the persona decription below to and articulate the Writing Style of the persona.

<Persona>
{persona}
</Persona>

Think step by step. Show your thinking.
Answer Format - Generate a json with the following fields
- "writing_style": <the writing style described in great detail in a paragraph>
"""

persona_rewrite_prompt = """\
Your task is to re-write the question like in the style of the persona below.
Use the Writing Style from the persona. It is okay to make non-sensical questions if the persona requires it.

<Style>
{persona}
</Style>

<Constraints>
- The reformated question shouldn't leak any information about the persona.
- The question should have enough identifiers to be understood in a vacuum. Don't replace too many proper nouns with pronouns.
</Constraints>

Old Question: {question}

Answer format should be a json with the following fields:
- "new_question": contains the new question.
"""
