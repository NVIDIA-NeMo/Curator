import ast
import json

from Endpoints import *
from prompts import *


class Generator:
    def __init__(self):
        self.llm = LLaMa_405B()

    def extract_points_of_interest(self, persona, file_name, passage):
        prompt = extract_user_interest_prompt.format(
            persona=persona, file_name=file_name, passage=passage
        )

        schema = {
            "guided_json": {
                "type": "object",
                "properties": {
                    "list_of_interest": {
                        "description": "[<fill with 1-5 word desription>]",
                        "type": "array",
                    }
                },
                "required": ["list_of_interest"],
            }
        }

        raw_answer = json.loads(self.llm.invoke(prompt, schema))
        return raw_answer

    def extract_compatible_question_type(self, interest, types, file_name, passage):
        prompt = extract_compatible_question_type_prompt.format(
            interest="\n".join(interest),
            types="\n".join(types),
            file_name=file_name,
            passage=passage,
        )

        schema = {
            "guided_json": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "description": "show your reasoning",
                        "type": "string",
                    },
                    "list_of_extractable_types_of_questions": {
                        "description": "list_of_extractable_types_of_questions",
                        "type": "array",
                    },
                },
                "required": ["reasoning", "list_of_extractable_types_of_questions"],
            }
        }
        answer = json.loads(self.llm.invoke(prompt, schema))
        return answer

    def generate_questions(self, file_name, passage, interest, types):
        prompt = extract_questions_prompt.format(
            file_name=file_name, passage=passage, interest=interest, types=types
        )

        schema = {
            "guided_json": {
                "type": "object",
                "properties": {
                    "generated_questions": {
                        "description": "[questions]",
                        "type": "array",
                    }
                },
                "required": ["generated_questions"],
            }
        }
        try:
            answer = json.loads(self.llm.invoke(prompt, schema))["generated_questions"]
            return answer
        except:
            return []

    def conversational_re_write(self, question, file_name, passage):
        prompt = conversational_re_write_prompt.format(
            question=question, file_name=file_name, passage=passage
        )
        schema = {
            "guided_json": {
                "type": "object",
                "properties": {
                    "re_written_question": {"description": "<fill>", "type": "string"}
                },
                "required": ["re_written_question"],
            }
        }

        answer = json.loads(self.llm.invoke(prompt, schema))
        return answer

    def writing_style(self, persona):
        prompt = extract_writing_style.format(persona=persona)
        schema = {
            "guided_json": {
                "type": "object",
                "properties": {
                    "writing_style": {
                        "description": "<the writing style described in great detail in a paragraph>",
                        "type": "string",
                    }
                },
                "required": ["writing_style"],
            }
        }
        answer = self.llm.invoke(prompt, schema)

        return answer

    def persona_rewrite(self, persona, question):
        prompt = persona_rewrite_prompt.format(persona=persona, question=question)

        schema = {
            "guided_json": {
                "type": "object",
                "properties": {
                    "new_question": {
                        "description": "<the writing style described in great detail in a paragraph>",
                        "type": "string",
                    }
                },
                "required": ["new_question"],
            }
        }
        try:
            answer = self.llm.invoke(prompt, schema)
            return answer
        except:
            return {"reasoning": "error", "new_question": question}


class Relevance_Filter:
    def __init__(self):
        self.llm = LLaMa_405B()

    def execute(self, question, file_name, passage):
        prompt = filter_relevance_prompt.format(
            question=question, file_name=file_name, passage=passage
        )

        schema = {
            "guided_json": {
                "type": "object",
                "properties": {
                    "Reasoning": {
                        "description": "1-10 words of reasoning",
                        "type": "string",
                    },
                    "Your_Decision": {
                        "description": "fill with judgement option",
                        "type": "string",
                    },
                },
                "required": ["Reasoning", "Your_Decision"],
            }
        }

        answer = json.loads(self.llm.invoke(prompt, schema))

        return answer


class Intelligent_Question_Filter:
    def __init__(self):
        self.llm = LLaMa_405B()

    def execute(self, question, file_name, passage):
        prompt = intelligent_question_filter_prompt.format(
            question=question, file_name=file_name, passage=passage
        )

        schema = {
            "guided_json": {
                "type": "object",
                "properties": {
                    "Type_of_question": {
                        "description": "<Fill with Type_A or Type_B or Type_C>",
                        "type": "string",
                    }
                },
                "required": ["Type_of_question"],
            }
        }

        answer = json.loads(self.llm.invoke(prompt, schema))
        return answer
