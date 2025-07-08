import json

from Endpoints import LLaMa_405B
from prompts import (
    conversational_re_write_prompt,
    extract_compatible_question_type_prompt,
    extract_questions_prompt,
    extract_user_interest_prompt,
    extract_writing_style,
    filter_relevance_prompt,
    intelligent_question_filter_prompt,
    persona_rewrite_prompt,
)


class Generator:
    def __init__(self):
        self.llm = LLaMa_405B()

    def extract_points_of_interest(self, persona: str, file_name: str, passage: str) -> dict[str, list[str]]:
        prompt = extract_user_interest_prompt.format(
            persona=persona,
            file_name=file_name,
            passage=passage,
        )

        schema = {
            "guided_json": {
                "type": "object",
                "properties": {
                    "list_of_interest": {
                        "description": "[<fill with 1-5 word desription>]",
                        "type": "array",
                    },
                },
                "required": ["list_of_interest"],
            },
        }

        return json.loads(self.llm.invoke(prompt, schema))

    def extract_compatible_question_type(
        self, interest: list[str], types: list[str], file_name: str, passage: str
    ) -> dict[str, list[str]]:
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
            },
        }
        return json.loads(self.llm.invoke(prompt, schema))

    def generate_questions(self, file_name: str, passage: str, interest: list[str], types: list[str]) -> list[str]:
        prompt = extract_questions_prompt.format(
            file_name=file_name,
            passage=passage,
            interest=interest,
            types=types,
        )

        schema = {
            "guided_json": {
                "type": "object",
                "properties": {
                    "generated_questions": {
                        "description": "[questions]",
                        "type": "array",
                    },
                },
                "required": ["generated_questions"],
            },
        }
        try:
            return json.loads(self.llm.invoke(prompt, schema))["generated_questions"]
        except:  # noqa: E722
            return []

    def conversational_re_write(self, question: str, file_name: str, passage: str) -> dict[str, str]:
        prompt = conversational_re_write_prompt.format(
            question=question,
            file_name=file_name,
            passage=passage,
        )
        schema = {
            "guided_json": {
                "type": "object",
                "properties": {
                    "re_written_question": {"description": "<fill>", "type": "string"},
                },
                "required": ["re_written_question"],
            },
        }

        return json.loads(self.llm.invoke(prompt, schema))

    def writing_style(self, persona: str) -> dict[str, str]:
        prompt = extract_writing_style.format(persona=persona)
        schema = {
            "guided_json": {
                "type": "object",
                "properties": {
                    "writing_style": {
                        "description": "<the writing style described in great detail in a paragraph>",
                        "type": "string",
                    },
                },
                "required": ["writing_style"],
            },
        }
        return self.llm.invoke(prompt, schema)

    def persona_rewrite(self, persona: str, question: str) -> dict[str, str]:
        prompt = persona_rewrite_prompt.format(persona=persona, question=question)

        schema = {
            "guided_json": {
                "type": "object",
                "properties": {
                    "new_question": {
                        "description": "<the writing style described in great detail in a paragraph>",
                        "type": "string",
                    },
                },
                "required": ["new_question"],
            },
        }
        try:
            return self.llm.invoke(prompt, schema)
        except:  # noqa: E722
            return {"reasoning": "error", "new_question": question}


class Relevance_Filter:  # noqa: N801
    def __init__(self):
        self.llm = LLaMa_405B()

    def execute(self, question: str, file_name: str, passage: str) -> dict[str, str]:
        prompt = filter_relevance_prompt.format(
            question=question,
            file_name=file_name,
            passage=passage,
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
            },
        }

        return json.loads(self.llm.invoke(prompt, schema))


class Intelligent_Question_Filter:  # noqa: N801
    def __init__(self):
        self.llm = LLaMa_405B()

    def execute(self, question: str, file_name: str, passage: str) -> dict[str, str]:
        prompt = intelligent_question_filter_prompt.format(
            question=question,
            file_name=file_name,
            passage=passage,
        )

        schema = {
            "guided_json": {
                "type": "object",
                "properties": {
                    "Type_of_question": {
                        "description": "<Fill with Type_A or Type_B or Type_C>",
                        "type": "string",
                    },
                },
                "required": ["Type_of_question"],
            },
        }

        return json.loads(self.llm.invoke(prompt, schema))
