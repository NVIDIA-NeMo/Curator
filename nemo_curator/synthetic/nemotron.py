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
from typing import List, Tuple, Union

import yaml

from nemo_curator.services.conversation_formatter import ConversationFormatter
from nemo_curator.services.model_client import LLMClient
from nemo_curator.synthetic.error import YamlConversionError
from nemo_curator.synthetic.prompts import (
    DEFAULT_CLOSED_QA_PROMPT_TEMPLATE,
    DEFAULT_MACRO_TOPICS_PROMPT_TEMPLATE,
    DEFAULT_MATH_CLASSIFICATION_PROMPT_TEMPLATE,
    DEFAULT_MATH_MACRO_TOPICS_PROMPT_TEMPLATE,
    DEFAULT_MATH_SUBTOPICS_PROMPT_TEMPLATE,
    DEFAULT_OPEN_QA_FROM_TOPICS_PROMPT_TEMPLATE,
    DEFAULT_PYTHON_CLASSIFICATION_PROMPT_TEMPLATE,
    DEFAULT_PYTHON_MACRO_TOPICS_PROMPT_TEMPLATE,
    DEFAULT_PYTHON_SUBTOPICS_PROMPT_TEMPLATE,
    DEFAULT_REVISE_OPEN_QA_PROMPT_TEMPLATE,
    DEFAULT_REVISE_WRITING_TASK_PROMPT_TEMPLATE,
    DEFAULT_SUBTOPICS_PROMPT_TEMPLATE,
    DEFAULT_WRITING_TASK_PROMPT_TEMPLATE,
    DEFAULT_YAML_CONVERSION_PROMPT_TEMPLATE,
    DIALOGUE_NORMAL_USER_TURN_PROMPT_TEMPLATE,
    MATH_PROBLEM_GENERAL_PROMPT_TEMPLATE,
    PYTHON_PROBLEM_BEGINNER_PROMPT_TEMPLATE,
)


class NemotronGenerator:
    """
    Provides a collection of methods for generating synthetic data
    described in the Nemotron-4 340B Technical Report
    (https://arxiv.org/abs/2406.11704v1) and inspired by the
    UltraChat paper (https://arxiv.org/abs/2305.14233)
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self.client = llm_client

    def _prompt(
        self, model: str, prompt_template: str, prompt_kwargs: dict, model_kwargs: dict
    ) -> List[str]:
        prompt = prompt_template.format(**prompt_kwargs)
        messages = [{"role": "user", "content": prompt}]

        return self.client.query_model(messages=messages, model=model, **model_kwargs)

    def convert_response_to_yaml_list(
        self,
        llm_response: str,
        model: str,
        prompt_template: str = DEFAULT_YAML_CONVERSION_PROMPT_TEMPLATE,
        prompt_kwargs: dict = {},
        model_kwargs: dict = {},
    ) -> List[str]:
        """
        Converts a response of an LLM to a list of strings by querying an LLM
        Args:
            llm_response: The original unformatted response of the LLM
            model: The name of the model that should be used to generate the response.
                Must be available in the LLMClient passed in the constructor.
            prompt_template: A format string of the prompt to use. It must have a {llm_response}
                parameter that will be populated with the llm_response value passed in this function.
            prompt_kwargs: Any additional keyword arguments that should be passed to the prompt template.
                None are needed for the default template.
            model_kwargs: Any additional keyword arguments that should be passed to the LLMClient.query_model call.
        Returns:
            A parsed list of elements from the original LLM response
        """
        prompt_kwargs["llm_response"] = llm_response
        yaml_response = self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        try:
            parsed_response = yaml.safe_load(yaml_response[0])
        except yaml.error.YAMLError as _:
            raise YamlConversionError(
                f"Error parsing yaml response: {yaml_response[0]}"
            )

        if not isinstance(parsed_response, list):
            raise YamlConversionError(
                f"Error: Parsed response was not a list: {parsed_response}"
            )

        for elem in parsed_response:
            if not isinstance(elem, str):
                raise YamlConversionError(
                    f"Error: Parsed response contains non-string elements in list: {parsed_response}"
                )
            if elem not in llm_response:
                raise YamlConversionError(
                    f"Conversion introduced hallucinations. Original response:\n{llm_response}\nConverted response:\n{parsed_response}\nHallucination:\n{elem}"
                )

        return parsed_response

    def generate_macro_topics(
        self,
        n_macro_topics: Union[int, str],
        model: str,
        prompt_template: str = DEFAULT_MACRO_TOPICS_PROMPT_TEMPLATE,
        prompt_kwargs: dict = {},
        model_kwargs: dict = {},
    ) -> List[str]:
        """
        Prompts an LLM to generate a list of macro topics about the world
        Args:
            n_macro_topics: The number of macro topics to generate.
            model: The name of the model that should be used to generate the macro topics.
                Must be available in the LLMClient passed in the constructor.
            prompt_template: A format string of the prompt to use. It must have the following parameters:
                - n_macro_topics: Will be populated with the n_macro_topics passed in this function
            prompt_kwargs: Any additional keyword arguments that should be passed to the prompt template.
                None are needed for the default template.
            model_kwargs: Any additional keyword arguments that should be passed to the LLMClient.query_model call.
        Returns:
            A list of responses from the LLM. The list is only greater than length 1 if n > 1 is set in model_kwargs.
        """
        prompt_kwargs["n_macro_topics"] = n_macro_topics
        macro_topics = self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return macro_topics

    def generate_subtopics(
        self,
        macro_topic: str,
        n_subtopics: Union[int, str],
        model: str,
        prompt_template: str = DEFAULT_SUBTOPICS_PROMPT_TEMPLATE,
        prompt_kwargs: dict = {},
        model_kwargs: dict = {},
    ) -> List[str]:
        """
        Prompts an LLM to generate a list of subtopics relating to a macro topic
        Args:
            macro_topic: The macro topic to generate subtopics for.
            n_subtopics: The number of subtopics to generate per macro topic
            model: The name of the model that should be used to generate the response.
                Must be available in the LLMClient passed in the constructor.
            prompt_template: A format string of the prompt to use. It must have the following parameters:
                - n_subtopics: Will be populated with the n_subtopics passed in this function
                - macro_topic: Will be populated with the macro_topic passed in this function
            prompt_kwargs: Any additional keyword arguments that should be passed to the prompt template.
                None are needed for the default template.
            model_kwargs: Any additional keyword arguments that should be passed to the LLMClient.query_model call.
        Returns:
            A list of responses from the LLM. The list is only greater than length 1 if n > 1 is set in model_kwargs.
        """
        prompt_kwargs["n_subtopics"] = n_subtopics
        prompt_kwargs["macro_topic"] = macro_topic
        subtopics_response = self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return subtopics_response

    def generate_open_qa_from_topic(
        self,
        topic: str,
        n_openlines: Union[str, int],
        model: str,
        prompt_template: str = DEFAULT_OPEN_QA_FROM_TOPICS_PROMPT_TEMPLATE,
        prompt_kwargs: dict = {},
        model_kwargs: dict = {},
    ) -> List[str]:
        """
        Prompts an LLM to generate a list of open Q&A questions based on a topic
        Args:
            topic: The topic to generate questions for.
            n_openlines: The number of questions to generate per topic.
            model: The name of the model that should be used to generate the response.
                Must be available in the LLMClient passed in the constructor.
            prompt_template: A format string of the prompt to use. It must have the following parameters:
                - n_openlines: Will be populated with the n_subtopics passed in this function
                - topic: Will be populated with the topic passed in this function
            prompt_kwargs: Any additional keyword arguments that should be passed to the prompt template.
                None are needed for the default template.
            model_kwargs: Any additional keyword arguments that should be passed to the LLMClient.query_model call.
        Returns:
            A list of responses from the LLM. The list is only greater than length 1 if n > 1 is set in model_kwargs.
        """
        prompt_kwargs["topic"] = topic
        prompt_kwargs["n_openlines"] = n_openlines
        openline_response = self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return openline_response

    def revise_open_qa(
        self,
        openline: str,
        n_revisions: Union[str, int],
        model: str,
        prompt_template: str = DEFAULT_REVISE_OPEN_QA_PROMPT_TEMPLATE,
        prompt_kwargs: dict = {},
        model_kwargs: dict = {},
    ) -> List[str]:
        """
        Prompts an LLM to revise an open Q&A question a given number of times
        Args:
            openline: An openline to revise
            n_revisions: The number of revisions to generate for the question.
            model: The name of the model that should be used to generate the response.
                Must be available in the LLMClient passed in the constructor.
            prompt_template: A format string of the prompt to use. It must have the following parameters:
                - openline: Will be populated with the openline passed in this function
                - n_revisions: Will be populated with the n_revisions passed in this function
            prompt_kwargs: Any additional keyword arguments that should be passed to the prompt template.
                None are needed for the default template.
            model_kwargs: Any additional keyword arguments that should be passed to the LLMClient.query_model call.
        Returns:
            A list of responses from the LLM. The list is only greater than length 1 if n > 1 is set in model_kwargs.
        """
        prompt_kwargs["openline"] = openline
        prompt_kwargs["n_revisions"] = n_revisions
        revisions = self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return revisions

    def generate_writing_tasks(
        self,
        topic: str,
        text_material_type: str,
        n_openlines: Union[str, int],
        model: str,
        prompt_template: str = DEFAULT_WRITING_TASK_PROMPT_TEMPLATE,
        prompt_kwargs: dict = {},
        model_kwargs: dict = {},
    ) -> List[str]:
        """
        Prompts an LLM to generate a list of writing tasks based on a topic and document type
        Args:
            topic: The topic to generate writing tasks for.
            text_material_type: The type of the document the question should ask to generate (e.g., "Email", "Poem")
            n_openlines: The number of tasks to generate per topic and text material pair.
            model: The name of the model that should be used to generate the response.
                Must be available in the LLMClient passed in the constructor.
            prompt_template: A format string of the prompt to use. It must have the following parameters:
                - topic: Will be populated with the topic passed in this function
                - text_material_type: Will be populated with the text_material_type passed in this function
                - n_openlines: Will be populated with the n_openlines passed in this function
            prompt_kwargs: Any additional keyword arguments that should be passed to the prompt template.
                None are needed for the default template.
            model_kwargs: Any additional keyword arguments that should be passed to the LLMClient.query_model call.
        Returns:
            A list of responses from the LLM. The list is only greater than length 1 if n > 1 is set in model_kwargs.
        """
        prompt_kwargs["topic"] = topic
        prompt_kwargs["text_material_type"] = text_material_type
        prompt_kwargs["n_openlines"] = n_openlines
        writing_tasks = self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return writing_tasks

    def revise_writing_tasks(
        self,
        openline: str,
        n_revisions: Union[str, int],
        model: str,
        prompt_template: str = DEFAULT_REVISE_WRITING_TASK_PROMPT_TEMPLATE,
        prompt_kwargs: dict = {},
        model_kwargs: dict = {},
    ) -> List[str]:
        """
        Prompts an LLM to revise a writing task a given number of times
        Args:
            openline: An openline to revise
            n_revisions: The number of revisions to generate for the task.
            model: The name of the model that should be used to generate the response.
                Must be available in the LLMClient passed in the constructor.
            prompt_template: A format string of the prompt to use. It must have the following parameters:
                - openline: Will be populated with the openline passed in this function
                - n_revisions: Will be populated with the n_revisions passed in this function
            prompt_kwargs: Any additional keyword arguments that should be passed to the prompt template.
                None are needed for the default template.
            model_kwargs: Any additional keyword arguments that should be passed to the LLMClient.query_model call.
        Returns:
            A list of responses from the LLM. The list is only greater than length 1 if n > 1 is set in model_kwargs.
        """
        prompt_kwargs["openline"] = openline
        prompt_kwargs["n_revisions"] = n_revisions
        revisions = self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return revisions

    def generate_closed_qa_instructions(
        self,
        document: str,
        n_openlines: Union[str, int],
        model: str,
        prompt_template: str = DEFAULT_CLOSED_QA_PROMPT_TEMPLATE,
        prompt_kwargs: dict = {},
        model_kwargs: dict = {},
    ) -> List[str]:
        """
        Prompts an LLM to generate a list of closed Q&A questions based on a reference document
        Args:
            document: The document to use when generating questions
            n_openlines: The number of questions to generate per document.
            model: The name of the model that should be used to generate the response.
                Must be available in the LLMClient passed in the constructor.
            prompt_template: A format string of the prompt to use. It must have the following parameters:
                - document: Will be populated with the document passed in this function
                - n_openlines: Will be populated with the n_openlines passed in this function
            prompt_kwargs: Any additional keyword arguments that should be passed to the prompt template.
                None are needed for the default template.
            model_kwargs: Any additional keyword arguments that should be passed to the LLMClient.query_model call.
        Returns:
            A list of responses from the LLM. The list is only greater than length 1 if n > 1 is set in model_kwargs.
        """
        prompt_kwargs["document"] = document
        prompt_kwargs["n_openlines"] = n_openlines
        openline_response = self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return openline_response

    def generate_math_macro_topics(
        self,
        n_macro_topics: Union[int, str],
        school_level: str,
        model: str,
        prompt_template: str = DEFAULT_MATH_MACRO_TOPICS_PROMPT_TEMPLATE,
        prompt_kwargs: dict = {},
        model_kwargs: dict = {},
    ) -> List[str]:
        """
        Prompts an LLM to generate a list of macro topics about math
        Args:
            n_macro_topics: The number of macro topics to generate. Can be an integer like 5 or a string like "five".
            school_level: The school level the math questions should be targeted at.
            model: The name of the model that should be used to generate the macro topics.
                Must be available in the LLMClient passed in the constructor.
            prompt_template: A format string of the prompt to use. It must have the following parameters:
                - n_macro_topics: Will be populated with the n_macro_topics passed in this function
                - school_level: Will be populated with the school_level passed in this function
            prompt_kwargs: Any additional keyword arguments that should be passed to the prompt template.
                None are needed for the default template.
            model_kwargs: Any additional keyword arguments that should be passed to the LLMClient.query_model call.
        Returns:
            A list of responses from the LLM. The list is only greater than length 1 if n > 1 is set in model_kwargs.
        """
        prompt_kwargs["n_macro_topics"] = n_macro_topics
        prompt_kwargs["school_level"] = school_level
        macro_topics = self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return macro_topics

    def generate_math_subtopics(
        self,
        macro_topic: str,
        n_subtopics: Union[int, str],
        model: str,
        prompt_template: str = DEFAULT_MATH_SUBTOPICS_PROMPT_TEMPLATE,
        prompt_kwargs: dict = {},
        model_kwargs: dict = {},
    ) -> List[str]:
        """
        Prompts an LLM to generate a list of subtopics relating to a math macro topic
        Args:
            macro_topic: The macro topic to generate subtopics for.
            n_subtopics: The number of subtopics to generate per macro topic
            model: The name of the model that should be used to generate the response.
                Must be available in the LLMClient passed in the constructor.
            prompt_template: A format string of the prompt to use. It must have the following parameters:
                - n_subtopics: Will be populated with the n_subtopics passed in this function
                - macro_topic: Will be populated with the macro_topic passed in this function
            prompt_kwargs: Any additional keyword arguments that should be passed to the prompt template.
                None are needed for the default template.
            model_kwargs: Any additional keyword arguments that should be passed to the LLMClient.query_model call.
        Returns:
            A list of responses from the LLM. The list is only greater than length 1 if n > 1 is set in model_kwargs.
        """
        prompt_kwargs["n_subtopics"] = n_subtopics
        prompt_kwargs["macro_topic"] = macro_topic
        subtopics_response = self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return subtopics_response

    def classify_math_entity(
        self,
        entity: str,
        model: str,
        prompt_template: str = DEFAULT_MATH_CLASSIFICATION_PROMPT_TEMPLATE,
        prompt_kwargs: dict = {},
        model_kwargs={},
    ) -> List[str]:
        """
        Prompts an LLM to classify if an entity is related to math
        Args:
            entity: The entity to classify
            model: The name of the model that should be used to generate the response.
                Must be available in the LLMClient passed in the constructor.
            prompt_template: A format string of the prompt to use. It must have the following parameters:
                - entity: Will be populated with the entity passed in this function
            prompt_kwargs: Any additional keyword arguments that should be passed to the prompt template.
                None are needed for the default template.
            model_kwargs: Any additional keyword arguments that should be passed to the LLMClient.query_model call.
        Returns:
            A list of responses from the LLM. The list is only greater than length 1 if n > 1 is set in model_kwargs.
        """
        prompt_kwargs["entity"] = entity
        classification_response = self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return classification_response

    def generate_math_problem(
        self,
        topic: str,
        n_openlines: Union[str, int],
        model: str,
        prompt_template: str = MATH_PROBLEM_GENERAL_PROMPT_TEMPLATE,
        prompt_kwargs: dict = {},
        model_kwargs: dict = {},
    ) -> List[str]:
        """
        Prompts an LLM to generate a list of math problems based on a topic
        Args:
            topic: The topic to generate problems for.
            n_openlines: The number of problems to generate per topic.
            model: The name of the model that should be used to generate the response.
                Must be available in the LLMClient passed in the constructor.
            prompt_template: A format string of the prompt to use. It must have the following parameters:
                - n_openlines: Will be populated with the n_subtopics passed in this function
                - topic: Will be populated with the topic passed in this function
                Some example templates found in nemo_curator.synthetic include:
                - MATH_PROBLEM_GENERAL_PROMPT_TEMPLATE
                - MATH_PROBLEM_BEGINNER_PROMPT_TEMPLATE
            prompt_kwargs: Any additional keyword arguments that should be passed to the prompt template.
                None are needed for the default template.
            model_kwargs: Any additional keyword arguments that should be passed to the LLMClient.query_model call.
        Returns:
            A list of responses from the LLM. The list is only greater than length 1 if n > 1 is set in model_kwargs.
        """
        prompt_kwargs["topic"] = topic
        prompt_kwargs["n_openlines"] = n_openlines
        openline_response = self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return openline_response

    def generate_python_macro_topics(
        self,
        n_macro_topics: Union[int, str],
        model: str,
        prompt_template: str = DEFAULT_PYTHON_MACRO_TOPICS_PROMPT_TEMPLATE,
        prompt_kwargs: dict = {},
        model_kwargs: dict = {},
    ) -> List[str]:
        """
        Prompts an LLM to generate a list of macro topics about the Python programming language
        Args:
            n_macro_topics: The number of macro topics to generate. Can be an integer like 5 or a string like "five".
            model: The name of the model that should be used to generate the macro topics.
                Must be available in the LLMClient passed in the constructor.
            prompt_template: A format string of the prompt to use. It must have the following parameters:
                - n_macro_topics: Will be populated with the n_macro_topics passed in this function
            prompt_kwargs: Any additional keyword arguments that should be passed to the prompt template.
                None are needed for the default template.
            model_kwargs: Any additional keyword arguments that should be passed to the LLMClient.query_model call.
        Returns:
            A list of responses from the LLM. The list is only greater than length 1 if n > 1 is set in model_kwargs.
        """
        prompt_kwargs["n_macro_topics"] = n_macro_topics
        macro_topics = self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return macro_topics

    def generate_python_subtopics(
        self,
        macro_topic: str,
        n_subtopics: Union[int, str],
        model: str,
        prompt_template: str = DEFAULT_PYTHON_SUBTOPICS_PROMPT_TEMPLATE,
        prompt_kwargs: dict = {},
        model_kwargs: dict = {},
    ) -> List[str]:
        """
        Prompts an LLM to generate a list of subtopics relating to a Python macro topic
        Args:
            macro_topic: The macro topic to generate subtopics for.
            n_subtopics: The number of subtopics to generate per macro topic
            model: The name of the model that should be used to generate the response.
                Must be available in the LLMClient passed in the constructor.
            prompt_template: A format string of the prompt to use. It must have the following parameters:
                - n_subtopics: Will be populated with the n_subtopics passed in this function
                - macro_topic: Will be populated with the macro_topic passed in this function
            prompt_kwargs: Any additional keyword arguments that should be passed to the prompt template.
                None are needed for the default template.
            model_kwargs: Any additional keyword arguments that should be passed to the LLMClient.query_model call.
        Returns:
            A list of responses from the LLM. The list is only greater than length 1 if n > 1 is set in model_kwargs.
        """
        prompt_kwargs["n_subtopics"] = n_subtopics
        prompt_kwargs["macro_topic"] = macro_topic
        subtopics_response = self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return subtopics_response

    def classify_python_entity(
        self,
        entity: str,
        model: str,
        prompt_template: str = DEFAULT_PYTHON_CLASSIFICATION_PROMPT_TEMPLATE,
        prompt_kwargs: dict = {},
        model_kwargs: dict = {},
    ) -> List[str]:
        """
        Prompts an LLM to classify if an entity is related to Python
        Args:
            entity: The entity to classify
            model: The name of the model that should be used to generate the response.
                Must be available in the LLMClient passed in the constructor.
            prompt_template: A format string of the prompt to use. It must have the following parameters:
                - entity: Will be populated with the entity passed in this function
            prompt_kwargs: Any additional keyword arguments that should be passed to the prompt template.
                None are needed for the default template.
            model_kwargs: Any additional keyword arguments that should be passed to the LLMClient.query_model call.
        Returns:
            A list of responses from the LLM. The list is only greater than length 1 if n > 1 is set in model_kwargs.
        """
        prompt_kwargs["entity"] = entity
        classification_response = self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return classification_response

    def generate_python_problem(
        self,
        topic: str,
        n_openlines: Union[str, int],
        model: str,
        language="Python",
        prompt_template: str = PYTHON_PROBLEM_BEGINNER_PROMPT_TEMPLATE,
        prompt_kwargs: dict = {},
        model_kwargs: dict = {},
    ) -> List[str]:
        """
        Prompts an LLM to generate a list of coding problems based on a topic
        Args:
            topic: The topic to generate problems for.
            n_openlines: The number of problems to generate per topic.
            model: The name of the model that should be used to generate the response.
                Must be available in the LLMClient passed in the constructor.
            language: The programming language to target when generating these questions.
            prompt_template: A format string of the prompt to use. It must have the following parameters:
                - n_openlines: Will be populated with the n_subtopics passed in this function
                - topic: Will be populated with the topic passed in this function
                - language: Will be populated with the language passed in this function
                Some example templates found in nemo_curator.synthetic include:
                - PYTHON_PROBLEM_BEGINNER_PROMPT_TEMPLATE
                - PYTHON_PROBLEM_INTERMEDIATE_PROMPT_TEMPLATE
                - PYTHON_PROBLEM_ADVANCED_PROMPT_TEMPLATE
            prompt_kwargs: Any additional keyword arguments that should be passed to the prompt template.
                None are needed for the default template.
            model_kwargs: Any additional keyword arguments that should be passed to the LLMClient.query_model call.
        Returns:
            A list of responses from the LLM. The list is only greater than length 1 if n > 1 is set in model_kwargs.
        """
        prompt_kwargs["topic"] = topic
        prompt_kwargs["n_openlines"] = n_openlines
        prompt_kwargs["language"] = language
        openline_response = self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return openline_response

    def generate_dialogue(
        self,
        openline: str,
        user_model: str,
        assistant_model: str,
        n_user_turns: int = 3,
        prompt_template: str = DIALOGUE_NORMAL_USER_TURN_PROMPT_TEMPLATE,
        prompt_kwargs: dict = {},
        user_model_kwargs: dict = {},
        assistant_model_kwargs: dict = {},
    ) -> List[dict]:
        """
        Prompts an LLM to generate a dialogue based on a given openline.
        The LLM will alternate impersonating the user and the assistant.
        Args:
            openline: The openline that will comprise the first user turn.
            user_model: The model that will be impersonating the user.
                Must be available in the LLMClient passed in the constructor.
            assistant_model: The model that will be impersonating the assistant
                Must be available in the LLMClient passed in the constructor.
            n_user_turns: The number of user turns to go through. The openline counts as 1 user turn.
                Therefore, if there are 3 user turns, 2 will be generated by the LLM impersonating the user.
            prompt_template: A format string of the prompt to use when impersonating the user.
                It must have the following parameters:
                - converstation_history: Will be populated with a formatted history of the dialogue up to that point.
                Some example templates found in nemo_curator.synthetic include:
                - DIALOGUE_NORMAL_USER_TURN_PROMPT_TEMPLATE
                - DIALOGUE_COMPLEX_USER_TURN_PROMPT_TEMPLATE
                - DIALOGUE_CONCISE_USER_TURN_PROMPT_TEMPLATE
            prompt_kwargs: Any additional keyword arguments that should be passed to the prompt template.
                None are needed for the default template.
            user_model_kwargs: Any additional keyword arguments that should be passed to the
                LLMClient.query_model call for the user.
            assistant_model_kwargs: Any additional keyword arguments that should be passed to the
                LLMClient.query_model call for the assistant.
        Returns:
            A conversation between a User and Assistant
        """
        conversation_history = [{"role": "user", "content": openline}]
        first_assistant_response = self.client.query_model(
            messages=conversation_history,
            model=assistant_model,
            **assistant_model_kwargs,
        )[0]
        conversation_history.append(
            {"role": "assistant", "content": first_assistant_response}
        )
        for _ in range(n_user_turns - 1):
            user_response = self._impersonate_user(
                conversation_history=conversation_history,
                model=user_model,
                prompt_template=prompt_template,
                prompt_kwargs=prompt_kwargs,
                model_kwargs=user_model_kwargs,
            )
            conversation_history.append({"role": "user", "content": user_response})
            assistant_response = self.client.query_model(
                messages=conversation_history,
                model=assistant_model,
                **assistant_model_kwargs,
            )[0]
            conversation_history.append(
                {"role": "assistant", "content": assistant_response}
            )

        return conversation_history

    def generate_two_turn_prompt(
        self,
        openline: str,
        user_model: str,
        assistant_model: str,
        prompt_template: str = DIALOGUE_NORMAL_USER_TURN_PROMPT_TEMPLATE,
        prompt_kwargs: dict = {},
        user_model_kwargs: dict = {},
        assistant_model_kwargs: dict = {},
    ) -> List[dict]:
        """
        Prompts an LLM to generate a response as an assistant, then as the user based on a given openline.
        The conversation will look like "User -> Assistant -> User"
        Args:
            openline: The openline that will comprise the first user turn.
            user_model: The model that will be impersonating the user.
                Must be available in the LLMClient passed in the constructor.
            assistant_model: The model that will be impersonating the assistant
                Must be available in the LLMClient passed in the constructor.
            prompt_template: A format string of the prompt to use when impersonating the user.
                It must have the following parameters:
                - converstation_history: Will be populated with a formatted history of the dialogue up to that point.
                Some example templates found in nemo_curator.synthetic include:
                - DIALOGUE_NORMAL_USER_TURN_PROMPT_TEMPLATE
                - DIALOGUE_COMPLEX_USER_TURN_PROMPT_TEMPLATE
                - DIALOGUE_CONCISE_USER_TURN_PROMPT_TEMPLATE
            prompt_kwargs: Any additional keyword arguments that should be passed to the prompt template.
                None are needed for the default template.
            user_model_kwargs: Any additional keyword arguments that should be passed to the
                LLMClient.query_model call for the user.
            assistant_model_kwargs: Any additional keyword arguments that should be passed to the
                LLMClient.query_model call for the assistant.
        Returns:
            A conversation between a User and Assistant
        """
        conversation_history = [{"role": "user", "content": openline}]
        first_assistant_response = self.client.query_model(
            messages=conversation_history,
            model=assistant_model,
            **assistant_model_kwargs,
        )[0]
        conversation_history.append(
            {"role": "assistant", "content": first_assistant_response}
        )

        user_response = self._impersonate_user(
            conversation_history=conversation_history,
            model=user_model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=user_model_kwargs,
        )
        conversation_history.append({"role": "user", "content": user_response})

        return conversation_history

    def _impersonate_user(
        self,
        conversation_history: List[dict],
        model: str,
        prompt_template: str,
        prompt_kwargs: dict,
        model_kwargs: dict,
    ) -> str:
        # Convert the conversation history to a string
        history_str = ""
        for turn in conversation_history:
            history_str += f"{turn['role'].capitalize()}: {turn['content']}"
        prompt_kwargs["conversation_history"] = history_str
        response = self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return response[0]

    def run_open_qa_pipeline(
        self,
        n_macro_topics: Union[str, int],
        n_subtopics: Union[str, int],
        n_openlines: Union[str, int],
        n_revisions: Union[str, int],
        model: str,
        macro_topic_prompt_template: str = DEFAULT_MACRO_TOPICS_PROMPT_TEMPLATE,
        subtopic_prompt_template: str = DEFAULT_SUBTOPICS_PROMPT_TEMPLATE,
        open_qa_from_topics_prompt_template: str = DEFAULT_OPEN_QA_FROM_TOPICS_PROMPT_TEMPLATE,
        revise_open_qa_prompt_template: str = DEFAULT_REVISE_OPEN_QA_PROMPT_TEMPLATE,
        yaml_conversion_prompt_template: str = DEFAULT_YAML_CONVERSION_PROMPT_TEMPLATE,
        base_model_kwargs: dict = {},
        conversion_model_kwargs: dict = {},
        additional_macro_topics: List[str] = [],
        additional_subtopics: List[str] = [],
        ignore_conversion_failure: bool = False,
        combine_topics: bool = True,
    ) -> List[str]:
        """
        Runs a pipeline for automatically generating Open Q&A openlines for a dialogue
        Args:
            n_macro_topics: The number of macro topics to generate
            n_subtopics: The number of subtopics to generate per macro topic
            n_openlines: The number of questions to generate per topic.
            n_revisions: The number of revisions to generate per original question.
            model: The name of the model that should be used to generate all the responses.
                Must be available in the LLMClient passed in the constructor.
            macro_topic_prompt_template: A format string of the prompt to use. It must have the following parameters:
                - n_macro_topics: Will be populated with the n_macro_topics passed in this function
                No additional parameters may be passed to this prompt template.
            subtopic_prompt_template: A format string of the prompt to use. It must have the following parameters:
                - n_subtopics: Will be populated with the n_subtopics passed in this function
                - macro_topic: Will be populated with a generated macro topic
                No additional parameters may be passed to this prompt template.
            open_qa_from_topics_prompt_template: A format string of the prompt to use. It must have the following parameters:
                - n_openlines: Will be populated with the n_openlines passed in this function
                - topic: Will be populated with a generated topic
                No additional parameters may be passed to this prompt template.
            revise_open_qa_prompt_template: A format string of the prompt to use. It must have the following parameters:
                - n_revisions: Will be populated with the n_revisions passed in this function
                - openline: Will be populated with a generated open Q&A openline
                No additional parameters may be passed to this prompt template.
            yaml_conversion_prompt_template: A format string of the prompt to use. It must have the following parameters:
                - llm_response: Will be populated with the raw LLM response from each stage of the pipeline
                No additional parameters may be passed to this prompt template.
            base_model_kwargs: Any additional keyword arguments that should be passed to the
                LLMClient.query_model call for the normal stages of the pipeline.
            conversion_model_kwargs: Any additional keyword arguments that should be passed to the
                LLMClient.query_model call for the yaml conversion stages of the pipeline.
            ignore_conversion_failure: Ignores yaml conversion failures when able and discards the data
                that conversion was attempted on
            combine_topics: If True, mixes the macro topics with the subtopics when generating openlines.
                If False, only the subtopics are used.
        Returns:
            A list of synthetically generated open Q&A prompts
        """
        # Generate the macro topics
        responses = self.generate_macro_topics(
            n_macro_topics=n_macro_topics,
            model=model,
            model_kwargs=base_model_kwargs,
            prompt_template=macro_topic_prompt_template,
        )
        macro_topics = self.convert_response_to_yaml_list(
            responses[0],
            model=model,
            prompt_template=yaml_conversion_prompt_template,
            model_kwargs=conversion_model_kwargs,
        )
        if len(macro_topics) != n_macro_topics and not ignore_conversion_failure:
            raise YamlConversionError(
                f"Error: Length of macro topics {len(macro_topics)} does not match desired n_macro_topics {n_macro_topics}: {macro_topics}"
            )
        macro_topics.extend(additional_macro_topics)

        # Generate the subtopics
        raw_topics = [
            self.generate_subtopics(
                macro_topic=macro_topic,
                n_subtopics=n_subtopics,
                model=model,
                model_kwargs=base_model_kwargs,
                prompt_template=subtopic_prompt_template,
            )[0]
            for macro_topic in macro_topics
        ]
        topic_list = []
        for topic in raw_topics:
            try:
                parsed_topics = self.convert_response_to_yaml_list(
                    topic,
                    model=model,
                    prompt_template=yaml_conversion_prompt_template,
                    model_kwargs=conversion_model_kwargs,
                )
                if len(parsed_topics) != n_subtopics:
                    raise YamlConversionError(
                        f"Error: Length of subtopics {len(parsed_topics)} does not match desired n_subtopics {n_subtopics}: {parsed_topics}"
                    )
                topic_list.extend(parsed_topics)
            except YamlConversionError as e:
                if ignore_conversion_failure:
                    continue
                else:
                    raise e
        topic_list.extend(additional_subtopics)

        # Mix the macro topics with the subtopics
        if combine_topics:
            topic_list.extend(macro_topics)

        # Generate the openlines
        raw_lines = [
            self.generate_open_qa_from_topic(
                topic=t,
                n_openlines=n_openlines,
                model=model,
                model_kwargs=base_model_kwargs,
                prompt_template=open_qa_from_topics_prompt_template,
            )[0]
            for t in topic_list
        ]
        openlines = []
        for line in raw_lines:
            try:
                parsed_line = self.convert_response_to_yaml_list(
                    line,
                    model=model,
                    prompt_template=yaml_conversion_prompt_template,
                    model_kwargs=conversion_model_kwargs,
                )
                if len(parsed_line) != n_openlines:
                    raise YamlConversionError(
                        f"Error: Length of openlines {len(parsed_line)} does not match desired n_openlines {n_openlines}: {parsed_line}"
                    )
                openlines.extend(parsed_line)
            except YamlConversionError as e:
                if ignore_conversion_failure:
                    continue
                else:
                    raise e

        # Revise the openlines
        raw_revisions = [
            self.revise_open_qa(
                openline=line,
                n_revisions=n_revisions,
                model=model,
                model_kwargs=base_model_kwargs,
                prompt_template=revise_open_qa_prompt_template,
            )[0]
            for line in openlines
        ]
        revised_openlines = []
        for line in raw_revisions:
            try:
                parsed_revision = self.convert_response_to_yaml_list(
                    line,
                    model=model,
                    prompt_template=yaml_conversion_prompt_template,
                    model_kwargs=conversion_model_kwargs,
                )
                if len(parsed_revision) != n_revisions:
                    raise YamlConversionError(
                        f"Error: Length of revisions {len(parsed_revision)} does not match desired n_revisions {n_revisions}: {parsed_revision}"
                    )
                revised_openlines.extend(parsed_revision)
            except YamlConversionError as e:
                if ignore_conversion_failure:
                    continue
                else:
                    raise e

        return revised_openlines

    def run_writing_pipeline(
        self,
        topics: List[str],
        text_material_types: List[str],
        n_openlines: Union[str, int],
        n_revisions: Union[str, int],
        model: str,
        writing_task_prompt_template: str = DEFAULT_WRITING_TASK_PROMPT_TEMPLATE,
        revise_writing_task_prompt_template: str = DEFAULT_REVISE_WRITING_TASK_PROMPT_TEMPLATE,
        yaml_conversion_prompt_template: str = DEFAULT_YAML_CONVERSION_PROMPT_TEMPLATE,
        base_model_kwargs: dict = {},
        conversion_model_kwargs: dict = {},
        ignore_conversion_failure: bool = False,
    ) -> List[str]:
        """
        Runs a pipeline for automatically generating writing task openlines for a dialogue
        Args:
            topics: A list of topics to generate tasks for
            text_material_types: A list of writing material types, like "Essay" or "Blog post"
            n_openlines: The number of tasks to generate per (topic, text_material_type) pair.
            n_revisions: The number of revisions to generate per original task.
            model: The name of the model that should be used to generate all the responses.
                Must be available in the LLMClient passed in the constructor.
            writing_task_prompt_template: A format string of the prompt to use. It must have the following parameters:
                - n_openlines: Will be populated with the n_openlines passed in this function
                - topic: Will be populated with one element of the topics list passed in this function
                - text_material_type: Will be populated with one element of the text_material_types list passed in this function
                No additional parameters may be passed to this prompt template.
            revise_writing_task_prompt_template: A format string of the prompt to use. It must have the following parameters:
                - n_revisions: Will be populated with the n_revisions passed in this function
                - openline: Will be populated with one of the writing tasks generated in the pipeline.
                No additional parameters may be passed to this prompt template.
            yaml_conversion_prompt_template: A format string of the prompt to use. It must have the following parameters:
                - llm_response: Will be populated with the raw LLM response from each stage of the pipeline
                No additional parameters may be passed to this prompt template.
            base_model_kwargs: Any additional keyword arguments that should be passed to the
                LLMClient.query_model call for the normal stages of the pipeline.
            conversion_model_kwargs: Any additional keyword arguments that should be passed to the
                LLMClient.query_model call for the yaml conversion stages of the pipeline.
            ignore_conversion_failure: Ignores yaml conversion failures when able and discards the data
                that conversion was attempted on
        Returns:
            A list of synthetically generated writing task prompts
        """
        # Generate the tasks
        writing_tasks = []
        for topic in topics:
            for material in text_material_types:
                raw_tasks = self.generate_writing_tasks(
                    topic=topic,
                    text_material_type=material,
                    n_openlines=n_openlines,
                    model=model,
                    model_kwargs=base_model_kwargs,
                    prompt_template=writing_task_prompt_template,
                )[0]
                try:
                    parsed_tasks = self.convert_response_to_yaml_list(
                        raw_tasks,
                        model=model,
                        prompt_template=yaml_conversion_prompt_template,
                        model_kwargs=conversion_model_kwargs,
                    )
                    if len(parsed_tasks) != n_openlines:
                        raise YamlConversionError(
                            f"Error: Length of writing tasks {len(parsed_tasks)} does not match desired n_openlines {n_openlines}: {parsed_tasks}"
                        )
                    writing_tasks.extend(parsed_tasks)
                except YamlConversionError as e:
                    if ignore_conversion_failure:
                        continue
                    else:
                        raise e

        # Revise the tasks
        raw_revisions = [
            self.revise_writing_tasks(
                openline=line,
                n_revisions=n_revisions,
                model=model,
                model_kwargs=base_model_kwargs,
                prompt_template=revise_writing_task_prompt_template,
            )[0]
            for line in writing_tasks
        ]
        revised_openlines = []
        for line in raw_revisions:
            try:
                parsed_revision = self.convert_response_to_yaml_list(
                    line,
                    model=model,
                    prompt_template=yaml_conversion_prompt_template,
                    model_kwargs=conversion_model_kwargs,
                )
                if len(parsed_revision) != n_revisions:
                    raise YamlConversionError(
                        f"Error: Length of revisions {len(parsed_revision)} does not match desired n_revisions {n_revisions}: {parsed_revision}"
                    )
                revised_openlines.extend(parsed_revision)
            except YamlConversionError as e:
                if ignore_conversion_failure:
                    continue
                else:
                    raise e

        return revised_openlines

    def run_closed_qa_pipeline(
        self,
        documents: List[str],
        n_openlines: Union[str, int],
        model: str,
        closed_qa_prompt_template: str = DEFAULT_CLOSED_QA_PROMPT_TEMPLATE,
        yaml_conversion_prompt_template: str = DEFAULT_YAML_CONVERSION_PROMPT_TEMPLATE,
        base_model_kwargs: dict = {},
        conversion_model_kwargs: dict = {},
        ignore_conversion_failure: bool = False,
    ) -> List[Tuple[int, str]]:
        """
        Runs a pipeline for automatically generating closed Q&A openlines for a dialogue
        Args:
            documents: A list of documents to generate closed Q&A questions for
            n_openlines: The number of questions to generate per document.
            model: The name of the model that should be used to generate all the responses.
                Must be available in the LLMClient passed in the constructor.
            closed_qa_prompt_template: A format string of the prompt to use. It must have the following parameters:
                - n_openlines: Will be populated with the n_openlines passed in this function
                - document: Will be populated with one element of the documents list passed in this function
                No additional parameters may be passed to this prompt template.
            yaml_conversion_prompt_template: A format string of the prompt to use. It must have the following parameters:
                - llm_response: Will be populated with the raw LLM response from each stage of the pipeline
                No additional parameters may be passed to this prompt template.
            base_model_kwargs: Any additional keyword arguments that should be passed to the
                LLMClient.query_model call for the normal stages of the pipeline.
            conversion_model_kwargs: Any additional keyword arguments that should be passed to the
                LLMClient.query_model call for the yaml conversion stages of the pipeline.
            ignore_conversion_failure: Ignores yaml conversion failures when able and discards the data
                that conversion was attempted on
        Returns:
            A list of pairs where the first element represents the index of the document used to generate the question in the documents list
            and the second element represents a synthetically generated closed Q&A prompt. Example: [(0, "Summarize this document"), ...]
        """
        raw_instructions = [
            self.generate_closed_qa_instructions(
                document=document,
                n_openlines=n_openlines,
                model=model,
                model_kwargs=base_model_kwargs,
                prompt_template=closed_qa_prompt_template,
            )[0]
            for document in documents
        ]
        document_openline_pairs = []
        for i, instruction in enumerate(raw_instructions):
            try:
                parsed_instructions = self.convert_response_to_yaml_list(
                    instruction,
                    model=model,
                    prompt_template=yaml_conversion_prompt_template,
                    model_kwargs=conversion_model_kwargs,
                )
                if len(parsed_instructions) != n_openlines:
                    raise YamlConversionError(
                        f"Error: Length of openlines {len(parsed_instructions)} does not match desired n_openlines {n_openlines}: {parsed_instructions}"
                    )
                document_openline_pairs.extend(
                    [(i, inst) for inst in parsed_instructions]
                )
            except YamlConversionError as e:
                if ignore_conversion_failure:
                    continue
                else:
                    raise e

        return document_openline_pairs

    def run_math_pipeline(
        self,
        n_macro_topics: Union[str, int],
        school_level: str,
        n_subtopics: Union[str, int],
        n_openlines: Union[str, int],
        model: str,
        macro_topic_prompt_template: str = DEFAULT_MATH_MACRO_TOPICS_PROMPT_TEMPLATE,
        subtopic_prompt_template: str = DEFAULT_MATH_SUBTOPICS_PROMPT_TEMPLATE,
        math_problem_prompt_template: str = MATH_PROBLEM_GENERAL_PROMPT_TEMPLATE,
        yaml_conversion_prompt_template: str = DEFAULT_YAML_CONVERSION_PROMPT_TEMPLATE,
        base_model_kwargs: dict = {},
        conversion_model_kwargs: dict = {},
        additional_macro_topics: List[str] = [],
        additional_subtopics: List[str] = [],
        ignore_conversion_failure: bool = False,
        combine_topics: bool = True,
    ) -> List[str]:
        """
        Runs a pipeline for automatically generating math questions for a dialogue
        Args:
            n_macro_topics: The number of macro topics to generate.
            school_level: The school level to target when generating macro topics.
            n_subtopics: The number of subtopics to generate per macro topic.
            n_openlines: The number of questions to generate per topic.
            model: The name of the model that should be used to generate all the responses.
                Must be available in the LLMClient passed in the constructor.
            macro_topic_prompt_template: A format string of the prompt to use. It must have the following parameters:
                - n_macro_topics: Will be populated with the n_macro_topics passed in this function
                - school_level: Will be populated with the school_level passed in this function
                No additional parameters may be passed to this prompt template.
            subtopic_prompt_template: A format string of the prompt to use. It must have the following parameters:
                - n_subtopics: Will be populated with the n_subtopics passed in this function
                - macro_topic: Will be populated with a generated macro topic
                No additional parameters may be passed to this prompt template.
            math_problem_prompt_template: A format string of the prompt to use. It must have the following parameters:
                - n_openlines: Will be populated with the n_openlines passed in this function
                - topic: Will be populated with a generated topic
                No additional parameters may be passed to this prompt template.
                Some example templates found in nemo_curator.synthetic include:
                - MATH_PROBLEM_GENERAL_PROMPT_TEMPLATE
                - MATH_PROBLEM_BEGINNER_PROMPT_TEMPLATE
            yaml_conversion_prompt_template: A format string of the prompt to use. It must have the following parameters:
                - llm_response: Will be populated with the raw LLM response from each stage of the pipeline
                No additional parameters may be passed to this prompt template.
            base_model_kwargs: Any additional keyword arguments that should be passed to the
                LLMClient.query_model call for the normal stages of the pipeline.
            conversion_model_kwargs: Any additional keyword arguments that should be passed to the
                LLMClient.query_model call for the yaml conversion stages of the pipeline.
            ignore_conversion_failure: Ignores yaml conversion failures when able and discards the data
                that conversion was attempted on
            combine_topics: If True, mixes the macro topics with the subtopics when generating openlines.
                If False, only the subtopics are used.
        Returns:
            A list of synthetically generated math prompts
        """
        # Generate the macro topics
        responses = self.generate_math_macro_topics(
            n_macro_topics=n_macro_topics,
            school_level=school_level,
            model=model,
            model_kwargs=base_model_kwargs,
            prompt_template=macro_topic_prompt_template,
        )
        macro_topics = self.convert_response_to_yaml_list(
            responses[0],
            model=model,
            prompt_template=yaml_conversion_prompt_template,
            model_kwargs=conversion_model_kwargs,
        )
        if len(macro_topics) != n_macro_topics and not ignore_conversion_failure:
            raise YamlConversionError(
                f"Error: Length of macro topics {len(macro_topics)} does not match desired n_macro_topics {n_macro_topics}: {macro_topics}"
            )
        macro_topics.extend(additional_macro_topics)

        # Generate the subtopics
        raw_topics = [
            self.generate_math_subtopics(
                macro_topic=macro_topic,
                n_subtopics=n_subtopics,
                model=model,
                model_kwargs=base_model_kwargs,
                prompt_template=subtopic_prompt_template,
            )[0]
            for macro_topic in macro_topics
        ]
        topic_list = []
        for topic in raw_topics:
            try:
                parsed_topics = self.convert_response_to_yaml_list(
                    topic,
                    model=model,
                    prompt_template=yaml_conversion_prompt_template,
                    model_kwargs=conversion_model_kwargs,
                )
                if len(parsed_topics) != n_subtopics:
                    raise YamlConversionError(
                        f"Error: Length of subtopics {len(parsed_topics)} does not match desired n_subtopics {n_subtopics}: {parsed_topics}"
                    )
                topic_list.extend(parsed_topics)
            except YamlConversionError as e:
                if ignore_conversion_failure:
                    continue
                else:
                    raise e
        topic_list.extend(additional_subtopics)

        # Mix the macro topics with the subtopics
        if combine_topics:
            topic_list.extend(macro_topics)

        # Generate the openlines
        raw_lines = [
            self.generate_math_problem(
                topic=t,
                n_openlines=n_openlines,
                model=model,
                model_kwargs=base_model_kwargs,
                prompt_template=math_problem_prompt_template,
            )[0]
            for t in topic_list
        ]
        openlines = []
        for line in raw_lines:
            try:
                parsed_line = self.convert_response_to_yaml_list(
                    line,
                    model=model,
                    prompt_template=yaml_conversion_prompt_template,
                    model_kwargs=conversion_model_kwargs,
                )
                if len(parsed_line) != n_openlines:
                    raise YamlConversionError(
                        f"Error: Length of openlines {len(parsed_line)} does not match desired n_openlines {n_openlines}: {parsed_line}"
                    )
                openlines.extend(parsed_line)
            except YamlConversionError as e:
                if ignore_conversion_failure:
                    continue
                else:
                    raise e

        return openlines

    def run_python_pipeline(
        self,
        n_macro_topics: Union[str, int],
        n_subtopics: Union[str, int],
        n_openlines: Union[str, int],
        model: str,
        macro_topic_prompt_template: str = DEFAULT_PYTHON_MACRO_TOPICS_PROMPT_TEMPLATE,
        subtopic_prompt_template: str = DEFAULT_PYTHON_SUBTOPICS_PROMPT_TEMPLATE,
        python_problem_prompt_template: str = PYTHON_PROBLEM_BEGINNER_PROMPT_TEMPLATE,
        yaml_conversion_prompt_template: str = DEFAULT_YAML_CONVERSION_PROMPT_TEMPLATE,
        base_model_kwargs: dict = {},
        conversion_model_kwargs: dict = {},
        additional_macro_topics: List[str] = [],
        additional_subtopics: List[str] = [],
        ignore_conversion_failure: bool = False,
        combine_topics: bool = True,
    ) -> List[str]:
        """
        Runs a pipeline for automatically generating Python questions for a dialogue
        Args:
            n_macro_topics: The number of macro topics to generate.
            n_subtopics: The number of subtopics to generate per macro topic.
            n_openlines: The number of questions to generate per topic.
            model: The name of the model that should be used to generate all the responses.
                Must be available in the LLMClient passed in the constructor.
            macro_topic_prompt_template: A format string of the prompt to use. It must have the following parameters:
                - n_macro_topics: Will be populated with the n_macro_topics passed in this function
                No additional parameters may be passed to this prompt template.
            subtopic_prompt_template: A format string of the prompt to use. It must have the following parameters:
                - n_subtopics: Will be populated with the n_subtopics passed in this function
                - macro_topic: Will be populated with a generated macro topic
                No additional parameters may be passed to this prompt template.
            python_problem_prompt_template: A format string of the prompt to use. It must have the following parameters:
                - n_openlines: Will be populated with the n_openlines passed in this function
                - language: Will be populated with "Python"
                - topic: Will be populated with a generated topic
                No additional parameters may be passed to this prompt template.
                Some example templates found in nemo_curator.synthetic include:
                - PYTHON_PROBLEM_BEGINNER_PROMPT_TEMPLATE
                - PYTHON_PROBLEM_INTERMEDIATE_PROMPT_TEMPLATE
                - PYTHON_PROBLEM_ADVANCED_PROMPT_TEMPLATE
            yaml_conversion_prompt_template: A format string of the prompt to use. It must have the following parameters:
                - llm_response: Will be populated with the raw LLM response from each stage of the pipeline
                No additional parameters may be passed to this prompt template.
            base_model_kwargs: Any additional keyword arguments that should be passed to the
                LLMClient.query_model call for the normal stages of the pipeline.
            conversion_model_kwargs: Any additional keyword arguments that should be passed to the
                LLMClient.query_model call for the yaml conversion stages of the pipeline.
            ignore_conversion_failure: Ignores yaml conversion failures when able and discards the data
                that conversion was attempted on
            combine_topics: If True, mixes the macro topics with the subtopics when generating openlines.
                If False, only the subtopics are used.
        Returns:
            A list of synthetically generated Python prompts
        """
        # Generate the macro topics
        responses = self.generate_python_macro_topics(
            n_macro_topics=n_macro_topics,
            model=model,
            model_kwargs=base_model_kwargs,
            prompt_template=macro_topic_prompt_template,
        )
        macro_topics = self.convert_response_to_yaml_list(
            responses[0],
            model=model,
            prompt_template=yaml_conversion_prompt_template,
            model_kwargs=conversion_model_kwargs,
        )
        if len(macro_topics) != n_macro_topics and not ignore_conversion_failure:
            raise YamlConversionError(
                f"Error: Length of macro topics {len(macro_topics)} does not match desired n_macro_topics {n_macro_topics}: {macro_topics}"
            )
        macro_topics.extend(additional_macro_topics)

        # Generate the subtopics
        raw_topics = [
            self.generate_python_subtopics(
                macro_topic=macro_topic,
                n_subtopics=n_subtopics,
                model=model,
                model_kwargs=base_model_kwargs,
                prompt_template=subtopic_prompt_template,
            )[0]
            for macro_topic in macro_topics
        ]
        topic_list = []
        for topic in raw_topics:
            try:
                parsed_topics = self.convert_response_to_yaml_list(
                    topic,
                    model=model,
                    prompt_template=yaml_conversion_prompt_template,
                    model_kwargs=conversion_model_kwargs,
                )
                if len(parsed_topics) != n_subtopics:
                    raise YamlConversionError(
                        f"Error: Length of subtopics {len(parsed_topics)} does not match desired n_subtopics {n_subtopics}: {parsed_topics}"
                    )
                topic_list.extend(parsed_topics)
            except YamlConversionError as e:
                if ignore_conversion_failure:
                    continue
                else:
                    raise e
        topic_list.extend(additional_subtopics)

        # Mix the macro topics with the subtopics
        if combine_topics:
            topic_list.extend(macro_topics)

        # Generate the openlines
        raw_lines = [
            self.generate_python_problem(
                topic=t,
                n_openlines=n_openlines,
                model=model,
                model_kwargs=base_model_kwargs,
                prompt_template=python_problem_prompt_template,
            )[0]
            for t in topic_list
        ]
        openlines = []
        for line in raw_lines:
            try:
                parsed_line = self.convert_response_to_yaml_list(
                    line,
                    model=model,
                    prompt_template=yaml_conversion_prompt_template,
                    model_kwargs=conversion_model_kwargs,
                )
                if len(parsed_line) != n_openlines:
                    raise YamlConversionError(
                        f"Error: Length of openlines {len(parsed_line)} does not match desired n_openlines {n_openlines}: {parsed_line}"
                    )
                openlines.extend(parsed_line)
            except YamlConversionError as e:
                if ignore_conversion_failure:
                    continue
                else:
                    raise e

        return openlines


class NemotronFormatter(ConversationFormatter):

    PROMPT_PREFIX = "<extra_id_0>System\n\n<extra_id_1>User\n"

    @staticmethod
    def format_conversation(conv: List[dict]) -> str:
        """
        Formats a converstation between a user and assistant in the Nemotron 340B format
        described here: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/nemotron-4-340b-instruct
        Args:
            conv: A conversation between a user and assistant
        Returns:
            A conversation formatted as text
        """
        prompt = NemotronFormatter.PROMPT_PREFIX

        for i, turn in enumerate(conv):
            user_turn = i % 2 == 0

            if user_turn:
                if turn["role"] != "user":
                    raise ValueError(
                        f"Conversation turn {i} is not 'user'. All even number turns should be."
                    )
                prompt += turn["content"] + "\n<extra_id_1>Assistant\n"
            else:
                if turn["role"] != "assistant":
                    raise ValueError(
                        f"Conversation turn {i} is not 'assistant'. All odd number turns should be."
                    )
                prompt += turn["content"] + "\n<extra_id_1>User\n"

        return prompt
