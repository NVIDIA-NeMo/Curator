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
import logging
import os
from typing import Any, Coroutine, List, Optional, Tuple, Union

import yaml
from tqdm.asyncio import tqdm

from nemo_curator.log import create_logger
from nemo_curator.services.model_client import AsyncLLMClient
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


class AsyncNemotronGenerator:
    """
    Provides a collection of methods for generating synthetic data
    described in the Nemotron-4 340B Technical Report
    (https://arxiv.org/abs/2406.11704v1) and inspired by the
    UltraChat paper (https://arxiv.org/abs/2305.14233)
    """

    def __init__(
        self,
        llm_client: AsyncLLMClient,
        logger: Union[logging.LoggerAdapter, str] = "./",
        max_concurrent_requests: Optional[int] = None,
    ) -> None:
        self.client = llm_client
        self.max_concurrent_requests = max_concurrent_requests
        if isinstance(logger, str):
            self.logger = create_logger(
                rank=0,
                log_file=os.path.join(logger, "nemotron-generator.log"),
                name="AsyncNemotronGenrator",
            )
        else:
            self.logger = logger

    async def _prompt(
        self, model: str, prompt_template: str, prompt_kwargs: dict, model_kwargs: dict
    ) -> List[str]:
        prompt = prompt_template.format(**prompt_kwargs)
        messages = [{"role": "user", "content": prompt}]

        return await self.client.query_model(
            messages=messages, model=model, **model_kwargs
        )

    async def convert_response_to_yaml_list(
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
        yaml_response = await self._prompt(
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

    async def _try_convert_yaml_list(
        self,
        response: str,
        model: str,
        yaml_conversion_prompt_template: str,
        conversion_model_kwargs: dict,
        expected_length: int,
        ignore_conversion_failure: bool,
    ):
        try:
            parsed_list = await self.convert_response_to_yaml_list(
                response,
                model=model,
                prompt_template=yaml_conversion_prompt_template,
                model_kwargs=conversion_model_kwargs,
            )
            if len(parsed_list) != expected_length:
                raise YamlConversionError(
                    f"Error: Length of parsed list {len(parsed_list)} does not match expected length {expected_length}: {parsed_list}"
                )
        except YamlConversionError as e:
            if ignore_conversion_failure:
                return []
            else:
                raise e

        return parsed_list

    async def _gather(
        self, requests: List[Coroutine[Any, Any, List[str]]]
    ) -> List[str]:
        max_requests = self.max_concurrent_requests
        if max_requests is None:
            max_requests = len(requests)

        final_list = []
        for i in tqdm(range(0, len(requests), max_requests)):
            request_slice = requests[i : i + max_requests]
            result = await tqdm.gather(*request_slice)
            final_list.extend(result)

        return final_list

    async def generate_macro_topics(
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
        macro_topics = await self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return macro_topics

    async def generate_subtopics(
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
        subtopics_response = await self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return subtopics_response

    async def generate_open_qa_from_topic(
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
        openline_response = await self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return openline_response

    async def revise_open_qa(
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
        revisions = await self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return revisions

    async def generate_writing_tasks(
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
        writing_tasks = await self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return writing_tasks

    async def revise_writing_tasks(
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
        revisions = await self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return revisions

    async def generate_closed_qa_instructions(
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
        openline_response = await self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return openline_response

    async def generate_math_macro_topics(
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
        macro_topics = await self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return macro_topics

    async def generate_math_subtopics(
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
        subtopics_response = await self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return subtopics_response

    async def classify_math_entity(
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
        classification_response = await self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return classification_response

    async def generate_math_problem(
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
        openline_response = await self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return openline_response

    async def generate_python_macro_topics(
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
        macro_topics = await self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return macro_topics

    async def generate_python_subtopics(
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
        subtopics_response = await self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return subtopics_response

    async def classify_python_entity(
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
        classification_response = await self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return classification_response

    async def generate_python_problem(
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
        openline_response = await self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return openline_response

    async def generate_dialogue(
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
        first_assistant_response = await self.client.query_model(
            messages=conversation_history,
            model=assistant_model,
            **assistant_model_kwargs,
        )
        first_assistant_response = first_assistant_response[0]
        conversation_history.append(
            {"role": "assistant", "content": first_assistant_response}
        )
        for _ in range(n_user_turns - 1):
            user_response = await self._impersonate_user(
                conversation_history=conversation_history,
                model=user_model,
                prompt_template=prompt_template,
                prompt_kwargs=prompt_kwargs,
                model_kwargs=user_model_kwargs,
            )
            conversation_history.append({"role": "user", "content": user_response})
            assistant_response = await self.client.query_model(
                messages=conversation_history,
                model=assistant_model,
                **assistant_model_kwargs,
            )
            assistant_response = assistant_response[0]
            conversation_history.append(
                {"role": "assistant", "content": assistant_response}
            )

        return conversation_history

    async def generate_two_turn_prompt(
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
        first_assistant_response = await self.client.query_model(
            messages=conversation_history,
            model=assistant_model,
            **assistant_model_kwargs,
        )
        first_assistant_response = first_assistant_response[0]
        conversation_history.append(
            {"role": "assistant", "content": first_assistant_response}
        )

        user_response = await self._impersonate_user(
            conversation_history=conversation_history,
            model=user_model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=user_model_kwargs,
        )
        conversation_history.append({"role": "user", "content": user_response})

        return conversation_history

    async def _impersonate_user(
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
        response = await self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return response[0]

    async def run_open_qa_pipeline(
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
        self.logger.info("Starting open q&a pipeline")
        # Generate the macro topics
        self.logger.info("Starting macro topic generation")
        responses = await self.generate_macro_topics(
            n_macro_topics=n_macro_topics,
            model=model,
            model_kwargs=base_model_kwargs,
            prompt_template=macro_topic_prompt_template,
        )
        macro_topics = await self.convert_response_to_yaml_list(
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
        self.logger.info("Finished macro topic generation")

        # Generate the subtopics
        raw_topics = [
            self._generate_parse_subtopic(
                macro_topic=macro_topic,
                n_subtopics=n_subtopics,
                model=model,
                subtopic_prompt_template=subtopic_prompt_template,
                yaml_conversion_prompt_template=yaml_conversion_prompt_template,
                base_model_kwargs=base_model_kwargs,
                conversion_model_kwargs=conversion_model_kwargs,
                ignore_conversion_failure=ignore_conversion_failure,
            )
            for macro_topic in macro_topics
        ]
        self.logger.info("Starting subtopic generation")
        raw_topics = await self._gather(raw_topics)
        topic_list = [item for subtopics in raw_topics for item in subtopics]
        topic_list.extend(additional_subtopics)
        self.logger.info("Finished subtopic generation")

        # Mix the macro topics with the subtopics
        if combine_topics:
            topic_list.extend(macro_topics)

        # Generate the openlines
        raw_lines = [
            self._generate_parse_openline(
                subtopic=subtopic,
                n_openlines=n_openlines,
                model=model,
                open_qa_from_topics_prompt_template=open_qa_from_topics_prompt_template,
                yaml_conversion_prompt_template=yaml_conversion_prompt_template,
                base_model_kwargs=base_model_kwargs,
                conversion_model_kwargs=conversion_model_kwargs,
                ignore_conversion_failure=ignore_conversion_failure,
            )
            for subtopic in topic_list
        ]
        self.logger.info("Starting openline generation")
        raw_lines = await self._gather(raw_lines)
        openlines = [item for lines in raw_lines for item in lines]
        self.logger.info("Finished openline generation")

        # Revise the openlines
        raw_revisions = [
            self._revise_parse_openline(
                openline=openline,
                n_revisions=n_revisions,
                model=model,
                revise_open_qa_prompt_template=revise_open_qa_prompt_template,
                yaml_conversion_prompt_template=yaml_conversion_prompt_template,
                base_model_kwargs=base_model_kwargs,
                conversion_model_kwargs=conversion_model_kwargs,
                ignore_conversion_failure=ignore_conversion_failure,
            )
            for openline in openlines
        ]
        self.logger.info("Starting openline revision")
        raw_revisions = await self._gather(raw_revisions)
        revised_openlines = [item for revisions in raw_revisions for item in revisions]
        self.logger.info("Finished openline revision")
        self.logger.info("Finished open q&a pipeline")

        return revised_openlines

    async def _generate_parse_subtopic(
        self,
        macro_topic: str,
        n_subtopics: Union[int, str],
        model: str,
        subtopic_prompt_template: str,
        yaml_conversion_prompt_template: str,
        base_model_kwargs: dict,
        conversion_model_kwargs: dict,
        ignore_conversion_failure: bool,
    ) -> List[str]:
        subtopic = await self.generate_subtopics(
            macro_topic=macro_topic,
            n_subtopics=n_subtopics,
            model=model,
            model_kwargs=base_model_kwargs,
            prompt_template=subtopic_prompt_template,
        )
        subtopic = subtopic[0]
        return await self._try_convert_yaml_list(
            subtopic,
            model=model,
            yaml_conversion_prompt_template=yaml_conversion_prompt_template,
            conversion_model_kwargs=conversion_model_kwargs,
            expected_length=n_subtopics,
            ignore_conversion_failure=ignore_conversion_failure,
        )

    async def _generate_parse_openline(
        self,
        subtopic: str,
        n_openlines: Union[int, str],
        model: str,
        open_qa_from_topics_prompt_template: str,
        yaml_conversion_prompt_template: str,
        base_model_kwargs: dict,
        conversion_model_kwargs: dict,
        ignore_conversion_failure: bool,
    ) -> List[str]:
        openline = await self.generate_open_qa_from_topic(
            topic=subtopic,
            n_openlines=n_openlines,
            model=model,
            model_kwargs=base_model_kwargs,
            prompt_template=open_qa_from_topics_prompt_template,
        )
        openline = openline[0]
        return await self._try_convert_yaml_list(
            openline,
            model=model,
            yaml_conversion_prompt_template=yaml_conversion_prompt_template,
            conversion_model_kwargs=conversion_model_kwargs,
            expected_length=n_openlines,
            ignore_conversion_failure=ignore_conversion_failure,
        )

    async def _revise_parse_openline(
        self,
        openline: str,
        n_revisions: Union[int, str],
        model: str,
        revise_open_qa_prompt_template: str,
        yaml_conversion_prompt_template: str,
        base_model_kwargs: dict,
        conversion_model_kwargs: dict,
        ignore_conversion_failure: bool,
    ) -> List[str]:
        revised_openline = await self.revise_open_qa(
            openline=openline,
            n_revisions=n_revisions,
            model=model,
            model_kwargs=base_model_kwargs,
            prompt_template=revise_open_qa_prompt_template,
        )
        revised_openline = revised_openline[0]
        return await self._try_convert_yaml_list(
            revised_openline,
            model=model,
            yaml_conversion_prompt_template=yaml_conversion_prompt_template,
            conversion_model_kwargs=conversion_model_kwargs,
            expected_length=n_revisions,
            ignore_conversion_failure=ignore_conversion_failure,
        )

    async def run_writing_pipeline(
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
        self.logger.info("Starting writing pipeline")
        # Generate the tasks
        raw_writing_tasks = []
        for topic in topics:
            for material in text_material_types:
                raw_writing_tasks.append(
                    self._generate_parse_writing_task(
                        topic=topic,
                        material=material,
                        n_openlines=n_openlines,
                        model=model,
                        writing_task_prompt_template=writing_task_prompt_template,
                        yaml_conversion_prompt_template=yaml_conversion_prompt_template,
                        base_model_kwargs=base_model_kwargs,
                        conversion_model_kwargs=conversion_model_kwargs,
                        ignore_conversion_failure=ignore_conversion_failure,
                    )
                )
        self.logger.info("Starting writing task generation")
        raw_writing_tasks = await self._gather(raw_writing_tasks)
        writing_tasks = [item for tasks in raw_writing_tasks for item in tasks]
        self.logger.info("Finished writing task generation")

        # Revise the tasks
        raw_revised_openlines = [
            self._revise_parse_writing_task(
                task=task,
                n_revisions=n_revisions,
                model=model,
                revise_writing_task_prompt_template=revise_writing_task_prompt_template,
                yaml_conversion_prompt_template=yaml_conversion_prompt_template,
                base_model_kwargs=base_model_kwargs,
                conversion_model_kwargs=conversion_model_kwargs,
                ignore_conversion_failure=ignore_conversion_failure,
            )
            for task in writing_tasks
        ]
        self.logger.info("Starting writing task revision")
        raw_revised_openlines = await self._gather(raw_revised_openlines)
        revised_openlines = [item for lines in raw_revised_openlines for item in lines]
        self.logger.info("Finished writing task revision")
        self.logger.info("Finished writing pipeline")

        return revised_openlines

    async def _generate_parse_writing_task(
        self,
        topic: str,
        material: str,
        n_openlines: Union[int, str],
        model: str,
        writing_task_prompt_template: str,
        yaml_conversion_prompt_template: str,
        base_model_kwargs: dict,
        conversion_model_kwargs: dict,
        ignore_conversion_failure: bool,
    ) -> List[str]:
        raw_tasks = await self.generate_writing_tasks(
            topic=topic,
            text_material_type=material,
            n_openlines=n_openlines,
            model=model,
            model_kwargs=base_model_kwargs,
            prompt_template=writing_task_prompt_template,
        )
        raw_tasks = raw_tasks[0]
        return await self._try_convert_yaml_list(
            raw_tasks,
            model=model,
            yaml_conversion_prompt_template=yaml_conversion_prompt_template,
            conversion_model_kwargs=conversion_model_kwargs,
            expected_length=n_openlines,
            ignore_conversion_failure=ignore_conversion_failure,
        )

    async def _revise_parse_writing_task(
        self,
        task: str,
        n_revisions: Union[int, str],
        model: str,
        revise_writing_task_prompt_template: str,
        yaml_conversion_prompt_template: str,
        base_model_kwargs: dict,
        conversion_model_kwargs: dict,
        ignore_conversion_failure: bool,
    ) -> List[str]:
        raw_revision = await self.revise_writing_tasks(
            openline=task,
            n_revisions=n_revisions,
            model=model,
            model_kwargs=base_model_kwargs,
            prompt_template=revise_writing_task_prompt_template,
        )
        raw_revision = raw_revision[0]
        return await self._try_convert_yaml_list(
            raw_revision,
            model=model,
            yaml_conversion_prompt_template=yaml_conversion_prompt_template,
            conversion_model_kwargs=conversion_model_kwargs,
            expected_length=n_revisions,
            ignore_conversion_failure=ignore_conversion_failure,
        )

    async def run_closed_qa_pipeline(
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
        self.logger.info("Starting closed q&a pipeline")
        raw_qa = [
            self._generate_parse_closed_qa(
                document_id=i,
                document=document,
                n_openlines=n_openlines,
                model=model,
                closed_qa_prompt_template=closed_qa_prompt_template,
                yaml_conversion_prompt_template=yaml_conversion_prompt_template,
                base_model_kwargs=base_model_kwargs,
                conversion_model_kwargs=conversion_model_kwargs,
                ignore_conversion_failure=ignore_conversion_failure,
            )
            for i, document in enumerate(documents)
        ]
        raw_qa = await self._gather(raw_qa)
        document_openline_pairs = [item for lines in raw_qa for item in lines]
        self.logger.info("Finished closed q&a pipeline")

        return document_openline_pairs

    async def _generate_parse_closed_qa(
        self,
        document_id: int,
        document: str,
        n_openlines: Union[int, str],
        model: str,
        closed_qa_prompt_template: str,
        yaml_conversion_prompt_template: str,
        base_model_kwargs: dict,
        conversion_model_kwargs: dict,
        ignore_conversion_failure: bool,
    ) -> List[str]:
        raw_instruction = await self.generate_closed_qa_instructions(
            document=document,
            n_openlines=n_openlines,
            model=model,
            model_kwargs=base_model_kwargs,
            prompt_template=closed_qa_prompt_template,
        )
        raw_instruction = raw_instruction[0]
        parsed_instructions = await self._try_convert_yaml_list(
            raw_instruction,
            model=model,
            yaml_conversion_prompt_template=yaml_conversion_prompt_template,
            conversion_model_kwargs=conversion_model_kwargs,
            expected_length=n_openlines,
            ignore_conversion_failure=ignore_conversion_failure,
        )

        return [(document_id, inst) for inst in parsed_instructions]

    async def run_math_pipeline(
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
        self.logger.info("Starting math pipeline")
        # Generate the macro topics
        self.logger.info("Starting math macro topic generation")
        responses = await self.generate_math_macro_topics(
            n_macro_topics=n_macro_topics,
            school_level=school_level,
            model=model,
            model_kwargs=base_model_kwargs,
            prompt_template=macro_topic_prompt_template,
        )
        macro_topics = await self.convert_response_to_yaml_list(
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
        self.logger.info("Finished math macro topic generation")

        # Generate the subtopics
        raw_topics = [
            self._generate_parse_math_subtopic(
                macro_topic=macro_topic,
                n_subtopics=n_subtopics,
                model=model,
                subtopic_prompt_template=subtopic_prompt_template,
                yaml_conversion_prompt_template=yaml_conversion_prompt_template,
                base_model_kwargs=base_model_kwargs,
                conversion_model_kwargs=conversion_model_kwargs,
                ignore_conversion_failure=ignore_conversion_failure,
            )
            for macro_topic in macro_topics
        ]
        self.logger.info("Starting math subtopic generation")
        raw_topics = await self._gather(raw_topics)
        topic_list = [item for subtopics in raw_topics for item in subtopics]
        topic_list.extend(additional_subtopics)
        self.logger.info("Finished math subtopic generation")

        # Mix the macro topics with the subtopics
        if combine_topics:
            topic_list.extend(macro_topics)

        # Generate the openlines
        raw_lines = [
            self._generate_parse_math_openline(
                subtopic=subtopic,
                n_openlines=n_openlines,
                model=model,
                math_problem_prompt_template=math_problem_prompt_template,
                yaml_conversion_prompt_template=yaml_conversion_prompt_template,
                base_model_kwargs=base_model_kwargs,
                conversion_model_kwargs=conversion_model_kwargs,
                ignore_conversion_failure=ignore_conversion_failure,
            )
            for subtopic in topic_list
        ]
        self.logger.info("Starting math openline generation")
        raw_lines = await self._gather(raw_lines)
        openlines = [item for lines in raw_lines for item in lines]
        self.logger.info("Finished math openline generation")
        self.logger.info("Finished math pipeline")

        return openlines

    async def _generate_parse_math_subtopic(
        self,
        macro_topic: str,
        n_subtopics: Union[int, str],
        model: str,
        subtopic_prompt_template: str,
        yaml_conversion_prompt_template: str,
        base_model_kwargs: dict,
        conversion_model_kwargs: dict,
        ignore_conversion_failure: bool,
    ) -> List[str]:
        raw_topic = await self.generate_math_subtopics(
            macro_topic=macro_topic,
            n_subtopics=n_subtopics,
            model=model,
            model_kwargs=base_model_kwargs,
            prompt_template=subtopic_prompt_template,
        )
        raw_topic = raw_topic[0]
        return await self._try_convert_yaml_list(
            raw_topic,
            model=model,
            yaml_conversion_prompt_template=yaml_conversion_prompt_template,
            conversion_model_kwargs=conversion_model_kwargs,
            expected_length=n_subtopics,
            ignore_conversion_failure=ignore_conversion_failure,
        )

    async def _generate_parse_math_openline(
        self,
        subtopic: str,
        n_openlines: Union[int, str],
        model: str,
        math_problem_prompt_template: str,
        yaml_conversion_prompt_template: str,
        base_model_kwargs: dict,
        conversion_model_kwargs: dict,
        ignore_conversion_failure: bool,
    ) -> List[str]:
        raw_line = await self.generate_math_problem(
            topic=subtopic,
            n_openlines=n_openlines,
            model=model,
            model_kwargs=base_model_kwargs,
            prompt_template=math_problem_prompt_template,
        )
        raw_line = raw_line[0]
        return await self._try_convert_yaml_list(
            raw_line,
            model=model,
            yaml_conversion_prompt_template=yaml_conversion_prompt_template,
            conversion_model_kwargs=conversion_model_kwargs,
            expected_length=n_openlines,
            ignore_conversion_failure=ignore_conversion_failure,
        )

    async def run_python_pipeline(
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
        self.logger.info("Starting python pipeline")
        # Generate the macro topics
        self.logger.info("Starting python macro topic generation")
        responses = await self.generate_python_macro_topics(
            n_macro_topics=n_macro_topics,
            model=model,
            model_kwargs=base_model_kwargs,
            prompt_template=macro_topic_prompt_template,
        )
        macro_topics = await self.convert_response_to_yaml_list(
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
        self.logger.info("Finished python macro topic generation")

        # Generate the subtopics
        raw_topics = [
            self._generate_parse_python_subtopic(
                macro_topic=macro_topic,
                n_subtopics=n_subtopics,
                model=model,
                subtopic_prompt_template=subtopic_prompt_template,
                yaml_conversion_prompt_template=yaml_conversion_prompt_template,
                base_model_kwargs=base_model_kwargs,
                conversion_model_kwargs=conversion_model_kwargs,
                ignore_conversion_failure=ignore_conversion_failure,
            )
            for macro_topic in macro_topics
        ]
        self.logger.info("Starting python subtopic generation")
        raw_topics = await self._gather(raw_topics)
        topic_list = [item for subtopics in raw_topics for item in subtopics]
        topic_list.extend(additional_subtopics)
        self.logger.info("Finished python subtopic generation")

        # Mix the macro topics with the subtopics
        if combine_topics:
            topic_list.extend(macro_topics)

        # Generate the openlines
        raw_lines = [
            self._generate_parse_python_openline(
                subtopic=subtopic,
                n_openlines=n_openlines,
                model=model,
                python_problem_prompt_template=python_problem_prompt_template,
                yaml_conversion_prompt_template=yaml_conversion_prompt_template,
                base_model_kwargs=base_model_kwargs,
                conversion_model_kwargs=conversion_model_kwargs,
                ignore_conversion_failure=ignore_conversion_failure,
            )
            for subtopic in topic_list
        ]
        self.logger.info("Starting python openline generation")
        raw_lines = await self._gather(raw_lines)
        openlines = [item for lines in raw_lines for item in lines]
        self.logger.info("Finished python openline generation")
        self.logger.info("Finished python pipeline")

        return openlines

    async def _generate_parse_python_subtopic(
        self,
        macro_topic: str,
        n_subtopics: Union[int, str],
        model: str,
        subtopic_prompt_template: str,
        yaml_conversion_prompt_template: str,
        base_model_kwargs: dict,
        conversion_model_kwargs: dict,
        ignore_conversion_failure: bool,
    ) -> List[str]:
        raw_topic = await self.generate_python_subtopics(
            macro_topic=macro_topic,
            n_subtopics=n_subtopics,
            model=model,
            model_kwargs=base_model_kwargs,
            prompt_template=subtopic_prompt_template,
        )
        raw_topic = raw_topic[0]
        return await self._try_convert_yaml_list(
            raw_topic,
            model=model,
            yaml_conversion_prompt_template=yaml_conversion_prompt_template,
            conversion_model_kwargs=conversion_model_kwargs,
            expected_length=n_subtopics,
            ignore_conversion_failure=ignore_conversion_failure,
        )

    async def _generate_parse_python_openline(
        self,
        subtopic: str,
        n_openlines: Union[int, str],
        model: str,
        python_problem_prompt_template: str,
        yaml_conversion_prompt_template: str,
        base_model_kwargs: dict,
        conversion_model_kwargs: dict,
        ignore_conversion_failure: bool,
    ) -> List[str]:
        raw_line = await self.generate_python_problem(
            topic=subtopic,
            n_openlines=n_openlines,
            model=model,
            model_kwargs=base_model_kwargs,
            prompt_template=python_problem_prompt_template,
        )
        raw_line = raw_line[0]
        return await self._try_convert_yaml_list(
            raw_line,
            model=model,
            yaml_conversion_prompt_template=yaml_conversion_prompt_template,
            conversion_model_kwargs=conversion_model_kwargs,
            expected_length=n_openlines,
            ignore_conversion_failure=ignore_conversion_failure,
        )
