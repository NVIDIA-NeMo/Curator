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

import asyncio
from itertools import cycle
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
import yaml
from tqdm.asyncio import tqdm

from nemo_curator.synthetic.async_nemotron import AsyncNemotronGenerator
from nemo_curator.synthetic.error import YamlConversionError


class TestAsyncNemotronGenerator:
    @pytest.fixture
    def mock_llm_client(self):
        mock_client = AsyncMock()
        mock_client.query_model.return_value = ["This is a mock response"]
        return mock_client

    @pytest.mark.asyncio
    async def test_init(self, mock_llm_client):
        """Test the constructor of AsyncNemotronGenerator."""
        generator = AsyncNemotronGenerator(mock_llm_client)
        assert generator.client == mock_llm_client

    @pytest.mark.asyncio
    async def test_prompt(self, mock_llm_client):
        """Test the internal _prompt method."""
        generator = AsyncNemotronGenerator(mock_llm_client)

        prompt_template = "Test prompt for {query}."
        prompt_kwargs = {"query": "test"}
        model_kwargs = {"temperature": 0.7}

        result = await generator._prompt(
            model="test_model",
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        # Check if query_model was called with the right parameters
        mock_llm_client.query_model.assert_called_once()
        call_args = mock_llm_client.query_model.call_args[1]
        assert call_args["model"] == "test_model"
        assert call_args["temperature"] == 0.7
        assert len(call_args["messages"]) == 1
        assert call_args["messages"][0]["content"] == "Test prompt for test."

        # Check return value
        assert result == ["This is a mock response"]

    @pytest.mark.asyncio
    async def test_convert_response_to_yaml_list(self, mock_llm_client):
        """Test the convert_response_to_yaml_list method with a valid response."""
        generator = AsyncNemotronGenerator(mock_llm_client)

        # Mock that the LLM returns a valid YAML string
        yaml_list = ["Item 1", "Item 2", "Item 3"]
        yaml_string = yaml.dump(yaml_list)
        mock_llm_client.query_model.return_value = [yaml_string]

        llm_response = "Some text containing Item 1, Item 2, and Item 3"
        result = await generator.convert_response_to_yaml_list(
            llm_response=llm_response,
            model="test_model",
        )

        assert result == yaml_list

    @pytest.mark.asyncio
    async def test_convert_response_to_yaml_list_invalid_yaml(self, mock_llm_client):
        """Test handling of invalid YAML in convert_response_to_yaml_list."""
        generator = AsyncNemotronGenerator(mock_llm_client)

        # Return invalid YAML
        mock_llm_client.query_model.return_value = ["[This is not valid YAML"]

        with pytest.raises(YamlConversionError):
            await generator.convert_response_to_yaml_list(
                llm_response="Some text",
                model="test_model",
            )

    @pytest.mark.asyncio
    async def test_convert_response_with_non_string_elements(self, mock_llm_client):
        """Test handling of YAML list with non-string elements."""
        generator = AsyncNemotronGenerator(mock_llm_client)

        # Test with list containing non-string elements
        mock_llm_client.query_model.return_value = [
            yaml.dump(["Valid string", 123, True, {"nested": "dict"}])
        ]

        with pytest.raises(YamlConversionError) as excinfo:
            await generator.convert_response_to_yaml_list(
                llm_response="Valid string",
                model="test_model",
            )
        assert "non-string element" in str(excinfo.value).lower()

    @pytest.mark.asyncio
    async def test_try_convert_yaml_list(self, mock_llm_client):
        """Test _try_convert_yaml_list method for handling YAML conversion errors."""
        generator = AsyncNemotronGenerator(mock_llm_client)

        # Test successful conversion
        mock_llm_client.query_model.return_value = [yaml.dump(["Item 1", "Item 2"])]
        result = await generator._try_convert_yaml_list(
            response="Original response with Item 1 and Item 2",
            model="test_model",
            yaml_conversion_prompt_template="Template: {llm_response}",
            conversion_model_kwargs={},
            expected_length=2,
            ignore_conversion_failure=False,
        )
        assert result == ["Item 1", "Item 2"]

        # Test with invalid YAML - should raise exception when ignore_conversion_failure=False
        mock_llm_client.query_model.return_value = ["[This is not valid YAML"]
        with pytest.raises(YamlConversionError):
            await generator._try_convert_yaml_list(
                response="Original response",
                model="test_model",
                yaml_conversion_prompt_template="Template: {llm_response}",
                conversion_model_kwargs={},
                expected_length=2,
                ignore_conversion_failure=False,
            )

        # Test with invalid YAML - should return empty list when ignore_conversion_failure=True
        result = await generator._try_convert_yaml_list(
            response="Original response",
            model="test_model",
            yaml_conversion_prompt_template="Template: {llm_response}",
            conversion_model_kwargs={},
            expected_length=2,
            ignore_conversion_failure=True,
        )
        assert result == []

        # Test with YAML that's not a list - should raise exception when ignore_conversion_failure=False
        mock_llm_client.query_model.return_value = [yaml.dump({"key": "value"})]
        with pytest.raises(YamlConversionError):
            await generator._try_convert_yaml_list(
                response="Original response",
                model="test_model",
                yaml_conversion_prompt_template="Template: {llm_response}",
                conversion_model_kwargs={},
                expected_length=2,
                ignore_conversion_failure=False,
            )

        # Test with YAML list of wrong length - should raise exception when ignore_conversion_failure=False
        mock_llm_client.query_model.return_value = [
            yaml.dump(["Item 1"])
        ]  # Only 1 item
        with pytest.raises(YamlConversionError):
            await generator._try_convert_yaml_list(
                response="Original response",
                model="test_model",
                yaml_conversion_prompt_template="Template: {llm_response}",
                conversion_model_kwargs={},
                expected_length=2,  # Expected 2 items
                ignore_conversion_failure=False,
            )

        # Test with hallucination - should raise exception when ignore_conversion_failure=False
        mock_llm_client.query_model.return_value = [
            yaml.dump(["Item in response", "Hallucinated item"])
        ]
        with pytest.raises(YamlConversionError):
            await generator._try_convert_yaml_list(
                response="Original response with Item in response",
                model="test_model",
                yaml_conversion_prompt_template="Template: {llm_response}",
                conversion_model_kwargs={},
                expected_length=2,
                ignore_conversion_failure=False,
            )

    @pytest.mark.asyncio
    async def test_generate_macro_topics(self, mock_llm_client):
        """Test generate_macro_topics method."""
        generator = AsyncNemotronGenerator(mock_llm_client)

        # Set up the return value for query_model
        mock_llm_client.query_model.return_value = ["Topic 1\nTopic 2\nTopic 3"]

        result = await generator.generate_macro_topics(
            n_macro_topics=3,
            model="test_model",
        )

        # Check the parameters sent to _prompt
        mock_llm_client.query_model.assert_called_once()
        messages = mock_llm_client.query_model.call_args[1]["messages"]
        assert (
            messages[0]["content"].find("3") != -1
        ), "Number of topics wasn't passed to the prompt"

        # Check the result
        assert result == ["Topic 1\nTopic 2\nTopic 3"]

    @pytest.mark.asyncio
    async def test_generate_subtopics(self, mock_llm_client):
        """Test generate_subtopics method."""
        generator = AsyncNemotronGenerator(mock_llm_client)

        result = await generator.generate_subtopics(
            macro_topic="Science",
            n_subtopics=5,
            model="test_model",
        )

        # Check the right parameters were passed to query_model
        mock_llm_client.query_model.assert_called_once()
        messages = mock_llm_client.query_model.call_args[1]["messages"]
        assert (
            "Science" in messages[0]["content"]
        ), "Macro topic wasn't passed to the prompt"
        assert (
            "5" in messages[0]["content"]
        ), "Number of subtopics wasn't passed to the prompt"

        # Check the result
        assert result == ["This is a mock response"]

    @pytest.mark.asyncio
    async def test_generate_open_qa_from_topic(self, mock_llm_client):
        """Test generate_open_qa_from_topic method."""
        generator = AsyncNemotronGenerator(mock_llm_client)

        result = await generator.generate_open_qa_from_topic(
            topic="Artificial Intelligence",
            n_openlines=3,
            model="test_model",
        )

        # Check the right parameters were passed to query_model
        mock_llm_client.query_model.assert_called_once()
        messages = mock_llm_client.query_model.call_args[1]["messages"]
        assert (
            "Artificial Intelligence" in messages[0]["content"]
        ), "Topic wasn't passed to the prompt"
        assert (
            "3" in messages[0]["content"]
        ), "Number of openlines wasn't passed to the prompt"

        # Check the result
        assert result == ["This is a mock response"]

    @pytest.mark.asyncio
    async def test_revise_open_qa(self, mock_llm_client):
        """Test revise_open_qa method."""
        generator = AsyncNemotronGenerator(mock_llm_client)

        result = await generator.revise_open_qa(
            openline="What is machine learning?",
            n_revisions=2,
            model="test_model",
        )

        # Check the right parameters were passed to query_model
        mock_llm_client.query_model.assert_called_once()
        messages = mock_llm_client.query_model.call_args[1]["messages"]
        assert (
            "What is machine learning?" in messages[0]["content"]
        ), "Openline wasn't passed to the prompt"
        assert (
            "2" in messages[0]["content"]
        ), "Number of revisions wasn't passed to the prompt"

        # Check the result
        assert result == ["This is a mock response"]

    @pytest.mark.asyncio
    async def test_generate_writing_tasks(self, mock_llm_client):
        """Test generate_writing_tasks method."""
        generator = AsyncNemotronGenerator(mock_llm_client)

        result = await generator.generate_writing_tasks(
            topic="Environment",
            text_material_type="Essay",
            n_openlines=4,
            model="test_model",
        )

        # Check the right parameters were passed to query_model
        mock_llm_client.query_model.assert_called_once()
        messages = mock_llm_client.query_model.call_args[1]["messages"]
        assert (
            "Environment" in messages[0]["content"]
        ), "Topic wasn't passed to the prompt"
        assert (
            "Essay" in messages[0]["content"]
        ), "Text material type wasn't passed to the prompt"
        assert (
            "4" in messages[0]["content"]
        ), "Number of openlines wasn't passed to the prompt"

        # Check the result
        assert result == ["This is a mock response"]

    @pytest.mark.asyncio
    async def test_revise_writing_tasks(self, mock_llm_client):
        """Test revise_writing_tasks method."""
        generator = AsyncNemotronGenerator(mock_llm_client)

        result = await generator.revise_writing_tasks(
            openline="Write an essay about climate change.",
            n_revisions=2,
            model="test_model",
        )

        # Check the right parameters were passed to query_model
        mock_llm_client.query_model.assert_called_once()
        messages = mock_llm_client.query_model.call_args[1]["messages"]
        assert (
            "Write an essay about climate change." in messages[0]["content"]
        ), "Openline wasn't passed to the prompt"
        assert (
            "2" in messages[0]["content"]
        ), "Number of revisions wasn't passed to the prompt"

        # Check the result
        assert result == ["This is a mock response"]

    @pytest.mark.asyncio
    async def test_generate_closed_qa_instructions(self, mock_llm_client):
        """Test generate_closed_qa_instructions method."""
        generator = AsyncNemotronGenerator(mock_llm_client)

        document = "This is a sample document about AI technology..."
        result = await generator.generate_closed_qa_instructions(
            document=document,
            n_openlines=2,
            model="test_model",
        )

        # Check the right parameters were passed to query_model
        mock_llm_client.query_model.assert_called_once()
        messages = mock_llm_client.query_model.call_args[1]["messages"]
        assert (
            document in messages[0]["content"]
        ), "Document wasn't passed to the prompt"
        assert (
            "2" in messages[0]["content"]
        ), "Number of openlines wasn't passed to the prompt"

        # Check the result
        assert result == ["This is a mock response"]

    @pytest.mark.asyncio
    async def test_gather(self, mock_llm_client):
        """Test the _gather method for concurrent requests."""
        generator = AsyncNemotronGenerator(mock_llm_client, max_concurrent_requests=2)

        # Create mock async functions that return different values
        async def mock_coro1():
            return ["Result 1"]

        async def mock_coro2():
            return ["Result 2"]

        async def mock_coro3():
            return ["Result 3"]

        # Test gathering results
        requests = [mock_coro1(), mock_coro2(), mock_coro3()]
        results = await generator._gather(requests)

        # Check that we get all results in the right order
        assert results == [["Result 1"], ["Result 2"], ["Result 3"]]

    @pytest.mark.asyncio
    async def test_generate_math_macro_topics(self, mock_llm_client):
        """Test generate_math_macro_topics method."""
        generator = AsyncNemotronGenerator(mock_llm_client)

        result = await generator.generate_math_macro_topics(
            n_macro_topics=3,
            school_level="High School",
            model="test_model",
        )

        # Check the right parameters were passed to query_model
        mock_llm_client.query_model.assert_called_once()
        messages = mock_llm_client.query_model.call_args[1]["messages"]
        assert (
            "3" in messages[0]["content"]
        ), "Number of topics wasn't passed to the prompt"
        assert (
            "High School" in messages[0]["content"]
        ), "School level wasn't passed to the prompt"

        # Check the result
        assert result == ["This is a mock response"]

    @pytest.mark.asyncio
    async def test_generate_math_subtopics(self, mock_llm_client):
        """Test generate_math_subtopics method."""
        generator = AsyncNemotronGenerator(mock_llm_client)

        result = await generator.generate_math_subtopics(
            macro_topic="Calculus",
            n_subtopics=4,
            model="test_model",
        )

        # Check the right parameters were passed to query_model
        mock_llm_client.query_model.assert_called_once()
        messages = mock_llm_client.query_model.call_args[1]["messages"]
        assert (
            "Calculus" in messages[0]["content"]
        ), "Macro topic wasn't passed to the prompt"
        assert (
            "4" in messages[0]["content"]
        ), "Number of subtopics wasn't passed to the prompt"

        # Check the result
        assert result == ["This is a mock response"]

    @pytest.mark.asyncio
    async def test_classify_math_entity(self, mock_llm_client):
        """Test classify_math_entity method."""
        generator = AsyncNemotronGenerator(mock_llm_client)

        result = await generator.classify_math_entity(
            entity="Linear Algebra",
            model="test_model",
        )

        # Check the right parameters were passed to query_model
        mock_llm_client.query_model.assert_called_once()
        messages = mock_llm_client.query_model.call_args[1]["messages"]
        assert (
            "Linear Algebra" in messages[0]["content"]
        ), "Entity wasn't passed to the prompt"

        # Check the result
        assert result == ["This is a mock response"]

    @pytest.mark.asyncio
    async def test_generate_math_problem(self, mock_llm_client):
        """Test generate_math_problem method."""
        generator = AsyncNemotronGenerator(mock_llm_client)

        result = await generator.generate_math_problem(
            topic="Calculus",
            n_openlines=3,
            model="test_model",
        )

        # Check the right parameters were passed to query_model
        mock_llm_client.query_model.assert_called_once()
        messages = mock_llm_client.query_model.call_args[1]["messages"]
        assert "Calculus" in messages[0]["content"], "Topic wasn't passed to the prompt"
        assert (
            "3" in messages[0]["content"]
        ), "Number of openlines wasn't passed to the prompt"

        # Check the result
        assert result == ["This is a mock response"]

    @pytest.mark.asyncio
    async def test_generate_python_macro_topics(self, mock_llm_client):
        """Test generate_python_macro_topics method."""
        generator = AsyncNemotronGenerator(mock_llm_client)

        result = await generator.generate_python_macro_topics(
            n_macro_topics=3,
            model="test_model",
        )

        # Check the right parameters were passed to query_model
        mock_llm_client.query_model.assert_called_once()
        messages = mock_llm_client.query_model.call_args[1]["messages"]
        assert (
            "3" in messages[0]["content"]
        ), "Number of topics wasn't passed to the prompt"

        # Check the result
        assert result == ["This is a mock response"]

    @pytest.mark.asyncio
    async def test_generate_python_subtopics(self, mock_llm_client):
        """Test generate_python_subtopics method."""
        generator = AsyncNemotronGenerator(mock_llm_client)

        result = await generator.generate_python_subtopics(
            macro_topic="Data Structures",
            n_subtopics=4,
            model="test_model",
        )

        # Check the right parameters were passed to query_model
        mock_llm_client.query_model.assert_called_once()
        messages = mock_llm_client.query_model.call_args[1]["messages"]
        assert (
            "Data Structures" in messages[0]["content"]
        ), "Macro topic wasn't passed to the prompt"
        assert (
            "4" in messages[0]["content"]
        ), "Number of subtopics wasn't passed to the prompt"

        # Check the result
        assert result == ["This is a mock response"]

    @pytest.mark.asyncio
    async def test_classify_python_entity(self, mock_llm_client):
        """Test classify_python_entity method."""
        generator = AsyncNemotronGenerator(mock_llm_client)

        result = await generator.classify_python_entity(
            entity="List Comprehension",
            model="test_model",
        )

        # Check the right parameters were passed to query_model
        mock_llm_client.query_model.assert_called_once()
        messages = mock_llm_client.query_model.call_args[1]["messages"]
        assert (
            "List Comprehension" in messages[0]["content"]
        ), "Entity wasn't passed to the prompt"

        # Check the result
        assert result == ["This is a mock response"]

    @pytest.mark.asyncio
    async def test_generate_python_problem(self, mock_llm_client):
        """Test generate_python_problem method."""
        generator = AsyncNemotronGenerator(mock_llm_client)

        result = await generator.generate_python_problem(
            topic="Lists and Dictionaries",
            n_openlines=2,
            model="test_model",
            language="Python",
        )

        # Check the right parameters were passed to query_model
        mock_llm_client.query_model.assert_called_once()
        messages = mock_llm_client.query_model.call_args[1]["messages"]
        assert (
            "Lists and Dictionaries" in messages[0]["content"]
        ), "Topic wasn't passed to the prompt"
        assert (
            "2" in messages[0]["content"]
        ), "Number of openlines wasn't passed to the prompt"
        assert (
            "Python" in messages[0]["content"]
        ), "Language wasn't passed to the prompt"

        # Check the result
        assert result == ["This is a mock response"]

    @pytest.mark.asyncio
    async def test_impersonate_user(self, mock_llm_client):
        """Test _impersonate_user private method."""
        generator = AsyncNemotronGenerator(mock_llm_client)

        conversation_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        result = await generator._impersonate_user(
            conversation_history=conversation_history,
            model="test_model",
            prompt_template="History: {conversation_history}",
            prompt_kwargs={},
            model_kwargs={},
        )

        # Check the right parameters were passed to query_model
        mock_llm_client.query_model.assert_called_once()
        messages = mock_llm_client.query_model.call_args[1]["messages"]
        assert "History:" in messages[0]["content"]
        assert "Hello" in messages[0]["content"]
        assert "Hi there!" in messages[0]["content"]

        # Check the result
        assert result == "This is a mock response"

    @pytest.mark.asyncio
    async def test_generate_dialogue(self, mock_llm_client):
        """Test generate_dialogue method that creates conversations."""
        generator = AsyncNemotronGenerator(mock_llm_client)

        # Patch the _impersonate_user method
        with patch.object(
            AsyncNemotronGenerator, "_impersonate_user", new_callable=AsyncMock
        ) as mock_impersonate:
            # Set up return values
            mock_llm_client.query_model.side_effect = [
                ["Assistant response 1"],
                ["Assistant response 2"],
            ]
            mock_impersonate.return_value = "User follow-up"

            # Call the method
            result = await generator.generate_dialogue(
                openline="Hello",
                user_model="user_model",
                assistant_model="assistant_model",
                n_user_turns=2,
            )

            # Check the result structure
            expected_conversation = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Assistant response 1"},
                {"role": "user", "content": "User follow-up"},
                {"role": "assistant", "content": "Assistant response 2"},
            ]

            assert result == expected_conversation

            # Check that query_model was called correctly for assistant responses
            assert mock_llm_client.query_model.call_count == 2

            # Check that _impersonate_user was called correctly
            mock_impersonate.assert_called_once()

            # Extract the first two items from conversation_history for asserting
            # since _impersonate_user should be called after the first assistant response
            history_arg = mock_impersonate.call_args[1]["conversation_history"]
            assert len(history_arg) >= 2
            assert history_arg[0]["role"] == "user"
            assert history_arg[0]["content"] == "Hello"
            assert history_arg[1]["role"] == "assistant"
            assert history_arg[1]["content"] == "Assistant response 1"

    @pytest.mark.asyncio
    async def test_generate_two_turn_prompt(self, mock_llm_client):
        """Test generate_two_turn_prompt method."""
        generator = AsyncNemotronGenerator(mock_llm_client)

        # Patch the _impersonate_user method
        with patch.object(
            AsyncNemotronGenerator, "_impersonate_user", new_callable=AsyncMock
        ) as mock_impersonate:
            # Set up return values
            mock_llm_client.query_model.return_value = ["Assistant response"]
            mock_impersonate.return_value = "User follow-up"

            # Call the method
            result = await generator.generate_two_turn_prompt(
                openline="Hello",
                user_model="user_model",
                assistant_model="assistant_model",
            )

            # Check the result structure
            expected_conversation = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Assistant response"},
                {"role": "user", "content": "User follow-up"},
            ]

            assert result == expected_conversation

            # Check that query_model was called correctly
            mock_llm_client.query_model.assert_called_once()

            # Check that _impersonate_user was called correctly
            mock_impersonate.assert_called_once()

            # Check only first two items of conversation history
            history_arg = mock_impersonate.call_args[1]["conversation_history"]
            assert len(history_arg) >= 2
            assert history_arg[0]["role"] == "user"
            assert history_arg[0]["content"] == "Hello"
            assert history_arg[1]["role"] == "assistant"
            assert history_arg[1]["content"] == "Assistant response"

    @pytest.mark.asyncio
    async def test_generate_parse_subtopic(self, mock_llm_client):
        """Test the _generate_parse_subtopic helper method."""
        generator = AsyncNemotronGenerator(mock_llm_client)

        # Patch the methods
        with patch.object(
            AsyncNemotronGenerator, "generate_subtopics", new_callable=AsyncMock
        ) as mock_subtopics:
            with patch.object(
                AsyncNemotronGenerator, "_try_convert_yaml_list", new_callable=AsyncMock
            ) as mock_convert:
                # Set up return values
                mock_subtopics.return_value = ["Raw subtopics"]
                mock_convert.return_value = ["Physics", "Chemistry"]

                # Call the method
                result = await generator._generate_parse_subtopic(
                    macro_topic="Science",
                    n_subtopics=2,
                    model="test_model",
                    subtopic_prompt_template="template",
                    yaml_conversion_prompt_template="conversion_template",
                    base_model_kwargs={},
                    conversion_model_kwargs={},
                    ignore_conversion_failure=False,
                )

                # Check that methods were called correctly
                mock_subtopics.assert_called_once_with(
                    macro_topic="Science",
                    n_subtopics=2,
                    model="test_model",
                    prompt_template="template",
                    model_kwargs={},
                )

                mock_convert.assert_called_once()

                # Check result
                assert result == ["Physics", "Chemistry"]

    @pytest.mark.asyncio
    async def test_generate_parse_openline(self, mock_llm_client):
        """Test the _generate_parse_openline helper method."""
        generator = AsyncNemotronGenerator(mock_llm_client)

        # Patch the methods
        with patch.object(
            AsyncNemotronGenerator,
            "generate_open_qa_from_topic",
            new_callable=AsyncMock,
        ) as mock_qa:
            with patch.object(
                AsyncNemotronGenerator, "_try_convert_yaml_list", new_callable=AsyncMock
            ) as mock_convert:
                # Set up return values
                mock_qa.return_value = ["Raw QA"]
                mock_convert.return_value = ["Question 1", "Question 2"]

                # Call the method
                result = await generator._generate_parse_openline(
                    subtopic="Physics",
                    n_openlines=2,
                    model="test_model",
                    open_qa_from_topics_prompt_template="template",
                    yaml_conversion_prompt_template="conversion_template",
                    base_model_kwargs={},
                    conversion_model_kwargs={},
                    ignore_conversion_failure=False,
                )

                # Check methods were called correctly
                mock_qa.assert_called_once_with(
                    topic="Physics",
                    n_openlines=2,
                    model="test_model",
                    prompt_template="template",
                    model_kwargs={},
                )

                mock_convert.assert_called_once()

                # Check result
                assert result == ["Question 1", "Question 2"]

    @pytest.mark.asyncio
    async def test_run_open_qa_pipeline(self, mock_llm_client):
        """Test run_open_qa_pipeline pipeline method."""
        generator = AsyncNemotronGenerator(mock_llm_client)

        # Patch all the required methods
        with patch.object(
            AsyncNemotronGenerator, "generate_macro_topics", new_callable=AsyncMock
        ) as mock_macro_topics:
            with patch.object(
                AsyncNemotronGenerator,
                "convert_response_to_yaml_list",
                new_callable=AsyncMock,
            ) as mock_convert:
                with patch.object(
                    AsyncNemotronGenerator,
                    "_generate_parse_subtopic",
                    new_callable=AsyncMock,
                ) as mock_parse_subtopic:
                    with patch.object(
                        AsyncNemotronGenerator,
                        "_generate_parse_openline",
                        new_callable=AsyncMock,
                    ) as mock_parse_openline:
                        with patch.object(
                            AsyncNemotronGenerator,
                            "_revise_parse_openline",
                            new_callable=AsyncMock,
                        ) as mock_revise_openline:

                            # Set up return values - use cycle to create infinite iterators for side effects
                            mock_macro_topics.return_value = ["Macro topics response"]
                            mock_convert.return_value = [
                                "Science",
                                "History",
                            ]  # Macro topics conversion
                            mock_parse_subtopic.side_effect = cycle(
                                [
                                    ["Physics", "Chemistry"],  # Subtopics for Science
                                    ["Ancient", "Modern"],  # Subtopics for History
                                ]
                            )
                            mock_parse_openline.side_effect = cycle(
                                [
                                    [
                                        "Physics Q1",
                                        "Physics Q2",
                                    ],  # Questions for Physics
                                    [
                                        "Chemistry Q1",
                                        "Chemistry Q2",
                                    ],  # Questions for Chemistry
                                    [
                                        "Ancient Q1",
                                        "Ancient Q2",
                                    ],  # Questions for Ancient
                                    ["Modern Q1", "Modern Q2"],  # Questions for Modern
                                ]
                            )
                            mock_revise_openline.side_effect = cycle(
                                [
                                    ["Revised Physics Q1", "Revised Physics Q2"],
                                    ["Revised Chemistry Q1", "Revised Chemistry Q2"],
                                    ["Revised Ancient Q1", "Revised Ancient Q2"],
                                    ["Revised Modern Q1", "Revised Modern Q2"],
                                ]
                            )

                            # Call the pipeline
                            result = await generator.run_open_qa_pipeline(
                                n_macro_topics=2,
                                n_subtopics=2,
                                n_openlines=2,
                                n_revisions=2,
                                model="test_model",
                            )

                            # Check that methods were called correctly
                            mock_macro_topics.assert_called_once()
                            mock_convert.assert_called_once()
                            assert (
                                mock_parse_subtopic.call_count == 2
                            )  # Called for each macro topic
                            assert (
                                mock_parse_openline.call_count >= 4
                            )  # Called at least once for each subtopic
                            assert (
                                mock_revise_openline.call_count >= 8
                            )  # Called for each openline item

                            # Check the result - should have all the revised questions
                            assert (
                                len(result) >= 8
                            )  # At least 2 macro topics x 2 subtopics x 2 questions
                            assert (
                                "Revised" in result[0]
                            )  # Verify we're getting the revised questions

    @pytest.mark.asyncio
    async def test_run_writing_pipeline(self, mock_llm_client):
        """Test run_writing_pipeline pipeline method."""
        generator = AsyncNemotronGenerator(mock_llm_client)

        # Patch the methods
        with patch.object(
            AsyncNemotronGenerator,
            "_generate_parse_writing_task",
            new_callable=AsyncMock,
        ) as mock_generate_task:
            with patch.object(
                AsyncNemotronGenerator,
                "_revise_parse_writing_task",
                new_callable=AsyncMock,
            ) as mock_revise_task:

                # Set up return values - use cycle to create infinite iterators for side effects
                mock_generate_task.side_effect = cycle(
                    [
                        ["Task 1", "Task 2"],  # For Science/Essay
                        ["Task 3", "Task 4"],  # For Science/Blog
                        ["Task 5", "Task 6"],  # For History/Essay
                        ["Task 7", "Task 8"],  # For History/Blog
                    ]
                )
                mock_revise_task.side_effect = cycle(
                    [
                        ["Revised Task 1", "Revised Task 2"],
                        ["Revised Task 3", "Revised Task 4"],
                        ["Revised Task 5", "Revised Task 6"],
                        ["Revised Task 7", "Revised Task 8"],
                    ]
                )

                topics = ["Science", "History"]
                text_material_types = ["Essay", "Blog Post"]

                # Call the pipeline
                result = await generator.run_writing_pipeline(
                    topics=topics,
                    text_material_types=text_material_types,
                    n_openlines=2,
                    n_revisions=2,
                    model="test_model",
                )

                # Check methods were called correctly
                assert (
                    mock_generate_task.call_count == 4
                )  # Called for each topic/material type pair
                assert (
                    mock_revise_task.call_count >= 8
                )  # Called for each task item (2 per generate call)

                # Check the result - may have more than 8 items if _revise_parse_writing_task is called more times
                assert (
                    len(result) >= 8
                )  # At least 2 topics x 2 material types x 2 tasks
                assert "Revised" in result[0]  # Verify we're getting the revised tasks

    @pytest.mark.asyncio
    async def test_run_pipeline_with_yaml_conversion_error(self, mock_llm_client):
        """Test pipeline error handling when YamlConversionError occurs."""
        generator = AsyncNemotronGenerator(mock_llm_client)

        # Patch methods
        with patch.object(
            AsyncNemotronGenerator, "generate_macro_topics", new_callable=AsyncMock
        ) as mock_macro_topics:
            with patch.object(
                AsyncNemotronGenerator,
                "convert_response_to_yaml_list",
                new_callable=AsyncMock,
            ) as mock_convert:
                with patch.object(
                    AsyncNemotronGenerator, "_gather", new_callable=AsyncMock
                ) as mock_gather:
                    # Set up mocks - first throw an error to test error propagation
                    mock_macro_topics.return_value = ["Raw response"]
                    mock_convert.side_effect = YamlConversionError(
                        "Test conversion error"
                    )
                    mock_gather.return_value = [
                        "Raw response"
                    ]  # Mock the gather method to avoid range() issue

                    # Verify that the error is propagated when ignore_conversion_failure=False
                    with pytest.raises(YamlConversionError):
                        await generator.run_open_qa_pipeline(
                            n_macro_topics=2,
                            n_subtopics=2,
                            n_openlines=2,
                            n_revisions=2,
                            model="test_model",
                            ignore_conversion_failure=False,
                        )

                    # Reset and now test with ignore_conversion_failure=True
                    mock_convert.reset_mock()
                    mock_convert.side_effect = None
                    mock_convert.return_value = (
                        []
                    )  # Return empty list to simulate ignored error
                    mock_gather.return_value = (
                        []
                    )  # Ensure gather also returns empty list

                    # Call with ignore_conversion_failure=True (expect empty result)
                    result = await generator.run_open_qa_pipeline(
                        n_macro_topics=2,
                        n_subtopics=2,
                        n_openlines=2,
                        n_revisions=2,
                        model="test_model",
                        ignore_conversion_failure=True,
                    )

                    # Should return empty list due to empty macro topics
                    assert result == []

    @pytest.mark.asyncio
    async def test_revise_parse_openline(self, mock_llm_client):
        """Test the _revise_parse_openline helper method."""
        generator = AsyncNemotronGenerator(mock_llm_client)

        # Patch the methods
        with patch.object(
            AsyncNemotronGenerator,
            "revise_open_qa",
            new_callable=AsyncMock,
        ) as mock_revise:
            with patch.object(
                AsyncNemotronGenerator, "_try_convert_yaml_list", new_callable=AsyncMock
            ) as mock_convert:
                # Set up return values
                mock_revise.return_value = ["Raw revision"]
                mock_convert.return_value = ["Revised Q1", "Revised Q2"]

                # Call the method
                result = await generator._revise_parse_openline(
                    openline="Original question",
                    n_revisions=2,
                    model="test_model",
                    revise_open_qa_prompt_template="template",
                    yaml_conversion_prompt_template="conversion_template",
                    base_model_kwargs={},
                    conversion_model_kwargs={},
                    ignore_conversion_failure=False,
                )

                # Check methods were called correctly
                mock_revise.assert_called_once_with(
                    openline="Original question",
                    n_revisions=2,
                    model="test_model",
                    prompt_template="template",
                    model_kwargs={},
                )

                mock_convert.assert_called_once()

                # Check result
                assert result == ["Revised Q1", "Revised Q2"]

    @pytest.mark.asyncio
    async def test_generate_parse_writing_task(self, mock_llm_client):
        """Test the _generate_parse_writing_task helper method."""
        generator = AsyncNemotronGenerator(mock_llm_client)

        # Patch the methods
        with patch.object(
            AsyncNemotronGenerator,
            "generate_writing_tasks",
            new_callable=AsyncMock,
        ) as mock_generate:
            with patch.object(
                AsyncNemotronGenerator, "_try_convert_yaml_list", new_callable=AsyncMock
            ) as mock_convert:
                # Set up return values
                mock_generate.return_value = ["Raw writing tasks"]
                mock_convert.return_value = ["Task 1", "Task 2"]

                # Call the method
                result = await generator._generate_parse_writing_task(
                    topic="Science",
                    material="Essay",
                    n_openlines=2,
                    model="test_model",
                    writing_task_prompt_template="template",
                    yaml_conversion_prompt_template="conversion_template",
                    base_model_kwargs={},
                    conversion_model_kwargs={},
                    ignore_conversion_failure=False,
                )

                # Check methods were called correctly
                mock_generate.assert_called_once_with(
                    topic="Science",
                    text_material_type="Essay",
                    n_openlines=2,
                    model="test_model",
                    prompt_template="template",
                    model_kwargs={},
                )

                mock_convert.assert_called_once()

                # Check result
                assert result == ["Task 1", "Task 2"]

    @pytest.mark.asyncio
    async def test_revise_parse_writing_task(self, mock_llm_client):
        """Test the _revise_parse_writing_task helper method."""
        generator = AsyncNemotronGenerator(mock_llm_client)

        # Patch the methods
        with patch.object(
            AsyncNemotronGenerator,
            "revise_writing_tasks",
            new_callable=AsyncMock,
        ) as mock_revise:
            with patch.object(
                AsyncNemotronGenerator, "_try_convert_yaml_list", new_callable=AsyncMock
            ) as mock_convert:
                # Set up return values
                mock_revise.return_value = ["Raw revision"]
                mock_convert.return_value = ["Revised Task 1", "Revised Task 2"]

                # Call the method
                result = await generator._revise_parse_writing_task(
                    task="Original task",
                    n_revisions=2,
                    model="test_model",
                    revise_writing_task_prompt_template="template",
                    yaml_conversion_prompt_template="conversion_template",
                    base_model_kwargs={},
                    conversion_model_kwargs={},
                    ignore_conversion_failure=False,
                )

                # Check methods were called correctly
                mock_revise.assert_called_once_with(
                    openline="Original task",
                    n_revisions=2,
                    model="test_model",
                    prompt_template="template",
                    model_kwargs={},
                )

                mock_convert.assert_called_once()

                # Check result
                assert result == ["Revised Task 1", "Revised Task 2"]

    @pytest.mark.asyncio
    async def test_generate_parse_closed_qa(self, mock_llm_client):
        """Test the _generate_parse_closed_qa helper method."""
        generator = AsyncNemotronGenerator(mock_llm_client)

        # Patch the methods
        with patch.object(
            AsyncNemotronGenerator,
            "generate_closed_qa_instructions",
            new_callable=AsyncMock,
        ) as mock_generate:
            with patch.object(
                AsyncNemotronGenerator, "_try_convert_yaml_list", new_callable=AsyncMock
            ) as mock_convert:
                # Set up return values
                mock_generate.return_value = ["Raw closed QA"]
                mock_convert.return_value = ["Question 1", "Question 2"]

                # Call the method
                result = await generator._generate_parse_closed_qa(
                    document_id=1,
                    document="Document content",
                    n_openlines=2,
                    model="test_model",
                    closed_qa_prompt_template="template",
                    yaml_conversion_prompt_template="conversion_template",
                    base_model_kwargs={},
                    conversion_model_kwargs={},
                    ignore_conversion_failure=False,
                )

                # Check methods were called correctly
                mock_generate.assert_called_once_with(
                    document="Document content",
                    n_openlines=2,
                    model="test_model",
                    prompt_template="template",
                    model_kwargs={},
                )

                mock_convert.assert_called_once()

                # Check result has the document ID and questions
                assert len(result) == 2
                assert all(isinstance(item, tuple) for item in result)
                assert result[0][0] == 1  # document_id
                assert result[0][1] in ["Question 1", "Question 2"]
                assert result[1][0] == 1  # document_id
                assert result[1][1] in ["Question 1", "Question 2"]

    @pytest.mark.asyncio
    async def test_run_closed_qa_pipeline(self, mock_llm_client):
        """Test run_closed_qa_pipeline pipeline method."""
        generator = AsyncNemotronGenerator(mock_llm_client)

        # Patch the methods
        with patch.object(
            AsyncNemotronGenerator,
            "_generate_parse_closed_qa",
            new_callable=AsyncMock,
        ) as mock_parse_qa:
            with patch.object(
                AsyncNemotronGenerator, "_gather", new_callable=AsyncMock
            ) as mock_gather:
                # Set up return values
                mock_parse_qa.side_effect = [
                    [(0, "Question 1"), (0, "Question 2")],  # Document 0
                    [(1, "Question 3"), (1, "Question 4")],  # Document 1
                ]
                mock_gather.return_value = [
                    [(0, "Question 1"), (0, "Question 2")],
                    [(1, "Question 3"), (1, "Question 4")],
                ]

                documents = ["Document 1", "Document 2"]

                # Call the pipeline
                result = await generator.run_closed_qa_pipeline(
                    documents=documents,
                    n_openlines=2,
                    model="test_model",
                )

                # Check methods were called correctly
                assert mock_parse_qa.call_count == 2  # Called for each document
                mock_gather.assert_called_once()

                # Verify the result format
                assert len(result) == 4  # 2 documents x 2 questions each
                assert all(isinstance(item, tuple) for item in result)
                assert result[0][0] == 0  # First document index
                assert result[2][0] == 1  # Second document index

    @pytest.mark.asyncio
    async def test_generate_parse_math_subtopic(self, mock_llm_client):
        """Test the _generate_parse_math_subtopic helper method."""
        generator = AsyncNemotronGenerator(mock_llm_client)

        # Patch the methods
        with patch.object(
            AsyncNemotronGenerator,
            "generate_math_subtopics",
            new_callable=AsyncMock,
        ) as mock_subtopics:
            with patch.object(
                AsyncNemotronGenerator, "_try_convert_yaml_list", new_callable=AsyncMock
            ) as mock_convert:
                # Set up return values
                mock_subtopics.return_value = ["Raw math subtopics"]
                mock_convert.return_value = ["Algebra", "Calculus"]

                # Call the method
                result = await generator._generate_parse_math_subtopic(
                    macro_topic="Mathematics",
                    n_subtopics=2,
                    model="test_model",
                    subtopic_prompt_template="template",
                    yaml_conversion_prompt_template="conversion_template",
                    base_model_kwargs={},
                    conversion_model_kwargs={},
                    ignore_conversion_failure=False,
                )

                # Check methods were called correctly
                mock_subtopics.assert_called_once_with(
                    macro_topic="Mathematics",
                    n_subtopics=2,
                    model="test_model",
                    prompt_template="template",
                    model_kwargs={},
                )

                mock_convert.assert_called_once()

                # Check result
                assert result == ["Algebra", "Calculus"]

    @pytest.mark.asyncio
    async def test_generate_parse_math_openline(self, mock_llm_client):
        """Test the _generate_parse_math_openline helper method."""
        generator = AsyncNemotronGenerator(mock_llm_client)

        # Patch the methods
        with patch.object(
            AsyncNemotronGenerator,
            "generate_math_problem",
            new_callable=AsyncMock,
        ) as mock_problem:
            with patch.object(
                AsyncNemotronGenerator, "_try_convert_yaml_list", new_callable=AsyncMock
            ) as mock_convert:
                # Set up return values
                mock_problem.return_value = ["Raw math problem"]
                mock_convert.return_value = ["Problem 1", "Problem 2"]

                # Call the method
                result = await generator._generate_parse_math_openline(
                    subtopic="Algebra",
                    n_openlines=2,
                    model="test_model",
                    math_problem_prompt_template="template",
                    yaml_conversion_prompt_template="conversion_template",
                    base_model_kwargs={},
                    conversion_model_kwargs={},
                    ignore_conversion_failure=False,
                )

                # Check methods were called correctly
                mock_problem.assert_called_once_with(
                    topic="Algebra",
                    n_openlines=2,
                    model="test_model",
                    prompt_template="template",
                    model_kwargs={},
                )

                mock_convert.assert_called_once()

                # Check result
                assert result == ["Problem 1", "Problem 2"]

    @pytest.mark.asyncio
    async def test_run_math_pipeline(self, mock_llm_client):
        """Test run_math_pipeline pipeline method."""
        generator = AsyncNemotronGenerator(mock_llm_client)

        # Patch all the required methods
        with patch.object(
            AsyncNemotronGenerator, "generate_math_macro_topics", new_callable=AsyncMock
        ) as mock_macro_topics:
            with patch.object(
                AsyncNemotronGenerator,
                "convert_response_to_yaml_list",
                new_callable=AsyncMock,
            ) as mock_convert:
                with patch.object(
                    AsyncNemotronGenerator,
                    "_generate_parse_math_subtopic",
                    new_callable=AsyncMock,
                ) as mock_parse_subtopic:
                    with patch.object(
                        AsyncNemotronGenerator,
                        "_generate_parse_math_openline",
                        new_callable=AsyncMock,
                    ) as mock_parse_openline:
                        with patch.object(
                            AsyncNemotronGenerator, "_gather", new_callable=AsyncMock
                        ) as mock_gather:

                            # Set up return values
                            mock_macro_topics.return_value = ["Math topics response"]
                            mock_convert.return_value = [
                                "Algebra",
                                "Geometry",
                            ]  # Macro topics conversion
                            mock_parse_subtopic.side_effect = [
                                ["Equations", "Polynomials"],  # Subtopics for Algebra
                                ["Triangles", "Circles"],  # Subtopics for Geometry
                            ]
                            mock_parse_openline.side_effect = [
                                ["Problem 1", "Problem 2"]
                            ] * 4  # For each subtopic

                            # Setup gather to return all problems
                            mock_gather.side_effect = [
                                [
                                    ["Equations", "Polynomials"],
                                    ["Triangles", "Circles"],
                                ],  # First gather for subtopics
                                [
                                    ["Problem 1", "Problem 2"],
                                    ["Problem 1", "Problem 2"],
                                    ["Problem 1", "Problem 2"],
                                    ["Problem 1", "Problem 2"],
                                ],  # Second gather for problems
                            ]

                            # Call the pipeline
                            result = await generator.run_math_pipeline(
                                n_macro_topics=2,
                                school_level="High School",
                                n_subtopics=2,
                                n_openlines=2,
                                model="test_model",
                            )

                            # Check that methods were called correctly
                            mock_macro_topics.assert_called_once_with(
                                n_macro_topics=2,
                                school_level="High School",
                                model="test_model",
                                prompt_template=mock_macro_topics.call_args[1][
                                    "prompt_template"
                                ],
                                model_kwargs={},
                            )
                            mock_convert.assert_called_once()
                            assert mock_gather.call_count >= 2

                            # Check the result
                            assert (
                                len(result) == 8
                            )  # Should have 8 problems (2 macro x 2 subtopics x 2 problems)
                            assert all("Problem" in item for item in result)

    @pytest.mark.asyncio
    async def test_generate_parse_python_subtopic(self, mock_llm_client):
        """Test the _generate_parse_python_subtopic helper method."""
        generator = AsyncNemotronGenerator(mock_llm_client)

        # Patch the methods
        with patch.object(
            AsyncNemotronGenerator,
            "generate_python_subtopics",
            new_callable=AsyncMock,
        ) as mock_subtopics:
            with patch.object(
                AsyncNemotronGenerator, "_try_convert_yaml_list", new_callable=AsyncMock
            ) as mock_convert:
                # Set up return values
                mock_subtopics.return_value = ["Raw python subtopics"]
                mock_convert.return_value = ["Lists", "Dictionaries"]

                # Call the method
                result = await generator._generate_parse_python_subtopic(
                    macro_topic="Data Structures",
                    n_subtopics=2,
                    model="test_model",
                    subtopic_prompt_template="template",
                    yaml_conversion_prompt_template="conversion_template",
                    base_model_kwargs={},
                    conversion_model_kwargs={},
                    ignore_conversion_failure=False,
                )

                # Check methods were called correctly
                mock_subtopics.assert_called_once_with(
                    macro_topic="Data Structures",
                    n_subtopics=2,
                    model="test_model",
                    prompt_template="template",
                    model_kwargs={},
                )

                mock_convert.assert_called_once()

                # Check result
                assert result == ["Lists", "Dictionaries"]

    @pytest.mark.asyncio
    async def test_generate_parse_python_openline(self, mock_llm_client):
        """Test the _generate_parse_python_openline helper method."""
        generator = AsyncNemotronGenerator(mock_llm_client)

        # Patch the methods
        with patch.object(
            AsyncNemotronGenerator,
            "generate_python_problem",
            new_callable=AsyncMock,
        ) as mock_problem:
            with patch.object(
                AsyncNemotronGenerator, "_try_convert_yaml_list", new_callable=AsyncMock
            ) as mock_convert:
                # Set up return values
                mock_problem.return_value = ["Raw python problem"]
                mock_convert.return_value = ["Problem 1", "Problem 2"]

                # Call the method
                result = await generator._generate_parse_python_openline(
                    subtopic="Lists",
                    n_openlines=2,
                    model="test_model",
                    python_problem_prompt_template="template",
                    yaml_conversion_prompt_template="conversion_template",
                    base_model_kwargs={},
                    conversion_model_kwargs={},
                    ignore_conversion_failure=False,
                )

                # Check methods were called correctly
                mock_problem.assert_called_once_with(
                    topic="Lists",
                    n_openlines=2,
                    model="test_model",
                    prompt_template="template",
                    model_kwargs={},
                )

                mock_convert.assert_called_once()

                # Check result
                assert result == ["Problem 1", "Problem 2"]

    @pytest.mark.asyncio
    async def test_run_python_pipeline(self, mock_llm_client):
        """Test run_python_pipeline pipeline method."""
        generator = AsyncNemotronGenerator(mock_llm_client)

        # Patch all the required methods
        with patch.object(
            AsyncNemotronGenerator,
            "generate_python_macro_topics",
            new_callable=AsyncMock,
        ) as mock_macro_topics:
            with patch.object(
                AsyncNemotronGenerator,
                "convert_response_to_yaml_list",
                new_callable=AsyncMock,
            ) as mock_convert:
                with patch.object(
                    AsyncNemotronGenerator,
                    "_generate_parse_python_subtopic",
                    new_callable=AsyncMock,
                ) as mock_parse_subtopic:
                    with patch.object(
                        AsyncNemotronGenerator,
                        "_generate_parse_python_openline",
                        new_callable=AsyncMock,
                    ) as mock_parse_openline:
                        with patch.object(
                            AsyncNemotronGenerator, "_gather", new_callable=AsyncMock
                        ) as mock_gather:

                            # Set up return values
                            mock_macro_topics.return_value = ["Python topics response"]
                            mock_convert.return_value = [
                                "Data Structures",
                                "Functions",
                            ]  # Macro topics conversion
                            mock_parse_subtopic.side_effect = [
                                [
                                    "Lists",
                                    "Dictionaries",
                                ],  # Subtopics for Data Structures
                                ["Lambda", "Decorators"],  # Subtopics for Functions
                            ]
                            mock_parse_openline.side_effect = [
                                ["Problem 1", "Problem 2"]
                            ] * 4  # For each subtopic

                            # Setup gather to return all problems
                            mock_gather.side_effect = [
                                [
                                    ["Lists", "Dictionaries"],
                                    ["Lambda", "Decorators"],
                                ],  # First gather for subtopics
                                [
                                    ["Problem 1", "Problem 2"],
                                    ["Problem 1", "Problem 2"],
                                    ["Problem 1", "Problem 2"],
                                    ["Problem 1", "Problem 2"],
                                ],  # Second gather for problems
                            ]

                            # Call the pipeline
                            result = await generator.run_python_pipeline(
                                n_macro_topics=2,
                                n_subtopics=2,
                                n_openlines=2,
                                model="test_model",
                            )

                            # Check that methods were called correctly
                            mock_macro_topics.assert_called_once_with(
                                n_macro_topics=2,
                                model="test_model",
                                prompt_template=mock_macro_topics.call_args[1][
                                    "prompt_template"
                                ],
                                model_kwargs={},
                            )
                            mock_convert.assert_called_once()
                            assert mock_gather.call_count >= 2

                            # Check the result
                            assert (
                                len(result) == 8
                            )  # Should have 8 problems (2 macro x 2 subtopics x 2 problems)
                            assert all("Problem" in item for item in result)

    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self, mock_llm_client):
        """Test error handling in all pipeline methods."""
        generator = AsyncNemotronGenerator(mock_llm_client)

        # Test open_qa_pipeline with error handling
        with patch.object(generator, "convert_response_to_yaml_list") as mock_convert:
            # Setup to raise error for ignore_conversion_failure=False test
            mock_convert.side_effect = YamlConversionError("Test error")

            # Should raise with ignore_conversion_failure=False
            with pytest.raises(YamlConversionError):
                await generator.run_open_qa_pipeline(
                    n_macro_topics=2,
                    n_subtopics=2,
                    n_openlines=2,
                    n_revisions=2,
                    model="test_model",
                    ignore_conversion_failure=False,
                )

            # Reset the mock for the next test and configure it to handle both calls appropriately
            mock_convert.reset_mock()
            # First call to generate macro topics fails, but we have additional_macro_topics
            # Later calls for subtopics and openlines succeed
            # Use cycle to create an infinite iterator that won't run out of values
            mock_convert.side_effect = cycle(
                [
                    YamlConversionError("Test error"),  # First call fails
                    ["Subtopic 1", "Subtopic 2"],  # Subsequent calls succeed
                    ["Question 1", "Question 2"],
                    ["Revised 1", "Revised 2"],
                ]
            )

            # Should not raise with ignore_conversion_failure=True
            result = await generator.run_open_qa_pipeline(
                n_macro_topics=2,
                n_subtopics=2,
                n_openlines=2,
                n_revisions=2,
                model="test_model",
                ignore_conversion_failure=True,
                additional_macro_topics=[
                    "Backup Topic"
                ],  # Ensure we have something to process
            )
            assert isinstance(result, list)

        # Test writing pipeline
        with patch.object(generator, "convert_response_to_yaml_list") as mock_convert:
            # Setup to raise error for ignore_conversion_failure=False test
            mock_convert.side_effect = YamlConversionError("Test error")

            # Should raise with ignore_conversion_failure=False
            with pytest.raises(YamlConversionError):
                await generator.run_writing_pipeline(
                    topics=["Topic"],
                    text_material_types=["Essay"],
                    n_openlines=2,
                    n_revisions=2,
                    model="test_model",
                    ignore_conversion_failure=False,
                )

            # Reset the mock for the next test and configure it
            mock_convert.reset_mock()
            # Use cycle to create an infinite iterator that won't run out of values
            mock_convert.side_effect = cycle(
                [
                    ["Task 1", "Task 2"],  # First call succeeds
                    ["Revised 1", "Revised 2"],  # Second call succeeds
                ]
            )

            # Should not raise with ignore_conversion_failure=True
            result = await generator.run_writing_pipeline(
                topics=["Topic"],
                text_material_types=["Essay"],
                n_openlines=2,
                n_revisions=2,
                model="test_model",
                ignore_conversion_failure=True,
            )
            assert isinstance(result, list)

        # Test closed_qa pipeline
        with patch.object(generator, "convert_response_to_yaml_list") as mock_convert:
            # Setup to raise error for ignore_conversion_failure=False test
            mock_convert.side_effect = YamlConversionError("Test error")

            # Should raise with ignore_conversion_failure=False
            with pytest.raises(YamlConversionError):
                await generator.run_closed_qa_pipeline(
                    documents=["Test document"],
                    n_openlines=2,
                    model="test_model",
                    ignore_conversion_failure=False,
                )

            # Reset the mock for the next test and configure it
            mock_convert.reset_mock()
            mock_convert.return_value = ["Question 1", "Question 2"]

            # Should not raise with ignore_conversion_failure=True
            result = await generator.run_closed_qa_pipeline(
                documents=["Test document"],
                n_openlines=2,
                model="test_model",
                ignore_conversion_failure=True,
            )
            assert isinstance(result, list)

        # Test math pipeline
        with patch.object(generator, "convert_response_to_yaml_list") as mock_convert:
            # Setup to raise error for ignore_conversion_failure=False test
            mock_convert.side_effect = YamlConversionError("Test error")

            # Should raise with ignore_conversion_failure=False
            with pytest.raises(YamlConversionError):
                await generator.run_math_pipeline(
                    n_macro_topics=2,
                    school_level="High School",
                    n_subtopics=2,
                    n_openlines=2,
                    model="test_model",
                    ignore_conversion_failure=False,
                )

            # Reset the mock for the next test and configure it
            mock_convert.reset_mock()
            # Use cycle to create an infinite iterator that won't run out of values
            mock_convert.side_effect = cycle(
                [
                    YamlConversionError("Test error"),  # First call fails
                    ["Subtopic 1", "Subtopic 2"],  # Subsequent calls succeed
                    ["Problem 1", "Problem 2"],
                ]
            )

            # Should not raise with ignore_conversion_failure=True
            result = await generator.run_math_pipeline(
                n_macro_topics=2,
                school_level="High School",
                n_subtopics=2,
                n_openlines=2,
                model="test_model",
                ignore_conversion_failure=True,
                additional_macro_topics=[
                    "Backup Topic"
                ],  # Ensure we have something to process
            )
            assert isinstance(result, list)

        # Test python pipeline
        with patch.object(generator, "convert_response_to_yaml_list") as mock_convert:
            # Setup to raise error for ignore_conversion_failure=False test
            mock_convert.side_effect = YamlConversionError("Test error")

            # Should raise with ignore_conversion_failure=False
            with pytest.raises(YamlConversionError):
                await generator.run_python_pipeline(
                    n_macro_topics=2,
                    n_subtopics=2,
                    n_openlines=2,
                    model="test_model",
                    ignore_conversion_failure=False,
                )

            # Reset the mock for the next test and configure it
            mock_convert.reset_mock()
            # Use cycle to create an infinite iterator that won't run out of values
            mock_convert.side_effect = cycle(
                [
                    YamlConversionError("Test error"),  # First call fails
                    ["Subtopic 1", "Subtopic 2"],  # Subsequent calls succeed
                    ["Problem 1", "Problem 2"],
                ]
            )

            # Should not raise with ignore_conversion_failure=True
            result = await generator.run_python_pipeline(
                n_macro_topics=2,
                n_subtopics=2,
                n_openlines=2,
                model="test_model",
                ignore_conversion_failure=True,
                additional_macro_topics=[
                    "Backup Topic"
                ],  # Ensure we have something to process
            )
            assert isinstance(result, list)
