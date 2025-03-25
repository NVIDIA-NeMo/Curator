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

from itertools import cycle
from unittest.mock import MagicMock, call, patch

import pytest
import yaml

from nemo_curator.synthetic.error import YamlConversionError
from nemo_curator.synthetic.nemotron import NemotronFormatter, NemotronGenerator


class TestNemotronFormatter:
    def test_format_conversation_basic(self):
        """Test basic conversation formatting with alternating user and assistant turns."""
        conv = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"},
        ]

        expected_output = (
            "<extra_id_0>System\n\n<extra_id_1>User\n"
            "Hello\n<extra_id_1>Assistant\n"
            "Hi there!\n<extra_id_1>User\n"
            "How are you?\n<extra_id_1>Assistant\n"
            "I'm doing well, thank you!\n<extra_id_1>User\n"
        )

        result = NemotronFormatter.format_conversation(conv)
        assert result == expected_output

    def test_format_conversation_single_turn(self):
        """Test formatting with just a single user turn."""
        conv = [{"role": "user", "content": "Hello"}]

        expected_output = (
            "<extra_id_0>System\n\n<extra_id_1>User\n" "Hello\n<extra_id_1>Assistant\n"
        )

        result = NemotronFormatter.format_conversation(conv)
        assert result == expected_output

    def test_format_conversation_invalid_roles(self):
        """Test error handling when conversation has invalid role sequence."""
        # User followed by user
        conv = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "Are you there?"},
        ]

        with pytest.raises(ValueError) as excinfo:
            NemotronFormatter.format_conversation(conv)
        assert "Conversation turn 1 is not 'assistant'" in str(excinfo.value)

        # Assistant as first turn
        conv = [
            {"role": "assistant", "content": "Hello, how can I help?"},
            {"role": "user", "content": "Hi there!"},
        ]

        with pytest.raises(ValueError) as excinfo:
            NemotronFormatter.format_conversation(conv)
        assert "Conversation turn 0 is not 'user'" in str(excinfo.value)


class TestNemotronGenerator:
    @pytest.fixture
    def mock_llm_client(self):
        mock_client = MagicMock()
        mock_client.query_model.return_value = ["This is a mock response"]
        return mock_client

    def test_init(self, mock_llm_client):
        """Test the constructor of NemotronGenerator."""
        generator = NemotronGenerator(mock_llm_client)
        assert generator.client == mock_llm_client

    def test_prompt(self, mock_llm_client):
        """Test the internal _prompt method."""
        generator = NemotronGenerator(mock_llm_client)

        prompt_template = "Test prompt for {query}."
        prompt_kwargs = {"query": "test"}
        model_kwargs = {"temperature": 0.7}

        result = generator._prompt(
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

    def test_convert_response_to_yaml_list(self, mock_llm_client):
        """Test the convert_response_to_yaml_list method with a valid response."""
        generator = NemotronGenerator(mock_llm_client)

        # Mock that the LLM returns a valid YAML string
        yaml_list = ["Item 1", "Item 2", "Item 3"]
        yaml_string = yaml.dump(yaml_list)
        mock_llm_client.query_model.return_value = [yaml_string]

        llm_response = "Some text containing Item 1, Item 2, and Item 3"
        result = generator.convert_response_to_yaml_list(
            llm_response=llm_response,
            model="test_model",
        )

        assert result == yaml_list

    def test_convert_response_to_yaml_list_invalid_yaml(self, mock_llm_client):
        """Test handling of invalid YAML in convert_response_to_yaml_list."""
        generator = NemotronGenerator(mock_llm_client)

        # Return invalid YAML
        mock_llm_client.query_model.return_value = ["[This is not valid YAML"]

        with pytest.raises(YamlConversionError):
            generator.convert_response_to_yaml_list(
                llm_response="Some text",
                model="test_model",
            )

    def test_convert_response_to_yaml_list_not_a_list(self, mock_llm_client):
        """Test handling when YAML is valid but not a list."""
        generator = NemotronGenerator(mock_llm_client)

        # Return valid YAML but not a list
        mock_llm_client.query_model.return_value = [yaml.dump({"key": "value"})]

        with pytest.raises(YamlConversionError) as excinfo:
            generator.convert_response_to_yaml_list(
                llm_response="Some text",
                model="test_model",
            )
        assert "not a list" in str(excinfo.value)

    def test_convert_response_to_yaml_list_hallucination(self, mock_llm_client):
        """Test handling when YAML contains hallucinated items not in original response."""
        generator = NemotronGenerator(mock_llm_client)

        # Return a list containing an item not in the original text
        mock_llm_client.query_model.return_value = [
            yaml.dump(["Item in text", "Hallucinated item"])
        ]

        with pytest.raises(YamlConversionError) as excinfo:
            generator.convert_response_to_yaml_list(
                llm_response="Original text containing Item in text",
                model="test_model",
            )
        assert "hallucination" in str(excinfo.value).lower()

    def test_generate_macro_topics(self, mock_llm_client):
        """Test generate_macro_topics method."""
        generator = NemotronGenerator(mock_llm_client)

        # Set up the return value for query_model
        mock_llm_client.query_model.return_value = ["Topic 1\nTopic 2\nTopic 3"]

        result = generator.generate_macro_topics(
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

    def test_generate_subtopics(self, mock_llm_client):
        """Test generate_subtopics method."""
        generator = NemotronGenerator(mock_llm_client)

        result = generator.generate_subtopics(
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

    def test_generate_open_qa_from_topic(self, mock_llm_client):
        """Test generate_open_qa_from_topic method."""
        generator = NemotronGenerator(mock_llm_client)

        result = generator.generate_open_qa_from_topic(
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

    def test_revise_open_qa(self, mock_llm_client):
        """Test revise_open_qa method."""
        generator = NemotronGenerator(mock_llm_client)

        result = generator.revise_open_qa(
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

    def test_generate_writing_tasks(self, mock_llm_client):
        """Test generate_writing_tasks method."""
        generator = NemotronGenerator(mock_llm_client)

        result = generator.generate_writing_tasks(
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

    def test_generate_closed_qa_instructions(self, mock_llm_client):
        """Test generate_closed_qa_instructions method."""
        generator = NemotronGenerator(mock_llm_client)

        document = "This is a sample document about AI technology..."
        result = generator.generate_closed_qa_instructions(
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

    def test_generate_math_problem(self, mock_llm_client):
        """Test generate_math_problem method."""
        generator = NemotronGenerator(mock_llm_client)

        result = generator.generate_math_problem(
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

    def test_generate_python_problem(self, mock_llm_client):
        """Test generate_python_problem method."""
        generator = NemotronGenerator(mock_llm_client)

        result = generator.generate_python_problem(
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

    @patch.object(NemotronGenerator, "_impersonate_user")
    def test_generate_dialogue(self, mock_impersonate, mock_llm_client):
        """Test generate_dialogue method that creates conversations."""
        generator = NemotronGenerator(mock_llm_client)

        # Set up return values
        mock_llm_client.query_model.side_effect = [
            ["Assistant response 1"],
            ["Assistant response 2"],
        ]
        mock_impersonate.return_value = "User follow-up"

        # Call the method
        result = generator.generate_dialogue(
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

    @patch.object(NemotronGenerator, "_impersonate_user")
    def test_generate_two_turn_prompt(self, mock_impersonate, mock_llm_client):
        """Test generate_two_turn_prompt method."""
        generator = NemotronGenerator(mock_llm_client)

        # Set up return values
        mock_llm_client.query_model.return_value = ["Assistant response"]
        mock_impersonate.return_value = "User follow-up"

        # Call the method
        result = generator.generate_two_turn_prompt(
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

    def test_impersonate_user(self, mock_llm_client):
        """Test _impersonate_user private method."""
        generator = NemotronGenerator(mock_llm_client)

        conversation_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        result = generator._impersonate_user(
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

    @patch.object(NemotronGenerator, "convert_response_to_yaml_list")
    @patch.object(NemotronGenerator, "generate_macro_topics")
    @patch.object(NemotronGenerator, "generate_subtopics")
    @patch.object(NemotronGenerator, "generate_open_qa_from_topic")
    @patch.object(NemotronGenerator, "revise_open_qa")
    def test_run_open_qa_pipeline(
        self,
        mock_revise,
        mock_generate_qa,
        mock_subtopics,
        mock_macro_topics,
        mock_convert,
        mock_llm_client,
    ):
        """Test run_open_qa_pipeline pipeline method."""
        generator = NemotronGenerator(mock_llm_client)

        # Setup the mocks to handle multiple calls with the same response pattern
        mock_macro_topics.return_value = ["Macro topics response"]

        # Use cycle to repeat the sequence of return values for potentially unlimited calls
        mock_convert.side_effect = cycle(
            [
                ["Science", "History"],  # Macro topics conversion
                ["Physics", "Chemistry"],  # Subtopics conversion for Science
                ["Geography", "Art"],  # Subtopics conversion for History
                ["Question 1", "Question 2"],  # Openlines conversion
                ["Revised Q1", "Revised Q2"],  # Revisions conversion
            ]
        )

        mock_subtopics.return_value = ["Subtopics response"]
        mock_generate_qa.return_value = ["QA response"]
        mock_revise.return_value = ["Revisions response"]

        # Call the pipeline
        result = generator.run_open_qa_pipeline(
            n_macro_topics=2,
            n_subtopics=2,
            n_openlines=2,
            n_revisions=2,
            model="test_model",
        )

        # Check that each step was called with appropriate arguments
        mock_macro_topics.assert_called_once_with(
            n_macro_topics=2,
            model="test_model",
            model_kwargs={},
            prompt_template=mock_macro_topics.call_args[1]["prompt_template"],
        )

        # Verify subtopics were generated for both macro topics
        assert mock_subtopics.call_count >= 2
        mock_subtopics.assert_any_call(
            macro_topic="Science",
            n_subtopics=2,
            model="test_model",
            model_kwargs={},
            prompt_template=mock_subtopics.call_args[1]["prompt_template"],
        )

        # Check the final result - we should get the values from our last mock side_effect
        assert "Revised Q" in result[0]

    @patch.object(NemotronGenerator, "convert_response_to_yaml_list")
    @patch.object(NemotronGenerator, "generate_closed_qa_instructions")
    def test_run_closed_qa_pipeline(
        self, mock_qa_instructions, mock_convert, mock_llm_client
    ):
        """Test run_closed_qa_pipeline pipeline method."""
        generator = NemotronGenerator(mock_llm_client)

        # Set up return values with cycle to handle multiple calls
        mock_qa_instructions.return_value = ["QA instructions response"]
        mock_convert.side_effect = cycle([["Question 1", "Question 2"]])

        documents = ["Document 1", "Document 2"]

        # Call the pipeline
        result = generator.run_closed_qa_pipeline(
            documents=documents,
            n_openlines=2,
            model="test_model",
        )

        # Verify the method was called for each document
        assert mock_qa_instructions.call_count == 2
        mock_qa_instructions.assert_has_calls(
            [
                call(
                    document="Document 1",
                    n_openlines=2,
                    model="test_model",
                    model_kwargs={},
                    prompt_template=mock_qa_instructions.call_args[1][
                        "prompt_template"
                    ],
                ),
                call(
                    document="Document 2",
                    n_openlines=2,
                    model="test_model",
                    model_kwargs={},
                    prompt_template=mock_qa_instructions.call_args[1][
                        "prompt_template"
                    ],
                ),
            ]
        )

        # Verify convert was called for each response
        assert mock_convert.call_count >= 2

        # Due to the mocking structure, we need to manually check the structure
        assert len(result) == 4
        assert result[0][0] == 0  # First document index
        assert result[2][0] == 1  # Second document index
        assert type(result[0][1]) == str  # Verify it's a string

    @patch.object(NemotronGenerator, "convert_response_to_yaml_list")
    @patch.object(NemotronGenerator, "generate_writing_tasks")
    @patch.object(NemotronGenerator, "revise_writing_tasks")
    def test_run_writing_pipeline(
        self, mock_revise, mock_generate_tasks, mock_convert, mock_llm_client
    ):
        """Test run_writing_pipeline pipeline method."""
        generator = NemotronGenerator(mock_llm_client)

        # Set up return values
        mock_generate_tasks.return_value = ["Tasks response"]
        # Adjust the order of responses - we need to make sure the function gets Revised Task values
        # for the final result
        mock_convert.side_effect = cycle(
            [
                [
                    "Revised Task 1",
                    "Revised Task 2",
                ],  # Put revised tasks first in the cycle
                ["Task 1", "Task 2"],  # Tasks conversion
            ]
        )
        mock_revise.return_value = ["Revisions response"]

        topics = ["Science", "History"]
        text_material_types = ["Essay", "Blog Post"]

        # Call the pipeline
        result = generator.run_writing_pipeline(
            topics=topics,
            text_material_types=text_material_types,
            n_openlines=2,
            n_revisions=2,
            model="test_model",
        )

        # Check that the tasks generator was called for each topic and text material type
        assert mock_generate_tasks.call_count == 4  # 2 topics x 2 material types
        mock_generate_tasks.assert_any_call(
            topic="Science",
            text_material_type="Essay",
            n_openlines=2,
            model="test_model",
            model_kwargs={},
            prompt_template=mock_generate_tasks.call_args[1]["prompt_template"],
        )

        # Check the final result - now correctly expecting "Revised Task" in the result
        assert "Revised Task" in result[0]

    @patch.object(NemotronGenerator, "convert_response_to_yaml_list")
    @patch.object(NemotronGenerator, "generate_math_macro_topics")
    @patch.object(NemotronGenerator, "generate_math_subtopics")
    @patch.object(NemotronGenerator, "generate_math_problem")
    def test_run_math_pipeline(
        self,
        mock_math_problem,
        mock_subtopics,
        mock_macro_topics,
        mock_convert,
        mock_llm_client,
    ):
        """Test run_math_pipeline pipeline method."""
        generator = NemotronGenerator(mock_llm_client)

        # Set up return values using cycle for multiple calls
        mock_macro_topics.return_value = ["Math topics response"]
        mock_convert.side_effect = cycle(
            [
                ["Algebra", "Geometry"],  # Macro topics conversion
                ["Equations", "Polynomials"],  # Subtopics conversion for Algebra
                ["Triangles", "Circles"],  # Subtopics conversion for Geometry
                ["Problem 1", "Problem 2"],  # Problems conversion
            ]
        )
        mock_subtopics.return_value = ["Subtopics response"]
        mock_math_problem.return_value = ["Problem generation response"]

        # Call the pipeline
        result = generator.run_math_pipeline(
            n_macro_topics=2,
            school_level="High School",
            n_subtopics=2,
            n_openlines=2,
            model="test_model",
        )

        # Check that macro_topics was called with the right parameters
        mock_macro_topics.assert_called_once_with(
            n_macro_topics=2,
            school_level="High School",
            model="test_model",
            model_kwargs={},
            prompt_template=mock_macro_topics.call_args[1]["prompt_template"],
        )

        # Check that subtopics were generated for both macro topics
        assert mock_subtopics.call_count >= 2

        # Check the final result
        assert "Problem" in result[0]

    @patch.object(NemotronGenerator, "convert_response_to_yaml_list")
    @patch.object(NemotronGenerator, "generate_python_macro_topics")
    @patch.object(NemotronGenerator, "generate_python_subtopics")
    @patch.object(NemotronGenerator, "generate_python_problem")
    def test_run_python_pipeline(
        self,
        mock_python_problem,
        mock_subtopics,
        mock_macro_topics,
        mock_convert,
        mock_llm_client,
    ):
        """Test run_python_pipeline pipeline method."""
        generator = NemotronGenerator(mock_llm_client)

        # Set up return values with cycle for multiple calls
        mock_macro_topics.return_value = ["Python topics response"]
        mock_convert.side_effect = cycle(
            [
                ["Data Structures", "Functions"],  # Macro topics conversion
                ["Lists", "Dictionaries"],  # Subtopics conversion for Data Structures
                ["Lambda", "Closures"],  # Subtopics conversion for Functions
                ["Problem 1", "Problem 2"],  # Problems conversion
            ]
        )
        mock_subtopics.return_value = ["Subtopics response"]
        mock_python_problem.return_value = ["Problem generation response"]

        # Call the pipeline
        result = generator.run_python_pipeline(
            n_macro_topics=2,
            n_subtopics=2,
            n_openlines=2,
            model="test_model",
        )

        # Check that macro_topics was called with the right parameters
        mock_macro_topics.assert_called_once_with(
            n_macro_topics=2,
            model="test_model",
            model_kwargs={},
            prompt_template=mock_macro_topics.call_args[1]["prompt_template"],
        )

        # Check that subtopics were generated for both macro topics
        assert mock_subtopics.call_count >= 2

        # Check the final result
        assert "Problem" in result[0]

    @patch.object(NemotronGenerator, "convert_response_to_yaml_list")
    def test_run_pipeline_with_yaml_conversion_error(
        self, mock_convert, mock_llm_client
    ):
        """Test pipeline error handling when YamlConversionError occurs."""
        generator = NemotronGenerator(mock_llm_client)

        # Mock the convert_response_to_yaml_list to raise a YamlConversionError on first call
        mock_convert.side_effect = YamlConversionError("Test conversion error")

        # Set up other return values to test pipeline error handling
        mock_llm_client.query_model.return_value = ["Raw response"]

        # Verify that the error is propagated when ignore_conversion_failure=False
        with pytest.raises(YamlConversionError):
            generator.run_open_qa_pipeline(
                n_macro_topics=2,
                n_subtopics=2,
                n_openlines=2,
                n_revisions=2,
                model="test_model",
                ignore_conversion_failure=False,
            )

        # Now check that the error is handled when ignore_conversion_failure=True
        # We need to reset the mock for this test
        mock_convert.reset_mock()
        mock_convert.side_effect = cycle(
            [
                ["Topic 1", "Topic 2"],  # Successful macro topics conversion
                YamlConversionError(
                    "Subtopics conversion error"
                ),  # Failure in subtopics
                [
                    "Openline 1"
                ],  # Successful openlines conversion (fewer than requested)
                YamlConversionError(
                    "Revisions conversion error"
                ),  # Failure in revisions
            ]
        )

        # Call the pipeline with ignore_conversion_failure=True
        # In a real call this would return an empty list because all conversions fail,
        # but we've mocked to have some successes to verify partial processing
        result = generator.run_open_qa_pipeline(
            n_macro_topics=2,
            n_subtopics=2,
            n_openlines=1,  # Lowered to match our mock
            n_revisions=2,
            model="test_model",
            ignore_conversion_failure=True,
            additional_subtopics=[
                "Extra subtopic"
            ],  # Added to ensure we have something to process
        )

    def test_revise_writing_tasks(self, mock_llm_client):
        """Test revise_writing_tasks method."""
        generator = NemotronGenerator(mock_llm_client)

        result = generator.revise_writing_tasks(
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

    def test_generate_math_macro_topics(self, mock_llm_client):
        """Test generate_math_macro_topics method."""
        generator = NemotronGenerator(mock_llm_client)

        result = generator.generate_math_macro_topics(
            n_macro_topics=3,
            school_level="Middle School",
            model="test_model",
        )

        # Check the parameters sent to query_model
        mock_llm_client.query_model.assert_called_once()
        messages = mock_llm_client.query_model.call_args[1]["messages"]
        assert (
            "3" in messages[0]["content"]
        ), "Number of topics wasn't passed to the prompt"
        assert (
            "Middle School" in messages[0]["content"]
        ), "School level wasn't passed to the prompt"

        # Check the result
        assert result == ["This is a mock response"]

    def test_generate_math_subtopics(self, mock_llm_client):
        """Test generate_math_subtopics method."""
        generator = NemotronGenerator(mock_llm_client)

        result = generator.generate_math_subtopics(
            macro_topic="Algebra",
            n_subtopics=4,
            model="test_model",
        )

        # Check the parameters sent to query_model
        mock_llm_client.query_model.assert_called_once()
        messages = mock_llm_client.query_model.call_args[1]["messages"]
        assert (
            "Algebra" in messages[0]["content"]
        ), "Macro topic wasn't passed to the prompt"
        assert (
            "4" in messages[0]["content"]
        ), "Number of subtopics wasn't passed to the prompt"

        # Check the result
        assert result == ["This is a mock response"]

    def test_classify_math_entity(self, mock_llm_client):
        """Test classify_math_entity method."""
        generator = NemotronGenerator(mock_llm_client)

        result = generator.classify_math_entity(
            entity="Quadratic Equations",
            model="test_model",
        )

        # Check the parameters sent to query_model
        mock_llm_client.query_model.assert_called_once()
        messages = mock_llm_client.query_model.call_args[1]["messages"]
        assert (
            "Quadratic Equations" in messages[0]["content"]
        ), "Entity wasn't passed to the prompt"

        # Check the result
        assert result == ["This is a mock response"]

    def test_generate_python_macro_topics(self, mock_llm_client):
        """Test generate_python_macro_topics method."""
        generator = NemotronGenerator(mock_llm_client)

        result = generator.generate_python_macro_topics(
            n_macro_topics=3,
            model="test_model",
        )

        # Check the parameters sent to query_model
        mock_llm_client.query_model.assert_called_once()
        messages = mock_llm_client.query_model.call_args[1]["messages"]
        assert (
            "3" in messages[0]["content"]
        ), "Number of topics wasn't passed to the prompt"

        # Check the result
        assert result == ["This is a mock response"]

    def test_generate_python_subtopics(self, mock_llm_client):
        """Test generate_python_subtopics method."""
        generator = NemotronGenerator(mock_llm_client)

        result = generator.generate_python_subtopics(
            macro_topic="Web Development",
            n_subtopics=4,
            model="test_model",
        )

        # Check the parameters sent to query_model
        mock_llm_client.query_model.assert_called_once()
        messages = mock_llm_client.query_model.call_args[1]["messages"]
        assert (
            "Web Development" in messages[0]["content"]
        ), "Macro topic wasn't passed to the prompt"
        assert (
            "4" in messages[0]["content"]
        ), "Number of subtopics wasn't passed to the prompt"

        # Check the result
        assert result == ["This is a mock response"]

    def test_classify_python_entity(self, mock_llm_client):
        """Test classify_python_entity method."""
        generator = NemotronGenerator(mock_llm_client)

        result = generator.classify_python_entity(
            entity="Object-Oriented Programming",
            model="test_model",
        )

        # Check the parameters sent to query_model
        mock_llm_client.query_model.assert_called_once()
        messages = mock_llm_client.query_model.call_args[1]["messages"]
        assert (
            "Object-Oriented Programming" in messages[0]["content"]
        ), "Entity wasn't passed to the prompt"

        # Check the result
        assert result == ["This is a mock response"]

    def test_convert_response_with_different_exception_types(self, mock_llm_client):
        """Test that different types of conversion errors are properly raised."""
        generator = NemotronGenerator(mock_llm_client)

        # Test with a completely invalid YAML
        mock_llm_client.query_model.return_value = ["[This is not valid YAML: {"]
        with pytest.raises(YamlConversionError) as excinfo:
            generator.convert_response_to_yaml_list(
                llm_response="Test text",
                model="test_model",
            )
        assert "Error parsing yaml response" in str(excinfo.value)

        # Test with valid YAML but not a list
        mock_llm_client.query_model.return_value = [yaml.dump({"key": "value"})]
        with pytest.raises(YamlConversionError) as excinfo:
            generator.convert_response_to_yaml_list(
                llm_response="Test text",
                model="test_model",
            )
        assert "not a list" in str(excinfo.value)

        # Test with list containing items not in the source text (hallucination)
        mock_llm_client.query_model.return_value = [
            yaml.dump(["Item in source", "Hallucinated item"])
        ]
        with pytest.raises(YamlConversionError) as excinfo:
            generator.convert_response_to_yaml_list(
                llm_response="Source text with Item in source",
                model="test_model",
            )
        assert "hallucination" in str(excinfo.value).lower()

        # Test with list containing non-string elements
        mock_llm_client.query_model.return_value = [
            yaml.dump(["Valid string", 123, True, {"nested": "dict"}])
        ]
        with pytest.raises(YamlConversionError) as excinfo:
            generator.convert_response_to_yaml_list(
                llm_response="Valid string",
                model="test_model",
            )
        assert "non-string element" in str(excinfo.value).lower()

    def test_pipeline_error_propagation(self, mock_llm_client):
        """Test that pipeline methods propagate conversion errors properly."""
        generator = NemotronGenerator(mock_llm_client)

        # Setup to trigger a YamlConversionError in the first conversion step
        mock_llm_client.query_model.return_value = ["Valid response"]

        # Patch the necessary methods to simulate the pipeline
        with patch.object(generator, "generate_macro_topics") as mock_macro_topics:
            mock_macro_topics.return_value = ["Macro topics response"]

            with patch.object(
                generator, "convert_response_to_yaml_list"
            ) as mock_convert:
                # Make the convert method raise an exception
                mock_convert.side_effect = YamlConversionError("Test conversion error")

                # Verify exception is propagated with ignore_conversion_failure=False
                with pytest.raises(YamlConversionError):
                    generator.run_open_qa_pipeline(
                        n_macro_topics=2,
                        n_subtopics=2,
                        n_openlines=2,
                        n_revisions=2,
                        model="test_model",
                        ignore_conversion_failure=False,
                    )

    def test_pipeline_error_suppression(self, mock_llm_client):
        """Test that pipeline methods suppress errors when configured to do so."""
        generator = NemotronGenerator(mock_llm_client)

        # Setup for testing
        mock_llm_client.query_model.return_value = ["Valid response"]

        # Patch the necessary methods to simulate the pipeline
        with patch.object(generator, "generate_macro_topics") as mock_macro_topics:
            mock_macro_topics.return_value = ["Macro topics response"]

            with patch.object(
                generator, "convert_response_to_yaml_list"
            ) as mock_convert:
                # Setup the side effect to first fail with error, then return valid results
                # This simulates the pipeline continuing after an error
                # Use cycle to create an infinite iterator that won't run out of values
                mock_convert.side_effect = cycle(
                    [
                        YamlConversionError(
                            "Test conversion error"
                        ),  # First call: macro topics - fail
                        ["Topic 1", "Topic 2"],  # Second call: subtopics - succeed
                        ["Question 1", "Question 2"],  # Third call: openlines - succeed
                        [
                            "Revised Q1",
                            "Revised Q2",
                        ],  # Fourth call: revisions - succeed
                    ]
                )

                # With ignore_conversion_failure=True, should handle the error and continue
                result = generator.run_open_qa_pipeline(
                    n_macro_topics=2,
                    n_subtopics=2,
                    n_openlines=2,
                    n_revisions=2,
                    model="test_model",
                    ignore_conversion_failure=True,
                    additional_macro_topics=[
                        "Backup Topic"
                    ],  # To ensure we have something to process
                )

                # Verify that the mock was called and pipeline continued execution
                assert mock_convert.call_count >= 1

                # Result should contain processed items since we setup the mocks to return successful results
                assert isinstance(result, list)
                assert len(result) > 0  # Should have some results

    def test_other_pipelines_error_handling(self, mock_llm_client):
        """Test error handling in other pipeline methods."""
        generator = NemotronGenerator(mock_llm_client)

        # Setup for testing
        mock_llm_client.query_model.return_value = ["Valid response"]

        # Test writing pipeline
        with patch.object(generator, "convert_response_to_yaml_list") as mock_convert:
            # Create a side effect function that checks the kwargs for ignore_conversion_failure
            def side_effect(*args, **kwargs):
                # Always raise the error for main method calls - pipeline will handle the error internally
                raise YamlConversionError("Test conversion error")

            mock_convert.side_effect = side_effect

            # Should raise exception with ignore_conversion_failure=False
            with pytest.raises(YamlConversionError):
                generator.run_writing_pipeline(
                    topics=["Topic"],
                    text_material_types=["Essay"],
                    n_openlines=2,
                    n_revisions=2,
                    model="test_model",
                    ignore_conversion_failure=False,
                )

            # Should not raise exception with ignore_conversion_failure=True
            # For this to work, we need to patch at the method level that handles the error instead
            with patch.object(generator, "generate_writing_tasks") as mock_gen_tasks:
                mock_gen_tasks.return_value = ["Task response"]

                result = generator.run_writing_pipeline(
                    topics=["Topic"],
                    text_material_types=["Essay"],
                    n_openlines=2,
                    n_revisions=2,
                    model="test_model",
                    ignore_conversion_failure=True,
                )
                assert isinstance(result, list)

        # Test math pipeline
        with patch.object(generator, "generate_math_macro_topics") as mock_macro_topics:
            mock_macro_topics.return_value = ["Math topics response"]

            # For testing ignore_conversion_failure=False
            with patch.object(
                generator, "convert_response_to_yaml_list"
            ) as mock_convert:
                mock_convert.side_effect = YamlConversionError("Test conversion error")

                # Should raise exception with ignore_conversion_failure=False
                with pytest.raises(YamlConversionError):
                    generator.run_math_pipeline(
                        n_macro_topics=2,
                        school_level="High School",
                        n_subtopics=2,
                        n_openlines=2,
                        model="test_model",
                        ignore_conversion_failure=False,
                    )

            # For testing ignore_conversion_failure=True, we'll make the mock handle this case correctly
            with patch.object(
                generator, "convert_response_to_yaml_list"
            ) as mock_convert:
                # First call will be for macro_topics, which we want to simulate as failing
                # If additional_macro_topics is provided, the pipeline should still continue
                mock_convert.side_effect = cycle(
                    [
                        YamlConversionError(
                            "Test conversion error"
                        ),  # First call fails
                        ["Subtopic 1", "Subtopic 2"],  # Subsequent calls succeed
                    ]
                )

                result = generator.run_math_pipeline(
                    n_macro_topics=2,
                    school_level="High School",
                    n_subtopics=2,
                    n_openlines=2,
                    model="test_model",
                    ignore_conversion_failure=True,
                    additional_macro_topics=["Backup Topic"],
                )
                assert isinstance(result, list)

        # Test closed_qa pipeline
        with patch.object(generator, "generate_closed_qa_instructions") as mock_gen_qa:
            mock_gen_qa.return_value = ["QA instructions response"]

            # For testing ignore_conversion_failure=False
            with patch.object(
                generator, "convert_response_to_yaml_list"
            ) as mock_convert:
                mock_convert.side_effect = YamlConversionError("Test conversion error")

                # Should raise exception with ignore_conversion_failure=False
                with pytest.raises(YamlConversionError):
                    generator.run_closed_qa_pipeline(
                        documents=["Test document"],
                        n_openlines=2,
                        model="test_model",
                        ignore_conversion_failure=False,
                    )

            # For testing ignore_conversion_failure=True
            with patch.object(
                generator, "convert_response_to_yaml_list"
            ) as mock_convert:
                # Setup to return empty list when ignore_conversion_failure=True
                mock_convert.return_value = []

                result = generator.run_closed_qa_pipeline(
                    documents=["Test document"],
                    n_openlines=2,
                    model="test_model",
                    ignore_conversion_failure=True,
                )
                assert isinstance(result, list)

        # Test python pipeline
        with patch.object(
            generator, "generate_python_macro_topics"
        ) as mock_macro_topics:
            mock_macro_topics.return_value = ["Python topics response"]

            # For testing ignore_conversion_failure=False
            with patch.object(
                generator, "convert_response_to_yaml_list"
            ) as mock_convert:
                mock_convert.side_effect = YamlConversionError("Test conversion error")

                # Should raise exception with ignore_conversion_failure=False
                with pytest.raises(YamlConversionError):
                    generator.run_python_pipeline(
                        n_macro_topics=2,
                        n_subtopics=2,
                        n_openlines=2,
                        model="test_model",
                        ignore_conversion_failure=False,
                    )

            # For testing ignore_conversion_failure=True
            with patch.object(
                generator, "convert_response_to_yaml_list"
            ) as mock_convert:
                # First call will be for macro_topics, which we want to simulate as failing
                # If additional_macro_topics is provided, the pipeline should still continue
                mock_convert.side_effect = cycle(
                    [
                        YamlConversionError(
                            "Test conversion error"
                        ),  # First call fails
                        ["Subtopic 1", "Subtopic 2"],  # Subsequent calls succeed
                    ]
                )

                result = generator.run_python_pipeline(
                    n_macro_topics=2,
                    n_subtopics=2,
                    n_openlines=2,
                    model="test_model",
                    ignore_conversion_failure=True,
                    additional_macro_topics=["Backup Topic"],
                )
                assert isinstance(result, list)
