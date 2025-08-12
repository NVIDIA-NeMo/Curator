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

from nemo_curator.services.conversation_formatter import ConversationFormatter


class Mixtral8x7BFormatter(ConversationFormatter):
    PROMPT_PREFIX = "<s> [INST] \n"

    @staticmethod
    def format_conversation(conv: list[dict]) -> str:
        """
        Formats a converstation between a user and assistant in the Mixtral-8x7B format
        described here: https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1
        Args:
            conv: A conversation between a user and assistant
        Returns:
            A conversation formatted as text
        """
        prompt = Mixtral8x7BFormatter.PROMPT_PREFIX

        for i, turn in enumerate(conv):
            user_turn = i % 2 == 0

            if user_turn:
                if turn["role"] != "user":
                    msg = f"Conversation turn {i} is not 'user'. All even number turns should be."
                    raise ValueError(msg)
                prompt += turn["content"] + " [/INST]"
            else:
                if turn["role"] != "assistant":
                    msg = f"Conversation turn {i} is not 'assistant'. All odd number turns should be."
                    raise ValueError(msg)
                prompt += turn["content"] + "</s>[INST] "

        return prompt
