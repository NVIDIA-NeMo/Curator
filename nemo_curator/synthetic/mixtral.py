from typing import List

from nemo_curator.services.conversation_formatter import ConversationFormatter


class Mixtral8x7BFormatter(ConversationFormatter):

    PROMPT_PREFIX = "<s> [INST] \n"

    @staticmethod
    def format_conversation(conv: List[dict]) -> str:
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
                    raise ValueError(
                        f"Conversation turn {i} is not 'user'. All even number turns should be."
                    )
                prompt += turn["content"] + " [/INST]"
            else:
                if turn["role"] != "assistant":
                    raise ValueError(
                        f"Conversation turn {i} is not 'assistant'. All odd number turns should be."
                    )
                prompt += turn["content"] + "</s>[INST] "

        return prompt
