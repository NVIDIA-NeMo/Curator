"""Generates conversations from description data."""

from dataclasses import dataclass
from pathlib import Path

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources

from nemo_curator.stages.synthetic.omni.utils.conversation import ConversationSample, ImageMedia, Message
from nemo_curator.stages.synthetic.omni.description_validator import DescriptionValidatedData
from nemo_curator.tasks.image import SingleDataTask


@dataclass(kw_only=True)
class DescriptionConversationData(DescriptionValidatedData):
    """Task data for description with conversation output."""

    conversation: ConversationSample | None = None


class DescriptionOutputStage(
    ProcessingStage[
        SingleDataTask[DescriptionValidatedData],
        SingleDataTask[DescriptionConversationData],
    ]
):
    """Generate conversation from description data."""

    name = "description_output"
    resources = Resources(cpus=1.0)
    batch_size = 1

    def process(
        self, task: SingleDataTask[DescriptionValidatedData]
    ) -> SingleDataTask[DescriptionConversationData]:
        """Convert description into a conversation sample.

        Skips tasks that already failed in previous stages (is_valid=False).

        Args:
            task: Task containing description data.

        Returns:
            Task with conversation data added.
        """
        output_data = DescriptionConversationData(
            image_path=task.data.image_path,
            image_id=task.data.image_id,
            is_valid=task.data.is_valid,
            error=task.data.error,
            verification_result=task.data.verification_result,
            verification_prompt=task.data.verification_prompt,
            verification_response=task.data.verification_response,
            description_prompt=task.data.description_prompt,
            description=task.data.description,
        )
        task.data = output_data

        # Skip tasks that already failed or have no description
        if not task.data.is_valid or not task.data.description:
            return task

        vr = task.data.verification_result
        if vr is None or not vr.accurate or not vr.complete:
            return task

        image_name = Path(task.data.image_path).name
        prompt = task.data.description_prompt

        task.data.conversation = ConversationSample(
            conversation=[
                Message(
                    sender="user",
                    fragments=[ImageMedia(value=image_name), prompt],
                ),
                Message(
                    sender="assistant",
                    fragments=[task.data.description],
                ),
            ]
        )
        return task
