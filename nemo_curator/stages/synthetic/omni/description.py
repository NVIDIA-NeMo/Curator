"""Description generation stage using Qwen3-VL."""

from dataclasses import dataclass
import random
from typing import Any

from nemo_curator.stages.resources import Resources

from nemo_curator.models.omni.base import (
    InferenceConfig,
    VLLMModelConfig,
)
from nemo_curator.models.omni.qwen3_vl import Qwen3VL
from nemo_curator.stages.synthetic.omni.base import ModelProcessingStage
from nemo_curator.tasks.image import ImageTaskData, SingleDataTask

# Target prompts for description generation.
TARGET_PROMPTS = [
    # Direct & Formal Instructions
    "Provide a detailed, grounded description of the image, noting objects, positions, and visible text.",
    "Write a comprehensive description of the image content, focusing on spatial layout and text.",
    "Describe the image in depth, including the arrangement of objects and any words visible.",
    "Generate a full textual representation of the image scene, covering all visual elements.",
    "Analyze the image and provide a detailed account of objects, their locations, and text.",
    "Create a grounded description of the image, ensuring all text and major objects are mentioned.",
    "Detail the contents of the image, specifically noting object placement and textual information.",
    "Provide a thorough explanation of the image's visual composition and specific details.",
    "Produce a descriptive summary of the image, highlighting spatial relationships and text.",
    "Construct a detailed narrative of the scene presented in the image.",
    "Examine the image and write a description that includes objects, positions, and text.",
    "Give a precise description of the image, focusing on grounding elements and readability.",
    "Report on the visual data in the image, including object identification and text extraction.",
    "Draft a complete description of the image scene, omitting no major visual details.",
    "Supply a detailed caption for the image that covers composition, objects, and text.",
    # Concise & Imperative
    "Describe the image in detail.",
    "What is in the image? Include text and object positions.",
    "Image description: detailed, grounded, with text.",
    "Analyze the image and write a detailed description.",
    "List and describe the contents of the image.",
    "Describe the scene, objects, and text in the image.",
    "Detail everything visible in the image.",
    "Write a grounded description of the image.",
    "Explain the image composition and content.",
    "Describe the image visuals, including text.",
    "Give me a detailed breakdown of this image.",
    "Describe objects, positions, and text in the image.",
    "Full image description required.",
    "Interpret the image scene and describe it fully.",
    "State what is visible in the image in detail.",
    # Conversational & Natural Language
    "Tell me exactly what you see in this image, including where things are.",
    "Can you describe the image in detail for me? Mention any text you see.",
    "Look at the image and describe everything that's going on.",
    "I need a detailed breakdown of what's in this image.",
    "Please describe the image, paying attention to the layout and any words.",
    "What does this image show? Be specific about objects and text.",
    "Explain the image to me like I can't see it, including positions and text.",
    "Take a look at the image and write a detailed description of it.",
    "Describe the image thoroughly, including who or what is where.",
    "Give me a full picture of what this image contains.",
    "Write down everything you notice about the image's composition and content.",
    "How would you describe this image in detail? Include text if there is any.",
    "Please provide a grounded text description of the image.",
    "Analyze the image and tell me about the objects and their arrangement.",
    "Describe the image scene, making sure to capture the text and layout.",
    # Strict Formatting (Focus on "No Extra Text")
    "Describe the image. Output only the description.",
    "Generate a description of the image. Do not include conversational text.",
    "Write a detailed description of the image. Just the description, nothing else.",
    "Provide the image description only, covering objects, positions, and text.",
    "Task: Describe image. Constraint: No extra text.",
    "Strictly describe the image content. No intro or outro.",
    "Return a description of the image. Keep it purely descriptive.",
    "Analyze the image and output the description text only.",
    "Give a direct description of the image without meta-commentary.",
    "Describe the image content, specifically objects and text, with no filler.",
    "Just write the description of the image.",
    "Provide a raw text description of the image scene.",
    "Description only: Analyze the image's objects and text.",
    "Generate text describing the image. No pleasantries.",
    "Simply describe the image in detail.",
    # Focused on Spatial/Grounding
    "Describe the spatial layout of objects in the image.",
    "Locate and describe all major elements in the image.",
    "Provide a description of the image focusing on where objects are placed.",
    "Detail the image composition, noting foreground and background elements.",
    "Describe the relative positions of objects in the image.",
    "Analyze the scene layout of the image and describe it.",
    "Map out the image in text, describing objects and their locations.",
    "Describe the image with a focus on spatial grounding.",
    "Explain how objects are arranged in the image.",
    "Write a description that highlights the positioning of items in the image.",
    # Focused on Text & Details
    "Transcribe any text in the image and describe the surrounding objects.",
    "Describe the image, paying special attention to any visible text.",
    "Identify all text and objects in the image and describe them.",
    "Read the text in the image and describe the context.",
    "Detail the image content, ensuring all text is captured in the description.",
    "Describe the visual and textual elements of the image.",
    "Focus on text and object details in your description of the image.",
    "Provide a description that integrates the visible text with the scene.",
    "What text and objects appear in the image? Describe them.",
    "Capture all visual details and text from the image in a description.",
    # "Act As" / Role-Based
    "Act as a captioning tool: Describe the image in detail.",
    "As an AI observer, describe the image content grounded in reality.",
    "Simulate a screen reader: Describe the image layout and text.",
    "Act as a visual analyzer and report on the image contents.",
    "Perform a detailed inspection of the image and write the results.",
    "As a vision model, generate a grounded description of the image.",
    "Provide a technical description of the image scene.",
    "Assume the role of a detail-oriented observer and describe the image.",
    "Generate a training caption for this image, including all details.",
    "Write a metadata description for the image.",
    # Abstract/Mixed Variations
    "Breakdown the visual narrative of the image.",
    "Elucidate the contents of the provided image.",
    "Convert the visual information in the image into a detailed text description.",
    "Provide a textual representation of the image's reality.",
    "Deconstruct the image into a descriptive paragraph.",
    "Verbalize the scene depicted in the image.",
    "Synthesize a comprehensive description of the image.",
    "Formulate a detailed account of the image's visual features.",
    "Translate the image scene into a grounded text description.",
    "Render the image content into words, focusing on precision and detail.",
]

DESCRIPTION_PROMPT = """\
Generate a detailed, grounded description of this image.

Include information about objects, their positions, any text visible,
and the overall scene composition. Just refer to the image as "image",
and write down the description, no extra text.
"""

DESCRIPTION_SUFFIX = """\


Just refer to the image as "image", and write down the description, no extra text.
Put the description in <description> tags:
<description>
[description of the image].
</description>
"""

@dataclass(kw_only=True)
class ImageCaptioningData(ImageTaskData):
    """Task data for image description/captioning."""

    description_prompt: str | None = None
    description: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImageCaptioningData":
        """Create from dictionary (for JsonlPipelineOutputReaderStage)."""
        from pathlib import Path

        return cls(
            image_path=Path(data["image_path"]) if data.get("image_path") else None,
            image_id=data.get("image_id"),
            is_valid=data.get("is_valid", True),
            error=data.get("error"),
            description_prompt=data.get("description_prompt"),
            description=data.get("description"),
        )


class DescriptionStage(ModelProcessingStage[ImageCaptioningData]):
    """Generate dense description."""

    name = "description_vllm"
    resources = Resources(cpus=0.0, gpus=1)
    batch_size = 16

    def __init__(
        self,
        num_workers: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize vLLM description stage.

        Args:
            num_workers: If set, number of workers for this stage (e.g. 8 for 8 GPUs).
                If None, Cosmos-Xenna autoscaler decides (often starts with fewer workers).
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(
            model=Qwen3VL(model_config=VLLMModelConfig(
                gpu_memory_utilization=self._get_gpu_memory_utilization(),
                tensor_parallel_size=self._get_tensor_parallel_size(),
                max_tokens=8 * 1024,
            )),
            inference_config=InferenceConfig(
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            ),
            batch_size=self.batch_size,
            **kwargs,
        )
        self.num_workers = num_workers

    def xenna_stage_spec(self) -> dict[str, Any]:
        spec: dict[str, Any] = {}
        if self.num_workers is not None:
            spec["num_workers"] = self.num_workers
        return spec

    def build_prompt(self, task: SingleDataTask[ImageCaptioningData]) -> str:
        """Build prompt incorporating grounding and OCR context.

        Args:
            task: The task to process

        Returns:
            Prompt string for description generation.
        """
        task.data.description_prompt = random.choice(TARGET_PROMPTS)
        return task.data.description_prompt + DESCRIPTION_SUFFIX

    def handle_response(self, task: SingleDataTask[ImageCaptioningData], response: str) -> SingleDataTask[ImageCaptioningData]:
        """Store the generated description in the task.

        Args:
            task: The task to update.
            response: The model's generated description.
        """
        try:
            assert "<description>" in response and "</description>" in response, "Description must be in <description> tags"
            start = response.index("<description>") + len("<description>")
            end = response.rindex("</description>")
            task.data.description = response[start:end].strip()
        except Exception:
            import traceback
            traceback.print_exc()
        return task
