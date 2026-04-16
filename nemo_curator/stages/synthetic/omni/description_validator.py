"""Validates image descriptions using a verification model."""

from dataclasses import dataclass
from typing import Any, Generator, Iterable

from nemo_curator.stages.resources import Resources

from nemo_curator.models.omni.base import InferenceConfig, NVInferenceModelConfig
from nemo_curator.models.omni.gemini import Gemini3Pro
from nemo_curator.stages.synthetic.omni.base import ModelProcessingStage
from nemo_curator.tasks.image import SingleDataTask
from nemo_curator.stages.synthetic.omni.description import ImageCaptioningData
from nemo_curator.stages.synthetic.omni.utils.json_streamer import process_json
from nemo_curator.tasks.image import SingleDataTask


@dataclass
class DescriptionVerificationResult:
    """Verification result for a single description."""

    accurate: bool | None = None
    complete: bool | None = None

    @property
    def is_valid(self) -> bool:
        """Check if the description passed validation."""
        return self.accurate is True and self.complete is True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DescriptionVerificationResult":
        """Create from dictionary."""
        return cls(
            accurate=data.get("accurate"),
            complete=data.get("complete"),
        )


@dataclass(kw_only=True)
class DescriptionValidatedData(ImageCaptioningData):
    """Task data for validated descriptions."""

    verification_result: DescriptionVerificationResult | None = None
    verification_prompt: str | None = None
    verification_response: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DescriptionValidatedData":
        """Create from dictionary (reads describe output or validated jsonl)."""
        base = ImageCaptioningData.from_dict(data)
        vr = data.get("verification_result")
        verification_result = (
            DescriptionVerificationResult.from_dict(vr) if isinstance(vr, dict) else None
        )
        return cls(
            image_path=base.image_path,
            image_id=base.image_id,
            is_valid=base.is_valid,
            error=base.error,
            description_prompt=base.description_prompt,
            description=base.description,
            verification_result=verification_result,
            verification_prompt=data.get("verification_prompt"),
            verification_response=data.get("verification_response"),
        )


def _to_validated_data(data: ImageCaptioningData) -> DescriptionValidatedData:
    """Convert to DescriptionValidatedData so downstream stages and serialization see all fields."""
    return DescriptionValidatedData(
        image_path=data.image_path,
        image_id=data.image_id,
        is_valid=data.is_valid,
        error=data.error,
        description_prompt=data.description_prompt,
        description=data.description,
        verification_result=getattr(data, "verification_result", None),
        verification_prompt=getattr(data, "verification_prompt", None),
        verification_response=getattr(data, "verification_response", None),
    )


VERIFICATION_DESCRIPTION_PROMPT = """\
You are a verification assistant. Given an image and a textual description of it, verify the description.

Description to verify:
```
{description}
```

Respond in JSON format:
{{
  "accurate": true|false,
  "complete": true|false,
}}

`accurate`: Is the description factually correct and grounded in the image (no hallucinations)?
`complete`: Does the description cover the main visual elements, layout, and any visible text?
Only output valid JSON. No markdown code fences."""


class DescriptionValidatorStage(ModelProcessingStage[DescriptionValidatedData]):
    """Validate descriptions using a VLM for verification."""

    name = "description_validator"
    resources = Resources(cpus=1.0)
    # Match or be smaller than upstream DescriptionStage.batch_size (16) so streaming can pass
    # work to this stage as soon as each description batch completes, improving GPU/API overlap.
    batch_size = 16

    def __init__(
        self,
        model_id: str = "gcp/google/gemini-3-flash-preview",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=Gemini3Pro(
                model_config=NVInferenceModelConfig(max_tokens=8 * 1024),
                model_id=model_id,
            ),
            inference_config=InferenceConfig(
                temperature=1.0,
                top_p=0.95,
                do_sample=True,
            ),
            batch_size=self.batch_size,
            **kwargs,
        )

    def build_prompt(self, task: SingleDataTask[DescriptionValidatedData]) -> str:
        desc = task.data.description or ""
        prompt = VERIFICATION_DESCRIPTION_PROMPT.format(description=desc)
        task.data.verification_prompt = prompt
        return prompt

    def handle_response(
        self,
        task: SingleDataTask[DescriptionValidatedData],
        response: str,
    ) -> SingleDataTask[DescriptionValidatedData]:
        try:
            obj = process_json(response, bounds="{}")
        except Exception as e:
            task.data.error = f"Error parsing verification JSON: {e}"
            task.data.is_valid = False
            return task
        task.data.verification_response = response
        task.data.verification_result = DescriptionVerificationResult(
            accurate=obj.get("accurate"),
            complete=obj.get("complete"),
        )
        return task

    def process_batch(
        self, tasks: list[SingleDataTask[DescriptionValidatedData]]
    ) -> list[SingleDataTask[DescriptionValidatedData]]:
        tasks = super().process_batch(tasks)
        for task in tasks:
            if not isinstance(task.data, DescriptionValidatedData):
                task.data = _to_validated_data(task.data)
        return tasks
