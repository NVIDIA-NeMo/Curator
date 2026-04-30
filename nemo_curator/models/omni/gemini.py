"""Gemini model implementation using NVIDIA Inference API."""

from nemo_curator.models.omni.base import NVInferenceModel, NVInferenceModelConfig

DEFAULT_MODEL_ID = "gcp/google/gemini-3-pro"


class Gemini3Pro(NVInferenceModel):
    """Gemini model using NVIDIA Inference API backend."""

    def __init__(
        self,
        model_config: NVInferenceModelConfig,
        model_id: str = DEFAULT_MODEL_ID,
    ) -> None:
        """Initialize Gemini with NVIDIA Inference API.

        Args:
            model_config: NVIDIA Inference API configuration.
            model_id: Model identifier for NVIDIA Inference API.
        """
        super().__init__(model_id, model_config)
