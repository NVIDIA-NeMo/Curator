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

"""Composite stage for PII redaction in documents."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from ray_curator.stages.base import CompositeStage, ProcessingStage
from ray_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    from ray_curator.stages.pii.ner_pii.config import PiiConfig


@dataclass
class PiiRedactionStage(CompositeStage[DocumentBatch, DocumentBatch]):
    """Composite stage for PII redaction in documents.

    This high-level stage decomposes into:
    1. PiiDetectionStage - Analyzes (detects) PII entities using NER models
    2. PiiAnonymizationStage - Applies anonymization strategies to detected entities

    This separation allows for:
    - Different resource allocation (GPU for analyzer/detection, CPU for anonymization)
    - Intermediate entity analysis if needed
    - Flexible anonymization strategies per entity type

    Example using config:
        >>> from ray_curator.stages.pii.ner_pii.config import PiiConfig
        >>> config = PiiConfig.from_yaml("pii_config.yaml")
        >>> pii_stage = PiiRedactionStage(config=config)
        >>> pipeline.add_stage(pii_stage)

    Example using parameters:
        >>> pii_stage = PiiRedactionStage(
        ...     language="en",
        ...     supported_entities=["PERSON", "EMAIL", "PHONE_NUMBER"],
        ...     anonymize_action="replace",
        ...     device="gpu"
        ... )
        >>> pipeline.add_stage(pii_stage)

    Args:
        config: Optional PiiConfig object containing all configuration. If provided,
                overrides all other parameters.
        language: Language code for NER model (default: "en")
        supported_entities: List of entity types to detect. If None, uses all supported entities.
        device: Device to use for NER model ("gpu" or "cpu")
        batch_size: Batch size for processing documents
        text_column: Name of the column containing text to process
        anonymize_action: Anonymization strategy ("replace", "redact", "hash", "mask", "custom")
        anonymize_kwargs: Additional arguments for anonymization (e.g., new_value for replace)
        entity_operators: Entity-specific anonymization operators
        max_doc_size: Maximum document size to process
    """

    # Optional config object (if provided, overrides other parameters)
    config: Optional["PiiConfig"] = None

    # Detection configuration (used if config is not provided)
    language: str = "en"
    supported_entities: list[str] | None = None
    device: str = "gpu"  # "gpu" or "cpu"
    batch_size: int = 2000

    # Data configuration
    text_column: str = "text"

    # Anonymization configuration (used if config is not provided)
    anonymize_action: str = "replace"  # "replace", "redact", "hash", "mask", "custom"
    anonymize_kwargs: dict[str, Any] = field(default_factory=dict)

    # Advanced configuration
    entity_operators: dict[str, dict[str, Any]] | None = None
    max_doc_size: int = 2000000

    def __post_init__(self):
        """Initialize internal config from parameters if not provided."""
        if self.config is None:
            # Create config from individual parameters
            from ray_curator.stages.pii.ner_pii.config import PiiAnalyzerConfig, PiiAnonymizationConfig, PiiConfig

            # Create analyzer config
            analyzer_config = PiiAnalyzerConfig(
                models=[{"language": self.language, "model_name": f"{self.language}_core_web_lg"}],
                supported_entities=self.supported_entities,
                device=self.device,
                batch_size=self.batch_size,
                max_doc_size=self.max_doc_size,
            )

            # Create anonymization config
            anon_config = PiiAnonymizationConfig(
                default_action=self.anonymize_action,
                default_params=self.anonymize_kwargs,
                entity_config=self.entity_operators or {},
            )

            # Create complete config
            self._internal_config = PiiConfig(
                analyzer=analyzer_config,
                anonymization=anon_config,
                text_column=self.text_column,
            )
        else:
            # Use provided config
            self._internal_config = self.config
            # Update text_column if it's different in the config
            if hasattr(self.config, "text_column"):
                self.text_column = self.config.text_column

    @property
    def name(self) -> str:
        return "pii_redaction"

    @classmethod
    def from_config(cls, config: "PiiConfig") -> "PiiRedactionStage":
        """Create a PiiRedactionStage from a PiiConfig object."""
        return cls(config=config)

    def decompose(self) -> list[ProcessingStage]:
        """Decompose into analyzer (detection) and anonymization stages.

        Returns:
            List containing PiiDetectionStage and PiiAnonymizationStage
        """
        # Import here to avoid circular imports
        from ray_curator.stages.pii.ner_pii.pii_anonymization import PiiAnonymizationStage
        from ray_curator.stages.pii.ner_pii.pii_detection import PiiDetectionStage

        return [
            # First stage: Analyze (detect) PII entities
            PiiDetectionStage(
                config=self._internal_config.analyzer,
                text_column=self.text_column,
            ),
            # Second stage: Anonymize detected entities
            PiiAnonymizationStage(
                config=self._internal_config.anonymization,
                text_column=self.text_column,
            ),
        ]

    def get_description(self) -> str:
        """Get a human-readable description of this composite stage."""
        parts = [f"PII Redaction for {self._internal_config.analyzer.get_languages()[0]} text"]

        if self._internal_config.analyzer.supported_entities:
            parts.append(f"detecting {', '.join(self._internal_config.analyzer.supported_entities)}")
        else:
            parts.append("detecting all supported entity types")

        parts.append(f"using {self._internal_config.anonymization.default_action} anonymization")

        if self._internal_config.analyzer.device == "gpu":
            parts.append("with GPU acceleration")

        return ", ".join(parts)
