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

"""Simplified configuration for PII analyzer (detection) and anonymization."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ray_curator.stages.pii.ner_pii.constants import SUPPORTED_ENTITIES


@dataclass
class PiiAnalyzerConfig:
    """Configuration for PII analyzer (detection).

    Focuses on the most commonly changed settings:
    - Language models for multi-language support
    - Entity types to detect
    - Processing device (GPU/CPU)
    """

    # SpaCy models for different languages
    # Format: [{"language": "en", "model_name": "en_core_web_lg"}, ...] # noqa: ERA001
    models: list[dict[str, str]] = field(default_factory=lambda: [{"language": "en", "model_name": "en_core_web_lg"}])

    # Entity types to detect (None = all supported entities)
    supported_entities: list[str] | None = None

    # Processing settings
    device: str = "gpu"  # "gpu" or "cpu"
    batch_size: int = 2000
    max_doc_size: int = 2000000

    def __post_init__(self):
        """Set defaults and validate models."""
        if self.supported_entities is None:
            self.supported_entities = SUPPORTED_ENTITIES.copy()

        # Validate model format
        for model in self.models:
            if not isinstance(model, dict) or "language" not in model or "model_name" not in model:
                exception = "Each model must be a dict with 'language' and 'model_name' keys"
                raise ValueError(exception)

    def get_languages(self) -> list[str]:
        """Get list of supported language codes."""
        return [model["language"] for model in self.models]


@dataclass
class PiiAnonymizationConfig:
    """Configuration for PII anonymization.

    Focuses on the most commonly changed settings:
    - Default anonymization action
    - Entity-specific anonymization rules
    """

    # Default action for all entities
    default_action: str = "replace"  # "replace", "redact", "hash", "mask"
    default_params: dict[str, Any] = field(default_factory=lambda: {"new_value": "<REDACTED>"})

    # Entity-specific actions (overrides default)
    entity_config: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get_action_for_entity(self, entity: str) -> tuple[str, dict[str, Any]]:
        """Get anonymization action and params for a specific entity."""
        if entity in self.entity_config:
            config = self.entity_config[entity]
            return config.get("action", self.default_action), config.get("params", {})
        return self.default_action, self.default_params


@dataclass
class PiiConfig:
    """Complete PII configuration combining analyzer (detection) and anonymization."""

    analyzer: PiiAnalyzerConfig = field(default_factory=PiiAnalyzerConfig)
    anonymization: PiiAnonymizationConfig = field(default_factory=PiiAnonymizationConfig)
    text_column: str = "text"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PiiConfig":
        """Load configuration from a simplified YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        # Parse models
        models = data.get("models", [{"language": "en", "model_name": "en_core_web_lg"}])

        # Parse analyzer (detection) config
        analyzer = PiiAnalyzerConfig(
            models=models,
            supported_entities=data.get("supported_entities"),
            device=data.get("device", "gpu"),
            batch_size=data.get("batch_size", 2000),
            max_doc_size=data.get("max_doc_size", 2000000),
        )

        # Parse anonymization config
        anon_data = data.get("anonymization", {})
        anonymization = PiiAnonymizationConfig(
            default_action=anon_data.get("default_action", "replace"),
            default_params=anon_data.get("default_params", {"new_value": "<REDACTED>"}),
            entity_config=anon_data.get("entity_config", {}),
        )

        return cls(
            analyzer=analyzer,
            anonymization=anonymization,
            text_column=data.get("text_column", "text"),
        )

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        data = {
            "models": self.analyzer.models,
            "supported_entities": self.analyzer.supported_entities,
            "device": self.analyzer.device,
            "batch_size": self.analyzer.batch_size,
            "max_doc_size": self.analyzer.max_doc_size,
            "anonymization": {
                "default_action": self.anonymization.default_action,
                "default_params": self.anonymization.default_params,
                "entity_config": self.anonymization.entity_config,
            },
            "text_column": self.text_column,
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def to_stage_params(self) -> dict[str, Any]:
        """Convert to parameters for PiiRedactionStage."""
        # Get primary language (first model)
        primary_language = self.analyzer.models[0]["language"] if self.analyzer.models else "en"

        # Convert entity config to stage format
        entity_operators = {}
        for entity, config in self.anonymization.entity_config.items():
            entity_operators[entity] = {
                "action": config.get("action", self.anonymization.default_action),
                **config.get("params", {}),
            }

        return {
            "language": primary_language,
            "supported_entities": self.analyzer.supported_entities,
            "device": self.analyzer.device,
            "batch_size": self.analyzer.batch_size,
            "max_doc_size": self.analyzer.max_doc_size,
            "text_column": self.text_column,
            "anonymize_action": self.anonymization.default_action,
            "anonymize_kwargs": self.anonymization.default_params,
            "entity_operators": entity_operators,
        }


# Pre-configured setups for common use cases


def get_multilingual_config() -> PiiConfig:
    """Configuration for common European languages."""
    return PiiConfig(
        analyzer=PiiAnalyzerConfig(
            models=[
                {"language": "en", "model_name": "en_core_web_lg"},
                {"language": "es", "model_name": "es_core_news_md"},
                {"language": "de", "model_name": "de_core_news_lg"},
                {"language": "fr", "model_name": "fr_core_news_md"},
            ]
        )
    )


def get_minimal_config() -> PiiConfig:
    """Minimal configuration detecting only critical PII."""
    return PiiConfig(
        analyzer=PiiAnalyzerConfig(
            supported_entities=[
                "PERSON",
                "EMAIL_ADDRESS",
                "PHONE_NUMBER",
                "CREDIT_CARD",
                "US_SSN",
            ]
        )
    )


def get_hash_config() -> PiiConfig:
    """Configuration that hashes all PII instead of replacing."""
    return PiiConfig(
        anonymization=PiiAnonymizationConfig(
            default_action="hash",
            default_params={"hash_type": "sha256"},
        )
    )


def get_custom_entity_config() -> PiiConfig:
    """Configuration with different actions per entity type."""
    return PiiConfig(
        anonymization=PiiAnonymizationConfig(
            default_action="replace",
            default_params={"new_value": "<REDACTED>"},
            entity_config={
                "PERSON": {"action": "replace", "params": {"new_value": "[NAME]"}},
                "EMAIL_ADDRESS": {"action": "hash", "params": {"hash_type": "sha256"}},
                "PHONE_NUMBER": {
                    "action": "mask",
                    "params": {"chars_to_mask": 7, "masking_char": "*", "from_end": False},
                },
                "CREDIT_CARD": {
                    "action": "mask",
                    "params": {"chars_to_mask": 12, "masking_char": "X", "from_end": False},
                },
            },
        )
    )
