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

"""PII anonymization stage that applies redaction strategies."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ray_curator.backends.base import WorkerMetadata
    from ray_curator.stages.pii.ner_pii.config import PiiAnonymizationConfig

from loguru import logger
from presidio_analyzer import RecognizerResult
from presidio_anonymizer import AnonymizerEngine, BatchAnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.pii.ner_pii.config import PiiAnonymizationConfig
from ray_curator.stages.resources import Resources
from ray_curator.tasks import DocumentBatch


class PiiAnonymizer:
    """Anonymizes detected PII entities in text.

    This class handles only the anonymization part of PII processing,
    separated from detection logic for better modularity.
    """

    def __init__(
        self,
        config: PiiAnonymizationConfig,
    ):
        """Initialize the PII anonymizer.

        Args:
            config: PiiAnonymizationConfig containing all configuration parameters
        """
        # Store config
        self.config = config

        # Initialize anonymizer engines
        self.anonymizer = AnonymizerEngine()
        self.batch_anonymizer = BatchAnonymizerEngine(self.anonymizer)

        # Set up operators from config
        self.operators = self._setup_operators_from_config()

    def _setup_operators_from_config(self) -> dict[str, OperatorConfig]:
        """Set up anonymization operators from configuration."""
        operators = {}

        # Set up default operator
        operators["DEFAULT"] = OperatorConfig(self.config.default_action, self.config.default_params)

        # Set up entity-specific operators
        for entity, entity_config in self.config.entity_config.items():
            action = entity_config.get("action", self.config.default_action)
            params = entity_config.get("params", self.config.default_params)
            operators[entity] = OperatorConfig(action, params)

        return operators

    def add_custom_operator(self, entity: str, operator: OperatorConfig) -> None:
        """Add a custom cleaning operation for a specific entity type."""
        self.operators[entity] = operator

    def anonymize_text(self, text: str, analyzer_results: list[RecognizerResult]) -> str:
        """Anonymize a single text based on detected entities."""
        result = self.anonymizer.anonymize(text=text, analyzer_results=analyzer_results, operators=self.operators)
        return result.text

    def anonymize_text_batch(self, texts: list[str], analyzer_results_list: list[list[RecognizerResult]]) -> list[str]:
        """Anonymize a batch of texts based on detected entities.

        Args:
            texts: List of original texts
            analyzer_results_list: List of analyzer results for each text

        Returns:
            List of anonymized texts
        """
        return self.batch_anonymizer.anonymize_list(texts, analyzer_results_list, operators=self.operators)


@dataclass
class PiiAnonymizationStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Stage that anonymizes detected PII entities in documents.

    This stage takes documents with detected PII entities (from PiiDetectionStage)
    and applies anonymization strategies to redact or replace the sensitive information.

    Args:
        config: PiiAnonymizationConfig containing all configuration parameters
        text_column: Name of the column containing text to process
    """

    config: PiiAnonymizationConfig
    text_column: str = "text"

    @property
    def name(self) -> str:
        return "pii_anonymization"

    @property
    def resources(self) -> Resources:
        """Resource requirements - anonymization is CPU-based."""
        return Resources(cpus=1.0)

    def inputs(self) -> tuple[list[str], list[str]]:
        """Define input requirements - needs pii_entities in metadata."""
        return ["data", "_pii_entities"], [self.text_column]

    def outputs(self) -> tuple[list[str], list[str]]:
        """Define outputs - modifies text column."""
        return ["data"], [self.text_column]

    def setup(self, worker_metadata: "WorkerMetadata | None" = None) -> None:  # noqa: ARG002
        """Initialize the anonymizer once per worker."""
        # Initialize the PII anonymizer
        self.anonymizer = PiiAnonymizer(config=self.config)

        logger.info(f"Initialized PII anonymizer with default action: {self.config.default_action}")
        if self.config.entity_config:
            logger.info(f"Entity-specific actions configured for: {list(self.config.entity_config.keys())}")

    def process(self, task: DocumentBatch) -> DocumentBatch:
        """Process a single document batch to anonymize PII entities."""
        try:
            # Check if PII entities were detected
            if not hasattr(task, "_pii_entities") or task._pii_entities is None:
                logger.warning(f"No PII entities found for task {task.task_id}")
                return task

            # Extract text and entities
            if self.text_column not in task.data.columns:
                logger.error(f"Column '{self.text_column}' not found in task {task.task_id}")
                return task

            # Pandas DataFrame
            df = task.data.to_pandas()
            texts = df[self.text_column].tolist()

            entities_list = task._pii_entities

            # Anonymize texts
            logger.debug(f"Anonymizing {len(texts)} documents")
            anonymized_texts = self.anonymizer.anonymize_text_batch(texts, entities_list)

            # Update the text column
            df[self.text_column] = anonymized_texts
            task.data = df

            # Log statistics
            total_entities = sum(len(entities) for entities in entities_list)
            logger.info(f"Anonymized {total_entities} PII entities in task {task.task_id}")

        except Exception as e:  # noqa: BLE001
            logger.error(f"Error anonymizing task {task.task_id}: {e}")
            # Return task unchanged on error

        return task

    def teardown(self) -> None:
        """Clean up resources."""
        if hasattr(self, "anonymizer"):
            del self.anonymizer
