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

"""PII detection stage using NER models."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ray_curator.backends.base import WorkerMetadata
    from ray_curator.stages.pii.ner_pii.config import PiiAnalyzerConfig

from loguru import logger
from presidio_analyzer import AnalyzerEngine, EntityRecognizer, RecognizerRegistry, RecognizerResult
from presidio_analyzer.nlp_engine import NerModelConfiguration
from presidio_analyzer.nlp_engine.ner_model_configuration import LABELS_TO_IGNORE
from presidio_analyzer.predefined_recognizers import (
    CreditCardRecognizer,
    EmailRecognizer,
    IpRecognizer,
    PhoneRecognizer,
    SpacyRecognizer,
    UsSsnRecognizer,
)

from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.pii.ner_pii.config import PiiAnalyzerConfig
from ray_curator.stages.pii.ner_pii.constants import DEFAULT_LANGUAGE, SUPPORTED_ENTITIES
from ray_curator.stages.pii.ner_pii.custom_batch_analyzer_engine import CustomBatchAnalyzerEngine
from ray_curator.stages.pii.ner_pii.custom_nlp_engine import CustomNlpEngine
from ray_curator.stages.pii.ner_pii.recognizers.address_recognizer import AddressRecognizer
from ray_curator.stages.resources import Resources
from ray_curator.tasks import DocumentBatch


class PiiAnalyzer:
    """Analyzes (detects) PII entities in text using NER models.

    This class handles only the detection/analysis part of PII processing,
    separated from anonymization logic for better modularity.
    """

    def __init__(
        self,
        config: PiiAnalyzerConfig,
        load_nlp_engine: bool = True,
    ):
        """Initialize the PII analyzer (detector).

        Args:
            config: PiiAnalyzerConfig containing all configuration parameters
            load_nlp_engine: Whether to load the NLP engine immediately (default: True)
        """
        # Store config
        self.config = config

        # Get primary language from first model
        self.language = config.models[0]["language"] if config.models else DEFAULT_LANGUAGE

        # Set up labels to ignore
        additional_labels_to_ignore = {
            "GPE",
            "NORP",
            "AGE",
            "ID",
            "PATIENT",
            "HOSP",
            "PATORG",
            "HCW",
            "HOSPITAL",
            "FAC",
        }
        LABELS_TO_IGNORE.update(additional_labels_to_ignore)

        # Set up recognizers
        recognizer_registry = RecognizerRegistry(
            recognizers=[
                EmailRecognizer(),
                PhoneRecognizer(),
                SpacyRecognizer(),
                UsSsnRecognizer(),
                CreditCardRecognizer(),
                IpRecognizer(),
            ]
        )

        # Initialize analyzer components
        ner_model_configuration = NerModelConfiguration(labels_to_ignore=LABELS_TO_IGNORE)
        nlp_engine = CustomNlpEngine(ner_model_configuration=ner_model_configuration)

        # Load the NLP engine if requested (can be deferred for node-level loading)
        if load_nlp_engine:
            nlp_engine.load()

        self.analyzer = AnalyzerEngine(
            registry=recognizer_registry,
            nlp_engine=nlp_engine,
        )
        self.batch_analyzer = CustomBatchAnalyzerEngine(self.analyzer)

        # Set supported entities from config
        self.supported_entities = config.supported_entities or SUPPORTED_ENTITIES

        # Add custom recognizers if needed
        if "ADDRESS" in self.supported_entities:
            self.add_custom_recognizer(AddressRecognizer(supported_entities=["ADDRESS"]))

        # Set max document size
        self.max_doc_size = config.max_doc_size

    def set_max_doc_size(self, size: int) -> None:
        """Set the maximum document size for the NLP engine."""
        if self.language in self.analyzer.nlp_engine.nlp:
            self.analyzer.nlp_engine.nlp[self.language].max_length = size

    def ensure_nlp_engine_loaded(self) -> None:
        """Ensure the NLP engine is loaded. Useful for lazy loading scenarios."""
        if not self.analyzer.nlp_engine.nlp:
            self.analyzer.nlp_engine.load()

    def add_custom_recognizer(self, recognizer: EntityRecognizer) -> None:
        """Add a custom recognizer to detect entities based on user-defined logic."""
        self.supported_entities.extend(recognizer.get_supported_entities())
        self.analyzer.registry.add_recognizer(recognizer)

    def analyze_text(
        self, text: str, entities: list[str] | None = None, language: str | None = None
    ) -> list[RecognizerResult]:
        """Analyze a single text for PII entities."""
        if not entities:
            entities = self.supported_entities
        if not language:
            language = self.language
        return self.analyzer.analyze(text, language, entities=entities)

    def analyze_text_batch(
        self,
        texts: list[str],
        entities: list[str] | None = None,
        language: str | None = None,
        batch_size: int | None = None,
    ) -> list[list[RecognizerResult]]:
        """Analyze a batch of texts for PII entities.

        Args:
            texts: List of texts to analyze
            entities: List of entity types to detect
            language: Language code
            batch_size: Number of texts to process in a batch

        Returns:
            List of lists of RecognizerResult objects
        """
        if not entities:
            entities = self.supported_entities
        if not language:
            language = self.language
        if batch_size is None:
            batch_size = self.config.batch_size

        return self.batch_analyzer.analyze_iterator(texts, language, entities=entities, batch_size=batch_size)


@dataclass
class PiiDetectionStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Stage that detects PII entities in documents using NER models.

    This stage uses Presidio and spaCy to detect personally identifiable information
    in text documents. Detected entities are stored in the task metadata for use
    by downstream stages.

    Args:
        config: PiiAnalyzerConfig containing all configuration parameters
        text_column: Name of the column containing text to process
    """

    config: PiiAnalyzerConfig
    text_column: str = "text"

    def __post_init__(self):
        """Initialize the analyzer without loading models yet."""
        # Create the analyzer but don't load NLP engine yet
        # This allows us to share the analyzer instance across setup methods
        self.analyzer = PiiAnalyzer(
            config=self.config,
            load_nlp_engine=False,  # Don't load models yet
        )

    @property
    def name(self) -> str:
        return "pii_detection"

    @property
    def resources(self) -> Resources:
        """Resource requirements for this stage."""
        return Resources(
            gpus=1.0 if self.config.device == "gpu" else 0.0,
            cpus=1.0,
        )

    def inputs(self) -> tuple[list[str], list[str]]:
        """Define input requirements."""
        return ["data"], [self.text_column]

    def outputs(self) -> tuple[list[str], list[str]]:
        """Define outputs - adds pii_entities to metadata."""
        return ["data", "_pii_entities"], [self.text_column]

    """TODO: Add this back in when we have a way to load the models on the node
    def setup_on_node(self) -> None:
        # Initialize spaCy and load NER models once per node.
        import spacy
        breakpoint()
        # Set up GPU if requested - this should be done once per node
        if self.config.device == "gpu":
            spacy.require_gpu()
            logger.info("Configured spaCy to use GPU for PII detection")

        # Load the spaCy models at node level
        # This is important for GPU memory efficiency and model sharing
        try:
            # Load the NLP engine (downloads and loads spaCy models)
            self.analyzer.analyzer.nlp_engine.load()
            primary_language = self.config.models[0]["language"] if self.config.models else "en"
            logger.info(f"Pre-loaded spaCy model for language: {primary_language} on node")
        except Exception as e:
            logger.warning(f"Failed to pre-load spaCy model on node: {e}")
        pass
    """

    def setup(self, worker_metadata: "WorkerMetadata | None" = None) -> None:  # noqa: ARG002
        """Initialize the PII detector once per worker."""
        # Ensure the NLP engine is loaded (in case setup_on_node wasn't called)
        self.analyzer.ensure_nlp_engine_loaded()

        # Set max document size in the NLP engine
        self.analyzer.set_max_doc_size(self.config.max_doc_size)

        primary_language = self.config.models[0]["language"] if self.config.models else "en"
        logger.info(f"Initialized PII analyzer (detector) for language: {primary_language}")
        logger.info(f"Detecting entities: {self.config.supported_entities or 'all supported'}")

    def process(self, task: DocumentBatch) -> DocumentBatch:
        """Process a single document batch to detect PII entities."""
        try:
            # Extract text column
            if self.text_column not in task.data.columns:
                logger.error(f"Column '{self.text_column}' not found in task {task.task_id}")
                return task

            # Convert to pandas if needed
            texts = task.data.to_arrow().to_pylist()

            # Detect PII entities in batch
            logger.debug(f"Detecting PII in {len(texts)} documents")
            detected_entities = self.analyzer.analyze_text_batch(
                texts,
                entities=self.config.supported_entities,
                language=self.analyzer.language,
                batch_size=self.config.batch_size,
            )

            task._pii_entities = detected_entities
            # Log statistics
            total_entities = sum(len(entities) for entities in detected_entities)
            logger.info(f"Detected {total_entities} PII entities in task {task.task_id}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error processing task {task.task_id}: {e}")
            # Return task unchanged on error
        return task

    def teardown(self) -> None:
        """Clean up resources."""
        # The spaCy model and other resources will be garbage collected
        if hasattr(self, "analyzer"):
            del self.analyzer
