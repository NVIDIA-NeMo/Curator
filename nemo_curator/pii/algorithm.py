# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from pathlib import Path
from typing import Any, List, Mapping, Union

# NOTE: Importing this module before cluster creation will create a primary CUDA context
# that leads to issues of all GPUs not being used when creating a cluster/client later on.
# Ensure that this module is always imported after cluster creation only when the algorithm
# needs to be executed. See: https://github.com/NVIDIA/NeMo-Curator/issues/64
import yaml
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
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
from presidio_anonymizer import AnonymizerEngine, BatchAnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from nemo_curator.pii.constants import DEFAULT_LANGUAGE, SUPPORTED_ENTITIES
from nemo_curator.pii.custom_batch_analyzer_engine import CustomBatchAnalyzerEngine
from nemo_curator.pii.custom_nlp_engine import CustomNlpEngine
from nemo_curator.pii.recognizers.address_recognizer import AddressRecognizer

__all__ = [
    "PiiDeidentifier",
]


class PiiDeidentifier(object):
    """Cleans PII from an unstructured text"""

    def __init__(
        self,
        language: str = DEFAULT_LANGUAGE,
        supported_entities: List[str] = None,
        anonymize_action: str = "replace",
        **kwargs
    ):
        """
        Parameters:
        :param language (str): 2-digit language code
        :param supported_entities (List[str]): List of entities to consider while doing deidentification
        :param anonymize_action (str): String that determines what to do for anonymization. Options are:
        redact, hash, replace, mask and custom

        kwargs are additional anonymization related arguments.
        if anonymize_action is 'replace', 'new_value' can be provided as a substitution string
        if anonymize_action is 'hash', 'hash_type' can be provided (sha256, sha512 or md5)
        if anonymize_action is 'mask', 'chars_to_mask' and 'masking_char' can be provided
        if anonymize_action is 'custom', 'lambda' function can be provided
        """
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

        self.language = language
        ner_model_configuration = NerModelConfiguration(
            labels_to_ignore=LABELS_TO_IGNORE
        )
        self.analyzer = AnalyzerEngine(
            registry=recognizer_registry,
            nlp_engine=CustomNlpEngine(ner_model_configuration=ner_model_configuration),
        )
        self.anonymizer = AnonymizerEngine()
        self.batch_analyzer = CustomBatchAnalyzerEngine(self.analyzer)
        self.batch_anonymizer = BatchAnonymizerEngine(self.anonymizer)
        self.operators = {}

        if anonymize_action == "redact":
            self.operators["DEFAULT"] = OperatorConfig("redact", {})

        elif anonymize_action == "hash":
            self.operators["DEFAULT"] = OperatorConfig(
                "hash", {"hash_type": kwargs.get("hash_type")}
            )

        elif anonymize_action == "mask":
            self.operators["DEFAULT"] = OperatorConfig(
                "mask",
                {
                    "chars_to_mask": kwargs.get("chars_to_mask", 100),
                    "masking_char": kwargs.get("masking_char", "*"),
                    "from_end": False,
                },
            )

        elif anonymize_action == "lambda":
            self.operators["DEFAULT"] = OperatorConfig(
                "custom", {"lambda": kwargs.get("lambda")}
            )

        else:
            self.operators["DEFAULT"] = OperatorConfig(
                "replace", {"new_value": kwargs.get("new_value")}
            )

        self.supported_entities = supported_entities or SUPPORTED_ENTITIES

        if "ADDRESS" in self.supported_entities:
            self.add_custom_recognizer(
                AddressRecognizer(supported_entities=["ADDRESS"])
            )

    @staticmethod
    def from_config(config: Mapping[str, Any]):
        config = config.get("pii_config")
        language = config.get("language")
        supported_entities = config.get("supported_entities")
        operator_config = config.get("anonymize", {})
        operator_name = operator_config.get("action")
        if operator_name:
            del operator_config["action"]

        return PiiDeidentifier(
            language=language,
            supported_entities=supported_entities,
            anonymize_action=operator_name,
            **operator_config
        )

    @staticmethod
    def from_yaml_file(path: Union[Path, str]):
        with open(path) as f:
            return PiiDeidentifier.from_config(yaml.safe_load(f))

    @staticmethod
    def from_default_config():
        return PiiDeidentifier(
            PiiDeidentifier.DEFAULT_LANGUAGE,
            supported_entities=SUPPORTED_ENTITIES,
            anonymize_action="replace",
        )

    def list_supported_entities(self):
        """List all entities that are detected while cleaning a text"""
        return self.supported_entities.copy()

    def list_operators(self):
        """List all operators used to clean PII entities"""
        return self.operators.copy()

    def add_custom_recognizer(self, recognizer):
        """Add a custom recognizer to detect entities based on user-defined logic"""
        self.supported_entities.extend(recognizer.get_supported_entities())
        self.analyzer.registry.add_recognizer(recognizer)

    def add_custom_operator(self, entity, operator):
        """Use a custom cleaning operation for a specific entity types"""
        self.operators[entity] = operator

    def analyze_text(self, text, entities: List[str] = None, language: str = "en"):
        if not entities:
            entities = self.supported_entities
        return self.analyzer.analyze(text, language, entities=entities)

    def analyze_text_batch(
        self,
        texts: List[str],
        entities: List[str] = None,
        language: str = "en",
        batch_size: int = 32,
    ):
        """
        For processing batches, use batch analyzer

        Parameters:
        texts (List[str]): List of texts to perform deidentification on
        batch_size (int): The number of texts to handle in a batch. This
                          parameter is useful when using spacy models.

        Returns:
        List(str): list of deidentified text
        """
        if not entities:
            entities = self.supported_entities

        return self.batch_analyzer.analyze_iterator(
            texts, language, entities=entities, batch_size=batch_size
        )

    def deidentify_text_batch(self, texts: List[str], batch_size: int = 32):
        """
        For processing batches, use batch analyzer

        Parameters:
        texts (List[str]): List of texts to perform deidentification on
        batch_size (int): The number of texts to handle in a batch. This
                          parameter is useful when using spacy models.

        Returns:
        List(str): list of deidentified text
        """
        analyzer_results_list = self.batch_analyzer.analyze_iterator(
            texts,
            self.language,
            entities=self.supported_entities,
            batch_size=batch_size,
        )

        anonymized_results_list = self.batch_anonymizer.anonymize_list(
            texts, analyzer_results_list, operators=self.operators
        )
        return anonymized_results_list

    def deidentify_text(self, text: str):
        """
        Cleans PII data from text

        Parameters:
        text (str): Text that may contain personally-identifiable information

        Returns:
        str: Returns anonymized text
        """
        analyzer_results = self.analyzer.analyze(
            text=text, entities=self.supported_entities, language=self.language
        )
        anonymized_results = self.anonymizer.anonymize(
            text=text, analyzer_results=analyzer_results, operators=self.operators
        )
        return anonymized_results.text


if __name__ == "__main__":
    txt = (
        "Hello, I am John. I was born on December 5, 1983. My email is john.doe@gmail.com and "
        "you can call me on (814) 566 4637"
    )
    piid = PiiDeidentifier("en", ["DATE_TIME"])
    print(piid.deidentify_text(txt))

    piid = PiiDeidentifier("en", ["ADDRESS", "PERSON"], anonymize_action="replace")
    print(piid.deidentify_text_batch([txt]))
