# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import logging
from typing import Dict, Iterator, List, Optional, Tuple, Union

import spacy
from presidio_analyzer.nlp_engine import (
    NerModelConfiguration,
    NlpArtifacts,
    SpacyNlpEngine,
)
from spacy import Language

logger = logging.getLogger("presidio-analyzer")


class CustomNlpEngine(SpacyNlpEngine):

    def __init__(
        self,
        models: Optional[List[Dict[str, str]]] = None,
        ner_model_configuration: Optional[NerModelConfiguration] = None,
    ):
        super().__init__(models, ner_model_configuration)
        self.nlp: Dict[str, Language] = None

    def load(self) -> None:
        """Load the spaCy NLP model."""
        logger.debug(f"Loading SpaCy models: {self.models}")

        self.nlp = {}
        # Download spaCy model if missing
        for model in self.models:
            self._validate_model_params(model)
            self._download_spacy_model_if_needed(model["model_name"])
            self.nlp[model["lang_code"]] = spacy.load(
                model["model_name"], enable=["ner"]
            )

    def process_batch(
        self,
        texts: Union[List[str], List[Tuple[str, object]]],
        language: str,
        as_tuples: bool = False,
        batch_size: int = 32,
    ) -> Iterator[Optional[NlpArtifacts]]:
        """Execute the NLP pipeline on a batch of texts using spacy pipe.

        :param texts: A list of texts to process.
        :param language: The language of the texts.
        :param as_tuples: If set to True, inputs should be a sequence of
            (text, context) tuples. Output will then be a sequence of
            (doc, context) tuples. Defaults to False.
        :param batch_size: The batch size.
        """

        if not self.nlp:
            raise ValueError("NLP engine is not loaded. Consider calling .load()")

        texts = [str(text) for text in texts]
        docs = self.nlp[language].pipe(
            texts, as_tuples=as_tuples, batch_size=batch_size
        )
        for doc in docs:
            yield doc.text, self._doc_to_nlp_artifact(doc, language)
