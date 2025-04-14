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

from typing import Dict, List

import pandas as pd

from nemo_curator.modifiers import DocumentModifier
from nemo_curator.pii.constants import DEFAULT_LANGUAGE, DEFAULT_MAX_DOC_SIZE
from nemo_curator.utils.decorators import batched
from nemo_curator.utils.distributed_utils import load_object_on_worker

__all__ = ["PiiModifier"]

DEFAULT_BATCH_SIZE = 2000


class PiiModifier(DocumentModifier):
    """
    This class is the entry point to using the PII de-identification module on documents stored as CSV, JSONL or
    other formats. It works with the `Modify` functionality as shown below:

    dataframe = pd.DataFrame({'text': ['Sarah and Ryan went out to play', 'Jensen is the CEO of NVIDIA']})
    dd = dask.dataframe.from_pandas(dataframe, npartitions=1)
    dataset = DocumentDataset(dd)

    modifier = PiiModifier(
        batch_size=2000,
        language='en',
        supported_entities=['PERSON', "EMAIL_ADDRESS"],
        anonymize_action='replace')

    modify = Modify(modifier)
    modified_dataset = modify(dataset)
    modified_dataset.df.to_json('output_files/*.jsonl', lines=True, orient='records')

    """

    def __init__(
        self,
        language: str = DEFAULT_LANGUAGE,
        supported_entities: List[str] = None,
        anonymize_action: str = "redact",
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str = "gpu",
        **kwargs,
    ):
        super().__init__()

        self.language = language
        self.supported_entities = supported_entities
        self.anonymize_action = anonymize_action
        self.kwargs = kwargs

        self.batch_size = batch_size
        self.device = device

    @batched
    def modify_document(self, text: pd.Series, partition_info: Dict = None):
        import logging

        logging.basicConfig(
            format="%(asctime)s %(levelname)s:%(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        deidentifier = load_object_on_worker("deidentifier", self.load_deidentifier, {})
        try:
            output: List[str] = deidentifier.deidentify_text_batch(
                text.tolist(), self.batch_size
            )
        except Exception as e:
            logging.error(
                f"Encountered error {str(e)} in partition {partition_info['number']}"
            )
            return pd.Series([True], index=text.index)
        output: pd.Series = pd.Series(output, text.index)
        return output

    def load_deidentifier(self):
        """
        Helper function to load the de-identifier
        """
        import spacy

        if self.device == "gpu":
            spacy.require_gpu()
        from nemo_curator.pii.algorithm import PiiDeidentifier

        deidentifier: PiiDeidentifier = PiiDeidentifier(
            language=self.language,
            supported_entities=self.supported_entities,
            anonymize_action=self.anonymize_action,
            **self.kwargs,
        )
        deidentifier.analyzer.nlp_engine.nlp[deidentifier.language].max_length = (
            DEFAULT_MAX_DOC_SIZE
        )

        return deidentifier
