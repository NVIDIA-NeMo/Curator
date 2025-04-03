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

import hashlib
import importlib
import re
import secrets
from typing import Any

import pandas as pd
from dask.base import normalize_token
from openai import OpenAI
from tqdm import tqdm

from nemo_curator import OpenAIClient
from nemo_curator.datasets import DocumentDataset
from nemo_curator.synthetic import NemotronGenerator
from nemo_curator.synthetic.generator import SyntheticDataGenerator
from nemo_curator.utils.distributed_utils import load_object_on_worker

tqdm.pandas()
config = importlib.import_module(
    "tutorials.nemo-retriever-synthetic-data-generation.config.config"
)
RetrieverEvalSDGConfig = config.RetrieverEvalSDGConfig


# ----------------------------------------------------------------------------80
class RetrieverEvalSetGenerator(SyntheticDataGenerator):
    """
    Main class that generates annotated datasets for retriever evaluation
    Question, Answers are generated for a given document chunk as input
    Datasets are annotated in format of (passage, question, answer) triplets
    """

    def __init__(
        self,
        pipeline_config: RetrieverEvalSDGConfig = None,
    ):
        super().__init__()
        self._name = self.__class__.__name__
        self.cfg = pipeline_config

        self._init_pipeline_params()

    def load_pipeline_config(self, cfg_path: str):
        self.cfg = RetrieverEvalSDGConfig.from_yaml(cfg_path)

    def _validate_config(self):
        return True  # TODO complete this

    def _init_pipeline_params(self):

        if self._validate_config():
            self.sys_prompt = self.cfg.generator_system_prompt
            self.user_prompt_template = self.cfg.generator_user_prompt_template
            self.generator_model = self.cfg.generator_model
            self.generator_model_kwargs = {
                "temperature": self.cfg.temperature,
                "top_p": self.cfg.top_p,
                "max_tokens": self.cfg.max_tokens,
            }
            self.num_qs = self.cfg.num_questions
        else:
            raise Exception("Validation Error: incorrect pipeline config file")

    # ----------------------------------------------------------------------------80

    def _create_generator(self):
        openai_client = OpenAI(
            base_url=self.cfg.base_url,
            api_key=self.cfg.api_key,
        )
        client = OpenAIClient(openai_client)
        generator = NemotronGenerator(client)
        return generator

    def _get_partition_id(self, df: pd.DataFrame, partition_info=None):
        df["partition-id"] = partition_info["number"]
        return df

    def __call__(self, dataset: DocumentDataset) -> DocumentDataset:

        ddf = dataset.df
        ddf["partition-id"] = ""
        ddf = ddf.map_partitions(self._get_partition_id, meta=ddf)
        ddf["llm_response"] = ""
        ddf["qa_pairs"] = ""
        ddf["question"] = ""
        ddf["answer"] = ""
        if "_id" not in ddf.columns:
            ddf["_id"] = ""
        ddf["question-id"] = ""
        ddf["score"] = ""

        ddf = ddf.map_partitions(self._process_on_partition, meta=ddf)

        return DocumentDataset(ddf)

    def _process_on_partition(self, df: pd.DataFrame) -> pd.DataFrame:

        self.generator = load_object_on_worker(
            attr="generator",
            load_object_function=self._create_generator,
            load_object_kwargs={},
        )

        _id = df["partition-id"].iloc[0]
        tqdm.pandas(desc=f"For partition_{_id}")

        if "_id" in df.columns:
            df["_id"] = df["_id"].apply(self._check_doc_id)
        else:
            df["_id"] = df["text"].apply(self._get_random_hash)

        df["llm_response"] = df["text"].progress_apply(self.generate)
        df["qa_pairs"] = df["llm_response"].apply(self.parse_response)

        df = df.explode("qa_pairs").reset_index(drop=True)
        df["question"] = df["qa_pairs"].apply(lambda x: x["question"])
        df["question-id"] = df["question"].apply(self._get_random_hash)
        df["answer"] = df["qa_pairs"].apply(lambda x: x["answer"])
        df["score"] = df["question"].apply(lambda x: 1)

        return df

    # ----------------------------------------------------------------------------80
    def parse_response(self, llm_response: str) -> Any:
        qa_pairs = []
        if llm_response:
            qa_list = llm_response.split("Question")[1:]
            try:
                for qa in qa_list:
                    qas = qa.split("Answer")
                    q = qas[0].split(":")[1].strip()
                    if re.search("Explanation", qas[1]):
                        a = qas[1].split("Explanation")[0].split(":")[1].strip()
                        explanation = qas[1].split("Explanation")[1].strip()
                    else:
                        a = qas[1].split(":")[1].strip()
                    qa_pairs.append({"question": q, "answer": a})
            except Exception as e:
                qa_pairs = [{"question": "", "answer": ""}]
                print(f"error: {e}")
        else:
            qa_pairs = [{"question": "", "answer": ""}]
        return qa_pairs

    # ----------------------------------------------------------------------------80
    def generate(self, doc_text):
        try:
            response = self.generator.generate_closed_qa_instructions(
                document=doc_text,
                prompt_template=self.sys_prompt + "\n" + self.user_prompt_template,
                n_openlines=self.num_qs,
                model=self.generator_model,
                model_kwargs=self.generator_model_kwargs,
            )
        except Exception as e:
            print(f"error: {e}")
            return ""

        return response[0]

    # ----------------------------------------------------------------------------80
    def _get_random_hash(self, question: str):
        """Generate random hash for synthetic question IDs"""
        # Generate a random string
        random_string = secrets.token_hex(16)
        # Generates a secure, random string of 16 bytes hex-encoded

        # Hash the random string using SHA-256
        hash_object = hashlib.sha256(
            random_string.encode()
        )  # Encode the string to bytes
        hex_dig = hash_object.hexdigest()
        return hex_dig

    # ----------------------------------------------------------------------------80
    def _check_doc_id(self, doc_id: Any) -> str:
        if doc_id:
            if str(doc_id) != "nan":
                return str(doc_id)
        return self._get_random_hash("some text")

    def __dask_tokenize__(self):
        return normalize_token(RetrieverEvalSetGenerator)


# ----------------------------------------------------------------------------80
