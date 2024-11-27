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

import asyncio
import hashlib
import importlib
import os
import re
import secrets
from abc import ABC, abstractmethod
from typing import Any

import dask.array as da
import dask.dataframe as dd
import pandas as pd
from dask.base import normalize_token, tokenize
from dask.diagnostics import ProgressBar
from dask.distributed import progress
from distributed import Client
from omegaconf import DictConfig, OmegaConf
from openai import AsyncOpenAI, OpenAI
from tqdm import tqdm
from tqdm.dask import TqdmCallback

from config.config import RetrieverEvalSDGConfig
from nemo_curator import AsyncOpenAIClient, OpenAIClient
from nemo_curator.datasets import DocumentDataset
from nemo_curator.filters.doc_filter import DocumentFilter
from nemo_curator.synthetic import AsyncNemotronGenerator, NemotronGenerator
from nemo_curator.synthetic.generator import SyntheticDataGenerator


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
        # synchronous
        self.openai_client = OpenAI(
            base_url=self.cfg.base_url,
            api_key=self.cfg.api_key,
        )
        self.client = OpenAIClient(self.openai_client)
        self.generator = NemotronGenerator(self.client)

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

    def __call__(self, dataset: DocumentDataset) -> DocumentDataset:

        df = dataset.df

        df["llm_response"] = df["text"].apply(
            self.generate, meta=("llm_response", "str")
        )
        df["qa_pairs"] = df["llm_response"].apply(
            self.parse_response, meta=("qa_pairs", "object")
        )

        df = df.explode("qa_pairs").reset_index(drop=True)

        df["question"] = df["qa_pairs"].apply(
            lambda x: x["question"], meta=("question", "str")
        )

        if "_id" in df.columns:
            df["_id"] = df["_id"].apply(self._check_doc_id, meta=("_id", "str"))
        else:
            df["_id"] = df["text"].apply(self._get_random_hash, meta=("_id", "str"))

        df["question-id"] = df["question"].apply(
            self._get_random_hash, meta=("question-id", "str")
        )

        df["answer"] = df["qa_pairs"].apply(
            lambda x: x["answer"], meta=("answer", "str")
        )

        df["score"] = df["question"].apply(lambda x: 1, meta=("score", "int"))

        df = df.drop(["llm_response", "qa_pairs"], axis=1)

        return DocumentDataset(df)

    # ----------------------------------------------------------------------------80
    def parse_response(self, llm_response: str) -> Any:
        qa_pairs = []
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
            print(f"error: {e}")
        return qa_pairs

    # ----------------------------------------------------------------------------80
    def generate(self, doc_text):
        response = self.generator.generate_closed_qa_instructions(
            document=doc_text,
            prompt_template=self.sys_prompt + "\n" + self.user_prompt_template,
            n_openlines=self.num_qs,
            model=self.generator_model,
            model_kwargs=self.generator_model_kwargs,
        )
        return response[0]

    # ----------------------------------------------------------------------------80
    def _get_random_hash(self, question: str):
        """Generate random hash for synthetic question IDs"""
        # Generate a random string
        random_string = secrets.token_hex(
            16
        )  # Generates a secure, random string of 16 bytes hex-encoded

        # Hash the random string using SHA-256
        hash_object = hashlib.sha256(
            random_string.encode()
        )  # Encode the string to bytes
        hex_dig = hash_object.hexdigest()
        return hex_dig

    # ----------------------------------------------------------------------------80
    def _check_doc_id(self, doc_id: Any) -> str:
        if str(doc_id) == "nan":
            return self._get_random_hash("")
        else:
            return str(doc_id)

    def __dask_tokenize__(self):
        return normalize_token(RetrieverEvalSetGenerator)


# ----------------------------------------------------------------------------80
