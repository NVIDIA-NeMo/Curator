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

from abc import ABC, abstractmethod
from typing import List

from huggingface_hub import hf_hub_download

from nemo_curator.utils.import_utils import safe_import

COMET_IMPORT_MSG = (
    "To run QE filtering with COMET, you need to install from PyPI with: `pip install unbabel-comet`. "
    + "More information at https://github.com/Unbabel/COMET."
)
PYMARIAN_IMPORT_MSG = (
    "To run QE filtering with Cometoid/PyMarian, you need to install PyMarian. "
    + "More information at https://github.com/marian-nmt/wmt23-metrics?tab=readme-ov-file#setup."
)
comet = safe_import("comet", msg=COMET_IMPORT_MSG)
pymarian = safe_import("pymarian", msg=PYMARIAN_IMPORT_MSG)


class QEModel(ABC):
    """Abstract model for all quality estimation models for bitext."""

    def __init__(self, name: str, model, gpu=False):
        """Args:
        name (str): A string named of the model. Not directly tied to `MODEL_NAME_TO_HF_PATH` as defined in some subclasses but it is suggested.
        model: A loaded model object. The type of the object depends on the loaded model type.
        gpu (bool, optional): Whether inference is on GPU. Defaults to False.
        """
        self._name = name
        self._model = model
        self._gpu = gpu

    @classmethod
    @abstractmethod
    def load_model(cls, model_name: str):
        """An abstract method that loads the model according to a model name.

        Args:
            model_name (str): The name of the model to be loaded.
                Could be a huggingface model name, a path, or something else, depending on the implementation.
        """
        pass

    @staticmethod
    @abstractmethod
    def wrap_qe_input(src: str, tgt: str, reverse=False, **kwargs):
        """An abstract method that implements the following: given the individual source and target string of the bitext,
        wrap them into proper format that can be accepted by the underlying model.

        Args:
            src (str): Source side string of the bitext.
            tgt (str): Target side string of the bitext.
            reverse (bool, optional): Whether to reverse the source and target side of the bitext. Defaults to False.
        """
        pass

    @abstractmethod
    def predict(self, **kwargs) -> List[float]:
        """An abstract method that calls the underlying model to produce estimated quality scores.

        Returns:
            List[float]: List of quality scores.
        """
        pass


class COMETQEModel(QEModel):
    """Wrapper class for any COMET quality estimation models (https://github.com/Unbabel/COMET)."""

    MODEL_NAME_TO_HF_PATH = {
        "comet-qe": "Unbabel/wmt20-comet-qe-da",
    }

    @classmethod
    def load_model(cls, model_name: str, gpu: bool = False):
        """See parent class docstring for details on functionality and arguments."""
        path = comet.download_model(cls.MODEL_NAME_TO_HF_PATH[model_name])
        return cls(model_name, comet.load_from_checkpoint(path), gpu)

    @staticmethod
    def wrap_qe_input(src: str, tgt: str, reverse=False, **kwargs):
        """See parent class docstring for details on functionality and arguments."""
        return {"src": src, "mt": tgt} if not reverse else {"src": tgt, "mt": src}

    def predict(self, input: List, **kwargs) -> List[float]:
        """Implements quality estimation score prediction for COMET model.

        Args:
            input (List): A list of bitext pairs wrapped as dictionaries.

        Returns:
            List[float]: List of quality scores.
        """
        return self._model.predict(
            input, gpus=int(self._gpu), num_workers=0
        ).scores  # it's critical to set num_workers=0 to avoid spawning new processes within a dask worker


class PyMarianQEModel(QEModel):

    MODEL_NAME_TO_HF_PATH = {
        "cometoid-wmt23": "marian-nmt/cometoid22-wmt23",
        "cometoid-wmt23-mqm": "marian-nmt/cometoid22-wmt23",
    }
    # Because PyMarian depends on its own deep learning library rather than PyTorch/Huggingface
    # there is unfortunately no model configuration interface that can automatically adapt to
    # individual systems (like hf `AutoConfig`).
    # Those should work on most systems, but if not please adjust as needed.
    MARIAN_GPU_ARGS = " -w 8000 --mini-batch 32 -d 0"
    MARIAN_CPU_ARGS = " --cpu-threads 1 -w 2000"
    # PyMarian has memory leakage when a very large input is passed.
    # Hence we limit the size of input passed into PyMarian within one API call.
    SHARD_SIZE = 5000

    @classmethod
    def load_model(cls, model_name: str, gpu: bool = False):
        """See parent class docstring for details on functionality and arguments."""
        repo_id = cls.MODEL_NAME_TO_HF_PATH[model_name]
        model_path = hf_hub_download(repo_id, filename="checkpoints/marian.model.bin")
        vocab_path = hf_hub_download(repo_id, filename="vocab.spm")
        marian_args = f"-m {model_path} -v {vocab_path} {vocab_path} --like comet-qe"
        if gpu:
            marian_args += cls.MARIAN_GPU_ARGS
        else:
            marian_args += cls.MARIAN_CPU_ARGS
        return cls(model_name, pymarian.Evaluator(marian_args), gpu)

    @staticmethod
    def wrap_qe_input(src: str, tgt: str, reverse=False, **kwargs):
        """See parent class docstring for details on functionality and arguments."""
        return [src, tgt] if not reverse else [tgt, src]

    def predict(self, input: List, **kwargs) -> List[float]:
        """Implements quality estimation score prediction for Cometoid/PyMarian model.

        Args:
            input (List): A list of bitext pairs wrapped as dictionaries.

        Returns:
            List[float]: List of quality scores.
        """
        scores = []
        for start_idx in range(0, len(input), self.SHARD_SIZE):
            scores.extend(
                self._model.evaluate(input[start_idx : start_idx + self.SHARD_SIZE])
            )

        if not self._name.endswith("mqm"):
            # using DA+SQM score by default
            # choice made based on paper: https://aclanthology.org/2023.wmt-1.62.pdf
            return [score[1] for score in scores]
        else:
            return [score[0] for score in scores]
