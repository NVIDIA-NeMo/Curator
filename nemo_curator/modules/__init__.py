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
import os

# Disables multiprocessing in torch.compile calls.
# Without this, Dasks multiprocessing combined with PyTorch's
# gives errors like "daemonic processes are not allowed to have children"
# See https://github.com/NVIDIA/NeMo-Curator/issues/31
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

from nemo_curator.utils.import_utils import gpu_only_import_from

from .add_id import AddId
from .config import FuzzyDuplicatesConfig
from .dataset_ops import blend_datasets, Shuffle
from .exact_dedup import ExactDuplicates
from .filter import Filter, Score, ScoreFilter
from .meta import Sequential
from .modify import Modify
from .task import TaskDecontamination

# GPU packages
LSH = gpu_only_import_from("nemo_curator.modules.fuzzy_dedup", "LSH")
MinHash = gpu_only_import_from("nemo_curator.modules.fuzzy_dedup", "MinHash")
FuzzyDuplicates = gpu_only_import_from(
    "nemo_curator.modules.fuzzy_dedup", "FuzzyDuplicates"
)

# Pytorch related imports must come after all imports that require cugraph,
# because of context cleanup issues b/w pytorch and cugraph
# See this issue: https://github.com/rapidsai/cugraph/issues/2718
from .distributed_data_classifier import DomainClassifier, QualityClassifier

__all__ = [
    "DomainClassifier",
    "ExactDuplicates",
    "Filter",
    "FuzzyDuplicatesConfig",
    "FuzzyDuplicates",
    "LSH",
    "MinHash",
    "Modify",
    "QualityClassifier",
    "Score",
    "ScoreFilter",
    "Sequential",
    "TaskDecontamination",
    "AddId",
    "blend_datasets",
    "Shuffle",
]
