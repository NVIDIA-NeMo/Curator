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
from .config import FuzzyDuplicatesConfig, SemDedupConfig
from .dataset_ops import blend_datasets, Shuffle
from .exact_dedup import ExactDuplicates
from .meta import Sequential
from .modify import Modify
from .task import TaskDecontamination

# GPU packages
MinHash = gpu_only_import_from("nemo_curator.modules.fuzzy_dedup.minhash", "MinHash")
LSH = gpu_only_import_from("nemo_curator.modules.fuzzy_dedup.lsh", "LSH")
JaccardSimilarity = gpu_only_import_from(
    "nemo_curator.modules.fuzzy_dedup.jaccardsimilarity", "JaccardSimilarity"
)
BucketsToEdges = gpu_only_import_from(
    "nemo_curator.modules.fuzzy_dedup.bucketstoedges", "BucketsToEdges"
)
ConnectedComponents = gpu_only_import_from(
    "nemo_curator.modules.fuzzy_dedup.connectedcomponents", "ConnectedComponents"
)
FuzzyDuplicates = gpu_only_import_from(
    "nemo_curator.modules.fuzzy_dedup.fuzzyduplicates", "FuzzyDuplicates"
)

EmbeddingCreator = gpu_only_import_from(
    "nemo_curator.modules.semantic_dedup.embeddings", "EmbeddingCreator"
)
ClusteringModel = gpu_only_import_from(
    "nemo_curator.modules.semantic_dedup.clusteringmodel", "ClusteringModel"
)
SemanticClusterLevelDedup = gpu_only_import_from(
    "nemo_curator.modules.semantic_dedup.semanticclusterleveldedup",
    "SemanticClusterLevelDedup",
)
SemDedup = gpu_only_import_from(
    "nemo_curator.modules.semantic_dedup.semdedup", "SemDedup"
)

# PyTorch-related imports must come after all imports that require cuGraph
# because of context cleanup issues between PyTorch and cuGraph
# See this issue: https://github.com/rapidsai/cugraph/issues/2718
from .filter import Filter, Score, ScoreFilter, ParallelScoreFilter

__all__ = [
    "AddId",
    "FuzzyDuplicatesConfig",
    "SemDedupConfig",
    "blend_datasets",
    "Shuffle",
    "ExactDuplicates",
    "Filter",
    "Score",
    "ScoreFilter",
    "ParallelScoreFilter",
    "Sequential",
    "Modify",
    "TaskDecontamination",
    "MinHash",
    "LSH",
    "JaccardSimilarity",
    "BucketsToEdges",
    "ConnectedComponents",
    "FuzzyDuplicates",
    "EmbeddingCreator",
    "ClusteringModel",
    "SemanticClusterLevelDedup",
    "SemDedup",
]
