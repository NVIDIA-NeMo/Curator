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

from .downstream_task import DownstreamTask, import_task
from .metrics import (
    ANLI,
    CB,
    COQA,
    MMLU,
    PIQA,
    RTE,
    WSC,
    ArcChallenge,
    ArcEasy,
    BigBenchHard,
    BigBenchLight,
    BoolQ,
    Copa,
    Drop,
    Lambada,
    Multilingual,
    MultiRC,
    NumDasc,
    OpenBookQA,
    Quac,
    Race,
    Record,
    Squad,
    StoryCloze,
    TriviaQA,
    WebQA,
    WiC,
    Winogrande,
)

__all__ = [
    "ANLI",
    "CB",
    "COQA",
    "MMLU",
    "PIQA",
    "RTE",
    "WSC",
    "ArcChallenge",
    "ArcEasy",
    "BigBenchHard",
    "BigBenchLight",
    "BoolQ",
    "Copa",
    "DownstreamTask",
    "Drop",
    "Lambada",
    "MultiRC",
    "Multilingual",
    "NumDasc",
    "OpenBookQA",
    "Quac",
    "Race",
    "Record",
    "Squad",
    "StoryCloze",
    "TriviaQA",
    "WebQA",
    "WiC",
    "Winogrande",
    "import_task",
]
