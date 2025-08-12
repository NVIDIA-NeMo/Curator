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

import os

os.environ["RAPIDS_NO_INITIALIZE"] = "1"
from .aegis import AegisClassifier, InstructionDataGuardClassifier
from .content_type import ContentTypeClassifier
from .domain import DomainClassifier, MultilingualDomainClassifier
from .fineweb_edu import (
    FineWebEduClassifier,
    FineWebMixtralEduClassifier,
    FineWebNemotronEduClassifier,
)
from .prompt_task_complexity import PromptTaskComplexityClassifier
from .quality import QualityClassifier

__all__ = [
    "AegisClassifier",
    "ContentTypeClassifier",
    "DomainClassifier",
    "FineWebEduClassifier",
    "FineWebMixtralEduClassifier",
    "FineWebNemotronEduClassifier",
    "InstructionDataGuardClassifier",
    "MultilingualDomainClassifier",
    "PromptTaskComplexityClassifier",
    "QualityClassifier",
]
