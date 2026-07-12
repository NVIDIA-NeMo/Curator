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

# Lazy-load every classifier so that importing this package does not pull in
# torch / transformers / numpy at module-parse time.  Each name is resolved
# on first access via __getattr__ (PEP 562, Python 3.7+).
_LAZY: dict[str, str] = {
    "AegisClassifier": ".aegis",
    "InstructionDataGuardClassifier": ".aegis",
    "ContentTypeClassifier": ".content_type",
    "DomainClassifier": ".domain",
    "MultilingualDomainClassifier": ".domain",
    "FineWebEduClassifier": ".fineweb_edu",
    "FineWebMixtralEduClassifier": ".fineweb_edu",
    "FineWebNemotronEduClassifier": ".fineweb_edu",
    "PromptTaskComplexityClassifier": ".prompt_task_complexity",
    "QualityClassifier": ".quality",
}

__all__ = list(_LAZY)


def __getattr__(name: str) -> object:
    if name in _LAZY:
        import importlib

        mod = importlib.import_module(_LAZY[name], package=__name__)
        return getattr(mod, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
