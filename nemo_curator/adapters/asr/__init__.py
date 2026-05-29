# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""ASR adapter family for the SDP-V2 stage-adapter split.

Public surface (the only symbols the stage imports):

* :class:`ASRAdapter` - structural protocol every ASR adapter implements.
* :class:`ASRResult` - canonical per-utterance result dataclass.

Concrete adapters live in their own modules (e.g. ``qwen_omni.py``) and are
resolved at runtime by their fully-qualified class path in YAML's
``adapter_target`` field.
"""

from nemo_curator.adapters.asr.base import ASRAdapter, ASRResult
from nemo_curator.adapters.asr.qwen_omni import QwenOmniASRAdapter

__all__ = ["ASRAdapter", "ASRResult", "QwenOmniASRAdapter"]
