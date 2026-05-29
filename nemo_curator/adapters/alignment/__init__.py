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

"""Forced-alignment adapter family for the SDP-V2 stage-adapter split.

Public surface (the only symbols the stage imports):

* :class:`ForcedAlignmentAdapter` - structural protocol every alignment
  adapter implements.
* :class:`AlignmentResult` - canonical per-utterance result dataclass.
* :class:`WordAlignment` - canonical per-word dataclass.

Concrete adapters live in their own modules (e.g. ``nemo_asr_align.py``,
``nemo_nfa.py``, ``whisperx_alignment.py``) and are resolved at runtime
by their fully-qualified class path in YAML's ``adapter_target`` field.
"""

from nemo_curator.adapters.alignment.base import (
    AlignmentResult,
    ForcedAlignmentAdapter,
    WordAlignment,
)
from nemo_curator.adapters.alignment.nemo_asr_align import NeMoASRAlignAdapter

__all__ = [
    "AlignmentResult",
    "ForcedAlignmentAdapter",
    "NeMoASRAlignAdapter",
    "WordAlignment",
]
