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

"""Model adapters for the SDP-V2 stage-adapter split.

Each adapter family (``asr``, ``diarization``, ``vad``, ...) lives in its
own subpackage and exposes:

* ``base.py`` - a ``Protocol`` plus a typed ``Result`` dataclass that every
  adapter in the family must implement.
* one module per concrete model that implements the protocol.

Stages in ``nemo_curator/stages/audio/inference/`` import the protocol and
typed result only; the concrete adapter is resolved at runtime from the
YAML's ``adapter_target`` string via ``hydra.utils.get_class``.
"""
