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

from typing import TYPE_CHECKING

from nemo_curator.core.serve.base import InferenceBackend

if TYPE_CHECKING:
    from nemo_curator.core.serve.server import InferenceServer


class DynamoBackend(InferenceBackend):
    """Dynamo backend for ``InferenceServer``."""

    def __init__(self, server: "InferenceServer") -> None:
        self._server = server

    def start(self) -> None:
        msg = "DynamoBackend is not yet implemented."
        raise NotImplementedError(msg)

    def stop(self) -> None:
        pass
