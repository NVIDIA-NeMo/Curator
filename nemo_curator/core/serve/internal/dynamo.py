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

from nemo_curator.core.serve.internal.base import InferenceBackend

if TYPE_CHECKING:
    from nemo_curator.core.serve.server import InferenceServer


class DynamoBackend(InferenceBackend):
    """Typed Dynamo backend placeholder.

    The real Dynamo implementation lands in the next stacked PR so this branch
    can stay focused on the public typed config surface.
    """

    def __init__(self, server: "InferenceServer") -> None:
        self._server = server

    def start(self) -> None:
        msg = "Dynamo backend implementation lands in PR 3."
        raise NotImplementedError(msg)

    def stop(self) -> None:
        return
