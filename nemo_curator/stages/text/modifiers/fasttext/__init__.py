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

if TYPE_CHECKING:
    from .fasttext_label import FastTextLabelModifier

__all__ = [
    "FastTextLabelModifier",
]


def __getattr__(name: str) -> type["FastTextLabelModifier"]:
    if name == "FastTextLabelModifier":
        from .fasttext_label import FastTextLabelModifier

        return FastTextLabelModifier
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
