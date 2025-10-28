# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from latex_augment import transforms as T


def test_split_latex_column_spec():
    assert T.split_latex_column_spec("c @{} | l") == ["c", "@{}", "|", "l"]

    assert T.split_latex_column_spec("p{2cm}  >{\\centering}m{3em}  !{--}") == [
        "p{2cm}",
        ">{\\centering}",
        "m{3em}",
        "!{--}",
    ]

    assert T.split_latex_column_spec("c@{}|l!{\\hspace}>{\\raggedright}p{2cm}") == [
        "c",
        "@{}",
        "|",
        "l",
        "!{\\hspace}",
        ">{\\raggedright}",
        "p{2cm}",
    ]
