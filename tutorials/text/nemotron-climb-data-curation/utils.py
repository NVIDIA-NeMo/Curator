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

# This script is adapted from the RegMix project:
# https://github.com/sail-sg/regmix/blob/main/mixture_config/synthesize_mixture.py

import glob
import os


def get_token_distribution(input_path: str) -> dict[str, float]:
    """
    Get the token distribution from the input path of the tokenized files.

    Args:
    input_path (str): Path to the input directory containing the tokenized files.

    Returns:
    dict: Dictionary of tokenized files and their corresponding weights.
    """

    files = sorted(glob.glob(f"{input_path}/*.bin"))

    sizes = [os.path.getsize(f) for f in files]
    total = sum(sizes)

    weights: list[float] = [s / total for s in sizes]

    return dict(zip(files, weights, strict=True))
