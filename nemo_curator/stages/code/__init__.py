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

"""Code annotation stages and modifiers for NeMo Curator.

This module provides document modifiers for annotating code data using the
code_annotation Rust library.

Modifiers:
    - CodeLanguageDetector: Detects programming language
    - CodeBasicStats: Computes basic statistics (bytes, lines, patterns)
    - CodeSoftwareMetrics: Computes code complexity metrics
    - CodeOpenCoderMetrics: Computes OpenCoder comment fractions
    - CodeTokenizer: Tokenizes code using BPE tokenizers
"""

from nemo_curator.stages.code.modifiers import (
    CodeAnnotator,
    CodeBasicStats,
    CodeLanguageDetector,
    CodeOpenCoderMetrics,
    CodeSoftwareMetrics,
    CodeTokenizer,
)

__all__ = [
    "CodeAnnotator",
    "CodeBasicStats",
    "CodeLanguageDetector",
    "CodeOpenCoderMetrics",
    "CodeSoftwareMetrics",
    "CodeTokenizer",
]
