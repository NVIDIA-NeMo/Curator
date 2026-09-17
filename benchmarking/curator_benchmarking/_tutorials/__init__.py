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

"""Build-time vendored tutorial files used by selected benchmarks.

This package is intentionally almost empty in the source tree. During package
builds, ``benchmarking/setup.py`` copies a small allowlist of tutorial files
from the peer ``tutorials/`` directory into this package so non-editable
``nemo-curator-benchmarking`` installs can run tutorial-backed benchmarks
without requiring a full source checkout.
"""
