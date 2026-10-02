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

"""Explicit compatibility for benchmark scripts, never Curator-under-test."""

import os


def selected_profile() -> str | None:
    profile = os.environ.get("CURATOR_BENCHMARK_COMPAT_PROFILE", "")
    if profile not in ("", "26.07"):
        message = f"Unknown benchmark compatibility profile: {profile!r}"
        raise ValueError(message)
    return profile or None


if __name__ == "__main__":
    print(f"Benchmark compatibility profile: {selected_profile() or 'none'}")
