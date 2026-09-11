# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from collections.abc import Iterable
from typing import Any

from nemo_curator.utils.stage_perf_collector import (
    PerformanceRecordStore,
    _new_spool_path,
    _StagePerfSpool,
)


def make_performance_record_store(
    records: Iterable[dict[str, Any]],
) -> PerformanceRecordStore:
    """Construct a disk-backed store for tests without expanding the production API."""
    spool = _StagePerfSpool(_new_spool_path())
    for record in records:
        spool.record(record)
    path, record_count = spool.finish()
    return PerformanceRecordStore(path=path, record_count=record_count)
