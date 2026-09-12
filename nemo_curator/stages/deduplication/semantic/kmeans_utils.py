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

"""Memory planning and GPU contexts for KMeans prediction workers."""

from collections.abc import Iterator
from contextlib import contextmanager

import cupy as cp
import rmm
from rmm.allocators.cupy import rmm_cupy_allocator

from .utils import ParquetFileInfo, break_parquet_partition_into_groups

# cuDF 26.08's dictionary maps, nested levels, and encoded/compressed buffers can use
# over 25 bytes per embedding element. Allow additional space for FP32 prediction.
_PREDICT_BYTES_PER_EMBEDDING_ELEMENT = 40
# Multiply retained columns' uncompressed Parquet bytes, including string contents.
# This allows for temporary copies; it is not a metadata element width.
_PREDICT_METADATA_MEMORY_FACTOR = 4


def plan_kmeans_prediction(
    file_info: list[ParquetFileInfo],
    *,
    memory_budget: int,
    max_workers: int,
    n_clusters: int,
    max_samples_per_batch: int,
) -> tuple[list[list[str]], int]:
    """Group complete files and choose how many groups can run concurrently.

    Each worker needs a read/write group plus an FP32 sample-by-centroid distance
    workspace. First divide memory among the requested workers, then reduce the
    worker count if a complete file exceeds that share. Groups also stay below
    cuDF's embedding child-column limit, independently of the memory estimate.

    Files cannot be split here. If one file has an estimated memory requirement above
    the total budget, try it with one worker; the conservative estimate is not an exact allocation
    bound and the underlying reader/writer can still raise an out-of-memory error.
    """
    file_memory_bytes = {
        info.path: info.embedding_elements * _PREDICT_BYTES_PER_EMBEDDING_ELEMENT
        + info.metadata_bytes * _PREDICT_METADATA_MEMORY_FACTOR
        for info in file_info
    }
    prediction_scratch_bytes = max_samples_per_batch * n_clusters * cp.dtype(cp.float32).itemsize
    group_memory_limit = memory_budget // max_workers - prediction_scratch_bytes
    # A complete input file is the smallest read unit, even when it exceeds a worker's share.
    largest_file_memory_bytes = max(file_memory_bytes.values())
    group_memory_limit = max(group_memory_limit, largest_file_memory_bytes)

    groups = []
    for element_bounded_group in break_parquet_partition_into_groups(file_info):
        group = []
        group_bytes = 0
        for path in element_bounded_group:
            if group and group_bytes + file_memory_bytes[path] > group_memory_limit:
                groups.append(group)
                group = []
                group_bytes = 0
            group.append(path)
            group_bytes += file_memory_bytes[path]
        groups.append(group)

    largest_group_bytes = max(sum(file_memory_bytes[path] for path in group) for group in groups)
    bytes_per_worker = largest_group_bytes + prediction_scratch_bytes
    workers_that_fit = memory_budget // bytes_per_worker
    worker_count = min(max_workers, len(groups), workers_that_fit)
    return groups, max(1, worker_count)


@contextmanager
def kmeans_prediction_memory_pool(memory_budget: int) -> Iterator[None]:
    """Share a bounded RMM pool across prediction workers, restoring it on exit.

    Call after releasing the fit array and before starting worker threads. The
    current RMM resource is shared by threads on this device, so it must remain
    installed until every worker has finished. Reusing allocations avoids repeated
    CUDA allocation costs; setting it up after fit preserves automatic fit sizing.
    """
    upstream = rmm.mr.get_current_device_resource()
    # RMM requires pool sizes to be multiples of 256 bytes.
    pool_bytes = memory_budget - memory_budget % 256
    pool = rmm.mr.PoolMemoryResource(upstream, initial_pool_size=min(2**30, pool_bytes), maximum_pool_size=pool_bytes)
    rmm.mr.set_current_device_resource(pool)
    try:
        yield
    finally:
        rmm.mr.set_current_device_resource(upstream)


@contextmanager
def kmeans_prediction_worker_context(device: int) -> Iterator[None]:
    """Use the actor's GPU, an independent stream, and its RMM pool in a new thread.

    CUDA device/stream selection and CuPy allocator overrides are thread-local.
    Explicitly select the actor's device because a new thread otherwise starts on
    device zero. The per-thread default stream matches cuDF's stream when
    CUDF_PER_THREAD_STREAM=1 is set before import, allowing workers to overlap.
    Routing CuPy allocations through RMM keeps embeddings and cuDF output in the
    same bounded pool instead of retaining memory in separate allocator caches.
    All three settings are restored when the worker exits, including on failure.
    """
    with cp.cuda.Device(device), cp.cuda.Stream.ptds, cp.cuda.using_allocator(rmm_cupy_allocator):
        yield
