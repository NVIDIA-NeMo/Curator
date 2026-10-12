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

"""Speaker/video-aware shard packing for GranaryHiFi SHARs (reference ``create_shar_tasks`` ``speaker_video``)."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from nemo_curator.stages.audio.tts.shar import SharEntry


def split_balanced(values: list[Any], max_size: int) -> list[list[Any]]:
    if max_size <= 0:
        msg = "max_samples_per_shard must be positive"
        raise ValueError(msg)
    if not values:
        return []
    num_chunks = (len(values) + max_size - 1) // max_size
    base_size, larger = divmod(len(values), num_chunks)
    chunks, start = [], 0
    for index in range(num_chunks):
        size = base_size + (1 if index < larger else 0)
        chunks.append(values[start : start + size])
        start += size
    return chunks


def pack_best_fit_decreasing(chunks: list[list[Any]], max_size: int) -> list[list[Any]]:
    """Pack largest chunks first into the shard leaving the least room; equal sizes keep input order."""
    shards: list[list[Any]] = []
    sizes: list[int] = []
    for _, chunk in sorted(enumerate(chunks), key=lambda item: (-len(item[1]), item[0])):
        best_index, best_remaining = None, None
        for index, current in enumerate(sizes):
            remaining = max_size - current
            if len(chunk) > remaining:
                continue
            if best_remaining is None or remaining - len(chunk) < best_remaining:
                best_index, best_remaining = index, remaining - len(chunk)
        if best_index is None:
            shards.append(list(chunk))
            sizes.append(len(chunk))
        else:
            shards[best_index].extend(chunk)
            sizes[best_index] += len(chunk)
    return shards


def plan_shards(
    entries: list[SharEntry],
    max_per_shard: int = 4096,
    seed: int = 17,
    *,
    keep_speaker_in_one_shard: bool = True,
) -> dict[tuple[str, int], list[list[SharEntry]]]:
    """Speaker/video-aware packing per ``(language, sampling_rate)`` (reference ``speaker_video`` strategy).

    Videos are shuffled with ``Random(seed)``. With ``keep_speaker_in_one_shard`` (default) all segments of a speaker
    (across videos, ordered by shuffled video then tar member) form one indivisible chunk, so a speaker and the
    context clips of its cuts live in exactly one shard; a speaker larger than ``max_per_shard`` gets its own
    oversized shard. Otherwise each (video, speaker) bucket is split into balanced chunks of at most
    ``max_per_shard`` (reference behaviour). Chunks are packed best-fit-decreasing.
    """
    grouped: dict[tuple[str, int], dict[str, dict[str, list[SharEntry]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for entry in entries:
        grouped[(entry.language, entry.sampling_rate)][entry.video_id][entry.speaker_id].append(entry)
    plan: dict[tuple[str, int], list[list[SharEntry]]] = {}
    for key in sorted(grouped):
        by_video = grouped[key]
        video_ids = sorted(by_video)
        random.Random(seed).shuffle(video_ids)  # noqa: S311
        if keep_speaker_in_one_shard:
            by_speaker: dict[str, list[SharEntry]] = {}
            for video_id in video_ids:
                for speaker_id in sorted(by_video[video_id]):
                    ordered = sorted(by_video[video_id][speaker_id], key=lambda e: e.audio_filepath)
                    by_speaker.setdefault(speaker_id, []).extend(ordered)
            chunks = list(by_speaker.values())
            for speaker_id, chunk in by_speaker.items():
                if len(chunk) > max_per_shard:
                    logger.warning(
                        "Speaker {} has {} cuts > max_per_shard={}; it gets its own oversized shard ({} / {})",
                        speaker_id,
                        len(chunk),
                        max_per_shard,
                        *key,
                    )
        else:
            chunks = []
            for video_id in video_ids:
                for speaker_id in sorted(by_video[video_id]):
                    ordered = sorted(by_video[video_id][speaker_id], key=lambda e: e.audio_filepath)
                    chunks.extend(split_balanced(ordered, max_per_shard))
        plan[key] = pack_best_fit_decreasing(chunks, max_per_shard)
    return plan
